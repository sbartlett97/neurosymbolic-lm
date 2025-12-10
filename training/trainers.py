"""Stage trainers for tiered training of the neurosymbolic model.

Includes:
- Gradient clipping for training stability
- Mixed precision training support
- Proper entity span alignment
- Learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from contextlib import nullcontext


class BaseTrainer:
    """Base trainer class with common functionality."""
    
    def __init__(
        self,
        model,
        optimizer,
        grad_clip_norm: float = 1.0,
        use_amp: bool = False,
        device: str = "cpu"
    ):
        self.model = model
        self.optimizer = optimizer
        self.grad_clip_norm = grad_clip_norm
        self.use_amp = use_amp and torch.cuda.is_available()
        self.device = device
        
        # Mixed precision scaler (using updated API)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
    
    def _get_amp_context(self):
        """Get appropriate autocast context for mixed precision."""
        if self.use_amp:
            return torch.amp.autocast('cuda')
        return nullcontext()
    
    def _backward_and_step(self, loss: torch.Tensor):
        """Perform backward pass with gradient clipping and optional AMP."""
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()


class Stage1_MLM_Trainer(BaseTrainer):
    """Stage 1: Masked Language Model pretraining for encoder."""
    
    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        grad_clip_norm: float = 1.0,
        use_amp: bool = False,
        device: str = "cpu"
    ):
        super().__init__(model, optimizer, grad_clip_norm, use_amp, device)
        self.tokenizer = tokenizer
        self.loss_fct = nn.CrossEntropyLoss()
    
    def train_step(self, batch: Dict[str, Any]) -> float:
        """Execute one training step."""
        # Skip if using pre-trained encoder
        if self.model.mlm_head is None:
            return 0.0
        
        self.optimizer.zero_grad()
        labels = batch["input_ids"].clone()
        masked = batch["input_ids"].clone()
        attention_mask = batch["attention_mask"]
        
        # Create random mask (15% masking)
        rand = torch.rand(masked.shape, device=masked.device)
        mask_positions = (rand < 0.15) & (attention_mask == 1)
        labels[~mask_positions] = -100
        
        masked[mask_positions] = self.tokenizer.mask_token_id
        
        with self._get_amp_context():
            enc = self.model.encode(masked, attention_mask)
            logits = self.model.mlm_head(enc)
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        self._backward_and_step(loss)
        return loss.item()


class Stage2_Symbolic_Trainer(BaseTrainer):
    """Stage 2: Entity extraction, concept mapping, and relation head training."""
    
    def __init__(
        self,
        model,
        optimizer,
        soft_logic_weight: float = 0.1,
        grad_clip_norm: float = 1.0,
        use_amp: bool = False,
        device: str = "cpu"
    ):
        super().__init__(model, optimizer, grad_clip_norm, use_amp, device)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.bce = nn.BCEWithLogitsLoss()
        self.soft_logic_weight = soft_logic_weight
    
    def train_step(self, batch: Dict[str, Any]) -> float:
        """Execute one training step."""
        self.optimizer.zero_grad()
        
        with self._get_amp_context():
            # Single forward pass - get all outputs at once
            out = self.model(
                batch["input_ids"],
                batch["attention_mask"],
                spans=None,
                y_ids=None
            )
            
            # Use encoder output from forward pass (no redundant encode call)
            enc = out.get("enc")
            if enc is None:
                # Fallback if enc not in output (when y_ids is provided)
                enc = self.model.encode(batch["input_ids"], batch["attention_mask"])
            
            # Entity classification using token-level entity span alignment
            ent_logits = out["token_ent_logits"]
            B, L, E = ent_logits.shape
            
            # Create entity token labels aligned to actual positions
            entity_token_labels = self._create_entity_token_labels(
                batch, B, L, E, enc.device
            )
            
            ent_loss = self.ce(ent_logits.view(-1, E), entity_token_labels.view(-1))
            
            # Concept classification
            concept_labels = batch["concept_labels"]
            n_concept_types = concept_labels.shape[2]
            
            # Use pooled encoder output
            pooled_enc = enc.mean(dim=1)
            concept_logits = self.model.concept_head(pooled_enc)
            n_concepts_bank = concept_logits.shape[1]
            
            # Aggregate concept labels across entities
            entity_mask = (batch["entity_ids"] > 0).float()
            num_valid_entities = entity_mask.sum(dim=1, keepdim=True).clamp(min=1)
            aggregated_concept_labels = (concept_labels * entity_mask.unsqueeze(-1)).sum(dim=1) / num_valid_entities
            
            # Handle dimension mismatch
            if n_concept_types == n_concepts_bank:
                target_labels = aggregated_concept_labels
            elif n_concept_types < n_concepts_bank:
                padding = torch.zeros(
                    aggregated_concept_labels.shape[0],
                    n_concepts_bank - n_concept_types,
                    device=aggregated_concept_labels.device,
                    dtype=aggregated_concept_labels.dtype
                )
                target_labels = torch.cat([aggregated_concept_labels, padding], dim=1)
            else:
                target_labels = aggregated_concept_labels[:, :n_concepts_bank]
            
            con_loss = self.bce(concept_logits, target_labels)
            
            # Soft logic constraint loss
            soft_logic_loss = torch.tensor(0.0, device=enc.device)
            if len(self.model.softlogic.rules) > 0:
                if "node_entity_type_probs" in out and "rel_logits_matrix" in out:
                    soft_logic_loss, _ = self.model.softlogic(
                        out["node_entity_type_probs"],
                        out["rel_logits_matrix"]
                    )
            
            loss = ent_loss + con_loss + self.soft_logic_weight * soft_logic_loss
        
        self._backward_and_step(loss)
        return loss.item()
    
    def _create_entity_token_labels(
        self,
        batch: Dict[str, Any],
        B: int,
        L: int,
        E: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create entity token labels aligned to actual token positions.
        
        Uses entity_token_spans if available, otherwise falls back to
        distributing entity types across first N tokens.
        """
        entity_token_labels = torch.zeros(B, L, dtype=torch.long, device=device)
        
        # Check if we have token-level spans
        entity_token_spans = batch.get("entity_token_spans")
        
        if entity_token_spans is not None:
            # Use actual token positions
            for i in range(B):
                spans = entity_token_spans[i]
                entity_types = batch["entity_type_labels"][i]
                for j, (start, end) in enumerate(spans):
                    if j >= len(entity_types):
                        break
                    entity_type = entity_types[j].item()
                    entity_type = min(entity_type, E - 1)
                    # Label all tokens in the span
                    for pos in range(start, min(end + 1, L)):
                        entity_token_labels[i, pos] = entity_type
        else:
            # Fallback: distribute entity types to first N positions
            for i in range(B):
                num_ents = (batch["entity_ids"][i] > 0).sum().item()
                for j in range(min(num_ents, L)):
                    entity_type = batch["entity_type_labels"][i, j].item()
                    entity_type = min(entity_type, E - 1)
                    entity_token_labels[i, j] = entity_type
        
        return entity_token_labels


class Stage3_Control_Trainer(BaseTrainer):
    """Stage 3: Response controller training."""
    
    def __init__(
        self,
        model,
        optimizer,
        grad_clip_norm: float = 1.0,
        use_amp: bool = False,
        device: str = "cpu"
    ):
        super().__init__(model, optimizer, grad_clip_norm, use_amp, device)
        self.bce = nn.BCEWithLogitsLoss()
    
    def train_step(self, batch: Dict[str, Any]) -> float:
        """Execute one training step."""
        self.optimizer.zero_grad()
        
        with self._get_amp_context():
            out = self.model(
                batch["input_ids"],
                batch["attention_mask"],
                spans=None,
                y_ids=None
            )
            
            answer_logit = out["controller_logits"][:, 0]
            
            should_respond = batch["should_respond"]
            if not isinstance(should_respond, torch.Tensor):
                should_respond = torch.tensor(
                    should_respond, dtype=torch.float, device=answer_logit.device
                )
            else:
                should_respond = should_respond.float()
            
            loss = self.bce(answer_logit, should_respond)
        
        self._backward_and_step(loss)
        return loss.item()


class Stage3_Decoder_Trainer(BaseTrainer):
    """Stage 3.5: Decoder response generation training."""
    
    def __init__(
        self,
        model,
        optimizer,
        grad_clip_norm: float = 1.0,
        use_amp: bool = False,
        device: str = "cpu"
    ):
        super().__init__(model, optimizer, grad_clip_norm, use_amp, device)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
    
    def train_step(self, batch: Dict[str, Any]) -> float:
        """Execute one training step.
        
        Note: The collator already prepares decoder_input_ids and decoder_labels
        such that they are aligned (no shift needed here):
        - decoder_input_ids[t] is the input at timestep t
        - decoder_labels[t] is the target prediction at timestep t
        """
        self.optimizer.zero_grad()
        
        if "decoder_input_ids" not in batch:
            return 0.0
        
        labels = batch["decoder_labels"]
        
        # Check if there are any valid labels (not all -100)
        valid_labels = (labels != -100).sum()
        if valid_labels == 0:
            return 0.0
        
        with self._get_amp_context():
            out = self.model(
                batch["input_ids"],
                batch["attention_mask"],
                spans=None,
                y_ids=batch["decoder_input_ids"]
            )
            
            logits = out["logits"]
            
            # No shift needed - collator already aligned input_ids and labels:
            # decoder_input_ids = [start, tok1, tok2, ...]
            # decoder_labels    = [tok1,  tok2, tok3, ...]
            # logits[t] predicts labels[t]
            decoder_loss = self.ce(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            # Check for NaN and skip if necessary
            if torch.isnan(decoder_loss) or torch.isinf(decoder_loss):
                print(f"Warning: NaN/Inf loss detected, skipping batch")
                return 0.0
        
        self._backward_and_step(decoder_loss)
        return decoder_loss.item()


class Stage4_Joint_Trainer(BaseTrainer):
    """Stage 4: Joint end-to-end training of all components."""
    
    def __init__(
        self,
        model,
        optimizer,
        soft_logic_weight: float = 0.1,
        grad_clip_norm: float = 1.0,
        use_amp: bool = False,
        device: str = "cpu"
    ):
        super().__init__(model, optimizer, grad_clip_norm, use_amp, device)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.ce_ignore_neg100 = nn.CrossEntropyLoss(ignore_index=-100)
        self.bce = nn.BCEWithLogitsLoss()
        self.soft_logic_weight = soft_logic_weight
    
    def train_step(self, batch: Dict[str, Any]) -> float:
        """Execute one training step."""
        self.optimizer.zero_grad()
        
        decoder_input_ids = batch.get("decoder_input_ids", None)
        
        with self._get_amp_context():
            out = self.model(
                batch["input_ids"],
                batch["attention_mask"],
                spans=None,
                y_ids=decoder_input_ids
            )
            
            # Entity loss with proper span alignment
            B, L, E = out["entity_logits"].shape
            entity_token_labels = self._create_entity_token_labels(
                batch, B, L, E, out["entity_logits"].device
            )
            ent_loss = self.ce(out["entity_logits"].view(-1, E), entity_token_labels.view(-1))
            
            # Concept loss
            concept_probs = out["concept_probs"]
            n_concepts_bank = concept_probs.shape[1]
            
            concept_labels = batch["concept_labels"]
            n_concept_types = concept_labels.shape[2]
            
            entity_mask = (batch["entity_ids"] > 0).float()
            num_valid_entities = entity_mask.sum(dim=1, keepdim=True).clamp(min=1)
            aggregated_concept_labels = (concept_labels * entity_mask.unsqueeze(-1)).sum(dim=1) / num_valid_entities
            
            if n_concept_types == n_concepts_bank:
                target_labels = aggregated_concept_labels
            elif n_concept_types < n_concepts_bank:
                padding = torch.zeros(
                    aggregated_concept_labels.shape[0],
                    n_concepts_bank - n_concept_types,
                    device=aggregated_concept_labels.device,
                    dtype=aggregated_concept_labels.dtype
                )
                target_labels = torch.cat([aggregated_concept_labels, padding], dim=1)
            else:
                target_labels = aggregated_concept_labels[:, :n_concepts_bank]
            
            eps = 1e-8
            concept_logits = torch.log(concept_probs + eps) - torch.log(1 - concept_probs + eps)
            con_loss = self.bce(concept_logits, target_labels)
            
            # Response controller loss
            should_respond = batch["should_respond"]
            if not isinstance(should_respond, torch.Tensor):
                should_respond = torch.tensor(
                    should_respond, dtype=torch.float, device=out["response_logit"].device
                )
            else:
                should_respond = should_respond.float()
            
            response_logit_squeezed = out["response_logit"].squeeze(-1)
            B_resp = response_logit_squeezed.shape[0]
            
            if should_respond.dim() == 0:
                should_respond = should_respond.unsqueeze(0).expand(B_resp)
            elif should_respond.shape[0] != B_resp:
                should_respond = should_respond.view(B_resp)
            
            should_respond = should_respond.to(response_logit_squeezed.device)
            resp_loss = self.bce(response_logit_squeezed, should_respond)
            
            # Decoder loss (no shift - collator already aligned input_ids and labels)
            decoder_loss = torch.tensor(0.0, device=out["entity_logits"].device)
            if "logits" in out and decoder_input_ids is not None and "decoder_labels" in batch:
                logits = out["logits"]
                labels = batch["decoder_labels"]
                
                decoder_loss = self.ce_ignore_neg100(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
            
            # Soft logic loss
            soft_logic_loss = torch.tensor(0.0, device=out["entity_logits"].device)
            if len(self.model.softlogic.rules) > 0:
                if "node_entity_type_probs" in out and "rel_logits_matrix" in out:
                    soft_logic_loss, _ = self.model.softlogic(
                        out["node_entity_type_probs"],
                        out["rel_logits_matrix"]
                    )
            
            loss = ent_loss + con_loss + resp_loss + decoder_loss + self.soft_logic_weight * soft_logic_loss
        
        self._backward_and_step(loss)
        return loss.item()
    
    def _create_entity_token_labels(
        self,
        batch: Dict[str, Any],
        B: int,
        L: int,
        E: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create entity token labels aligned to actual token positions."""
        entity_token_labels = torch.zeros(B, L, dtype=torch.long, device=device)
        
        entity_token_spans = batch.get("entity_token_spans")
        
        if entity_token_spans is not None:
            for i in range(B):
                spans = entity_token_spans[i]
                entity_types = batch["entity_type_labels"][i]
                for j, (start, end) in enumerate(spans):
                    if j >= len(entity_types):
                        break
                    entity_type = entity_types[j].item()
                    entity_type = min(entity_type, E - 1)
                    for pos in range(start, min(end + 1, L)):
                        entity_token_labels[i, pos] = entity_type
        else:
            for i in range(B):
                num_ents = (batch["entity_ids"][i] > 0).sum().item()
                for j in range(min(num_ents, L)):
                    entity_type = batch["entity_type_labels"][i, j].item()
                    entity_type = min(entity_type, E - 1)
                    entity_token_labels[i, j] = entity_type
        
        return entity_token_labels
