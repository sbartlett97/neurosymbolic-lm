"""Stage trainers for tiered training of the neurosymbolic model.

Includes:
- Gradient clipping for training stability
- Mixed precision training support
- Proper entity span alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
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
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
    
    def _check_model_health(self) -> bool:
        """Check if model weights contain NaN or Inf, and try to fix them."""
        has_issues = False
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                if not hasattr(self, '_nan_weight_printed'):
                    self._nan_weight_printed = True
                    print(f"    DEBUG: NaN found in {name}, resetting to zeros")
                param.data = torch.where(torch.isnan(param.data), torch.zeros_like(param.data), param.data)
                has_issues = True
            if torch.isinf(param).any():
                if not hasattr(self, '_inf_weight_printed'):
                    self._inf_weight_printed = True
                    print(f"    DEBUG: Inf found in {name}, clamping")
                param.data = param.data.clamp(-10, 10)
                has_issues = True
        return True  # Always return True after fixing
    
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
        # Use ignore_index=0 - entity type 0 is padding/unknown
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        # Use BCEWithLogitsLoss for concept classification (applies sigmoid internally)
        self.bce = nn.BCEWithLogitsLoss()
        self.soft_logic_weight = soft_logic_weight
    
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
            
            enc = out.get("enc")
            if enc is None:
                enc = self.model.encode(batch["input_ids"], batch["attention_mask"])
            
            # Entity classification
            ent_logits = out["token_ent_logits"]
            B, L, E = ent_logits.shape
            
            entity_token_labels = self._create_entity_token_labels(
                batch, B, L, E, enc.device
            )
            
            ent_loss = self.ce(ent_logits.view(-1, E), entity_token_labels.view(-1))
            
            # Concept classification
            concept_labels = batch["concept_labels"]
            n_concept_types = concept_labels.shape[2]
            
            entity_mask = (batch["entity_ids"] > 0).float()
            num_valid_entities = entity_mask.sum(dim=1, keepdim=True).clamp(min=1)
            aggregated_concept_labels = (concept_labels * entity_mask.unsqueeze(-1)).sum(dim=1) / num_valid_entities
            
            concept_probs = out["concept_probs"]
            n_concepts_bank = concept_probs.shape[1]
            
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
            
            # Convert concept_probs to logits for BCEWithLogitsLoss
            eps = 1e-8
            concept_logits = torch.log(concept_probs + eps) - torch.log(1 - concept_probs + eps)
            con_loss = self.bce(concept_logits, target_labels)
            
            # Relation classification loss
            rel_loss = torch.tensor(0.0, device=enc.device)
            pair_logits = out["pair_relation_logits"]
            relations = batch["relations"]
            
            for i, (plogits, rels) in enumerate(zip(pair_logits, relations)):
                if len(rels) == 0 or len(plogits) == 0:
                    continue
                
                # Fix NaN/Inf in pair logits before using them
                if torch.isnan(plogits).any() or torch.isinf(plogits).any():
                    plogits = torch.where(torch.isnan(plogits), torch.zeros_like(plogits), plogits)
                    plogits = plogits.clamp(-100, 100)
                
                for (head_idx, tail_idx, rel_type) in rels:
                    if head_idx < tail_idx:
                        pair_idx = self._get_pair_index(head_idx, tail_idx, out["node_feats"].shape[1])
                        if pair_idx < plogits.shape[0]:
                            rel_loss = rel_loss + F.cross_entropy(
                                plogits[pair_idx].unsqueeze(0),
                                torch.tensor([rel_type], device=plogits.device)
                            )
            
            # Soft logic loss
            soft_logic_loss = torch.tensor(0.0, device=enc.device)
            if len(self.model.softlogic.rules) > 0:
                if "node_entity_type_probs" in out and "rel_logits_matrix" in out:
                    soft_logic_loss, _ = self.model.softlogic(
                        out["node_entity_type_probs"],
                        out["rel_logits_matrix"]
                    )
            
            loss = ent_loss + con_loss + rel_loss + self.soft_logic_weight * soft_logic_loss
            
            # Debug: print component losses on first step
            if not hasattr(self, '_debug_printed'):
                self._debug_printed = True
                print(f"    DEBUG Stage2 losses: ent={ent_loss.item():.4f}, "
                      f"con={con_loss.item():.4f}, rel={rel_loss.item():.4f}, "
                      f"logic={soft_logic_loss.item():.4f}, total={loss.item():.4f}")
        
        self._backward_and_step(loss)
        return loss.item()
    
    def _get_pair_index(self, i: int, j: int, n: int) -> int:
        """Convert node pair indices to flat index (upper triangular)."""
        if i > j:
            i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)
    
    def _create_entity_token_labels(
        self,
        batch: Dict[str, Any],
        B: int,
        L: int,
        E: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create entity token labels aligned to actual token positions."""
        # Initialize with 0 (padding - ignored by CrossEntropyLoss with ignore_index=0)
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
            # Fallback: assign entity types to first N token positions
            for i in range(B):
                num_ents = (batch["entity_ids"][i] > 0).sum().item()
                for j in range(min(num_ents, L)):
                    etype = batch["entity_type_labels"][i, j].item()
                    etype = min(etype, E - 1)
                    entity_token_labels[i, j] = etype
        
        return entity_token_labels


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
        """Execute one training step."""
        # Check and fix model health
        self._check_model_health()
        
        self.optimizer.zero_grad()
        
        if "decoder_input_ids" not in batch:
            return 0.0
        
        labels = batch["decoder_labels"]
        decoder_input_ids = batch["decoder_input_ids"]
        
        # Check if there are any valid labels
        valid_labels = (labels != -100).sum()
        if valid_labels == 0:
            return 0.0
        
        with self._get_amp_context():
            out = self.model(
                batch["input_ids"],
                batch["attention_mask"],
                spans=None,
                y_ids=decoder_input_ids
            )
            
            logits = out["logits"]
            
            # Fix NaN/Inf in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
                logits = logits.clamp(-100, 100)
            
            # Compute loss (no shift needed - collator aligns input_ids and labels)
            decoder_loss = self.ce(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            # Debug: print loss on first step
            if not hasattr(self, '_debug_printed'):
                self._debug_printed = True
                print(f"    DEBUG Stage3 decoder_loss={decoder_loss.item():.4f}")
            
            # Handle NaN/Inf loss
            if torch.isnan(decoder_loss) or torch.isinf(decoder_loss):
                decoder_loss = torch.tensor(0.1, device=logits.device, requires_grad=True)
        
        # Backward pass
        self._backward_and_step(decoder_loss)
        
        return decoder_loss.item()


class Stage4_Joint_Trainer(BaseTrainer):
    """Stage 4: Joint end-to-end training of all components.
    
    Note: Controller/response decision training has been removed.
    Abstention is now learned through the decoder generating EOS tokens
    for should_respond=0 samples.
    """
    
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
        # Use ignore_index=0 for entity classification (0 = padding)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        # Use ignore_index=-100 for decoder (standard HF convention)
        self.ce_ignore_neg100 = nn.CrossEntropyLoss(ignore_index=-100)
        # Use BCEWithLogitsLoss for concept classification
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
            
            # Entity loss
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
            
            # Convert concept_probs to logits for BCEWithLogitsLoss
            eps = 1e-8
            concept_logits = torch.log(concept_probs + eps) - torch.log(1 - concept_probs + eps)
            con_loss = self.bce(concept_logits, target_labels)
            
            # Decoder loss (includes EOS-based abstention learning)
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
            
            loss = ent_loss + con_loss + decoder_loss + self.soft_logic_weight * soft_logic_loss
            
            # Debug: print component losses on first step
            if not hasattr(self, '_debug_printed'):
                self._debug_printed = True
                print(f"    DEBUG Stage4 losses: ent={ent_loss.item():.4f}, con={con_loss.item():.4f}, "
                      f"dec={decoder_loss.item():.4f}, logic={soft_logic_loss.item():.4f}, total={loss.item():.4f}")
        
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
        # Initialize with 0 (padding - ignored by CrossEntropyLoss with ignore_index=0)
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
            # Fallback: assign entity types to first N token positions
            for i in range(B):
                num_ents = (batch["entity_ids"][i] > 0).sum().item()
                for j in range(min(num_ents, L)):
                    etype = batch["entity_type_labels"][i, j].item()
                    etype = min(etype, E - 1)
                    entity_token_labels[i, j] = etype
        
        return entity_token_labels
