"""Main NeuroSymbolicLM model combining all components.

This implementation uses T5/LongT5 as a unified encoder-decoder backbone,
avoiding tokenizer mismatches between separate encoder/decoder models.

Supports:
- Standard T5 (t5-small, t5-base, t5-large)
- LongT5 with transient global attention (google/long-t5-tglobal-base)
- Gradient checkpointing for memory efficiency
- Flash attention where available
"""

from typing import List, Tuple, Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pooling import MultiQueryPool, span_mean_pool
from .entity import (
    TokenEntityClassifier,
    ContextAwareEntityClassifier,
    ConceptBank,
    HierarchicalConceptBank,
)
from .gnn import SimpleGNN, AttentionGNN
from .logic import SoftLogicConstraints, pair_logits_to_matrix


class NeuroSymbolicLM(nn.Module):
    """
    Neurosymbolic Language Model using T5/LongT5 as unified encoder-decoder backbone.
    
    This design uses a single model and tokenizer throughout, avoiding the
    complexity of mixing different encoder/decoder architectures.
    
    For long context (8k-16k tokens), use LongT5 models:
    - google/long-t5-tglobal-base (recommended for RTX 4090)
    - google/long-t5-tglobal-large (requires more VRAM)
    
    Components:
    - T5/LongT5 encoder for text understanding
    - Entity/concept extraction from encoder outputs
    - Graph neural network for relational reasoning
    - Soft logic constraints for symbolic reasoning
    - T5/LongT5 decoder for response generation
    
    Abstention Behavior:
    The model learns when to abstain through the decoder generating EOS tokens.
    For samples where should_respond=0, the decoder is trained to output EOS
    immediately, effectively learning to "stay silent" without needing a
    separate decision head. This is simpler and more effective than using
    a separate Controller head for abstention decisions.
    
    Memory Efficiency:
    - Supports gradient checkpointing to reduce VRAM usage
    - Compatible with mixed precision training (AMP)
    - LongT5's transient global attention is more efficient than full attention
    """
    
    def __init__(
        self,
        model_name: str = "google/long-t5-tglobal-base",
        n_entity_types: int = 8,
        n_relations: int = 32,
        n_concepts: int = 512,
        concept_dim: int = 256,
        node_dim: int = 256,
        max_nodes: int = 16,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
        gradient_checkpointing: bool = False,
        max_input_length: int = 4096,
        max_output_length: int = 1024,
        # New options for enhanced components
        use_context_aware_entity: bool = False,
        use_hierarchical_concepts: bool = False,
        concept_hierarchy_levels: Optional[List[int]] = None,
        use_attention_gnn: bool = False,
        gnn_n_heads: int = 4,
        # Soft entity selection
        use_soft_entity_selection: bool = False,
        entity_selection_initial_temp: float = 2.0,
        entity_selection_min_temp: float = 0.1,
        # Linear graph transformer
        use_linear_graph_transformer: bool = False,
        linear_attn_n_random_features: int = 64,
        # Global workspace
        use_global_workspace: bool = False,
        workspace_n_slots: int = 16,
        workspace_n_cycles: int = 1,
        # Vision (T5Gemma)
        use_vision: bool = False,
        # Backbone dtype: None = auto (bfloat16 on CUDA, float32 elsewhere —
        # bf16 support on MPS/CPU is incomplete)
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # Determine model type and load appropriately
        self.model_name = model_name
        self.is_long_t5 = "long-t5" in model_name.lower()
        self.is_t5gemma = "t5gemma" in model_name.lower()
        self.has_vision = use_vision and self.is_t5gemma
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Load the appropriate model
        if self.is_t5gemma:
            from transformers import AutoModelForSeq2SeqLM, AutoProcessor
            print(f"Loading T5Gemma 2 model: {model_name} ({torch_dtype})")
            self.t5 = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, torch_dtype=torch_dtype
            )
            # Store processor for vision input handling
            if self.has_vision:
                self._processor = AutoProcessor.from_pretrained(model_name)
                print("Vision input enabled via SigLIP encoder")
        elif self.is_long_t5:
            from transformers import LongT5ForConditionalGeneration
            print(f"Loading LongT5 model: {model_name}")
            self.t5 = LongT5ForConditionalGeneration.from_pretrained(model_name)
        else:
            from transformers import T5ForConditionalGeneration
            print(f"Loading T5 model: {model_name}")
            self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)

        self.config = self.t5.config
        self.d_model = self.config.d_model
        self.vocab_size = self.config.vocab_size

        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            self.t5.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

        # Store configuration
        self.n_entity_types = n_entity_types
        self.n_relations = n_relations
        self.n_concepts = n_concepts
        self.concept_dim = concept_dim
        self.node_dim = node_dim
        self.max_nodes = max_nodes
        self.gradient_checkpointing = gradient_checkpointing
        self.use_context_aware_entity = use_context_aware_entity
        self.use_hierarchical_concepts = use_hierarchical_concepts
        self.use_attention_gnn = use_attention_gnn

        # Freeze options
        if freeze_encoder:
            for param in self.t5.encoder.parameters():
                param.requires_grad = False

        if freeze_decoder:
            for param in self.t5.decoder.parameters():
                param.requires_grad = False
            for param in self.t5.lm_head.parameters():
                param.requires_grad = False

        # Token processing
        self.token_pool = MultiQueryPool(self.d_model, n_queries=6)

        # Entity classifier - standard or context-aware
        if use_context_aware_entity:
            self.token_ent = ContextAwareEntityClassifier(
                self.d_model,
                n_entity_types,
                n_heads=4,
                context_layers=1,
            )
            print("Using context-aware entity classifier")
        else:
            self.token_ent = TokenEntityClassifier(self.d_model, n_entity_types)

        # Entity selection -- soft or hard
        self.use_soft_entity_selection = use_soft_entity_selection
        if use_soft_entity_selection:
            from .entity_selector import SoftEntitySelector
            self.entity_selector = SoftEntitySelector(
                max_nodes=max_nodes,
                d_model=self.d_model,
                node_dim=node_dim,
                initial_temp=entity_selection_initial_temp,
                min_temp=entity_selection_min_temp,
            )
            print(f"Using soft entity selection (temp={entity_selection_initial_temp})")
        else:
            self.entity_selector = None

        # Concept bank - flat or hierarchical
        if use_hierarchical_concepts:
            if concept_hierarchy_levels is None:
                concept_hierarchy_levels = [n_concepts, n_concepts // 4, n_concepts // 16]
            self.concept_bank = HierarchicalConceptBank(
                n_concepts_per_level=concept_hierarchy_levels,
                c_dim=concept_dim,
            )
            print(f"Using hierarchical concept bank: {concept_hierarchy_levels}")
        else:
            self.concept_bank = ConceptBank(n_concepts, concept_dim)

        self.concept_proj = nn.Linear(self.d_model, concept_dim)
        self.node_proj = nn.Linear(self.d_model, node_dim)

        # GNN - select variant
        if use_linear_graph_transformer:
            from .linear_graph_transformer import LinearGraphTransformer
            self.gnn = LinearGraphTransformer(
                node_dim,
                n_heads=gnn_n_heads,
                n_layers=2,
                n_random_features=linear_attn_n_random_features,
            )
            print(f"Using linear graph transformer ({linear_attn_n_random_features} random features)")
        elif use_attention_gnn:
            self.gnn = AttentionGNN(
                node_dim,
                n_heads=gnn_n_heads,
                n_layers=2,
            )
            print(f"Using attention-based GNN with {gnn_n_heads} heads")
        else:
            self.gnn = SimpleGNN(node_dim, n_layers=2)

        # Relation scorer
        self.rel_scorer = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(node_dim, n_relations)
        )

        # Soft logic constraints
        self.softlogic = SoftLogicConstraints(n_entity_types, n_relations)

        # Concept classification head
        self.concept_head = nn.Linear(self.d_model, n_concepts)

        # Projection for node features to feed into decoder
        if node_dim != self.d_model:
            self.node_to_encoder_proj = nn.Linear(node_dim, self.d_model)
        else:
            self.node_to_encoder_proj = nn.Identity()

        # Global workspace for decoder memory integration
        self.use_global_workspace = use_global_workspace
        if use_global_workspace:
            from .global_workspace import GlobalWorkspace
            self.global_workspace = GlobalWorkspace(
                d_model=self.d_model,
                n_slots=workspace_n_slots,
                n_cycles=workspace_n_cycles,
            )
            print(f"Using Global Workspace ({workspace_n_slots} slots, {workspace_n_cycles} cycles)")
        else:
            self.global_workspace = None

    @property
    def entity_head(self):
        """Entity classification head."""
        return self.token_ent
    
    @property
    def encoder(self):
        """Return the T5 encoder for compatibility."""
        return self.t5.encoder
    
    @property
    def decoder(self):
        """Return a wrapper for decoder access."""
        return self
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode input tokens (and optional images) using backbone encoder."""
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        if self.has_vision and pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
        encoder_outputs = self.t5.encoder(**kwargs)
        return encoder_outputs.last_hidden_state
    
    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Decode using T5 decoder, returning logits."""
        outputs = self.t5(
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )
        return outputs.logits
    
    def _compute_pairwise_relations_vectorized(
        self,
        node_feats: torch.Tensor,
        N: int,
        n_rel: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Compute pairwise relation logits using vectorized operations."""
        B, _, D = node_feats.shape
        device = node_feats.device
        
        # Create all pairs using broadcasting
        node_i = node_feats.unsqueeze(2).expand(-1, -1, N, -1)
        node_j = node_feats.unsqueeze(1).expand(-1, N, -1, -1)
        pair_feats = torch.cat([node_i, node_j], dim=-1)
        
        # Clamp pair features for numerical stability
        pair_feats = pair_feats.clamp(-100, 100)
        
        # Apply relation scorer: (B, N, N, n_rel)
        rel_logits_matrix = self.rel_scorer(pair_feats)
        
        # Clamp relation logits for numerical stability
        rel_logits_matrix = rel_logits_matrix.clamp(-50, 50)
        
        # Extract upper triangular for compatibility
        pair_logits = []
        pairs_index_map = []
        triu_indices = torch.triu_indices(N, N, offset=1, device=device)
        num_pairs = triu_indices.shape[1]
        
        if num_pairs > 0:
            indices_tensor = triu_indices.t()
            for b in range(B):
                batch_pair_logits = rel_logits_matrix[b, triu_indices[0], triu_indices[1]]
                pair_logits.append(batch_pair_logits)
                pairs_index_map.append(indices_tensor.clone())
        else:
            for b in range(B):
                pair_logits.append(torch.zeros((0, n_rel), device=device))
                pairs_index_map.append(torch.zeros((0, 2), dtype=torch.long, device=device))
        
        return pair_logits, pairs_index_map, rel_logits_matrix
    
    def _extract_node_features(
        self,
        enc: torch.Tensor,
        token_ent_logits: torch.Tensor,
        spans: Optional[List[List[Tuple[int, int]]]] = None
    ) -> torch.Tensor:
        """Extract node features from encoder outputs."""
        B, L, _ = enc.shape

        if spans is not None:
            # Use provided spans - extract each span's mean representation per batch item
            nodes = []
            for i in range(B):
                batch_spans = spans[i] if i < len(spans) else []
                batch_nodes = []
                for (s, e) in batch_spans[:self.max_nodes]:
                    span_repr = enc[i, s:e+1].mean(dim=0)
                    batch_nodes.append(self.node_proj(span_repr))
                # Pad to max_nodes
                while len(batch_nodes) < self.max_nodes:
                    batch_nodes.append(torch.zeros(self.node_dim, device=enc.device))
                nodes.append(torch.stack(batch_nodes[:self.max_nodes], dim=0))
            node_feats = torch.stack(nodes, dim=0)  # (B, max_nodes, node_dim)
        else:
            # Use top-k entity scores
            ent_scores = token_ent_logits.max(dim=-1).values
            topk_vals, topk_idx = torch.topk(
                ent_scores, k=min(self.max_nodes, L), dim=-1
            )
            
            nodes = []
            for i in range(B):
                idxs = topk_idx[i].tolist()
                reps = torch.stack([enc[i, j] for j in idxs], dim=0)
                pad_size = self.max_nodes - reps.size(0)
                if pad_size > 0:
                    pad = enc.new_zeros((pad_size, enc.size(-1)))
                    reps = torch.cat([reps, pad], dim=0)
                nodes.append(reps)
            
            node_feats = torch.stack(nodes, dim=0)
            node_feats = self.node_proj(node_feats)
        
        return node_feats
    
    def _compute_node_entity_probs(
        self,
        token_ent_logits: torch.Tensor,
        spans: Optional[List[List[Tuple[int, int]]]] = None
    ) -> torch.Tensor:
        """Compute entity type probabilities for nodes."""
        B, L, _ = token_ent_logits.shape
        N = self.max_nodes
        device = token_ent_logits.device
        
        if spans is not None:
            node_entity_logits = []
            for i in range(B):
                node_logits = []
                for (s, e) in spans[i]:
                    span_logits = token_ent_logits[i, s:e+1].mean(dim=0)
                    node_logits.append(span_logits)
                while len(node_logits) < N:
                    node_logits.append(torch.zeros(self.n_entity_types, device=device))
                node_entity_logits.append(torch.stack(node_logits[:N], dim=0))
            node_entity_logits = torch.stack(node_entity_logits, dim=0)
        else:
            ent_scores = token_ent_logits.max(dim=-1).values
            topk_vals, topk_idx = torch.topk(
                ent_scores, k=min(self.max_nodes, L), dim=-1
            )
            
            node_entity_logits = []
            for i in range(B):
                node_logits = []
                for j in topk_idx[i].tolist():
                    node_logits.append(token_ent_logits[i, j])
                while len(node_logits) < N:
                    node_logits.append(torch.zeros(self.n_entity_types, device=device))
                node_entity_logits.append(torch.stack(node_logits[:N], dim=0))
            node_entity_logits = torch.stack(node_entity_logits, dim=0)
        
        return F.softmax(node_entity_logits, dim=-1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        spans: Optional[List[List[Tuple[int, int]]]] = None,
        y_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: (B, L) input token IDs
            attention_mask: (B, L) attention mask
            spans: Optional gold entity spans
            y_ids: Decoder input IDs for teacher forcing
            pixel_values: Optional image tensors for vision models

        Returns:
            Dictionary with model outputs
        """
        B, L = input_ids.shape

        # Encode input
        enc = self.encode(input_ids, attention_mask, pixel_values=pixel_values)

        # Token-level entity classification
        # Context-aware classifier can use attention mask
        if self.use_context_aware_entity:
            token_ent_logits = self.token_ent(enc, attention_mask)
        else:
            token_ent_logits = self.token_ent(enc)

        # Pool tokens
        token_pool, _ = self.token_pool(enc, attention_mask)

        # Extract node features
        selection_weights = None
        if self.use_soft_entity_selection and self.entity_selector is not None:
            node_feats, selection_weights = self.entity_selector(
                enc, token_ent_logits, attention_mask
            )
        else:
            node_feats = self._extract_node_features(enc, token_ent_logits, spans)

        # GNN relational reasoning over extracted nodes
        node_feats_refined = self.gnn(node_feats)

        # Concept mapping
        concept_query = self.concept_proj(token_pool)

        # Handle hierarchical vs flat concept bank
        if self.use_hierarchical_concepts:
            concept_vec, concept_probs, hierarchy_probs = self.concept_bank.soft_assign(
                concept_query, aggregate_hierarchy=True
            )
        else:
            concept_vec, concept_probs = self.concept_bank.soft_assign(concept_query)
            hierarchy_probs = None

        # Compute pairwise relations
        N = node_feats_refined.shape[1]
        n_rel = self.rel_scorer[-1].out_features
        pair_logits, pairs_index_map, rel_logits_matrix = self._compute_pairwise_relations_vectorized(
            node_feats_refined, N, n_rel
        )

        # Node entity type probabilities
        node_entity_type_probs = self._compute_node_entity_probs(token_ent_logits, spans)

        # Base outputs
        outputs = {
            "entity_logits": token_ent_logits,
            "concept_logits": concept_probs,
            "concept_probs": concept_probs,
            "token_ent_logits": token_ent_logits,
            "pair_relation_logits": pair_logits,
            "node_entity_type_probs": node_entity_type_probs,
            "rel_logits_matrix": rel_logits_matrix,
            "enc": enc,
            "node_feats": node_feats_refined,
        }

        # Add hierarchical concept outputs if available
        if hierarchy_probs is not None:
            outputs["hierarchy_concept_probs"] = hierarchy_probs

        # Add selection weights if available
        if selection_weights is not None:
            outputs["selection_weights"] = selection_weights

        # Decoder forward if y_ids provided
        if y_ids is not None:
            memory_nodes = self.node_to_encoder_proj(node_feats_refined)

            if self.use_global_workspace and self.global_workspace is not None:
                # Optionally include concept vector as a specialist
                concept_input = concept_vec.unsqueeze(1) if concept_vec.dim() == 2 else None

                combined_memory, combined_mask, ws_attn = self.global_workspace(
                    encoder_states=enc,
                    node_features=memory_nodes,
                    concept_features=concept_input,
                    encoder_mask=attention_mask,
                )
                outputs["workspace_attention"] = ws_attn
            else:
                # Original concatenation path
                combined_memory = torch.cat([enc, memory_nodes], dim=1)
                node_mask = torch.ones(B, memory_nodes.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
                combined_mask = torch.cat([attention_mask, node_mask], dim=1)

            # Get decoder outputs
            logits = self.decode(y_ids, combined_memory, combined_mask)
            outputs["logits"] = logits

        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate response using T5's built-in generation.

        Args:
            input_ids: (B, L) input token IDs
            attention_mask: (B, L) attention mask
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling
            pixel_values: Optional image tensors for vision models

        Returns:
            Generated token IDs
        """
        from transformers.modeling_outputs import BaseModelOutput

        self.eval()

        with torch.no_grad():
            # Encode input
            enc = self.encode(input_ids, attention_mask, pixel_values=pixel_values)

            # Get node features for enhanced context
            if self.use_context_aware_entity:
                token_ent_logits = self.token_ent(enc, attention_mask)
            else:
                token_ent_logits = self.token_ent(enc)

            if self.use_soft_entity_selection and self.entity_selector is not None:
                node_feats, _ = self.entity_selector(enc, token_ent_logits, attention_mask)
            else:
                node_feats = self._extract_node_features(enc, token_ent_logits)

            node_feats_refined = self.gnn(node_feats)
            memory_nodes = self.node_to_encoder_proj(node_feats_refined)

            B = enc.shape[0]

            if self.use_global_workspace and self.global_workspace is not None:
                # Build concept features for workspace
                token_pool, _ = self.token_pool(enc, attention_mask)
                concept_query = self.concept_proj(token_pool)
                if self.use_hierarchical_concepts:
                    concept_vec, _, _ = self.concept_bank.soft_assign(
                        concept_query, aggregate_hierarchy=True
                    )
                else:
                    concept_vec, _ = self.concept_bank.soft_assign(concept_query)
                concept_input = concept_vec.unsqueeze(1) if concept_vec.dim() == 2 else None

                combined_memory, combined_mask, _ = self.global_workspace(
                    encoder_states=enc,
                    node_features=memory_nodes,
                    concept_features=concept_input,
                    encoder_mask=attention_mask,
                )
            else:
                combined_memory = torch.cat([enc, memory_nodes], dim=1)
                node_mask = torch.ones(B, memory_nodes.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
                combined_mask = torch.cat([attention_mask, node_mask], dim=1)

            # Wrap in BaseModelOutput for T5's generate method
            encoder_outputs = BaseModelOutput(last_hidden_state=combined_memory)

            # Use T5's generate method with enhanced encoder outputs
            generated = self.t5.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=combined_mask,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_k=top_k,
                top_p=top_p,
                **kwargs
            )

        return generated


def compute_losses(model_outputs: Dict, targets: Dict, lambda_weights: Dict) -> Tuple[torch.Tensor, Dict]:
    """
    Compute training losses from model outputs and targets.
    
    Args:
        model_outputs: Dict of model outputs
        targets: Dict of target tensors
        lambda_weights: Dict of loss weights
    
    Returns:
        total_loss: Scalar loss tensor
        losses: Dict of individual losses
    """
    losses = {}
    
    # LM/Decoder loss
    if "logits" in model_outputs and targets.get("lm_ids") is not None:
        logits = model_outputs["logits"]
        labels = targets["lm_ids"]
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        losses["lm"] = ce * lambda_weights.get("lm", 1.0)
    
    # Entity token loss
    if targets.get("entity_token_labels") is not None:
        ent_logits = model_outputs["token_ent_logits"]
        ent_labels = targets["entity_token_labels"]
        ent_loss = F.cross_entropy(
            ent_logits.view(-1, ent_logits.size(-1)),
            ent_labels.view(-1),
            ignore_index=-100
        )
        losses["entity"] = ent_loss * lambda_weights.get("entity", 1.0)
    
    # Concept loss - use nll_loss since cp is already probabilities (avoids double log)
    if targets.get("concept_id") is not None:
        cp = model_outputs["concept_probs"]
        cid = targets["concept_id"]
        c_loss = F.nll_loss(torch.log(cp + 1e-8), cid)
        losses["concept"] = c_loss * lambda_weights.get("concept", 1.0)
    
    if len(losses) > 0:
        total = sum(losses.values())
    else:
        # Get device from model outputs to avoid device mismatch
        device = model_outputs.get("entity_logits", model_outputs.get("concept_probs")).device
        total = torch.tensor(0.0, device=device, requires_grad=True)
    return total, losses
