"""Main NeuroSymbolicLM model combining all components."""

from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .encoders import SimpleTransformerEncoder, PretrainedEncoderWrapper
from .decoders import SimpleTransformerDecoder, PretrainedDecoderWrapper
from .pooling import MultiQueryPool, span_mean_pool
from .entity import TokenEntityClassifier, ConceptBank
from .gnn import SimpleGNN, KGAwareGNN, KGPathReasoner
from .logic import SoftLogicConstraints, Controller, pair_logits_to_matrix

# Try to import KG utilities
try:
    from kg_utils import KGEmbeddingLoader, EntityLinker, KGGraph
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False


class NeuroSymbolicLM(nn.Module):
    """
    Neurosymbolic Language Model combining neural encoding with symbolic reasoning.
    
    Integrates:
    - Transformer encoder (custom or pre-trained)
    - Entity/concept extraction
    - Graph neural network for relational reasoning
    - Soft logic constraints
    - Transformer decoder for generation
    - Optional knowledge graph integration
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_entity_types: int = 8,
        n_relations: int = 32,
        n_concepts: int = 512,
        concept_dim: int = 256,
        node_dim: int = 256,
        max_nodes: int = 16,
        encoder=None,
        use_pretrained_encoder: bool = False,
        pretrained_model_name: str = "bert-base-uncased",
        freeze_encoder: bool = False,
        decoder=None,
        use_pretrained_decoder: bool = False,
        pretrained_decoder_name: str = "t5-small",
        freeze_decoder: bool = False,
        use_kg: bool = False,
        kg_embed_dim: int = 300,
        use_kg_gnn: bool = False,
        use_path_reasoning: bool = False,
        max_path_length: int = 3
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_entity_types = n_entity_types
        self.n_relations = n_relations
        self.n_concepts = n_concepts
        self.concept_dim = concept_dim
        self.node_dim = node_dim
        self.max_nodes = max_nodes
        self.use_pretrained_encoder = use_pretrained_encoder
        self.use_pretrained_decoder = use_pretrained_decoder
        self.use_kg = use_kg and KG_AVAILABLE
        self.kg_embed_dim = kg_embed_dim
        self.use_kg_gnn = use_kg_gnn and self.use_kg
        self.use_path_reasoning = use_path_reasoning and self.use_kg
        self.max_path_length = max_path_length
        
        # KG components
        self.kg_loader: Optional['KGEmbeddingLoader'] = None
        self.entity_linker: Optional['EntityLinker'] = None
        self.kg_graph: Optional['KGGraph'] = None
        self.entity_mapping: Dict[str, str] = {}
        
        # Initialize encoder
        if encoder is not None:
            self.encoder = encoder
            self.d_model = encoder.d_model
        elif use_pretrained_encoder:
            self.encoder = PretrainedEncoderWrapper(pretrained_model_name, freeze_encoder)
            self.d_model = self.encoder.d_model
        else:
            self.encoder = SimpleTransformerEncoder(vocab_size, d_model, nhead=8, nlayer=6)
            self.d_model = d_model
        
        # Token processing
        self.token_pool = MultiQueryPool(self.d_model, n_queries=6)
        self.token_ent = TokenEntityClassifier(self.d_model, n_entity_types)
        self.span_pool = MultiQueryPool(self.d_model, n_queries=4)
        
        # Concept bank
        self.concept_bank = ConceptBank(n_concepts, concept_dim)
        self.concept_proj = nn.Linear(self.d_model, concept_dim)
        self.node_proj = nn.Linear(self.d_model, node_dim)
        
        # GNN
        if self.use_kg_gnn:
            self.gnn = KGAwareGNN(node_dim, kg_embed_dim, n_layers=2, use_kg=True)
        else:
            self.gnn = SimpleGNN(node_dim, n_layers=2)
        
        # Path reasoner
        if self.use_path_reasoning:
            self.path_reasoner = KGPathReasoner(
                node_dim, kg_embed_dim, max_path_length=max_path_length
            )
        else:
            self.path_reasoner = None
        
        # Decoder
        if decoder is not None:
            self.decoder = decoder
        elif use_pretrained_decoder:
            self.decoder = PretrainedDecoderWrapper(
                pretrained_decoder_name,
                vocab_size=vocab_size,
                freeze_decoder=freeze_decoder,
                d_model=self.d_model
            )
        else:
            self.decoder = SimpleTransformerDecoder(vocab_size, self.d_model, nhead=8, nlayer=6)
        
        # Controller and relation scorer with dropout
        self.controller = Controller(d_model, node_dim)
        self.rel_scorer = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(node_dim, n_relations)
        )
        self.softlogic = SoftLogicConstraints(n_entity_types, n_relations)
        
        # Path reasoning integration projection
        if self.use_path_reasoning:
            self.path_to_rel_proj = nn.Linear(node_dim, n_relations)
        
        # MLM head (only for custom encoder)
        if not use_pretrained_encoder:
            self.mlm_head = nn.Linear(self.d_model, vocab_size)
        else:
            self.mlm_head = None
        
        # Concept head for Stage 2
        self.concept_head = nn.Linear(d_model, n_concepts)
        
        # Projection layer for node features
        if node_dim != self.d_model:
            self.node_to_encoder_proj = nn.Linear(node_dim, self.d_model)
        else:
            self.node_to_encoder_proj = nn.Identity()
    
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode input tokens."""
        return self.encoder(input_ids, attention_mask)
    
    @property
    def entity_head(self):
        """Entity classification head."""
        return self.token_ent
    
    def load_kg_data(
        self, 
        kg_loader: 'KGEmbeddingLoader', 
        entity_linker: Optional['EntityLinker'] = None,
        kg_graph: Optional['KGGraph'] = None, 
        entity_mapping: Optional[Dict[str, str]] = None
    ):
        """Load knowledge graph data for KG-aware processing."""
        if not self.use_kg:
            print("Warning: use_kg=False, KG data will not be used")
            return
        
        self.kg_loader = kg_loader
        self.entity_mapping = entity_mapping or {}
        
        if entity_linker is None:
            self.entity_linker = EntityLinker(kg_loader, entity_mapping)
        else:
            self.entity_linker = entity_linker
        
        self.kg_graph = kg_graph
        print(f"Loaded KG data: {len(kg_loader.entity_embeddings)} entities")
    
    def _get_kg_embeddings_for_entities(
        self, 
        entity_texts: List[str], 
        device: str
    ) -> Tuple[torch.Tensor, List[Optional[str]]]:
        """Get KG embeddings for text entities."""
        if not self.use_kg or self.entity_linker is None:
            return torch.zeros((len(entity_texts), self.kg_embed_dim), device=device), [None] * len(entity_texts)
        
        kg_entity_ids = self.entity_linker.link_entities_batch(entity_texts)
        entity_embeddings = self.kg_loader.get_embedding_tensor(kg_entity_ids, device=device)
        
        return entity_embeddings, kg_entity_ids
    
    def _compute_pairwise_relations_vectorized(
        self,
        node_feats: torch.Tensor,
        N: int,
        n_rel: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Compute pairwise relation logits using vectorized operations.
        
        Args:
            node_feats: (B, N, D) node features
            N: Number of nodes
            n_rel: Number of relation types
        
        Returns:
            pair_logits: List of (num_pairs, n_rel) tensors per batch
            pairs_index_map: List of (num_pairs, 2) index tensors per batch
            rel_logits_matrix: (B, N, N, n_rel) dense relation matrix
        """
        B, _, D = node_feats.shape
        device = node_feats.device
        
        # Create all pairs using broadcasting (vectorized)
        # node_i: (B, N, 1, D) -> (B, N, N, D)
        # node_j: (B, 1, N, D) -> (B, N, N, D)
        node_i = node_feats.unsqueeze(2).expand(-1, -1, N, -1)
        node_j = node_feats.unsqueeze(1).expand(-1, N, -1, -1)
        
        # Concatenate pairs: (B, N, N, 2D)
        pair_feats = torch.cat([node_i, node_j], dim=-1)
        
        # Apply relation scorer to all pairs at once: (B, N, N, n_rel)
        rel_logits_matrix = self.rel_scorer(pair_feats)
        
        # Extract upper triangular indices for pair_logits list (for compatibility)
        pair_logits = []
        pairs_index_map = []
        
        # Create index tensors once
        triu_indices = torch.triu_indices(N, N, offset=1, device=device)
        num_pairs = triu_indices.shape[1]
        
        if num_pairs > 0:
            indices_tensor = triu_indices.t()  # (num_pairs, 2)
            
            for b in range(B):
                # Extract upper triangular pairs for this batch
                batch_pair_logits = rel_logits_matrix[b, triu_indices[0], triu_indices[1]]
                pair_logits.append(batch_pair_logits)
                pairs_index_map.append(indices_tensor.clone())
        else:
            for b in range(B):
                pair_logits.append(torch.zeros((0, n_rel), device=device))
                pairs_index_map.append(torch.zeros((0, 2), dtype=torch.long, device=device))
        
        return pair_logits, pairs_index_map, rel_logits_matrix
    
    def _integrate_path_reasoning(
        self,
        rel_logits_matrix: torch.Tensor,
        node_feats: torch.Tensor,
        entity_texts: Optional[List[List[str]]] = None
    ) -> torch.Tensor:
        """
        Integrate path reasoning into relation predictions.
        
        Args:
            rel_logits_matrix: (B, N, N, n_rel) relation logits
            node_feats: (B, N, D) node features
            entity_texts: Optional entity text for each batch and node
        
        Returns:
            Updated relation logits matrix
        """
        if not self.use_path_reasoning or self.path_reasoner is None:
            return rel_logits_matrix
        
        if self.kg_graph is None or entity_texts is None:
            return rel_logits_matrix
        
        B, N, _, n_rel = rel_logits_matrix.shape
        device = rel_logits_matrix.device
        
        # Process each batch
        for b in range(B):
            batch_entities = entity_texts[b] if b < len(entity_texts) else []
            
            # Link entities to KG
            kg_entity_ids = []
            for ent_text in batch_entities[:N]:
                kg_id = self.entity_linker.link_entity(ent_text) if self.entity_linker else None
                kg_entity_ids.append(kg_id)
            
            # Find paths between entity pairs
            for i in range(min(len(kg_entity_ids), N)):
                for j in range(i + 1, min(len(kg_entity_ids), N)):
                    if kg_entity_ids[i] and kg_entity_ids[j]:
                        paths = self.kg_graph.find_paths(
                            kg_entity_ids[i],
                            kg_entity_ids[j],
                            max_length=self.max_path_length,
                            max_paths=5
                        )
                        
                        if paths:
                            # Encode paths and add to relation logits
                            path_contribution = self._encode_paths_for_relation(
                                paths, device
                            )
                            rel_logits_matrix[b, i, j] = rel_logits_matrix[b, i, j] + path_contribution
                            rel_logits_matrix[b, j, i] = rel_logits_matrix[b, j, i] + path_contribution
        
        return rel_logits_matrix
    
    def _encode_paths_for_relation(
        self,
        paths: List[List[Tuple[str, str]]],
        device: torch.device
    ) -> torch.Tensor:
        """
        Encode KG paths into relation logit contributions.
        
        Args:
            paths: List of paths, each path is list of (relation, entity) tuples
            device: Device to place tensors
        
        Returns:
            Relation logit contribution tensor of shape (n_relations,)
        """
        if not paths or self.path_reasoner is None:
            return torch.zeros(self.n_relations, device=device)
        
        # Get embeddings for paths
        path_embeddings = []
        for path in paths[:5]:  # Limit number of paths
            if not path:
                continue
            
            # Simple encoding: average relation embeddings in path
            path_emb = torch.zeros(self.node_dim, device=device)
            valid_steps = 0
            
            for rel, ent in path:
                if self.kg_loader:
                    rel_emb = self.kg_loader.get_embedding(rel)
                    if rel_emb is not None:
                        rel_tensor = torch.tensor(rel_emb, dtype=torch.float32, device=device)
                        # Project to node_dim
                        if hasattr(self.path_reasoner, 'kg_proj'):
                            rel_proj = self.path_reasoner.kg_proj(rel_tensor.unsqueeze(0)).squeeze(0)
                            path_emb = path_emb + rel_proj
                            valid_steps += 1
            
            if valid_steps > 0:
                path_emb = path_emb / valid_steps
                path_embeddings.append(path_emb)
        
        if not path_embeddings:
            return torch.zeros(self.n_relations, device=device)
        
        # Aggregate path embeddings
        aggregated = torch.stack(path_embeddings).mean(dim=0)
        
        # Project to relation space
        return self.path_to_rel_proj(aggregated)
    
    def generate(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        max_length: int = 128, 
        bos_token_id: int = 1, 
        eos_token_id: int = 2, 
        pad_token_id: int = 0, 
        temperature: float = 1.0, 
        do_sample: bool = False,
        top_k: Optional[int] = None, 
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate a sequence autoregressively.
        
        Args:
            input_ids: (B, L) input token IDs
            attention_mask: (B, L) attention mask
            max_length: Maximum generation length
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated token IDs of shape (B, T)
        """
        self.eval()
        device = input_ids.device
        B = input_ids.shape[0]
        
        with torch.no_grad():
            enc = self.encoder(input_ids, attention_mask)
            token_ent_logits = self.token_ent(enc)
            
            # Create node features
            ent_scores = token_ent_logits.max(dim=-1).values
            topk_vals, topk_idx = torch.topk(
                ent_scores, k=min(self.max_nodes, enc.size(1)), dim=-1
            )
            
            nodes = []
            for i in range(B):
                idxs = topk_idx[i].tolist()
                reps = torch.stack([enc[i, j] for j in idxs], dim=0)
                pad = enc.new_zeros((self.max_nodes - reps.size(0), enc.size(-1)))
                reps_padded = torch.cat([reps, pad], dim=0)
                nodes.append(reps_padded)
            
            node_feats = torch.stack(nodes, dim=0)
            node_feats = self.node_proj(node_feats)
            node_feats_refined = self.gnn(node_feats)
            memory_nodes = self.node_to_encoder_proj(node_feats_refined)
            
            # Initialize generation
            generated_ids = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            
            for step in range(max_length - 1):
                if finished.all():
                    break
                
                logits = self.decoder(generated_ids, enc, memory_nodes, memory_token_mask=attention_mask)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample or argmax
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        return generated_ids
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        spans: Optional[List[List[Tuple[int, int]]]] = None, 
        y_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: (B, L) input token IDs
            attention_mask: (B, L) attention mask
            spans: Optional gold entity spans for supervision
            y_ids: Target decoder IDs for teacher forcing
        
        Returns:
            Dictionary with model outputs
        """
        B, L = input_ids.shape
        enc = self.encoder(input_ids, attention_mask)
        token_pool, _ = self.token_pool(enc, attention_mask)
        token_ent_logits = self.token_ent(enc)
        
        # Create node features
        if spans is not None:
            node_repr = span_mean_pool(enc, spans)
            node_feats = self.node_proj(node_repr).unsqueeze(1)
        else:
            ent_scores = token_ent_logits.max(dim=-1).values
            topk_vals, topk_idx = torch.topk(ent_scores, k=min(self.max_nodes, L), dim=-1)
            nodes = []
            for i in range(B):
                idxs = topk_idx[i].tolist()
                reps = torch.stack([enc[i, j] for j in idxs], dim=0)
                pad = enc.new_zeros((self.max_nodes - reps.size(0), enc.size(-1)))
                reps_padded = torch.cat([reps, pad], dim=0)
                nodes.append(reps_padded)
            node_feats = torch.stack(nodes, dim=0)
            node_feats = self.node_proj(node_feats)
        
        # Concept mapping
        concept_query = self.concept_proj(token_pool)
        concept_vec, concept_probs = self.concept_bank.soft_assign(concept_query)
        
        # KG-aware GNN processing
        kg_relation_embeddings = None
        kg_adjacency = None
        
        if self.use_kg and self.use_kg_gnn:
            # Placeholder entity extraction - in production, decode from tokenizer
            entity_texts = [[f"entity_{j}" for j in range(min(self.max_nodes, L))] for _ in range(B)]
            
            if len(entity_texts) > 0:
                kg_entity_embs, kg_entity_ids = self._get_kg_embeddings_for_entities(
                    entity_texts[0], device=str(node_feats.device)
                )
                
                if kg_entity_ids and any(kg_entity_ids):
                    B_kg, N_kg = node_feats.shape[0], node_feats.shape[1]
                    kg_rel_embs = torch.zeros(B_kg, N_kg, N_kg, self.kg_embed_dim, device=node_feats.device)
                    kg_adj = torch.zeros(B_kg, N_kg, N_kg, device=node_feats.device)
                    
                    for b in range(B_kg):
                        for i in range(N_kg):
                            for j in range(N_kg):
                                if i != j and kg_entity_ids[i] and kg_entity_ids[j]:
                                    if self.kg_graph and kg_entity_ids[i] in self.kg_graph.edges:
                                        neighbors = self.kg_graph.get_neighbors(kg_entity_ids[i])
                                        for rel, target in neighbors:
                                            if target == kg_entity_ids[j]:
                                                rel_emb = self.kg_loader.get_embedding(rel)
                                                if rel_emb is not None:
                                                    kg_rel_embs[b, i, j] = torch.tensor(
                                                        rel_emb, dtype=torch.float32, device=node_feats.device
                                                    )
                                                    kg_adj[b, i, j] = 1.0
                    
                    kg_relation_embeddings = kg_rel_embs
                    kg_adjacency = kg_adj
        
        # Run GNN
        if self.use_kg_gnn and kg_relation_embeddings is not None:
            node_feats_refined = self.gnn(
                node_feats,
                kg_relation_embeddings=kg_relation_embeddings,
                kg_adjacency=kg_adjacency
            )
        else:
            node_feats_refined = self.gnn(node_feats)
        
        # Compute pairwise relation logits (vectorized)
        B_n, N, Dn = node_feats_refined.shape
        n_rel = self.rel_scorer[-1].out_features
        
        pair_logits, pairs_index_map, rel_logits_matrix = self._compute_pairwise_relations_vectorized(
            node_feats_refined, N, n_rel
        )
        
        # Node entity type probabilities
        if spans is not None:
            node_entity_logits = []
            for i in range(B):
                node_logits = []
                for (s, e) in spans[i]:
                    span_logits = token_ent_logits[i, s:e+1].mean(dim=0)
                    node_logits.append(span_logits)
                while len(node_logits) < N:
                    node_logits.append(torch.zeros(self.n_entity_types, device=token_ent_logits.device))
                node_entity_logits.append(torch.stack(node_logits[:N], dim=0))
            node_entity_logits = torch.stack(node_entity_logits, dim=0)
        else:
            node_entity_logits = []
            ent_scores = token_ent_logits.max(dim=-1).values
            topk_vals, topk_idx = torch.topk(ent_scores, k=min(self.max_nodes, L), dim=-1)
            for i in range(B):
                node_logits = []
                idxs = topk_idx[i].tolist()
                for j in idxs:
                    node_logits.append(token_ent_logits[i, j])
                while len(node_logits) < N:
                    node_logits.append(torch.zeros(self.n_entity_types, device=token_ent_logits.device))
                node_entity_logits.append(torch.stack(node_logits[:N], dim=0))
            node_entity_logits = torch.stack(node_entity_logits, dim=0)
        
        node_entity_type_probs = F.softmax(node_entity_logits, dim=-1)
        
        # Controller
        token_pool2, _ = self.token_pool(enc, attention_mask)
        node_pool = node_feats_refined.mean(dim=1)
        controller_logits = self.controller(token_pool2, node_pool)
        
        # Decoder memory
        memory_nodes = self.node_to_encoder_proj(node_feats_refined)
        
        if y_ids is not None:
            logits = self.decoder(y_ids, enc, memory_nodes, memory_token_mask=attention_mask)
            return {
                "logits": logits,
                "entity_logits": token_ent_logits,
                "concept_logits": concept_probs,
                "response_logit": controller_logits[:, 0:1],
                "token_ent_logits": token_ent_logits,
                "concept_probs": concept_probs,
                "pair_relation_logits": pair_logits,
                "controller_logits": controller_logits,
                "node_entity_type_probs": node_entity_type_probs,
                "rel_logits_matrix": rel_logits_matrix,
            }
        else:
            return {
                "enc": enc,
                "node_feats": node_feats_refined,
                "entity_logits": token_ent_logits,
                "concept_logits": concept_probs,
                "response_logit": controller_logits[:, 0:1],
                "token_ent_logits": token_ent_logits,
                "concept_probs": concept_probs,
                "pair_relation_logits": pair_logits,
                "controller_logits": controller_logits,
                "node_entity_type_probs": node_entity_type_probs,
                "rel_logits_matrix": rel_logits_matrix,
            }


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
    
    # LM loss
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
    
    # Concept loss
    if targets.get("concept_id") is not None:
        cp = model_outputs["concept_probs"]
        cid = targets["concept_id"]
        c_loss = F.cross_entropy(cp.log(), cid)
        losses["concept"] = c_loss * lambda_weights.get("concept", 1.0)
    
    # Controller loss
    if targets.get("controller_label") is not None:
        cl = model_outputs["controller_logits"]
        lab = targets["controller_label"]
        losses["controller"] = F.cross_entropy(cl, lab) * lambda_weights.get("controller", 1.0)
    
    total = sum(losses.values()) if len(losses) > 0 else torch.tensor(0.0)
    return total, losses
