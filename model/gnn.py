"""Graph Neural Network modules for relational reasoning."""

from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import numpy as np


class SimpleGNN(nn.Module):
    """
    Simple message-passing GNN for entity relation reasoning.
    
    Performs pairwise message passing between all nodes with dropout for regularization.
    """
    
    def __init__(self, node_dim: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(node_dim)
        self.layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(node_dim * 2, node_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(node_dim, node_dim)
            ))
    
    def forward(
        self, 
        node_feats: torch.Tensor, 
        adj_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Propagate messages through the graph.
        
        Args:
            node_feats: (B, N, D) node features
            adj_mask: (B, N, N) optional adjacency mask
        
        Returns:
            Refined node features of shape (B, N, D)
        """
        B, N, D = node_feats.shape
        h = node_feats
        
        for layer in self.layers:
            hi = h.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
            hj = h.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
            pairs = torch.cat([hi, hj], dim=-1)  # (B, N, N, 2D)
            m = layer(pairs)  # (B, N, N, D)
            
            if adj_mask is not None:
                m = m * adj_mask.unsqueeze(-1)
            
            # Average instead of sum to prevent value explosion
            msg = m.mean(dim=2)  # (B, N, D)
            h = self.layer_norm(h + self.dropout(msg))  # Residual + LayerNorm
        
        # Clamp output for numerical stability
        h = h.clamp(-100, 100)
        
        return h


class KGAwareGNN(nn.Module):
    """
    GNN that incorporates knowledge graph structure and relation embeddings.
    
    Uses KG relation embeddings to guide message passing between nodes.
    """
    
    def __init__(
        self, 
        node_dim: int, 
        kg_embed_dim: int, 
        n_layers: int = 2, 
        use_kg: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.use_kg = use_kg
        self.kg_embed_dim = kg_embed_dim
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(node_dim)
        self.layers = nn.ModuleList()
        
        if self.use_kg:
            self.kg_proj = nn.Linear(kg_embed_dim, node_dim)
            input_dim = node_dim * 3  # node_i, node_j, kg_relation
        else:
            input_dim = node_dim * 2
        
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim, node_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(node_dim, node_dim)
            ))
    
    def forward(
        self, 
        node_feats: torch.Tensor, 
        kg_relation_embeddings: Optional[torch.Tensor] = None,
        kg_adjacency: Optional[torch.Tensor] = None,
        adj_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Propagate messages with KG-aware attention.
        
        Args:
            node_feats: (B, N, node_dim) node features
            kg_relation_embeddings: (B, N, N, kg_embed_dim) relation embeddings
            kg_adjacency: (B, N, N) binary mask for KG edges
            adj_mask: (B, N, N) optional additional adjacency mask
        
        Returns:
            Refined node features of shape (B, N, node_dim)
        """
        B, N, D = node_feats.shape
        h = node_feats
        
        for layer in self.layers:
            hi = h.unsqueeze(2).expand(-1, -1, N, -1)
            hj = h.unsqueeze(1).expand(-1, N, -1, -1)
            
            if self.use_kg and kg_relation_embeddings is not None:
                kg_rel_proj = self.kg_proj(kg_relation_embeddings)
                pairs = torch.cat([hi, hj, kg_rel_proj], dim=-1)
            else:
                pairs = torch.cat([hi, hj], dim=-1)
            
            m = layer(pairs)
            
            if kg_adjacency is not None:
                m = m * kg_adjacency.unsqueeze(-1)
            
            if adj_mask is not None:
                m = m * adj_mask.unsqueeze(-1)
            
            msg = m.sum(dim=2)
            h = self.layer_norm(h + self.dropout(msg))
        
        return h


class KGPathReasoner(nn.Module):
    """
    Multi-hop path reasoning over knowledge graph.
    
    Extracts and encodes paths between entity pairs for enhanced reasoning.
    """
    
    def __init__(
        self, 
        node_dim: int, 
        kg_embed_dim: int, 
        max_path_length: int = 3, 
        path_aggregation: str = "attention"
    ):
        super().__init__()
        self.max_path_length = max_path_length
        self.path_aggregation = path_aggregation
        self.kg_embed_dim = kg_embed_dim
        
        self.kg_proj = nn.Linear(kg_embed_dim, node_dim)
        
        self.path_encoder = nn.LSTM(
            node_dim * 2,  # relation + entity embeddings
            node_dim,
            batch_first=True,
            bidirectional=False
        )
        
        if path_aggregation == "attention":
            self.path_attention = nn.MultiheadAttention(
                embed_dim=node_dim,
                num_heads=4,
                batch_first=True
            )
        elif path_aggregation != "mean":
            raise ValueError(f"Unknown path_aggregation: {path_aggregation}")
        
        self.path_proj = nn.Linear(node_dim, node_dim)
    
    def encode_path(
        self, 
        path: List[Tuple[str, str]], 
        kg_entity_embeddings: Dict[str, torch.Tensor],
        kg_relation_embeddings: Dict[str, torch.Tensor],
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Encode a single path into an embedding.
        
        Args:
            path: List of (relation, entity) tuples
            kg_entity_embeddings: Dict mapping entity strings to embeddings
            kg_relation_embeddings: Dict mapping relation strings to embeddings
            device: Device to place tensors on
        
        Returns:
            Path embedding of shape (node_dim,)
        """
        if len(path) == 0:
            return torch.zeros(self.path_encoder.hidden_size, device=device)
        
        path_steps = []
        for relation, entity in path:
            rel_emb = kg_relation_embeddings.get(relation)
            ent_emb = kg_entity_embeddings.get(entity)
            
            if rel_emb is None:
                rel_emb = torch.zeros(self.kg_embed_dim, device=device)
            if ent_emb is None:
                ent_emb = torch.zeros(self.kg_embed_dim, device=device)
            
            rel_proj = self.kg_proj(rel_emb.unsqueeze(0))
            ent_proj = self.kg_proj(ent_emb.unsqueeze(0))
            
            step = torch.cat([rel_proj, ent_proj], dim=-1)
            path_steps.append(step)
        
        path_tensor = torch.cat(path_steps, dim=0).unsqueeze(0)
        path_encoded, (hidden, _) = self.path_encoder(path_tensor)
        path_embedding = hidden.squeeze(0)
        
        return path_embedding
    
    def forward(
        self, 
        entity_pairs: List[Tuple[int, int]], 
        paths: List[List[List[Tuple[str, str]]]],
        kg_entity_embeddings: Dict[str, torch.Tensor],
        kg_relation_embeddings: Dict[str, torch.Tensor],
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Encode and aggregate paths for entity pairs.
        
        Args:
            entity_pairs: List of (entity_i, entity_j) index pairs
            paths: List of lists of paths per pair
            kg_entity_embeddings: Dict mapping entity strings to embeddings
            kg_relation_embeddings: Dict mapping relation strings to embeddings
            device: Device to place tensors on
        
        Returns:
            Aggregated path embeddings of shape (len(entity_pairs), node_dim)
        """
        if len(entity_pairs) == 0:
            return torch.zeros((0, self.path_encoder.hidden_size), device=device)
        
        all_path_embeddings = []
        
        for pair_paths in paths:
            pair_embeddings = []
            for path in pair_paths[:10]:  # Limit paths per pair
                path_emb = self.encode_path(
                    path, kg_entity_embeddings, kg_relation_embeddings, device
                )
                pair_embeddings.append(path_emb)
            
            if len(pair_embeddings) == 0:
                pair_embeddings = [torch.zeros(self.path_encoder.hidden_size, device=device)]
            
            pair_embeddings_tensor = torch.stack(pair_embeddings).unsqueeze(0)
            
            if self.path_aggregation == "attention":
                aggregated, _ = self.path_attention(
                    pair_embeddings_tensor, pair_embeddings_tensor, pair_embeddings_tensor
                )
                aggregated = aggregated.mean(dim=1)
            else:
                aggregated = pair_embeddings_tensor.mean(dim=1)
            
            aggregated = self.path_proj(aggregated)
            all_path_embeddings.append(aggregated.squeeze(0))
        
        return torch.stack(all_path_embeddings)
