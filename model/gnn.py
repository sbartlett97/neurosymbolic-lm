"""Graph Neural Network modules for relational reasoning."""

from typing import List, Tuple, Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class AttentionGNN(nn.Module):
    """
    Graph Neural Network with attention-based message passing.

    Uses scaled dot-product attention between nodes instead of
    dense all-to-all message passing, improving relation reasoning
    quality and reducing noise from irrelevant connections.
    """

    def __init__(
        self,
        node_dim: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
    ):
        """
        Initialize attention-based GNN.

        Args:
            node_dim: Node feature dimension
            n_heads: Number of attention heads
            n_layers: Number of GNN layers
            dropout: Dropout rate
            edge_dim: Optional edge feature dimension
        """
        super().__init__()
        self.node_dim = node_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_dim = node_dim // n_heads
        self.dropout = nn.Dropout(dropout)

        assert node_dim % n_heads == 0, "node_dim must be divisible by n_heads"

        # Attention layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                GraphAttentionLayer(
                    node_dim=node_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            )

        # Final layer norm
        self.final_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        node_feats: torch.Tensor,
        adj_mask: Optional[torch.Tensor] = None,
        edge_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Propagate messages through the graph with attention.

        Args:
            node_feats: (B, N, D) node features
            adj_mask: (B, N, N) optional adjacency mask (1 = connected)
            edge_feats: (B, N, N, E) optional edge features

        Returns:
            Refined node features of shape (B, N, D)
        """
        h = node_feats

        for layer in self.layers:
            h = layer(h, adj_mask=adj_mask, edge_feats=edge_feats)

        h = self.final_norm(h)

        # Clamp for numerical stability
        h = h.clamp(-100, 100)

        return h


class GraphAttentionLayer(nn.Module):
    """
    Single graph attention layer with multi-head attention.

    Implements GAT-style attention with optional edge features.
    """

    def __init__(
        self,
        node_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
    ):
        """
        Initialize graph attention layer.

        Args:
            node_dim: Node feature dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            edge_dim: Optional edge feature dimension
        """
        super().__init__()
        self.node_dim = node_dim
        self.n_heads = n_heads
        self.head_dim = node_dim // n_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        self.o_proj = nn.Linear(node_dim, node_dim)

        # Edge feature projection (optional)
        self.edge_proj = None
        if edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, n_heads)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(node_dim, node_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim * 4, node_dim),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_feats: torch.Tensor,
        adj_mask: Optional[torch.Tensor] = None,
        edge_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply graph attention.

        Args:
            node_feats: (B, N, D) node features
            adj_mask: (B, N, N) adjacency mask (1 = connected, 0 = not)
            edge_feats: (B, N, N, E) edge features

        Returns:
            Updated node features (B, N, D)
        """
        B, N, D = node_feats.shape
        residual = node_feats

        # Pre-norm
        h = self.norm1(node_feats)

        # Compute Q, K, V
        Q = self.q_proj(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores: (B, H, N, N)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Add edge feature bias if available
        if edge_feats is not None and self.edge_proj is not None:
            edge_bias = self.edge_proj(edge_feats)  # (B, N, N, H)
            edge_bias = edge_bias.permute(0, 3, 1, 2)  # (B, H, N, N)
            attn_scores = attn_scores + edge_bias

        # Apply adjacency mask
        if adj_mask is not None:
            # adj_mask: (B, N, N) -> (B, 1, N, N)
            mask = adj_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # Handle all-masked rows
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (B, H, N, head_dim)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.o_proj(out)
        out = self.dropout(out)

        # Residual connection
        h = residual + out

        # FFN with residual
        h = h + self.ffn(self.norm2(h))

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

            # Use mean instead of sum for numerical stability
            msg = m.mean(dim=2)
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
        # hidden shape: (num_layers * num_directions, batch, hidden_size)
        # Get the last layer's hidden state for the single batch item
        path_embedding = hidden[-1, 0]  # (hidden_size,)
        
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
