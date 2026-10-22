"""Soft-token injection of symbolic node features into a causal backbone.

The decoder-only equivalent of the encoder-decoder architecture's
``cat([encoder_states, node_features])`` cross-attention memory: GNN node
features are projected into the backbone's embedding space and inserted as
extra token slots between prompt and response (causal order
``prompt < nodes < response``), so response tokens attend to both.

The learned scalar gate starts small so injection is near-invisible at
initialization and the model grows into using it; the gate's magnitude over
training (and a masked-nodes ablation) tells you whether the broadcast is
earning its keep.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodePrefixInjector(nn.Module):
    """Project node features into gated soft-token embeddings.

    Args:
        node_dim: Dimension of GNN node features.
        hidden_size: Backbone embedding size.
        gate_init: Initial value of the learned scalar gate.
    """

    def __init__(self, node_dim: int, hidden_size: int, gate_init: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(node_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_feats: (B, K, node_dim) refined node features.

        Returns:
            (B, K, hidden_size) soft-token embeddings.
        """
        return self.norm(self.proj(node_feats)) * self.gate


class NodeCrossAttention(nn.Module):
    """Gated cross-attention from backbone hidden states to graph nodes.

    The alternative to soft tokens (Flamingo style). The token sequence stays
    the same as the base model's, and the selected decoder layers add
    ``tanh(gate) * attn(hidden -> nodes)`` to their output. The gate starts
    at 0, so a new model is exactly its backbone. The 16 soft tokens sat
    between the prompt and the answer, and they cost about 9 acc_norm points
    on ARC-Easy whatever their content. This path has no such cost.

    Args:
        hidden_size: Backbone hidden size.
        node_dim: Dimension of GNN node features.
        n_heads: Attention heads (must divide hidden_size).
    """

    def __init__(self, hidden_size: int, node_dim: int, n_heads: int = 8):
        super().__init__()
        if hidden_size % n_heads:
            raise ValueError(f"hidden_size {hidden_size} is not divisible by "
                             f"n_heads {n_heads}")
        self.n_heads = n_heads
        self.norm = nn.LayerNorm(hidden_size)
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.kv = nn.Linear(node_dim, 2 * hidden_size, bias=False)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: torch.Tensor, nodes: torch.Tensor,
                valid: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (B, T, H) output of a decoder layer.
            nodes: (B, N, node_dim) refined node features.
            valid: (B, N) bool, False for empty node slots.
            pos_mask: (B, T, 1) 1 where a position may read the nodes.

        Returns:
            (B, T, H) the term to add to ``hidden``, in hidden's dtype.
        """
        B, T, H = hidden.shape
        N, h = nodes.shape[1], self.n_heads
        dtype = self.q.weight.dtype
        x = self.norm(hidden.to(dtype))
        q = self.q(x).view(B, T, h, H // h).transpose(1, 2)
        k, v = self.kv(nodes.to(dtype)).view(B, N, 2, h, H // h).permute(2, 0, 3, 1, 4)
        # A prompt with no nodes would mask every key (NaN). Let it read
        # slot 0, then zero its output below.
        has_nodes = valid.any(dim=-1)
        keys = valid.clone()
        keys[:, 0] |= ~has_nodes
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=keys[:, None, None, :])
        out = self.o(out.transpose(1, 2).reshape(B, T, H))
        scale = torch.tanh(self.gate) * has_nodes.to(out.dtype)[:, None, None]
        return (out * scale * pos_mask.to(out.dtype)).to(hidden.dtype)
