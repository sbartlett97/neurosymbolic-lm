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
