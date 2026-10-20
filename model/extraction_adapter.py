"""Bidirectional extraction adapter for decoder-only backbones.

Decoder-only hidden states are causal: a mention's representation only sees
left context, which weakens entity typing. This adapter is a small
transformer with *bidirectional* attention that runs over the tapped
prompt states before the symbolic heads read them. It is never part of the
generation path, so bidirectionality here is legitimate — it restores
encoder-quality extraction without touching the backbone.

``n_layers=0`` yields an identity module: the pure-causal baseline used to
quantify the bidirectionality gap (Phase 1 ablation in
docs/DECODER_ONLY_MIGRATION.md).
"""

import torch
import torch.nn as nn


class ExtractionAdapter(nn.Module):
    """Bidirectional refinement of tapped decoder states.

    Args:
        d_model: Backbone hidden size.
        n_layers: Transformer layers (0 = identity / pure-causal baseline).
        n_heads: Attention heads.
        ff_mult: Feed-forward expansion factor.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 1,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers
        if n_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * ff_mult,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            self.encoder = None

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L, d_model) tapped backbone states.
            attention_mask: (B, L) 1 for real tokens, 0 for padding.

        Returns:
            (B, L, d_model) refined states, dtype matching adapter params.
        """
        if self.encoder is None:
            return hidden_states
        # Match adapter parameter dtype (backbone may run bf16, heads fp32)
        param_dtype = next(self.encoder.parameters()).dtype
        h = hidden_states.to(param_dtype)
        pad_mask = attention_mask == 0
        return self.encoder(h, src_key_padding_mask=pad_mask)
