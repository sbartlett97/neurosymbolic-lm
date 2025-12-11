"""Decoder modules for the neurosymbolic model.

Note: When using the T5-based NeuroSymbolicLM, the decoder is built-in
to the T5 model. This module provides a standalone decoder option
for custom architectures.
"""

from typing import Optional
import torch
import torch.nn as nn

from .encoders import PositionalEmbedding


class SimpleTransformerDecoder(nn.Module):
    """
    Simple transformer decoder with cross-attention to encoder and node features.
    
    Use this for custom encoder-decoder setups where you want full control
    over the architecture. For most cases, use the T5-based NeuroSymbolicLM instead.
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 512, 
        nhead: int = 8, 
        nlayer: int = 6
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEmbedding(d_model, max_len=2048)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, nhead, d_model * 4, batch_first=True) 
            for _ in range(nlayer)
        ])
        self.to_vocab = nn.Linear(d_model, vocab_size)
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self, 
        y_ids: torch.Tensor, 
        memory_tokens: torch.Tensor, 
        memory_nodes: torch.Tensor, 
        tgt_mask: Optional[torch.Tensor] = None,
        memory_token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode with cross-attention to encoder outputs and node features.
        
        Args:
            y_ids: (B, T) decoder input token IDs
            memory_tokens: (B, L, d_model) encoder outputs
            memory_nodes: (B, N, d_model) node representations
            tgt_mask: Optional causal mask for decoder
            memory_token_mask: Optional mask for encoder outputs
        
        Returns:
            Logits of shape (B, T, vocab_size)
        """
        b, t = y_ids.shape
        y = self.tok_emb(y_ids) + self.pos(t)
        
        # Concatenate encoder tokens and nodes as memory
        memory = torch.cat([memory_tokens, memory_nodes], dim=1)
        
        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(t, y.device)
        
        # Convert attention mask to key padding mask format
        memory_kpm = (memory_token_mask == 0) if memory_token_mask is not None else None
        
        out = y
        for layer in self.layers:
            out = layer(out, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_kpm)
        
        logits = self.to_vocab(out)
        return logits
