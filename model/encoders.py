"""Encoder modules for the neurosymbolic model."""

from typing import Optional
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Learnable positional embeddings."""
    
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, d_model) * 0.01)
    
    def forward(self, x_len: int) -> torch.Tensor:
        """
        Get positional embeddings for sequence length.
        
        Args:
            x_len: Sequence length
        
        Returns:
            Positional embeddings of shape (1, x_len, d_model)
        """
        return self.pe[:x_len].unsqueeze(0)


class SimpleTransformerEncoder(nn.Module):
    """Simple transformer encoder with learnable positional embeddings."""
    
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 512, 
        nhead: int = 8, 
        nlayer: int = 6
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEmbedding(d_model, max_len=4096)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayer)
        self.d_model = d_model
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode input tokens.
        
        Args:
            input_ids: (B, L) input token IDs
            attn_mask: (B, L) attention mask (1 for real, 0 for padding)
        
        Returns:
            Hidden states of shape (B, L, d_model)
        """
        x = self.tok_emb(input_ids) + self.pos(input_ids.size(1))
        src_kpm = (attn_mask == 0) if attn_mask is not None else None
        return self.enc(x, src_key_padding_mask=src_kpm)


class PretrainedEncoderWrapper(nn.Module):
    """
    Wrapper for pre-trained transformer encoders (e.g., BERT).
    
    Matches the interface of SimpleTransformerEncoder.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", freeze_encoder: bool = False):
        super().__init__()
        from transformers import AutoModel
        
        self.pretrained_model = AutoModel.from_pretrained(model_name)
        self.d_model = self.pretrained_model.config.hidden_size
        self.freeze_encoder = freeze_encoder
        
        if freeze_encoder:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through pre-trained encoder.
        
        Args:
            input_ids: (B, L) token IDs
            attention_mask: (B, L) attention mask (1 for real tokens, 0 for padding)
        
        Returns:
            Hidden states of shape (B, L, d_model)
        """
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state
