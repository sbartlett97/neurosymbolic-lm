"""Decoder modules for the neurosymbolic model."""

from typing import Optional
import torch
import torch.nn as nn

from .encoders import PositionalEmbedding


class SimpleTransformerDecoder(nn.Module):
    """Simple transformer decoder with cross-attention to encoder and node features."""
    
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 512, 
        nhead: int = 8, 
        nlayer: int = 6
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEmbedding(d_model, max_len=2048)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, nhead, d_model * 4, batch_first=True) 
            for _ in range(nlayer)
        ])
        self.to_vocab = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
    
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
        
        memory_kpm = (memory_token_mask == 0) if memory_token_mask is not None else None
        
        out = y
        for layer in self.layers:
            out = layer(out, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_kpm)
        
        logits = self.to_vocab(out)
        return logits


class PretrainedDecoderWrapper(nn.Module):
    """
    Wrapper for pre-trained decoder models (e.g., T5, BART).
    
    Supports combining encoder outputs with node representations as memory.
    """
    
    def __init__(
        self, 
        model_name: str = "t5-small", 
        vocab_size: Optional[int] = None, 
        freeze_decoder: bool = False, 
        d_model: Optional[int] = None
    ):
        super().__init__()
        from transformers import AutoModelForSeq2SeqLM, AutoConfig
        
        config = AutoConfig.from_pretrained(model_name)
        self.pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        self.decoder = self.pretrained_model.decoder
        self.lm_head = self.pretrained_model.lm_head
        
        decoder_d_model = config.d_model if hasattr(config, 'd_model') else config.hidden_size
        
        if d_model is None:
            self.d_model = decoder_d_model
            self.input_proj = nn.Identity()
        else:
            self.d_model = d_model
            if decoder_d_model != d_model:
                self.input_proj = nn.Linear(d_model, decoder_d_model)
            else:
                self.input_proj = nn.Identity()
        
        self.vocab_size = vocab_size if vocab_size else config.vocab_size
        self.freeze_decoder = freeze_decoder
        self.decoder_d_model = decoder_d_model
        
        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.lm_head.parameters():
                param.requires_grad = False
    
    def forward(
        self, 
        y_ids: torch.Tensor, 
        memory_tokens: torch.Tensor, 
        memory_nodes: torch.Tensor, 
        tgt_mask: Optional[torch.Tensor] = None,
        memory_token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through pre-trained decoder.
        
        Args:
            y_ids: (B, T) decoder input token IDs
            memory_tokens: (B, L, d_model) encoder outputs
            memory_nodes: (B, N, d_model) node representations
            tgt_mask: Optional causal mask (not used in T5/BART)
            memory_token_mask: Optional mask for encoder outputs
        
        Returns:
            Logits of shape (B, T, vocab_size)
        """
        # Project memory to decoder dimension
        memory_tokens_proj = self.input_proj(memory_tokens)
        memory_nodes_proj = self.input_proj(memory_nodes)
        
        # Concatenate encoder outputs and nodes
        encoder_hidden_states = torch.cat([memory_tokens_proj, memory_nodes_proj], dim=1)
        
        # Create attention mask for combined memory
        if memory_token_mask is not None:
            B, L = memory_token_mask.shape
            N = memory_nodes.shape[1]
            node_mask = torch.ones(B, N, dtype=memory_token_mask.dtype, device=memory_token_mask.device)
            encoder_attention_mask = torch.cat([memory_token_mask, node_mask], dim=1)
        else:
            encoder_attention_mask = None
        
        # Get decoder outputs
        decoder_outputs = self.decoder(
            input_ids=y_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False
        )
        
        hidden_states = decoder_outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        return logits
