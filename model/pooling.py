"""Pooling modules for aggregating token representations."""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQueryPool(nn.Module):
    """
    Multi-query attention pooling.
    
    Uses learnable query vectors to pool variable-length sequences
    into fixed-size representations.
    """
    
    def __init__(self, hidden_dim: int, n_queries: int = 6):
        super().__init__()
        self.nq = n_queries
        self.q = nn.Parameter(torch.randn(n_queries, hidden_dim) * 0.02)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(n_queries * hidden_dim, hidden_dim)
    
    def forward(
        self, 
        tokens: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pool tokens using multi-query attention.
        
        Args:
            tokens: (B, L, H) token representations
            mask: (B, L) attention mask (1 for real, 0 for padding)
        
        Returns:
            pooled: (B, H) pooled representation
            queries: (B, n_queries, H) individual query outputs
        """
        B, L, H = tokens.shape
        Q = self.q.unsqueeze(0).expand(B, -1, -1)  # (B, k, H)
        K = self.key(tokens)  # (B, L, H)
        V = self.value(tokens)
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (H ** 0.5)  # (B, k, L)
        
        if mask is not None:
            # Use -1e4 instead of -1e9 for float16 compatibility (max ~65504)
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e4)
        
        # Clamp scores for numerical stability
        scores = scores.clamp(-1e4, 1e4)
        attn = F.softmax(scores, dim=-1)
        
        # Handle case where softmax produces NaN (all tokens masked)
        if torch.isnan(attn).any():
            attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
            # If entire row is zero, use uniform distribution
            row_sums = attn.sum(dim=-1, keepdim=True)
            attn = torch.where(row_sums == 0, torch.ones_like(attn) / L, attn)
        
        pooled = torch.matmul(attn, V)  # (B, k, H)
        flat = pooled.view(B, -1)
        
        return self.out(flat), pooled


def span_mean_pool(
    tokens: torch.Tensor, 
    spans: List[List[Tuple[int, int]]]
) -> torch.Tensor:
    """
    Pool tokens by averaging over specified spans.
    
    Args:
        tokens: (B, L, H) token representations
        spans: Per-batch list of (start, end) inclusive token indices
    
    Returns:
        Pooled representations of shape (B, H)
    """
    B, L, H = tokens.shape
    out = tokens.new_zeros((B, H))
    
    for i in range(B):
        sps = spans[i]
        if len(sps) == 0:
            out[i] = tokens[i].mean(dim=0)
            continue
        
        reps = []
        for (s, e) in sps:
            seg = tokens[i, s:e+1]
            reps.append(seg.mean(dim=0))
        out[i] = torch.stack(reps, dim=0).mean(dim=0)
    
    return out
