"""Entity and concept modules for symbolic representation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEntityClassifier(nn.Module):
    """
    Token-level entity type classifier.
    
    Predicts entity type for each token position.
    """
    
    def __init__(self, d_model: int, n_entity_types: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_entity_types)
    
    def forward(self, token_hidden: torch.Tensor) -> torch.Tensor:
        """
        Classify tokens by entity type.
        
        Args:
            token_hidden: (B, L, d_model) token representations
        
        Returns:
            Logits of shape (B, L, n_entity_types)
        """
        return self.fc(token_hidden)


class ConceptBank(nn.Module):
    """
    Learnable concept bank for soft concept assignment.
    
    Maintains a bank of concept embeddings and supports both
    discrete lookup and soft (probabilistic) assignment.
    """
    
    def __init__(self, n_concepts: int = 512, c_dim: int = 256):
        super().__init__()
        self.n = n_concepts
        self.dim = c_dim
        self.emb = nn.Parameter(torch.randn(n_concepts, c_dim) * 0.02)
    
    def lookup(self, concept_ids: torch.LongTensor) -> torch.Tensor:
        """
        Lookup concept embeddings by ID.
        
        Args:
            concept_ids: (B, M) concept indices
        
        Returns:
            Concept embeddings of shape (B, M, c_dim)
        """
        return F.embedding(concept_ids, self.emb)
    
    def soft_assign(self, query: torch.Tensor) -> tuple:
        """
        Compute soft assignment to concept bank.
        
        Args:
            query: (B, c_dim) query vectors
        
        Returns:
            vec: (B, c_dim) soft-weighted concept vector
            probs: (B, n_concepts) assignment probabilities
        """
        # Use eps to prevent NaN from zero-norm vectors
        eps = 1e-8
        qn = F.normalize(query + eps, dim=-1, eps=eps)
        bankn = F.normalize(self.emb + eps, dim=-1, eps=eps)
        sim = torch.matmul(qn, bankn.t())  # (B, n_concepts)
        # Clamp similarity to prevent extreme softmax values
        sim = sim.clamp(-20, 20)
        p = F.softmax(sim, dim=-1)
        vec = torch.matmul(p, self.emb)  # (B, c_dim)
        return vec, p
