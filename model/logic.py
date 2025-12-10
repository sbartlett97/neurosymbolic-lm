"""Soft logic constraints and controller modules."""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def pair_logits_to_matrix(
    pair_logits_list: List[torch.Tensor], 
    pairs_index_map: List[torch.Tensor], 
    num_nodes: int, 
    n_rel: int, 
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert sparse pair logits to dense (B, N, N, R) tensor.
    
    Args:
        pair_logits_list: List of (P_b, R) tensors per batch
        pairs_index_map: List of (P_b, 2) index tensors
        num_nodes: Fixed node count N
        n_rel: Number of relation types R
        device: Device for output tensor
    
    Returns:
        Relation logits tensor of shape (B, N, N, R)
    """
    B = len(pair_logits_list)
    rel_tensor = torch.zeros((B, num_nodes, num_nodes, n_rel), device=device)
    
    for b in range(B):
        pl = pair_logits_list[b]
        if pl is None or pl.numel() == 0:
            continue
        idxmap = pairs_index_map[b]
        assert idxmap.shape[0] == pl.shape[0]
        
        for p in range(pl.shape[0]):
            i, j = int(idxmap[p, 0].item()), int(idxmap[p, 1].item())
            rel_tensor[b, i, j] = pl[p]
            rel_tensor[b, j, i] = pl[p]  # Symmetric
    
    return rel_tensor


class SoftLogicConstraints(nn.Module):
    """
    Differentiable soft constraints as auxiliary losses.
    
    Rules specify relationships between entity types and relations:
    (etype_a, etype_b, rel_idx, weight, polarity)
    
    polarity=1 encourages the relation, -1 discourages it.
    """
    
    def __init__(self, n_entity_types: int, n_relations: int):
        super().__init__()
        self.rules = []
        self.n_entity_types = n_entity_types
        self.n_relations = n_relations
    
    def add_rule(
        self, 
        etype_a: int, 
        etype_b: int, 
        rel_idx: int, 
        weight: float = 1.0, 
        polarity: int = 1
    ):
        """
        Add a soft logic rule.
        
        Args:
            etype_a: Entity type index for first entity
            etype_b: Entity type index for second entity
            rel_idx: Relation index
            weight: Rule weight (default 1.0)
            polarity: 1 to encourage, -1 to discourage
        """
        assert 0 <= etype_a < self.n_entity_types
        assert 0 <= etype_b < self.n_entity_types
        assert 0 <= rel_idx < self.n_relations
        assert polarity in (-1, 1)
        self.rules.append((etype_a, etype_b, rel_idx, float(weight), int(polarity)))
    
    def forward(
        self, 
        entity_type_probs: torch.Tensor, 
        rel_logits_matrix: torch.Tensor
    ) -> tuple:
        """
        Compute soft logic constraint loss.
        
        Args:
            entity_type_probs: (B, N, E) probability over entity types
            rel_logits_matrix: (B, N, N, R) relation logits
        
        Returns:
            total_loss: Scalar loss tensor
            details: Dict with per-rule losses
        """
        device = entity_type_probs.device
        B, N, E = entity_type_probs.shape
        _, _, _, R = rel_logits_matrix.shape
        assert E == self.n_entity_types and R == self.n_relations
        
        total_loss = torch.tensor(0.0, device=device)
        details = {"rules": []}
        
        if len(self.rules) == 0:
            return total_loss, details
        
        for (etype_a, etype_b, rel_idx, weight, polarity) in self.rules:
            p_a = entity_type_probs[:, :, etype_a].unsqueeze(-1)  # (B, N, 1)
            p_b = entity_type_probs[:, :, etype_b].unsqueeze(1)   # (B, 1, N)
            type_pair_prob = (p_a * p_b).squeeze(-1)  # (B, N, N)
            
            # Work with logits directly for numerical stability
            rel_logits = rel_logits_matrix[:, :, :, rel_idx]  # (B, N, N)
            
            # Compute weighted average of logits
            weighted_logits = (type_pair_prob * rel_logits).sum(dim=(1, 2))
            normalizer = type_pair_prob.sum(dim=(1, 2)).clamp(min=1e-6)
            expected_logit = weighted_logits / normalizer  # (B,)
            
            # Target: 1 for polarity=1 (encourage), 0 for polarity=-1 (discourage)
            if polarity == 1:
                target = torch.ones_like(expected_logit)
            else:
                target = torch.zeros_like(expected_logit)
            
            # Use BCEWithLogits - safe with autocast/AMP
            rule_loss = F.binary_cross_entropy_with_logits(expected_logit, target)
            scaled = weight * rule_loss
            total_loss = total_loss + scaled
            
            details["rules"].append({
                "etype_a": etype_a, 
                "etype_b": etype_b, 
                "rel_idx": rel_idx, 
                "weight": weight, 
                "polarity": polarity, 
                "loss": rule_loss.detach().cpu()
            })
        
        return total_loss, details


class Controller(nn.Module):
    """
    Response controller for deciding model behavior.
    
    Outputs logits for [answer, abstain, ask_clarify].
    """
    
    def __init__(self, token_dim: int, node_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(token_dim + node_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, 3)
        )
    
    def forward(
        self, 
        token_pool: torch.Tensor, 
        node_pool: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute controller decision logits.
        
        Args:
            token_pool: (B, token_dim) pooled token representation
            node_pool: (B, node_dim) pooled node representation
        
        Returns:
            Logits of shape (B, 3) for [answer, abstain, ask_clarify]
        """
        x = torch.cat([token_pool, node_pool], dim=-1)
        return self.fc(x)
