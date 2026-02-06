"""Soft logic constraints module.

Provides differentiable soft logic constraints for neurosymbolic reasoning:
- Simple pairwise constraints (entity type -> relation preferences)
- First-order logic with conjunction, disjunction, transitivity
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
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
            # Broadcast produces (B, N, N) directly - no squeeze needed
            type_pair_prob = p_a * p_b  # (B, N, N)
            
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


class LogicOperator(Enum):
    """Logical operators for first-order logic rules."""
    AND = "and"
    OR = "or"
    IMPLIES = "implies"
    NOT = "not"


@dataclass
class FOLAtom:
    """A first-order logic atom representing a relation between entities.

    Example: located_in(X, Y) where X, Y are entity variables.
    """
    relation: int  # Relation type index
    arg1: str  # Variable name for first argument
    arg2: str  # Variable name for second argument
    negated: bool = False

    def __repr__(self):
        neg = "NOT " if self.negated else ""
        return f"{neg}R{self.relation}({self.arg1}, {self.arg2})"


@dataclass
class FOLRule:
    """A first-order logic rule.

    Supports rules like:
    - Transitivity: IF R(A,B) AND R(B,C) THEN R(A,C)
    - Implication: IF R1(A,B) THEN R2(A,B)
    - Mutual exclusion: NOT (R1(A,B) AND R2(A,B))
    """
    name: str
    antecedent: List[FOLAtom]  # Premises (body of rule)
    consequent: List[FOLAtom]  # Conclusion (head of rule)
    operator: LogicOperator = LogicOperator.AND  # How to combine antecedent atoms
    weight: float = 1.0


class FirstOrderLogicConstraints(nn.Module):
    """
    First-order logic constraints with conjunction, disjunction, and transitivity.

    Implements differentiable soft logic using t-norms for conjunction
    and t-conorms for disjunction, enabling gradient-based optimization.

    T-norm semantics:
    - AND (product t-norm): p(A AND B) = p(A) * p(B)
    - OR (probabilistic t-conorm): p(A OR B) = p(A) + p(B) - p(A) * p(B)
    - NOT: p(NOT A) = 1 - p(A)
    - IMPLIES: p(A -> B) = 1 - p(A) + p(A) * p(B)

    Example rules:
    - Transitivity: IF located_in(A,B) AND located_in(B,C) THEN located_in(A,C)
    - Type constraint: IF person(X) AND organization(Y) THEN works_for(X,Y) OR member_of(X,Y)
    - Mutual exclusion: NOT (child_of(X,Y) AND parent_of(X,Y))
    """

    def __init__(
        self,
        n_entity_types: int,
        n_relations: int,
        use_godel_tnorm: bool = False,
    ):
        """
        Initialize FOL constraints.

        Args:
            n_entity_types: Number of entity type classes
            n_relations: Number of relation types
            use_godel_tnorm: Use Godel t-norm (min/max) instead of product
        """
        super().__init__()
        self.n_entity_types = n_entity_types
        self.n_relations = n_relations
        self.use_godel_tnorm = use_godel_tnorm
        self.rules: List[FOLRule] = []

    def add_rule(self, rule: FOLRule):
        """Add a first-order logic rule."""
        self.rules.append(rule)

    def add_transitivity_rule(
        self,
        relation: int,
        weight: float = 1.0,
        name: Optional[str] = None,
    ):
        """
        Add a transitivity rule: IF R(A,B) AND R(B,C) THEN R(A,C).

        Args:
            relation: Relation index
            weight: Rule weight
            name: Optional rule name
        """
        if name is None:
            name = f"transitivity_r{relation}"

        rule = FOLRule(
            name=name,
            antecedent=[
                FOLAtom(relation=relation, arg1="A", arg2="B"),
                FOLAtom(relation=relation, arg1="B", arg2="C"),
            ],
            consequent=[
                FOLAtom(relation=relation, arg1="A", arg2="C"),
            ],
            operator=LogicOperator.AND,
            weight=weight,
        )
        self.rules.append(rule)

    def add_implication_rule(
        self,
        premise_relation: int,
        conclusion_relation: int,
        weight: float = 1.0,
        name: Optional[str] = None,
    ):
        """
        Add an implication rule: IF R1(A,B) THEN R2(A,B).

        Args:
            premise_relation: Relation in premise
            conclusion_relation: Relation in conclusion
            weight: Rule weight
            name: Optional rule name
        """
        if name is None:
            name = f"implies_r{premise_relation}_r{conclusion_relation}"

        rule = FOLRule(
            name=name,
            antecedent=[
                FOLAtom(relation=premise_relation, arg1="A", arg2="B"),
            ],
            consequent=[
                FOLAtom(relation=conclusion_relation, arg1="A", arg2="B"),
            ],
            operator=LogicOperator.IMPLIES,
            weight=weight,
        )
        self.rules.append(rule)

    def add_mutual_exclusion_rule(
        self,
        relation1: int,
        relation2: int,
        weight: float = 1.0,
        name: Optional[str] = None,
    ):
        """
        Add a mutual exclusion rule: NOT (R1(A,B) AND R2(A,B)).

        Args:
            relation1: First relation
            relation2: Second relation
            weight: Rule weight
            name: Optional rule name
        """
        if name is None:
            name = f"mutex_r{relation1}_r{relation2}"

        rule = FOLRule(
            name=name,
            antecedent=[
                FOLAtom(relation=relation1, arg1="A", arg2="B"),
                FOLAtom(relation=relation2, arg1="A", arg2="B"),
            ],
            consequent=[],  # Empty consequent means we penalize the antecedent
            operator=LogicOperator.NOT,
            weight=weight,
        )
        self.rules.append(rule)

    def add_disjunction_rule(
        self,
        premise_relation: int,
        conclusion_relations: List[int],
        weight: float = 1.0,
        name: Optional[str] = None,
    ):
        """
        Add a disjunction rule: IF R1(A,B) THEN R2(A,B) OR R3(A,B).

        Args:
            premise_relation: Relation in premise
            conclusion_relations: List of relations in disjunctive conclusion
            weight: Rule weight
            name: Optional rule name
        """
        if name is None:
            name = f"disjunct_r{premise_relation}"

        rule = FOLRule(
            name=name,
            antecedent=[
                FOLAtom(relation=premise_relation, arg1="A", arg2="B"),
            ],
            consequent=[
                FOLAtom(relation=r, arg1="A", arg2="B")
                for r in conclusion_relations
            ],
            operator=LogicOperator.OR,
            weight=weight,
        )
        self.rules.append(rule)

    def _t_norm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute t-norm (fuzzy AND)."""
        if self.use_godel_tnorm:
            return torch.min(a, b)
        else:
            # Product t-norm
            return a * b

    def _t_conorm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute t-conorm (fuzzy OR)."""
        if self.use_godel_tnorm:
            return torch.max(a, b)
        else:
            # Probabilistic t-conorm
            return a + b - a * b

    def _negation(self, a: torch.Tensor) -> torch.Tensor:
        """Compute fuzzy negation."""
        return 1.0 - a

    def _implication(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute fuzzy implication (Reichenbach)."""
        return 1.0 - a + a * b

    def _get_relation_prob(
        self,
        rel_probs: torch.Tensor,
        relation: int,
        i: int,
        j: int,
    ) -> torch.Tensor:
        """Get probability of relation(i, j)."""
        return rel_probs[:, i, j, relation]

    def _evaluate_transitivity_rule(
        self,
        rule: FOLRule,
        rel_probs: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """
        Evaluate transitivity rule over all entity triplets.

        IF R(A,B) AND R(B,C) THEN R(A,C)
        """
        B = rel_probs.shape[0]
        device = rel_probs.device
        rel = rule.antecedent[0].relation

        total_loss = torch.tensor(0.0, device=device)
        count = 0

        # Iterate over all triplets (A, B, C)
        for a in range(N):
            for b in range(N):
                if a == b:
                    continue
                for c in range(N):
                    if c == a or c == b:
                        continue

                    # P(R(A,B) AND R(B,C))
                    p_ab = rel_probs[:, a, b, rel]
                    p_bc = rel_probs[:, b, c, rel]
                    p_antecedent = self._t_norm(p_ab, p_bc)

                    # P(R(A,C))
                    p_ac = rel_probs[:, a, c, rel]

                    # Implication loss: 1 - P(antecedent -> consequent)
                    p_rule = self._implication(p_antecedent, p_ac)
                    loss = (1.0 - p_rule).mean()
                    total_loss = total_loss + loss
                    count += 1

        if count > 0:
            total_loss = total_loss / count

        return total_loss

    def _evaluate_implication_rule(
        self,
        rule: FOLRule,
        rel_probs: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """Evaluate implication rule: IF R1(A,B) THEN R2(A,B)."""
        device = rel_probs.device
        r1 = rule.antecedent[0].relation
        r2 = rule.consequent[0].relation

        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for a in range(N):
            for b in range(N):
                if a == b:
                    continue

                p_premise = rel_probs[:, a, b, r1]
                p_conclusion = rel_probs[:, a, b, r2]
                p_rule = self._implication(p_premise, p_conclusion)
                loss = (1.0 - p_rule).mean()
                total_loss = total_loss + loss
                count += 1

        if count > 0:
            total_loss = total_loss / count

        return total_loss

    def _evaluate_mutex_rule(
        self,
        rule: FOLRule,
        rel_probs: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """Evaluate mutual exclusion: NOT (R1(A,B) AND R2(A,B))."""
        device = rel_probs.device
        r1 = rule.antecedent[0].relation
        r2 = rule.antecedent[1].relation

        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for a in range(N):
            for b in range(N):
                if a == b:
                    continue

                p_r1 = rel_probs[:, a, b, r1]
                p_r2 = rel_probs[:, a, b, r2]
                p_both = self._t_norm(p_r1, p_r2)
                # Penalize: want P(R1 AND R2) = 0
                loss = p_both.mean()
                total_loss = total_loss + loss
                count += 1

        if count > 0:
            total_loss = total_loss / count

        return total_loss

    def _evaluate_disjunction_rule(
        self,
        rule: FOLRule,
        rel_probs: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """Evaluate disjunction: IF R1(A,B) THEN R2(A,B) OR R3(A,B)."""
        device = rel_probs.device
        r_premise = rule.antecedent[0].relation
        r_conclusions = [atom.relation for atom in rule.consequent]

        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for a in range(N):
            for b in range(N):
                if a == b:
                    continue

                p_premise = rel_probs[:, a, b, r_premise]

                # Compute disjunction of conclusions
                p_disjunct = rel_probs[:, a, b, r_conclusions[0]]
                for r in r_conclusions[1:]:
                    p_disjunct = self._t_conorm(p_disjunct, rel_probs[:, a, b, r])

                p_rule = self._implication(p_premise, p_disjunct)
                loss = (1.0 - p_rule).mean()
                total_loss = total_loss + loss
                count += 1

        if count > 0:
            total_loss = total_loss / count

        return total_loss

    def forward(
        self,
        rel_logits_matrix: torch.Tensor,
        entity_type_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute first-order logic constraint loss.

        Args:
            rel_logits_matrix: (B, N, N, R) relation logits
            entity_type_probs: (B, N, E) entity type probabilities (optional)

        Returns:
            total_loss: Scalar loss tensor
            details: Dict with per-rule losses
        """
        device = rel_logits_matrix.device
        B, N, _, R = rel_logits_matrix.shape

        # Convert logits to probabilities
        rel_probs = torch.sigmoid(rel_logits_matrix)

        total_loss = torch.tensor(0.0, device=device)
        details = {"rules": []}

        if len(self.rules) == 0:
            return total_loss, details

        for rule in self.rules:
            # Determine rule type and evaluate
            if rule.operator == LogicOperator.NOT:
                rule_loss = self._evaluate_mutex_rule(rule, rel_probs, N)
            elif rule.operator == LogicOperator.OR:
                rule_loss = self._evaluate_disjunction_rule(rule, rel_probs, N)
            elif len(rule.antecedent) == 2 and rule.antecedent[0].arg2 == rule.antecedent[1].arg1:
                # Transitivity pattern
                rule_loss = self._evaluate_transitivity_rule(rule, rel_probs, N)
            else:
                # Standard implication
                rule_loss = self._evaluate_implication_rule(rule, rel_probs, N)

            weighted_loss = rule.weight * rule_loss
            total_loss = total_loss + weighted_loss

            details["rules"].append({
                "name": rule.name,
                "weight": rule.weight,
                "loss": rule_loss.detach().cpu().item(),
            })

        return total_loss, details
