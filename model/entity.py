"""Entity and concept modules for symbolic representation."""

from typing import Optional, List, Tuple
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


class ContextAwareEntityClassifier(nn.Module):
    """
    Context-aware entity type classifier with self-attention.

    Uses self-attention to incorporate surrounding context before
    classifying entity types, improving accuracy for ambiguous entities.
    """

    def __init__(
        self,
        d_model: int,
        n_entity_types: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        context_layers: int = 1,
    ):
        """
        Initialize context-aware entity classifier.

        Args:
            d_model: Hidden dimension
            n_entity_types: Number of entity type classes
            n_heads: Number of attention heads
            dropout: Dropout rate
            context_layers: Number of self-attention layers
        """
        super().__init__()
        self.d_model = d_model
        self.n_entity_types = n_entity_types

        # Self-attention layers for context
        self.context_layers = nn.ModuleList()
        for _ in range(context_layers):
            self.context_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,  # Pre-LN for better training
                )
            )

        # Classification head with intermediate layer
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_entity_types),
        )

        # Layer norm for stability
        self.pre_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        token_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Classify tokens by entity type with context awareness.

        Args:
            token_hidden: (B, L, d_model) token representations
            attention_mask: (B, L) attention mask (1 = attend, 0 = ignore)

        Returns:
            Logits of shape (B, L, n_entity_types)
        """
        # Normalize input
        h = self.pre_norm(token_hidden)

        # Create attention mask for transformer (True = ignore)
        if attention_mask is not None:
            # Convert from (B, L) to (B, L) bool mask where True means ignore
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None

        # Apply self-attention layers
        for layer in self.context_layers:
            h = layer(h, src_key_padding_mask=key_padding_mask)

        # Residual connection
        h = h + token_hidden

        # Classify
        logits = self.classifier(h)

        return logits

    def get_entity_confidence(
        self,
        token_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        top_k: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get entity type predictions with confidence scores.

        Args:
            token_hidden: (B, L, d_model) token representations
            attention_mask: (B, L) attention mask
            top_k: Number of top predictions to return

        Returns:
            logits: (B, L, n_entity_types) raw logits
            top_types: (B, L, top_k) top-k entity type indices
            top_probs: (B, L, top_k) top-k probabilities
        """
        logits = self.forward(token_hidden, attention_mask)
        probs = F.softmax(logits, dim=-1)

        top_probs, top_types = torch.topk(probs, k=min(top_k, self.n_entity_types), dim=-1)

        return logits, top_types, top_probs


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
        # F.normalize already handles zero-norm vectors via its eps parameter
        eps = 1e-8
        qn = F.normalize(query, dim=-1, eps=eps)
        bankn = F.normalize(self.emb, dim=-1, eps=eps)
        sim = torch.matmul(qn, bankn.t())  # (B, n_concepts)
        # Clamp similarity to prevent extreme softmax values
        sim = sim.clamp(-20, 20)
        p = F.softmax(sim, dim=-1)
        vec = torch.matmul(p, self.emb)  # (B, c_dim)
        return vec, p


class HierarchicalConceptBank(nn.Module):
    """
    Hierarchical concept bank with parent-child relationships.

    Enables multi-scale semantic reasoning by organizing concepts
    into a hierarchy (e.g., "dog" -> "animal" -> "living_thing").

    The hierarchy is represented as:
    - Level 0: Most specific concepts (leaf nodes)
    - Level 1: Intermediate concepts
    - Level 2+: More abstract concepts (root-like)

    Supports:
    - Hierarchical soft assignment (aggregate up the tree)
    - Multi-level concept retrieval
    - Dynamic hierarchy updates during training
    """

    def __init__(
        self,
        n_concepts_per_level: List[int] = None,
        c_dim: int = 256,
        use_learned_hierarchy: bool = True,
    ):
        """
        Initialize hierarchical concept bank.

        Args:
            n_concepts_per_level: Number of concepts at each level [leaf, ..., root]
                Default: [512, 128, 32] (3 levels)
            c_dim: Concept embedding dimension
            use_learned_hierarchy: Learn parent assignments vs fixed
        """
        super().__init__()

        if n_concepts_per_level is None:
            n_concepts_per_level = [512, 128, 32]

        self.n_levels = len(n_concepts_per_level)
        self.n_concepts_per_level = n_concepts_per_level
        self.dim = c_dim
        self.use_learned_hierarchy = use_learned_hierarchy

        # Concept embeddings for each level
        self.level_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(n, c_dim) * 0.02)
            for n in n_concepts_per_level
        ])

        # Parent assignment matrices (child -> parent)
        # Soft assignments allow learning the hierarchy
        if use_learned_hierarchy:
            self.parent_assignments = nn.ParameterList()
            for i in range(self.n_levels - 1):
                # Assignment from level i to level i+1
                n_children = n_concepts_per_level[i]
                n_parents = n_concepts_per_level[i + 1]
                # Initialize with slight preference for certain parents
                init = torch.randn(n_children, n_parents) * 0.1
                self.parent_assignments.append(nn.Parameter(init))
        else:
            self.parent_assignments = None

        # Layer norms for stability
        self.level_norms = nn.ModuleList([
            nn.LayerNorm(c_dim) for _ in n_concepts_per_level
        ])

        # Total concepts (for compatibility)
        self.n = sum(n_concepts_per_level)

    def get_level_embeddings(self, level: int, normalize: bool = True) -> torch.Tensor:
        """Get embeddings for a specific level."""
        emb = self.level_embeddings[level]
        if normalize:
            emb = self.level_norms[level](emb)
        return emb

    def get_parent_probs(self, level: int) -> torch.Tensor:
        """
        Get soft parent assignment probabilities.

        Args:
            level: Child level (0 = leaf)

        Returns:
            (n_children, n_parents) probability matrix
        """
        if self.parent_assignments is None or level >= self.n_levels - 1:
            return None

        logits = self.parent_assignments[level]
        return F.softmax(logits, dim=-1)

    def lookup(self, concept_ids: torch.LongTensor, level: int = 0) -> torch.Tensor:
        """
        Lookup concept embeddings by ID at a specific level.

        Args:
            concept_ids: (B, M) concept indices
            level: Hierarchy level (0 = most specific)

        Returns:
            Concept embeddings of shape (B, M, c_dim)
        """
        emb = self.get_level_embeddings(level)
        return F.embedding(concept_ids, emb)

    def soft_assign(
        self,
        query: torch.Tensor,
        level: int = 0,
        aggregate_hierarchy: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Compute soft assignment with optional hierarchical aggregation.

        Args:
            query: (B, c_dim) or (B, L, c_dim) query vectors
            level: Starting level for assignment
            aggregate_hierarchy: Whether to aggregate up the hierarchy

        Returns:
            vec: Soft-weighted concept vector (same shape as query)
            probs: Assignment probabilities at base level
            hierarchy_probs: List of probs at each level (if aggregate)
        """
        eps = 1e-8

        # Handle both 2D and 3D inputs
        input_shape = query.shape
        if query.dim() == 3:
            B, L, D = query.shape
            query = query.view(B * L, D)
        else:
            L = None

        # Normalize query
        qn = F.normalize(query, dim=-1, eps=eps)

        # Get base level embeddings
        base_emb = self.get_level_embeddings(level)
        base_emb_n = F.normalize(base_emb, dim=-1, eps=eps)

        # Compute similarity and soft assignment
        sim = torch.matmul(qn, base_emb_n.t())
        sim = sim.clamp(-20, 20)
        base_probs = F.softmax(sim, dim=-1)

        # Compute weighted embedding
        vec = torch.matmul(base_probs, base_emb)

        hierarchy_probs = [base_probs] if aggregate_hierarchy else None

        # Aggregate up hierarchy if requested
        if aggregate_hierarchy and self.parent_assignments is not None:
            current_probs = base_probs

            for i in range(level, self.n_levels - 1):
                parent_probs = self.get_parent_probs(i)  # (n_child, n_parent)

                # Propagate probabilities up: p(parent) = sum_child p(child) * p(parent|child)
                next_probs = torch.matmul(current_probs, parent_probs)

                hierarchy_probs.append(next_probs)

                # Add parent-level contribution to embedding
                parent_emb = self.get_level_embeddings(i + 1)
                vec = vec + 0.5 * torch.matmul(next_probs, parent_emb)

                current_probs = next_probs

        # Reshape output if input was 3D
        if L is not None:
            vec = vec.view(B, L, -1)
            base_probs = base_probs.view(B, L, -1)
            if hierarchy_probs:
                hierarchy_probs = [p.view(B, L, -1) for p in hierarchy_probs]

        return vec, base_probs, hierarchy_probs

    def get_concept_ancestors(
        self,
        concept_ids: torch.LongTensor,
        level: int = 0,
    ) -> List[torch.Tensor]:
        """
        Get ancestor concepts for given concept IDs.

        Args:
            concept_ids: (B,) or (B, M) concept indices at base level
            level: Starting level

        Returns:
            List of ancestor indices at each higher level
        """
        if self.parent_assignments is None:
            return []

        ancestors = []
        current_ids = concept_ids

        for i in range(level, self.n_levels - 1):
            parent_probs = self.get_parent_probs(i)
            # Get most likely parent
            if current_ids.dim() == 1:
                parent_probs_selected = parent_probs[current_ids]
            else:
                # Batch selection
                parent_probs_selected = parent_probs[current_ids.view(-1)].view(
                    *current_ids.shape, -1
                )

            parent_ids = parent_probs_selected.argmax(dim=-1)
            ancestors.append(parent_ids)
            current_ids = parent_ids

        return ancestors

    def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Default forward pass: soft assignment at leaf level with hierarchy.

        Args:
            query: (B, c_dim) query vectors

        Returns:
            vec: (B, c_dim) soft-weighted concept vector
            probs: (B, n_concepts_level_0) assignment probabilities
        """
        vec, probs, _ = self.soft_assign(query, level=0, aggregate_hierarchy=True)
        return vec, probs
