"""Learned embedding encoder for ConceptNet relation types.

Provides a differentiable mapping from ConceptNet relation type indices
to dense embeddings, suitable for feeding into KGAwareGNN.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn

# ConceptNet relation types (47 core relations)
CONCEPTNET_RELATIONS = [
    "/r/RelatedTo",
    "/r/FormOf",
    "/r/IsA",
    "/r/PartOf",
    "/r/HasA",
    "/r/UsedFor",
    "/r/CapableOf",
    "/r/AtLocation",
    "/r/Causes",
    "/r/HasSubevent",
    "/r/HasFirstSubevent",
    "/r/HasLastSubevent",
    "/r/HasPrerequisite",
    "/r/HasProperty",
    "/r/MotivatedByGoal",
    "/r/ObstructedBy",
    "/r/Desires",
    "/r/CreatedBy",
    "/r/Synonym",
    "/r/Antonym",
    "/r/DistinctFrom",
    "/r/DerivedFrom",
    "/r/SymbolOf",
    "/r/DefinedAs",
    "/r/MannerOf",
    "/r/LocatedNear",
    "/r/HasContext",
    "/r/SimilarTo",
    "/r/EtymologicallyRelatedTo",
    "/r/EtymologicallyDerivedFrom",
    "/r/CausesDesire",
    "/r/MadeOf",
    "/r/ReceivesAction",
    "/r/ExternalURL",
    "/r/InstanceOf",
    "/r/NotDesires",
    "/r/NotUsedFor",
    "/r/NotCapableOf",
    "/r/NotHasProperty",
    "/r/NotIsA",
    "/r/NotHasA",
    "/r/dbpedia/genre",
    "/r/dbpedia/influencedBy",
    "/r/dbpedia/knownFor",
    "/r/dbpedia/occupation",
    "/r/dbpedia/language",
    "/r/dbpedia/field",
]


class KGRelationEncoder(nn.Module):
    """Learned embeddings for ConceptNet relation types.

    Maps integer relation type indices to dense vectors. Index 0 is a
    padding/unknown embedding. Indices 1..N correspond to the relations
    in CONCEPTNET_RELATIONS.

    Args:
        kg_embed_dim: Dimension of relation embeddings.
        num_relations: Number of relation types (excluding padding).
            Defaults to len(CONCEPTNET_RELATIONS).
    """

    def __init__(
        self,
        kg_embed_dim: int = 300,
        num_relations: Optional[int] = None,
    ):
        super().__init__()
        if num_relations is None:
            num_relations = len(CONCEPTNET_RELATIONS)

        self.num_relations = num_relations
        self.kg_embed_dim = kg_embed_dim

        # +1 for padding index 0
        self.embedding = nn.Embedding(
            num_relations + 1, kg_embed_dim, padding_idx=0
        )

        # Build string -> index mapping (1-indexed)
        self._relation_to_idx: Dict[str, int] = {
            rel: i + 1 for i, rel in enumerate(CONCEPTNET_RELATIONS[:num_relations])
        }

    def encode_relation(self, rel_str: str) -> int:
        """Map a relation string to its integer index.

        Returns 0 (padding) for unknown relations.
        """
        return self._relation_to_idx.get(rel_str, 0)

    def forward(self, relation_ids: torch.Tensor) -> torch.Tensor:
        """Produce dense relation embeddings.

        Args:
            relation_ids: (B, N, N) long tensor of relation type indices.

        Returns:
            (B, N, N, kg_embed_dim) dense relation embedding tensor.
        """
        return self.embedding(relation_ids)
