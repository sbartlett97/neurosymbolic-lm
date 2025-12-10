"""Model components for the neurosymbolic architecture."""

from .encoders import (
    PositionalEmbedding,
    SimpleTransformerEncoder,
    PretrainedEncoderWrapper,
)
from .decoders import (
    SimpleTransformerDecoder,
    PretrainedDecoderWrapper,
)
from .pooling import (
    MultiQueryPool,
    span_mean_pool,
)
from .entity import (
    TokenEntityClassifier,
    ConceptBank,
)
from .gnn import (
    SimpleGNN,
    KGAwareGNN,
    KGPathReasoner,
)
from .logic import (
    SoftLogicConstraints,
    Controller,
    pair_logits_to_matrix,
)
from .neurosymbolic import (
    NeuroSymbolicLM,
    compute_losses,
)

__all__ = [
    # Encoders
    "PositionalEmbedding",
    "SimpleTransformerEncoder",
    "PretrainedEncoderWrapper",
    # Decoders
    "SimpleTransformerDecoder",
    "PretrainedDecoderWrapper",
    # Pooling
    "MultiQueryPool",
    "span_mean_pool",
    # Entity/Concept
    "TokenEntityClassifier",
    "ConceptBank",
    # GNN
    "SimpleGNN",
    "KGAwareGNN",
    "KGPathReasoner",
    # Logic/Control
    "SoftLogicConstraints",
    "Controller",
    "pair_logits_to_matrix",
    # Main model
    "NeuroSymbolicLM",
    "compute_losses",
]
