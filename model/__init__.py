"""Model components for the neurosymbolic architecture."""

from .encoders import (
    PositionalEmbedding,
    SimpleTransformerEncoder,
    PretrainedEncoderWrapper,
)
from .decoders import (
    SimpleTransformerDecoder,
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
# Optional experimental modules (referenced by ModelConfig flags that
# default to off). Their source files are not yet in the repo, so import
# them tolerantly — NeuroSymbolicLM imports them lazily only when the
# corresponding use_* flag is enabled.
try:
    from .entity_selector import SoftEntitySelector
except ImportError:
    SoftEntitySelector = None
try:
    from .linear_graph_transformer import LinearGraphTransformer, LinearGraphTransformerLayer
except ImportError:
    LinearGraphTransformer = LinearGraphTransformerLayer = None
try:
    from .global_workspace import GlobalWorkspace
except ImportError:
    GlobalWorkspace = None
from .kg_relation_encoder import (
    KGRelationEncoder,
    CONCEPTNET_RELATIONS,
)
from .logic import (
    SoftLogicConstraints,
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
    # KG Relation Encoder
    "KGRelationEncoder",
    "CONCEPTNET_RELATIONS",
    # Logic
    "SoftLogicConstraints",
    "pair_logits_to_matrix",
    # Entity selector
    "SoftEntitySelector",
    # Linear graph transformer
    "LinearGraphTransformer",
    "LinearGraphTransformerLayer",
    # Global workspace
    "GlobalWorkspace",
    # Main model
    "NeuroSymbolicLM",
    "compute_losses",
]
