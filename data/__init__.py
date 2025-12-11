"""Data loading and processing utilities."""

from .dataset import (
    ToyCognitiveDataset,
    statement_to_question,
    recalculate_entity_spans,
    generate_response_text,
)
from .collator import CognitiveCollator
from .pipeline import (
    DataPipeline,
    DatasetConfig,
    ConvertedSample,
    DatasetConverter,
    DocREDConverter,
    MetaQAConverter,
    InstructionConverter,
    build_vocab_from_datasets,
)
from .staged_pipeline import (
    StagedDataPipeline,
    EntityRelationSample,
    EntityRelationDataset,
    InstructionSample,
    InstructionDataset,
    REBELLoader,
    DollyLoader,
    AlpacaLoader,
)

__all__ = [
    # Dataset
    "ToyCognitiveDataset",
    "CognitiveCollator",
    "statement_to_question",
    "recalculate_entity_spans",
    "generate_response_text",
    # Pipeline (unified)
    "DataPipeline",
    "DatasetConfig",
    "ConvertedSample",
    "DatasetConverter",
    "DocREDConverter",
    "MetaQAConverter",
    "InstructionConverter",
    "build_vocab_from_datasets",
    # Staged Pipeline (recommended)
    "StagedDataPipeline",
    "EntityRelationSample",
    "EntityRelationDataset",
    "InstructionSample",
    "InstructionDataset",
    "REBELLoader",
    "DollyLoader",
    "AlpacaLoader",
]
