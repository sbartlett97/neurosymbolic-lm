"""Data loading and processing utilities."""

from .dataset import (
    ToyCognitiveDataset,
    statement_to_question,
    recalculate_entity_spans,
    generate_response_text,
)
from .collator import CognitiveCollator
from .chat_template import ChatTemplate, CHAT_SPECIAL_TOKENS
from .chat_converter import ChatConverter
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
    # Chat
    "ChatTemplate",
    "CHAT_SPECIAL_TOKENS",
    "ChatConverter",
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
