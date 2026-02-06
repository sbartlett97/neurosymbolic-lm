"""Dataset curation pipeline for generating neurosymbolic training data.

This module provides tools for curating datasets with LLM-generated annotations:
- Entity extraction with character spans
- Concept assignment (1-3 concepts per entity)
- Relation triplet extraction
- QA pair generation

Usage:
    python data/curate_dataset.py \
        --target-samples 50000 \
        --sources allenai/c4 wikipedia \
        --output-dir data/curated
"""

from .config import CurationConfig
from .source_loader import SourceLoader
from .document_filter import DocumentFilter
from .llm_annotator import LLMAnnotator
from .quality_control import QualityControl
from .output_writer import OutputWriter

__all__ = [
    "CurationConfig",
    "SourceLoader",
    "DocumentFilter",
    "LLMAnnotator",
    "QualityControl",
    "OutputWriter",
]
