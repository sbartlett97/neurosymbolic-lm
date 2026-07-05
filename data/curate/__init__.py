"""Dataset curation pipeline for generating neurosymbolic training data.

This module provides tools for curating datasets with schema-driven
(GLiNER2) or LLM-generated annotations:
- Entity extraction with character spans and entity types
- Concept assignment (1-3 concepts per entity) from a diverse taxonomy
- Relation triplet extraction
- QA pair generation (hybrid mode)

Usage:
    python data/curate_dataset.py \
        --target-samples 50000 \
        --annotator gliner \
        --sources allenai/c4 wikipedia \
        --output-dir data/curated
"""

from .config import CurationConfig
from .source_loader import SourceLoader
from .document_filter import DocumentFilter
from .llm_annotator import LLMAnnotator
from .gliner_annotator import GLiNER2Annotator
from .quality_control import QualityControl
from .output_writer import OutputWriter
from .taxonomy import Taxonomy, get_default_taxonomy
from .code_annotator import CodeAnnotator
from .trace_annotator import TraceAnnotator
from .trace_loader import TraceDocument, TraceSourceLoader

__all__ = [
    "CurationConfig",
    "SourceLoader",
    "DocumentFilter",
    "LLMAnnotator",
    "GLiNER2Annotator",
    "QualityControl",
    "OutputWriter",
    "Taxonomy",
    "get_default_taxonomy",
    "CodeAnnotator",
    "TraceAnnotator",
    "TraceDocument",
    "TraceSourceLoader",
]
