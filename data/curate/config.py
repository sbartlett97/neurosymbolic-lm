"""Configuration for dataset curation pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path


@dataclass
class CurationConfig:
    """Configuration for the dataset curation pipeline.

    Attributes:
        target_samples: Target number of curated samples
        sources: List of dataset sources (HuggingFace paths or 'wikipedia')
        source_ratios: Ratio of samples from each source (must sum to 1.0)
        output_dir: Directory for output files
        output_name: Base name for output files (<name>.jsonl, <name>_val.jsonl,
            <name>_vocab.json, <name>_checkpoint.json)
        overwrite: Replace existing output files of the same name
        val_ratio: Fraction of samples assigned to the validation split
        text_field: Text column name, or a {source: column} mapping
        llm_model: LLM model name/path for annotation (None = backend default)
        llm_backend: Backend for LLM inference ('vllm', 'transformers',
            'openai', 'anthropic', 'api', 'mock'). 'api' picks 'anthropic'
            for 'claude*' model names, else 'openai'. API keys come only
            from OPENAI_API_KEY / ANTHROPIC_API_KEY and are never stored here.
        api_concurrency: Parallel requests per batch for API backends
        api_effort: Effort or reasoning level for API backends (None =
            provider default)
        batch_size: Batch size for LLM inference
        max_workers: Number of parallel workers for processing
        should_respond_ratio: Ratio of samples with should_respond=1
        min_doc_length: Minimum document length in characters
        max_doc_length: Maximum document length in characters
        min_entities: Minimum entities per sample
        max_entities: Maximum entities per sample
        checkpoint_every: Save checkpoint every N samples
        resume_from: Path to checkpoint to resume from
        seed: Random seed for reproducibility
        enable_safety_filter: Use SafetyRegulator for content filtering
        safety_strictness: Safety filter strictness level
    """

    # Target and sources
    target_samples: int = 50000
    sources: List[str] = field(default_factory=lambda: ["allenai/c4", "wikipedia"])
    source_ratios: List[float] = field(default_factory=lambda: [0.6, 0.4])

    # Output
    output_dir: str = "data/curated"
    output_name: str = "curated"
    output_format: str = "jsonl"
    overwrite: bool = False
    val_ratio: float = 0.02

    # Source text column (None = per-source default, usually "text")
    text_field: Optional[Dict[str, str]] = None

    # Annotator selection
    # 'gliner'  — GLiNER2 schema-driven extraction, should_respond=0 samples
    # 'llm'     — legacy free-form LLM JSON annotation
    # 'hybrid'  — GLiNER2 extraction + LLM-generated QA pairs
    annotator: str = "gliner"

    # GLiNER2 settings
    gliner_model: str = "fastino/gliner2-base-v1"
    gliner_entity_threshold: float = 0.5
    gliner_relation_threshold: float = 0.4
    gliner_device: Optional[str] = None  # None = auto (cuda > mps > cpu)
    # Word tokens GLiNER2 encodes per text. None = the whole text. GLiNER2
    # drops words past this cap silently, so a cap below the document
    # length leaves late entities unlabeled (trained as "none").
    gliner_max_len: Optional[int] = None
    gliner_batch_size: Optional[int] = None  # texts per pass; None = 8 on CUDA, 1 elsewhere

    # LLM settings (used by 'llm' annotator and 'hybrid' QA generation)
    # None = backend default: Llama-3.1-8B-Instruct for vllm/transformers,
    # gpt-6-luna for openai, claude-haiku-4-5 for anthropic
    llm_model: Optional[str] = None
    llm_backend: str = "vllm"  # 'vllm', 'transformers', 'openai', 'anthropic', 'api'
    api_concurrency: int = 8
    api_effort: Optional[str] = None
    llm_quantization: Optional[str] = None  # "awq", "gptq", None
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.1
    # vLLM GPU memory fraction. None = 0.6 in hybrid mode (GLiNER2 shares
    # the GPU), else 0.9.
    vllm_gpu_memory_utilization: Optional[float] = None

    # Batch processing
    batch_size: int = 16
    max_workers: int = 4

    # Content ratios
    should_respond_ratio: float = 0.7  # 70% QA samples

    # Document filtering
    min_doc_length: int = 100
    # ~3.7k Qwen tokens (4.3 chars/token): fits the 4096-token training
    # context with room for the appended QA question
    max_doc_length: int = 16000
    min_entities: int = 1
    # Long documents hold many entities; dropped ones would train as "none"
    max_entities: int = 128

    # Checkpointing
    checkpoint_every: int = 1000
    resume_from: Optional[str] = None

    # Quality control
    min_entity_span_ratio: float = 0.5  # Min ratio of valid entity spans
    max_json_retries: int = 3

    # Safety
    enable_safety_filter: bool = True
    safety_strictness: str = "medium"
    safety_log_path: Optional[str] = None

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Validate configuration."""
        # Normalize source ratios
        if len(self.source_ratios) != len(self.sources):
            # Default to equal ratios
            self.source_ratios = [1.0 / len(self.sources)] * len(self.sources)

        ratio_sum = sum(self.source_ratios)
        if abs(ratio_sum - 1.0) > 0.01:
            self.source_ratios = [r / ratio_sum for r in self.source_ratios]

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_testing(cls) -> "CurationConfig":
        """Small config for testing the pipeline."""
        return cls(
            target_samples=100,
            sources=["wikipedia"],
            source_ratios=[1.0],
            batch_size=4,
            checkpoint_every=50,
            enable_safety_filter=False,
        )

    @classmethod
    def for_production(cls) -> "CurationConfig":
        """Production config for 50K samples."""
        return cls(
            target_samples=50000,
            sources=["allenai/c4", "wikipedia"],
            source_ratios=[0.6, 0.4],
            batch_size=16,
            llm_backend="vllm",
            checkpoint_every=1000,
            enable_safety_filter=True,
            safety_strictness="medium",
        )

    @classmethod
    def for_large_scale(cls) -> "CurationConfig":
        """Config for large-scale curation (100K+ samples)."""
        return cls(
            target_samples=100000,
            sources=["allenai/c4", "wikipedia", "HuggingFaceFW/fineweb"],
            source_ratios=[0.4, 0.35, 0.25],
            batch_size=32,
            llm_backend="vllm",
            checkpoint_every=2000,
            max_workers=8,
            enable_safety_filter=True,
            safety_strictness="high",
        )
