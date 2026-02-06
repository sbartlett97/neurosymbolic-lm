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
        llm_model: LLM model name/path for annotation
        llm_backend: Backend for LLM inference ('vllm', 'transformers', 'api')
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
    output_format: str = "jsonl"

    # LLM settings
    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    llm_backend: str = "vllm"  # 'vllm', 'transformers', 'api'
    llm_quantization: Optional[str] = "awq"  # 'awq', 'gptq', None
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.1

    # Batch processing
    batch_size: int = 16
    max_workers: int = 4

    # Content ratios
    should_respond_ratio: float = 0.7  # 70% QA samples

    # Document filtering
    min_doc_length: int = 100
    max_doc_length: int = 4000
    min_entities: int = 1
    max_entities: int = 20

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
            llm_quantization="awq",
            checkpoint_every=1000,
            enable_safety_filter=True,
            safety_strictness="medium",
        )

    @classmethod
    def for_large_scale(cls) -> "CurationConfig":
        """Config for large-scale curation (100K+ samples)."""
        return cls(
            target_samples=100000,
            sources=["allenai/c4", "wikipedia", "togethercomputer/RedPajama-Data-1T-Sample"],
            source_ratios=[0.4, 0.35, 0.25],
            batch_size=32,
            llm_backend="vllm",
            llm_quantization="awq",
            checkpoint_every=2000,
            max_workers=8,
            enable_safety_filter=True,
            safety_strictness="high",
        )
