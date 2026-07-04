"""Output writing for curated datasets."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

from .llm_annotator import AnnotationResult


@dataclass
class CheckpointState:
    """State for checkpointing."""

    samples_processed: int = 0
    samples_written: int = 0
    samples_failed: int = 0
    concepts_vocab: Set[str] = field(default_factory=set)
    relations_vocab: Set[str] = field(default_factory=set)
    entity_types_vocab: Set[str] = field(default_factory=set)
    source_counts: Dict[str, int] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "samples_processed": self.samples_processed,
            "samples_written": self.samples_written,
            "samples_failed": self.samples_failed,
            "concepts_vocab": list(self.concepts_vocab),
            "relations_vocab": list(self.relations_vocab),
            "entity_types_vocab": list(self.entity_types_vocab),
            "source_counts": self.source_counts,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointState":
        """Create from dictionary."""
        return cls(
            samples_processed=data.get("samples_processed", 0),
            samples_written=data.get("samples_written", 0),
            samples_failed=data.get("samples_failed", 0),
            concepts_vocab=set(data.get("concepts_vocab", [])),
            relations_vocab=set(data.get("relations_vocab", [])),
            entity_types_vocab=set(data.get("entity_types_vocab", [])),
            source_counts=data.get("source_counts", {}),
            timestamp=data.get("timestamp", ""),
        )


class OutputWriter:
    """Write curated samples to output files.

    Outputs:
    - JSONL file with curated samples
    - Vocabulary files for concepts, relations, entity types
    - Checkpoint files for resumability
    """

    def __init__(
        self,
        output_dir: str,
        output_name: str = "curated",
        checkpoint_every: int = 1000,
        compress: bool = False,
        taxonomy=None,
    ):
        """
        Initialize output writer.

        Args:
            output_dir: Directory for output files
            output_name: Base name for output files
            checkpoint_every: Save checkpoint every N samples
            compress: Whether to compress output files
            taxonomy: Optional Taxonomy; when given, the saved vocabulary is
                the full taxonomy inventory (stable across runs) rather than
                just the labels observed in this run
        """
        self.output_dir = Path(output_dir)
        self.output_name = output_name
        self.checkpoint_every = checkpoint_every
        self.compress = compress
        self.taxonomy = taxonomy

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Output paths
        self.output_path = self.output_dir / f"{output_name}.jsonl"
        self.checkpoint_path = self.output_dir / f"{output_name}_checkpoint.json"
        self.vocab_path = self.output_dir / f"{output_name}_vocab.json"

        # State
        self.state = CheckpointState()
        self._output_file = None
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_size = 100

    def _open_output(self):
        """Open output file for appending."""
        if self._output_file is None:
            mode = "a" if self.output_path.exists() else "w"
            if self.compress:
                import gzip
                self._output_file = gzip.open(str(self.output_path) + ".gz", mode + "t")
            else:
                self._output_file = open(self.output_path, mode)

    def _flush_buffer(self):
        """Flush buffer to file."""
        if not self._buffer:
            return

        self._open_output()
        for sample in self._buffer:
            self._output_file.write(json.dumps(sample) + "\n")
        self._output_file.flush()
        self._buffer = []

    def write(self, result: AnnotationResult, source: str = "unknown"):
        """
        Write a single annotation result.

        Args:
            result: AnnotationResult to write
            source: Source identifier
        """
        self.state.samples_processed += 1

        if not result.success:
            self.state.samples_failed += 1
            return

        # Convert to output format
        sample = result.to_dict()

        # Update vocabulary
        for concept_list in result.concepts:
            for concept in concept_list:
                self.state.concepts_vocab.add(concept)

        for relation in result.relations:
            if len(relation) >= 3:
                self.state.relations_vocab.add(relation[2])

        for entity_type in result.entity_types:
            self.state.entity_types_vocab.add(entity_type)

        # Track source
        self.state.source_counts[source] = self.state.source_counts.get(source, 0) + 1

        # Add to buffer
        self._buffer.append(sample)
        self.state.samples_written += 1

        # Flush buffer if full
        if len(self._buffer) >= self._buffer_size:
            self._flush_buffer()

        # Save checkpoint periodically
        if self.state.samples_written % self.checkpoint_every == 0:
            self.save_checkpoint()

    def write_batch(
        self, results: List[AnnotationResult], sources: Optional[List[str]] = None
    ):
        """
        Write a batch of results.

        Args:
            results: List of AnnotationResults
            sources: Optional list of source identifiers
        """
        if sources is None:
            sources = ["unknown"] * len(results)

        for result, source in zip(results, sources):
            self.write(result, source)

    def save_checkpoint(self):
        """Save checkpoint state."""
        self._flush_buffer()
        self.state.timestamp = datetime.now().isoformat()

        with open(self.checkpoint_path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

        # Also save vocabulary
        self.save_vocabulary()

        print(f"Checkpoint saved: {self.state.samples_written} samples")

    def load_checkpoint(self) -> bool:
        """
        Load checkpoint state if available.

        Returns:
            True if checkpoint was loaded
        """
        if not self.checkpoint_path.exists():
            return False

        try:
            with open(self.checkpoint_path, "r") as f:
                data = json.load(f)
            self.state = CheckpointState.from_dict(data)
            print(f"Checkpoint loaded: {self.state.samples_written} samples")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def save_vocabulary(self):
        """Save vocabulary mappings.

        With a taxonomy, the full (stable) taxonomy vocab is written so
        index assignments do not depend on which labels happened to occur;
        observed-label statistics are still recorded. Without a taxonomy,
        falls back to the labels observed in this run (1-indexed, 0 reserved
        for unknown/padding).
        """
        if self.taxonomy is not None:
            vocab = self.taxonomy.vocab()
            vocab["statistics"] = {
                "num_concepts": len(vocab["concepts"]),
                "num_relations": len(vocab["relations"]),
                "num_samples": self.state.samples_written,
                "observed_concepts": len(self.state.concepts_vocab),
                "observed_relations": len(self.state.relations_vocab),
                "observed_entity_types": len(self.state.entity_types_vocab),
            }
        else:
            concepts_sorted = sorted(self.state.concepts_vocab)
            relations_sorted = sorted(self.state.relations_vocab)
            entity_types_sorted = sorted(self.state.entity_types_vocab)

            entity_types = {t: i + 1 for i, t in enumerate(entity_types_sorted)}
            if not entity_types:
                # Legacy default for LLM-annotated data without entity types
                entity_types = {
                    "person": 1,
                    "organization": 2,
                    "location": 3,
                    "date": 4,
                    "time": 5,
                    "quantity": 6,
                    "object": 7,
                    "event": 8,
                    "concept": 9,
                }

            vocab = {
                "concepts": {c: i + 1 for i, c in enumerate(concepts_sorted)},
                "relations": {r: i + 1 for i, r in enumerate(relations_sorted)},
                "entity_types": entity_types,
                "statistics": {
                    "num_concepts": len(concepts_sorted),
                    "num_relations": len(relations_sorted),
                    "num_samples": self.state.samples_written,
                },
            }

        with open(self.vocab_path, "w") as f:
            json.dump(vocab, f, indent=2)

    def finalize(self):
        """Finalize output and close files."""
        self._flush_buffer()

        if self._output_file:
            self._output_file.close()
            self._output_file = None

        # Final checkpoint and vocabulary
        self.save_checkpoint()
        self.save_vocabulary()

        print(f"\nOutput finalized:")
        print(f"  Samples written: {self.state.samples_written}")
        print(f"  Samples failed: {self.state.samples_failed}")
        print(f"  Concepts: {len(self.state.concepts_vocab)}")
        print(f"  Relations: {len(self.state.relations_vocab)}")
        print(f"  Output: {self.output_path}")
        print(f"  Vocabulary: {self.vocab_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get output statistics."""
        return {
            "samples_processed": self.state.samples_processed,
            "samples_written": self.state.samples_written,
            "samples_failed": self.state.samples_failed,
            "success_rate": self.state.samples_written / self.state.samples_processed
            if self.state.samples_processed > 0
            else 0,
            "num_concepts": len(self.state.concepts_vocab),
            "num_relations": len(self.state.relations_vocab),
            "source_distribution": self.state.source_counts,
        }


class SplitWriter:
    """Write curated samples with train/val/test splits."""

    def __init__(
        self,
        output_dir: str,
        output_name: str = "curated",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize split writer.

        Args:
            output_dir: Directory for output files
            output_name: Base name for output files
            train_ratio: Ratio of training samples
            val_ratio: Ratio of validation samples
            test_ratio: Ratio of test samples
            seed: Random seed for splitting
        """
        import random

        self.output_dir = Path(output_dir)
        self.output_name = output_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        random.seed(seed)

        # Create writers for each split
        self.writers = {
            "train": OutputWriter(output_dir, f"{output_name}_train"),
            "val": OutputWriter(output_dir, f"{output_name}_val"),
            "test": OutputWriter(output_dir, f"{output_name}_test"),
        }

        self._sample_count = 0

    def _get_split(self) -> str:
        """Determine which split to assign sample to."""
        import random

        r = random.random()
        if r < self.train_ratio:
            return "train"
        elif r < self.train_ratio + self.val_ratio:
            return "val"
        else:
            return "test"

    def write(self, result: AnnotationResult, source: str = "unknown"):
        """Write a sample to appropriate split."""
        split = self._get_split()
        self.writers[split].write(result, source)
        self._sample_count += 1

    def write_batch(
        self, results: List[AnnotationResult], sources: Optional[List[str]] = None
    ):
        """Write batch of samples to splits."""
        if sources is None:
            sources = ["unknown"] * len(results)

        for result, source in zip(results, sources):
            self.write(result, source)

    def finalize(self):
        """Finalize all splits."""
        for split, writer in self.writers.items():
            print(f"\n{split.upper()} split:")
            writer.finalize()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all splits."""
        return {split: writer.get_statistics() for split, writer in self.writers.items()}
