"""Output writing for curated datasets."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

from .dedup import is_validation, text_hash
from .llm_annotator import AnnotationResult


@dataclass
class CheckpointState:
    """State for checkpointing."""

    samples_processed: int = 0
    samples_written: int = 0
    samples_failed: int = 0
    # Lines in the train and validation files at checkpoint time
    train_written: int = 0
    val_written: int = 0
    concepts_vocab: Set[str] = field(default_factory=set)
    relations_vocab: Set[str] = field(default_factory=set)
    entity_types_vocab: Set[str] = field(default_factory=set)
    source_counts: Dict[str, int] = field(default_factory=dict)
    # Raw source rows consumed per source (resume skips these)
    source_offsets: Dict[str, int] = field(default_factory=dict)
    # Hashes of normalized text already seen (deduplication)
    seen_hashes: Set[str] = field(default_factory=set)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "samples_processed": self.samples_processed,
            "samples_written": self.samples_written,
            "samples_failed": self.samples_failed,
            "train_written": self.train_written,
            "val_written": self.val_written,
            "concepts_vocab": list(self.concepts_vocab),
            "relations_vocab": list(self.relations_vocab),
            "entity_types_vocab": list(self.entity_types_vocab),
            "source_counts": self.source_counts,
            "source_offsets": self.source_offsets,
            "seen_hashes": sorted(self.seen_hashes),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointState":
        """Create from dictionary."""
        samples_written = data.get("samples_written", 0)
        return cls(
            samples_processed=data.get("samples_processed", 0),
            samples_written=samples_written,
            samples_failed=data.get("samples_failed", 0),
            # Old checkpoints have no split: all rows are train rows
            train_written=data.get("train_written", samples_written),
            val_written=data.get("val_written", 0),
            concepts_vocab=set(data.get("concepts_vocab", [])),
            relations_vocab=set(data.get("relations_vocab", [])),
            entity_types_vocab=set(data.get("entity_types_vocab", [])),
            source_counts=data.get("source_counts", {}),
            source_offsets=data.get("source_offsets", {}),
            seen_hashes=set(data.get("seen_hashes", [])),
            timestamp=data.get("timestamp", ""),
        )


def _truncate_lines(path: Path, n_lines: int) -> int:
    """Keep the first n_lines lines of a file. Return lines dropped."""
    if not path.exists():
        path.touch()
        return 0
    with open(path, "rb") as f:
        lines = f.readlines()
    kept = lines[:n_lines]
    if len(kept) < n_lines:
        raise RuntimeError(
            f"{path} has {len(lines)} lines but the checkpoint expects "
            f"{n_lines}. The output does not match the checkpoint."
        )
    if len(kept) != len(lines):
        with open(path, "wb") as f:
            f.writelines(kept)
    return len(lines) - len(kept)


class OutputWriter:
    """Write curated samples to output files.

    Outputs:
    - ``<name>.jsonl``: train samples
    - ``<name>_val.jsonl``: validation samples (hash-based assignment)
    - ``<name>_vocab.json``: vocabulary for concepts, relations, entity types
    - ``<name>_checkpoint.json``: checkpoint for resumability

    Call :meth:`start` before the first write. A fresh start refuses to
    replace existing outputs unless ``overwrite`` is set. A resumed start
    truncates the output files to the checkpointed line counts, so rows
    written after the last checkpoint are not duplicated.
    """

    def __init__(
        self,
        output_dir: str,
        output_name: str = "curated",
        checkpoint_every: int = 1000,
        compress: bool = False,
        taxonomy=None,
        val_ratio: float = 0.0,
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
            val_ratio: Fraction of samples written to the validation file
        """
        self.output_dir = Path(output_dir)
        self.output_name = output_name
        self.checkpoint_every = checkpoint_every
        self.compress = compress
        self.taxonomy = taxonomy
        self.val_ratio = val_ratio

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Output paths
        suffix = ".jsonl.gz" if compress else ".jsonl"
        self.output_path = self.output_dir / f"{output_name}{suffix}"
        self.val_path = self.output_dir / f"{output_name}_val{suffix}"
        self.checkpoint_path = self.output_dir / f"{output_name}_checkpoint.json"
        self.vocab_path = self.output_dir / f"{output_name}_vocab.json"

        # State
        self.state = CheckpointState()
        self._files: Dict[str, Any] = {"train": None, "val": None}
        self._buffers: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
        self._buffer_size = 100
        self._started = False
        self._append = False
        self._last_checkpoint_at = 0

    # ------------------------------------------------------------------
    # Start / resume
    # ------------------------------------------------------------------

    def existing_outputs(self) -> List[Path]:
        """Output files of this name that already exist and are not empty."""
        paths = [self.output_path, self.val_path, self.checkpoint_path, self.vocab_path]
        return [p for p in paths if p.exists() and p.stat().st_size > 0]

    def start(self, overwrite: bool = False):
        """Start a fresh run.

        Raises:
            FileExistsError: Outputs of this name exist and overwrite is False
        """
        existing = self.existing_outputs()
        if existing and not overwrite:
            names = ", ".join(p.name for p in existing)
            raise FileExistsError(
                f"Output files already exist in {self.output_dir}: {names}. "
                f"Use --overwrite to replace them or --resume-from to continue."
            )
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        self.state = CheckpointState()
        self._append = False
        self._last_checkpoint_at = 0
        self._started = True
        # Truncate both files now so no stale rows survive an overwrite
        self._open_output("train")
        self._open_output("val")
        # The vocabulary comes from the taxonomy, so it is usable at once
        self.save_vocabulary()

    def resume(self, checkpoint_path: Optional[str] = None) -> bool:
        """Resume from a checkpoint.

        Loads the checkpoint and truncates the train and validation files to
        the checkpointed line counts.

        Returns:
            True if a checkpoint was loaded
        """
        if self.compress:
            raise ValueError("Resume does not support compressed output")
        if checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path)
        if not self.load_checkpoint():
            return False

        dropped = _truncate_lines(self.output_path, self.state.train_written)
        dropped += _truncate_lines(self.val_path, self.state.val_written)
        if dropped:
            print(f"Dropped {dropped} rows written after the last checkpoint")

        if not self.state.seen_hashes:
            # Old checkpoints have no hash set: rebuild it from the output
            self.state.seen_hashes = self._hashes_from_output()

        self._append = True
        self._last_checkpoint_at = self.state.samples_written
        self._started = True
        return True

    def _hashes_from_output(self) -> Set[str]:
        """Hash the text of every row in the train and validation files."""
        hashes: Set[str] = set()
        for path in (self.output_path, self.val_path):
            if not path.exists():
                continue
            with self._open_path(path, "r") as f:
                for line in f:
                    if line.strip():
                        hashes.add(text_hash(json.loads(line)["text"]))
        return hashes

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def is_duplicate(self, hash_hex: str) -> bool:
        """True if this text hash was already seen in this run."""
        return hash_hex in self.state.seen_hashes

    def mark_seen(self, hash_hex: str):
        """Record a text hash as seen."""
        self.state.seen_hashes.add(hash_hex)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def _open_path(self, path: Path, mode: str):
        if self.compress:
            import gzip
            return gzip.open(str(path), mode + "t", encoding="utf-8")
        return open(path, mode, encoding="utf-8")

    def _open_output(self, split: str):
        """Open the output file of a split."""
        if self._files[split] is None:
            path = self.output_path if split == "train" else self.val_path
            self._files[split] = self._open_path(path, "a" if self._append else "w")

    def _flush_buffer(self):
        """Flush buffers to files."""
        for split, buffer in self._buffers.items():
            if not buffer:
                continue
            self._open_output(split)
            for sample in buffer:
                self._files[split].write(json.dumps(sample) + "\n")
            self._files[split].flush()
            self._buffers[split] = []

    def write(
        self,
        result: AnnotationResult,
        source: str = "unknown",
        hash_hex: Optional[str] = None,
    ):
        """
        Write a single annotation result.

        Args:
            result: AnnotationResult to write
            source: Source identifier
            hash_hex: Text hash for split assignment. Pass the hash of the
                source document so the split does not change when a QA
                question is appended. Defaults to the hash of result.text.
        """
        if not self._started:
            # Direct use without start(): behave as a fresh run
            self.start(overwrite=True)

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

        # Pick the split from the text hash
        if hash_hex is None:
            hash_hex = text_hash(result.text)
        split = "val" if is_validation(hash_hex, self.val_ratio) else "train"

        # Add to buffer
        self._buffers[split].append(sample)
        self.state.samples_written += 1
        if split == "train":
            self.state.train_written += 1
        else:
            self.state.val_written += 1

        # Flush buffer if full
        if len(self._buffers[split]) >= self._buffer_size:
            self._flush_buffer()

    def write_batch(
        self,
        results: List[AnnotationResult],
        sources: Optional[List[str]] = None,
        hashes: Optional[List[Optional[str]]] = None,
    ):
        """
        Write a batch of results.

        Args:
            results: List of AnnotationResults
            sources: Optional list of source identifiers
            hashes: Optional list of text hashes for split assignment
        """
        if sources is None:
            sources = ["unknown"] * len(results)
        if hashes is None:
            hashes = [None] * len(results)

        for result, source, hash_hex in zip(results, sources, hashes):
            self.write(result, source, hash_hex)

    def maybe_checkpoint(self, source_offsets: Optional[Dict[str, int]] = None) -> bool:
        """Save a checkpoint if checkpoint_every samples passed since the last one.

        Call this between batches, when every consumed document has been
        written. Then the source offsets match the written rows.
        """
        if source_offsets is not None:
            self.state.source_offsets = dict(source_offsets)
        if self.state.samples_written - self._last_checkpoint_at >= self.checkpoint_every:
            self.save_checkpoint()
            return True
        return False

    def save_checkpoint(self):
        """Save checkpoint state."""
        self._flush_buffer()
        self.state.timestamp = datetime.now().isoformat()

        # Write to a temp file first so a crash cannot leave a half checkpoint
        tmp_path = self.checkpoint_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.state.to_dict(), f)
        tmp_path.replace(self.checkpoint_path)
        self._last_checkpoint_at = self.state.samples_written

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
        if not self._started:
            self.start(overwrite=True)

        # Final checkpoint (flushes buffers) and vocabulary
        self.save_checkpoint()

        for split, f in self._files.items():
            if f is not None:
                f.close()
                self._files[split] = None

        print(f"\nOutput finalized:")
        print(
            f"  Samples written: {self.state.samples_written} "
            f"(train {self.state.train_written}, val {self.state.val_written})"
        )
        print(f"  Samples failed: {self.state.samples_failed}")
        print(f"  Concepts: {len(self.state.concepts_vocab)}")
        print(f"  Relations: {len(self.state.relations_vocab)}")
        print(f"  Output: {self.output_path}")
        print(f"  Validation: {self.val_path}")
        print(f"  Vocabulary: {self.vocab_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get output statistics."""
        return {
            "samples_processed": self.state.samples_processed,
            "samples_written": self.state.samples_written,
            "train_written": self.state.train_written,
            "val_written": self.state.val_written,
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
