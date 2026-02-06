"""Source data loading from HuggingFace datasets."""

from typing import Iterator, Dict, Any, List, Optional
from dataclasses import dataclass
import random


@dataclass
class DocumentSource:
    """A document from a source dataset."""

    text: str
    source: str
    doc_id: str
    metadata: Dict[str, Any]


class SourceLoader:
    """Load and stream documents from multiple HuggingFace sources.

    Supports:
    - Wikipedia (wikimedia/wikipedia)
    - C4 (allenai/c4)
    - RedPajama (togethercomputer/RedPajama-Data-1T-Sample)
    - Other HuggingFace text datasets

    Uses streaming mode to handle large datasets without downloading fully.
    """

    # Known dataset configurations
    DATASET_CONFIGS = {
        "wikipedia": {
            "hf_path": "wikimedia/wikipedia",
            "subset": "20231101.en",
            "text_field": "text",
            "split": "train",
        },
        "allenai/c4": {
            "hf_path": "allenai/c4",
            "subset": "en",
            "text_field": "text",
            "split": "train",
        },
        "togethercomputer/RedPajama-Data-1T-Sample": {
            "hf_path": "togethercomputer/RedPajama-Data-1T-Sample",
            "subset": None,
            "text_field": "text",
            "split": "train",
        },
    }

    def __init__(
        self,
        sources: List[str],
        ratios: List[float],
        seed: int = 42,
        buffer_size: int = 10000,
    ):
        """
        Initialize source loader.

        Args:
            sources: List of source identifiers
            ratios: Sampling ratios for each source
            seed: Random seed for shuffling
            buffer_size: Buffer size for shuffling
        """
        self.sources = sources
        self.ratios = ratios
        self.seed = seed
        self.buffer_size = buffer_size
        self._iterators: Dict[str, Iterator] = {}
        self._doc_counts: Dict[str, int] = {s: 0 for s in sources}

        random.seed(seed)

    def _get_dataset_config(self, source: str) -> Dict[str, Any]:
        """Get configuration for a dataset source."""
        if source in self.DATASET_CONFIGS:
            return self.DATASET_CONFIGS[source]

        # Assume it's a HuggingFace path
        return {
            "hf_path": source,
            "subset": None,
            "text_field": "text",
            "split": "train",
        }

    def _create_iterator(self, source: str) -> Iterator[DocumentSource]:
        """Create a streaming iterator for a source."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        config = self._get_dataset_config(source)

        # Load dataset in streaming mode
        try:
            if config["subset"]:
                dataset = load_dataset(
                    config["hf_path"],
                    config["subset"],
                    split=config["split"],
                    streaming=True,
                    trust_remote_code=True,
                )
            else:
                dataset = load_dataset(
                    config["hf_path"],
                    split=config["split"],
                    streaming=True,
                    trust_remote_code=True,
                )
        except Exception as e:
            print(f"Error loading {source}: {e}")
            raise

        # Shuffle with buffer
        dataset = dataset.shuffle(seed=self.seed, buffer_size=self.buffer_size)

        # Yield documents
        for idx, item in enumerate(dataset):
            text = item.get(config["text_field"], "")

            # Handle Wikipedia's nested structure
            if source == "wikipedia" and not text:
                text = item.get("text", "")

            if text:
                yield DocumentSource(
                    text=text,
                    source=source,
                    doc_id=f"{source}_{idx}",
                    metadata={
                        k: v
                        for k, v in item.items()
                        if k != config["text_field"] and isinstance(v, (str, int, float))
                    },
                )

    def _get_iterator(self, source: str) -> Iterator[DocumentSource]:
        """Get or create iterator for a source."""
        if source not in self._iterators:
            self._iterators[source] = self._create_iterator(source)
        return self._iterators[source]

    def _sample_source(self) -> str:
        """Sample a source according to ratios."""
        r = random.random()
        cumulative = 0.0
        for source, ratio in zip(self.sources, self.ratios):
            cumulative += ratio
            if r < cumulative:
                return source
        return self.sources[-1]

    def __iter__(self) -> Iterator[DocumentSource]:
        """Iterate over documents from all sources according to ratios."""
        while True:
            source = self._sample_source()
            iterator = self._get_iterator(source)

            try:
                doc = next(iterator)
                self._doc_counts[source] += 1
                yield doc
            except StopIteration:
                # Source exhausted, recreate iterator
                print(f"Source {source} exhausted, restarting...")
                self._iterators[source] = self._create_iterator(source)
                continue

    def get_documents(self, n: int) -> List[DocumentSource]:
        """Get n documents from the sources.

        Args:
            n: Number of documents to retrieve

        Returns:
            List of DocumentSource objects
        """
        docs = []
        iterator = iter(self)
        for _ in range(n):
            try:
                docs.append(next(iterator))
            except StopIteration:
                break
        return docs

    def get_statistics(self) -> Dict[str, Any]:
        """Get loading statistics."""
        total = sum(self._doc_counts.values())
        return {
            "total_documents": total,
            "by_source": dict(self._doc_counts),
            "source_ratios_actual": {
                s: c / total if total > 0 else 0 for s, c in self._doc_counts.items()
            },
            "source_ratios_target": dict(zip(self.sources, self.ratios)),
        }


class SimpleTextLoader:
    """Simple loader for local text files or directories.

    Useful for testing or processing local data.
    """

    def __init__(self, paths: List[str], shuffle: bool = True, seed: int = 42):
        """
        Initialize from local paths.

        Args:
            paths: List of file or directory paths
            shuffle: Whether to shuffle documents
            seed: Random seed
        """
        self.paths = paths
        self.shuffle = shuffle
        self.seed = seed
        self._documents: List[DocumentSource] = []
        self._loaded = False

    def _load_documents(self):
        """Load documents from paths."""
        from pathlib import Path

        for path_str in self.paths:
            path = Path(path_str)

            if path.is_file():
                self._load_file(path)
            elif path.is_dir():
                for file_path in path.glob("**/*.txt"):
                    self._load_file(file_path)
                for file_path in path.glob("**/*.json"):
                    self._load_json_file(file_path)
                for file_path in path.glob("**/*.jsonl"):
                    self._load_jsonl_file(file_path)

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self._documents)

        self._loaded = True

    def _load_file(self, path):
        """Load a text file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            if text.strip():
                self._documents.append(
                    DocumentSource(
                        text=text,
                        source="local",
                        doc_id=str(path),
                        metadata={"path": str(path)},
                    )
                )
        except Exception as e:
            print(f"Error loading {path}: {e}")

    def _load_json_file(self, path):
        """Load a JSON file with text field."""
        import json

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "text" in data:
                self._documents.append(
                    DocumentSource(
                        text=data["text"],
                        source="local",
                        doc_id=str(path),
                        metadata=data,
                    )
                )
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict) and "text" in item:
                        self._documents.append(
                            DocumentSource(
                                text=item["text"],
                                source="local",
                                doc_id=f"{path}_{i}",
                                metadata=item,
                            )
                        )
        except Exception as e:
            print(f"Error loading {path}: {e}")

    def _load_jsonl_file(self, path):
        """Load a JSONL file."""
        import json

        try:
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if line.strip():
                        item = json.loads(line)
                        if "text" in item:
                            self._documents.append(
                                DocumentSource(
                                    text=item["text"],
                                    source="local",
                                    doc_id=f"{path}_{i}",
                                    metadata=item,
                                )
                            )
        except Exception as e:
            print(f"Error loading {path}: {e}")

    def __iter__(self) -> Iterator[DocumentSource]:
        """Iterate over documents."""
        if not self._loaded:
            self._load_documents()
        return iter(self._documents)

    def get_documents(self, n: int) -> List[DocumentSource]:
        """Get n documents."""
        if not self._loaded:
            self._load_documents()
        return self._documents[:n]
