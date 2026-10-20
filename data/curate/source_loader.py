"""Source data loading from HuggingFace datasets."""

from typing import Iterator, Dict, Any, List, Optional, Union
from dataclasses import dataclass
from itertools import islice
import random


@dataclass
class DocumentSource:
    """A document from a source dataset."""

    text: str
    source: str
    doc_id: str
    metadata: Dict[str, Any]


def parse_text_field_args(values: Optional[List[str]]) -> Union[str, Dict[str, str], None]:
    """Parse ``--text-field`` values.

    A bare ``field`` applies to every source. A ``source=field`` entry
    applies to one source and wins over a bare value. The bare value is
    stored under the ``"*"`` key.
    """
    if not values:
        return None
    mapping: Dict[str, str] = {}
    for value in values:
        if "=" in value:
            source, field_name = value.rsplit("=", 1)
            mapping[source] = field_name
        else:
            mapping["*"] = value
    return mapping


class SourceLoader:
    """Load and stream documents from multiple HuggingFace sources.

    Supports:
    - Wikipedia (wikimedia/wikipedia)
    - C4 (allenai/c4)
    - FineWeb (HuggingFaceFW/fineweb, sample-10BT)
    - Other HuggingFace text datasets

    Uses streaming mode to handle large datasets without downloading fully.
    The loader counts the raw rows it consumes per source. A resumed run
    passes these counts as ``start_offsets`` to skip rows it already saw.
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
        "HuggingFaceFW/fineweb": {
            "hf_path": "HuggingFaceFW/fineweb",
            "subset": "sample-10BT",
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
        text_field: Union[str, Dict[str, str], None] = None,
        start_offsets: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize source loader.

        Args:
            sources: List of source identifiers
            ratios: Sampling ratios for each source
            seed: Random seed for shuffling
            buffer_size: Buffer size for shuffling
            text_field: Name of the text column. A string applies to all
                sources. A dict maps source to column; the ``"*"`` key is
                the default for sources not in the dict.
            start_offsets: Raw rows to skip per source (resume)
        """
        self.sources = sources
        self.ratios = ratios
        self.seed = seed
        self.buffer_size = buffer_size
        if isinstance(text_field, str):
            text_field = {"*": text_field}
        self.text_field_overrides: Dict[str, str] = dict(text_field or {})
        self._iterators: Dict[str, Iterator] = {}
        self._doc_counts: Dict[str, int] = {s: 0 for s in sources}
        # Raw rows consumed per source in the current pass over the stream.
        # Start at the resume offsets, so a checkpoint saved before a source
        # is sampled again keeps its position.
        self._start_offsets: Dict[str, int] = dict(start_offsets or {})
        self._offsets: Dict[str, int] = {
            s: self._start_offsets.get(s, 0) for s in sources
        }
        # Usable documents yielded per source since the last (re)start
        self._pass_yielded: Dict[str, int] = {s: 0 for s in sources}
        # True when the current pass started at a resume offset
        self._pass_resumed: Dict[str, bool] = {s: False for s in sources}

        random.seed(seed)

    def _get_dataset_config(self, source: str) -> Dict[str, Any]:
        """Get configuration for a dataset source."""
        if source in self.DATASET_CONFIGS:
            config = dict(self.DATASET_CONFIGS[source])
        else:
            # Assume it's a HuggingFace path
            config = {
                "hf_path": source,
                "subset": None,
                "text_field": "text",
                "split": "train",
            }
        if source in self.text_field_overrides:
            config["text_field"] = self.text_field_overrides[source]
        elif "*" in self.text_field_overrides:
            config["text_field"] = self.text_field_overrides["*"]
        return config

    def _load_stream(self, source: str, config: Dict[str, Any]):
        """Load a shuffled streaming dataset for a source."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        args = [config["hf_path"]]
        if config["subset"]:
            args.append(config["subset"])
        try:
            dataset = load_dataset(*args, split=config["split"], streaming=True)
        except Exception as e:
            print(f"Error loading {source}: {e}")
            raise

        # Shuffle with buffer
        return dataset.shuffle(seed=self.seed, buffer_size=self.buffer_size)

    def _create_iterator(self, source: str, skip: int = 0) -> Iterator[DocumentSource]:
        """Create a streaming iterator for a source.

        Args:
            source: Source identifier
            skip: Number of raw rows to skip at the start of the stream
        """
        config = self._get_dataset_config(source)
        field_name = config["text_field"]
        dataset = self._load_stream(source, config)

        # The shuffled stream is deterministic for a fixed seed, so the
        # first `skip` rows are the rows a previous run already consumed.
        rows = iter(dataset)
        if skip:
            print(f"Skipping {skip} rows already consumed from {source}")
            rows = islice(rows, skip, None)

        idx = skip
        self._offsets[source] = skip
        for item in rows:
            if idx == skip and field_name not in item:
                raise ValueError(
                    f"Source {source!r} has no text field {field_name!r}. "
                    f"Available fields: {sorted(item.keys())}. "
                    f"Set --text-field {source}=<field>."
                )
            idx += 1
            self._offsets[source] = idx
            text = item.get(field_name, "")
            if isinstance(text, str) and text.strip():
                yield DocumentSource(
                    text=text,
                    source=source,
                    doc_id=f"{source}_{idx - 1}",
                    metadata={
                        k: v
                        for k, v in item.items()
                        if k != field_name and isinstance(v, (str, int, float))
                    },
                )

    def _get_iterator(self, source: str) -> Iterator[DocumentSource]:
        """Get or create iterator for a source."""
        if source not in self._iterators:
            skip = self._start_offsets.pop(source, 0)
            self._iterators[source] = self._create_iterator(source, skip=skip)
            self._pass_yielded[source] = 0
            self._pass_resumed[source] = skip > 0
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
                self._pass_yielded[source] += 1
                yield doc
            except StopIteration:
                # A restart cannot help a source with no usable documents.
                # A resumed pass can end with zero documents legitimately.
                if self._pass_yielded[source] == 0 and not self._pass_resumed[source]:
                    raise RuntimeError(
                        f"Source {source!r} yielded zero usable documents "
                        f"(text field {self._get_dataset_config(source)['text_field']!r}). "
                        f"Check --text-field and the dataset split."
                    )
                # Source exhausted, recreate iterator
                print(f"Source {source} exhausted, restarting...")
                self._iterators[source] = self._create_iterator(source)
                self._pass_yielded[source] = 0
                self._pass_resumed[source] = False
                continue

    def get_offsets(self) -> Dict[str, int]:
        """Raw rows consumed per source in the current pass."""
        return dict(self._offsets)

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
