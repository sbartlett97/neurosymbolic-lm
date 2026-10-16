"""Tests for curation fixes: overwrite guard, truncation, source errors,
dedup and validation split, resume, and the output name."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.curate.config import CurationConfig
from data.curate.dedup import is_validation, normalize_text, text_hash
from data.curate.document_filter import DocumentFilter, truncate_text
from data.curate.gliner_annotator import GLiNER2Annotator, select_device
from data.curate.llm_annotator import AnnotationResult, apply_chat_template
from data.curate.output_writer import OutputWriter
from data.curate.source_loader import (
    DocumentSource,
    SourceLoader,
    parse_text_field_args,
)
from data.curate.taxonomy import get_default_taxonomy

REPO = Path(__file__).parent.parent


def load_curate_dataset():
    """Import data/curate_dataset.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "curate_dataset_under_test", REPO / "data" / "curate_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_doc_text(i: int) -> str:
    """A unique document that passes DocumentFilter."""
    return (
        f"Albert Einstein visited Brazil in the spring of year number {i}. "
        f"He gave a long lecture about physics to a large audience there. "
        f"The local newspaper wrote a detailed report about lecture {i} later."
    )


class StubExtractor:
    """GLiNER2-shaped stub. Records the kwargs of every call."""

    LABELS = {"Einstein": ("scientist", 0.9), "Brazil": ("country", 0.9)}

    def __init__(self):
        self.calls = []

    def extract_entities(self, text, entity_types, **kw):
        self.calls.append(kw)
        found = {}
        for word, (label, conf) in self.LABELS.items():
            if label in entity_types and word in text:
                start = text.index(word)
                found.setdefault(label, []).append(
                    {"text": word, "confidence": conf,
                     "start": start, "end": start + len(word)}
                )
        return {"entities": found}

    def extract_relations(self, text, relation_types, **kw):
        self.calls.append(kw)
        return {"relation_extraction": {}}


def result(text: str) -> AnnotationResult:
    return AnnotationResult(
        text=text, entities=["x"], entity_spans=[[0, 1]], concepts=[["object"]],
        entity_types=["product"],
    )


def read_rows(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ----------------------------------------------------------------------
# C1: overwrite guard
# ----------------------------------------------------------------------


def test_fresh_start_refuses_existing_output(tmp_path):
    writer = OutputWriter(str(tmp_path), "run")
    writer.start()
    writer.write(result("first run row"))
    writer.finalize()

    again = OutputWriter(str(tmp_path), "run")
    with pytest.raises(FileExistsError, match="--overwrite"):
        again.start()

    # Overwrite truncates the old rows
    again.start(overwrite=True)
    again.write(result("second run row"))
    again.finalize()
    rows = read_rows(tmp_path / "run.jsonl")
    assert [r["text"] for r in rows] == ["second run row"]


def test_other_output_name_is_not_blocked(tmp_path):
    first = OutputWriter(str(tmp_path), "a")
    first.start()
    first.finalize()
    OutputWriter(str(tmp_path), "b").start()


# ----------------------------------------------------------------------
# C3: max document length
# ----------------------------------------------------------------------


def test_truncate_text_cuts_at_sentence_boundary():
    text = "First sentence here. Second sentence is here. Third one goes on and on."
    cut = truncate_text(text, 50)
    assert cut == "First sentence here. Second sentence is here."
    assert truncate_text(text, 1000) == text


def test_truncate_text_falls_back_to_whitespace():
    text = "word " * 50
    cut = truncate_text(text, 23)
    assert len(cut) <= 23
    assert not cut.endswith(" ")
    assert cut.split() == ["word"] * len(cut.split())


def test_filter_truncates_doc_text():
    long_text = " ".join(make_doc_text(i) for i in range(40))
    assert len(long_text) > 2000
    doc = DocumentSource(text=long_text, source="t", doc_id="0", metadata={})
    result_ = DocumentFilter(max_length=1000).filter(doc)
    assert result_.passed
    assert len(doc.text) <= 1000
    assert doc.text.endswith(".")
    assert long_text.startswith(doc.text)


def test_gliner_max_len_is_passed_and_falls_back():
    stub = StubExtractor()
    ann = GLiNER2Annotator(extractor=stub, taxonomy=get_default_taxonomy(), max_len=256)
    ann.annotate_text(make_doc_text(1))
    assert stub.calls and all(c.get("max_len") == 256 for c in stub.calls)

    class NoMaxLen(StubExtractor):
        def extract_entities(self, text, entity_types, threshold=0.5,
                             include_confidence=False, include_spans=False):
            return super().extract_entities(text, entity_types)

    ann = GLiNER2Annotator(extractor=NoMaxLen(), taxonomy=get_default_taxonomy(), max_len=256)
    entities, *_ = ann.annotate_text(make_doc_text(1))
    assert "Einstein" in entities


def test_select_device_respects_request():
    assert select_device("cpu") == "cpu"
    assert select_device(None) in ("cuda", "mps", "cpu")


# ----------------------------------------------------------------------
# C4: text field and zero-document sources
# ----------------------------------------------------------------------


def patch_stream(monkeypatch, rows_by_source):
    """Replace HF loading with in-memory rows."""
    monkeypatch.setattr(
        SourceLoader,
        "_load_stream",
        lambda self, source, config: list(rows_by_source[source]),
    )


def test_missing_text_field_raises_clear_error(monkeypatch):
    patch_stream(monkeypatch, {"src": [{"content": "hello"}]})
    loader = SourceLoader(["src"], [1.0])
    with pytest.raises(ValueError, match="no text field 'text'.*content"):
        next(iter(loader))


def test_zero_usable_documents_raises_instead_of_looping(monkeypatch):
    patch_stream(monkeypatch, {"src": [{"text": ""}, {"text": "   "}]})
    loader = SourceLoader(["src"], [1.0])
    with pytest.raises(RuntimeError, match="zero usable documents"):
        next(iter(loader))


def test_text_field_override(monkeypatch):
    patch_stream(monkeypatch, {"src": [{"content": "hello world"}]})
    loader = SourceLoader(["src"], [1.0], text_field=parse_text_field_args(["src=content"]))
    assert next(iter(loader)).text == "hello world"
    assert parse_text_field_args(["body", "a=b"]) == {"*": "body", "a": "b"}
    assert parse_text_field_args(None) is None


def test_start_offsets_skip_consumed_rows(monkeypatch):
    patch_stream(monkeypatch, {"src": [{"text": f"doc {i}"} for i in range(6)]})
    loader = SourceLoader(["src"], [1.0], start_offsets={"src": 4})
    it = iter(loader)
    assert [next(it).text for _ in range(2)] == ["doc 4", "doc 5"]
    assert loader.get_offsets() == {"src": 6}


# ----------------------------------------------------------------------
# C13: dedup and validation split
# ----------------------------------------------------------------------


def test_hash_is_normalized_and_split_is_deterministic():
    assert text_hash("Hello   World\n") == text_hash("hello world")
    assert normalize_text("  A\tB  ") == "a b"
    hashes = [text_hash(f"doc {i}") for i in range(2000)]
    val = [is_validation(h, 0.1) for h in hashes]
    assert val == [is_validation(h, 0.1) for h in hashes]
    assert 100 < sum(val) < 300
    assert not any(is_validation(h, 0.0) for h in hashes)


def test_writer_split_is_deterministic_across_runs(tmp_path):
    texts = [f"sample text {i}" for i in range(200)]
    outputs = []
    for run in ("one", "two"):
        writer = OutputWriter(str(tmp_path), run, val_ratio=0.2)
        writer.start()
        writer.write_batch([result(t) for t in texts])
        writer.finalize()
        val = [r["text"] for r in read_rows(tmp_path / f"{run}_val.jsonl")]
        train = [r["text"] for r in read_rows(tmp_path / f"{run}.jsonl")]
        assert sorted(val + train) == sorted(texts)
        assert 10 < len(val) < 80
        outputs.append(val)
    assert outputs[0] == outputs[1]


def test_writer_dedup_state():
    writer = OutputWriter.__new__(OutputWriter)
    from data.curate.output_writer import CheckpointState
    writer.state = CheckpointState()
    h = text_hash("Some Text")
    assert not writer.is_duplicate(h)
    writer.mark_seen(h)
    assert writer.is_duplicate(text_hash("some   text"))


# ----------------------------------------------------------------------
# C2: resume truncates rows written after the checkpoint
# ----------------------------------------------------------------------


def test_resume_truncates_to_checkpoint(tmp_path):
    writer = OutputWriter(str(tmp_path), "r", checkpoint_every=1000, val_ratio=0.3)
    writer.start()
    for i in range(10):
        writer.mark_seen(text_hash(f"row {i}"))
        writer.write(result(f"row {i}"))
    writer.save_checkpoint()
    checkpointed = writer.state.train_written, writer.state.val_written
    for i in range(10, 15):
        writer.write(result(f"row {i}"))
    writer._flush_buffer()

    resumed = OutputWriter(str(tmp_path), "r", val_ratio=0.3)
    assert resumed.resume()
    n_train = len(read_rows(tmp_path / "r.jsonl"))
    n_val = len(read_rows(tmp_path / "r_val.jsonl"))
    assert (n_train, n_val) == checkpointed
    assert n_train + n_val == 10
    assert resumed.is_duplicate(text_hash("row 3"))
    assert not resumed.is_duplicate(text_hash("row 12"))


# ----------------------------------------------------------------------
# End to end through run_curation: C14 output name, C1 refusal, C2 resume
# ----------------------------------------------------------------------


@pytest.fixture
def curate(monkeypatch):
    module = load_curate_dataset()
    rows = [{"text": make_doc_text(i)} for i in range(40)]
    # Two exact duplicates (after normalization) to exercise dedup
    rows.insert(5, {"text": make_doc_text(2).upper()})
    rows.insert(9, {"text": make_doc_text(3) + "   "})
    patch_stream(monkeypatch, {"stub": rows})

    import data.curate as curate_pkg

    real = curate_pkg.GLiNER2Annotator
    monkeypatch.setattr(
        curate_pkg,
        "GLiNER2Annotator",
        lambda **kw: real(extractor=StubExtractor(), **{
            k: v for k, v in kw.items() if k not in ("model_name", "device")
        }),
    )
    return module


def make_config(out_dir, **kw):
    defaults = dict(
        target_samples=8, sources=["stub"], source_ratios=[1.0],
        output_dir=str(out_dir), output_name="mydata", annotator="gliner",
        batch_size=3, checkpoint_every=3, enable_safety_filter=False,
        val_ratio=0.25,
    )
    defaults.update(kw)
    return CurationConfig(**defaults)


def test_run_curation_honors_output_name_and_refuses_rerun(curate, tmp_path):
    curate.run_curation(make_config(tmp_path))
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "mydata.jsonl", "mydata_checkpoint.json", "mydata_val.jsonl", "mydata_vocab.json",
    ]
    rows = read_rows(tmp_path / "mydata.jsonl") + read_rows(tmp_path / "mydata_val.jsonl")
    assert len(rows) == 8
    assert len({normalize_text(r["text"]) for r in rows}) == 8
    for r in rows:
        for ent, (a, b) in zip(r["entities"], r["entity_spans"]):
            assert r["text"][a:b] == ent

    with pytest.raises(SystemExit):
        curate.run_curation(make_config(tmp_path))
    curate.run_curation(make_config(tmp_path, overwrite=True))
    assert len(read_rows(tmp_path / "mydata.jsonl")) + len(
        read_rows(tmp_path / "mydata_val.jsonl")
    ) == 8


def test_run_curation_resume_matches_straight_run(curate, tmp_path):
    straight = tmp_path / "straight"
    curate.run_curation(make_config(straight, target_samples=12))

    resumed = tmp_path / "resumed"
    curate.run_curation(make_config(resumed, target_samples=6))
    # Add rows past the checkpoint to simulate a crash after a flush
    with open(resumed / "mydata.jsonl", "a") as f:
        f.write(json.dumps({"text": "orphan row"}) + "\n")
    curate.run_curation(make_config(
        resumed, target_samples=12,
        resume_from=str(resumed / "mydata_checkpoint.json"),
    ))

    for name in ("mydata.jsonl", "mydata_val.jsonl"):
        a = [r["text"] for r in read_rows(straight / name)]
        b = [r["text"] for r in read_rows(resumed / name)]
        assert a == b


# ----------------------------------------------------------------------
# C7: chat template for QA prompts
# ----------------------------------------------------------------------


def test_apply_chat_template_only_for_instruct_tokenizers():
    class Base:
        chat_template = None

    class Instruct:
        chat_template = "{{ messages }}"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is True
            return "<chat>" + messages[-1]["content"]

    assert apply_chat_template(Base(), "hi") == "hi"
    assert apply_chat_template(None, "hi") == "hi"
    assert apply_chat_template(Instruct(), "hi") == "<chat>hi"


def test_unsampled_source_keeps_resume_offset(monkeypatch):
    patch_stream(monkeypatch, {
        "a": [{"text": f"a {i}"} for i in range(10)],
        "b": [{"text": f"b {i}"} for i in range(10)],
    })
    loader = SourceLoader(["a", "b"], [1.0, 0.0], start_offsets={"a": 2, "b": 7})
    next(iter(loader))
    assert loader.get_offsets() == {"a": 3, "b": 7}
