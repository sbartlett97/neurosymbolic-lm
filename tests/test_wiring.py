"""Regression tests for training-pipeline wiring: trace samples through the
dataset loader, device resolution, and the KG-free model surface."""

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")

from data.dataset import ToyCognitiveDataset  # noqa: E402


def load_train_module():
    spec = importlib.util.spec_from_file_location("train_mod", REPO / "train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TRACE_SAMPLE = {
    "messages": [
        {"role": "user", "content": "Check src/app/main.py please."},
        {"role": "assistant", "content": "Done - it looks fine."},
    ],
    "message_annotations": [
        {
            "message_idx": 0,
            "entities": ["src/app/main.py"],
            "entity_spans": [[6, 21]],
            "concepts": [["file_path"]],
            "entity_types": ["digital_artifact"],
            "relations": [],
        }
    ],
    "should_respond": 1,
    "response": "Done - it looks fine.",
}


def test_dataset_loads_trace_samples(tmp_path):
    """Trace samples (no top-level 'text') must load without KeyError and
    must not be rewritten by the statement->question processing."""
    path = tmp_path / "traces.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps(TRACE_SAMPLE) + "\n")
        f.write(json.dumps({
            "text": "Paris is the capital of France.",
            "entities": ["Paris", "France"],
            "concepts": [["city"], ["country"]],
            "relations": [[0, 1, "capital_of"]],
            "should_respond": 0,
        }) + "\n")

    ds = ToyCognitiveDataset(str(path))
    trace_entries = [s for s in ds if "messages" in s]
    assert len(trace_entries) == 1
    assert trace_entries[0]["message_annotations"][0]["entities"] == ["src/app/main.py"]
    assert trace_entries[0]["should_respond"] == 1  # not duplicated/rewritten


def test_extract_vocab_harvests_message_annotations(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(TRACE_SAMPLE) + "\n")
    m = load_train_module()
    ds = ToyCognitiveDataset(str(path))
    concept_map, relation_map, _, _, _, _ = m.extract_vocab_from_dataset(ds)
    assert "file_path" in concept_map


def test_resolve_device():
    m = load_train_module()
    # In this environment no accelerator is present; auto and explicit
    # requests must both degrade gracefully rather than crash.
    resolved = m.resolve_device("auto")
    assert resolved in ("cuda", "mps", "cpu")
    assert m.resolve_device("cpu") == "cpu"
    if not torch.cuda.is_available():
        assert m.resolve_device("cuda") in ("mps", "cpu")


def test_model_surface_has_no_kg_parameters():
    from model.neurosymbolic import NeuroSymbolicLM

    init_params = set(inspect.signature(NeuroSymbolicLM.__init__).parameters)
    for removed in ("use_kg", "kg_embed_dim", "use_kg_gnn", "use_path_reasoning"):
        assert removed not in init_params

    fwd_params = set(inspect.signature(NeuroSymbolicLM.forward).parameters)
    for removed in ("kg_paths", "kg_relation_ids", "kg_adjacency", "entity_names"):
        assert removed not in fwd_params
    # The broadcast-critical inputs are still there
    for kept in ("input_ids", "attention_mask", "spans", "y_ids"):
        assert kept in fwd_params


def test_packages_import_without_kg_modules():
    import model  # noqa: F401
    import training  # noqa: F401
    import continual_learning  # noqa: F401
    from continual_learning.symbolic_updates import SymbolicUpdateManager  # noqa: F401

    assert not (REPO / "kg_utils.py").exists()
    assert not (REPO / "model" / "kg_relation_encoder.py").exists()


def test_collator_handles_annotationless_chat_samples():
    """A messages-only sample without annotations must not KeyError."""
    from data.chat_template import ChatTemplate
    from data.collator import CognitiveCollator
    from data.curate.taxonomy import get_default_taxonomy
    from tests.test_collator import FakeTokenizer

    vocab = get_default_taxonomy().vocab()
    collator = CognitiveCollator(
        tokenizer=FakeTokenizer(),
        concept_map=vocab["concepts"],
        relation_map=vocab["relations"],
        chat_mode=True,
        chat_template=ChatTemplate(),
        include_responses=True,
    )
    batch = collator([
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
            "should_respond": 1,
        }
    ])
    assert batch["input_ids"].shape[0] == 1
