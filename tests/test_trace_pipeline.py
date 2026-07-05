"""Tests for trace annotation: chat offsets, code annotator, trace annotator,
trace loader parsing, and collator flattening of message annotations."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.chat_template import ChatTemplate
from data.curate.code_annotator import CodeAnnotator
from data.curate.trace_annotator import TraceAnnotator, regex_artifact_matches
from data.curate.trace_loader import (
    parse_glaive_sample,
    parse_messages_sample,
    parse_sharegpt_sample,
)
from data.curate.taxonomy import get_default_taxonomy


# ---------------------------------------------------------------------------
# ChatTemplate offsets
# ---------------------------------------------------------------------------

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Run the tests in tests/test_app.py please."},
    {"role": "assistant", "content": '{"name": "run_tests", "arguments": {"path": "tests/test_app.py"}}'},
    {"role": "tool", "content": "3 passed in 1.2s"},
    {"role": "assistant", "content": "All 3 tests passed."},
]


def test_offsets_point_at_message_content():
    tmpl = ChatTemplate()
    input_text, target, offsets = tmpl.format_messages_with_offsets(MESSAGES)

    # Final assistant turn is the target, not in the input
    assert target == "All 3 tests passed."
    assert offsets[4] is None

    for i in (0, 1, 2, 3):
        content = MESSAGES[i]["content"].strip()
        assert offsets[i] is not None
        assert input_text[offsets[i]:offsets[i] + len(content)] == content

    assert input_text.rstrip().endswith("<assistant>")


def test_offsets_match_format_messages():
    tmpl = ChatTemplate()
    a = tmpl.format_messages(MESSAGES)
    b = tmpl.format_messages_with_offsets(MESSAGES)
    assert a == (b[0], b[1])


def test_offsets_with_injected_default_system():
    tmpl = ChatTemplate()
    msgs = [
        {"role": "user", "content": "Hello there."},
        {"role": "assistant", "content": "Hi!"},
    ]
    input_text, target, offsets = tmpl.format_messages_with_offsets(msgs)
    assert target == "Hi!"
    assert input_text.startswith("<system>")
    assert input_text[offsets[0]:offsets[0] + len("Hello there.")] == "Hello there."


def test_tool_role_marker():
    tmpl = ChatTemplate()
    input_text, _, offsets = tmpl.format_messages_with_offsets(MESSAGES)
    assert "<tool> 3 passed in 1.2s" in input_text


# ---------------------------------------------------------------------------
# Code annotator
# ---------------------------------------------------------------------------

CODE = '''import os
from pathlib import Path

class Loader:
    def read(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return Path(path).read_text()
'''


def test_code_annotator_extracts_defs_imports_calls():
    ann = CodeAnnotator().annotate(CODE, "python")

    assert "Loader" in ann["entities"]
    assert "read" in ann["entities"]
    assert "os" in ann["entities"]
    assert "pathlib" in ann["entities"]

    # Spans are exact
    for ent, span in zip(ann["entities"], ann["entity_spans"]):
        assert CODE[span[0]:span[1]] == ent

    # Types and concepts come from the taxonomy code branch
    idx = ann["entities"].index("Loader")
    assert ann["concepts"][idx] == ["class_name"]
    assert ann["entity_types"][idx] == "code_construct"

    # Relations: read defined_in Loader, read calls exists, read raises FileNotFoundError
    read_i = ann["entities"].index("read")
    loader_i = ann["entities"].index("Loader")
    exc_i = ann["entities"].index("FileNotFoundError")
    assert [read_i, loader_i, "defined_in"] in ann["relations"]
    assert [read_i, exc_i, "raises"] in ann["relations"]
    assert any(r[2] == "calls" and r[0] == read_i for r in ann["relations"])


def test_code_annotator_handles_syntax_errors_and_unknown_languages():
    ann = CodeAnnotator()
    assert ann.annotate("def broken(:", "python")["entities"] == []
    assert ann.annotate("SELECT * FROM users;", "sql")["entities"] == []
    assert ann.annotate("", "python")["entities"] == []


def test_code_annotator_taxonomy_alignment():
    tax = get_default_taxonomy()
    ann = CodeAnnotator().annotate(CODE, "python")
    for concepts, etype in zip(ann["concepts"], ann["entity_types"]):
        assert etype in tax.entity_types
        for c in concepts:
            assert c in tax.concept_labels
    for _, _, rel in ann["relations"]:
        assert rel in tax.relation_labels


# ---------------------------------------------------------------------------
# Regex artifact extraction
# ---------------------------------------------------------------------------

def test_regex_artifacts():
    text = (
        "Deploy v2.1.0 to https://api.example.com/health and check "
        "src/utils/config.py, then email ops@example.com and set API_KEY."
    )
    matches = regex_artifact_matches(text)
    by_label = {}
    for m in matches:
        by_label.setdefault(m.label, []).append(text[m.start:m.end])

    assert "https://api.example.com/health" in by_label["url"]
    assert "src/utils/config.py" in by_label["file_path"]
    assert "ops@example.com" in by_label["email_address"]
    assert "v2.1.0" in by_label["version_number"]
    assert "API_KEY" in by_label["environment_variable"]


# ---------------------------------------------------------------------------
# Trace annotator (deterministic only — no GLiNER2 needed)
# ---------------------------------------------------------------------------

TRACE = [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "Fix the bug in src/app/main.py:\n```python\ndef load(path):\n    raise ValueError(path)\n```"},
    {"role": "assistant", "content": "Looking at https://docs.example.com/errors now."},
    {"role": "tool", "content": "GET https://docs.example.com/errors -> 200"},
    {"role": "assistant", "content": "Fixed: load() now validates its input."},
]


def test_trace_annotator_produces_message_annotations():
    annotator = TraceAnnotator()  # no GLiNER2: regex + AST only
    sample = annotator.annotate_trace(TRACE)

    assert sample["should_respond"] == 1
    assert sample["response"] == "Fixed: load() now validates its input."

    anns = {a["message_idx"]: a for a in sample["message_annotations"]}
    # Final assistant turn (idx 4) must not be annotated
    assert 4 not in anns

    # User message: file path from regex + function/exception from AST
    user_ann = anns[1]
    content = TRACE[1]["content"].strip()
    assert "src/app/main.py" in user_ann["entities"]
    assert "load" in user_ann["entities"]
    assert "ValueError" in user_ann["entities"]
    for ent, span in zip(user_ann["entities"], user_ann["entity_spans"]):
        assert content[span[0]:span[1]] == ent

    load_i = user_ann["entities"].index("load")
    exc_i = user_ann["entities"].index("ValueError")
    assert [load_i, exc_i, "raises"] in user_ann["relations"]

    # Assistant intermediate turn: URL extracted
    assert "https://docs.example.com/errors" in anns[2]["entities"]


def test_trace_annotator_no_assistant_turn():
    annotator = TraceAnnotator()
    sample = annotator.annotate_trace([
        {"role": "user", "content": "See https://example.com now."}
    ])
    assert sample["should_respond"] == 0
    assert sample["response"] == ""


# ---------------------------------------------------------------------------
# Trace loader parsing
# ---------------------------------------------------------------------------

def test_parse_glaive():
    system = 'SYSTEM: You are a helpful assistant with access to functions.'
    chat = (
        "USER: What's the tip on $50? "
        "ASSISTANT: <functioncall> {\"name\": \"calculate_tip\"} <|endoftext|> "
        "FUNCTION RESPONSE: {\"tip_amount\": 7.5} "
        "ASSISTANT: The tip is $7.50. <|endoftext|>"
    )
    messages = parse_glaive_sample(system, chat)
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert "calculate_tip" in messages[2]["content"]
    assert "<|endoftext|>" not in messages[-1]["content"]
    assert messages[-1]["content"] == "The tip is $7.50."


def test_parse_glaive_rejects_no_assistant():
    assert parse_glaive_sample("SYSTEM: hi", "USER: hello") is None


def test_parse_sharegpt():
    conversations = [
        {"from": "system", "value": "Be helpful."},
        {"from": "human", "value": "Hi"},
        {"from": "gpt", "value": "Hello!"},
        {"from": "observation", "value": "tool output"},
        {"from": "gpt", "value": "Done."},
    ]
    messages = parse_sharegpt_sample(conversations)
    assert [m["role"] for m in messages] == [
        "system", "user", "assistant", "tool", "assistant"
    ]


def test_parse_messages_passthrough():
    messages = parse_messages_sample([
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "invalid", "content": "drop me"},
    ])
    assert [m["role"] for m in messages] == ["user", "assistant"]


# ---------------------------------------------------------------------------
# Collator flattening (requires torch)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")

from data.collator import CognitiveCollator  # noqa: E402
from tests.test_collator import FakeTokenizer  # noqa: E402


def make_chat_collator():
    tax = get_default_taxonomy()
    vocab = tax.vocab()
    return CognitiveCollator(
        tokenizer=FakeTokenizer(),
        concept_map=vocab["concepts"],
        relation_map=vocab["relations"],
        concept_to_entity_type_map=vocab["concept_to_entity_type"],
        entity_type_map=vocab["entity_types"],
        chat_mode=True,
        chat_template=ChatTemplate(),
        include_responses=True,
    )


def test_collator_flattens_message_annotations():
    annotator = TraceAnnotator()
    sample = annotator.annotate_trace(TRACE)

    collator = make_chat_collator()
    flat = collator._flatten_message_annotations(sample)

    input_text, _, _ = ChatTemplate().format_messages_with_offsets(TRACE)
    assert flat["text"] == input_text

    # All flattened spans must point at their entity in the encoder input
    assert flat["entities"], "expected flattened entities"
    for ent, span in zip(flat["entities"], flat["entity_spans"]):
        assert input_text[span[0]:span[1]] == ent

    # Relation indices were re-based across messages
    for h, t, rel in flat["relations"]:
        assert 0 <= h < len(flat["entities"])
        assert 0 <= t < len(flat["entities"])

    load_i = flat["entities"].index("load")
    exc_i = flat["entities"].index("ValueError")
    assert [load_i, exc_i, "raises"] in flat["relations"]


def test_collator_batches_trace_samples_end_to_end():
    annotator = TraceAnnotator()
    sample = annotator.annotate_trace(TRACE)

    collator = make_chat_collator()
    batch = collator([sample])

    assert batch["input_ids"].shape[0] == 1
    assert batch["should_respond"][0].item() == 1
    assert "decoder_input_ids" in batch

    # Entity types resolved through the explicit entity_types field
    tax = get_default_taxonomy()
    types = batch["entity_type_labels"][0]
    flat = collator._flatten_message_annotations(sample)
    path_i = flat["entities"].index("src/app/main.py")
    assert types[path_i].item() == tax.entity_type_id("digital_artifact")
    load_i = flat["entities"].index("load")
    assert types[load_i].item() == tax.entity_type_id("code_construct")

    # Token spans line up with the flattened char spans (FakeTokenizer is
    # whitespace-based, so multi-token entities resolve to word ranges)
    token_spans = batch["entity_token_spans"][0]
    assert len(token_spans) == len(flat["entities"])
