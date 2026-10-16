"""Tests for per-turn trace expansion and canonical tool-call normalization."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.chat_template import ChatTemplate
from data.curate.trace_annotator import (
    TraceAnnotator,
    expand_trace,
    select_target_turns,
)
from data.curate.trace_loader import (
    TraceDocument,
    format_tool_calls,
    is_tool_call,
    parse_glaive_sample,
    parse_messages_sample,
    parse_sharegpt_sample,
    parse_tool_call_text,
)

TRACE = [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "Read src/app/main.py please."},
    {"role": "assistant", "content": '{"name": "read_file", "arguments": {"path": "src/app/main.py"}}'},
    {"role": "tool", "content": "def load(path): see https://docs.example.com/load"},
    {"role": "assistant", "content": "It calls https://docs.example.com/load."},
    {"role": "user", "content": "Now check config/settings.yaml."},
    {"role": "assistant", "content": "config/settings.yaml looks fine."},
]


# ---------------------------------------------------------------------------
# Per-turn expansion
# ---------------------------------------------------------------------------

def expand(messages, max_turns=8):
    sample = TraceAnnotator().annotate_trace(messages)
    return sample, expand_trace(sample, select_target_turns(messages, max_turns))


def test_one_sample_per_assistant_turn_with_prefix():
    _, samples = expand(TRACE)
    assert len(samples) == 3
    for s, i in zip(samples, (2, 4, 6)):
        assert s["messages"] == TRACE[: i + 1]
        assert s["response"] == TRACE[i]["content"]
        assert s["should_respond"] == 1
        # The chat template sees the same target
        _, target = ChatTemplate().format_messages(s["messages"])
        assert target == s["response"]


def test_annotations_never_on_or_after_target():
    sample, samples = expand(TRACE)
    # Earlier assistant turns are encoder input, so they carry annotations
    assert {a["message_idx"] for a in sample["message_annotations"]} >= {1, 3, 4, 5}
    for s in samples:
        target_idx = len(s["messages"]) - 1
        assert all(a["message_idx"] < target_idx for a in s["message_annotations"])
    # The first sample keeps only the annotations before its target
    assert {a["message_idx"] for a in samples[0]["message_annotations"]} == {1}


def test_turn_cap_keeps_first_and_last_evenly():
    messages = [{"role": "user", "content": "go"}]
    for k in range(10):
        messages.append({"role": "assistant", "content": f"step {k}"})
        messages.append({"role": "tool", "content": f"result {k}"})
    turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]

    picked = select_target_turns(messages, 4)
    assert len(picked) == 4
    assert picked[0] == turns[0] and picked[-1] == turns[-1]
    assert picked == sorted(set(picked))
    assert select_target_turns(messages, 1) == [turns[-1]]
    assert select_target_turns(messages, 0) == turns


# ---------------------------------------------------------------------------
# Tool-call normalization
# ---------------------------------------------------------------------------

def test_glaive_single_quoted_arguments_become_object():
    chat = (
        "USER: Find books by George Orwell\n\n"
        "ASSISTANT: <functioncall> {\"name\": \"search_books\", "
        "\"arguments\": '{\"query\": \"can't stop\", \"author\": \"George Orwell\"}'} <|endoftext|>\n\n"
        "FUNCTION RESPONSE: {\"results\": [\"1984\"]}\n\n"
        "ASSISTANT: I found 1984. <|endoftext|>"
    )
    messages = parse_glaive_sample("SYSTEM: You can call functions.", chat)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]
    call = json.loads(messages[2]["content"])
    assert call == {
        "name": "search_books",
        "arguments": {"query": "can't stop", "author": "George Orwell"},
    }
    assert is_tool_call(messages[2]["content"])
    assert not is_tool_call(messages[4]["content"])


def test_glaive_escaped_apostrophe_in_quoted_arguments():
    body = "{\"name\": \"analyze\", \"arguments\": '{\"text\": \"The author\\'s book\"}'}"
    assert json.loads(parse_tool_call_text(body)) == {
        "name": "analyze", "arguments": {"text": "The author's book"},
    }


def test_glaive_object_arguments_and_missing_arguments():
    assert json.loads(parse_tool_call_text('{"name": "f", "arguments": {"x": 1}}')) == {
        "name": "f", "arguments": {"x": 1},
    }
    assert json.loads(parse_tool_call_text('{"name": "f"}')) == {"name": "f", "arguments": {}}


def test_openai_tool_calls_normalized():
    rows = [
        {"role": "user", "content": "Weather in Paris and Rome?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
            {"id": "b", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'}},
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "18C"},
        {"role": "function", "content": "21C"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"type": "function", "function": {"name": "done", "arguments": "{}"}},
        ]},
        {"role": "ipython", "content": "ok"},
        {"role": "assistant", "content": "Paris 18C, Rome 21C."},
    ]
    messages = parse_messages_sample(rows)
    assert [m["role"] for m in messages] == [
        "user", "assistant", "tool", "tool", "assistant", "tool", "assistant",
    ]
    # Several calls in one turn -> a JSON list
    assert json.loads(messages[1]["content"]) == [
        {"name": "get_weather", "arguments": {"city": "Paris"}},
        {"name": "get_weather", "arguments": {"city": "Rome"}},
    ]
    # One call -> a bare object
    assert json.loads(messages[4]["content"]) == {"name": "done", "arguments": {}}
    assert is_tool_call(messages[1]["content"]) and is_tool_call(messages[4]["content"])


def test_legacy_function_call_and_sharegpt_function_call():
    messages = parse_messages_sample([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None,
         "function_call": {"name": "greet", "arguments": '{"to": "you"}'}},
    ])
    assert json.loads(messages[1]["content"]) == {"name": "greet", "arguments": {"to": "you"}}

    messages = parse_sharegpt_sample([
        {"from": "human", "value": "hi"},
        {"from": "function-call", "value": '{"name": "greet", "arguments": "{\\"to\\": \\"you\\"}"}'},
        {"from": "observation", "value": "hello"},
        {"from": "gpt", "value": "Done."},
    ])
    assert json.loads(messages[1]["content"]) == {"name": "greet", "arguments": {"to": "you"}}


def test_malformed_arguments_cut_conversation():
    stats = {}
    chat = (
        "USER: Weather?\n"
        "ASSISTANT: Let me ask first. <|endoftext|>\n"
        "USER: Paris\n"
        "ASSISTANT: <functioncall> {\"name\": \"get_weather\", \"arguments\": '{\"city\": Paris}'} <|endoftext|>\n"
        "FUNCTION RESPONSE: 18C\n"
        "ASSISTANT: It is 18C. <|endoftext|>"
    )
    messages = parse_glaive_sample("", chat, stats)
    # Cut before the malformed call: no malformed target, no later turns
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert stats["malformed_tool_calls"] == 1

    # A malformed first call leaves no assistant turn at all
    stats = {}
    assert parse_glaive_sample(
        "", "USER: x\nASSISTANT: <functioncall> {\"name\": \"f\", \"arguments\": '[1, 2]'}", stats
    ) is None
    assert stats["malformed_tool_calls"] == 1

    stats = {}
    assert parse_messages_sample([
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"function": {"name": "f", "arguments": "{not json"}}]},
    ], stats) is None
    assert stats["malformed_tool_calls"] == 1


def test_plain_json_text_is_not_a_malformed_call():
    # Assistant text that looks like JSON but has no call marker stays as is
    messages = parse_messages_sample([
        {"role": "user", "content": "Give me JSON"},
        {"role": "assistant", "content": '{"a": 1}'},
    ])
    assert messages[1]["content"] == '{"a": 1}'
    assert not is_tool_call('{"a": 1}')


def test_target_json_round_trips():
    text = format_tool_calls([{"name": "f", "arguments": {"msg": "café ü", "n": [1, 2]}}])
    assert json.loads(text) == {"name": "f", "arguments": {"msg": "café ü", "n": [1, 2]}}
    assert "café" in text  # ensure_ascii=False keeps the text readable
    assert json.loads(parse_tool_call_text(text)) == json.loads(text)


# ---------------------------------------------------------------------------
# curate_traces.py end to end (fake loader, no network)
# ---------------------------------------------------------------------------

def make_conversation(k):
    return [
        {"role": "user", "content": f"Check https://example.com/item/{k}"},
        {"role": "assistant", "content": json.dumps(
            {"name": "fetch", "arguments": {"url": f"https://example.com/item/{k}"}}
        )},
        {"role": "tool", "content": f"item {k} ok"},
        {"role": "assistant", "content": f"Item {k} is ok."},
    ]


class FakeLoader:
    def __init__(self, *args, **kwargs):
        self.convs = [make_conversation(k) for k in range(60)]
        self.convs.append(make_conversation(0))  # an exact duplicate

    def __iter__(self):
        for k, messages in enumerate(self.convs):
            yield TraceDocument(messages=messages, source="fake", doc_id=f"fake_{k}")

    def get_statistics(self):
        return {"yielded": len(self.convs), "skipped": 0, "format": "messages",
                "malformed_tool_calls": 0}


def run_curate(monkeypatch, tmp_path, *extra):
    import data.curate_traces as ct

    monkeypatch.setattr(ct, "TraceSourceLoader", FakeLoader)
    monkeypatch.setattr(sys, "argv", [
        "curate_traces.py", "--no-gliner", "--output-dir", str(tmp_path),
        "--val-ratio", "0.3", *extra,
    ])
    ct.main()
    read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
    return read(tmp_path / "traces.jsonl"), read(tmp_path / "traces_val.jsonl")


def test_split_by_conversation(monkeypatch, tmp_path):
    train, val = run_curate(monkeypatch, tmp_path, "--target-samples", "1000")
    assert train and val
    train_ids = {r["conversation_id"] for r in train}
    val_ids = {r["conversation_id"] for r in val}
    assert not train_ids & val_ids
    # 60 unique conversations (duplicate skipped), 2 turns each
    assert len(train_ids | val_ids) == 60
    assert len(train) + len(val) == 120
    # Every target that is a tool call round-trips as JSON
    tool_targets = [r for r in train + val if is_tool_call(r["response"])]
    assert len(tool_targets) == 60
    for r in tool_targets:
        assert json.loads(r["response"])["name"] == "fetch"
    assert (tmp_path / "traces_vocab.json").exists()


def test_target_samples_stops_at_conversation_boundary(monkeypatch, tmp_path):
    train, val = run_curate(monkeypatch, tmp_path, "--target-samples", "5")
    rows = train + val
    # 5 samples requested, 2 per conversation -> 3 whole conversations
    assert len(rows) == 6
    ids = [r["conversation_id"] for r in rows]
    assert all(ids.count(i) == 2 for i in ids)


def test_overwrite_guard(monkeypatch, tmp_path):
    run_curate(monkeypatch, tmp_path, "--target-samples", "2")
    with pytest.raises(SystemExit):
        run_curate(monkeypatch, tmp_path, "--target-samples", "2")
