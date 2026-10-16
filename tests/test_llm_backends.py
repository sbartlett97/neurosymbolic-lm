"""Tests for the OpenAI and Anthropic API backends with mocked clients.

No test makes a network call. The fake clients copy the response shapes
of the official SDKs (``messages.create`` and ``chat.completions.create``).
"""

import json
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.curate.gliner_annotator import GLiNER2Annotator
from data.curate.llm_annotator import (
    AnthropicBackend,
    LLMAnnotator,
    OpenAIBackend,
    is_retryable_error,
)
from data.curate.prompts import QA_JSON_SCHEMA

SECRET = "sk-test-SECRET-do-not-log-123"
QA_JSON = json.dumps({"question": "Who won?", "answer": "Curie.", "response_reason": "question"})


class FakeStatusError(Exception):
    """Like an SDK APIStatusError: carries status_code."""

    def __init__(self, status_code):
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


class APITimeoutError(Exception):
    """Same class name as the SDK timeout error."""


def anthropic_response(text, stop_reason="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason
    )


def openai_response(text, finish_reason="stop", refusal=None):
    message = SimpleNamespace(content=text, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


class FakeAnthropicClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.lock = threading.Lock()
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        with self.lock:
            self.calls.append(kwargs)
            outcome = self.outcomes.pop(0) if self.outcomes else anthropic_response(QA_JSON)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeOpenAIClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        with self.lock:
            self.calls.append(kwargs)
            outcome = self.outcomes.pop(0) if self.outcomes else openai_response(QA_JSON)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def no_sleep(_):
    pass


# ----------------------------------------------------------------------
# Anthropic
# ----------------------------------------------------------------------


def test_anthropic_success_sends_structured_output():
    client = FakeAnthropicClient([anthropic_response(QA_JSON)])
    backend = AnthropicBackend(client=client, sleep=no_sleep)
    out = backend.generate_chat("Q?", system="sys", json_schema=QA_JSON_SCHEMA)
    assert json.loads(out)["answer"] == "Curie."
    call = client.calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["system"] == "sys"
    assert call["messages"] == [{"role": "user", "content": "Q?"}]
    assert call["output_config"] == {
        "format": {"type": "json_schema", "schema": QA_JSON_SCHEMA}
    }
    # Parameters that current models reject are never sent
    assert "temperature" not in call and "thinking" not in call


def test_anthropic_retries_429_then_succeeds():
    delays = []
    client = FakeAnthropicClient([FakeStatusError(429), FakeStatusError(529), anthropic_response(QA_JSON)])
    backend = AnthropicBackend(client=client, sleep=delays.append, base_delay=1.0)
    assert backend.generate_single("Q?") == QA_JSON
    assert len(client.calls) == 3
    # Exponential backoff with jitter: delay k is in [2^k / 2, 2^k]
    assert len(delays) == 2
    assert 0.5 <= delays[0] <= 1.0 and 1.0 <= delays[1] <= 2.0
    assert backend.get_statistics()["retries"] == 2


def test_anthropic_does_not_retry_bad_request():
    client = FakeAnthropicClient([FakeStatusError(400)])
    backend = AnthropicBackend(client=client, sleep=no_sleep)
    assert backend.generate_single("Q?") == ""
    assert len(client.calls) == 1


def test_anthropic_gives_up_after_max_retries():
    client = FakeAnthropicClient([FakeStatusError(503)] * 10)
    backend = AnthropicBackend(client=client, sleep=no_sleep, max_retries=3)
    assert backend.generate_single("Q?") == ""
    assert len(client.calls) == 4


def test_anthropic_refusal_and_truncation_are_failures():
    client = FakeAnthropicClient([
        anthropic_response("", stop_reason="refusal"),
        anthropic_response('{"question": "Wh', stop_reason="max_tokens"),
    ])
    backend = AnthropicBackend(client=client, sleep=no_sleep)
    assert backend.generate_chat_batch(["a", "b"]) == ["", ""]


def test_anthropic_effort_sent_only_when_set():
    client = FakeAnthropicClient([])
    AnthropicBackend(client=client, model_name="claude-sonnet-5-5", effort="low").generate_single("x")
    AnthropicBackend(client=client, effort="default").generate_single("x")
    assert client.calls[0]["output_config"] == {"effort": "low"}
    assert "output_config" not in client.calls[1]


# ----------------------------------------------------------------------
# OpenAI
# ----------------------------------------------------------------------


def test_openai_success_sends_json_schema():
    client = FakeOpenAIClient([openai_response(QA_JSON)])
    backend = OpenAIBackend(client=client, sleep=no_sleep)
    out = backend.generate_chat("Q?", system="sys", json_schema=QA_JSON_SCHEMA)
    assert json.loads(out)["question"] == "Who won?"
    call = client.calls[0]
    assert call["model"] == "gpt-6-luna"
    assert call["messages"][0] == {"role": "system", "content": "sys"}
    assert call["max_completion_tokens"] == 1024
    assert call["reasoning_effort"] == "low"
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["schema"] == QA_JSON_SCHEMA
    assert call["response_format"]["json_schema"]["strict"] is True
    assert "temperature" not in call and "max_tokens" not in call


def test_openai_without_schema_uses_json_object_mode():
    client = FakeOpenAIClient([])
    OpenAIBackend(client=client).generate_single("Return JSON")
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_openai_retries_429_and_timeout_then_succeeds():
    client = FakeOpenAIClient([FakeStatusError(429), APITimeoutError("timed out"), openai_response(QA_JSON)])
    backend = OpenAIBackend(client=client, sleep=no_sleep)
    assert backend.generate_single("Q?") == QA_JSON
    assert len(client.calls) == 3


def test_openai_length_and_refusal_are_failures():
    client = FakeOpenAIClient([
        openai_response('{"q', finish_reason="length"),
        openai_response(None, refusal="I can't help"),
    ])
    backend = OpenAIBackend(client=client, sleep=no_sleep)
    assert backend.generate_chat_batch(["a", "b"]) == ["", ""]


# ----------------------------------------------------------------------
# Shared behavior
# ----------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 529])
def test_retryable_statuses(status):
    assert is_retryable_error(FakeStatusError(status))


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_non_retryable_statuses(status):
    assert not is_retryable_error(FakeStatusError(status))


def test_batch_keeps_order_and_isolates_failures():
    class OrderClient(FakeOpenAIClient):
        def _create(self, **kwargs):
            prompt = kwargs["messages"][-1]["content"]
            if prompt == "p3":
                raise FakeStatusError(400)
            return openai_response(prompt.upper())

    backend = OpenAIBackend(client=OrderClient([]), concurrency=4, sleep=no_sleep)
    prompts = [f"p{i}" for i in range(10)]
    out = backend.generate(prompts)
    assert out == [("" if p == "p3" else p.upper()) for p in prompts]


def test_api_key_read_from_env_only(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicBackend()
    seen = {}
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setattr(OpenAIBackend, "_make_client", lambda self, key: seen.setdefault("key", key))
    backend = OpenAIBackend()
    assert seen["key"] == SECRET
    # The key is not kept on the backend
    assert SECRET not in json.dumps({k: str(v) for k, v in vars(backend).items() if k != "client"})


def test_api_key_never_in_repr_or_logs(monkeypatch, caplog, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)

    class KeyedClient(FakeAnthropicClient):
        def __init__(self, key):
            super().__init__([FakeStatusError(401)])
            self.api_key = key

    monkeypatch.setattr(AnthropicBackend, "_make_client", lambda self, key: KeyedClient(key))
    backend = AnthropicBackend(sleep=no_sleep)
    with caplog.at_level(logging.DEBUG):
        assert backend.generate_single("x") == ""
    assert SECRET not in repr(backend) and SECRET not in str(backend)
    assert SECRET not in caplog.text
    captured = capsys.readouterr()
    assert SECRET not in captured.out and SECRET not in captured.err


def test_llm_annotator_api_backend_selection(monkeypatch):
    monkeypatch.setattr(AnthropicBackend, "_make_client", lambda self, key: FakeAnthropicClient([]))
    monkeypatch.setattr(OpenAIBackend, "_make_client", lambda self, key: FakeOpenAIClient([]))
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    a = LLMAnnotator(model_name=None, backend="anthropic", api_concurrency=3)
    assert isinstance(a.backend, AnthropicBackend)
    assert a.model_name == "claude-haiku-4-5" and a.backend.concurrency == 3
    o = LLMAnnotator(model_name=None, backend="openai")
    assert o.model_name == "gpt-6-luna"
    # 'api' picks the provider from the model name
    assert isinstance(LLMAnnotator(model_name="claude-sonnet-5-5", backend="api").backend, AnthropicBackend)
    assert isinstance(LLMAnnotator(model_name="gpt-6-luna", backend="api").backend, OpenAIBackend)


def test_hybrid_qa_through_api_backend_parses():
    """GLiNER2 hybrid QA with an API backend: one batch call, JSON parses."""
    from tests.test_gliner_annotator import TEXT, StubExtractor
    from data.curate.source_loader import DocumentSource

    client = FakeAnthropicClient([anthropic_response(QA_JSON), FakeStatusError(429), anthropic_response("not json")])
    backend = AnthropicBackend(client=client, sleep=no_sleep, concurrency=1)
    ann = GLiNER2Annotator(extractor=StubExtractor(), response_backend=backend)
    docs = [DocumentSource(text=TEXT, source="t", doc_id=str(i), metadata={}) for i in range(2)]
    results = ann.annotate_batch(docs, include_response=True)
    assert [r.should_respond for r in results] == [1, 0]
    assert results[0].response == "Curie."
    assert all(c["output_config"]["format"]["schema"] == QA_JSON_SCHEMA for c in client.calls)
    stats = ann.get_statistics()
    assert stats["qa_generated"] == 1 and stats["qa_failed"] == 1
    assert stats["qa_failed_rate"] == 0.5
