"""Loading and normalization of assistant-trace datasets.

Streams conversation datasets from HuggingFace and normalizes them into the
internal message format:

    [{"role": "system"|"user"|"assistant"|"tool", "content": str}, ...]

Tool/function calls are kept inline as the assistant turn's content, and
tool results become "tool" turns — the decoder learns to emit the call, and
the symbolic heads see the result as encoder input.

Every source uses one canonical tool-call format. A tool call is assistant
content that is exactly a JSON object:

    {"name": "get_weather", "arguments": {"city": "Paris"}}

"arguments" is always a JSON object, never a JSON string. A turn with
several calls (OpenAI parallel tool_calls) is a JSON list of such objects,
so one decoder target holds the whole turn. A missing "arguments" key
becomes {}. A call whose arguments do not parse into an object is
malformed: the conversation is cut just before that turn, so a malformed
call never becomes a target.

Supported formats:
- glaive: glaiveai/glaive-function-calling-v2 ("system" + "chat" strings
  with USER:/ASSISTANT:/FUNCTION RESPONSE: markers)
- sharegpt: {"conversations": [{"from": ..., "value": ...}]}
- messages: already-normalized {"messages": [{"role", "content"}]}
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

END_OF_TEXT = "<|endoftext|>"

_GLAIVE_MARKER_RE = re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)")
GLAIVE_CALL_MARKER = "<functioncall>"

# Glaive writes the arguments as a single-quoted JSON string:
#   {"name": "f", "arguments": '{"x": 1}'}
# The match is greedy, so an apostrophe inside a value ("can't") does not
# end the string early.
_GLAIVE_QUOTED_ARGS_RE = re.compile(
    r'^\{\s*"name"\s*:\s*(".*?")\s*,\s*"arguments"\s*:\s*\'(.*)\'\s*\}$', re.DOTALL
)

_SHAREGPT_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "chatgpt": "assistant",
    "function-call": "assistant",
    "function_call": "assistant",
    "tool": "tool",
    "observation": "tool",
    "function-response": "tool",
    "function_response": "tool",
    "function": "tool",
    "ipython": "tool",
}

# ShareGPT roles whose value is a tool call, not free text
_SHAREGPT_CALL_ROLES = {"function-call", "function_call"}

_MESSAGE_ROLE_MAP = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
    "function": "tool",
    "ipython": "tool",
}


class MalformedToolCall(ValueError):
    """A tool call whose name or arguments do not parse."""


def normalize_tool_call(call: Any) -> Dict[str, Any]:
    """Return {"name": str, "arguments": dict} or raise MalformedToolCall.

    Accepts a dict with "name"/"arguments" or an OpenAI tool_calls entry
    ({"type": "function", "function": {...}}). String arguments are
    parsed as JSON.
    """
    if isinstance(call, dict) and isinstance(call.get("function"), dict):
        call = call["function"]
    if not isinstance(call, dict):
        raise MalformedToolCall(f"tool call is not an object: {call!r:.80}")
    name = call.get("name")
    if not isinstance(name, str) or not name.strip():
        raise MalformedToolCall("tool call has no name")
    arguments = call.get("arguments", {})
    if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
        arguments = {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as e:
            raise MalformedToolCall(f"arguments do not parse: {e}") from None
    if not isinstance(arguments, dict):
        raise MalformedToolCall("arguments are not a JSON object")
    return {"name": name.strip(), "arguments": arguments}


def format_tool_calls(calls: List[Any]) -> str:
    """Serialize calls in the canonical format (object, or list for >1)."""
    if not calls:
        raise MalformedToolCall("empty tool call list")
    normalized = [normalize_tool_call(c) for c in calls]
    payload = normalized[0] if len(normalized) == 1 else normalized
    return json.dumps(payload, ensure_ascii=False)


def is_tool_call(content: str) -> bool:
    """True when content is a canonical tool call (object or list)."""
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return False
    calls = payload if isinstance(payload, list) else [payload]
    if not calls:
        return False
    for call in calls:
        if not (
            isinstance(call, dict)
            and set(call) == {"name", "arguments"}
            and isinstance(call["name"], str)
            and isinstance(call["arguments"], dict)
        ):
            return False
    return True


def parse_tool_call_text(text: str) -> str:
    """Normalize a tool call written as text (JSON object or list).

    Handles the glaive single-quoted arguments string. Raises
    MalformedToolCall when the text does not parse.
    """
    text = text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        m = _GLAIVE_QUOTED_ARGS_RE.match(text)
        if m is None:
            raise MalformedToolCall(f"tool call does not parse: {text[:80]!r}") from None
        try:
            name = json.loads(m.group(1))
            # Inside the single-quoted string glaive puts a backslash before
            # each apostrophe. JSON does not allow that escape.
            arguments = json.loads(m.group(2).replace("\\'", "'"))
        except json.JSONDecodeError as e:
            raise MalformedToolCall(f"arguments do not parse: {e}") from None
        payload = {"name": name, "arguments": arguments}
    calls = payload if isinstance(payload, list) else [payload]
    return format_tool_calls(calls)


def _bump(stats: Optional[Dict[str, int]], key: str):
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1


def _finish(messages: List[Dict[str, str]]) -> Optional[List[Dict[str, str]]]:
    """Return the messages, or None when no assistant turn is left."""
    if not any(m["role"] == "assistant" for m in messages):
        return None
    return messages


@dataclass
class TraceDocument:
    """A normalized multi-turn conversation."""

    messages: List[Dict[str, str]]
    source: str
    doc_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def parse_glaive_sample(
    system: str, chat: str, stats: Optional[Dict[str, int]] = None
) -> Optional[List[Dict[str, str]]]:
    """Parse a glaive-function-calling-v2 row into normalized messages.

    ``<functioncall> {...}`` turns become canonical tool calls. A malformed
    call cuts the conversation before that turn (counted in ``stats``).
    """
    messages: List[Dict[str, str]] = []

    system = (system or "").strip()
    if system.upper().startswith("SYSTEM:"):
        system = system[len("SYSTEM:"):].strip()
    if system:
        messages.append({"role": "system", "content": system})

    chat = (chat or "").replace(END_OF_TEXT, "")
    pieces = _GLAIVE_MARKER_RE.split(chat)
    # pieces = [prefix, marker, content, marker, content, ...]
    for marker, content in zip(pieces[1::2], pieces[2::2]):
        content = content.strip()
        if not content:
            continue
        if marker == "USER:":
            role = "user"
        elif marker == "ASSISTANT:":
            role = "assistant"
            if GLAIVE_CALL_MARKER in content:
                # Text before the marker is not part of the call. The
                # glaive data has none, so it is dropped.
                body = content.split(GLAIVE_CALL_MARKER, 1)[1]
                try:
                    content = parse_tool_call_text(body)
                except MalformedToolCall:
                    _bump(stats, "malformed_tool_calls")
                    break
        else:  # FUNCTION RESPONSE:
            role = "tool"
        messages.append({"role": role, "content": content})

    return _finish(messages)


def parse_sharegpt_sample(
    conversations: List[Dict[str, Any]], stats: Optional[Dict[str, int]] = None
) -> Optional[List[Dict[str, str]]]:
    """Parse a ShareGPT-style conversations list into normalized messages."""
    messages: List[Dict[str, str]] = []
    for turn in conversations or []:
        raw_role = str(turn.get("from", "")).lower()
        role = _SHAREGPT_ROLE_MAP.get(raw_role)
        content = str(turn.get("value", "")).strip()
        if role is None or not content:
            continue
        if raw_role in _SHAREGPT_CALL_ROLES:
            try:
                content = parse_tool_call_text(content)
            except MalformedToolCall:
                _bump(stats, "malformed_tool_calls")
                break
        messages.append({"role": role, "content": content})

    return _finish(messages)


def parse_messages_sample(
    messages: List[Dict[str, Any]], stats: Optional[Dict[str, int]] = None
) -> Optional[List[Dict[str, str]]]:
    """Validate/normalize an already message-shaped row.

    OpenAI-style assistant turns with ``tool_calls`` (or the legacy
    ``function_call``) become canonical tool calls. Any text content on
    such a turn is dropped, because the target must be the call alone.
    """
    out: List[Dict[str, str]] = []
    for msg in messages or []:
        role = _MESSAGE_ROLE_MAP.get(str(msg.get("role", "")).lower())
        if role is None:
            continue
        calls = msg.get("tool_calls") or (
            [msg["function_call"]] if msg.get("function_call") else None
        )
        if role == "assistant" and calls:
            try:
                content = format_tool_calls(list(calls))
            except MalformedToolCall:
                _bump(stats, "malformed_tool_calls")
                break
        else:
            content = str(msg.get("content", "") or "").strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    return _finish(out)


class TraceSourceLoader:
    """Stream and normalize conversation datasets.

    Known sources get a format automatically; other HuggingFace paths can be
    used by passing ``fmt`` explicitly.
    """

    KNOWN_SOURCES = {
        "glaiveai/glaive-function-calling-v2": {"fmt": "glaive", "split": "train"},
        "teknium/OpenHermes-2.5": {"fmt": "sharegpt", "split": "train"},
    }

    def __init__(
        self,
        source: str,
        fmt: Optional[str] = None,
        split: Optional[str] = None,
        seed: int = 42,
        buffer_size: int = 10000,
        streaming: bool = True,
    ):
        known = self.KNOWN_SOURCES.get(source, {})
        self.source = source
        self.fmt = fmt or known.get("fmt", "messages")
        self.split = split or known.get("split", "train")
        self.seed = seed
        self.buffer_size = buffer_size
        self.streaming = streaming
        self._count = 0
        self._skipped = 0
        # malformed_tool_calls: conversations cut at a malformed call
        self._parse_stats: Dict[str, int] = {"malformed_tool_calls": 0}

    def _normalize(self, item: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        stats = self._parse_stats
        if self.fmt == "glaive":
            return parse_glaive_sample(item.get("system", ""), item.get("chat", ""), stats)
        if self.fmt == "sharegpt":
            return parse_sharegpt_sample(item.get("conversations", []), stats)
        return parse_messages_sample(item.get("messages", []), stats)

    def __iter__(self) -> Iterator[TraceDocument]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        dataset = load_dataset(
            self.source, split=self.split, streaming=self.streaming
        )
        if self.streaming:
            dataset = dataset.shuffle(seed=self.seed, buffer_size=self.buffer_size)

        for idx, item in enumerate(dataset):
            messages = self._normalize(item)
            if messages is None:
                self._skipped += 1
                continue
            self._count += 1
            yield TraceDocument(
                messages=messages,
                source=self.source,
                doc_id=f"{self.source}_{idx}",
            )

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "yielded": self._count,
            "skipped": self._skipped,
            "format": self.fmt,
            **self._parse_stats,
        }
