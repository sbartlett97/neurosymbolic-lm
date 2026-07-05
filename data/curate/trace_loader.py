"""Loading and normalization of assistant-trace datasets.

Streams conversation datasets from HuggingFace and normalizes them into the
internal message format:

    [{"role": "system"|"user"|"assistant"|"tool", "content": str}, ...]

Tool/function calls are kept inline as the assistant turn's content (JSON
text), and tool results become "tool" turns — the decoder learns to emit
the call, and the symbolic heads see the result as encoder input.

Supported formats:
- glaive: glaiveai/glaive-function-calling-v2 ("system" + "chat" strings
  with USER:/ASSISTANT:/FUNCTION RESPONSE: markers)
- sharegpt: {"conversations": [{"from": ..., "value": ...}]}
- messages: already-normalized {"messages": [{"role", "content"}]}
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

END_OF_TEXT = "<|endoftext|>"

_GLAIVE_MARKER_RE = re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)")

_SHAREGPT_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "chatgpt": "assistant",
    "function-call": "assistant",
    "tool": "tool",
    "observation": "tool",
    "function-response": "tool",
}


@dataclass
class TraceDocument:
    """A normalized multi-turn conversation."""

    messages: List[Dict[str, str]]
    source: str
    doc_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def parse_glaive_sample(system: str, chat: str) -> Optional[List[Dict[str, str]]]:
    """Parse a glaive-function-calling-v2 row into normalized messages."""
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
        else:  # FUNCTION RESPONSE:
            role = "tool"
        messages.append({"role": role, "content": content})

    if not any(m["role"] == "assistant" for m in messages):
        return None
    return messages


def parse_sharegpt_sample(conversations: List[Dict[str, Any]]) -> Optional[List[Dict[str, str]]]:
    """Parse a ShareGPT-style conversations list into normalized messages."""
    messages: List[Dict[str, str]] = []
    for turn in conversations or []:
        role = _SHAREGPT_ROLE_MAP.get(str(turn.get("from", "")).lower())
        content = str(turn.get("value", "")).strip()
        if role is None or not content:
            continue
        messages.append({"role": role, "content": content})

    if not any(m["role"] == "assistant" for m in messages):
        return None
    return messages


def parse_messages_sample(messages: List[Dict[str, Any]]) -> Optional[List[Dict[str, str]]]:
    """Validate/normalize an already message-shaped row."""
    out: List[Dict[str, str]] = []
    for msg in messages or []:
        role = str(msg.get("role", "")).lower()
        content = str(msg.get("content", "") or "").strip()
        if role not in ("system", "user", "assistant", "tool") or not content:
            continue
        out.append({"role": role, "content": content})
    if not any(m["role"] == "assistant" for m in out):
        return None
    return out


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

    def _normalize(self, item: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        if self.fmt == "glaive":
            return parse_glaive_sample(item.get("system", ""), item.get("chat", ""))
        if self.fmt == "sharegpt":
            return parse_sharegpt_sample(item.get("conversations", []))
        return parse_messages_sample(item.get("messages", []))

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
        return {"yielded": self._count, "skipped": self._skipped, "format": self.fmt}
