"""Convert instruction-tuning datasets to chat JSONL format.

Supports Dolly, Alpaca, and OASST1 multi-turn conversations. Each output
sample has a ``messages`` list with role/content dicts suitable for
ChatTemplate formatting.

Usage (CLI):
    python data/chat_converter.py --source dolly --output data/chat_dolly.jsonl
    python data/chat_converter.py --source alpaca --output data/chat_alpaca.jsonl
    python data/chat_converter.py --source oasst1 --output data/chat_oasst1.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable assistant. Answer clearly and accurately."
)


class ChatConverter:
    """Convert various datasets into chat-message JSONL format."""

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Dolly
    # ------------------------------------------------------------------
    def convert_dolly_to_chat(
        self,
        max_samples: Optional[int] = None,
        cache_dir: str = "data/cache",
    ) -> List[Dict]:
        """Load Dolly from HuggingFace and wrap as chat messages."""
        from datasets import load_dataset

        ds = load_dataset(
            "databricks/databricks-dolly-15k", cache_dir=cache_dir
        )
        samples = []
        for item in ds["train"]:
            instruction = item.get("instruction", "").strip()
            response = item.get("response", "").strip()
            context = item.get("context", "").strip()
            if not instruction or not response:
                continue

            user_content = instruction
            if context:
                user_content = f"{instruction}\n\nContext: {context}"

            sample = self._build_sample(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": response},
                ],
                source_text=user_content,
            )
            samples.append(sample)
            if max_samples and len(samples) >= max_samples:
                break
        return samples

    # ------------------------------------------------------------------
    # Alpaca
    # ------------------------------------------------------------------
    def convert_alpaca_to_chat(
        self,
        max_samples: Optional[int] = None,
        cache_dir: str = "data/cache",
    ) -> List[Dict]:
        """Load Alpaca from HuggingFace and wrap as chat messages."""
        from datasets import load_dataset

        ds = load_dataset("tatsu-lab/alpaca", cache_dir=cache_dir)
        samples = []
        for item in ds["train"]:
            instruction = item.get("instruction", "").strip()
            output = item.get("output", "").strip()
            input_text = item.get("input", "").strip()
            if not instruction or not output:
                continue

            user_content = instruction
            if input_text:
                user_content = f"{instruction}\n\nInput: {input_text}"

            sample = self._build_sample(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output},
                ],
                source_text=user_content,
            )
            samples.append(sample)
            if max_samples and len(samples) >= max_samples:
                break
        return samples

    # ------------------------------------------------------------------
    # OASST1 multi-turn
    # ------------------------------------------------------------------
    def convert_oasst1_multiturn(
        self,
        max_samples: Optional[int] = None,
        cache_dir: str = "data/cache",
    ) -> List[Dict]:
        """Load OASST1 and rebuild multi-turn conversation threads."""
        from datasets import load_dataset

        ds = load_dataset(
            "OpenAssistant/oasst1", cache_dir=cache_dir
        )

        # Build parent->children map and index by message_id
        msg_by_id: Dict[str, dict] = {}
        children: Dict[str, List[str]] = {}
        roots: List[str] = []

        for split in ds:
            for item in ds[split]:
                mid = item["message_id"]
                pid = item.get("parent_id")
                msg_by_id[mid] = item
                if pid is None:
                    roots.append(mid)
                else:
                    children.setdefault(pid, []).append(mid)

        # Walk each thread — pick the highest-ranked child at each step
        samples = []
        for root_id in roots:
            thread = self._walk_thread(root_id, msg_by_id, children)
            if len(thread) < 2:
                continue  # Need at least user + assistant

            messages = [{"role": "system", "content": self.system_prompt}]
            for msg in thread:
                role = "user" if msg["role"] == "prompter" else "assistant"
                messages.append({"role": role, "content": msg["text"].strip()})

            # Ensure last message is from assistant
            if messages[-1]["role"] != "assistant":
                messages.pop()
            if len(messages) < 3:  # system + user + assistant
                continue

            user_texts = " ".join(
                m["content"] for m in messages if m["role"] == "user"
            )
            sample = self._build_sample(
                messages=messages, source_text=user_texts
            )
            samples.append(sample)
            if max_samples and len(samples) >= max_samples:
                break

        return samples

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _walk_thread(
        root_id: str,
        msg_by_id: Dict[str, dict],
        children: Dict[str, List[str]],
    ) -> List[dict]:
        """Greedily walk a conversation tree picking the highest-ranked child."""
        thread = []
        current = root_id
        while current is not None:
            msg = msg_by_id.get(current)
            if msg is None:
                break
            thread.append(msg)
            kids = children.get(current, [])
            if not kids:
                break
            # Pick child with highest rank (lowest rank number)
            kids_sorted = sorted(
                kids,
                key=lambda c: msg_by_id.get(c, {}).get("rank", 999),
            )
            current = kids_sorted[0]
        return thread

    @staticmethod
    def _extract_simple_entities(text: str) -> List[str]:
        """Cheap entity extraction from text for chat samples.

        Uses simple capitalized-word heuristic. Full NER can be done
        offline via the annotation pipeline.
        """
        words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        # Deduplicate preserving order
        seen = set()
        entities = []
        for w in words:
            if w not in seen:
                seen.add(w)
                entities.append(w)
        return entities[:20]

    @classmethod
    def _build_sample(
        cls,
        messages: List[Dict[str, str]],
        source_text: str,
    ) -> Dict:
        """Build a training-ready sample dict with messages and basic entities."""
        entities = cls._extract_simple_entities(source_text)
        return {
            "text": source_text,
            "entities": entities,
            "concepts": [["concept"] for _ in entities],
            "relations": [],
            "should_respond": 1,
            "response": messages[-1]["content"] if messages[-1]["role"] == "assistant" else "",
            "response_reason": "question",
            "messages": messages,
        }


def write_jsonl(samples: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {len(samples)} samples to {path}")


def main():
    parser = argparse.ArgumentParser(description="Convert datasets to chat JSONL")
    parser.add_argument(
        "--source",
        choices=["dolly", "alpaca", "oasst1"],
        required=True,
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    args = parser.parse_args()

    converter = ChatConverter(system_prompt=args.system_prompt)

    if args.source == "dolly":
        samples = converter.convert_dolly_to_chat(args.max_samples, args.cache_dir)
    elif args.source == "alpaca":
        samples = converter.convert_alpaca_to_chat(args.max_samples, args.cache_dir)
    elif args.source == "oasst1":
        samples = converter.convert_oasst1_multiturn(args.max_samples, args.cache_dir)

    write_jsonl(samples, Path(args.output))


if __name__ == "__main__":
    main()
