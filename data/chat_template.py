"""Chat template formatting for T5-based chat fine-tuning.

Defines special tokens for role markers and provides formatting utilities
for single-turn and multi-turn conversations.

T5 doesn't have native chat templates, so we add <system>, <user>,
<assistant>, and <tool> as special tokens and format conversations as flat
text with role markers. The <tool> role carries tool/function results in
assistant traces.
"""

from typing import List, Dict, Optional, Tuple

# Special tokens for chat role markers
CHAT_SPECIAL_TOKENS = ["<system>", "<user>", "<assistant>", "<tool>"]


class ChatTemplate:
    """Formats conversations with role markers for T5 chat fine-tuning.

    Input format:
        <system> You are helpful. <user> What is X? <assistant>
    Multi-turn:
        <system> ... <user> msg1 <assistant> resp1 <user> msg2 <assistant>

    The last assistant turn becomes the decoder target. Everything before
    (including the final <assistant> marker) is the encoder input.
    """

    def __init__(self, default_system_prompt: str = "You are a helpful assistant."):
        self.default_system_prompt = default_system_prompt

    @staticmethod
    def add_special_tokens(tokenizer) -> int:
        """Add chat role tokens to a tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer to extend.

        Returns:
            Number of tokens added.
        """
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": CHAT_SPECIAL_TOKENS}
        )
        return added

    def format_messages(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[str, str]:
        """Format a multi-turn conversation into (input_text, target_text).

        Args:
            messages: List of dicts with "role" and "content" keys.
                      Roles: "system", "user", "assistant", "tool".

        Returns:
            (input_text, target_text) where target_text is the last
            assistant message (or empty string if none).
        """
        input_text, target_text, _ = self.format_messages_with_offsets(messages)
        return input_text, target_text

    def format_messages_with_offsets(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[str, str, List[Optional[int]]]:
        """Format a conversation and report where each message's content
        landed in the encoder input.

        Message content is stripped before formatting, so annotations must
        be relative to ``msg["content"].strip()``.

        Returns:
            (input_text, target_text, content_offsets) where
            content_offsets[i] is the character offset of messages[i]'s
            stripped content within input_text, or None if that content is
            not part of the encoder input (the final assistant turn, which
            becomes the decoder target, or an empty message).
        """
        parts: List[str] = []          # formatted chunks joined by " "
        owners: List[Optional[int]] = []   # message index owning each chunk
        prefixes: List[int] = []       # chars before content within chunk
        target_text = ""
        system_seen = False

        def add_part(text: str, owner: Optional[int], prefix: int):
            parts.append(text)
            owners.append(owner)
            prefixes.append(prefix)

        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"].strip()

            if role == "system":
                add_part(f"<system> {content}", i, len("<system> "))
                system_seen = True
            elif role in ("user", "tool"):
                if not system_seen:
                    add_part(f"<system> {self.default_system_prompt}", None, 0)
                    system_seen = True
                marker = f"<{role}>"
                add_part(f"{marker} {content}", i, len(marker) + 1)
            elif role == "assistant":
                is_last_assistant = all(
                    m["role"] != "assistant" for m in messages[i + 1:]
                )
                if is_last_assistant:
                    # Last assistant turn is the target
                    add_part("<assistant>", None, 0)
                    target_text = content
                else:
                    add_part(f"<assistant> {content}", i, len("<assistant> "))

        if not system_seen:
            parts.insert(0, f"<system> {self.default_system_prompt}")
            owners.insert(0, None)
            prefixes.insert(0, 0)

        input_text = " ".join(parts)
        # Ensure the input ends with <assistant> marker (appending never
        # shifts earlier offsets)
        if not input_text.rstrip().endswith("<assistant>"):
            input_text = input_text.rstrip() + " <assistant>"

        # Walk the joined parts to compute each message's content offset
        content_offsets: List[Optional[int]] = [None] * len(messages)
        pos = 0
        for part, owner, prefix in zip(parts, owners, prefixes):
            if owner is not None and messages[owner]["content"].strip():
                content_offsets[owner] = pos + prefix
            pos += len(part) + 1  # +1 for the " " join separator

        return input_text, target_text, content_offsets

    def format_single_turn(
        self,
        user_message: str,
        response: str = "",
        system_message: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Convenience wrapper for single-turn formatting.

        Args:
            user_message: The user's message.
            response: The assistant's response (decoder target).
            system_message: Optional system prompt override.

        Returns:
            (input_text, target_text)
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        if response:
            messages.append({"role": "assistant", "content": response})

        return self.format_messages(messages)
