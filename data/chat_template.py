"""Chat template formatting for T5-based chat fine-tuning.

Defines special tokens for role markers and provides formatting utilities
for single-turn and multi-turn conversations.

T5 doesn't have native chat templates, so we add <system>, <user>, <assistant>
as special tokens and format conversations as flat text with role markers.
"""

from typing import List, Dict, Tuple, Optional

# Special tokens for chat role markers
CHAT_SPECIAL_TOKENS = ["<system>", "<user>", "<assistant>"]


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
                      Roles: "system", "user", "assistant".

        Returns:
            (input_text, target_text) where target_text is the last
            assistant message (or empty string if none).
        """
        parts = []
        target_text = ""
        system_seen = False

        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"].strip()

            if role == "system":
                parts.append(f"<system> {content}")
                system_seen = True
            elif role == "user":
                if not system_seen:
                    parts.append(f"<system> {self.default_system_prompt}")
                    system_seen = True
                parts.append(f"<user> {content}")
            elif role == "assistant":
                is_last_assistant = all(
                    m["role"] != "assistant" for m in messages[i + 1:]
                )
                if is_last_assistant:
                    # Last assistant turn is the target
                    parts.append("<assistant>")
                    target_text = content
                else:
                    parts.append(f"<assistant> {content}")

        if not system_seen:
            parts.insert(0, f"<system> {self.default_system_prompt}")

        # Ensure the input ends with <assistant> marker
        input_text = " ".join(parts)
        if not input_text.rstrip().endswith("<assistant>"):
            input_text = input_text.rstrip() + " <assistant>"

        return input_text, target_text

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
