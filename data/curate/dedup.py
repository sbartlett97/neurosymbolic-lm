"""Text hashing for deduplication and deterministic train/val splits.

The hash is a stable digest of normalized text, so the same document gets
the same hash in every process and every run. A run uses it twice:

- Deduplication: a document whose hash was already seen is skipped.
- Split assignment: the hash value decides train or validation, so a
  document always lands in the same split for a given ``val_ratio``.
"""

import hashlib
import re

_WHITESPACE = re.compile(r"\s+")

# Hex digits kept from the digest (64 bits)
HASH_HEX_CHARS = 16


def normalize_text(text: str) -> str:
    """Lowercase the text and collapse all whitespace runs to one space."""
    return _WHITESPACE.sub(" ", text).strip().lower()


def text_hash(text: str) -> str:
    """Stable hex digest of the normalized text."""
    digest = hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()
    return digest[:HASH_HEX_CHARS]


def is_validation(hash_hex: str, val_ratio: float) -> bool:
    """Assign a hash to the validation split with probability val_ratio."""
    if val_ratio <= 0:
        return False
    bucket = int(hash_hex[:8], 16) / 0x100000000
    return bucket < val_ratio
