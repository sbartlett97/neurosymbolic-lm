"""Document filtering for quality control."""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .source_loader import DocumentSource


@dataclass
class FilterResult:
    """Result of document filtering."""

    passed: bool
    reason: Optional[str] = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentFilter:
    """Filter documents based on quality criteria.

    Filters:
    - Length constraints
    - Language detection (English)
    - Content quality (not too repetitive, not mostly boilerplate)
    - Encoding issues
    - Adult/harmful content (basic)
    """

    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 4000,
        min_words: int = 20,
        max_repetition_ratio: float = 0.3,
        min_alpha_ratio: float = 0.7,
        max_line_noise_ratio: float = 0.3,
        block_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize document filter.

        Args:
            min_length: Minimum document length in characters
            max_length: Maximum document length in characters
            min_words: Minimum word count
            max_repetition_ratio: Maximum ratio of repeated lines
            min_alpha_ratio: Minimum ratio of alphabetic characters
            max_line_noise_ratio: Maximum ratio of short/noisy lines
            block_patterns: Additional regex patterns to block
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        self.max_repetition_ratio = max_repetition_ratio
        self.min_alpha_ratio = min_alpha_ratio
        self.max_line_noise_ratio = max_line_noise_ratio

        # Compile block patterns
        self.block_patterns = []
        if block_patterns:
            for pattern in block_patterns:
                try:
                    self.block_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error:
                    continue

        # Default patterns to block
        self._default_block_patterns = [
            re.compile(r"javascript:", re.IGNORECASE),
            re.compile(r"<script", re.IGNORECASE),
            re.compile(r"cookie.*policy", re.IGNORECASE),
            re.compile(r"terms.*service", re.IGNORECASE),
            re.compile(r"privacy.*policy", re.IGNORECASE),
            re.compile(r"subscribe.*newsletter", re.IGNORECASE),
            re.compile(r"click\s+here\s+to", re.IGNORECASE),
            re.compile(r"error\s*404", re.IGNORECASE),
            re.compile(r"page\s+not\s+found", re.IGNORECASE),
        ]

        # Statistics
        self._stats = {
            "total": 0,
            "passed": 0,
            "failed_length": 0,
            "failed_words": 0,
            "failed_repetition": 0,
            "failed_alpha": 0,
            "failed_noise": 0,
            "failed_pattern": 0,
            "failed_encoding": 0,
        }

    def filter(self, doc: DocumentSource) -> FilterResult:
        """
        Filter a single document.

        Args:
            doc: Document to filter

        Returns:
            FilterResult with pass/fail status and reason
        """
        self._stats["total"] += 1
        text = doc.text

        # Length check
        if len(text) < self.min_length:
            self._stats["failed_length"] += 1
            return FilterResult(False, f"Too short: {len(text)} < {self.min_length}")

        if len(text) > self.max_length:
            # Truncate instead of rejecting
            text = text[: self.max_length]

        # Word count check
        words = text.split()
        if len(words) < self.min_words:
            self._stats["failed_words"] += 1
            return FilterResult(False, f"Too few words: {len(words)} < {self.min_words}")

        # Encoding check
        try:
            text.encode("utf-8").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            self._stats["failed_encoding"] += 1
            return FilterResult(False, "Encoding issues")

        # Alpha ratio check
        alpha_chars = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_chars / len(text) if text else 0
        if alpha_ratio < self.min_alpha_ratio:
            self._stats["failed_alpha"] += 1
            return FilterResult(
                False, f"Low alpha ratio: {alpha_ratio:.2f} < {self.min_alpha_ratio}"
            )

        # Repetition check
        lines = text.split("\n")
        if len(lines) > 1:
            unique_lines = set(line.strip().lower() for line in lines if line.strip())
            repetition_ratio = 1 - len(unique_lines) / len(
                [l for l in lines if l.strip()]
            )
            if repetition_ratio > self.max_repetition_ratio:
                self._stats["failed_repetition"] += 1
                return FilterResult(
                    False,
                    f"High repetition: {repetition_ratio:.2f} > {self.max_repetition_ratio}",
                )

        # Line noise check (many short lines often indicates boilerplate)
        non_empty_lines = [l for l in lines if l.strip()]
        if non_empty_lines:
            short_lines = sum(1 for l in non_empty_lines if len(l.strip()) < 30)
            noise_ratio = short_lines / len(non_empty_lines)
            if noise_ratio > self.max_line_noise_ratio:
                self._stats["failed_noise"] += 1
                return FilterResult(
                    False,
                    f"High noise ratio: {noise_ratio:.2f} > {self.max_line_noise_ratio}",
                )

        # Block pattern check
        all_patterns = self._default_block_patterns + self.block_patterns
        for pattern in all_patterns:
            if pattern.search(text):
                self._stats["failed_pattern"] += 1
                return FilterResult(False, f"Blocked pattern: {pattern.pattern[:30]}")

        self._stats["passed"] += 1
        return FilterResult(
            True,
            metadata={
                "length": len(text),
                "words": len(words),
                "alpha_ratio": alpha_ratio,
            },
        )

    def filter_batch(
        self, docs: List[DocumentSource]
    ) -> Tuple[List[DocumentSource], List[Tuple[DocumentSource, str]]]:
        """
        Filter a batch of documents.

        Args:
            docs: List of documents to filter

        Returns:
            (passed_docs, failed_docs_with_reasons)
        """
        passed = []
        failed = []

        for doc in docs:
            result = self.filter(doc)
            if result.passed:
                passed.append(doc)
            else:
                failed.append((doc, result.reason))

        return passed, failed

    def get_statistics(self) -> dict:
        """Get filtering statistics."""
        total = self._stats["total"]
        return {
            **self._stats,
            "pass_rate": self._stats["passed"] / total if total > 0 else 0,
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        for key in self._stats:
            self._stats[key] = 0


class ContentCleaner:
    """Clean document content before annotation."""

    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_whitespace: bool = True,
        remove_html_tags: bool = True,
        max_consecutive_newlines: int = 2,
    ):
        """
        Initialize content cleaner.

        Args:
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses
            normalize_whitespace: Normalize whitespace
            remove_html_tags: Remove HTML tags
            max_consecutive_newlines: Max consecutive newlines to keep
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.remove_html_tags = remove_html_tags
        self.max_consecutive_newlines = max_consecutive_newlines

        # Compile patterns
        self._url_pattern = re.compile(
            r"https?://[^\s<>\"']+|www\.[^\s<>\"']+"
        )
        self._email_pattern = re.compile(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        )
        self._html_pattern = re.compile(r"<[^>]+>")
        self._whitespace_pattern = re.compile(r"[ \t]+")
        self._newline_pattern = re.compile(r"\n{3,}")

    def clean(self, text: str) -> str:
        """
        Clean document text.

        Args:
            text: Raw document text

        Returns:
            Cleaned text
        """
        # Remove HTML tags
        if self.remove_html_tags:
            text = self._html_pattern.sub(" ", text)

        # Remove URLs
        if self.remove_urls:
            text = self._url_pattern.sub("[URL]", text)

        # Remove emails
        if self.remove_emails:
            text = self._email_pattern.sub("[EMAIL]", text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = self._whitespace_pattern.sub(" ", text)
            text = self._newline_pattern.sub(
                "\n" * self.max_consecutive_newlines, text
            )

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts."""
        return [self.clean(text) for text in texts]
