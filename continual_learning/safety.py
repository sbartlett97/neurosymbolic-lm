"""
Safety and morality regulator for filtering harmful content.

Implements multiple layers of content filtering:
- Keyword/pattern-based blocklists
- Toxicity classification
- Semantic similarity to harmful concepts
- Ethical constraint checking
- Audit logging

This module helps prevent the model from learning harmful, biased,
or inappropriate content during continuous learning.
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
from pathlib import Path
import hashlib


class ContentCategory(Enum):
    """Categories of potentially harmful content."""
    SAFE = "safe"
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    SELF_HARM = "self_harm"
    SEXUAL = "sexual"
    MISINFORMATION = "misinformation"
    ILLEGAL = "illegal"
    PRIVACY = "privacy_violation"
    MANIPULATION = "manipulation"
    BIAS = "bias"
    UNKNOWN = "unknown"


@dataclass
class SafetyVerdict:
    """Result of safety check on content."""
    is_safe: bool
    category: ContentCategory
    confidence: float
    reasons: List[str] = field(default_factory=list)
    triggered_rules: List[str] = field(default_factory=list)
    should_log: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "is_safe": self.is_safe,
            "category": self.category.value,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "triggered_rules": self.triggered_rules
        }


class KeywordFilter:
    """
    Pattern-based content filter using keywords and regex.
    
    Provides fast initial filtering before more expensive checks.
    """
    
    def __init__(self):
        # Category-specific patterns
        self.patterns: Dict[ContentCategory, List[re.Pattern]] = {
            ContentCategory.VIOLENCE: [],
            ContentCategory.HATE_SPEECH: [],
            ContentCategory.HARASSMENT: [],
            ContentCategory.SELF_HARM: [],
            ContentCategory.ILLEGAL: [],
            ContentCategory.MANIPULATION: [],
        }
        
        # Blocklist terms (exact match, case-insensitive)
        self.blocklist: Set[str] = set()
        
        # Allowlist for overriding false positives
        self.allowlist: Set[str] = set()
        
        # Context patterns (require context to be harmful)
        self.context_patterns: List[Tuple[re.Pattern, re.Pattern]] = []
    
    def add_patterns(self, category: ContentCategory, patterns: List[str]):
        """Add regex patterns for a category."""
        for pattern in patterns:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                self.patterns[category].append(compiled)
            except re.error:
                continue
    
    def add_to_blocklist(self, terms: List[str]):
        """Add terms to blocklist."""
        self.blocklist.update(t.lower() for t in terms)
    
    def add_to_allowlist(self, terms: List[str]):
        """Add terms to allowlist (override blocklist)."""
        self.allowlist.update(t.lower() for t in terms)
    
    def check(self, text: str) -> Tuple[bool, ContentCategory, List[str]]:
        """
        Check text against keyword filters.
        
        Returns:
            (is_safe, category, triggered_rules)
        """
        text_lower = text.lower()
        triggered = []
        
        # Check allowlist first
        for term in self.allowlist:
            if term in text_lower:
                return True, ContentCategory.SAFE, []
        
        # Check blocklist
        for term in self.blocklist:
            if term in text_lower:
                triggered.append(f"blocklist:{term}")
        
        if triggered:
            return False, ContentCategory.UNKNOWN, triggered
        
        # Check category patterns
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    triggered.append(f"pattern:{category.value}:{pattern.pattern[:30]}")
                    return False, category, triggered
        
        return True, ContentCategory.SAFE, []


class SemanticSafetyChecker:
    """
    Semantic similarity-based safety checking.
    
    Uses embeddings to detect content similar to known harmful examples.
    """
    
    def __init__(
        self,
        embedding_model: Optional[nn.Module] = None,
        similarity_threshold: float = 0.7
    ):
        """
        Args:
            embedding_model: Model to generate text embeddings
            similarity_threshold: Threshold for flagging similar content
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        
        # Store embeddings of known harmful content
        self.harmful_embeddings: Dict[ContentCategory, List[torch.Tensor]] = {
            cat: [] for cat in ContentCategory
        }
        
        # Store harmful example descriptions (for interpretability)
        self.harmful_examples: Dict[ContentCategory, List[str]] = {
            cat: [] for cat in ContentCategory
        }
    
    def add_harmful_example(
        self,
        text: str,
        category: ContentCategory,
        embedding: Optional[torch.Tensor] = None
    ):
        """Add a known harmful example."""
        self.harmful_examples[category].append(text[:100])  # Store truncated
        
        if embedding is not None:
            self.harmful_embeddings[category].append(embedding.detach().cpu())
        elif self.embedding_model is not None:
            # Generate embedding
            with torch.no_grad():
                emb = self._get_embedding(text)
                self.harmful_embeddings[category].append(emb.cpu())
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for text using the embedding model."""
        if self.embedding_model is None:
            return torch.zeros(768)  # Placeholder
        
        # Implementation depends on embedding model type
        # This is a placeholder - would need tokenizer in practice
        return torch.randn(768)
    
    def check(
        self,
        text: str,
        embedding: Optional[torch.Tensor] = None
    ) -> Tuple[bool, ContentCategory, float, str]:
        """
        Check text against known harmful examples.
        
        Returns:
            (is_safe, category, similarity_score, most_similar_example)
        """
        if embedding is None:
            embedding = self._get_embedding(text)
        
        max_similarity = 0.0
        max_category = ContentCategory.SAFE
        most_similar = ""
        
        for category, embeddings in self.harmful_embeddings.items():
            for i, harmful_emb in enumerate(embeddings):
                similarity = F.cosine_similarity(
                    embedding.unsqueeze(0),
                    harmful_emb.unsqueeze(0)
                ).item()
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_category = category
                    if i < len(self.harmful_examples[category]):
                        most_similar = self.harmful_examples[category][i]
        
        is_safe = max_similarity < self.similarity_threshold
        
        return is_safe, max_category, max_similarity, most_similar


class EthicalConstraintChecker:
    """
    Rule-based ethical constraint checking.
    
    Enforces explicit ethical rules and principles.
    """
    
    def __init__(self):
        self.constraints: List[Dict[str, Any]] = []
        self._initialize_default_constraints()
    
    def _initialize_default_constraints(self):
        """Initialize default ethical constraints."""
        self.constraints = [
            {
                "name": "no_personal_attacks",
                "description": "Content should not attack individuals personally",
                "check": lambda text, entities: not any(
                    self._is_person(e) and self._has_negative_context(text, e)
                    for e in entities
                ),
                "category": ContentCategory.HARASSMENT
            },
            {
                "name": "no_group_denigration",
                "description": "Content should not denigrate groups of people",
                "check": lambda text, entities: not self._has_group_denigration(text),
                "category": ContentCategory.HATE_SPEECH
            },
            {
                "name": "no_deceptive_claims",
                "description": "Content should not make deceptive factual claims",
                "check": lambda text, entities: not self._has_deceptive_markers(text),
                "category": ContentCategory.MISINFORMATION
            },
            {
                "name": "respect_privacy",
                "description": "Content should not expose private information",
                "check": lambda text, entities: not self._has_privacy_violation(text),
                "category": ContentCategory.PRIVACY
            },
        ]
    
    def _is_person(self, entity: str) -> bool:
        """Check if entity is likely a person."""
        # Simple heuristic - would need NER in practice
        person_indicators = ["mr", "mrs", "ms", "dr", "prof"]
        return any(ind in entity.lower() for ind in person_indicators)
    
    def _has_negative_context(self, text: str, entity: str) -> bool:
        """Check if entity appears in negative context."""
        negative_words = ["stupid", "idiot", "fool", "hate", "terrible", "worst"]
        text_lower = text.lower()
        entity_lower = entity.lower()
        
        if entity_lower not in text_lower:
            return False
        
        # Check for negative words near entity
        entity_pos = text_lower.find(entity_lower)
        context_window = text_lower[max(0, entity_pos-50):entity_pos+len(entity)+50]
        
        return any(neg in context_window for neg in negative_words)
    
    def _has_group_denigration(self, text: str) -> bool:
        """Check for denigration of groups."""
        group_terms = ["all", "every", "those people", "they always", "they never"]
        negative_terms = ["stupid", "lazy", "criminal", "dangerous", "inferior"]
        
        text_lower = text.lower()
        has_group = any(term in text_lower for term in group_terms)
        has_negative = any(term in text_lower for term in negative_terms)
        
        return has_group and has_negative
    
    def _has_deceptive_markers(self, text: str) -> bool:
        """Check for markers of deceptive content."""
        deceptive_patterns = [
            r"(?:doctors|scientists|experts)\s+(?:don't want|hate|are hiding)",
            r"(?:secret|hidden)\s+(?:truth|cure|knowledge)",
            r"(?:100%|guaranteed|proven)\s+(?:cure|solution|fix)",
        ]
        
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in deceptive_patterns)
    
    def _has_privacy_violation(self, text: str) -> bool:
        """Check for potential privacy violations."""
        # Patterns for sensitive information
        patterns = [
            r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',  # SSN-like
            r'\b\d{16}\b',  # Credit card-like
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
        ]
        
        return any(re.search(p, text) for p in patterns)
    
    def add_constraint(
        self,
        name: str,
        description: str,
        check_fn,
        category: ContentCategory
    ):
        """Add a custom ethical constraint."""
        self.constraints.append({
            "name": name,
            "description": description,
            "check": check_fn,
            "category": category
        })
    
    def check(
        self,
        text: str,
        entities: Optional[List[str]] = None
    ) -> Tuple[bool, List[str], ContentCategory]:
        """
        Check text against ethical constraints.
        
        Returns:
            (passes_all, violated_constraints, most_severe_category)
        """
        entities = entities or []
        violations = []
        categories = []
        
        for constraint in self.constraints:
            try:
                passes = constraint["check"](text, entities)
                if not passes:
                    violations.append(constraint["name"])
                    categories.append(constraint["category"])
            except Exception:
                continue
        
        if violations:
            # Return most severe category
            severity_order = [
                ContentCategory.ILLEGAL,
                ContentCategory.VIOLENCE,
                ContentCategory.HATE_SPEECH,
                ContentCategory.HARASSMENT,
                ContentCategory.SELF_HARM,
                ContentCategory.MISINFORMATION,
                ContentCategory.MANIPULATION,
                ContentCategory.PRIVACY,
                ContentCategory.BIAS,
            ]
            
            for cat in severity_order:
                if cat in categories:
                    return False, violations, cat
            
            return False, violations, categories[0]
        
        return True, [], ContentCategory.SAFE


class SafetyAuditLog:
    """
    Audit logging for safety decisions.
    
    Records all safety checks for review and improvement.
    """
    
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = Path(log_path) if log_path else None
        self.entries: List[Dict] = []
        
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log(
        self,
        text: str,
        verdict: SafetyVerdict,
        source: str = "unknown",
        metadata: Optional[Dict] = None
    ):
        """Log a safety check result."""
        # Hash text for privacy (don't store raw harmful content)
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "text_hash": text_hash,
            "text_preview": text[:50] + "..." if len(text) > 50 else text,
            "verdict": verdict.to_dict(),
            "source": source,
            "metadata": metadata or {}
        }
        
        self.entries.append(entry)
        
        # Write to file if path specified
        if self.log_path and not verdict.is_safe:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about safety checks."""
        if not self.entries:
            return {"total": 0}
        
        total = len(self.entries)
        safe = sum(1 for e in self.entries if e["verdict"]["is_safe"])
        
        category_counts = {}
        for entry in self.entries:
            cat = entry["verdict"]["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            "total": total,
            "safe": safe,
            "unsafe": total - safe,
            "safe_rate": safe / total if total > 0 else 0,
            "by_category": category_counts
        }


class SafetyRegulator:
    """
    Main safety regulator combining all filtering methods.
    
    Provides a unified interface for content safety checking
    with configurable strictness levels.
    """
    
    def __init__(
        self,
        strictness: str = "medium",
        enable_keyword_filter: bool = True,
        enable_semantic_filter: bool = True,
        enable_ethical_filter: bool = True,
        enable_logging: bool = True,
        log_path: Optional[str] = "safety_audit.jsonl"
    ):
        """
        Args:
            strictness: Safety strictness level ('low', 'medium', 'high', 'maximum')
            enable_keyword_filter: Use keyword-based filtering
            enable_semantic_filter: Use semantic similarity filtering
            enable_ethical_filter: Use rule-based ethical checking
            enable_logging: Enable audit logging
            log_path: Path for audit log
        """
        self.strictness = strictness
        
        # Initialize components
        self.keyword_filter = KeywordFilter() if enable_keyword_filter else None
        self.semantic_checker = SemanticSafetyChecker() if enable_semantic_filter else None
        self.ethical_checker = EthicalConstraintChecker() if enable_ethical_filter else None
        self.audit_log = SafetyAuditLog(log_path) if enable_logging else None
        
        # Configure based on strictness
        self._configure_strictness()
    
    def _configure_strictness(self):
        """Configure thresholds based on strictness level."""
        thresholds = {
            "low": {"semantic": 0.85, "require_all": False},
            "medium": {"semantic": 0.7, "require_all": False},
            "high": {"semantic": 0.6, "require_all": True},
            "maximum": {"semantic": 0.5, "require_all": True}
        }
        
        config = thresholds.get(self.strictness, thresholds["medium"])
        
        if self.semantic_checker:
            self.semantic_checker.similarity_threshold = config["semantic"]
        
        self.require_all_pass = config["require_all"]
    
    def check(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        embedding: Optional[torch.Tensor] = None,
        source: str = "unknown"
    ) -> SafetyVerdict:
        """
        Perform comprehensive safety check.
        
        Args:
            text: Text to check
            entities: Optional list of entities in text
            embedding: Optional pre-computed embedding
            source: Source identifier for logging
        
        Returns:
            SafetyVerdict with detailed results
        """
        reasons = []
        triggered_rules = []
        is_safe = True
        category = ContentCategory.SAFE
        confidences = []
        
        # Keyword filter (fast, first check)
        if self.keyword_filter:
            kw_safe, kw_cat, kw_rules = self.keyword_filter.check(text)
            if not kw_safe:
                is_safe = False
                category = kw_cat
                triggered_rules.extend(kw_rules)
                reasons.append("Failed keyword filter")
                confidences.append(0.9)
        
        # Semantic similarity check
        if self.semantic_checker and (is_safe or self.require_all_pass):
            sem_safe, sem_cat, similarity, similar_example = self.semantic_checker.check(
                text, embedding
            )
            if not sem_safe:
                is_safe = False
                category = sem_cat
                triggered_rules.append(f"semantic_similarity:{similarity:.2f}")
                reasons.append(f"Similar to known harmful content (similarity: {similarity:.2f})")
                confidences.append(similarity)
        
        # Ethical constraint check
        if self.ethical_checker and (is_safe or self.require_all_pass):
            eth_safe, violations, eth_cat = self.ethical_checker.check(text, entities)
            if not eth_safe:
                is_safe = False
                category = eth_cat
                triggered_rules.extend([f"ethical:{v}" for v in violations])
                reasons.append(f"Violated ethical constraints: {', '.join(violations)}")
                confidences.append(0.8)
        
        # Compute overall confidence
        confidence = max(confidences) if confidences else 1.0 if is_safe else 0.0
        
        verdict = SafetyVerdict(
            is_safe=is_safe,
            category=category,
            confidence=confidence,
            reasons=reasons,
            triggered_rules=triggered_rules
        )
        
        # Log the check
        if self.audit_log:
            self.audit_log.log(text, verdict, source)
        
        return verdict
    
    def check_batch(
        self,
        samples: List[Dict],
        source: str = "batch"
    ) -> List[Tuple[Dict, SafetyVerdict]]:
        """
        Check a batch of samples.
        
        Returns:
            List of (sample, verdict) tuples
        """
        results = []
        
        for sample in samples:
            text = sample.get("text", "")
            entities = sample.get("entities", [])
            
            verdict = self.check(text, entities, source=source)
            results.append((sample, verdict))
        
        return results
    
    def filter_safe(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter samples into safe and unsafe.
        
        Returns:
            (safe_samples, unsafe_samples)
        """
        safe = []
        unsafe = []
        
        for sample in samples:
            verdict = self.check(
                sample.get("text", ""),
                sample.get("entities", []),
                source="filter"
            )
            
            if verdict.is_safe:
                safe.append(sample)
            else:
                unsafe.append(sample)
        
        return safe, unsafe
    
    def add_harmful_pattern(
        self,
        pattern: str,
        category: ContentCategory
    ):
        """Add a harmful pattern to detect."""
        if self.keyword_filter:
            self.keyword_filter.add_patterns(category, [pattern])
    
    def add_harmful_example(
        self,
        text: str,
        category: ContentCategory,
        embedding: Optional[torch.Tensor] = None
    ):
        """Add a known harmful example for semantic matching."""
        if self.semantic_checker:
            self.semantic_checker.add_harmful_example(text, category, embedding)
    
    def add_blocklist_terms(self, terms: List[str]):
        """Add terms to the blocklist."""
        if self.keyword_filter:
            self.keyword_filter.add_to_blocklist(terms)
    
    def add_allowlist_terms(self, terms: List[str]):
        """Add terms to the allowlist (override blocklist)."""
        if self.keyword_filter:
            self.keyword_filter.add_to_allowlist(terms)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get safety check statistics."""
        if self.audit_log:
            return self.audit_log.get_statistics()
        return {}
    
    def save_config(self, path: str):
        """Save safety configuration."""
        config = {
            "strictness": self.strictness,
            "blocklist": list(self.keyword_filter.blocklist) if self.keyword_filter else [],
            "allowlist": list(self.keyword_filter.allowlist) if self.keyword_filter else [],
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, path: str):
        """Load safety configuration."""
        with open(path, 'r') as f:
            config = json.load(f)
        
        self.strictness = config.get("strictness", "medium")
        self._configure_strictness()
        
        if self.keyword_filter:
            self.keyword_filter.blocklist = set(config.get("blocklist", []))
            self.keyword_filter.allowlist = set(config.get("allowlist", []))
