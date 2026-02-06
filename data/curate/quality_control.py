"""Quality control for annotated samples."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .llm_annotator import AnnotationResult


@dataclass
class QCResult:
    """Result of quality control check."""

    passed: bool
    issues: List[str] = field(default_factory=list)
    fixed_result: Optional[AnnotationResult] = None
    stats: Dict[str, Any] = field(default_factory=dict)


class QualityControl:
    """Quality control for LLM annotations.

    Validates and fixes:
    - Entity span accuracy
    - Concept validity
    - Relation index bounds
    - Content safety
    """

    def __init__(
        self,
        min_valid_span_ratio: float = 0.5,
        max_span_tolerance: int = 10,
        valid_concepts: Optional[List[str]] = None,
        valid_relations: Optional[List[str]] = None,
        safety_regulator=None,
    ):
        """
        Initialize quality control.

        Args:
            min_valid_span_ratio: Minimum ratio of valid entity spans
            max_span_tolerance: Maximum character tolerance for span matching
            valid_concepts: List of valid concept names (None = accept all)
            valid_relations: List of valid relation types (None = accept all)
            safety_regulator: Optional SafetyRegulator for content filtering
        """
        self.min_valid_span_ratio = min_valid_span_ratio
        self.max_span_tolerance = max_span_tolerance
        self.valid_concepts = set(valid_concepts) if valid_concepts else None
        self.valid_relations = set(valid_relations) if valid_relations else None
        self.safety_regulator = safety_regulator

        # Default concepts if none provided
        if self.valid_concepts is None:
            self.valid_concepts = {
                "person", "organization", "location", "date", "time",
                "quantity", "object", "event", "concept",
                "scientist", "politician", "artist", "athlete", "writer",
                "actor", "musician", "leader", "company", "government",
                "university", "non_profit", "sports_team", "band",
                "city", "country", "region", "continent", "building",
                "landmark", "address", "year", "month", "day", "period",
                "era", "money", "percentage", "distance", "weight",
                "count", "age", "product", "vehicle", "document",
                "artwork", "food", "weapon", "tool", "war", "election",
                "disaster", "ceremony", "meeting", "competition",
                "theory", "law", "disease", "technology", "language", "religion",
            }

        # Default relations if none provided
        if self.valid_relations is None:
            self.valid_relations = {
                "born_in", "died_in", "lived_in", "nationality", "educated_at",
                "works_for", "employed_by", "founded", "created", "wrote",
                "directed", "married_to", "child_of", "parent_of", "sibling_of",
                "member_of", "headquartered_in", "subsidiary_of", "parent_company",
                "acquired", "merged_with", "partner_of", "competitor_of",
                "located_in", "capital_of", "part_of", "borders", "near",
                "occurred_in", "started_on", "ended_on", "participant_in",
                "owned_by", "made_by", "used_by", "contains", "made_of",
                "related_to", "instance_of", "subclass_of", "same_as",
            }

        # Statistics
        self._stats = {
            "total": 0,
            "passed": 0,
            "failed_spans": 0,
            "failed_concepts": 0,
            "failed_relations": 0,
            "failed_safety": 0,
            "fixed": 0,
        }

    def _validate_spans(
        self, text: str, entities: List[str], spans: List[List[int]]
    ) -> Tuple[List[List[int]], List[str], int]:
        """
        Validate and fix entity spans.

        Returns:
            (fixed_spans, issues, valid_count)
        """
        issues = []
        fixed_spans = []
        valid_count = 0

        if len(entities) != len(spans):
            issues.append(f"Entity/span count mismatch: {len(entities)} vs {len(spans)}")
            # Try to fix by finding entities in text
            spans = []
            for ent in entities:
                start = text.lower().find(ent.lower())
                if start >= 0:
                    spans.append([start, start + len(ent)])
                else:
                    spans.append([0, 0])

        for i, (entity, span) in enumerate(zip(entities, spans)):
            if len(span) != 2:
                issues.append(f"Invalid span format for entity {i}: {span}")
                # Try to find entity in text
                start = text.lower().find(entity.lower())
                if start >= 0:
                    fixed_spans.append([start, start + len(entity)])
                    valid_count += 1
                else:
                    fixed_spans.append([0, 0])
                continue

            start, end = span
            start = max(0, start)
            end = min(len(text), end)

            if end <= start:
                issues.append(f"Invalid span range for entity {i}: [{start}, {end}]")
                # Try to find entity
                found_start = text.lower().find(entity.lower())
                if found_start >= 0:
                    fixed_spans.append([found_start, found_start + len(entity)])
                    valid_count += 1
                else:
                    fixed_spans.append([0, 0])
                continue

            # Check if span matches entity
            span_text = text[start:end]
            if span_text.lower() == entity.lower():
                fixed_spans.append([start, end])
                valid_count += 1
            else:
                # Try fuzzy matching within tolerance
                found = False
                for offset in range(-self.max_span_tolerance, self.max_span_tolerance + 1):
                    new_start = max(0, start + offset)
                    new_end = min(len(text), new_start + len(entity))
                    if new_end <= new_start:
                        continue
                    check_text = text[new_start:new_end]
                    if check_text.lower() == entity.lower():
                        fixed_spans.append([new_start, new_end])
                        valid_count += 1
                        found = True
                        break

                if not found:
                    # Search entire text
                    found_start = text.lower().find(entity.lower())
                    if found_start >= 0:
                        fixed_spans.append([found_start, found_start + len(entity)])
                        valid_count += 1
                    else:
                        issues.append(
                            f"Cannot find entity '{entity}' at span [{start}, {end}]"
                        )
                        fixed_spans.append([start, end])

        return fixed_spans, issues, valid_count

    def _validate_concepts(
        self, concepts: List[List[str]], num_entities: int
    ) -> Tuple[List[List[str]], List[str]]:
        """
        Validate and fix concepts.

        Returns:
            (fixed_concepts, issues)
        """
        issues = []
        fixed_concepts = []

        # Ensure concepts list matches entity count
        while len(concepts) < num_entities:
            concepts.append(["object"])

        for i, concept_list in enumerate(concepts[:num_entities]):
            if not isinstance(concept_list, list):
                concept_list = [str(concept_list)]

            fixed_list = []
            for concept in concept_list:
                concept_lower = concept.lower().replace("-", "_").replace(" ", "_")
                if self.valid_concepts and concept_lower not in self.valid_concepts:
                    # Try to map to valid concept
                    mapped = self._map_concept(concept_lower)
                    if mapped:
                        fixed_list.append(mapped)
                    else:
                        issues.append(f"Invalid concept: {concept}")
                        fixed_list.append("object")  # Default fallback
                else:
                    fixed_list.append(concept_lower)

            if not fixed_list:
                fixed_list = ["object"]

            fixed_concepts.append(fixed_list[:3])  # Max 3 concepts per entity

        return fixed_concepts, issues

    def _map_concept(self, concept: str) -> Optional[str]:
        """Try to map an invalid concept to a valid one."""
        # Common mappings
        mappings = {
            "place": "location",
            "human": "person",
            "people": "person",
            "org": "organization",
            "corp": "company",
            "firm": "company",
            "date_time": "date",
            "datetime": "date",
            "number": "quantity",
            "num": "quantity",
            "thing": "object",
            "item": "object",
            "misc": "object",
            "other": "object",
        }
        return mappings.get(concept)

    def _validate_relations(
        self, relations: List[List], num_entities: int
    ) -> Tuple[List[List], List[str]]:
        """
        Validate and fix relations.

        Returns:
            (fixed_relations, issues)
        """
        issues = []
        fixed_relations = []

        for rel in relations:
            if len(rel) < 3:
                issues.append(f"Invalid relation format: {rel}")
                continue

            head_idx, tail_idx, rel_type = rel[0], rel[1], rel[2]

            # Validate indices
            try:
                head_idx = int(head_idx)
                tail_idx = int(tail_idx)
            except (ValueError, TypeError):
                issues.append(f"Invalid relation indices: {rel}")
                continue

            if head_idx < 0 or head_idx >= num_entities:
                issues.append(f"Head index out of bounds: {head_idx}")
                continue
            if tail_idx < 0 or tail_idx >= num_entities:
                issues.append(f"Tail index out of bounds: {tail_idx}")
                continue
            if head_idx == tail_idx:
                issues.append(f"Self-relation: {head_idx}")
                continue

            # Validate relation type
            rel_type_normalized = str(rel_type).lower().replace("-", "_").replace(" ", "_")
            if self.valid_relations and rel_type_normalized not in self.valid_relations:
                # Try common mappings
                mapped = self._map_relation(rel_type_normalized)
                if mapped:
                    rel_type_normalized = mapped
                else:
                    issues.append(f"Invalid relation type: {rel_type}")
                    rel_type_normalized = "related_to"

            fixed_relations.append([head_idx, tail_idx, rel_type_normalized])

        return fixed_relations, issues

    def _map_relation(self, relation: str) -> Optional[str]:
        """Try to map an invalid relation to a valid one."""
        mappings = {
            "is_in": "located_in",
            "in": "located_in",
            "at": "located_in",
            "from": "born_in",
            "works_at": "works_for",
            "employed": "employed_by",
            "founded_by": "founded",
            "created_by": "created",
            "written_by": "wrote",
            "spouse": "married_to",
            "spouse_of": "married_to",
            "parent": "parent_of",
            "child": "child_of",
            "sibling": "sibling_of",
            "belongs_to": "member_of",
            "hq_in": "headquartered_in",
            "owns": "owned_by",
            "produces": "made_by",
            "association": "related_to",
            "link": "related_to",
            "connection": "related_to",
        }
        return mappings.get(relation)

    def _check_safety(self, result: AnnotationResult) -> Tuple[bool, List[str]]:
        """Check content safety."""
        if self.safety_regulator is None:
            return True, []

        issues = []
        verdict = self.safety_regulator.check(
            result.text,
            entities=result.entities,
            source="curation_qc",
        )

        if not verdict.is_safe:
            issues.append(f"Safety filter: {verdict.category.value}")
            return False, issues

        # Also check response if present
        if result.response:
            resp_verdict = self.safety_regulator.check(
                result.response,
                source="curation_qc_response",
            )
            if not resp_verdict.is_safe:
                issues.append(f"Response safety filter: {resp_verdict.category.value}")
                return False, issues

        return True, []

    def validate(self, result: AnnotationResult) -> QCResult:
        """
        Validate and fix an annotation result.

        Args:
            result: AnnotationResult to validate

        Returns:
            QCResult with validation status and fixed result
        """
        self._stats["total"] += 1

        if not result.success:
            return QCResult(passed=False, issues=["Annotation failed"])

        all_issues = []

        # Validate spans
        fixed_spans, span_issues, valid_span_count = self._validate_spans(
            result.text, result.entities, result.entity_spans
        )
        all_issues.extend(span_issues)

        # Check span validity ratio
        if result.entities:
            span_ratio = valid_span_count / len(result.entities)
            if span_ratio < self.min_valid_span_ratio:
                self._stats["failed_spans"] += 1
                return QCResult(
                    passed=False,
                    issues=[f"Too few valid spans: {span_ratio:.2f} < {self.min_valid_span_ratio}"]
                    + all_issues,
                )

        # Validate concepts
        fixed_concepts, concept_issues = self._validate_concepts(
            result.concepts, len(result.entities)
        )
        all_issues.extend(concept_issues)

        # Validate relations
        fixed_relations, relation_issues = self._validate_relations(
            result.relations, len(result.entities)
        )
        all_issues.extend(relation_issues)

        # Check safety
        is_safe, safety_issues = self._check_safety(result)
        if not is_safe:
            self._stats["failed_safety"] += 1
            return QCResult(passed=False, issues=safety_issues + all_issues)

        # Create fixed result
        fixed_result = AnnotationResult(
            text=result.text,
            entities=result.entities,
            entity_spans=fixed_spans,
            concepts=fixed_concepts,
            relations=fixed_relations,
            should_respond=result.should_respond,
            response=result.response,
            success=True,
        )

        self._stats["passed"] += 1
        if all_issues:
            self._stats["fixed"] += 1

        return QCResult(
            passed=True,
            issues=all_issues,
            fixed_result=fixed_result,
            stats={
                "entities": len(result.entities),
                "valid_spans": valid_span_count,
                "relations": len(fixed_relations),
            },
        )

    def validate_batch(
        self, results: List[AnnotationResult]
    ) -> Tuple[List[AnnotationResult], List[Tuple[AnnotationResult, List[str]]]]:
        """
        Validate a batch of results.

        Returns:
            (passed_results, failed_results_with_issues)
        """
        passed = []
        failed = []

        for result in results:
            qc_result = self.validate(result)
            if qc_result.passed:
                passed.append(qc_result.fixed_result or result)
            else:
                failed.append((result, qc_result.issues))

        return passed, failed

    def get_statistics(self) -> Dict[str, Any]:
        """Get QC statistics."""
        total = self._stats["total"]
        return {
            **self._stats,
            "pass_rate": self._stats["passed"] / total if total > 0 else 0,
            "fix_rate": self._stats["fixed"] / self._stats["passed"]
            if self._stats["passed"] > 0
            else 0,
        }

    def reset_statistics(self):
        """Reset statistics."""
        for key in self._stats:
            self._stats[key] = 0
