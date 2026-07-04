"""GLiNER2-based annotation for the neurosymbolic dataset.

Replaces free-form LLM JSON annotation with schema-driven extraction:

- Entities + character spans + confidence come from GLiNER2 zero-shot NER
  against the fine-grained label inventory in :mod:`taxonomy`.
- Concepts per entity are the set of fine-grained labels that matched the
  span (GLiNER2 "picks" from the diverse inventory we supply — it cannot
  invent labels, so diversity lives in the taxonomy, not the prompt).
- The coarse entity type is the parent type of the best-scoring label.
- Relations come from GLiNER2 relation extraction, restricted to relation
  labels that are plausible given the entity types present in the document,
  then mapped back to entity indices.
- ``should_respond``/``response`` supervision is optional: in ``hybrid``
  mode an LLM backend generates a grounded question+answer which is
  *appended* to the document text, so all previously-extracted character
  spans remain valid.

The annotator returns :class:`AnnotationResult` objects, so it is a drop-in
replacement for :class:`LLMAnnotator` in the curation pipeline.
"""

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .llm_annotator import AnnotationResult, LLMBackend
from .prompts import get_qa_prompt
from .source_loader import DocumentSource
from .taxonomy import (
    DEFAULT_GLINER_MODEL,
    MAX_CONCEPTS_PER_ENTITY,
    Taxonomy,
    get_default_taxonomy,
)


class _EntityMatch:
    """A single labelled span candidate before merging."""

    __slots__ = ("start", "end", "text", "label", "score")

    def __init__(self, start: int, end: int, text: str, label: str, score: float):
        self.start = start
        self.end = end
        self.text = text
        self.label = label
        self.score = score


class GLiNER2Annotator:
    """Annotate documents with GLiNER2 entity/relation extraction.

    Args:
        model_name: GLiNER2 checkpoint (default ``fastino/gliner2-base-v1``)
        taxonomy: Label inventory; defaults to :func:`get_default_taxonomy`
        entity_threshold: Minimum confidence for entity matches
        relation_threshold: Minimum confidence for relation matches
        device: torch device string, or None for auto
        response_backend: Optional LLM backend used only to generate grounded
            question/answer pairs (hybrid mode). When None, all samples are
            emitted with ``should_respond=0`` (pure Stage-1 symbolic data).
        should_respond_ratio: Fraction of samples that get a QA pair appended
            (only used when ``response_backend`` is set)
        extractor: Pre-constructed GLiNER2-compatible object (dependency
            injection for tests). Must expose ``extract_entities`` and
            ``extract_relations``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_GLINER_MODEL,
        taxonomy: Optional[Taxonomy] = None,
        entity_threshold: float = 0.5,
        relation_threshold: float = 0.4,
        device: Optional[str] = None,
        response_backend: Optional[LLMBackend] = None,
        should_respond_ratio: float = 0.5,
        max_entities: int = 24,
        extractor: Any = None,
    ):
        self.model_name = model_name
        self.taxonomy = taxonomy or get_default_taxonomy()
        self.entity_threshold = entity_threshold
        self.relation_threshold = relation_threshold
        self.response_backend = response_backend
        self.should_respond_ratio = should_respond_ratio
        self.max_entities = max_entities

        if extractor is not None:
            self.extractor = extractor
        else:
            try:
                from gliner2 import GLiNER2
            except ImportError as e:
                raise ImportError(
                    "GLiNER2 is required for GLiNER2Annotator: "
                    "pip install 'gliner2[local]'"
                ) from e
            self.extractor = GLiNER2.from_pretrained(model_name)
            if device is not None:
                self.extractor = self.extractor.to(device)
            self.extractor.eval()

        self._stats = {
            "total": 0,
            "success": 0,
            "empty": 0,
            "qa_generated": 0,
            "qa_failed": 0,
            "relation_pairs_dropped": 0,
        }

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def _extract_entity_matches(self, text: str) -> List[_EntityMatch]:
        """Run GLiNER2 over every label batch and collect raw matches."""
        matches: List[_EntityMatch] = []
        for label_batch in self.taxonomy.entity_label_batches():
            try:
                raw = self.extractor.extract_entities(
                    text,
                    label_batch,
                    include_confidence=True,
                    include_spans=True,
                    threshold=self.entity_threshold,
                )
            except TypeError:
                # Older/newer API without some kwargs — retry minimally
                raw = self.extractor.extract_entities(text, label_batch)
            matches.extend(self._normalize_entity_output(raw, text))
        return [m for m in matches if m.score >= self.entity_threshold]

    def _normalize_entity_output(self, raw: Any, text: str) -> List[_EntityMatch]:
        """Normalize GLiNER2 entity output shapes into _EntityMatch objects.

        Handles: {"entities": {...}} wrappers, {label: [match, ...]} dicts,
        and flat [ {label, text, start, end, score}, ... ] lists. Matches may
        be plain strings (no span info) — those are anchored by text search.
        """
        out: List[_EntityMatch] = []
        if raw is None:
            return out
        if isinstance(raw, dict) and "entities" in raw and isinstance(raw["entities"], (dict, list)):
            raw = raw["entities"]

        if isinstance(raw, dict):
            for label, items in raw.items():
                if not isinstance(items, (list, tuple)):
                    items = [items]
                for item in items:
                    m = self._match_from_item(item, label, text)
                    if m:
                        out.append(m)
        elif isinstance(raw, (list, tuple)):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                label = item.get("label") or item.get("type") or item.get("entity")
                if label is None:
                    continue
                m = self._match_from_item(item, label, text)
                if m:
                    out.append(m)
        return out

    def _match_from_item(
        self, item: Any, label: str, text: str
    ) -> Optional[_EntityMatch]:
        if label not in self.taxonomy.concept_labels:
            return None
        if isinstance(item, str):
            span = self._find_span(text, item)
            if span is None:
                return None
            return _EntityMatch(span[0], span[1], item, label, 1.0)
        if isinstance(item, dict):
            ent_text = item.get("text") or item.get("span") or item.get("value")
            if not ent_text or not isinstance(ent_text, str):
                return None
            score = float(
                item.get("score", item.get("confidence", item.get("prob", 1.0)))
            )
            start = item.get("start")
            end = item.get("end")
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
                # Trust offsets only if they actually point at the text
                if text[start:end].strip().lower() == ent_text.strip().lower():
                    return _EntityMatch(start, end, text[start:end], label, score)
            span = self._find_span(text, ent_text)
            if span is None:
                return None
            return _EntityMatch(span[0], span[1], ent_text, label, score)
        return None

    @staticmethod
    def _find_span(text: str, entity: str) -> Optional[Tuple[int, int]]:
        """First case-insensitive, word-boundary-preferring occurrence."""
        entity = entity.strip()
        if not entity:
            return None
        pattern = re.escape(entity)
        m = re.search(rf"\b{pattern}\b", text, flags=re.IGNORECASE)
        if m is None:
            m = re.search(pattern, text, flags=re.IGNORECASE)
        if m is None:
            return None
        return m.start(), m.end()

    def _merge_matches(
        self, matches: List[_EntityMatch]
    ) -> Tuple[List[str], List[List[int]], List[List[str]], List[str]]:
        """Merge overlapping label matches into entities.

        Two matches referring to overlapping spans are the same entity seen
        through different labels: the labels become its concepts, and the
        best-scoring label's coarse parent becomes its entity type.

        Returns (entities, spans, concepts, entity_types), span end exclusive.
        """
        if not matches:
            return [], [], [], []

        matches = sorted(matches, key=lambda m: (m.start, -(m.end - m.start)))
        groups: List[List[_EntityMatch]] = []
        for m in matches:
            placed = False
            for group in groups:
                g_start = min(x.start for x in group)
                g_end = max(x.end for x in group)
                if m.start < g_end and m.end > g_start:  # overlap
                    group.append(m)
                    placed = True
                    break
            if not placed:
                groups.append([m])

        # Rank groups by best score so truncation keeps confident entities
        groups.sort(key=lambda g: -max(x.score for x in g))
        groups = groups[: self.max_entities]
        groups.sort(key=lambda g: min(x.start for x in g))

        entities, spans, concepts, entity_types = [], [], [], []
        for group in groups:
            best = max(group, key=lambda x: x.score)
            # Dedup labels keeping the highest score per label
            by_label: Dict[str, float] = {}
            for x in group:
                by_label[x.label] = max(by_label.get(x.label, 0.0), x.score)
            ranked = sorted(by_label.items(), key=lambda kv: -kv[1])
            group_concepts = [lbl for lbl, _ in ranked[:MAX_CONCEPTS_PER_ENTITY]]

            entities.append(best.text)
            spans.append([best.start, best.end])
            concepts.append(group_concepts)
            entity_types.append(self.taxonomy.coarse_type_of(best.label) or "abstract_concept")

        return entities, spans, concepts, entity_types

    # ------------------------------------------------------------------
    # Relation extraction
    # ------------------------------------------------------------------

    def _extract_relations(
        self,
        text: str,
        entities: List[str],
        entity_types: List[str],
        spans: Optional[List[List[int]]] = None,
    ) -> List[List[Any]]:
        """Extract relations and map them to entity indices."""
        if len(entities) < 2:
            return []

        present_types = set(entity_types)
        candidate_rels = self.taxonomy.relations_for_types(present_types)
        if not candidate_rels:
            return []

        raw_pairs: List[Dict[str, Any]] = []
        for rel_batch in self.taxonomy.relation_batches(candidate_rels):
            try:
                raw = self.extractor.extract_relations(
                    text,
                    rel_batch,
                    threshold=self.relation_threshold,
                    include_confidence=True,
                    include_spans=True,
                )
            except TypeError:
                raw = self.extractor.extract_relations(text, list(rel_batch))
            raw_pairs.extend(self._normalize_relation_output(raw))

        relations: List[List[Any]] = []
        seen = set()
        for pair in raw_pairs:
            rel = pair["relation"]
            if pair["score"] < self.relation_threshold or rel not in candidate_rels:
                continue
            head_idx = self._match_entity_index(
                pair["head"], entities, pair.get("head_span"), spans
            )
            tail_idx = self._match_entity_index(
                pair["tail"], entities, pair.get("tail_span"), spans
            )
            if head_idx is None or tail_idx is None or head_idx == tail_idx:
                self._stats["relation_pairs_dropped"] += 1
                continue
            if not self.taxonomy.relation_type_plausible(
                rel, entity_types[head_idx], entity_types[tail_idx]
            ):
                self._stats["relation_pairs_dropped"] += 1
                continue
            key = (head_idx, tail_idx, rel)
            if key in seen:
                continue
            seen.add(key)
            relations.append([head_idx, tail_idx, rel])
        return relations

    def _normalize_relation_output(self, raw: Any) -> List[Dict[str, Any]]:
        """Normalize GLiNER2 relation output.

        With ``include_confidence=True, include_spans=True`` GLiNER2 returns
        ``{"relation_extraction": {rel: [{"head": {"text", "confidence",
        "start", "end"}, "tail": {...}}, ...]}}``; without the flags each
        pair is a plain ``(head, tail)`` tuple. Both are handled.

        Returns dicts with keys: relation, head, tail, head_span, tail_span,
        score (spans are None when unavailable).
        """
        out: List[Dict[str, Any]] = []
        if raw is None:
            return out
        if isinstance(raw, dict):
            for wrapper in ("relation_extraction", "relations"):
                if wrapper in raw and isinstance(raw[wrapper], dict):
                    raw = raw[wrapper]
                    break
        if not isinstance(raw, dict):
            return out

        def arg_parts(arg: Any) -> Tuple[Optional[str], Optional[Tuple[int, int]], float]:
            if isinstance(arg, str):
                return arg, None, 1.0
            if isinstance(arg, dict):
                arg_text = arg.get("text")
                span = None
                if isinstance(arg.get("start"), int) and isinstance(arg.get("end"), int):
                    span = (arg["start"], arg["end"])
                return arg_text, span, float(arg.get("confidence", 1.0))
            return None, None, 0.0

        for rel, pairs in raw.items():
            if not isinstance(pairs, (list, tuple)):
                continue
            for pair in pairs:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    head, tail = pair[0], pair[1]
                elif isinstance(pair, dict):
                    head = pair.get("head") or pair.get("source") or pair.get("subject")
                    tail = pair.get("tail") or pair.get("target") or pair.get("object")
                else:
                    continue
                head_text, head_span, head_score = arg_parts(head)
                tail_text, tail_span, tail_score = arg_parts(tail)
                if not head_text or not tail_text:
                    continue
                out.append(
                    {
                        "relation": rel,
                        "head": head_text,
                        "tail": tail_text,
                        "head_span": head_span,
                        "tail_span": tail_span,
                        "score": min(head_score, tail_score),
                    }
                )
        return out

    @staticmethod
    def _match_entity_index(
        surface: str,
        entities: Sequence[str],
        arg_span: Optional[Tuple[int, int]] = None,
        entity_spans: Optional[List[List[int]]] = None,
    ) -> Optional[int]:
        """Map a relation argument to an entity index.

        Prefers character-span overlap (exact anchoring from GLiNER2) and
        falls back to surface-form matching when spans are unavailable.
        """
        if arg_span is not None and entity_spans:
            a_start, a_end = arg_span
            best, best_overlap = None, 0
            for i, span in enumerate(entity_spans):
                if len(span) != 2:
                    continue
                overlap = min(a_end, span[1]) - max(a_start, span[0])
                if overlap > best_overlap:
                    best, best_overlap = i, overlap
            if best is not None:
                return best

        s = surface.strip().lower()
        if not s:
            return None
        for i, ent in enumerate(entities):
            if ent.strip().lower() == s:
                return i
        # Containment fallback (e.g. "Apple" vs "Apple Inc.") — prefer the
        # longest-overlap candidate to avoid matching "New York" to "York".
        best, best_len = None, 0
        for i, ent in enumerate(entities):
            e = ent.strip().lower()
            if (s in e or e in s) and min(len(e), len(s)) > best_len:
                best, best_len = i, min(len(e), len(s))
        return best

    # ------------------------------------------------------------------
    # QA generation (hybrid mode)
    # ------------------------------------------------------------------

    def _maybe_generate_qa(
        self, text: str, entities: List[str], relations: List[List[Any]]
    ) -> Optional[Tuple[str, str]]:
        """Generate a grounded (question, answer) pair via the LLM backend."""
        if self.response_backend is None:
            return None
        prompt = get_qa_prompt(text, entities, relations)
        try:
            response = self.response_backend.generate_single(prompt)
            match = re.search(r"\{[\s\S]*\}", response)
            parsed = json.loads(match.group() if match else response)
            question = str(parsed.get("question", "")).strip()
            answer = str(parsed.get("answer", "")).strip()
            if question and answer:
                self._stats["qa_generated"] += 1
                return question, answer
        except Exception:
            pass
        self._stats["qa_failed"] += 1
        return None

    # ------------------------------------------------------------------
    # Public API (mirrors LLMAnnotator)
    # ------------------------------------------------------------------

    def annotate(
        self, doc: DocumentSource, include_response: Optional[bool] = None
    ) -> AnnotationResult:
        """Annotate a single document."""
        import random

        self._stats["total"] += 1
        text = doc.text

        try:
            matches = self._extract_entity_matches(text)
            entities, spans, concepts, entity_types = self._merge_matches(matches)
            relations = self._extract_relations(text, entities, entity_types, spans)
        except Exception as e:
            return AnnotationResult(text=text, success=False, error=str(e))

        if not entities:
            self._stats["empty"] += 1
            return AnnotationResult(
                text=text, success=False, error="No entities extracted"
            )

        should_respond = 0
        response = ""
        if include_response is None:
            include_response = (
                self.response_backend is not None
                and random.random() < self.should_respond_ratio
            )
        if include_response:
            qa = self._maybe_generate_qa(text, entities, relations)
            if qa is not None:
                question, answer = qa
                # Append the question so existing char spans stay valid.
                text = f"{text.rstrip()}\n\n{question}"
                should_respond = 1
                response = answer

        self._stats["success"] += 1
        return AnnotationResult(
            text=text,
            entities=entities,
            entity_spans=spans,
            concepts=concepts,
            relations=relations,
            entity_types=entity_types,
            should_respond=should_respond,
            response=response,
            success=True,
        )

    def annotate_batch(
        self, docs: List[DocumentSource], include_response: Optional[bool] = None
    ) -> List[AnnotationResult]:
        """Annotate a batch of documents.

        GLiNER2 batch APIs vary between releases, so this iterates; the
        per-document passes are already small (encoder-only model).
        """
        return [self.annotate(doc, include_response) for doc in docs]

    def get_statistics(self) -> Dict[str, Any]:
        total = self._stats["total"]
        return {
            **self._stats,
            "success_rate": self._stats["success"] / total if total > 0 else 0,
        }
