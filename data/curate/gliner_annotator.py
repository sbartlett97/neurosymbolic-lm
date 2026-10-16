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
from .prompts import QA_JSON_SCHEMA, get_qa_prompt
from .source_loader import DocumentSource
from .taxonomy import (
    DEFAULT_GLINER_MODEL,
    MAX_CONCEPTS_PER_ENTITY,
    Taxonomy,
    get_default_taxonomy,
)


class EntityMatch:
    """A single labelled span candidate before merging."""

    __slots__ = ("start", "end", "text", "label", "score")

    def __init__(self, start: int, end: int, text: str, label: str, score: float):
        self.start = start
        self.end = end
        self.text = text
        self.label = label
        self.score = score


# Backwards-compatible private alias
_EntityMatch = EntityMatch

# Number of QA failures to log before going quiet
_QA_FAILURE_LOG_LIMIT = 5
# Fail fast: a failed QA document is written as a should_respond=0 row, so
# a rejected request or schema would silently give an abstain-only "QA"
# set. Stop when most of the first attempts fail.
QA_FAIL_FAST_MIN_ATTEMPTS = 50
QA_FAIL_FAST_RATE = 0.8

# Coarse types for code and tool-use text. Corpus prose skips these labels.
CODE_ENTITY_TYPES = frozenset({"code_construct", "digital_artifact"})


def select_device(requested: Optional[str] = None) -> str:
    """Pick a torch device: the requested one, else cuda > mps > cpu."""
    if requested:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def parse_qa_reply(response: Optional[str]) -> Tuple[Optional[Tuple[str, str]], str]:
    """Parse a QA JSON reply into (question, answer).

    Returns ((question, answer), "") or (None, error text).
    """
    try:
        match = re.search(r"\{[\s\S]*\}", response or "")
        parsed = json.loads(match.group() if match else response)
        question = str(parsed.get("question", "")).strip()
        answer = str(parsed.get("answer", "")).strip()
        if question and answer:
            return (question, answer), ""
        return None, "empty question or answer"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def normalize_surface(text: str) -> str:
    """Normalized surface form: lower case, single spaces."""
    return " ".join(text.lower().split())


def merge_entity_matches(
    matches: List[EntityMatch],
    taxonomy: Taxonomy,
    max_entities: int,
    consistent_types: bool = True,
) -> Tuple[List[str], List[List[int]], List[List[str]], List[str]]:
    """Merge overlapping label matches into entities.

    Two matches referring to overlapping spans are the same entity seen
    through different labels: the labels become its concepts, and the
    best-scoring label's coarse parent becomes its entity type.

    With ``consistent_types`` (the default), all mentions of one surface
    form in the input get one coarse type. The type is the one with the
    highest match score over all mentions of that surface form. Each
    mention keeps its own span. Its concepts are the labels of that type
    seen on the mention or on other mentions of the same surface form.
    Labels of other types are dropped. If no label is left, the coarse
    type name is the concept.

    Returns (entities, spans, concepts, entity_types), span end exclusive.
    """
    if not matches:
        return [], [], [], []

    matches = sorted(matches, key=lambda m: (m.start, -(m.end - m.start)))
    groups: List[List[EntityMatch]] = []
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

    # Best score per coarse type and per label, for each surface form
    type_votes: Dict[str, Dict[str, float]] = {}
    surface_labels: Dict[str, Dict[str, float]] = {}
    if consistent_types:
        for m in matches:
            coarse = taxonomy.coarse_type_of(m.label)
            if coarse is None:
                continue
            key = normalize_surface(m.text)
            votes = type_votes.setdefault(key, {})
            votes[coarse] = max(votes.get(coarse, 0.0), m.score)
            labels = surface_labels.setdefault(key, {})
            labels[m.label] = max(labels.get(m.label, 0.0), m.score)

    # Rank groups by best score so truncation keeps confident entities
    groups.sort(key=lambda g: -max(x.score for x in g))
    groups = groups[:max_entities]
    groups.sort(key=lambda g: min(x.start for x in g))

    entities, spans, concepts, entity_types = [], [], [], []
    for group in groups:
        best = max(group, key=lambda x: x.score)
        # Dedup labels keeping the highest score per label
        by_label: Dict[str, float] = {}
        for x in group:
            by_label[x.label] = max(by_label.get(x.label, 0.0), x.score)
        entity_type = taxonomy.coarse_type_of(best.label) or "abstract_concept"

        key = normalize_surface(best.text)
        votes = type_votes.get(key)
        if votes:
            # Ties keep the type that was seen first
            entity_type = max(votes.items(), key=lambda kv: kv[1])[0]
            for lbl, score in surface_labels.get(key, {}).items():
                by_label[lbl] = max(by_label.get(lbl, 0.0), score)
            by_label = {
                lbl: score
                for lbl, score in by_label.items()
                if taxonomy.coarse_type_of(lbl) == entity_type
            }

        ranked = sorted(by_label.items(), key=lambda kv: -kv[1])
        group_concepts = [lbl for lbl, _ in ranked[:MAX_CONCEPTS_PER_ENTITY]]
        if not group_concepts:
            group_concepts = [entity_type]

        entities.append(best.text)
        spans.append([best.start, best.end])
        concepts.append(group_concepts)
        entity_types.append(entity_type)

    return entities, spans, concepts, entity_types


class GLiNER2Annotator:
    """Annotate documents with GLiNER2 entity/relation extraction.

    Args:
        model_name: GLiNER2 checkpoint (default ``fastino/gliner2-base-v1``)
        taxonomy: Label inventory; defaults to :func:`get_default_taxonomy`
        entity_threshold: Minimum confidence for entity matches
        relation_threshold: Minimum confidence for relation matches
        device: torch device string, or None for auto (cuda > mps > cpu)
        response_backend: Optional LLM backend used only to generate grounded
            question/answer pairs (hybrid mode). When None, all samples are
            emitted with ``should_respond=0`` (pure Stage-1 symbolic data).
        should_respond_ratio: Fraction of samples that get a QA pair appended
            (only used when ``response_backend`` is set)
        max_len: Maximum word tokens GLiNER2 encodes per text. Longer
            texts are cut by GLiNER2; spans stay valid because offsets are
            character positions in the full text. None means no limit.
        extractor: Pre-constructed GLiNER2-compatible object (dependency
            injection for tests). Must expose ``extract_entities`` and
            ``extract_relations``. ``annotate_batch`` also uses
            ``batch_extract_entities``, ``batch_extract`` and
            ``create_schema`` when the extractor has them.
        include_code_labels: Score the code_construct and digital_artifact
            labels. True (the default) keeps the old behavior for traces.
            Set False for corpus prose, where these labels add wrong types.
        batch_size: Texts per GLiNER2 forward pass in ``annotate_batch``.
            None means 8 on CUDA and 1 elsewhere. On Apple MPS, passes of
            more than one text were slower than one text at a time.
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
        max_len: Optional[int] = None,
        extractor: Any = None,
        include_code_labels: bool = True,
        batch_size: Optional[int] = None,
    ):
        self.model_name = model_name
        self.taxonomy = taxonomy or get_default_taxonomy()
        self.entity_threshold = entity_threshold
        self.relation_threshold = relation_threshold
        self.response_backend = response_backend
        self.should_respond_ratio = should_respond_ratio
        self.max_entities = max_entities
        self.max_len = max_len
        self.include_code_labels = include_code_labels
        self.batch_size = batch_size or 8

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
            device = select_device(device)
            try:
                self.extractor = self.extractor.to(device)
            except Exception as e:
                print(f"Could not move GLiNER2 to {device} ({e}); using cpu")
                device = "cpu"
                self.extractor = self.extractor.to(device)
            print(f"GLiNER2 device: {device}")
            if batch_size is None:
                # 1 by default: on MPS, 8 texts per pass ran at 0.40 docs/s
                # against 0.95 per document. Measure on the GPU before raising it.
                self.batch_size = 1
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

    def _entity_label_batches(self) -> List[Dict[str, str]]:
        """Label batches to score, without code labels if they are off.

        Only labels are taken out. The batches keep their other labels, so
        the batches without code labels are the same as on the trace path.
        """
        batches = self.taxonomy.entity_label_batches()
        if self.include_code_labels:
            return batches
        out = []
        for batch in batches:
            kept = {
                name: desc
                for name, desc in batch.items()
                if self.taxonomy.coarse_type_of(name) not in CODE_ENTITY_TYPES
            }
            if kept:
                out.append(kept)
        return out

    def _extract_entity_matches(self, text: str) -> List[_EntityMatch]:
        """Run GLiNER2 over every label batch and collect raw matches."""
        matches: List[_EntityMatch] = []
        for label_batch in self._entity_label_batches():
            raw = self._call_extractor(
                self.extractor.extract_entities,
                text,
                label_batch,
                include_confidence=True,
                include_spans=True,
                threshold=self.entity_threshold,
            )
            matches.extend(self._normalize_entity_output(raw, text))
        return [m for m in matches if m.score >= self.entity_threshold]

    def _call_extractor(self, fn, text: str, labels, **kwargs) -> Any:
        """Call a GLiNER2 method, dropping kwargs the API does not accept.

        The first try passes ``max_len`` when it is set. The second try
        drops ``max_len``. The last try passes only text and labels.
        """
        if self.max_len is not None:
            try:
                return fn(text, labels, max_len=self.max_len, **kwargs)
            except TypeError:
                pass
        try:
            return fn(text, labels, **kwargs)
        except TypeError:
            # Older/newer API without some kwargs — retry minimally
            return fn(text, list(labels))

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
        return merge_entity_matches(matches, self.taxonomy, self.max_entities)

    def annotate_text(
        self, text: str
    ) -> Tuple[List[str], List[List[int]], List[List[str]], List[str], List[List]]:
        """Extract (entities, spans, concepts, entity_types, relations)
        from a plain text string. Used directly by the trace annotator."""
        matches = self._extract_entity_matches(text)
        entities, spans, concepts, entity_types = self._merge_matches(matches)
        relations = self._extract_relations(text, entities, entity_types, spans)
        return entities, spans, concepts, entity_types, relations

    # ------------------------------------------------------------------
    # Batched extraction (corpus path)
    # ------------------------------------------------------------------

    def _call_batch_extractor(self, fn, texts: List[str], labels, **kwargs) -> Any:
        """Call a GLiNER2 batch method. Drop ``max_len`` if it is refused."""
        kwargs.setdefault("batch_size", self.batch_size)
        if self.max_len is not None:
            try:
                return fn(texts, labels, max_len=self.max_len, **kwargs)
            except TypeError:
                pass
        return fn(texts, labels, **kwargs)

    def _extract_entity_matches_batch(self, texts: List[str]) -> List[List[_EntityMatch]]:
        """Entity matches for many texts, one GLiNER2 call per label batch.

        Extractors without ``batch_extract_entities`` use the per-text path.
        """
        fn = getattr(self.extractor, "batch_extract_entities", None)
        if fn is None:
            return [self._extract_entity_matches(t) for t in texts]
        per_text: List[List[_EntityMatch]] = [[] for _ in texts]
        for label_batch in self._entity_label_batches():
            raws = self._call_batch_extractor(
                fn,
                texts,
                label_batch,
                include_confidence=True,
                include_spans=True,
                threshold=self.entity_threshold,
            )
            for out, raw, text in zip(per_text, raws, texts):
                out.extend(self._normalize_entity_output(raw, text))
        return [[m for m in ms if m.score >= self.entity_threshold] for ms in per_text]

    def _extract_relations_batch(
        self,
        texts: List[str],
        entities_list: List[List[str]],
        types_list: List[List[str]],
        spans_list: List[List[List[int]]],
    ) -> List[List[List[Any]]]:
        """Relations for many texts.

        Each text has its own candidate relations, because the candidates
        depend on the entity types in that text. So each text gets its own
        schema in one ``batch_extract`` call. Call k sends the k-th
        relation batch of every text that has one. Extractors without
        ``batch_extract`` use the per-text path.
        """
        fn = getattr(self.extractor, "batch_extract", None)
        create_schema = getattr(self.extractor, "create_schema", None)
        if fn is None or create_schema is None:
            return [
                self._extract_relations(t, e, ty, sp)
                for t, e, ty, sp in zip(texts, entities_list, types_list, spans_list)
            ]
        candidates = [
            self._candidate_relation_batches(e, ty)
            for e, ty in zip(entities_list, types_list)
        ]
        raw_pairs: List[List[Dict[str, Any]]] = [[] for _ in texts]
        rounds = max((len(batches) for _, batches in candidates), default=0)
        for k in range(rounds):
            idx = [i for i, (_, batches) in enumerate(candidates) if len(batches) > k]
            schemas = [create_schema().relations(candidates[i][1][k]) for i in idx]
            raws = self._call_batch_extractor(
                fn,
                [texts[i] for i in idx],
                schemas,
                threshold=self.relation_threshold,
                include_confidence=True,
                include_spans=True,
            )
            for i, raw in zip(idx, raws):
                raw_pairs[i].extend(self._normalize_relation_output(raw))
        return [
            self._relations_from_pairs(
                raw_pairs[i], entities_list[i], types_list[i], spans_list[i], candidates[i][0]
            )
            if candidates[i][0]
            else []
            for i in range(len(texts))
        ]

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
            raw = self._call_extractor(
                self.extractor.extract_relations,
                text,
                rel_batch,
                threshold=self.relation_threshold,
                include_confidence=True,
                include_spans=True,
            )
            raw_pairs.extend(self._normalize_relation_output(raw))
        return self._relations_from_pairs(
            raw_pairs, entities, entity_types, spans, candidate_rels
        )

    def _candidate_relation_batches(
        self, entities: List[str], entity_types: List[str]
    ) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        """Candidate relations for one text and their label batches."""
        if len(entities) < 2:
            return {}, []
        candidate_rels = self.taxonomy.relations_for_types(set(entity_types))
        if not candidate_rels:
            return {}, []
        return candidate_rels, self.taxonomy.relation_batches(candidate_rels)

    def _relations_from_pairs(
        self,
        raw_pairs: List[Dict[str, Any]],
        entities: List[str],
        entity_types: List[str],
        spans: Optional[List[List[int]]],
        candidate_rels: Dict[str, str],
    ) -> List[List[Any]]:
        """Map normalized relation pairs to [head_idx, tail_idx, relation]."""
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

    def _parse_qa_response(self, response: Optional[str]) -> Optional[Tuple[str, str]]:
        """Parse one QA reply. Count it as generated or failed."""
        qa, error = parse_qa_reply(response)
        if qa is not None:
            self._stats["qa_generated"] += 1
            return qa
        self._stats["qa_failed"] += 1
        if self._stats["qa_failed"] <= _QA_FAILURE_LOG_LIMIT:
            preview = (response or "")[:200].replace("\n", "\\n")
            print(f"QA generation failed ({error}). Output: {preview!r}")
        return None

    def _maybe_generate_qa(
        self, text: str, entities: List[str], relations: List[List[Any]]
    ) -> Optional[Tuple[str, str]]:
        """Generate a grounded (question, answer) pair via the LLM backend."""
        if self.response_backend is None:
            return None
        return self._generate_qa_batch([(text, entities, relations)])[0]

    def _generate_qa_batch(
        self, items: List[Tuple[str, List[str], List[List[Any]]]]
    ) -> List[Optional[Tuple[str, str]]]:
        """Generate QA pairs for many texts with one backend call.

        vLLM batches the prompts. API backends send them in parallel. The
        schema gives structured output on API backends.
        """
        if not items:
            return []
        prompts = [get_qa_prompt(t, e, r) for t, e, r in items]
        try:
            # Instruct backends wrap each prompt in their chat template
            responses = self.response_backend.generate_chat_batch(
                prompts, json_schema=QA_JSON_SCHEMA
            )
        except Exception as e:
            print(f"QA batch generation failed ({type(e).__name__}: {e})")
            responses = [None] * len(prompts)
        results = [self._parse_qa_response(r) for r in responses]
        self._check_qa_failure_rate()
        return results

    def _check_qa_failure_rate(self) -> None:
        generated, failed = self._stats["qa_generated"], self._stats["qa_failed"]
        attempts = generated + failed
        if (
            not getattr(self, "_qa_rate_checked", False)
            and attempts >= QA_FAIL_FAST_MIN_ATTEMPTS
        ):
            self._qa_rate_checked = True
            if failed / attempts > QA_FAIL_FAST_RATE:
                raise RuntimeError(
                    f"QA generation failed for {failed}/{attempts} of the first "
                    "documents. Failed documents become should_respond=0 rows, "
                    "so the run stops. Check the logged errors (model name, "
                    "API key, schema) and run a 5-sample job first."
                )

    def _wants_qa(self, include_response: Optional[bool]) -> bool:
        """Decide whether one document gets a QA pair."""
        import random

        if include_response is None:
            return (
                self.response_backend is not None
                and random.random() < self.should_respond_ratio
            )
        return include_response

    def _build_result(
        self,
        text: str,
        extracted: Tuple[List[str], List[List[int]], List[List[str]], List[str], List[List]],
        qa: Optional[Tuple[str, str]],
    ) -> AnnotationResult:
        """Build a successful result. Append the QA question if there is one."""
        entities, spans, concepts, entity_types, relations = extracted
        should_respond = 0
        response = ""
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

    def _empty_result(self, text: str) -> AnnotationResult:
        self._stats["empty"] += 1
        return AnnotationResult(text=text, success=False, error="No entities extracted")

    # ------------------------------------------------------------------
    # Public API (mirrors LLMAnnotator)
    # ------------------------------------------------------------------

    def annotate(
        self, doc: DocumentSource, include_response: Optional[bool] = None
    ) -> AnnotationResult:
        """Annotate a single document."""
        self._stats["total"] += 1
        text = doc.text

        try:
            extracted = self.annotate_text(text)
        except Exception as e:
            return AnnotationResult(text=text, success=False, error=str(e))

        entities, _, _, _, relations = extracted
        if not entities:
            return self._empty_result(text)

        qa = None
        if self._wants_qa(include_response):
            qa = self._maybe_generate_qa(text, entities, relations)
        return self._build_result(text, extracted, qa)

    def annotate_batch(
        self, docs: List[DocumentSource], include_response: Optional[bool] = None
    ) -> List[AnnotationResult]:
        """Annotate a batch of documents.

        GLiNER2 scores all texts of the batch in one call per label batch
        (and per relation batch). QA prompts for the batch go to the backend
        in one call. Random QA draws happen in document order, as in
        :meth:`annotate`, so seeded runs give the same QA choices. If batch
        extraction fails, each document goes through :meth:`annotate`.
        """
        if not docs:
            return []
        texts = [doc.text for doc in docs]
        try:
            all_matches = self._extract_entity_matches_batch(texts)
            merged = [self._merge_matches(m) for m in all_matches]
            relations = self._extract_relations_batch(
                texts,
                [m[0] for m in merged],
                [m[3] for m in merged],
                [m[1] for m in merged],
            )
        except Exception as e:
            print(f"Batch extraction failed ({type(e).__name__}: {e}); per-document fallback")
            return [self.annotate(doc, include_response) for doc in docs]

        extracted = [(*m, r) for m, r in zip(merged, relations)]
        qa_index = []
        for i, (entities, *_rest) in enumerate(extracted):
            if entities and self._wants_qa(include_response):
                qa_index.append(i)
        qa_pairs: List[Optional[Tuple[str, str]]] = [None] * len(docs)
        if qa_index and self.response_backend is not None:
            generated = self._generate_qa_batch(
                [(texts[i], extracted[i][0], extracted[i][4]) for i in qa_index]
            )
            for i, qa in zip(qa_index, generated):
                qa_pairs[i] = qa

        results = []
        for i, text in enumerate(texts):
            self._stats["total"] += 1
            if not extracted[i][0]:
                results.append(self._empty_result(text))
            else:
                results.append(self._build_result(text, extracted[i], qa_pairs[i]))
        return results

    def get_statistics(self) -> Dict[str, Any]:
        total = self._stats["total"]
        qa_attempts = self._stats["qa_generated"] + self._stats["qa_failed"]
        return {
            **self._stats,
            "success_rate": self._stats["success"] / total if total > 0 else 0,
            "qa_failed_rate": self._stats["qa_failed"] / qa_attempts if qa_attempts > 0 else 0,
        }
