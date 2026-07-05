"""Annotation of multi-turn assistant traces.

A trace sample keeps its ``messages`` list intact and carries symbolic
supervision per message, with spans relative to each message's *stripped*
content (the same coordinates ChatTemplate.format_messages_with_offsets
reports):

    {
        "messages": [{"role": ..., "content": ...}, ...],
        "message_annotations": [
            {"message_idx": 1, "entities": [...], "entity_spans": [...],
             "concepts": [...], "entity_types": [...], "relations": [...]},
            ...
        ],
        "should_respond": 1,
        "response": "<content of the final assistant turn>"
    }

Each message is segmented into fenced code blocks and prose:

- prose  -> GLiNER2 zero-shot extraction (optional) + deterministic regex
            extraction of digital artifacts (URLs, file paths, versions...)
- code   -> CodeAnnotator (AST-based, exact spans)

Segment-relative spans are shifted back into message coordinates here; the
collator later shifts message coordinates into encoder-input coordinates.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from .code_annotator import CodeAnnotator
from .gliner_annotator import EntityMatch, merge_entity_matches
from .taxonomy import Taxonomy, get_default_taxonomy

# Fenced code blocks: ```lang\n ... ```
CODE_BLOCK_RE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)

# Deterministic extractors for digital artifacts. Regex beats zero-shot NER
# for these: the patterns are exact and never hallucinate.
ARTIFACT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("url", re.compile(r"https?://[^\s)\]}>\"']+")),
    ("email_address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("file_path", re.compile(
        r"(?:(?<=\s)|^)(?:~?/|\./|\.\./)?(?:[\w.-]+/)+[\w.-]+\.\w{1,8}\b"
    )),
    ("version_number", re.compile(r"\bv?\d+\.\d+(?:\.\d+)+(?:[-+][\w.]+)?\b")),
    ("environment_variable", re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")),
]


def regex_artifact_matches(text: str) -> List[EntityMatch]:
    """Extract digital-artifact entities with exact regex spans."""
    matches: List[EntityMatch] = []
    for label, pattern in ARTIFACT_PATTERNS:
        for m in pattern.finditer(text):
            surface = m.group().rstrip(".,;:")
            if not surface:
                continue
            matches.append(
                EntityMatch(m.start(), m.start() + len(surface), surface, label, 1.0)
            )
    return matches


class TraceAnnotator:
    """Annotate assistant traces per message.

    Args:
        gliner_annotator: Optional GLiNER2Annotator for prose NER. When None,
            only deterministic extraction (regex artifacts + code AST) runs —
            useful for tests and cheap pure-code corpora.
        code_annotator: CodeAnnotator for fenced code blocks.
        taxonomy: Label inventory (defaults to the shared default taxonomy).
        max_entities_per_message: Cap per message to bound node counts.
        annotate_roles: Which message roles receive symbolic annotation.
            The final assistant turn is never annotated (it is the decoder
            target, not encoder input).
    """

    def __init__(
        self,
        gliner_annotator=None,
        code_annotator: Optional[CodeAnnotator] = None,
        taxonomy: Optional[Taxonomy] = None,
        max_entities_per_message: int = 16,
        annotate_roles: Tuple[str, ...] = ("system", "user", "assistant", "tool"),
    ):
        self.gliner = gliner_annotator
        self.code = code_annotator or CodeAnnotator(max_entities=max_entities_per_message)
        self.taxonomy = taxonomy or (
            gliner_annotator.taxonomy if gliner_annotator is not None else get_default_taxonomy()
        )
        self.max_entities_per_message = max_entities_per_message
        self.annotate_roles = set(annotate_roles)

        self._stats = {"traces": 0, "messages_annotated": 0, "entities": 0, "relations": 0}

    # ------------------------------------------------------------------

    def annotate_trace(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Produce a trace training sample from a normalized message list."""
        self._stats["traces"] += 1

        last_assistant_idx = max(
            (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
            default=None,
        )

        message_annotations = []
        for idx, msg in enumerate(messages):
            role = msg.get("role", "")
            content = (msg.get("content") or "").strip()
            if not content or role not in self.annotate_roles:
                continue
            if idx == last_assistant_idx:
                continue  # decoder target, not encoder input

            ann = self.annotate_message(content)
            if ann["entities"]:
                ann["message_idx"] = idx
                message_annotations.append(ann)
                self._stats["messages_annotated"] += 1
                self._stats["entities"] += len(ann["entities"])
                self._stats["relations"] += len(ann["relations"])

        response = ""
        if last_assistant_idx is not None:
            response = (messages[last_assistant_idx].get("content") or "").strip()

        return {
            "messages": messages,
            "message_annotations": message_annotations,
            "should_respond": 1 if response else 0,
            "response": response,
        }

    # ------------------------------------------------------------------

    def annotate_message(self, content: str) -> Dict[str, Any]:
        """Annotate one message's stripped content.

        Returns {entities, entity_spans, concepts, entity_types, relations}
        with spans relative to ``content``.
        """
        entities: List[str] = []
        spans: List[List[int]] = []
        concepts: List[List[str]] = []
        entity_types: List[str] = []
        relations: List[List[Any]] = []

        for seg_start, seg_text, language in self._segments(content):
            if language is None:
                e, s, c, t, r = self._annotate_prose(seg_text)
            else:
                ann = self.code.annotate(seg_text, language)
                e, s, c, t, r = (
                    ann["entities"], ann["entity_spans"], ann["concepts"],
                    ann["entity_types"], ann["relations"],
                )

            index_offset = len(entities)
            entities.extend(e)
            spans.extend([[seg_start + a, seg_start + b] for a, b in s])
            concepts.extend(c)
            entity_types.extend(t)
            relations.extend([[h + index_offset, tl + index_offset, rel] for h, tl, rel in r])

            if len(entities) >= self.max_entities_per_message:
                entities = entities[: self.max_entities_per_message]
                spans = spans[: self.max_entities_per_message]
                concepts = concepts[: self.max_entities_per_message]
                entity_types = entity_types[: self.max_entities_per_message]
                relations = [
                    rel for rel in relations
                    if rel[0] < self.max_entities_per_message
                    and rel[1] < self.max_entities_per_message
                ]
                break

        # Span integrity guarantee
        keep = [
            i for i, (ent, sp) in enumerate(zip(entities, spans))
            if content[sp[0]:sp[1]] == ent
        ]
        if len(keep) != len(entities):
            remap = {old: new for new, old in enumerate(keep)}
            entities = [entities[i] for i in keep]
            spans = [spans[i] for i in keep]
            concepts = [concepts[i] for i in keep]
            entity_types = [entity_types[i] for i in keep]
            relations = [
                [remap[h], remap[t], r] for h, t, r in relations
                if h in remap and t in remap
            ]

        return {
            "entities": entities,
            "entity_spans": spans,
            "concepts": concepts,
            "entity_types": entity_types,
            "relations": relations,
        }

    # ------------------------------------------------------------------

    def _segments(self, content: str) -> List[Tuple[int, str, Optional[str]]]:
        """Split content into (offset, text, language) segments.

        language is None for prose and the fence tag for code blocks (code
        offset points at the code body, after the opening fence line).
        """
        segments: List[Tuple[int, str, Optional[str]]] = []
        pos = 0
        for m in CODE_BLOCK_RE.finditer(content):
            if m.start() > pos:
                segments.append((pos, content[pos:m.start()], None))
            language = (m.group(1) or "").lower() or None
            code_start = m.start(2)
            segments.append((code_start, m.group(2), language or "unknown"))
            pos = m.end()
        if pos < len(content):
            segments.append((pos, content[pos:], None))
        return segments

    def _annotate_prose(
        self, text: str
    ) -> Tuple[List[str], List[List[int]], List[List[str]], List[str], List[List]]:
        matches: List[EntityMatch] = list(regex_artifact_matches(text))
        if self.gliner is not None and text.strip():
            matches.extend(self.gliner._extract_entity_matches(text))

        entities, spans, concepts, entity_types = merge_entity_matches(
            matches, self.taxonomy, self.max_entities_per_message
        )

        relations: List[List] = []
        if self.gliner is not None and len(entities) >= 2:
            relations = self.gliner._extract_relations(
                text, entities, entity_types, spans
            )
        return entities, spans, concepts, entity_types, relations

    def get_statistics(self) -> Dict[str, Any]:
        return dict(self._stats)
