"""Tests for GLiNER2Annotator using a stub extractor.

The stub emits the exact output shapes produced by gliner2's
``format_results`` (verified against gliner2's inference engine):

- entities: {"entities": {label: [{"text", "confidence", "start", "end"}]}}
- relations: {"relation_extraction": {rel: [{"head": {...}, "tail": {...}}]}}
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.curate.gliner_annotator import GLiNER2Annotator
from data.curate.llm_annotator import LLMBackend
from data.curate.quality_control import QualityControl
from data.curate.source_loader import DocumentSource
from data.curate.taxonomy import get_default_taxonomy

TEXT = "Marie Curie won the Nobel Prize in 1903. She worked at the University of Paris."


class StubExtractor:
    """Mimics GLiNER2's extract_entities / extract_relations output."""

    def __init__(self):
        self.entity_calls = 0
        self.relation_calls = 0

    def extract_entities(self, text, entity_types, threshold=0.5,
                         include_confidence=False, include_spans=False, **kw):
        self.entity_calls += 1
        matches = {
            "scientist": [
                {"text": "Marie Curie", "confidence": 0.95,
                 "start": TEXT.index("Marie Curie"),
                 "end": TEXT.index("Marie Curie") + len("Marie Curie")}
            ],
            "generic_person": [
                {"text": "Marie Curie", "confidence": 0.60,
                 "start": TEXT.index("Marie Curie"),
                 "end": TEXT.index("Marie Curie") + len("Marie Curie")}
            ],
            "university": [
                {"text": "University of Paris", "confidence": 0.91,
                 "start": TEXT.index("University of Paris"),
                 "end": TEXT.index("University of Paris") + len("University of Paris")}
            ],
            "year": [
                {"text": "1903", "confidence": 0.88,
                 "start": TEXT.index("1903"),
                 "end": TEXT.index("1903") + len("1903")}
            ],
            "document": [
                {"text": "Nobel Prize", "confidence": 0.55,
                 "start": TEXT.index("Nobel Prize"),
                 "end": TEXT.index("Nobel Prize") + len("Nobel Prize")}
            ],
        }
        # Only return labels that were asked about in this pass
        return {"entities": {k: v for k, v in matches.items() if k in entity_types}}

    def extract_relations(self, text, relation_types, threshold=0.5,
                          include_confidence=False, include_spans=False, **kw):
        self.relation_calls += 1
        rels = {
            "works_for": [
                {
                    "head": {"text": "Marie Curie", "confidence": 0.9,
                             "start": TEXT.index("Marie Curie"),
                             "end": TEXT.index("Marie Curie") + len("Marie Curie")},
                    "tail": {"text": "University of Paris", "confidence": 0.85,
                             "start": TEXT.index("University of Paris"),
                             "end": TEXT.index("University of Paris") + len("University of Paris")},
                }
            ],
            # Implausible pair: person born_in year — must be dropped by
            # taxonomy type constraints (born_in tail must be a location)
            "born_in": [
                {
                    "head": {"text": "Marie Curie", "confidence": 0.8,
                             "start": TEXT.index("Marie Curie"),
                             "end": TEXT.index("Marie Curie") + len("Marie Curie")},
                    "tail": {"text": "1903", "confidence": 0.7,
                             "start": TEXT.index("1903"),
                             "end": TEXT.index("1903") + len("1903")},
                }
            ],
        }
        return {
            "relation_extraction": {
                k: v for k, v in rels.items() if k in relation_types
            }
        }


class StubQABackend(LLMBackend):
    def generate(self, prompts):
        return [self.generate_single(p) for p in prompts]

    def generate_single(self, prompt):
        return json.dumps({
            "question": "Who won the Nobel Prize in 1903?",
            "answer": "Marie Curie won the Nobel Prize in 1903.",
        })


def make_annotator(**kwargs):
    return GLiNER2Annotator(extractor=StubExtractor(), **kwargs)


def doc():
    return DocumentSource(text=TEXT, source="test", doc_id="t0", metadata={})


def test_entities_spans_concepts_types():
    ann = make_annotator()
    result = ann.annotate(doc(), include_response=False)

    assert result.success
    assert "Marie Curie" in result.entities
    assert "University of Paris" in result.entities
    assert "1903" in result.entities

    # Spans are end-exclusive and point at the entity text
    for ent, span in zip(result.entities, result.entity_spans):
        assert result.text[span[0]:span[1]] == ent

    idx = result.entities.index("Marie Curie")
    # Both matching labels become concepts, best-scoring first
    assert result.concepts[idx][0] == "scientist"
    assert "generic_person" in result.concepts[idx]
    assert result.entity_types[idx] == "person"

    uni = result.entities.index("University of Paris")
    assert result.entity_types[uni] == "organization"


def test_relations_mapped_to_indices_and_type_checked():
    ann = make_annotator()
    result = ann.annotate(doc(), include_response=False)

    mc = result.entities.index("Marie Curie")
    uni = result.entities.index("University of Paris")
    assert [mc, uni, "works_for"] in result.relations

    # born_in(person, temporal) violates tail-type constraint -> dropped
    assert not any(r[2] == "born_in" for r in result.relations)


def test_gliner_only_mode_emits_abstain_samples():
    ann = make_annotator()
    result = ann.annotate(doc(), include_response=False)
    assert result.should_respond == 0
    assert result.response == ""


def test_hybrid_mode_appends_question_and_keeps_spans_valid():
    ann = make_annotator(response_backend=StubQABackend(), should_respond_ratio=1.0)
    result = ann.annotate(doc(), include_response=True)

    assert result.should_respond == 1
    assert result.response == "Marie Curie won the Nobel Prize in 1903."
    assert result.text.startswith(TEXT.rstrip())
    assert "Who won the Nobel Prize in 1903?" in result.text
    # Document spans must remain valid after appending the question
    for ent, span in zip(result.entities, result.entity_spans):
        assert result.text[span[0]:span[1]] == ent


def test_qc_passes_gliner_output():
    tax = get_default_taxonomy()
    ann = make_annotator(taxonomy=tax)
    result = ann.annotate(doc(), include_response=False)

    qc = QualityControl(taxonomy=tax)
    qc_result = qc.validate(result)
    assert qc_result.passed, qc_result.issues
    fixed = qc_result.fixed_result
    assert fixed.entity_types == result.entity_types
    assert fixed.entity_spans == result.entity_spans


def test_to_dict_contains_all_training_fields():
    ann = make_annotator()
    sample = ann.annotate(doc(), include_response=False).to_dict()
    for key in ("text", "entities", "entity_spans", "entity_types",
                "concepts", "relations", "should_respond", "response"):
        assert key in sample


def test_empty_document_fails_gracefully():
    ann = GLiNER2Annotator(extractor=_EmptyExtractor())
    result = ann.annotate(DocumentSource(text="Nothing here.", source="t", doc_id="1", metadata={}))
    assert not result.success
    assert result.error == "No entities extracted"


class _EmptyExtractor:
    def extract_entities(self, text, entity_types, **kw):
        return {"entities": {}}

    def extract_relations(self, text, relation_types, **kw):
        return {"relation_extraction": {}}


def test_relation_arg_matching_by_span_overlap():
    matched = GLiNER2Annotator._match_entity_index(
        "Curie", ["Marie Curie", "University of Paris"],
        arg_span=(0, 11), entity_spans=[[0, 11], [60, 79]],
    )
    assert matched == 0


def test_relation_arg_matching_by_surface_fallback():
    matched = GLiNER2Annotator._match_entity_index(
        "Apple", ["Apple Inc.", "Tim Cook"],
    )
    assert matched == 0
