"""Tests for GLiNER2 batching, type merge by surface form, and code-label skip."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.curate.gliner_annotator import (
    CODE_ENTITY_TYPES,
    EntityMatch,
    GLiNER2Annotator,
    merge_entity_matches,
)
from data.curate.source_loader import DocumentSource
from data.curate.taxonomy import get_default_taxonomy

TEXTS = [
    "Marie Curie won the Nobel Prize in 1903. Curie worked at the University of Paris.",
    "Albert Einstein visited Brazil in 1925. Einstein lectured in Rio de Janeiro.",
    "Nothing to see here.",
]
LEXICON = {
    "Marie Curie": ("scientist", 0.95),
    "University of Paris": ("university", 0.91),
    "1903": ("year", 0.88),
    "Albert Einstein": ("scientist", 0.93),
    "Brazil": ("country", 0.9),
    "Rio de Janeiro": ("city", 0.85),
    "1925": ("year", 0.8),
}


class FakeSchema:
    def __init__(self):
        self.rels = None

    def relations(self, rels):
        self.rels = dict(rels)
        return self


class BatchStub:
    """Stub with per-text and batch APIs. Batch calls must equal per-text calls."""

    def __init__(self):
        self.calls = {"single": 0, "batch": 0}

    def _entities(self, text, labels):
        found = {}
        for surface, (label, conf) in LEXICON.items():
            if label in labels and surface in text:
                start = text.index(surface)
                found.setdefault(label, []).append(
                    {"text": surface, "confidence": conf, "start": start, "end": start + len(surface)}
                )
        return {"entities": found}

    def _relations(self, text, rels):
        out = {}
        if "works_for" in rels and "Marie Curie" in text:
            h, t = "Marie Curie", "University of Paris"
            out["works_for"] = [{
                "head": {"text": h, "confidence": 0.9, "start": text.index(h), "end": text.index(h) + len(h)},
                "tail": {"text": t, "confidence": 0.9, "start": text.index(t), "end": text.index(t) + len(t)},
            }]
        if "located_in" in rels and "Rio de Janeiro" in text:
            h, t = "Rio de Janeiro", "Brazil"
            out["located_in"] = [{
                "head": {"text": h, "confidence": 0.8, "start": text.index(h), "end": text.index(h) + len(h)},
                "tail": {"text": t, "confidence": 0.8, "start": text.index(t), "end": text.index(t) + len(t)},
            }]
        return {"relation_extraction": out}

    def extract_entities(self, text, labels, **kw):
        self.calls["single"] += 1
        return self._entities(text, labels)

    def extract_relations(self, text, rels, **kw):
        self.calls["single"] += 1
        return self._relations(text, rels)

    def batch_extract_entities(self, texts, labels, batch_size=8, **kw):
        self.calls["batch"] += 1
        return [self._entities(t, labels) for t in texts]

    def create_schema(self):
        return FakeSchema()

    def batch_extract(self, texts, schemas, batch_size=8, **kw):
        self.calls["batch"] += 1
        assert len(texts) == len(schemas)
        return [self._relations(t, s.rels) for t, s in zip(texts, schemas)]


def docs():
    return [DocumentSource(text=t, source="t", doc_id=str(i), metadata={}) for i, t in enumerate(TEXTS)]


def key(r):
    return (r.success, r.entities, r.entity_spans, r.entity_types, r.concepts, r.relations)


def test_batched_equals_per_document():
    per_stub, bat_stub = BatchStub(), BatchStub()
    per = [GLiNER2Annotator(extractor=per_stub).annotate(d) for d in docs()]
    bat = GLiNER2Annotator(extractor=bat_stub).annotate_batch(docs())
    assert [key(r) for r in per] == [key(r) for r in bat]
    assert per[0].relations and per[1].relations
    assert bat_stub.calls["single"] == 0 and bat_stub.calls["batch"] > 0


def test_batch_without_batch_api_falls_back():
    from tests.test_gliner_annotator import StubExtractor

    ann = GLiNER2Annotator(extractor=StubExtractor())
    out = ann.annotate_batch(docs()[:1])
    assert out[0].success and "Marie Curie" in out[0].entities


def test_batch_qa_one_backend_call_and_same_draws():
    import random

    class Backend:
        def __init__(self):
            self.batches = []

        def generate(self, prompts):
            return ['{"question": "Q?", "answer": "A."}' for _ in prompts]

        def generate_single(self, prompt):
            return self.generate([prompt])[0]

        def generate_chat_batch(self, prompts, system=None, json_schema=None):
            self.batches.append(len(prompts))
            return self.generate(prompts)

    from data.curate.llm_annotator import LLMBackend

    class B(Backend, LLMBackend):
        pass

    backend = B()
    random.seed(0)
    bat = GLiNER2Annotator(extractor=BatchStub(), response_backend=backend, should_respond_ratio=0.5).annotate_batch(docs())
    random.seed(0)
    per = [GLiNER2Annotator(extractor=BatchStub(), response_backend=B(), should_respond_ratio=0.5).annotate(d) for d in docs()]
    assert [r.should_respond for r in bat] == [r.should_respond for r in per]
    assert len(backend.batches) <= 1


def m(text, start, label, score):
    return EntityMatch(start, start + len(text), text, label, score)


def test_type_merge_one_type_per_surface_form():
    tax = get_default_taxonomy()
    text = "Jordan spoke. Later Jordan left. Jordan again."
    matches = [
        m("Jordan", 0, "scientist", 0.6),
        m("Jordan", 0, "function", 0.55),
        m("Jordan", 20, "country", 0.9),
        m("jordan", 33, "scientist", 0.7),
    ]
    ents, spans, concepts, types = merge_entity_matches(matches, tax, 10)
    assert len(ents) == 3  # every mention keeps its own span
    assert spans == [[0, 6], [20, 26], [33, 39]]
    coarse = tax.coarse_type_of("country")
    assert types == [coarse] * 3
    for c in concepts:
        assert all(tax.coarse_type_of(x) == coarse for x in c)
        assert "country" in c


def test_type_merge_can_be_turned_off():
    tax = get_default_taxonomy()
    matches = [m("Jordan", 0, "scientist", 0.6), m("Jordan", 20, "country", 0.9)]
    _, _, concepts, types = merge_entity_matches(matches, tax, 10, consistent_types=False)
    assert types == [tax.coarse_type_of("scientist"), tax.coarse_type_of("country")]
    assert concepts == [["scientist"], ["country"]]


def test_code_labels_skipped_only_when_flag_off():
    tax = get_default_taxonomy()
    default = GLiNER2Annotator(extractor=BatchStub())._entity_label_batches()
    prose = GLiNER2Annotator(extractor=BatchStub(), include_code_labels=False)._entity_label_batches()
    assert default == tax.entity_label_batches()
    labels = [l for b in prose for l in b]
    assert labels and not any(tax.coarse_type_of(l) in CODE_ENTITY_TYPES for l in labels)
    # Non-code labels are all kept
    expected = [l for l in tax.concept_labels if tax.coarse_type_of(l) not in CODE_ENTITY_TYPES]
    assert sorted(labels) == sorted(expected)
