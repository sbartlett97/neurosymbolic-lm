"""Integration test: annotate -> QC -> write, with stub models only."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.curate.gliner_annotator import GLiNER2Annotator
from data.curate.output_writer import OutputWriter
from data.curate.quality_control import QualityControl
from data.curate.source_loader import DocumentSource
from data.curate.taxonomy import get_default_taxonomy

DOCS = [
    "Albert Einstein developed the theory of relativity in 1905.",
    "Amazon acquired Whole Foods for 13.7 billion dollars in 2017.",
    "The Amazon River flows through Brazil and Peru.",
]


class TinyStubExtractor:
    """Emits one plausible entity dict per capitalized word, GLiNER2-shaped."""

    LABELS = {
        "Albert": ("scientist", 0.9),
        "Einstein": ("scientist", 0.9),
        "Amazon": ("company", 0.8),
        "Whole": ("company", 0.7),
        "Foods": ("company", 0.7),
        "Brazil": ("country", 0.9),
        "Peru": ("country", 0.9),
        "1905": ("year", 0.85),
        "2017": ("year", 0.85),
    }

    def extract_entities(self, text, entity_types, **kw):
        found = {}
        for word, (label, conf) in self.LABELS.items():
            if label in entity_types and word in text:
                start = text.index(word)
                found.setdefault(label, []).append(
                    {"text": word, "confidence": conf,
                     "start": start, "end": start + len(word)}
                )
        return {"entities": found}

    def extract_relations(self, text, relation_types, **kw):
        return {"relation_extraction": {}}


def test_pipeline_writes_valid_jsonl_and_vocab(tmp_path):
    tax = get_default_taxonomy()
    annotator = GLiNER2Annotator(extractor=TinyStubExtractor(), taxonomy=tax)
    qc = QualityControl(taxonomy=tax)
    writer = OutputWriter(output_dir=str(tmp_path), output_name="test", taxonomy=tax)

    docs = [
        DocumentSource(text=t, source="test", doc_id=str(i), metadata={})
        for i, t in enumerate(DOCS)
    ]

    results = annotator.annotate_batch(docs, include_response=False)
    passed = []
    for r in results:
        qc_r = qc.validate(r)
        if qc_r.passed:
            passed.append(qc_r.fixed_result or r)

    assert len(passed) == len(DOCS)
    writer.write_batch(passed, ["test"] * len(passed))
    writer.finalize()

    # Validate JSONL output
    out_file = tmp_path / "test.jsonl"
    lines = [json.loads(l) for l in out_file.read_text().splitlines() if l.strip()]
    assert len(lines) == len(DOCS)
    for sample in lines:
        assert len(sample["entities"]) == len(sample["entity_spans"])
        assert len(sample["entities"]) == len(sample["concepts"])
        assert len(sample["entities"]) == len(sample["entity_types"])
        for ent, span in zip(sample["entities"], sample["entity_spans"]):
            assert sample["text"][span[0]:span[1]] == ent
        for etype in sample["entity_types"]:
            assert etype in tax.entity_types
        for concept_list in sample["concepts"]:
            for c in concept_list:
                assert c in tax.concept_labels or c in tax.entity_types

    # Validate vocab file matches taxonomy (stable across runs)
    vocab = json.loads((tmp_path / "test_vocab.json").read_text())
    assert vocab["concepts"] == tax.vocab()["concepts"]
    assert vocab["relations"] == tax.vocab()["relations"]
    assert vocab["entity_types"] == tax.vocab()["entity_types"]
    assert vocab["concept_to_entity_type"] == tax.vocab()["concept_to_entity_type"]
    assert vocab["statistics"]["num_samples"] == len(DOCS)
