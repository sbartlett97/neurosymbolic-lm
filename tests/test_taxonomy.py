"""Tests for the annotation label taxonomy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.curate.taxonomy import (
    CONCEPT_LABELS,
    ENTITY_TYPES,
    MAX_LABELS_PER_PASS,
    RELATION_LABELS,
    Taxonomy,
    get_default_taxonomy,
)


def test_budgets_match_model_config():
    """Taxonomy must fit the architecture budgets in config.ModelConfig."""
    # n_entity_types=24 with index 0 reserved for none/padding
    assert len(ENTITY_TYPES) <= 23
    # n_concepts defaults to 1024; concepts + coarse types must fit
    assert len(CONCEPT_LABELS) + len(ENTITY_TYPES) <= 1024
    # n_relations defaults to 128 with index 0 reserved
    assert len(RELATION_LABELS) <= 127


def test_every_concept_maps_to_valid_entity_type():
    for concept, (parent, desc) in CONCEPT_LABELS.items():
        assert parent in ENTITY_TYPES, f"{concept} -> unknown type {parent}"
        assert desc.strip(), f"{concept} has no description"


def test_relation_type_constraints_reference_valid_types():
    for rel, (desc, heads, tails) in RELATION_LABELS.items():
        assert desc.strip()
        for t in list(heads) + list(tails):
            assert t in ENTITY_TYPES, f"{rel} constraint uses unknown type {t}"


def test_entity_label_batches_cover_all_labels():
    tax = get_default_taxonomy()
    batches = tax.entity_label_batches()
    seen = set()
    for batch in batches:
        assert len(batch) <= MAX_LABELS_PER_PASS
        seen.update(batch)
    assert seen == set(CONCEPT_LABELS)


def test_relations_for_types_prunes():
    tax = get_default_taxonomy()
    person_org = tax.relations_for_types({"person", "organization"})
    assert "works_for" in person_org
    assert "borders" not in person_org  # needs two locations
    # Unconstrained relations always survive
    assert "related_to" in person_org


def test_relation_plausibility():
    tax = get_default_taxonomy()
    assert tax.relation_type_plausible("born_in", "person", "location")
    assert not tax.relation_type_plausible("born_in", "organization", "location")
    assert tax.relation_type_plausible("related_to", "product", "event")


def test_vocab_is_one_indexed_and_includes_coarse_concepts():
    tax = get_default_taxonomy()
    vocab = tax.vocab()
    for section in ("concepts", "relations", "entity_types"):
        values = sorted(vocab[section].values())
        assert values[0] == 1, f"{section} must be 1-indexed (0 reserved)"
        assert values == list(range(1, len(values) + 1))
    # coarse types double as concepts
    for t in ENTITY_TYPES:
        assert t in vocab["concepts"]
    # concept -> entity type map covers all concepts and coarse types
    c2t = vocab["concept_to_entity_type"]
    for c in CONCEPT_LABELS:
        assert c2t[c] == vocab["entity_types"][CONCEPT_LABELS[c][0]]
    for t in ENTITY_TYPES:
        assert c2t[t] == vocab["entity_types"][t]


def test_trace_and_code_branches_present():
    tax = get_default_taxonomy()
    assert tax.coarse_type_of("file_path") == "digital_artifact"
    assert tax.coarse_type_of("tool_name") == "digital_artifact"
    assert tax.coarse_type_of("function") == "code_construct"
    assert tax.coarse_type_of("module_or_package") == "code_construct"
    # Trace/code relations are pruned in when their types are present
    code_rels = tax.relations_for_types({"code_construct", "digital_artifact"})
    for rel in ("calls", "defined_in", "imports", "raises", "argument_of"):
        assert rel in code_rels
    # ... and pruned out for purely encyclopedic documents
    wiki_rels = tax.relations_for_types({"person", "location", "organization"})
    assert "defined_in" not in wiki_rels
    assert "imports" not in wiki_rels


def test_custom_taxonomy():
    tax = Taxonomy(
        concept_labels={"cat": ("animal", "a cat")},
        relation_labels={"chases": ("x chases y", (), ())},
        entity_types=["animal"],
    )
    assert tax.coarse_type_of("cat") == "animal"
    assert tax.entity_type_id("animal") == 1
    assert tax.entity_type_id("unknown") == 0
