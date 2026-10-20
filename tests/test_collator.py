"""Tests for CognitiveCollator span/type handling with a fake tokenizer."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")

from data.collator import CognitiveCollator
from data.curate.taxonomy import get_default_taxonomy


class FakeTokenizer:
    """Whitespace tokenizer with offset mapping, enough for the collator."""

    pad_token_id = 0
    eos_token_id = 1

    def _tokenize(self, text):
        tokens, offsets = [], []
        pos = 0
        for word in text.split():
            start = text.index(word, pos)
            tokens.append(word)
            offsets.append((start, start + len(word)))
            pos = start + len(word)
        return tokens, offsets

    def __call__(self, texts, padding=False, truncation=False, max_length=None,
                 return_tensors=None, add_special_tokens=True,
                 return_offsets_mapping=False):
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        all_ids, all_offsets = [], []
        for text in texts:
            tokens, offsets = self._tokenize(text)
            # Deterministic ids: hash of token, offset by 2 for specials
            ids = [2 + (abs(hash(t)) % 1000) for t in tokens]
            if max_length and truncation:
                ids = ids[:max_length]
                offsets = offsets[:max_length]
            all_ids.append(ids)
            all_offsets.append(offsets)

        if padding and len(all_ids) > 1:
            width = max(len(x) for x in all_ids)
            all_ids = [x + [self.pad_token_id] * (width - len(x)) for x in all_ids]

        masks = [[1 if t != 0 else 0 for t in x] for x in all_ids]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(all_ids), "attention_mask": torch.tensor(masks)}
        result = {
            "input_ids": all_ids[0] if single else all_ids,
            "attention_mask": masks[0] if single else masks,
        }
        if return_offsets_mapping:
            result["offset_mapping"] = all_offsets[0] if single else all_offsets
        return result

    def encode(self, text, add_special_tokens=True):
        tokens, _ = self._tokenize(text)
        return [2 + (abs(hash(t)) % 1000) for t in tokens]


TEXT = "Marie Curie worked at the University of Paris in 1903 ."


def make_collator(**kwargs):
    tax = get_default_taxonomy()
    vocab = tax.vocab()
    return CognitiveCollator(
        tokenizer=FakeTokenizer(),
        concept_map=vocab["concepts"],
        relation_map=vocab["relations"],
        concept_to_entity_type_map=vocab["concept_to_entity_type"],
        entity_type_map=vocab["entity_types"],
        **kwargs,
    )


def sample(**overrides):
    s = {
        "text": TEXT,
        "entities": ["Marie Curie", "University of Paris", "1903"],
        "entity_spans": [
            [TEXT.index("Marie Curie"), TEXT.index("Marie Curie") + len("Marie Curie")],
            [TEXT.index("University of Paris"), TEXT.index("University of Paris") + len("University of Paris")],
            [TEXT.index("1903"), TEXT.index("1903") + len("1903")],
        ],
        "entity_types": ["person", "organization", "temporal"],
        "concepts": [["scientist"], ["university"], ["year"]],
        "relations": [[0, 1, "works_for"]],
        "should_respond": 0,
    }
    s.update(overrides)
    return s


def test_gold_spans_produce_correct_token_spans():
    collator = make_collator()
    batch = collator([sample()])
    token_spans = batch["entity_token_spans"][0]

    words = TEXT.split()
    # "Marie Curie" = word tokens 0-1
    assert token_spans[0] == (0, 1)
    # "University of Paris" = words 4-6
    assert token_spans[1] == (words.index("University"), words.index("Paris"))
    # "1903" = word 8
    assert token_spans[2] == (words.index("1903"), words.index("1903"))


def test_explicit_entity_types_used():
    collator = make_collator()
    batch = collator([sample()])
    tax = get_default_taxonomy()
    types = batch["entity_type_labels"][0]
    assert types[0].item() == tax.entity_type_id("person")
    assert types[1].item() == tax.entity_type_id("organization")
    assert types[2].item() == tax.entity_type_id("temporal")


def test_entity_type_fallback_from_concepts():
    collator = make_collator()
    s = sample()
    del s["entity_types"]
    batch = collator([s])
    tax = get_default_taxonomy()
    types = batch["entity_type_labels"][0]
    # Derived via concept_to_entity_type_map: scientist -> person
    assert types[0].item() == tax.entity_type_id("person")


def test_relations_encoded_with_vocab_indices():
    collator = make_collator()
    batch = collator([sample()])
    tax = get_default_taxonomy()
    rels = batch["relations"][0]
    assert rels == [(0, 1, tax.vocab()["relations"]["works_for"])]


def test_bad_gold_spans_fall_back_to_search():
    collator = make_collator()
    s = sample()
    # Corrupt the spans; entities are still findable by search
    s["entity_spans"] = [[0, 3], [0, 2], [1, 2]]
    batch = collator([s])
    token_spans = batch["entity_token_spans"][0]
    assert token_spans[0] == (0, 1)  # found "Marie Curie" by search


def test_concept_multilabels_set():
    collator = make_collator()
    batch = collator([sample()])
    tax = get_default_taxonomy()
    concept_map = tax.vocab()["concepts"]
    labels = batch["concept_labels"][0]
    assert labels[0, concept_map["scientist"] - 1].item() == 1.0
    assert labels[1, concept_map["university"] - 1].item() == 1.0
