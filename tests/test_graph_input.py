"""A separate read-pass input (graph_input_ids): the graph comes from other
text than the prompt. Both injection modes, on a tiny offline backbone."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")

from model.neurosymbolic_causal import NeuroSymbolicCausalLM  # noqa: E402

VOCAB = 2048
MODES = ["prefix", "xattn"]


def make_model(injection, gate=1.0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    backbone = LlamaForCausalLM(LlamaConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512))
    model = NeuroSymbolicCausalLM(
        backbone=backbone, tap_layer_ratio=0.5, n_entity_types=8,
        n_relations=16, n_concepts=16, concept_dim=16, node_dim=32,
        max_nodes=4, adapter_layers=1, adapter_heads=2, injection=injection,
        xattn_every=1, xattn_heads=4)
    with torch.no_grad():
        for g in model.injection_gates():
            g.fill_(gate)
    return model.eval()


def batch(B=2, L=10, short=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, VOCAB, (B, L), generator=g)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[1, L - short:] = 0  # right padding
    ids[1, L - short:] = 0
    return ids, mask


@pytest.mark.parametrize("injection", MODES)
def test_graph_input_equal_to_prompt_changes_nothing(injection):
    model = make_model(injection)
    ids, mask = batch()
    y = torch.randint(2, VOCAB, (2, 5))
    with torch.no_grad():
        a = model(ids, mask, y_ids=y)
        b = model(ids, mask, y_ids=y, graph_input_ids=ids, graph_attention_mask=mask)
    for k in ("logits", "entity_logits", "node_feats", "rel_logits_matrix"):
        assert torch.equal(a[k], b[k]), k


@pytest.mark.parametrize("injection", MODES)
def test_graph_input_drives_the_read_pass_only(injection):
    model = make_model(injection)
    ids, mask = batch(L=6, short=2)
    gids, gmask = batch(L=14, short=5, seed=7)  # longer than the prompt
    y = torch.randint(2, VOCAB, (2, 5))
    with torch.no_grad():
        own = model(ids, mask, y_ids=y)
        other = model(ids, mask, y_ids=y, graph_input_ids=gids, graph_attention_mask=gmask)
        alone = model(gids, gmask)  # the graph text as a prompt
        own_off = model(ids, mask, y_ids=y, use_soft_nodes=False)
        other_off = model(ids, mask, y_ids=y, use_soft_nodes=False,
                          graph_input_ids=gids, graph_attention_mask=gmask)
    # The heads and nodes come from the graph text
    assert other["entity_logits"].shape[1] == gids.shape[1]
    assert torch.equal(other["node_feats"], alone["node_feats"])
    assert torch.equal(other["entity_logits"], alone["entity_logits"])
    # The graph changes the answer logits through the nodes only
    assert not torch.allclose(own["logits"], other["logits"])
    assert torch.equal(own_off["logits"], other_off["logits"])


@pytest.mark.parametrize("injection", MODES)
def test_generate_with_graph_input_agrees_with_training_pass(injection):
    model = make_model(injection)
    ids, mask = batch(L=6, short=2)
    gids, gmask = batch(L=14, short=5, seed=7)
    out = model.generate(ids, mask, max_length=4, do_sample=False,
                         pad_token_id=0, eos_token_id=None,
                         output_logits=True, return_dict_in_generate=True,
                         graph_input_ids=gids, graph_attention_mask=gmask)
    gen_logits = torch.stack(out.logits, dim=1)
    with torch.no_grad():
        logits = model(ids, mask, y_ids=out.sequences, graph_input_ids=gids,
                       graph_attention_mask=gmask)["logits"]
        own = model(ids, mask, y_ids=out.sequences)["logits"]
    assert torch.allclose(gen_logits, logits, atol=1e-4)
    assert not torch.allclose(logits, own, atol=1e-3)


@pytest.mark.parametrize("injection", MODES)
def test_shuffle_with_graph_input(injection):
    model = make_model(injection)
    ids, mask = batch(L=6, short=2)
    gids, gmask = batch(L=14, short=5, seed=7)
    y = torch.randint(2, VOCAB, (2, 5))
    kw = dict(y_ids=y, graph_input_ids=gids, graph_attention_mask=gmask)
    with torch.no_grad():
        right = model(ids, mask, **kw)
        model.shuffle_nodes = True
        wrong = model(ids, mask, **kw)
        model.shuffle_nodes = False
    assert not torch.allclose(right["logits"], wrong["logits"])
    assert torch.equal(right["node_feats"], wrong["node_feats"])


def test_graph_input_needs_a_mask_and_the_batch_size():
    model = make_model("xattn")
    ids, mask = batch()
    with pytest.raises(ValueError, match="graph_attention_mask"):
        model(ids, mask, graph_input_ids=ids)
    with pytest.raises(ValueError, match="batch size"):
        model(ids, mask, graph_input_ids=ids[:1], graph_attention_mask=mask[:1])
    left = mask.flip(1)  # a left-padded prompt
    with pytest.raises(ValueError):
        model(ids, left, graph_input_ids=ids, graph_attention_mask=mask)


def make_collator():
    from data.collator_causal import CausalCognitiveCollator
    from data.curate.taxonomy import get_default_taxonomy
    from tests.test_collator import FakeTokenizer
    vocab = get_default_taxonomy().vocab()
    return CausalCognitiveCollator(
        tokenizer=FakeTokenizer(), concept_map=vocab["concepts"],
        relation_map=vocab["relations"], include_responses=True), FakeTokenizer()


def test_collator_aligns_spans_to_graph_text():
    collator, tok = make_collator()
    doc = "Ada Lovelace was born in London in 1815 ."
    rows = [
        {"text": "Where was Ada born ?", "graph_text": doc,
         "entities": ["Ada Lovelace", "London"], "entity_spans": [[0, 12], [25, 31]],
         "concepts": [], "relations": [], "should_respond": 1, "response": "London ."},
        # No graph_text: the prompt is the graph input
        {"text": "Paris is in France .", "entities": ["France"],
         "entity_spans": [[12, 18]], "concepts": [], "relations": [],
         "should_respond": 1, "response": "Yes ."},
    ]
    batch = collator(rows)
    assert batch["input_ids"].shape[1] == 5  # the longer prompt
    assert batch["graph_input_ids"].shape[1] == len(doc.split())
    assert batch["graph_input_ids"][0].tolist() == tok(doc)["input_ids"]
    assert batch["graph_input_ids"][1, :5].tolist() == tok(rows[1]["text"])["input_ids"]
    assert batch["graph_attention_mask"][1].tolist() == [1] * 5 + [0] * 4
    # Token spans in graph_text: "Ada Lovelace" = 0-1, "London" = 5
    assert batch["entity_token_spans"][0] == [(0, 1), (5, 5)]
    assert batch["entity_token_spans"][1] == [(3, 3)]


def test_collator_without_graph_text_has_no_graph_keys():
    collator, _ = make_collator()
    batch = collator([{"text": "Paris is in France .", "entities": [], "concepts": [],
                       "relations": [], "should_respond": 0}])
    assert "graph_input_ids" not in batch and "graph_attention_mask" not in batch


@pytest.mark.parametrize("injection", MODES)
def test_trainer_losses_use_the_graph_input(injection):
    from training.trainers import Stage3_Decoder_Trainer, Stage4_Joint_Trainer
    model = make_model(injection)
    ids, mask = batch(L=6, short=2)
    gids, gmask = batch(L=14, short=5, seed=7)
    y = torch.randint(2, VOCAB, (2, 5))
    common = {"input_ids": ids, "attention_mask": mask, "decoder_input_ids": y,
              "decoder_labels": y, "entity_ids": torch.zeros(2, 1, dtype=torch.long),
              "entity_type_labels": torch.zeros(2, 1, dtype=torch.long),
              "concept_labels": torch.zeros(2, 1, 16), "relations": [[], []],
              "should_respond": torch.ones(2, dtype=torch.long),
              "entity_token_spans": [[(2, 3)], [(12, 13)]]}  # (12, 13) is only in the graph text
    with_graph = dict(common, graph_input_ids=gids, graph_attention_mask=gmask)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    dec = Stage3_Decoder_Trainer(model, opt)
    joint = Stage4_Joint_Trainer(model, opt)
    with torch.no_grad():
        assert not torch.allclose(dec.compute_loss(common), dec.compute_loss(with_graph))
        joint.compute_loss(with_graph)  # spans past the prompt length work
