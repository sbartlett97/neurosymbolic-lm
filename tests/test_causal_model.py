"""End-to-end tests for NeuroSymbolicCausalLM on a tiny offline backbone.

Unlike the encoder-decoder tests (which can't download T5 weights), these
run real forward/backward passes: the backbone is a from-config Llama with
random weights, so every mechanism — tap, adapter, heads, packing,
injection, generation, staged freezing — is exercised for real.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")

from data.chat_template import ChatTemplate  # noqa: E402
from data.collator_causal import CausalCognitiveCollator  # noqa: E402
from data.curate.taxonomy import get_default_taxonomy  # noqa: E402
from model.neurosymbolic_causal import (  # noqa: E402
    NeuroSymbolicCausalLM,
    build_tiny_backbone,
)
from tests.test_collator import FakeTokenizer  # noqa: E402

VOCAB = 2048
HIDDEN = 64


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    # n_relations/n_entity_types sized to the taxonomy vocab, as train_causal
    # does from the vocab file (relation indices must fit the head)
    return NeuroSymbolicCausalLM(
        backbone=build_tiny_backbone(vocab_size=VOCAB, hidden_size=HIDDEN),
        n_entity_types=24,
        n_relations=64,
        n_concepts=32,
        concept_dim=32,
        node_dim=32,
        max_nodes=4,
        adapter_layers=1,
        adapter_heads=2,
    )


def make_prompt_batch(B=2, P=12):
    torch.manual_seed(1)
    input_ids = torch.randint(2, VOCAB, (B, P))
    attention_mask = torch.ones(B, P, dtype=torch.long)
    # Give the second sample a shorter prompt (right padding)
    if B > 1:
        attention_mask[1, P - 3:] = 0
        input_ids[1, P - 3:] = 0
    return input_ids, attention_mask


def test_read_pass_output_keys_and_shapes(model):
    input_ids, mask = make_prompt_batch()
    out = model(input_ids, mask)

    B, P = input_ids.shape
    assert out["token_ent_logits"].shape == (B, P, 24)
    assert out["entity_logits"].shape == (B, P, 24)
    assert out["concept_probs"].shape == (B, 32)
    assert out["node_feats"].shape == (B, 4, 32)
    assert out["rel_logits_matrix"].shape == (B, 4, 4, 64)
    assert out["node_entity_type_probs"].shape == (B, 4, 24)
    assert len(out["pair_relation_logits"]) == B
    assert out["pair_relation_logits"][0].shape == (6, 64)  # C(4,2) pairs
    # Same key set the trainers and continual learning read
    for key in ("enc", "concept_logits", "pair_relation_logits"):
        assert key in out


def test_topk_node_selection_ignores_padding(model):
    """Padding positions must never be selected as entity nodes."""
    input_ids, mask = make_prompt_batch()
    enc = model._read(input_ids, mask)
    logits = model.token_ent(enc)
    # Force pad positions to look maximally entity-like
    logits = logits.clone()
    logits[1, -3:, :] = 100.0
    ent_scores = logits.max(dim=-1).values
    ent_scores = ent_scores.masked_fill(mask == 0, float("-inf"))
    _, topk = torch.topk(ent_scores, k=4, dim=-1)
    assert all(int(i) < 9 for i in topk[1])  # sample 1 real length is 9


def test_write_pass_logits_aligned_to_response(model):
    input_ids, mask = make_prompt_batch()
    y_ids = torch.randint(2, VOCAB, (2, 5))
    out = model(input_ids, mask, y_ids=y_ids)
    assert out["logits"].shape == (2, 5, VOCAB)

    labels = y_ids.clone()
    loss = torch.nn.functional.cross_entropy(
        out["logits"].reshape(-1, VOCAB), labels.reshape(-1)
    )
    assert torch.isfinite(loss)


def test_no_nodes_matches_vanilla_backbone_exactly(model):
    """With soft nodes disabled, the packed write pass must reproduce the
    plain backbone forward on [prompt ; response] bit-for-bit."""
    input_ids = torch.randint(2, VOCAB, (1, 8))
    mask = torch.ones(1, 8, dtype=torch.long)
    y_ids = torch.randint(2, VOCAB, (1, 4))

    model.eval()
    with torch.no_grad():
        resp_logits = model._write(input_ids, mask, None, y_ids)

        combined = torch.cat([input_ids, y_ids], dim=1)
        vanilla = model.backbone(
            input_ids=combined,
            attention_mask=torch.ones_like(combined),
        ).logits
        # y_ids[:, j] is predicted at combined position 8 + j - 1
        expected = vanilla[:, 7:11]

    assert torch.allclose(resp_logits, expected, atol=1e-5)


def test_lm_loss_backprops_into_injector_and_gnn(model):
    """The broadcast path must be differentiable: LM loss alone should
    produce gradients in the injector, GNN, and node projection."""
    model.train()
    model.zero_grad(set_to_none=True)
    input_ids, mask = make_prompt_batch()
    y_ids = torch.randint(2, VOCAB, (2, 5))

    out = model(input_ids, mask, y_ids=y_ids, use_soft_nodes=True)
    loss = torch.nn.functional.cross_entropy(
        out["logits"].reshape(-1, VOCAB), y_ids.reshape(-1)
    )
    loss.backward()

    assert model.injector.proj.weight.grad is not None
    assert model.injector.gate.grad is not None
    assert model.node_proj.weight.grad is not None
    gnn_grads = [p.grad for p in model.gnn.parameters() if p.grad is not None]
    assert gnn_grads and any(g.abs().sum() > 0 for g in gnn_grads)


def test_staged_freezing(model):
    model.freeze_backbone(True)
    model.freeze_symbolic(False)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.token_ent.parameters())
    assert all(p.requires_grad for p in model.adapter.parameters())

    model.freeze_symbolic(True)
    model.freeze_backbone(False)
    assert all(p.requires_grad for p in model.backbone.parameters())
    assert not any(p.requires_grad for p in model.gnn.parameters())

    # restore
    model.freeze_symbolic(False)


def test_generate_returns_new_tokens(model):
    input_ids, mask = make_prompt_batch(B=2, P=10)
    gen = model.generate(input_ids, mask, max_length=6, do_sample=False)
    assert gen.shape[0] == 2
    assert 1 <= gen.shape[1] <= 6


def test_stage_trainers_run_one_step(model):
    """The existing stage trainers must work verbatim on the causal model."""
    from torch.optim import AdamW
    from training.trainers import Stage2_Symbolic_Trainer, Stage3_Decoder_Trainer

    tax = get_default_taxonomy()
    vocab = tax.vocab()
    collator = CausalCognitiveCollator(
        tokenizer=FakeTokenizer(),
        concept_map=vocab["concepts"],
        relation_map=vocab["relations"],
        concept_to_entity_type_map=vocab["concept_to_entity_type"],
        entity_type_map=vocab["entity_types"],
        include_responses=True,
    )
    text = "Marie Curie worked at the University of Paris in 1903 ."
    sample = {
        "text": text,
        "entities": ["Marie Curie", "University of Paris", "1903"],
        "entity_spans": [
            [text.index("Marie Curie"), text.index("Marie Curie") + 11],
            [text.index("University of Paris"), text.index("University of Paris") + 19],
            [text.index("1903"), text.index("1903") + 4],
        ],
        "entity_types": ["person", "organization", "temporal"],
        "concepts": [["scientist"], ["university"], ["year"]],
        "relations": [[0, 1, "works_for"]],
        "should_respond": 1,
        "response": "She won the Nobel Prize .",
    }
    abstain = dict(sample, should_respond=0, response="")
    batch = collator([sample, abstain])

    # Stage 1: symbolic step with backbone frozen
    model.zero_grad(set_to_none=True)  # clear grads left by earlier tests
    model.freeze_backbone(True)
    model.freeze_symbolic(False)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    trainer = Stage2_Symbolic_Trainer(model, opt, device="cpu")
    loss1 = trainer.train_step(batch)
    assert loss1 > 0 and torch.isfinite(torch.tensor(loss1))
    assert all(p.grad is None for p in model.backbone.parameters())

    # Stage 2: decoder step with heads frozen
    model.freeze_symbolic(True)
    model.freeze_backbone(False)
    for p in model.injector.parameters():
        p.requires_grad = True
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    trainer = Stage3_Decoder_Trainer(model, opt, device="cpu")
    loss2 = trainer.train_step(batch)
    assert loss2 > 0 and torch.isfinite(torch.tensor(loss2))

    model.freeze_symbolic(False)


def test_causal_collator_targets():
    tax = get_default_taxonomy()
    vocab = tax.vocab()
    collator = CausalCognitiveCollator(
        tokenizer=FakeTokenizer(),
        concept_map=vocab["concepts"],
        relation_map=vocab["relations"],
        include_responses=True,
    )
    batch = collator([
        {"text": "Hello world .", "entities": [], "concepts": [],
         "relations": [], "should_respond": 1, "response": "Hi there ."},
        {"text": "Hello world .", "entities": [], "concepts": [],
         "relations": [], "should_respond": 0},
    ])

    labels = batch["decoder_labels"]
    inputs = batch["decoder_input_ids"]
    eos = FakeTokenizer.eos_token_id

    # Unshifted: inputs equal labels wherever labels are real
    real = labels != -100
    assert torch.equal(inputs[real], labels[real])
    # Respond row ends with EOS
    row0 = labels[0][labels[0] != -100]
    assert row0[-1].item() == eos
    assert len(row0) == 4  # "Hi there ." = 3 tokens + EOS
    # Abstain row is exactly [EOS]
    row1 = labels[1][labels[1] != -100]
    assert row1.tolist() == [eos]


def test_causal_collator_trace_samples():
    """Trace samples flow through inherited flattening + causal targets."""
    from data.curate.trace_annotator import TraceAnnotator

    trace = [
        {"role": "user", "content": "Check src/app/main.py please ."},
        {"role": "assistant", "content": "It looks fine ."},
    ]
    sample = TraceAnnotator().annotate_trace(trace)

    tax = get_default_taxonomy()
    vocab = tax.vocab()
    collator = CausalCognitiveCollator(
        tokenizer=FakeTokenizer(),
        concept_map=vocab["concepts"],
        relation_map=vocab["relations"],
        concept_to_entity_type_map=vocab["concept_to_entity_type"],
        entity_type_map=vocab["entity_types"],
        include_responses=True,
        chat_mode=True,
        chat_template=ChatTemplate(),
    )
    batch = collator([sample])
    assert batch["should_respond"][0].item() == 1
    labels = batch["decoder_labels"][0]
    assert (labels != -100).sum() > 1  # response tokens + EOS
    types = batch["entity_type_labels"][0]
    assert (types == tax.entity_type_id("digital_artifact")).any()


def test_adapter_zero_layers_is_identity():
    from model.extraction_adapter import ExtractionAdapter

    adapter = ExtractionAdapter(HIDDEN, n_layers=0)
    h = torch.randn(2, 5, HIDDEN)
    mask = torch.ones(2, 5, dtype=torch.long)
    assert torch.equal(adapter(h, mask), h)
