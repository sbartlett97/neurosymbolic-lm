"""Regression tests for the training-readiness fixes (2026-10 audit).

Covers label conventions, span handling under truncation, pad == EOS
targets, relation negatives, node scoring, trace rows outside chat mode,
toy-augmentation scoping, and a CLI end-to-end run of train_causal.

Tests that need the real Qwen3 tokenizer skip when it is not cached and
the network is unavailable.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")

from data.chat_template import ChatTemplate  # noqa: E402
from data.collator_causal import CausalCognitiveCollator  # noqa: E402
from data.dataset import ToyCognitiveDataset  # noqa: E402
from model.entity import entity_node_scores  # noqa: E402
from training.trainers import (  # noqa: E402
    create_entity_token_labels,
    entity_loss,
    relation_loss,
)
from tests.test_collator import FakeTokenizer, make_collator, sample  # noqa: E402

QWEN = "Qwen/Qwen3-0.6B-Base"


@pytest.fixture(scope="module")
def qwen_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(QWEN)
    except Exception as e:  # offline and not cached
        pytest.skip(f"Qwen tokenizer unavailable: {e}")
    tok.padding_side = "right"
    ChatTemplate.add_special_tokens(tok)
    return tok


def causal_collator(tok, **kwargs):
    return CausalCognitiveCollator(
        tokenizer=tok,
        concept_map={"scientist": 1},
        relation_map={"works_for": 1},
        entity_type_map={"person": 1, "organization": 2, "temporal": 3},
        chat_template=ChatTemplate(),
        **kwargs,
    )


# --- labels ----------------------------------------------------------------

def test_entity_labels_train_none_class_and_ignore_padding():
    collator = make_collator()
    short = sample(text="Marie Curie .", entities=["Marie Curie"],
                   entity_spans=[[0, 11]], entity_types=["person"],
                   concepts=[["scientist"]], relations=[])
    batch = collator([sample(), short])
    B, L = batch["input_ids"].shape
    labels = create_entity_token_labels(batch, B, L, 24, torch.device("cpu"))

    # Real non-entity tokens are class 0, padding is ignored
    assert labels[0, 3].item() == 0  # "at"
    assert (labels[1][batch["attention_mask"][1] == 0] == -100).all()
    assert labels[0, 0].item() > 0  # "Marie" is an entity


def test_entity_loss_is_finite_without_valid_labels():
    logits = torch.randn(2, 5, 8, requires_grad=True)
    labels = torch.full((2, 5), -100)
    loss = entity_loss(logits, labels, none_weight=0.1)
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()  # still connected to the graph


def test_relation_loss_uses_directed_pairs_and_negatives():
    N, R = 4, 5
    logits = torch.zeros(1, N, N, R, requires_grad=True)
    spans = [[(0, 0), (2, 2), (-1, -1)]]  # entity 2 truncated away
    loss = relation_loss(logits, [[(1, 0, 3)]], spans, negative_weight=0.1)
    loss.backward()
    grad = logits.grad[0]
    # Only the two present entities form pairs, both directions
    assert grad[1, 0].abs().sum() > 0 and grad[0, 1].abs().sum() > 0
    assert grad[0, 2].abs().sum() == 0 and grad[2, 0].abs().sum() == 0
    # Gold direction targets class 3, reverse direction targets "none"
    assert grad[1, 0, 3] < 0 and grad[0, 1, 0] < 0


def test_node_scores_exclude_none_class_and_padding():
    logits = torch.tensor([[[9.0, 0.0], [0.0, 5.0], [0.0, 9.0]]])
    mask = torch.tensor([[1, 1, 0]])
    scores = entity_node_scores(logits, mask)
    # A confident "none" token must score below a likely entity
    assert scores[0, 1] > scores[0, 0]
    assert scores[0, 2] == float("-inf")


# --- collator ----------------------------------------------------------------

def test_trace_row_outside_chat_mode_does_not_crash():
    collator = make_collator(chat_template=ChatTemplate())
    trace = {
        "messages": [{"role": "user", "content": "Marie Curie ?"},
                     {"role": "assistant", "content": "A scientist ."}],
        "message_annotations": [{"message_idx": 0, "entities": ["Marie Curie"],
                                 "entity_spans": [[0, 11]],
                                 "entity_types": ["person"], "concepts": [["scientist"]],
                                 "relations": []}],
        "should_respond": 1,
    }
    batch = collator([sample(), trace])
    assert batch["entity_token_spans"][1][0][0] >= 0


def test_eos_label_kept_when_pad_equals_eos():
    class PadIsEos(FakeTokenizer):
        pad_token_id = 1
        eos_token_id = 1

    collator = CausalCognitiveCollator(
        tokenizer=PadIsEos(), concept_map={"x": 1}, relation_map={"r": 1},
        include_responses=True,
    )
    batch = collator([
        sample(should_respond=1, response="yes it is"),
        sample(should_respond=0),
    ])
    labels = batch["decoder_labels"]
    assert labels[0].tolist()[:4][-1] == 1  # EOS after "yes it is"
    assert labels[1, 0].item() == 1  # abstention row = [EOS]
    assert (labels[1, 1:] == -100).all()


def test_spans_past_truncation_are_dropped_not_clamped(qwen_tokenizer):
    text = "Marie Curie was born in Warsaw. " + "filler " * 200 + "Paris is late."
    s = {"text": text, "entities": ["Marie Curie", "Paris"],
         "entity_spans": [[0, 11], [text.index("Paris"), text.index("Paris") + 5]],
         "entity_types": ["person", "organization"], "concepts": [[], []],
         "relations": [], "should_respond": 0}
    collator = causal_collator(qwen_tokenizer, max_length=64)
    batch = collator([s])
    spans = batch["entity_token_spans"][0]
    # Causal collator truncates the start: the late entity survives
    assert spans[0] == (-1, -1)
    ids = batch["input_ids"][0]
    start, end = spans[1]
    assert qwen_tokenizer.decode(ids[start:end + 1]).strip() == "Paris"


def test_offsets_spans_match_entities(qwen_tokenizer):
    text = "Marie Curie worked at the University of Paris in 1903."
    ents = ["Marie Curie", "University of Paris", "1903"]
    s = {"text": text, "entities": ents,
         "entity_spans": [[text.index(e), text.index(e) + len(e)] for e in ents],
         "entity_types": ["person", "organization", "temporal"],
         "concepts": [[], [], []], "relations": [], "should_respond": 0}
    batch = causal_collator(qwen_tokenizer)([s])
    ids = batch["input_ids"][0]
    for ent, (a, b) in zip(ents, batch["entity_token_spans"][0]):
        assert qwen_tokenizer.decode(ids[a:b + 1]).strip() == ent


# --- dataset -----------------------------------------------------------------

def test_jsonl_rows_are_not_rewritten(tmp_path):
    path = tmp_path / "qa.jsonl"
    row = {"text": "Doc text. Tell me who won.", "entities": [], "concepts": [],
           "relations": [], "should_respond": 1, "response": "Curie."}
    path.write_text(json.dumps(row) + "\n")
    ds = ToyCognitiveDataset(str(path))
    assert len(ds) == 1
    assert ds[0]["text"] == row["text"] and ds[0]["should_respond"] == 1


# --- CLI end to end ------------------------------------------------------------

def test_train_causal_cli_end_to_end(tmp_path, qwen_tokenizer, capsys):
    import train_causal

    text = "Marie Curie worked at the University of Paris in 1903."
    ents = ["Marie Curie", "University of Paris", "1903"]
    base = {"text": text, "entities": ents,
            "entity_spans": [[text.index(e), text.index(e) + len(e)] for e in ents],
            "entity_types": ["person", "organization", "temporal"],
            "concepts": [["scientist"], ["university"], ["year"]],
            "relations": [[0, 1, "works_for"]]}
    corpus = [dict(base, should_respond=0) for _ in range(8)]
    qa = [dict(base, should_respond=i % 2,
               response="She was a physicist." if i % 2 else "")
          for i in range(8)]
    trace = {"messages": [{"role": "user", "content": "Who was Marie Curie?"},
                          {"role": "assistant", "content": "A physicist."}],
             "should_respond": 1}

    def write(name, rows):
        (tmp_path / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    # Stage-1 corpus is abstain-only (as curated); a trace row is mixed in
    write("d.jsonl", corpus + [trace])
    write("d_val.jsonl", corpus[:4])
    write("s2.jsonl", qa)
    write("s2_val.jsonl", qa[:4])
    out = tmp_path / "ck"
    train_causal.main([
        "--preset", "causal-testing", "--dataset", str(tmp_path / "d.jsonl"),
        "--stage2-dataset", str(tmp_path / "s2.jsonl"),
        "--epochs", "1", "--batch-size", "4", "--num-workers", "0",
        "--device", "cpu", "--output-dir", str(out),
        "--log-dir", str(tmp_path / "runs"),
    ])
    log = capsys.readouterr().out
    # Each stage validates on data like its own
    assert "Validation for d.jsonl" in log and "Validation for s2.jsonl" in log
    assert "respond_rate" in log
    for name in ("stage_symbolic.pt", "stage_decoder.pt", "stage_joint.pt",
                 "final_causal_model.pt"):
        assert (out / name).exists()
    model, tok, ckpt = train_causal.load_causal_checkpoint(
        str(out / "final_causal_model.pt")
    )
    assert "<assistant>" in tok.get_vocab()
    assert ckpt["entity_type_map"]["person"] >= 1


# --- mixed precision ------------------------------------------------------------

def _tiny_model(backbone_dtype=torch.float32, adapter_layers=1):
    from model.neurosymbolic_causal import NeuroSymbolicCausalLM, build_tiny_backbone

    torch.manual_seed(0)
    return NeuroSymbolicCausalLM(
        backbone=build_tiny_backbone(vocab_size=2048).to(backbone_dtype),
        n_entity_types=24, n_relations=64, n_concepts=32, concept_dim=32,
        node_dim=32, max_nodes=4, adapter_layers=adapter_layers, adapter_heads=2,
    )


def _gold_batch():
    from data.collator_causal import CausalCognitiveCollator
    from tests.test_collator import FakeTokenizer

    tax_collator = make_collator()
    collator = CausalCognitiveCollator(
        tokenizer=FakeTokenizer(), concept_map=tax_collator.concept_map,
        relation_map=tax_collator.relation_map, include_responses=True,
        entity_type_map=tax_collator.entity_type_map,
    )
    return collator([sample(should_respond=1, response="yes it is"),
                     sample(should_respond=0)])


@pytest.mark.parametrize("adapter_layers", [0, 1])
def test_bf16_base_without_autocast(adapter_layers):
    """LoRA keeps a bf16 base with fp32 heads; eval and generate run
    without autocast and must not hit a dtype mismatch (adapter_layers=0
    is the pure-causal ablation, with no adapter cast)."""
    model = _tiny_model(torch.bfloat16, adapter_layers).eval()
    batch = _gold_batch()
    out = model(batch["input_ids"], batch["attention_mask"],
                spans=batch["entity_token_spans"], y_ids=batch["decoder_input_ids"])
    assert torch.isfinite(out["token_ent_logits"]).all()
    assert torch.isfinite(out["logits"].float()).all()
    gen = model.generate(batch["input_ids"], batch["attention_mask"], max_length=3,
                         do_sample=False, pad_token_id=0)
    assert gen.shape[0] == 2


@pytest.mark.parametrize("trainer_name", [
    "Stage2_Symbolic_Trainer", "Stage3_Decoder_Trainer", "Stage4_Joint_Trainer",
])
def test_trainers_under_bf16_autocast(trainer_name):
    import training.trainers as T

    model = _tiny_model()
    trainer = getattr(T, trainer_name)(model, torch.optim.AdamW(model.parameters()))
    batch = _gold_batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = trainer.compute_loss(batch)
        gen = model.generate(batch["input_ids"], batch["attention_mask"],
                             max_length=2, do_sample=False, pad_token_id=0)
    assert torch.isfinite(loss)
    loss.backward()
    assert gen.shape[0] == 2
