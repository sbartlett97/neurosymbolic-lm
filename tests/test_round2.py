"""Tests for gradient accumulation, DDP, soft-logic type constraints, and
the encoder-decoder path under bf16 autocast."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")

from model.logic import SoftLogicConstraints  # noqa: E402
from tests.test_collator import make_collator, sample  # noqa: E402
from tests.test_training_readiness import _gold_batch, _tiny_model  # noqa: E402
import training.trainers as T  # noqa: E402

REPO = Path(__file__).parent.parent


# --- gradient accumulation ----------------------------------------------------

def _no_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    return model


def _one_update(accum: int):
    model = _no_dropout(_tiny_model())
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0)
    trainer = T.Stage4_Joint_Trainer(model, opt, grad_clip_norm=1e9)
    half = _gold_batch()
    if accum == 1:
        full = {k: (torch.cat([v, v]) if isinstance(v, torch.Tensor) else v + v)
                for k, v in half.items()}
        trainer.train_step(full)
    else:
        trainer.train_step(half, step=False, loss_scale=0.5)
        assert not trainer.stepped
        trainer.train_step(half, step=True, loss_scale=0.5)
    assert trainer.stepped
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def test_grad_accumulation_matches_full_batch():
    a, b = _one_update(1), _one_update(2)
    worst = max((a[n] - b[n]).abs().max().item() for n in a)
    assert worst < 1e-5, worst


def test_new_trainer_clears_stale_grads():
    model = _tiny_model()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    T.Stage3_Decoder_Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1))
    assert all(p.grad is None for p in model.parameters())


# --- soft-logic type constraints -------------------------------------------------

def _constraints():
    # types: 0 none, 1 person, 2 location; relations: 0 none, 1 born_in
    sl = SoftLogicConstraints(n_entity_types=3, n_relations=2)
    allowed = torch.ones(3, 3, 2, dtype=torch.bool)
    allowed[1, 1, 1] = False  # person born_in person
    allowed[2, 1, 1] = allowed[2, 2, 1] = False
    sl.set_type_constraints(allowed)
    return sl


def _types(*rows):
    return torch.tensor([rows], dtype=torch.float)


def test_type_constraint_penalizes_disallowed_pair_only():
    sl = _constraints()
    rel = torch.zeros(1, 2, 2, 2)
    rel[..., 1] = 5.0  # confident born_in everywhere
    person_person = sl.type_constraint_loss(_types([0, 1, 0], [0, 1, 0]), rel)
    # Only the location->person direction is disallowed for person/location
    person_loc = sl.type_constraint_loss(_types([0, 1, 0], [0, 0, 1]), rel)
    assert person_person > 0.9
    assert 0.4 < person_loc < 0.6
    none = sl.type_constraint_loss(_types([1, 0, 0], [1, 0, 0]), rel)
    assert none.item() == 0.0


def test_type_constraint_masks_nodes_and_detaches_types():
    sl = _constraints()
    type_logits = torch.zeros(1, 2, 3, requires_grad=True)
    rel = torch.zeros(1, 2, 2, 2, requires_grad=True)
    loss, _ = sl(torch.softmax(type_logits, -1), rel,
                 node_mask=torch.tensor([[True, True]]))
    loss.backward()
    assert type_logits.grad is None or type_logits.grad.abs().sum() == 0
    assert rel.grad.abs().sum() > 0
    masked, _ = sl(torch.softmax(type_logits, -1), rel,
                   node_mask=torch.tensor([[True, False]]))
    assert masked.item() == 0.0


def test_type_constraints_not_in_state_dict():
    assert "type_allowed" not in _constraints().state_dict()


def test_taxonomy_constraints_built_from_maps():
    from data.curate.taxonomy import get_default_taxonomy
    from train_causal import taxonomy_type_constraints

    vocab = get_default_taxonomy().vocab()
    maps = {"relation_map": vocab["relations"], "entity_type_map": vocab["entity_types"]}
    E = max(vocab["entity_types"].values()) + 1
    R = max(vocab["relations"].values()) + 5
    allowed = taxonomy_type_constraints(maps, E, R)
    p, loc = vocab["entity_types"]["person"], vocab["entity_types"]["location"]
    born = vocab["relations"]["born_in"]
    assert allowed[p, loc, born] and not allowed[p, p, born]
    assert allowed[:, :, R - 1].all()  # headroom stays unconstrained


# --- encoder-decoder path ------------------------------------------------------

def test_encoder_decoder_trainers_under_bf16_autocast(tmp_path):
    from transformers import T5Config, T5ForConditionalGeneration
    from model.neurosymbolic import NeuroSymbolicLM

    T5ForConditionalGeneration(T5Config(
        vocab_size=1100, d_model=32, d_kv=8, d_ff=64, num_layers=1,
        num_heads=4, decoder_start_token_id=0,
    )).save_pretrained(tmp_path / "t5")
    model = NeuroSymbolicLM(model_name=str(tmp_path / "t5"), n_entity_types=24,
                            n_relations=64, n_concepts=200, max_nodes=4,
                            concept_dim=32, node_dim=32)
    batch = make_collator(include_responses=True)(
        [sample(should_respond=1, response="yes it is"), sample(should_respond=0)]
    )
    for name in ("Stage2_Symbolic_Trainer", "Stage3_Decoder_Trainer", "Stage4_Joint_Trainer"):
        trainer = getattr(T, name)(model, torch.optim.AdamW(model.parameters()))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = trainer.compute_loss(batch)
        assert torch.isfinite(loss)
        loss.backward()


# --- DDP -------------------------------------------------------------------------

DDP_SKIP_SCRIPT = r'''
import os, sys, torch, torch.distributed as dist
sys.path.insert(0, sys.argv[1])
from torch.nn.parallel import DistributedDataParallel as DDP
from tests.test_training_readiness import _gold_batch, _tiny_model
import training.trainers as T

class NanOnRank1(T.Stage4_Joint_Trainer):
    calls = 0
    def compute_loss(self, batch, model=None):
        loss = super().compute_loss(batch, model)
        NanOnRank1.calls += 1
        if dist.get_rank() == 1 and NanOnRank1.calls == 2:
            return loss * float("nan")
        return loss

dist.init_process_group("gloo")
torch.manual_seed(0)
model = _tiny_model()
ddp = DDP(model, find_unused_parameters=True, broadcast_buffers=False)
tr = NanOnRank1(ddp, torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], 1e-3))
batch = _gold_batch()
for i in range(4):
    tr.train_step(batch, step=(i % 2 == 1), loss_scale=0.5)
sums = torch.tensor([sum(p.double().sum().item() for p in model.parameters())])
gathered = [torch.zeros(1, dtype=torch.float64) for _ in range(2)]
dist.all_gather(gathered, sums.double())
if dist.get_rank() == 0:
    print("CHECKSUMS", gathered[0].item(), gathered[1].item())
dist.destroy_process_group()
'''


def _torchrun(args, timeout=600, cwd=REPO):
    env = dict(os.environ, OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    return subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2",
         "--master_port", "29517"] + args,
        cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout,
    )


def test_ddp_rank_local_nan_skips_on_all_ranks(tmp_path):
    script = tmp_path / "ddp_skip.py"
    script.write_text(DDP_SKIP_SCRIPT)
    r = _torchrun([str(script), str(REPO)])
    assert r.returncode == 0, r.stderr[-3000:]
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("CHECKSUMS")][0]
    a, b = map(float, line.split()[1:])
    assert a == b  # ranks stayed in sync after the rank-1 NaN


def test_ddp_end_to_end_cli(tmp_path, qwen_tokenizer):
    text = "Marie Curie worked at the University of Paris in 1903."
    ents = ["Marie Curie", "University of Paris", "1903"]
    base = {"text": text, "entities": ents,
            "entity_spans": [[text.index(e), text.index(e) + len(e)] for e in ents],
            "entity_types": ["person", "organization", "temporal"],
            "concepts": [["scientist"], ["university"], ["year"]],
            "relations": [[0, 1, "works_for"]]}
    rows = [dict(base, should_respond=i % 2,
                 response="She was a physicist." if i % 2 else "") for i in range(16)]
    data = tmp_path / "d.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "d_val.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:4]) + "\n")
    out = tmp_path / "ck"
    r = _torchrun([
        "train_causal.py", "--preset", "causal-testing", "--dataset", str(data),
        "--epochs", "1", "--batch-size", "2", "--grad-accum", "2",
        "--gradient-checkpointing", "--num-workers", "0", "--device", "cpu",
        "--output-dir", str(out), "--log-dir", str(tmp_path / "runs"),
    ])
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-3000:])
    assert "Distributed: 2 processes" in r.stdout
    for name in ("stage_symbolic.pt", "stage_decoder.pt", "stage_joint.pt",
                 "final_causal_model.pt"):
        assert (out / name).exists()
    from train_causal import load_causal_checkpoint
    load_causal_checkpoint(str(out / "final_causal_model.pt"))


@pytest.fixture(scope="module")
def qwen_tokenizer():
    from tests.test_training_readiness import QWEN
    transformers = pytest.importorskip("transformers")
    try:
        transformers.AutoTokenizer.from_pretrained(QWEN)
    except Exception as e:
        pytest.skip(f"Qwen tokenizer unavailable: {e}")


# --- QA fail-fast -------------------------------------------------------------

def test_qa_fail_fast_stops_abstain_only_runs():
    from data.curate import gliner_annotator as G

    class Rejecting:
        def generate_chat_batch(self, prompts, system=None, json_schema=None):
            return [""] * len(prompts)

    ann = G.GLiNER2Annotator.__new__(G.GLiNER2Annotator)
    ann.response_backend = Rejecting()
    ann._stats = {"qa_generated": 0, "qa_failed": 0}
    items = [("Paris is in France.", ["Paris"], [])] * 10
    with pytest.raises(RuntimeError, match="should_respond=0"):
        for _ in range(G.QA_FAIL_FAST_MIN_ATTEMPTS // 10 + 1):
            ann._generate_qa_batch(items)


def test_api_first_rejection_is_printed_once(capsys):
    from data.curate.llm_annotator import AnthropicBackend

    class BadRequest(Exception):
        status_code = 400

    class Client:
        class messages:
            @staticmethod
            def create(**kw):
                raise BadRequest("model: claude-nope not found")

    b = AnthropicBackend(client=Client(), sleep=lambda s: None)
    b.generate_chat_batch(["a JSON q", "another JSON q"])
    out = capsys.readouterr().out
    assert out.count("request rejected") == 1 and "claude-nope" in out


def test_injection_gate_gets_its_own_lr():
    from train_causal import make_optimizer

    model = _tiny_model()
    opt = make_optimizer(model, lr=1e-5, weight_decay=0.01, gate_lr=1e-3)
    gate_groups = [g for g in opt.param_groups
                   if any(p is model.injector.gate for p in g["params"])]
    assert len(gate_groups) == 1
    g = gate_groups[0]
    assert g["params"] == [model.injector.gate]
    assert g["lr"] == 1e-3 and g["weight_decay"] == 0.0
    # The step log reads group 0: it keeps the stage LR
    assert opt.param_groups[0]["lr"] == 1e-5

    # Default: the gate follows the stage LR
    opt = make_optimizer(model, lr=1e-5, weight_decay=0.01)
    assert all(g["lr"] == 1e-5 for g in opt.param_groups)
