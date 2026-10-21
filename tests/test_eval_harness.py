"""The lm-eval wrapper (eval_harness.py) on the tiny offline backbone."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")
pytest.importorskip("lm_eval")

from config import CAUSAL_MODEL_PRESETS  # noqa: E402
from eval_harness import NeuroSymLM  # noqa: E402
from train_causal import build_backbone_and_tokenizer, build_model  # noqa: E402


@pytest.fixture(scope="module")
def parts():
    torch.manual_seed(0)
    config = CAUSAL_MODEL_PRESETS["causal-testing"]()
    try:
        backbone, tokenizer = build_backbone_and_tokenizer(config, "cpu")
    except OSError as e:  # no cached tokenizer and no network
        pytest.skip(f"tokenizer unavailable: {e}")
    return build_model(config, backbone, tokenizer), tokenizer


def make_lm(parts, **kw):
    model, tokenizer = parts
    return NeuroSymLM(model=model, tokenizer=tokenizer, device="cpu", **kw)


def req(*args):
    return SimpleNamespace(args=args)


REQS = [req("The capital of France is", " Paris"),
        req("Water boils at", " 100 degrees Celsius"),
        req("Question: 2 + 2 =", " 4"),
        req("", "An empty context")]


def test_trailing_space_moves_into_continuation(parts):
    lm = make_lm(parts)
    ctx, cont = lm._encode_pair("The answer is ", "Paris")
    assert ctx == lm._encode("The answer is")
    assert ctx + cont == lm._encode("The answer is Paris")


def test_loglikelihood_matches_manual_forward(parts):
    lm = make_lm(parts, batch_size=1)
    model, tokenizer = parts
    (lp, _), = lm.loglikelihood([REQS[0]])

    ctx, cont = lm._encode_pair(*REQS[0].args)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ctx]),
                    attention_mask=torch.ones(1, len(ctx), dtype=torch.long),
                    y_ids=torch.tensor([cont]))
    manual = torch.log_softmax(out["logits"], -1)[0, torch.arange(len(cont)),
                                                  torch.tensor(cont)].sum()
    assert lp == pytest.approx(float(manual), abs=1e-4)
    assert lp < 0


def test_batching_does_not_change_scores(parts):
    one = make_lm(parts, batch_size=1).loglikelihood(REQS)
    many = make_lm(parts, batch_size=4).loglikelihood(REQS)
    for (a, ga), (b, gb) in zip(one, many):
        assert a == pytest.approx(b, abs=1e-3)
        assert ga == gb


def test_greedy_flag_true_for_the_models_own_tokens(parts):
    lm = make_lm(parts)
    ctx = lm._encode("Once upon a time")
    gen = parts[0].generate(torch.tensor([ctx]),
                            torch.ones(1, len(ctx), dtype=torch.long),
                            max_length=3, do_sample=False,
                            eos_token_id=lm.eos_id, pad_token_id=lm.pad_id)
    cont = gen[0].tolist()
    (_, greedy), = lm._score([(ctx, cont)])
    assert greedy


def test_rolling_equals_scoring_the_text_after_eos(parts):
    lm = make_lm(parts)
    text = "A short text that fits in one window."
    (rolling,) = lm.loglikelihood_rolling([req(text)])
    ((direct, _),) = lm._score([([lm.eos_id], lm._encode(text))])
    assert rolling == pytest.approx(direct, abs=1e-4)


def test_rolling_windows_cover_long_text(parts):
    lm = make_lm(parts, max_length=8)
    text = " ".join(["word"] * 30)
    (rolling,) = lm.loglikelihood_rolling([req(text)])
    assert rolling < 0


def test_generate_until_cuts_at_stop_string(parts):
    lm = make_lm(parts)
    (full,) = lm.generate_until([req("Hello", {"max_gen_toks": 8})])
    assert isinstance(full, str)
    if len(full) > 2:
        stop = full[1:3]
        (cut,) = lm.generate_until(
            [req("Hello", {"until": [stop], "max_gen_toks": 8})])
        assert stop not in cut
        assert full.startswith(cut)


def test_no_soft_nodes_changes_scores(parts):
    on = make_lm(parts, use_soft_nodes=True).loglikelihood(REQS[:1])
    off = make_lm(parts, use_soft_nodes=False).loglikelihood(REQS[:1])
    assert on[0][0] != pytest.approx(off[0][0], abs=1e-6)
