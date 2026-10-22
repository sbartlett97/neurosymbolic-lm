"""v2 run pieces: --init-from (reuse stage-1 heads on a LoRA backbone) and
data/mix_stage2.py (unanswerable questions + general text)."""

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")

from data import mix_stage2  # noqa: E402

TEXT = "Marie Curie worked at the University of Paris in 1903."
ENTS = ["Marie Curie", "University of Paris", "1903"]


def base_row(text=TEXT):
    return {"text": text, "entities": ENTS,
            "entity_spans": [[text.index(e), text.index(e) + len(e)] for e in ENTS],
            "entity_types": ["person", "organization", "temporal"],
            "concepts": [["scientist"], ["university"], ["year"]],
            "relations": [[0, 1, "works_for"]]}


def write(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


# --- --init-from ----------------------------------------------------------------

@pytest.fixture(scope="module")
def qwen_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    except OSError as e:
        pytest.skip(f"tokenizer unavailable: {e}")


def test_init_from_symbolic_into_lora_model(tmp_path, qwen_tokenizer, capsys):
    """v2 path: stage 1 on the full model, then a LoRA run that reuses its
    heads via --init-from and trains only decoder + joint."""
    import train_causal

    corpus = [dict(base_row(), should_respond=0) for _ in range(8)]
    qa = [dict(base_row(), should_respond=i % 2,
               response="She was a physicist." if i % 2 else "")
          for i in range(8)]
    write(tmp_path / "d.jsonl", corpus)
    write(tmp_path / "d_val.jsonl", corpus[:4])
    write(tmp_path / "s2.jsonl", qa)
    write(tmp_path / "s2_val.jsonl", qa[:4])
    common = ["--dataset", str(tmp_path / "d.jsonl"),
              "--stage2-dataset", str(tmp_path / "s2.jsonl"),
              "--epochs", "1", "--batch-size", "4", "--num-workers", "0",
              "--device", "cpu", "--log-dir", str(tmp_path / "runs")]

    v1 = tmp_path / "v1"
    train_causal.main(["--preset", "causal-testing", "--stages", "symbolic",
                       "--output-dir", str(v1)] + common)
    v1_ckpt = torch.load(v1 / "stage_symbolic.pt", weights_only=False)

    v2 = tmp_path / "v2"
    capsys.readouterr()
    train_causal.main(["--preset", "causal-testing-lora",
                       "--init-from", str(v1 / "stage_symbolic.pt"),
                       "--stages", "decoder", "--output-dir", str(v2)] + common)
    log = capsys.readouterr().out
    assert "non-backbone tensors" in log
    assert "backbone differs" not in log

    v2_ckpt = torch.load(v2 / "stage_decoder.pt", weights_only=False)
    assert v2_ckpt["config"]["use_lora"] is True
    assert v2_ckpt["relation_map"] == v1_ckpt["relation_map"]
    # The decoder stage freezes the symbolic heads: still the v1 weights
    for k, v in v1_ckpt["model_state_dict"].items():
        if k.startswith(("token_ent.", "rel_scorer.", "adapter.")):
            assert torch.equal(v, v2_ckpt["model_state_dict"][k]), k
    assert any("lora_" in k for k in v2_ckpt["model_state_dict"])


def test_init_from_refuses_other_head_sizes(qwen_tokenizer):
    import train_causal
    from config import CAUSAL_MODEL_PRESETS

    config = CAUSAL_MODEL_PRESETS["causal-testing"]()
    backbone, tok = train_causal.build_backbone_and_tokenizer(config, "cpu")
    model = train_causal.build_model(config, backbone, tok)
    other = CAUSAL_MODEL_PRESETS["causal-testing"]()
    other.n_relations = 16
    b2, t2 = train_causal.build_backbone_and_tokenizer(other, "cpu")
    small = train_causal.build_model(other, b2, t2)
    with pytest.raises((SystemExit, RuntimeError)):
        train_causal.load_init_weights(
            model, {"model_state_dict": small.state_dict()})


def test_init_config_takes_heads_from_ckpt_and_lora_from_preset():
    import train_causal
    from dataclasses import asdict
    from config import CAUSAL_MODEL_PRESETS

    ckpt = CAUSAL_MODEL_PRESETS["causal-testing"]()
    ckpt.n_relations = 77
    config = train_causal.init_config(CAUSAL_MODEL_PRESETS["causal-testing-lora"](),
                                      asdict(ckpt))
    assert config.n_relations == 77 and config.use_lora is True
    with pytest.raises(SystemExit):
        train_causal.init_config(CAUSAL_MODEL_PRESETS["qwen3-0.6b-lora"](),
                                 asdict(ckpt))


# --- data/mix_stage2.py -------------------------------------------------------

def long_doc(i):
    sents = [f"Document {i} sentence {j} talks about Marie Curie and Paris."
             for j in range(40)]
    return " ".join(sents)


def test_general_row_keeps_only_prompt_labels():
    text = long_doc(0)
    ents = ["Marie Curie", "Paris"]
    first = [text.index(e) for e in ents]
    last = text.rindex("Marie Curie")
    row = {"text": text, "entities": ents + ["Marie Curie"],
           "entity_spans": [[first[0], first[0] + 11], [first[1], first[1] + 5],
                            [last, last + 11]],
           "entity_types": ["person", "location", "person"],
           "concepts": [["scientist"], ["city"], ["scientist"]],
           "relations": [[0, 1, "located_in"], [2, 1, "located_in"]],
           "should_respond": 0, "response": ""}
    g = mix_stage2.make_general(row, random.Random(0), 275, 550)
    assert g is not None and g["should_respond"] == 1
    assert 275 // 2 <= len(g["response"]) <= 550
    assert text.startswith(g["text"])
    # The last entity sits after the cut: dropped, with its relation
    assert len(g["entities"]) == 2
    assert g["relations"] == [[0, 1, "located_in"]]
    for (s, e), name in zip(g["entity_spans"], g["entities"]):
        assert g["text"][s:e] == name


def test_unanswerable_question_from_another_doc_without_its_answer():
    respond = []
    for i, (q, a) in enumerate([("Who led lab A?", "Alice"),
                                ("Who led lab B?", "Marie Curie")]):
        r = base_row(f"Doc {i}. " + TEXT)
        r.update(text=r["text"] + "\n\n" + q, should_respond=1, response=a)
        respond.append(r)
    abstain = [dict(base_row(), should_respond=0, response="")]
    out = mix_stage2.make_unanswerable(abstain, respond, random.Random(0))
    assert len(out) == 1
    doc, question = mix_stage2.split_question(out[0]["text"])
    assert doc == TEXT
    # "Marie Curie" occurs in the document, so only the Alice question fits
    assert question == "Who led lab A?"
    assert out[0]["should_respond"] == 0 and out[0]["response"] == ""


def test_mix_cli_writes_splits_and_vocab(tmp_path):
    corpus = []
    for i in range(30):
        text = long_doc(i)
        s = text.index("Marie Curie")
        corpus.append({"text": text, "entities": ["Marie Curie"],
                       "entity_spans": [[s, s + 11]], "entity_types": ["person"],
                       "concepts": [["scientist"]], "relations": [],
                       "should_respond": 0, "response": ""})
    qa = []
    for i in range(6):
        r = dict(corpus[i])
        r.update(text=r["text"] + "\n\nWhat year was it?", should_respond=1,
                 response=f"In 19{i:02d}.")
        qa.append(r)
    qa += [dict(corpus[i]) for i in range(6, 9)]
    write(tmp_path / "c.jsonl", corpus[:20])
    write(tmp_path / "c_val.jsonl", corpus[20:])
    write(tmp_path / "q.jsonl", qa)
    write(tmp_path / "q_val.jsonl", qa[:2])
    (tmp_path / "q_vocab.json").write_text("{}")
    out = tmp_path / "v2" / "mix.jsonl"
    mix_stage2.main(["--corpus", str(tmp_path / "c.jsonl"), "--qa",
                     str(tmp_path / "q.jsonl"), "--out", str(out),
                     "--general-rows", "5"])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    kinds = {"respond": 0, "unanswerable": 0, "general": 0}
    for r in rows:
        if r["should_respond"] == 0:
            kinds["unanswerable"] += 1
            assert "\n\n" in r["text"]  # never a document without a question
        elif r["text"].endswith("?"):
            kinds["respond"] += 1
        else:
            kinds["general"] += 1
    assert kinds == {"respond": 6, "unanswerable": 3, "general": 5}
    # General rows never come from documents the QA set uses
    qa_docs = {r["text"].rpartition("\n\n")[0] or r["text"] for r in qa}
    for r in rows:
        if r["should_respond"] == 1 and not r["text"].endswith("?"):
            assert not any(d.startswith(r["text"]) for d in qa_docs)
    assert (tmp_path / "v2" / "mix_val.jsonl").exists()
    assert (tmp_path / "v2" / "mix_vocab.json").exists()


# --- v3 negatives: same document, entity swap --------------------------------

DOC = ("Marie Curie was born in Warsaw. She worked at the University of Paris. "
       "In 1903 she won the Nobel Prize in Physics. She died in 1934.")


def qa_row(doc=DOC, question="When did she win the Nobel Prize?",
           answer="In 1903."):
    ents = ["Marie Curie", "Warsaw", "University of Paris", "1903", "1934"]
    types = ["person", "location", "organization", "temporal", "temporal"]
    rels = [[0, 1, "born_in"], [0, 3, "won_in"], [0, 4, "died_in"]]
    if doc != DOC:  # another document: no labels
        ents, types, rels = [], [], []
    return {"text": doc + "\n\n" + question, "entities": ents,
            "entity_spans": [[doc.index(e), doc.index(e) + len(e)] for e in ents],
            "entity_types": types, "concepts": [[t] for t in types],
            "relations": rels,
            "should_respond": 1, "response": answer, "kind": "respond"}


def test_same_doc_deletes_answer_and_shifts_spans():
    neg = mix_stage2.make_same_doc(qa_row())
    assert neg is not None and neg["kind"] == "unanswerable_same"
    assert neg["should_respond"] == 0 and neg["response"] == ""
    doc, question = mix_stage2.split_question(neg["text"])
    assert question == "When did she win the Nobel Prize?"
    assert "1903" not in doc and "Nobel" not in doc
    assert "1934" in doc  # the sentence after the cut stays
    # "1903" sat in the cut: dropped with its relation. Later spans moved
    assert "1903" not in neg["entities"]
    for (s, e), name in zip(neg["entity_spans"], neg["entities"]):
        assert doc[s:e] == name
    names = neg["entities"]
    assert [[names[h], names[t], r] for h, t, r in neg["relations"]] == [
        ["Marie Curie", "Warsaw", "born_in"], ["Marie Curie", "1934", "died_in"]]


def test_same_doc_skips_unsafe_rows():
    # The answer occurs in 4 sentences: too many to delete
    doc = " ".join(f"Line {i} names Warsaw." for i in range(4))
    assert mix_stage2.make_same_doc(
        qa_row(doc, "Which city is named?", "Warsaw.")) is None
    # One plain word: too weak to find the answer sentence
    assert mix_stage2.make_same_doc(
        qa_row(question="What did she do?", answer="Research.")) is None
    # No sentence holds the answer
    assert mix_stage2.make_same_doc(
        qa_row(question="Who was her husband?", answer="Pierre Curie.")) is None


def test_swap_changes_a_name_that_the_document_lacks():
    row = qa_row(question="Where was Marie Curie born?", answer="Warsaw.")
    by_type = {"person": ["Marie Curie", "Ada Lovelace"]}
    neg = mix_stage2.make_swap(row, by_type, random.Random(0))
    assert neg is not None and neg["kind"] == "unanswerable_swap"
    doc, question = mix_stage2.split_question(neg["text"])
    assert doc == DOC and question == "Where was Ada Lovelace born?"
    assert neg["should_respond"] == 0 and neg["response"] == ""
    # A date is not a name: no swap
    row = qa_row(question="What happened in 1903?", answer="The Nobel Prize.")
    assert mix_stage2.make_swap(row, {"temporal": ["1850"]}, random.Random(0)) is None


def test_build_split_counts_each_kind():
    respond = [qa_row(), qa_row(question="Where was Marie Curie born?",
                                answer="She was born in Warsaw.")]
    abstain = [dict(base_row(), should_respond=0, response="")]
    rows, stats = mix_stage2.build_split(
        [], respond + abstain, 0, random.Random(0), 275, 550,
        n_cross=-1, n_same=1, n_swap=1)
    kinds = sorted(r["kind"] for r in rows)
    assert stats["unanswerable_same"] == 1
    assert kinds.count("respond") == 2
    assert kinds.count("unanswerable_same") == 1
    # Defaults: no new kinds, so the output stays the v2 set
    rows, stats = mix_stage2.build_split([], respond + abstain, 0,
                                         random.Random(0), 275, 550)
    assert stats["unanswerable_same"] == stats["unanswerable_swap"] == 0


def test_generation_report_scores_each_negative_kind():
    import train_causal

    EOS = 7

    class Model:
        def generate(self, input_ids, attention_mask, use_soft_nodes=True, **kw):
            # Abstain when the prompt token says so (graph on only)
            hit = int(input_ids[0, 0]) == 1 and use_soft_nodes
            return torch.tensor([[EOS if hit else 3]])

    class Tok:
        eos_token_id, pad_token_id = EOS, 0

    def collator(rows):
        return {"input_ids": torch.tensor([[rows[0]["abstains"]]]),
                "attention_mask": torch.ones(1, 1, dtype=torch.long)}

    rows = ([{"kind": "unanswerable_cross", "should_respond": 0, "abstains": 1}] * 4
            + [{"kind": "unanswerable_same", "should_respond": 0, "abstains": a}
               for a in (1, 0, 0, 0)]
            + [{"kind": "respond", "should_respond": 1, "abstains": 0}] * 2)
    m = train_causal.generation_report(Model(), Tok(), collator, rows, "cpu",
                                       max_new_tokens=4, n_show=0)
    assert m["abstain_acc"] == 5 / 8 and m["respond_rate"] == 1.0
    assert m["abstain_acc_unanswerable_cross"] == 1.0
    assert m["abstain_acc_unanswerable_same"] == 0.25
    off = train_causal.generation_report(Model(), Tok(), collator, rows, "cpu",
                                         max_new_tokens=4, n_show=0,
                                         use_soft_nodes=False)
    assert off["abstain_acc_unanswerable_cross"] == 0.0
