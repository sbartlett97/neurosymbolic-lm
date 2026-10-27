"""A separate read-pass input (graph_input_ids): the graph comes from other
text than the prompt. Both injection modes, on a tiny offline backbone."""

import re
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


def test_eval_modes_and_token_f1():
    from eval_graph_only import mode_row, token_f1
    row = {"text": "Who won?", "graph_text": "Ada won the prize.", "kind": "graph_only_respond",
           "entity_spans": [[0, 3]], "entities": ["Ada"]}
    assert mode_row(row, "graph") == row
    assert "graph_text" not in mode_row(row, "closed_book")
    assert mode_row(row, "closed_book")["text"] == "Who won?"
    passage = mode_row(row, "passage")
    assert passage["text"] == "Ada won the prize.\n\nWho won?" and "graph_text" not in passage
    # The document is the prefix, so the gold spans stay valid in passage mode
    s, e = passage["entity_spans"][0]
    assert passage["text"][s:e] == "Ada"
    assert "graph_text" in row  # the input row stays the same
    assert token_f1("Ada won", "Ada won the prize") == pytest.approx(2 * 1 * 0.5 / 1.5)
    assert token_f1("", "Ada") == 0.0 and token_f1("Bob", "Ada") == 0.0


def test_eval_summary_and_paired_contrasts():
    from eval_graph_only import contrasts, summarize
    records = []
    for i in range(20):
        records.append({"row": i, "kind": "graph_only_respond",
                        "answer_type": "value" if i % 2 else "other",
                        "closed_book": {"key_recall": 0.0, "f1": 0.1, "respond": 1},
                        "graph": {"key_recall": 0.5, "f1": 0.6, "respond": 1},
                        "graph_shuffled": {"key_recall": 0.5, "f1": 0.6, "respond": 1},
                        "passage": {"key_recall": 1.0, "f1": 0.9, "respond": 1}})
    records.append({"row": 20, "kind": "graph_only_unanswerable_same", "answer_type": "other",
                    "graph": {"abstain": 1}, "closed_book": {"abstain": 0}})
    s = summarize(records)
    assert s["graph"]["key_recall"] == 0.5 and s["graph"]["n_respond"] == 20
    assert s["graph"]["f1_value"] == 0.6
    assert s["graph"]["abstain_unanswerable_same"] == 1.0
    assert s["closed_book"]["abstain_unanswerable_same"] == 0.0
    c = contrasts(records)
    book = c["key_recall:graph-closed_book"]
    assert book["n"] == 20 and book["mean"] == 0.5 and book["lo"] == 0.5
    shuf = c["key_recall:graph-graph_shuffled"]
    assert shuf["mean"] == 0.0 and shuf["lo"] <= 0.0  # no evidence of content
    assert c["f1_value:graph-closed_book"]["n"] == 10


def test_word_spans_gives_whole_words_in_text_order():
    from eval_graph_only import word_spans
    text = "Ed Fendell of Mission Control, in 1969."
    # Subword ranges inside "Fendell" and "Control"; "Ed" next to "Fendell"
    assert word_spans(text, [(5, 8), (0, 2), (24, 27)]) == "Ed Fendell; Control"
    assert word_spans(text, [(29, 30), (34, 36)]) == "1969"  # "," alone is no word
    assert word_spans(text, []) == ""
    assert word_spans("BRAT and BRAT", [(0, 4), (9, 13)]) == "BRAT"


def test_widen_grows_each_range_by_whole_words():
    from eval_graph_only import widen, word_spans
    text = "Ed Fendell of Mission Control, in 1969."
    assert widen(text, [(5, 8)], 0) == [(5, 8)]
    # "Fendell" with one word on each side; the window stops at the text ends
    assert word_spans(text, widen(text, [(5, 8)], 1)) == "Ed Fendell of"
    assert word_spans(text, widen(text, [(0, 2)], 2)) == "Ed Fendell of"
    # The window crosses punctuation; overlapping windows join into one phrase
    assert word_spans(text, widen(text, [(24, 27), (34, 36)], 1)) == \
        "Mission Control, in 1969"
    assert widen(text, [(29, 30)], 1) == []  # "," holds no word


def test_budget_text_stops_at_the_token_budget():
    from eval_graph_only import budget_text

    class WordTokenizer:  # one token per word or separator
        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": re.findall(r"\w+|[^\w\s]", text)}

    text = "one two three four five six seven eight nine ten"
    ranges = [(0, 3), (19, 23), (34, 39)]  # one, five, eight
    assert budget_text(text, ranges, 0, 1, WordTokenizer()) == "one"
    assert budget_text(text, ranges, 0, 2, WordTokenizer()) == "one; five"
    assert budget_text(text, ranges, 1, 4, WordTokenizer()) == "one two; four five six"
    assert budget_text(text, ranges, 0, 99, WordTokenizer()) == "one; five; eight"


def test_lead_and_question_baselines_keep_the_budget():
    from eval_graph_only import cut, question_sentences

    class WordTokenizer:  # one token per word or punctuation mark
        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": re.findall(r"\w+|[^\w\s]", text)}

    tok = WordTokenizer()
    text = "Ada was born in 1815. She wrote notes on the engine. Bob liked tea."
    assert cut(text, 3, tok) == "Ada was born"
    assert cut(text, 6, tok) == "Ada was born in 1815."
    assert cut(text, 99, tok) == text
    # "engine" and "notes" match the second sentence; stop words do not count
    assert question_sentences(text, "What did the notes say about the engine?", 7, tok) == \
        "She wrote notes on the engine."
    # Two sentences, back in text order
    assert question_sentences(text, "When was Ada born and what notes?", 13, tok) == \
        "Ada was born in 1815. She wrote notes on the engine."
    assert question_sentences(text, "engine?", 3, tok) == "She wrote notes"


def test_text_modes_put_the_text_before_the_question():
    from eval_graph_only import TEXT_FIELDS, TEXT_MODES, mode_row
    row = {"text": "Who won?", "graph_text": "Ada won the prize.", "kind": "graph_only_respond",
           "entity_spans": [[0, 3]], "entities": ["Ada"], "node_text": "Ada; prize",
           "partner_node_text": "Bob", "random_text": "the", "gold_text": "Ada",
           "lead_text": "Ada won", "question_text": "Ada won the prize.",
           "node_q_text": "Ada; prize"}
    for mode in TEXT_MODES:
        new = mode_row(row, mode)
        assert new["text"] == row[TEXT_FIELDS[mode]] + "\n\nWho won?"
        assert "graph_text" not in new and "entity_spans" not in new
        assert not set(TEXT_FIELDS.values()) & set(new)
        assert new["entities"] == ["Ada"]


def test_add_text_fields_uses_the_top_k_node_words():
    from eval_graph_only import add_text_fields
    from model.entity import entity_node_scores
    model = make_model("xattn")
    collator, tok = make_collator()
    docs = ["Ada Lovelace was born in London in 1815 and died in 1852 .",
            "Paris is the capital of France and has many old bridges ."]
    rows = [{"text": "Q?", "graph_text": d, "entity_spans": [[0, 12]],
             "entities": ["Ada Lovelace"]} for d in docs]
    add_text_fields(model, collator, rows, "cpu")
    for r, d in zip(rows, docs):
        ids = torch.tensor([tok(d)["input_ids"]])
        with torch.no_grad():
            scores = entity_node_scores(model.token_ent(model._read(ids, torch.ones_like(ids))))
        top = sorted(torch.topk(scores[0], k=model.max_nodes).indices.tolist())
        words = d.split()
        # Each node word is in the node text; the text has no other words.
        # A punctuation node ("." here) gives no word
        node_words = set(r["node_text"].replace(";", " ").split())
        assert node_words == {words[i] for i in top if words[i].isalnum()}
        assert len(r["random_text"].replace(";", " ").split()) <= model.max_nodes
    assert rows[0]["gold_text"] == "Ada Lovelace"
    assert rows[1]["gold_text"] == ""  # the span does not point at the entity there
    assert rows[0]["partner_node_text"] == rows[1]["node_text"]
    assert rows[1]["partner_node_text"] == rows[0]["node_text"]
    # Question first: the node words come from the passage, never the question
    q_rows = [{"text": "Zed Quux ?", "graph_text": d} for d in docs]
    add_text_fields(model, collator, q_rows, "cpu")
    for r, d in zip(q_rows, docs):
        q_words = set(r["node_q_text"].replace(";", " ").split())
        assert q_words and q_words <= set(d.split())


def test_gap_closed_and_text_contrasts():
    from eval_graph_only import contrasts, gap_closed, summarize
    records = [{"row": i, "kind": "graph_only_respond", "answer_type": "other",
                "closed_book": {"loss_topk": 2.0, "f1": 0.0},
                "node_text": {"loss_topk": 1.5 - 0.01 * (i % 3), "f1": 0.3},
                "node_text_shuffled": {"loss_topk": 1.9, "f1": 0.1},
                "passage": {"loss_topk": 0.5, "f1": 0.8}} for i in range(12)]
    s = summarize(records)
    assert set(s) == {"closed_book", "node_text", "node_text_shuffled", "passage"}
    g = gap_closed(s)
    assert g["passage"] == 1.0 and g["node_text_shuffled"] == pytest.approx(0.0667, abs=1e-3)
    c = contrasts(records)
    assert c["loss_topk:node_text-node_text_shuffled"]["hi"] < 0
    assert "loss_topk:node_text-random_text" not in c  # no random_text: left out
    assert gap_closed({"node_text": {"loss_topk": 1.0}}) == {}
