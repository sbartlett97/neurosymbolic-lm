#!/usr/bin/env python3
"""Round 7: can the graph path carry an answer?

Rows: the val rows of kind graph_only_* (data/mix_stage2.py
--graph-only-frac). Each row is scored in four modes:

  closed_book     the question only, no graph
  graph           the question in the prompt, the graph from the passage
  graph_shuffled  as graph, with the nodes of another row (control). Rows
                  run in pairs for this mode, so a row gets its partner's nodes
  passage         passage + question in the prompt, the graph from the prompt
                  (the v3 setting: the upper bound)

With --modes, also the text modes (Step 2: the node-text test). Each puts
words of the passage as text in front of the question, with no graph
input and the nodes off, so only the text interface carries them:

  node_text           the words at the model's top-K node positions
  node_text_shuffled  the node words of the partner row (control)
  random_text         the words at K random passage positions (control)
  gold_text           the text of the first K gold entity spans (ceiling)

With --window W (Step 3: the context-window test), each selected range
grows by W words on each side, so a mode shows the words in context.
Node windows overlap more than random ones. So random_text then takes
random positions until it has as many tokens as node_text (same budget).

Respond rows: answer loss (mean token cross-entropy per row, teacher
forcing), with top-K nodes and, in the graph modes, with gold-span nodes
(the channel capacity); greedy generation scored by token F1 against the
reference and by key-word recall (answer words that the question does not
hold). Unanswerable rows: abstain rate (the first token is EOS).

Usage:
  python eval_graph_only.py --checkpoint ckpt/final_causal_model.pt \\
      --val data/curated_r7/curated_r7_val.jsonl --out results.json
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from data.chat_template import ChatTemplate
from data.collator_causal import CausalCognitiveCollator
from data.mix_stage2 import answer_key, words
from train_causal import load_causal_checkpoint, to_device
from model.entity import entity_node_scores
from training.trainers import graph_kwargs

MODES = ("closed_book", "graph", "graph_shuffled", "passage")
TEXT_MODES = ("node_text", "node_text_shuffled", "random_text", "gold_text")
ALL_MODES = MODES + TEXT_MODES
# The row field that holds the text of each text mode (add_text_fields)
TEXT_FIELDS = {"node_text": "node_text", "node_text_shuffled": "partner_node_text",
               "random_text": "random_text", "gold_text": "gold_text"}


def mode_row(row: dict, mode: str) -> dict:
    """The row as the model sees it in one mode."""
    if mode == "passage":
        new = {k: v for k, v in row.items() if k != "graph_text"}
        new["text"] = row["graph_text"] + "\n\n" + row["text"]
        return new
    if mode == "closed_book":
        return {k: v for k, v in row.items() if k != "graph_text"}
    if mode in TEXT_MODES:
        new = {k: v for k, v in row.items()
               if k not in ("graph_text", "entity_spans") and k not in TEXT_FIELDS.values()}
        new["text"] = row[TEXT_FIELDS[mode]] + "\n\n" + row["text"]
        return new
    return dict(row)


def widen(text: str, char_ranges, window: int):
    """Each character range, grown by `window` words on each side."""
    if window <= 0:
        return list(char_ranges)
    bounds = [m.span() for m in re.finditer(r"\w+", text)]
    out = []
    for a, b in char_ranges:
        inside = [i for i, (s, e) in enumerate(bounds) if e > a and s < b]
        if not inside:
            continue
        i = max(0, inside[0] - window)
        j = min(len(bounds) - 1, inside[-1] + window)
        out.append((bounds[i][0], bounds[j][1]))
    return out


def n_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def budget_text(text: str, char_ranges, window: int, budget: int, tokenizer) -> str:
    """word_spans of the first ranges (each widened) that reach `budget`
    tokens. Ranges past the budget are not used."""
    ranges = widen(text, char_ranges, window)
    lo, hi = 1, len(ranges)
    while lo < hi:  # the fewest ranges that reach the budget
        mid = (lo + hi) // 2
        if n_tokens(tokenizer, word_spans(text, ranges[:mid])) >= budget:
            hi = mid
        else:
            lo = mid + 1
    return word_spans(text, ranges[:lo])


def word_spans(text: str, char_ranges) -> str:
    """The whole words that hold the given character ranges, in text order.
    Adjacent words join into one phrase. Phrases are separated by "; ",
    and a phrase that repeats shows once."""
    spans = set()
    for a, b in char_ranges:
        if b <= a:
            continue
        # Grow a word piece to its word; a punctuation token stays as it is
        while a > 0 and text[a].isalnum() and text[a - 1].isalnum():
            a -= 1
        while b < len(text) and text[b - 1].isalnum() and text[b].isalnum():
            b += 1
        if any(c.isalnum() for c in text[a:b]):
            spans.add((a, b))
    phrases: List[List[int]] = []
    for a, b in sorted(spans):
        if phrases and (a <= phrases[-1][1] or not text[phrases[-1][1]:a].strip()):
            phrases[-1][1] = max(phrases[-1][1], b)
        else:
            phrases.append([a, b])
    out: List[str] = []
    for a, b in phrases:  # a phrase that repeats shows once
        if text[a:b].strip().lower() not in {o.lower() for o in out}:
            out.append(text[a:b].strip())
    return "; ".join(out)


@torch.no_grad()
def add_text_fields(model, collator, rows: List[dict], device: str,
                    seed: int = 0, window: int = 0) -> None:
    """Adds the text of each text mode to each row (in place). The node
    positions come from the read pass over graph_text, tokenized as the
    collator tokenizes it. Each range grows by `window` words on each side;
    with a window, random_text gets the token budget of node_text.
    Partners are the pairs of evaluate()."""
    rng = random.Random(seed)
    k = model.max_nodes
    for r in rows:
        tok, offsets = collator._tokenize_inputs([r["graph_text"]])
        ids = tok["input_ids"].to(device)
        mask = tok["attention_mask"].to(device)
        enc = model._read(ids, mask)
        scores = entity_node_scores(model.token_ent(enc), mask)[0]
        top = torch.topk(scores, k=min(k, ids.shape[1])).indices.tolist()
        if offsets is None:  # a slow tokenizer: offsets without tensors
            offsets = [collator.tokenizer(
                r["graph_text"], truncation=True, max_length=collator.max_length,
                return_offsets_mapping=True)["offset_mapping"]]
        offs = [tuple(o) for o in (offsets[0].tolist() if torch.is_tensor(offsets)
                                   else offsets[0])]
        if len(offs) != ids.shape[1]:
            raise ValueError("offsets do not match the graph input ids")
        real = [i for i, (a, b) in enumerate(offs) if b > a]
        g = r["graph_text"]
        r["node_text"] = word_spans(g, widen(g, [offs[i] for i in top], window))
        if window:
            r["random_text"] = budget_text(g, [offs[i] for i in rng.sample(real, len(real))],
                                           window, n_tokens(collator.tokenizer, r["node_text"]),
                                           collator.tokenizer)
        else:
            r["random_text"] = word_spans(g, widen(
                g, [offs[i] for i in rng.sample(real, min(k, len(real)))], window))
        # The first K gold spans, as the gold nodes; a span must point at
        # its entity (as in the collator)
        gold = [(s, e) for (s, e), ent in zip((r.get("entity_spans") or [])[:k],
                                               r.get("entities", []))
                if 0 <= s < e <= len(g) and g[s:e].strip().lower() == ent.strip().lower()]
        r["gold_text"] = word_spans(g, widen(g, gold, window))
    for i in range(0, len(rows) - 1, 2):
        rows[i]["partner_node_text"] = rows[i + 1]["node_text"]
        rows[i + 1]["partner_node_text"] = rows[i]["node_text"]


def token_f1(pred: str, ref: str) -> float:
    p, r = words(pred), words(ref)
    common = sum((Counter(p) & Counter(r)).values())
    if not p or not r or common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def is_value(answer_words: set) -> bool:
    return any(any(c.isdigit() for c in w) for w in answer_words)


@torch.no_grad()
def answer_loss(model, batch, use_soft_nodes: bool, spans) -> List[float]:
    """Mean cross-entropy over each row's answer tokens."""
    out = model(batch["input_ids"], batch["attention_mask"], spans=spans,
                y_ids=batch["decoder_input_ids"], use_soft_nodes=use_soft_nodes,
                **graph_kwargs(batch))
    labels = batch["decoder_labels"]
    ce = F.cross_entropy(out["logits"].float().transpose(1, 2), labels,
                         ignore_index=-100, reduction="none")
    n = (labels != -100).sum(1).clamp(min=1)
    return (ce.sum(1) / n).tolist()


@torch.no_grad()
def score_pair(model, batch, pair: List[dict], mode: str, gen_kw: Dict,
               force_answer: bool):
    """Losses and greedy generations of one pair of rows in one mode.
    Returns (topk losses, gold-span losses, generations, forced generations)."""
    eos = gen_kw["eos_token_id"]
    use_nodes = mode not in ("closed_book",) + TEXT_MODES
    model.shuffle_nodes = mode == "graph_shuffled"
    try:
        topk = answer_loss(model, batch, use_nodes, None)
        gold = (answer_loss(model, batch, use_nodes, batch["entity_token_spans"])
                if mode in ("graph", "graph_shuffled") else [None, None])
        gens, forced = [], []
        for j in range(len(pair)):
            one = {k: (v[j:j + 1] if torch.is_tensor(v) else v) for k, v in batch.items()}
            gk = graph_kwargs(one)
            if mode == "graph_shuffled":  # the partner's graph text
                gk = graph_kwargs({k: (v[1 - j:2 - j] if torch.is_tensor(v) else v)
                                   for k, v in batch.items()})
            kw = dict(gen_kw, use_soft_nodes=use_nodes, **gk)
            ids = model.generate(one["input_ids"], one["attention_mask"], **kw)[0].tolist()
            gens.append(ids[:ids.index(eos)] if eos in ids else ids)
            if force_answer and not gens[-1] and int(pair[j].get("should_respond", 0)) == 1:
                ids = model.generate(one["input_ids"], one["attention_mask"],
                                     min_new_tokens=1, **kw)[0].tolist()
                forced.append(ids[:ids.index(eos)] if eos in ids else ids)
            else:
                forced.append(gens[-1])
    finally:
        model.shuffle_nodes = False
    return topk, gold, gens, forced


def evaluate(model, tokenizer, collator, rows: List[dict], device: str,
             max_new_tokens: int = 64, n_show: int = 3,
             modes=MODES, force_answer: bool = False) -> Dict:
    """Returns (summary, records). records holds one dict per row with its
    metrics in each mode, so modes can be compared row by row later.
    force_answer: a respond row that abstains generates again with EOS
    blocked at the first step (f1_forced, key_recall_forced)."""
    gen_kw = dict(max_length=max_new_tokens, do_sample=False,
                  eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    records: List[Dict] = []
    shown = cpu_pairs = 0
    # Pairs: graph_shuffled gives each row its partner's nodes
    for i in range(0, len(rows) - 1, 2):
        pair = rows[i:i + 2]
        recs = []
        for r in pair:
            key = answer_key(r["text"], r["response"]) if r.get("response") else set()
            recs.append({"row": len(records) + len(recs), "kind": r["kind"],
                         "answer_type": "value" if is_value(key) else "other"})
        records.extend(recs)
        for mode in modes:
            batch = collator([mode_row(r, mode) for r in pair])
            try:
                topk, gold, gens, forced = score_pair(
                    model, to_device(batch, device), pair, mode, gen_kw, force_answer)
            except RuntimeError as e:
                # MPS out of memory on a long passage: run this pair on the CPU
                if device != "mps" or "out of memory" not in str(e):
                    raise
                torch.mps.empty_cache()
                model.to("cpu")
                try:
                    topk, gold, gens, forced = score_pair(
                        model, to_device(batch, "cpu"), pair, mode, gen_kw, force_answer)
                finally:
                    model.to(device)
                cpu_pairs += 1

            for r, rec, lt, lg, g, gf in zip(pair, recs, topk, gold, gens, forced):
                abstained = len(g) == 0
                if int(r.get("should_respond", 0)) == 0:
                    rec[mode] = {"abstain": int(abstained)}
                    continue
                text = tokenizer.decode(g, skip_special_tokens=True)
                key = answer_key(r["text"], r["response"])
                m = {"loss_topk": lt, "f1": token_f1(text, r["response"]),
                     "respond": int(not abstained), "gen": text[:200]}
                if lg is not None:
                    m["loss_gold"] = lg
                if key:
                    m["key_recall"] = len(key & set(words(text))) / len(key)
                if force_answer:
                    ftext = tokenizer.decode(gf, skip_special_tokens=True)
                    m["f1_forced"] = token_f1(ftext, r["response"])
                    m["gen_forced"] = ftext[:200]
                    if key:
                        m["key_recall_forced"] = len(key & set(words(ftext))) / len(key)
                rec[mode] = m
                if mode == "graph" and shown < n_show:
                    shown += 1
                    print(f"  Q: {r['text'][:120]}\n  ref: {r['response'][:120]}\n"
                          f"  gen: {text[:120]!r}")

    if cpu_pairs:
        print(f"  {cpu_pairs} pair-modes ran on the CPU (MPS out of memory)")
    return summarize(records), records


def summarize(records: List[Dict]) -> Dict:
    """Mean of each metric per mode; f1 and key_recall also per answer type;
    abstain rate per unanswerable kind."""
    summary = {}
    for mode in [m for m in ALL_MODES if any(m in rec for rec in records)]:
        vals: Dict[str, list] = {}
        for rec in records:
            m = rec.get(mode)
            if m is None:
                continue
            if "abstain" in m:
                kind = rec["kind"].removeprefix("graph_only_")
                vals.setdefault(f"abstain_{kind}", []).append(m["abstain"])
                continue
            for k, v in m.items():
                if k in ("gen", "gen_forced"):
                    continue
                vals.setdefault(k, []).append(v)
                if k in ("f1", "key_recall"):
                    vals.setdefault(f"{k}_{rec['answer_type']}", []).append(v)
        summary[mode] = {k: round(sum(v) / len(v), 4) for k, v in vals.items()}
        summary[mode]["n_respond"] = len(vals.get("f1", []))
    return summary


def paired_ci(records: List[Dict], metric: str, a: str, b: str,
              answer_type: Optional[str] = None, n_boot: int = 2000,
              seed: int = 0) -> Dict:
    """Mean of metric(a) - metric(b) over the rows that have it in both
    modes, with a paired bootstrap 95% interval."""
    diffs = [rec[a][metric] - rec[b][metric] for rec in records
             if metric in rec.get(a, {}) and metric in rec.get(b, {})
             and (answer_type is None or rec["answer_type"] == answer_type)]
    if not diffs:
        return {"n": 0}
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(n_boot))
    return {"n": n, "mean": round(sum(diffs) / n, 4),
            "lo": round(means[int(0.025 * n_boot)], 4),
            "hi": round(means[int(0.975 * n_boot) - 1], 4)}


# The Round 7 decision: the graph carries the answer only if both lower
# bounds of the key_recall contrasts are above 0
CONTRASTS = [("key_recall", "graph", "closed_book", None),
             ("key_recall", "graph", "graph_shuffled", None),
             ("f1", "graph", "closed_book", "value"),
             ("f1", "graph", "graph_shuffled", "value"),
             ("key_recall", "passage", "graph", None)]
# Step 2: node_text passes if its lower bound against node_text_shuffled is
# above 0. loss_topk is the main metric (it does not depend on abstaining)
TEXT_CONTRASTS = [(m, a, b, None) for m in ("loss_topk", "key_recall", "key_recall_forced")
                  for a, b in (("node_text", "closed_book"),
                               ("node_text", "node_text_shuffled"),
                               ("node_text", "random_text"),
                               ("gold_text", "node_text"),
                               ("passage", "node_text"))]


def contrasts(records: List[Dict]) -> Dict:
    out = {f"{m}{'_' + t if t else ''}:{a}-{b}": paired_ci(records, m, a, b, t)
           for m, a, b, t in CONTRASTS + TEXT_CONTRASTS}
    return {k: v for k, v in out.items() if v["n"]}


def gap_closed(summary: Dict, metric: str = "loss_topk") -> Dict:
    """(closed_book - mode) / (closed_book - passage) for each other mode."""
    if "closed_book" not in summary or "passage" not in summary:
        return {}
    book, top = summary["closed_book"].get(metric), summary["passage"].get(metric)
    if book is None or top is None or book == top:
        return {}
    return {m: round((book - v[metric]) / (book - top), 4)
            for m, v in summary.items() if metric in v and m != "closed_book"}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val", required=True, help="val JSONL with graph_only_* rows")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="Use at most this many graph-only rows")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                   ("mps" if torch.backends.mps.is_available() else "cpu"))
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--force-answer", action="store_true",
                   help="Generate again with EOS blocked at the first step when a "
                        "respond row abstains (f1_forced, key_recall_forced)")
    p.add_argument("--window", type=int, default=0,
                   help="Text modes: grow each selected range by this many words "
                        "on each side")
    p.add_argument("--modes", default=",".join(MODES),
                   help=f"Comma-separated, from {','.join(ALL_MODES)}")
    args = p.parse_args(argv)
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    unknown = set(modes) - set(ALL_MODES)
    if unknown:
        p.error(f"unknown modes: {sorted(unknown)}")

    model, tokenizer, ck = load_causal_checkpoint(args.checkpoint, args.device)
    model.to(getattr(torch, args.dtype)).eval()
    config = ck["config"]
    collator = CausalCognitiveCollator(
        tokenizer=tokenizer, concept_map=ck["concept_map"],
        relation_map=ck["relation_map"], include_responses=True,
        concept_to_entity_type_map=ck.get("concept_to_entity_type_map", {}),
        max_length=config["max_input_length"],
        max_output_length=config["max_output_length"],
        chat_mode=False,
        chat_template=ChatTemplate(default_system_prompt=ck.get("chat_system_prompt")),
        max_nodes=config["max_nodes"], entity_type_map=ck.get("entity_type_map", {}))

    with open(args.val) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = [r for r in rows if r.get("kind", "").startswith("graph_only_")
            and r.get("graph_text")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} graph-only rows: "
          f"{dict(Counter(r['kind'] for r in rows))}")
    if set(modes) & set(TEXT_MODES):
        add_text_fields(model, collator, rows, args.device, window=args.window)
        print(f"  node text: {rows[0]['node_text'][:200]!r}")
    # The mean length in tokens of each text, for the gap per token
    lengths = {m: round(sum(n_tokens(tokenizer, r[f]) for r in rows) / len(rows), 1)
               for m, f in [("passage", "graph_text")] + list(TEXT_FIELDS.items())
               if m in modes and rows and f in rows[0]}
    print(f"  mean text tokens: {lengths}")
    summary, records = evaluate(model, tokenizer, collator, rows, args.device,
                                max_new_tokens=args.max_new_tokens, modes=modes,
                                force_answer=args.force_answer)
    paired = contrasts(records)
    gaps = gap_closed(summary)
    print(json.dumps({"summary": summary, "contrasts": paired, "gap_closed": gaps},
                     indent=1))
    for rec in records:  # the text each text mode showed the model
        if set(modes) & set(TEXT_MODES):
            rec["texts"] = {m: rows[rec["row"]][TEXT_FIELDS[m]]
                            for m in modes if m in TEXT_MODES}
    with open(args.out, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "val": args.val, "modes": modes,
                   "window": args.window, "text_tokens": lengths, "rows": len(rows),
                   "summary": summary, "contrasts": paired, "gap_closed": gaps,
                   "records": records},
                  f, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
