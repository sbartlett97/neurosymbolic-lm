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
import sys
from collections import Counter
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from data.chat_template import ChatTemplate
from data.collator_causal import CausalCognitiveCollator
from data.mix_stage2 import answer_key, words
from train_causal import load_causal_checkpoint, to_device
from training.trainers import graph_kwargs

MODES = ("closed_book", "graph", "graph_shuffled", "passage")


def mode_row(row: dict, mode: str) -> dict:
    """The row as the model sees it in one mode."""
    if mode == "passage":
        new = {k: v for k, v in row.items() if k != "graph_text"}
        new["text"] = row["graph_text"] + "\n\n" + row["text"]
        return new
    if mode == "closed_book":
        return {k: v for k, v in row.items() if k != "graph_text"}
    return dict(row)


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


def evaluate(model, tokenizer, collator, rows: List[dict], device: str,
             max_new_tokens: int = 64, n_show: int = 3) -> Dict:
    """Returns (summary, records). records holds one dict per row with its
    metrics in each mode, so modes can be compared row by row later."""
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    records: List[Dict] = []
    shown = 0
    # Pairs: graph_shuffled gives each row its partner's nodes
    for i in range(0, len(rows) - 1, 2):
        pair = rows[i:i + 2]
        recs = []
        for r in pair:
            key = answer_key(r["text"], r["response"]) if r.get("response") else set()
            recs.append({"row": len(records) + len(recs), "kind": r["kind"],
                         "answer_type": "value" if is_value(key) else "other"})
        records.extend(recs)
        for mode in MODES:
            batch = to_device(collator([mode_row(r, mode) for r in pair]), device)
            use_nodes = mode != "closed_book"
            model.shuffle_nodes = mode == "graph_shuffled"
            try:
                topk = answer_loss(model, batch, use_nodes, None)
                gold = (answer_loss(model, batch, use_nodes, batch["entity_token_spans"])
                        if mode in ("graph", "graph_shuffled") else [None, None])
                gens = []
                for j in range(len(pair)):
                    one = {k: (v[j:j + 1] if torch.is_tensor(v) else v)
                           for k, v in batch.items()}
                    gk = graph_kwargs(one)
                    if mode == "graph_shuffled":  # the partner's graph text
                        gk = graph_kwargs({k: (v[1 - j:2 - j] if torch.is_tensor(v) else v)
                                           for k, v in batch.items()})
                    gen = model.generate(one["input_ids"], one["attention_mask"],
                                         max_length=max_new_tokens, do_sample=False,
                                         eos_token_id=eos, pad_token_id=pad,
                                         use_soft_nodes=use_nodes, **gk)
                    ids = gen[0].tolist()
                    gens.append(ids[:ids.index(eos)] if eos in ids else ids)
            finally:
                model.shuffle_nodes = False

            for r, rec, lt, lg, g in zip(pair, recs, topk, gold, gens):
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
                rec[mode] = m
                if mode == "graph" and shown < n_show:
                    shown += 1
                    print(f"  Q: {r['text'][:120]}\n  ref: {r['response'][:120]}\n"
                          f"  gen: {text[:120]!r}")

    return summarize(records), records


def summarize(records: List[Dict]) -> Dict:
    """Mean of each metric per mode; f1 and key_recall also per answer type;
    abstain rate per unanswerable kind."""
    summary = {}
    for mode in MODES:
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
                if k == "gen":
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


def contrasts(records: List[Dict]) -> Dict:
    return {f"{m}{'_' + t if t else ''}:{a}-{b}": paired_ci(records, m, a, b, t)
            for m, a, b, t in CONTRASTS}


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
    args = p.parse_args(argv)

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
    summary, records = evaluate(model, tokenizer, collator, rows, args.device,
                                max_new_tokens=args.max_new_tokens)
    paired = contrasts(records)
    print(json.dumps({"summary": summary, "contrasts": paired}, indent=1))
    with open(args.out, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "val": args.val,
                   "rows": len(rows), "summary": summary, "contrasts": paired,
                   "records": records},
                  f, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
