#!/usr/bin/env python3
"""Build the v2 stage 2/3 set from the v1 corpus and QA files. No API calls.

Three kinds of rows:

  respond       the v1 QA rows: document + question -> answer (unchanged)
  unanswerable  a v1 abstain document + a question from another document.
                The two documents share an entity type, and the borrowed
                answer does not occur in the document. Target: EOS.
                (v1 abstain rows had no question at all, so the model only
                learned "no question -> EOS".)
  general       the start of a corpus document that the QA set does not use.
                Target: the text that follows (plain language modelling, so
                the backbone keeps its general ability). Entities and
                relations are kept only if they sit inside the prompt.

Each row keeps the train/val split of its source document. The output vocab
is the QA vocab: general rows only hold a subset of the corpus labels.

Usage:
    python data/mix_stage2.py --corpus data/curated/curated.jsonl \\
        --qa data/curated_qa/curated.jsonl --out data/curated_v2/curated_v2.jsonl
"""

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_jsonl(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def val_path(path: Path) -> Path:
    return path.with_name(path.stem + "_val.jsonl")


def split_question(text: str) -> Tuple[str, str]:
    """A v1 respond row is document + "\\n\\n" + question."""
    doc, _, question = text.rpartition("\n\n")
    return doc, question


def make_unanswerable(abstain_rows: List[dict], respond_rows: List[dict],
                      rng: random.Random) -> List[dict]:
    """Pair each abstain document with a question from another document
    that shares an entity type. The borrowed answer must not occur in the
    document, so the document cannot answer the question."""
    pool = []
    for r in respond_rows:
        doc, question = split_question(r["text"])
        pool.append((doc, question, str(r.get("response", "")),
                     set(r.get("entity_types", []))))
    by_type: Dict[str, List[int]] = {}
    for i, (_, _, _, types) in enumerate(pool):
        for t in types:
            by_type.setdefault(t, []).append(i)

    out = []
    for row in abstain_rows:
        doc = row["text"]
        low = doc.lower()
        cands = sorted({i for t in set(row.get("entity_types", []))
                        for i in by_type.get(t, [])})
        # No shared type: any question. The document still cannot answer it
        if not cands:
            cands = list(range(len(pool)))
        rng.shuffle(cands)
        for i in cands[:50]:
            other_doc, question, answer, _ = pool[i]
            if other_doc == doc or not answer.strip():
                continue
            if answer.strip().lower() in low:
                continue
            new = dict(row)
            new["text"] = doc + "\n\n" + question
            new["should_respond"] = 0
            new["response"] = ""
            out.append(new)
            break
    return out


def cut_at_boundary(text: str, pos: int) -> int:
    """Move pos back to just after the nearest sentence end or space."""
    head = text[:pos]
    m = None
    for m in re.finditer(r"[.!?]\s", head):
        pass
    if m is not None and pos - m.end() < 200:
        return m.end()
    space = head.rfind(" ")
    return space + 1 if space > 0 else pos


def make_general(row: dict, rng: random.Random, min_chars: int,
                 max_chars: int) -> Optional[dict]:
    """Prefix -> continuation row. Labels stay only inside the prefix."""
    text = row["text"]
    if len(text) < 1000:
        return None
    cut = cut_at_boundary(text, int(len(text) * rng.uniform(0.4, 0.8)))
    prompt = text[:cut].rstrip()
    rest = text[cut:]
    n = rng.randint(min_chars, max_chars)
    if len(rest) > n:
        end = rest.rfind(" ", 0, n)
        rest = rest[:end if end > 0 else n]
    response = rest.strip()
    if len(prompt) < 200 or len(response) < min_chars // 2:
        return None

    keep: Dict[int, int] = {}
    entities, spans, types, concepts = [], [], [], []
    for i, span in enumerate(row.get("entity_spans", [])):
        s, e = int(span[0]), int(span[1])
        if s < 0 or e > len(prompt):
            continue
        keep[i] = len(entities)
        entities.append(row["entities"][i])
        spans.append([s, e])
        if i < len(row.get("entity_types", [])):
            types.append(row["entity_types"][i])
        if i < len(row.get("concepts", [])):
            concepts.append(row["concepts"][i])
    relations = [[keep[h], keep[t], rel] for h, t, rel in row.get("relations", [])
                 if h in keep and t in keep]

    new = dict(row)
    new.update(text=prompt, entities=entities, entity_spans=spans,
               entity_types=types, concepts=concepts, relations=relations,
               should_respond=1, response=response)
    return new


def build_split(corpus: List[dict], qa: List[dict], n_general: int,
                rng: random.Random, min_chars: int, max_chars: int):
    respond = [r for r in qa if int(r.get("should_respond", 0)) == 1]
    abstain = [r for r in qa if int(r.get("should_respond", 0)) == 0]
    unanswerable = make_unanswerable(abstain, respond, rng)

    used = {r["text"] for r in abstain}
    used.update(split_question(r["text"])[0] for r in respond)
    free = [r for r in corpus if r["text"] not in used]
    rng.shuffle(free)
    general = []
    for r in free:
        if len(general) >= n_general:
            break
        g = make_general(r, rng, min_chars, max_chars)
        if g is not None:
            general.append(g)

    rows = respond + unanswerable + general
    rng.shuffle(rows)
    stats = {
        "respond": len(respond),
        "unanswerable": len(unanswerable),
        "abstain_without_question": len(abstain) - len(unanswerable),
        "general": len(general),
        "corpus_docs_not_in_qa": len(free),
        "response_chars_qa": sum(len(r.get("response", "")) for r in respond),
        "response_chars_general": sum(len(r["response"]) for r in general),
    }
    return rows, stats


def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--corpus", type=Path, required=True,
                   help="Stage 1 corpus JSONL (with <name>_val.jsonl)")
    p.add_argument("--qa", type=Path, required=True,
                   help="v1 QA JSONL (with <name>_val.jsonl and _vocab.json)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--general-rows", type=int, default=4000,
                   help="General-text rows in train; val gets the same share")
    p.add_argument("--min-chars", type=int, default=275,
                   help="Shortest continuation (about 64 Qwen tokens)")
    p.add_argument("--max-chars", type=int, default=550,
                   help="Longest continuation (about 128 Qwen tokens)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    qa_train = read_jsonl(args.qa)
    qa_val = read_jsonl(val_path(args.qa))
    n_val = round(args.general_rows * len(qa_val) / max(len(qa_train), 1))

    for name, corpus_file, qa, n, out in [
        ("train", args.corpus, qa_train, args.general_rows, args.out),
        ("val", val_path(args.corpus), qa_val, n_val, val_path(args.out)),
    ]:
        rows, stats = build_split(read_jsonl(corpus_file), qa, n, rng,
                                  args.min_chars, args.max_chars)
        write_jsonl(out, rows)
        share = stats["response_chars_general"] / max(
            stats["response_chars_general"] + stats["response_chars_qa"], 1)
        print(f"{name}: {len(rows)} rows -> {out}")
        print(f"  {json.dumps(stats)}")
        print(f"  general text share of response chars: {share:.0%}")

    vocab = args.qa.with_name(args.qa.stem + "_vocab.json")
    if vocab.exists():
        shutil.copy(vocab, args.out.with_name(args.out.stem + "_vocab.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
