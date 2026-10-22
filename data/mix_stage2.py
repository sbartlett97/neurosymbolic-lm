#!/usr/bin/env python3
"""Build the stage 2/3 set from the v1 corpus and QA files. No API calls.

Row kinds (the "kind" field):

  respond             the v1 QA rows: document + question -> answer (unchanged)
  unanswerable_cross  a v1 abstain document + a question from another
                      document. The two documents share an entity type, and
                      the borrowed answer does not occur in the document.
                      Target: EOS. (v1 abstain rows had no question at all,
                      so the model only learned "no question -> EOS".)
  unanswerable_same   a respond row with the answer sentences deleted. Same
                      question, same topic. Target: EOS. A cross-document
                      question is often off-topic, which is an easy cue; this
                      kind removes the cue.
  unanswerable_swap   a respond row with one question entity changed to a
                      same-type entity that the document does not contain.
                      Target: EOS.
  general             the start of a corpus document that the QA set does not
                      use. Target: the text that follows (plain language
                      modelling, so the backbone keeps its general ability).
                      Entities and relations are kept only if they sit inside
                      the prompt.

A respond row gives at most one same-document or swap negative.

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


def remap_labels(row: dict, new_spans: Dict[int, Tuple[int, int]]) -> dict:
    """Keep only the entities in new_spans (old index -> new span), in order,
    with their types and concepts. Drop the relations of the other entities."""
    keep: Dict[int, int] = {}
    entities, spans, types, concepts = [], [], [], []
    for i in sorted(new_spans):
        keep[i] = len(entities)
        entities.append(row["entities"][i])
        spans.append(list(new_spans[i]))
        if i < len(row.get("entity_types", [])):
            types.append(row["entity_types"][i])
        if i < len(row.get("concepts", [])):
            concepts.append(row["concepts"][i])
    relations = [[keep[h], keep[t], rel] for h, t, rel in row.get("relations", [])
                 if h in keep and t in keep]
    return dict(entities=entities, entity_spans=spans, entity_types=types,
                concepts=concepts, relations=relations)


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
            new["kind"] = "unanswerable_cross"
            out.append(new)
            break
    return out


# --- same-document negatives ----------------------------------------------------

STOP = set("""a an the of in on at to for from by with and or but is are was were
be been being it its this that these those as into about over after before than
then there their they he she his her him we our you your i not no do does did has
have had which who whom whose what when where why how text author article says said
also can could would will may might one""".split())
WORD = re.compile(r"[A-Za-z0-9]+(?:[.,'’][A-Za-z0-9]+)*")
SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+[\"'”’)]*|\n|$)")


def words(text: str) -> List[str]:
    return [w.lower() for w in WORD.findall(text)]


def answer_key(question: str, answer: str) -> set:
    """The answer words that the question does not already hold."""
    q = set(words(question))
    return {w for w in words(answer) if w not in STOP and w not in q}


def holds_answer(sentence: str, key: set) -> bool:
    found = set(words(sentence))
    if len(key) <= 3:
        return key <= found
    return len(key & found) >= max(3, int(0.6 * len(key)))


def sentence_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in SENTENCE.finditer(text) if m.group().strip()]


# Questions with a short, factual answer. A "why" or "how" answer is often a
# paraphrase, so a word match can delete the wrong sentence and leave the answer
FACTUAL = re.compile(r"^(who|whom|whose|when|where|which|what|how (many|much|long|far|old|tall|big))\b", re.I)


def make_same_doc(row: dict, max_sentences: int = 3,
                  max_key: int = 6) -> Optional[dict]:
    """Delete the sentences that hold the answer. None if the answer cannot be
    found with confidence, or if it survives the deletion."""
    doc, question = split_question(row["text"])
    answer = str(row.get("response", ""))
    if not FACTUAL.match(question.strip()):
        return None
    key = answer_key(question, answer)
    if len(key) > max_key:
        return None
    # A single plain word (no digit, not a name) matches too many sentences
    strong = any(any(c.isdigit() for c in w) for w in key) or any(
        w[:1].isupper() and w.lower() in key for w in WORD.findall(answer))
    if not doc or not key or (len(key) < 2 and not strong):
        return None
    cuts = [(s, e) for s, e in sentence_spans(doc) if holds_answer(doc[s:e], key)]
    if not cuts or len(cuts) > max_sentences:
        return None
    parts, pos = [], 0
    for s, e in cuts:
        parts.append(doc[pos:s])
        pos = e
    parts.append(doc[pos:])
    new_doc = "".join(parts)
    if any(holds_answer(new_doc[s:e], key) for s, e in sentence_spans(new_doc)):
        return None
    flat = answer.strip().rstrip(".").lower()
    if len(flat) > 3 and flat in new_doc.lower():
        return None

    new_spans: Dict[int, Tuple[int, int]] = {}
    for i, span in enumerate(row.get("entity_spans", [])):
        s, e = int(span[0]), int(span[1])
        if any(s < ce and e > cs for cs, ce in cuts):
            continue
        shift = sum(ce - cs for cs, ce in cuts if ce <= s)
        if new_doc[s - shift:e - shift] == doc[s:e]:
            new_spans[i] = (s - shift, e - shift)

    new = dict(row)
    new.update(remap_labels(row, new_spans))
    new.update(text=new_doc + "\n\n" + question, should_respond=0, response="",
               kind="unanswerable_same")
    return new


# Only proper names swap cleanly. A swapped date, quantity or common noun
# often gives a nonsense question, which is an easy cue of its own
SWAP_TYPES = ("person", "organization", "location")


def is_name(text: str) -> bool:
    parts = text.split()
    return bool(parts) and all(p[:1].isupper() for p in parts if p[:1].isalpha())


def make_swap(row: dict, by_type: Dict[str, List[str]],
              rng: random.Random) -> Optional[dict]:
    """Change one question name (person, organization, location) to another
    name of the same type that the document does not contain. The document
    stays the same."""
    doc, question = split_question(row["text"])
    low_doc = doc.lower()
    found = []
    for i, ent in enumerate(row.get("entities", [])):
        if (len(ent) < 3 or i >= len(row.get("entity_types", []))
                or row["entity_types"][i] not in SWAP_TYPES or not is_name(ent)):
            continue
        m = re.search(r"\b" + re.escape(ent) + r"\b", question, re.I)
        if m:
            found.append((len(ent), m, row["entity_types"][i]))
    if not found:
        return None
    _, m, etype = max(found, key=lambda f: f[0])
    names = by_type.get(etype, [])
    for _ in range(20):
        if not names:
            return None
        other = rng.choice(names)
        if other.lower() in low_doc or other.lower() in question.lower():
            continue
        new_q = question[:m.start()] + other + question[m.end():]
        new = dict(row)
        new.update(text=doc + "\n\n" + new_q, should_respond=0, response="",
                   kind="unanswerable_swap")
        return new
    return None


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

    new_spans = {i: (int(s), int(e))
                 for i, (s, e) in enumerate(row.get("entity_spans", []))
                 if int(s) >= 0 and int(e) <= len(prompt)}
    new = dict(row)
    new.update(remap_labels(row, new_spans))
    new.update(text=prompt, should_respond=1, response=response, kind="general")
    return new


def build_split(corpus: List[dict], qa: List[dict], n_general: int,
                rng: random.Random, min_chars: int, max_chars: int,
                n_cross: int = -1, n_same: int = 0, n_swap: int = 0):
    respond = [dict(r, kind="respond") for r in qa
               if int(r.get("should_respond", 0)) == 1]
    abstain = [r for r in qa if int(r.get("should_respond", 0)) == 0]
    unanswerable = make_unanswerable(abstain, respond, rng)
    if n_cross >= 0:
        unanswerable = unanswerable[:n_cross]

    # One negative per respond row at most: same-document first, then swap
    order = list(range(len(respond)))
    if n_same or n_swap:  # no extra draw: the default output stays the v2 set
        rng.shuffle(order)
    same, swap, rest = [], [], []
    for i in order:
        if len(same) >= n_same:
            rest.append(i)
            continue
        neg = make_same_doc(respond[i])
        if neg is not None:
            same.append(neg)
        else:
            rest.append(i)
    by_type: Dict[str, List[str]] = {}
    for r in qa:
        for ent, t in zip(r.get("entities", []), r.get("entity_types", [])):
            if len(ent) >= 3 and t in SWAP_TYPES and is_name(ent):
                by_type.setdefault(t, []).append(ent)
    for t in by_type:
        by_type[t] = sorted(set(by_type[t]))
    for i in rest:
        if len(swap) >= n_swap:
            break
        neg = make_swap(respond[i], by_type, rng)
        if neg is not None:
            swap.append(neg)

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

    rows = respond + unanswerable + same + swap + general
    rng.shuffle(rows)
    stats = {
        "respond": len(respond),
        "unanswerable": len(unanswerable),
        "unanswerable_same": len(same),
        "unanswerable_swap": len(swap),
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
    p.add_argument("--cross-doc-rows", type=int, default=-1,
                   help="Cross-document unanswerable rows in train (-1: all)")
    p.add_argument("--same-doc-rows", type=int, default=0,
                   help="Same-document unanswerable rows in train")
    p.add_argument("--swap-rows", type=int, default=0,
                   help="Entity-swap unanswerable rows in train")
    p.add_argument("--min-chars", type=int, default=275,
                   help="Shortest continuation (about 64 Qwen tokens)")
    p.add_argument("--max-chars", type=int, default=550,
                   help="Longest continuation (about 128 Qwen tokens)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    qa_train = read_jsonl(args.qa)
    qa_val = read_jsonl(val_path(args.qa))
    share = len(qa_val) / max(len(qa_train), 1)

    def val_n(n):
        return n if n < 0 else round(n * share)

    counts = (args.general_rows, args.cross_doc_rows, args.same_doc_rows, args.swap_rows)
    for name, corpus_file, qa, (n_gen, n_cross, n_same, n_swap), out in [
        ("train", args.corpus, qa_train, counts, args.out),
        ("val", val_path(args.corpus), qa_val, tuple(val_n(n) for n in counts),
         val_path(args.out)),
    ]:
        rows, stats = build_split(read_jsonl(corpus_file), qa, n_gen, rng,
                                  args.min_chars, args.max_chars,
                                  n_cross=n_cross, n_same=n_same, n_swap=n_swap)
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
