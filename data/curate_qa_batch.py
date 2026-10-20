#!/usr/bin/env python3
"""Stage-2/3 QA data through the OpenAI Batch API (50% of the sync price).

The QA set is built from an existing GLiNER2 corpus (curate_dataset.py
--annotator gliner). No GPU is needed, and the batch runs for up to 24 h,
so this can run from a laptop after the curation pod is gone.

Steps (all state lives in --out-dir/batch_state.json, so each step can be
re-run and a crash loses nothing):

    prepare  Pick --qa-docs corpus documents. A hash of each document
             decides respond (QA request) or abstain. Write request files.
    submit   Upload request files and create batches. Chunks that hit the
             org's enqueued-token limit are submitted again later.
    status   Show batch progress.
    collect  Download results. Write <name>.jsonl / <name>_val.jsonl /
             <name>_vocab.json for train_causal.py --stage2-dataset.
    run      submit + poll + collect until done.

Usage:
    python data/curate_qa_batch.py prepare --corpus data/curated/curated.jsonl \\
        --out-dir data/curated_qa --qa-docs 20000 --should-respond-ratio 0.7
    python data/curate_qa_batch.py run --out-dir data/curated_qa

Rows:
- respond: corpus text + "\\n\\n" + question, should_respond=1, the answer
  as response. The question is appended, so the entity spans stay valid.
- abstain: the corpus row unchanged (should_respond=0).
- A failed QA request is dropped. It never becomes an abstain row.
Each row keeps the train/val split of its corpus document.

The key comes from OPENAI_API_KEY. Check the provider terms on training
with model outputs before a large run.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.curate.dedup import text_hash  # noqa: E402
from data.curate.gliner_annotator import parse_qa_reply  # noqa: E402
from data.curate.llm_annotator import OpenAIBackend  # noqa: E402
from data.curate.prompts import QA_JSON_SCHEMA, get_qa_prompt  # noqa: E402

ENDPOINT = "/v1/chat/completions"
# Stop when most QA requests fail: dropped rows would leave a tiny set
FAIL_RATE_LIMIT = 0.8
STATE_FILE = "batch_state.json"


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------

def load_state(out_dir: Path) -> Dict[str, Any]:
    path = out_dir / STATE_FILE
    if not path.exists():
        raise SystemExit(f"No {STATE_FILE} in {out_dir}; run prepare first")
    return json.loads(path.read_text())


def save_state(out_dir: Path, state: Dict[str, Any]) -> None:
    tmp = out_dir / (STATE_FILE + ".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(out_dir / STATE_FILE)


def corpus_files(corpus: Path) -> Dict[str, Path]:
    """The corpus train file and its _val file, when present."""
    files = {"train": corpus}
    val = corpus.with_name(corpus.stem + "_val.jsonl")
    if val.exists():
        files["val"] = val
    return files


def unit_hash(key: str) -> float:
    """Deterministic number in [0, 1) from a string."""
    return int(text_hash(key)[:8], 16) / 2 ** 32


# ----------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------

def make_backend(model: str, max_tokens: int, effort: Optional[str]) -> OpenAIBackend:
    # Only build_request is used, so no client and no key are needed
    return OpenAIBackend(model_name=model, max_tokens=max_tokens,
                         effort=effort, client=object())


def cmd_prepare(args) -> None:
    out_dir = Path(args.out_dir)
    if (out_dir / STATE_FILE).exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} already has a batch; pass --overwrite to replace it")
    corpus = Path(args.corpus)
    if not corpus.exists():
        raise SystemExit(f"Corpus not found: {corpus}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("requests_*.jsonl"):
        old.unlink()

    # Rank documents by a salted hash, so the pick does not depend on the
    # order of the corpus file
    docs = []
    for split, path in corpus_files(corpus).items():
        with open(path) as f:
            for line_no, line in enumerate(f):
                row = json.loads(line)
                h = text_hash(row["text"])
                docs.append((unit_hash("qa-pick:" + h), h, split, line_no, row))
    docs.sort()
    picked = docs[: args.qa_docs]

    backend = make_backend(args.model, args.max_tokens, args.effort)
    selection: Dict[str, Dict[str, Any]] = {}
    chunks: List[Dict[str, Any]] = []
    out = None
    n_requests = 0
    for _, h, split, line_no, row in picked:
        respond = unit_hash("qa-respond:" + h) < args.should_respond_ratio
        selection[h] = {"split": split, "line": line_no, "respond": respond}
        if not respond:
            continue
        if out is None or chunks[-1]["requests"] >= args.chunk_requests:
            if out is not None:
                out.close()
            name = f"requests_{len(chunks):03d}.jsonl"
            chunks.append({"file": name, "requests": 0, "batch_id": None,
                           "status": "pending"})
            out = open(out_dir / name, "w")
        body = backend.build_request(
            get_qa_prompt(row["text"], row.get("entities", []), row.get("relations", [])),
            None, QA_JSON_SCHEMA,
        )
        out.write(json.dumps({"custom_id": h, "method": "POST",
                              "url": ENDPOINT, "body": body}) + "\n")
        chunks[-1]["requests"] += 1
        n_requests += 1
    if out is not None:
        out.close()

    state = {
        "corpus": str(corpus.resolve()),
        "output_name": args.output_name,
        "model": args.model,
        "selection": selection,
        "chunks": chunks,
    }
    save_state(out_dir, state)
    n_abstain = len(selection) - n_requests
    print(f"Picked {len(selection)} documents: {n_requests} QA requests in "
          f"{len(chunks)} file(s), {n_abstain} abstain rows")


# ----------------------------------------------------------------------
# submit / status
# ----------------------------------------------------------------------

def openai_client():
    try:
        import openai
    except ImportError as e:
        raise SystemExit("pip install openai") from e
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY")
    return openai.OpenAI()


def _limit_error(batch) -> bool:
    errors = getattr(getattr(batch, "errors", None), "data", None) or []
    return any("limit" in str(getattr(e, "code", "")) for e in errors)


def refresh(client, state: Dict[str, Any]) -> None:
    """Update chunk status. A chunk refused for the token limit goes back
    to pending, so the next submit sends it again."""
    for chunk in state["chunks"]:
        if not chunk["batch_id"] or chunk["status"] in ("completed", "collected"):
            continue
        batch = client.batches.retrieve(chunk["batch_id"])
        chunk["status"] = batch.status
        chunk["output_file_id"] = getattr(batch, "output_file_id", None)
        chunk["error_file_id"] = getattr(batch, "error_file_id", None)
        counts = getattr(batch, "request_counts", None)
        if counts is not None:
            chunk["counts"] = {"completed": counts.completed,
                               "failed": counts.failed, "total": counts.total}
        if batch.status == "failed" and _limit_error(batch):
            print(f"  {chunk['file']}: over the enqueued-token limit; it waits")
            chunk["batch_id"] = None
            chunk["status"] = "pending"


def cmd_submit(args, client=None, state=None) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    client = client or openai_client()
    state = state or load_state(out_dir)
    refresh(client, state)
    for chunk in state["chunks"]:
        if chunk["status"] != "pending":
            continue
        with open(out_dir / chunk["file"], "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id, endpoint=ENDPOINT, completion_window="24h",
        )
        chunk["batch_id"] = batch.id
        chunk["status"] = batch.status
        print(f"  Submitted {chunk['file']} ({chunk['requests']} requests): {batch.id}")
        save_state(out_dir, state)
    save_state(out_dir, state)
    return state


def cmd_status(args, client=None) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    client = client or openai_client()
    state = load_state(out_dir)
    refresh(client, state)
    save_state(out_dir, state)
    for chunk in state["chunks"]:
        print(f"  {chunk['file']}: {chunk['status']} {chunk.get('counts', '')}")
    return state


# ----------------------------------------------------------------------
# collect
# ----------------------------------------------------------------------

TERMINAL = ("completed", "expired", "cancelled", "collected")


def _file_lines(client, file_id: Optional[str]) -> List[Dict[str, Any]]:
    if not file_id:
        return []
    text = client.files.content(file_id).text
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_batch_line(line: Dict[str, Any]):
    """((question, answer), "") for a usable reply, else (None, reason)."""
    if line.get("error"):
        return None, f"error {line['error'].get('code')}"
    resp = line.get("response") or {}
    if resp.get("status_code") != 200:
        return None, f"status {resp.get('status_code')}"
    choice = (resp.get("body") or {}).get("choices", [{}])[0]
    if choice.get("finish_reason") == "length":
        # Reasoning tokens count against max_completion_tokens
        return None, "finish_reason=length"
    message = choice.get("message") or {}
    if message.get("refusal"):
        return None, "refusal"
    return parse_qa_reply(message.get("content"))


def cmd_collect(args, client=None) -> int:
    out_dir = Path(args.out_dir)
    client = client or openai_client()
    state = load_state(out_dir)
    refresh(client, state)
    waiting = [c["file"] for c in state["chunks"] if c["status"] not in TERMINAL]
    if waiting:
        print(f"Not finished: {waiting}. Run status, or run to wait.")
        save_state(out_dir, state)
        return 2

    qa: Dict[str, Any] = {}
    reasons: Dict[str, int] = {}
    for chunk in state["chunks"]:
        for line in (_file_lines(client, chunk.get("output_file_id"))
                     + _file_lines(client, chunk.get("error_file_id"))):
            result, reason = parse_batch_line(line)
            if result is not None:
                qa[line["custom_id"]] = result
            else:
                reasons[reason] = reasons.get(reason, 0) + 1

    corpus = Path(state["corpus"])
    files = corpus_files(corpus)
    name = state["output_name"]
    outs = {split: open(out_dir / f"{name}{'' if split == 'train' else '_val'}.jsonl", "w")
            for split in ("train", "val")}
    n_respond = n_abstain = n_dropped = 0
    selection = state["selection"]
    for split, path in files.items():
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                h = text_hash(row["text"])
                pick = selection.get(h)
                if pick is None:
                    continue
                if not pick["respond"]:
                    row["should_respond"], row["response"] = 0, ""
                    n_abstain += 1
                elif h in qa:
                    question, answer = qa[h]
                    row["text"] = f"{row['text'].rstrip()}\n\n{question}"
                    row["should_respond"], row["response"] = 1, answer
                    n_respond += 1
                else:
                    n_dropped += 1
                    continue
                outs[pick["split"]].write(json.dumps(row) + "\n")
    for f in outs.values():
        f.close()
    vocab = corpus.with_name(corpus.stem + "_vocab.json")
    if vocab.exists():
        shutil.copy(vocab, out_dir / f"{name}_vocab.json")

    for chunk in state["chunks"]:
        chunk["status"] = "collected"
    save_state(out_dir, state)

    requested = n_respond + n_dropped
    fail_rate = n_dropped / max(requested, 1)
    total = n_respond + n_abstain
    print(f"QA rows: {n_respond} respond, {n_abstain} abstain "
          f"(respond ratio {n_respond / max(total, 1):.2f})")
    print(f"QA requests failed: {n_dropped}/{requested} ({fail_rate:.1%}) {reasons}")
    if fail_rate > FAIL_RATE_LIMIT:
        print("Most QA requests failed; check the reasons above before training.")
        return 1
    return 0


def cmd_run(args) -> int:
    client = openai_client()
    out_dir = Path(args.out_dir)
    state = load_state(out_dir)
    while True:
        state = cmd_submit(args, client, state)
        if all(c["status"] in TERMINAL for c in state["chunks"]):
            return cmd_collect(args, client)
        done = sum(c["status"] in TERMINAL for c in state["chunks"])
        print(f"  {done}/{len(state['chunks'])} batch(es) finished; next check in "
              f"{args.poll_seconds}s")
        time.sleep(args.poll_seconds)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("prepare")
    pr.add_argument("--corpus", required=True, help="GLiNER2 corpus <name>.jsonl")
    pr.add_argument("--out-dir", required=True)
    pr.add_argument("--output-name", default="curated")
    pr.add_argument("--qa-docs", type=int, default=20000,
                    help="Corpus documents in the QA set (respond + abstain)")
    pr.add_argument("--should-respond-ratio", type=float, default=0.7)
    pr.add_argument("--model", default=OpenAIBackend.default_model)
    pr.add_argument("--max-tokens", type=int, default=2048,
                    help="max_completion_tokens; reasoning tokens count too")
    pr.add_argument("--effort", default=OpenAIBackend.default_effort)
    pr.add_argument("--chunk-requests", type=int, default=5000,
                    help="Requests per batch file (API cap: 50000 and 200 MB)")
    pr.add_argument("--overwrite", action="store_true")

    for name in ("submit", "status", "collect", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--out-dir", required=True)
        if name == "run":
            sp.add_argument("--poll-seconds", type=int, default=300)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.cmd == "prepare":
        cmd_prepare(args)
        return 0
    if args.cmd == "submit":
        cmd_submit(args)
        return 0
    if args.cmd == "status":
        cmd_status(args)
        return 0
    if args.cmd == "collect":
        return cmd_collect(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
