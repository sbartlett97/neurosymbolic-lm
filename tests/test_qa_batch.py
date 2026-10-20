"""OpenAI Batch QA pipeline (data/curate_qa_batch.py) with a fake client."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import data.curate_qa_batch as qb  # noqa: E402
from data.curate.llm_annotator import OpenAIBackend  # noqa: E402
from data.curate.prompts import QA_JSON_SCHEMA, get_qa_prompt  # noqa: E402


def write_corpus(tmp_path, n_train=12, n_val=4):
    def row(i):
        text = f"Document {i} says Marie Curie worked in Paris in {1900 + i}."
        start = text.index("Marie Curie")
        return {"text": text, "entities": ["Marie Curie"],
                "entity_spans": [[start, start + len("Marie Curie")]],
                "entity_types": ["person"], "concepts": [["scientist"]],
                "relations": [], "should_respond": 0, "response": ""}
    corpus = tmp_path / "corpus" / "curated.jsonl"
    corpus.parent.mkdir()
    corpus.write_text("".join(json.dumps(row(i)) + "\n" for i in range(n_train)))
    (corpus.parent / "curated_val.jsonl").write_text(
        "".join(json.dumps(row(100 + i)) + "\n" for i in range(n_val)))
    (corpus.parent / "curated_vocab.json").write_text(json.dumps({"concepts": {}, "relations": {}}))
    return corpus


class FakeClient:
    """Files + batches API. Batches complete on the first retrieve."""

    def __init__(self, reply=None, limit_first=False):
        self.uploaded, self.batches_made = {}, {}
        self.reply = reply or (lambda cid: {"question": f"Who? {cid[:4]}", "answer": "Curie"})
        self.limit_first = limit_first
        outer = self

        class Files:
            def create(self, file, purpose):
                assert purpose == "batch"
                fid = f"file-{len(outer.uploaded)}"
                outer.uploaded[fid] = file.read().decode()
                return SimpleNamespace(id=fid)

            def content(self, fid):
                return SimpleNamespace(text=outer.uploaded[fid])

        class Batches:
            def create(self, input_file_id, endpoint, completion_window):
                assert endpoint == "/v1/chat/completions" and completion_window == "24h"
                bid = f"batch-{len(outer.batches_made)}"
                outer.batches_made[bid] = input_file_id
                return SimpleNamespace(id=bid, status="validating")

            def retrieve(self, bid):
                if outer.limit_first and bid == "batch-0":
                    err = SimpleNamespace(data=[SimpleNamespace(code="token_limit_exceeded")])
                    return SimpleNamespace(status="failed", errors=err, output_file_id=None,
                                           error_file_id=None, request_counts=None)
                lines = []
                for req in outer.uploaded[outer.batches_made[bid]].splitlines():
                    cid = json.loads(req)["custom_id"]
                    reply = outer.reply(cid)
                    if reply is None:
                        lines.append({"custom_id": cid, "error": {"code": "server_error"}})
                        continue
                    body = {"choices": [{"finish_reason": "stop",
                                         "message": {"content": json.dumps(reply)}}]}
                    lines.append({"custom_id": cid, "error": None,
                                  "response": {"status_code": 200, "body": body}})
                out = f"file-out-{bid}"
                outer.uploaded[out] = "\n".join(json.dumps(x) for x in lines)
                return SimpleNamespace(status="completed", errors=None, output_file_id=out,
                                       error_file_id=None, request_counts=None)

        self.files, self.batches = Files(), Batches()


def prepare(tmp_path, **kw):
    corpus = write_corpus(tmp_path)
    out = tmp_path / "qa"
    argv = ["prepare", "--corpus", str(corpus), "--out-dir", str(out),
            "--qa-docs", str(kw.get("qa_docs", 10)), "--should-respond-ratio", "0.7",
            "--chunk-requests", str(kw.get("chunk", 100))]
    assert qb.main(argv) == 0
    return corpus, out


def test_batch_body_matches_sync_request(tmp_path):
    corpus, out = prepare(tmp_path)
    req = json.loads((out / "requests_000.jsonl").read_text().splitlines()[0])
    row = next(json.loads(l) for l in open(corpus)
               if qb.text_hash(json.loads(l)["text"]) == req["custom_id"]) \
        if any(qb.text_hash(json.loads(l)["text"]) == req["custom_id"] for l in open(corpus)) \
        else next(json.loads(l) for l in open(corpus.with_name("curated_val.jsonl"))
                  if qb.text_hash(json.loads(l)["text"]) == req["custom_id"])
    sync = OpenAIBackend(client=object(), max_tokens=2048).build_request(
        get_qa_prompt(row["text"], row["entities"], row["relations"]), None, QA_JSON_SCHEMA)
    assert req["body"] == sync and req["url"] == "/v1/chat/completions"


def test_collect_builds_rows_keeps_split_and_drops_failures(tmp_path):
    corpus, out = prepare(tmp_path, qa_docs=16)
    state = json.loads((out / "batch_state.json").read_text())
    respond = [h for h, s in state["selection"].items() if s["respond"]]
    failed = respond[0]
    client = FakeClient(reply=lambda cid: None if cid == failed else
                        {"question": "Who worked in Paris?", "answer": "Marie Curie."})
    args = qb.parse_args(["submit", "--out-dir", str(out)])
    qb.cmd_submit(args, client)
    assert qb.cmd_collect(args, client) == 0

    rows = {}
    for name, split in (("curated.jsonl", "train"), ("curated_val.jsonl", "val")):
        for line in open(out / name):
            r = json.loads(line)
            rows[qb.text_hash(r["text"].split("\n\n")[0])] = (split, r)
    sel = state["selection"]
    assert failed not in rows  # dropped, not turned into an abstain row
    assert len(rows) == len(sel) - 1
    for h, (split, r) in rows.items():
        assert split == sel[h]["split"]
        if sel[h]["respond"]:
            assert r["should_respond"] == 1 and r["text"].endswith("Who worked in Paris?")
            s, e = r["entity_spans"][0]
            assert r["text"][s:e] == "Marie Curie"  # spans still valid
        else:
            assert r["should_respond"] == 0
    assert (out / "curated_vocab.json").exists()


def test_token_limit_chunk_is_resubmitted(tmp_path):
    _, out = prepare(tmp_path, qa_docs=16, chunk=3)
    client = FakeClient(limit_first=True)
    args = qb.parse_args(["submit", "--out-dir", str(out)])
    state = qb.cmd_submit(args, client)
    first = state["chunks"][0]
    assert first["batch_id"] == "batch-0"
    state = qb.cmd_submit(args, client, state)  # refresh sees the limit error
    assert state["chunks"][0]["batch_id"] not in (None, "batch-0")


def test_collect_waits_for_unfinished_batches(tmp_path):
    _, out = prepare(tmp_path)

    class Pending(FakeClient):
        pass

    client = Pending()
    args = qb.parse_args(["submit", "--out-dir", str(out)])
    qb.cmd_submit(args, client)
    client.batches.retrieve = lambda bid: SimpleNamespace(
        status="in_progress", errors=None, output_file_id=None,
        error_file_id=None, request_counts=None)
    assert qb.cmd_collect(args, client) == 2


def test_prepare_refuses_existing_batch(tmp_path):
    corpus, out = prepare(tmp_path)
    with pytest.raises(SystemExit):
        qb.main(["prepare", "--corpus", str(corpus), "--out-dir", str(out)])


def test_hub_push_one_commit_allow_list(tmp_path, monkeypatch):
    import huggingface_hub
    from data.curate.hub import push_files

    commits = []

    class FakeApi:
        def create_repo(self, repo_id, repo_type, private, exist_ok):
            assert private is True

        def create_commit(self, repo_id, repo_type, operations, commit_message):
            commits.append([op.path_in_repo for op in operations])

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    (tmp_path / "a.jsonl").write_text("x")
    ok = push_files("u/r", {"c/a.jsonl": tmp_path / "a.jsonl",
                            "c/missing.json": tmp_path / "missing.json"})
    assert ok and commits == [["c/a.jsonl"]]

    class Broken(FakeApi):
        def create_commit(self, **kw):
            raise ConnectionError("down")

    monkeypatch.setattr(huggingface_hub, "HfApi", Broken)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert push_files("u/r", {"c/a.jsonl": tmp_path / "a.jsonl"}) is False
    with pytest.raises(RuntimeError):
        push_files("u/r", {"c/a.jsonl": tmp_path / "a.jsonl"}, required=True)
