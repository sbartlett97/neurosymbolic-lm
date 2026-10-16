# RunPod runbook: dataset generation and training

This runbook takes a fresh RunPod pod to a curated dataset and a trained
`NeuroSymbolicCausalLM`. All commands run on the pod unless noted.

## 1. Pick the pod

| Job | Preset / mode | Recommended GPU | Notes |
|---|---|---|---|
| GLiNER2 curation (`--annotator gliner`) | — | 24 GB+ (L4 / A5000 / 4090) | Documents go up to 16k characters, and GLiNER2 encodes all of each one. Memory grows with the square of the length. |
| Hybrid QA curation (`--annotator hybrid`, 8B LLM) | vLLM + GLiNER2 | 80 GB (A100 / H100) | 48 GB also fits. Long documents make long QA prompts, so leave room for the vLLM KV cache. |
| Hybrid QA through an API | `--backend anthropic` / `openai` | as GLiNER2 curation | The LLM runs at the provider |
| Training | `qwen3-0.6b` (full FT) | 48 GB (L40S / A6000) | 24 GB works with `--batch-size 1-2 --grad-accum 4` |
| Training | `qwen3-1.7b-lora` | 48 GB | bf16 base, LoRA adapters, chat-token rows train |
| Training | `qwen3-1.7b` (full FT) | 80 GB (A100 / H100) | ≈ 27 GB for weights + optimizer state alone |
| Training | `qwen3-4b-lora` | 80 GB | |
| Training | `gemma3-1b`, `llama3.2-1b` | 48 GB | gated: needs `HF_TOKEN` |

All training presets use a 4096-token context (`--max-input-length`). It
holds a full 16k-character curated document (about 4.3 characters per Qwen
token) plus its QA question. If you raise `--max-doc-length` in curation,
raise `--max-input-length` to match. Otherwise the start of long documents
is cut, and its entities are dropped from the labels.

- Template: an official **RunPod PyTorch** template with Python ≥ 3.11 and
  a CUDA torch build, torch ≥ 2.6. The setup script reuses that
  torch, and it stops if the venv ends up with a different torch.
- Storage: attach a **network volume** (100 GB+) at `/workspace`. Data,
  checkpoints, venvs, and the HF cache all go there. The container disk
  is wiped on stop.
- Checkpoint size: each `.pt` holds the full model in fp32. That is about
  2.5 GB for 0.6B and 7 GB for 1.7B. One run keeps one file per stage,
  plus `latest.pt` and `final_causal_model.pt`.

## 2. Set up (once per volume)

```bash
cd /workspace
git clone <your-repo-url> neurosymbolic-lm
cd neurosymbolic-lm
export HF_TOKEN=hf_...          # needed for Llama / Gemma; optional for Qwen
bash scripts/runpod_setup.sh    # add WITH_VLLM=1 for hybrid QA curation
. /workspace/neurosym.env && . /workspace/venv-train/bin/activate
```

The script does these steps:
1. It puts `HF_HOME` and the pip/uv caches on `/workspace`.
2. It installs `tmux`.
3. It makes `/workspace/venv-train` with `--system-site-packages`, which
   inherits the template torch.
4. It installs `requirements.txt` and prints the GPU and bf16 support.
5. It runs the offline test suite.

vLLM goes in its own `/workspace/venv-vllm`, because vLLM pins an exact
torch build.

## 3. Run long jobs detached

An SSH disconnect kills a plain background job. Use tmux:

```bash
tmux new -s curate        # detach: Ctrl-b d   reattach: tmux attach -t curate
```

Or use `setsid` from a one-shot SSH command:

```bash
setsid bash -c '<command>' > /workspace/logs/job.log 2>&1 < /dev/null &
tail -f /workspace/logs/job.log
```

## 4. Generate the datasets

Every curation command writes three files:
- `<name>.jsonl` (train)
- `<name>_val.jsonl` (deterministic hash split)
- `<name>_vocab.json`

`train_causal.py` uses the vocab file next to `--dataset`. Each stage
validates on the `_val.jsonl` next to its own training file: Stage 1 on the
corpus, decoder/joint on `--stage2-dataset`, chat on `--chat-dataset`.

A fresh run refuses to write into a non-empty output directory. Pass
`--overwrite` to start over, or `--resume-from` to continue.

**4a. Stage-1 symbolic corpus (GLiNER2 only):**

```bash
python data/curate_dataset.py --annotator gliner \
    --sources allenai/c4 wikipedia --source-ratios 0.6 0.4 \
    --target-samples 50000 --output-dir /workspace/data/curated
```

**4b. Stage-2/3 QA data (GLiNER2 + vLLM). Use the vLLM environment:**

```bash
. /workspace/venv-vllm/bin/activate
python data/curate_dataset.py --annotator hybrid --backend vllm \
    --llm meta-llama/Llama-3.1-8B-Instruct \
    --should-respond-ratio 0.7 --target-samples 20000 \
    --output-dir /workspace/data/curated_qa
```

At the end of the run, check the `qa_failed` rate in the printed stats. A
high rate means the LLM output did not parse as QA JSON.

**4b (alternative). QA through a hosted API (no GPU needed for the LLM):**

Use the training environment (`venv-train`). It has the `anthropic` and
`openai` SDKs; `venv-vllm` does not.

```bash
. /workspace/venv-train/bin/activate
export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY with --backend openai

# 1. Live smoke run: 5 samples, costs almost nothing
python data/curate_dataset.py --annotator hybrid --backend anthropic \
    --should-respond-ratio 1.0 --target-samples 5 \
    --output-dir /workspace/data/qa_smoke --overwrite
grep -c '"should_respond": 1' /workspace/data/qa_smoke/curated*.jsonl

# 2. The real run
python data/curate_dataset.py --annotator hybrid --backend anthropic \
    --api-concurrency 8 --should-respond-ratio 0.7 --target-samples 20000 \
    --output-dir /workspace/data/curated_qa
```

- The smoke run must give `should_respond=1` rows. If a request is
  rejected (bad model name, key, or schema), the first rejection message
  is printed. If more than 80% of the first 50 QA attempts fail, the run
  stops, because failed documents would become abstain rows.

- Default models: `claude-haiku-4-5` (anthropic) and `gpt-6-luna`
  (openai), the lowest-cost current models with structured output. Pick
  another with `--llm`.
- Keys come only from the environment. They are never logged or saved.
- **Check the provider terms first.** Anthropic's and OpenAI's terms limit
  the use of outputs to train competing models. This project trains a
  language model on the QA outputs, so get a legal review before a large
  run.

GLiNER2 throughput: `--gliner-batch-size` (default 1) sets how many texts
go through one forward pass. The output is the same at any value. Run a
200-sample job at 1 and at 8 on the pod, and keep the faster one.

Corpus prose does not use the `code_construct` or `digital_artifact`
labels. One surface form has one entity type within each document.

**4c. Assistant traces:**

```bash
python data/curate_traces.py --source glaiveai/glaive-function-calling-v2 \
    --target-samples 20000 --output-dir /workspace/data/traces
```

- Every assistant turn is a sample, so tool calls are decoder targets.
  About 22% of glaive targets are tool calls.
- `--target-samples` counts samples after expansion. Glaive gives about
  3.2 samples per conversation, so 20000 samples is about 6300
  conversations.
- The causal collator cuts long prompts from the left, so the system
  prompt (the function definitions) would go first. In a 200-sample glaive
  run the longest prompt was 1169 tokens, so nothing is cut at the default
  4096 limit. Keep `--max-input-length` at 2048 or more for traces.

**Before you train, check the data:**

```bash
wc -l /workspace/data/*/*.jsonl
python - <<'EOF'
import json
rows = [json.loads(l) for l in open("/workspace/data/curated_qa/curated.jsonl")]
print("respond ratio:", sum(r["should_respond"] for r in rows) / len(rows))
bad = sum(r["text"][s:e] != ent for r in rows
          for ent, (s, e) in zip(r["entities"], r["entity_spans"]))
print("span mismatches:", bad)
EOF
```

## 5. Smoke test before the long run

Run this first on every new pod. It takes about two minutes and checks
the real GPU path end to end (bf16 autocast, gradient checkpointing,
logits_to_keep, checkpoint save):

```bash
python train_causal.py --preset qwen3-0.6b \
    --dataset /workspace/data/curated/curated.jsonl \
    --stage2-dataset /workspace/data/curated_qa/curated.jsonl \
    --max-samples 64 --max-eval-samples 16 --epochs 1 --log-every 5 \
    --output-dir /workspace/checkpoints/smoke --log-dir /workspace/runs
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

On a pod with several GPUs, also run the multi-GPU path once. It is the
only check of NCCL:

```bash
torchrun --nproc_per_node 2 train_causal.py --preset qwen3-0.6b \
    --dataset /workspace/data/curated/curated.jsonl \
    --stage2-dataset /workspace/data/curated_qa/curated.jsonl \
    --max-samples 64 --max-eval-samples 16 --epochs 1 --log-every 5 \
    --output-dir /workspace/checkpoints/smoke_ddp --log-dir /workspace/runs
```

Ranks 1.. wait while rank 0 validates and saves. The collective timeout is
120 minutes; set `NEUROSYM_DIST_TIMEOUT_MIN` to change it.

The run passes when:
- every stage prints a finite `train=` and `val=`
- `stage_*.pt` files appear
- peak memory leaves headroom for your real `--batch-size`

## 6. Train

```bash
python train_causal.py --preset qwen3-0.6b \
    --dataset /workspace/data/curated/curated.jsonl \
    --stage2-dataset /workspace/data/curated_qa/curated.jsonl \
    --chat-dataset /workspace/data/traces/traces.jsonl --phase both \
    --epochs 3 --batch-size 8 \
    --output-dir /workspace/checkpoints/q06b --log-dir /workspace/runs \
    2>&1 | tee /workspace/logs/train_q06b.log
```

- Stage 1 trains only the adapter and heads at `--symbolic-lr`. Stage 2
  trains at `--lr`×0.5, joint at `--lr`×0.1, chat at `--chat-lr`.
- `--batch-size` is per GPU. Effective batch = `--batch-size` ×
  `--grad-accum` × number of GPUs. Use `--grad-accum` to reach a larger
  batch on one card.
- `--soft-logic-weight` (default 0.1, 0 = off) adds the taxonomy's
  relation domain/range constraint. Example: `born_in` must have a person
  head and a location tail. Only the relation scorer learns from it.
- A data path that does not exist stops the run. A decoder stage with no
  `should_respond=1` samples also stops the run.
- To continue after a crash, pass `--resume <dir>/stage_<last>.pt` with
  the remaining `--stages`. Resume takes its config and label maps from
  the checkpoint, and it refuses a checkpoint whose keys do not match the
  model.

### Several GPUs on one pod

```bash
torchrun --nproc_per_node 4 train_causal.py --preset qwen3-1.7b \
    --dataset ... --stage2-dataset ... --batch-size 4 --grad-accum 2 \
    --output-dir /workspace/checkpoints/q17b --log-dir /workspace/runs
```

- Each GPU holds a full model copy (DDP), so the per-preset VRAM table
  above still applies per GPU.
- Rank 0 alone runs validation, logs, and saves. It decides early
  stopping and sends the decision to the other ranks.
- One node only. Multi-node runs are not tested.

### What to watch (the migration-plan gates)

| Log value | Stage | Gate |
|---|---|---|
| `entity_f1`, `relation_f1` | Symbolic | Phase 1: within ~2 points of the encoder-decoder baseline |
| `val` vs `val_gate0` | Decoder / Joint / Chat | Phase 2a: `val_gate0` must be higher than `val`. If they are equal, the injected nodes are a no-op. |
| `abstain_acc` | Decoder / Joint / Chat | Phase 2b: share of `should_respond=0` val samples that emit EOS first |
| `gate=` | all | Injection gate value. If it falls to ~0, the model has learned to ignore the nodes. |

TensorBoard: `tensorboard --logdir /workspace/runs --port 6006`. Expose
port 6006 on the pod, or copy `runs/` to your machine.

## 7. Get results off the pod

```bash
# from your machine
runpodctl receive <code>     # after: runpodctl send /workspace/checkpoints/q06b
```

Load a checkpoint anywhere:

```python
from train_causal import load_causal_checkpoint
model, tokenizer, ckpt = load_causal_checkpoint("final_causal_model.pt", device="cuda")
```

`load_causal_checkpoint` reads the `tokenizer/` folder next to the
checkpoint, which holds the added chat tokens.

## Known limits

- The decoder stage builds nodes from top-K entity scores, which is the
  inference path. Joint and chat feed gold-span nodes (teacher forcing).
