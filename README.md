# NeuroSymbolic Language Model

A decoder-only language model that builds a small knowledge graph from its own
prompt, then reads that graph back while it writes the answer.

The model adds new modules to a pretrained causal LM (Qwen3, Gemma 3 or
Llama 3.2):
- An extraction adapter reads a middle layer of the backbone.
- Symbolic heads predict entities, concepts and relations.
- A graph neural network (GNN) refines the entity nodes.
- A gated injector writes the nodes back into the context as soft tokens.

All parts train together in one model, with one forward pass for extraction
and generation.

**Status:** the first full run, `qwen3-0.6b-v1`, trained on an A40 on
2026-10-20 and 2026-10-21. Benchmarks follow on the same pod. See [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
for the data, training details, results and limitations.

## How the model works

```
prompt tokens
    |
    v
Pretrained causal LM (Qwen3 / Gemma 3 / Llama 3.2)
    |
    +-- tapped mid-layer hidden states (tap_layer_ratio, default 0.66)
    |       |
    |       v
    |   Bidirectional extraction adapter (1 transformer layer)
    |       |
    |       +--> Token entity head      (entity type per token)
    |       +--> Concept bank           (soft concept assignment)
    |       +--> Top-K entity nodes --> GNN --> Pairwise relation scorer
    |                                    |          |
    |                                    |      Soft-logic constraints
    |                                    v      (taxonomy type rules)
    |                             Injector: proj -> norm -> x gate
    |                                    |
    v                                    v
[prompt tokens] [K gated node tokens] [response tokens] --> same causal LM --> answer
```

1. **Read.** The backbone runs over the prompt. The model taps one middle
   layer. A small bidirectional adapter reads it, so each token sees the whole
   prompt.
2. **Extract.** The heads predict entity types, concepts and relations. The
   top-K entity positions become graph nodes (K = 16 by default). The GNN refines
   them. Soft logic penalizes relations that the type taxonomy does not allow.
3. **Write.** The injector maps the nodes into the token-embedding space and
   multiplies them by a learned scalar gate. The model puts them as K soft
   tokens between the prompt and the response. The backbone then generates
   as usual.

**Abstention.** A `should_respond=0` sample has the target EOS. The model
learns to stop at once.

**Ablation.** `use_soft_nodes=False` (the same as gate = 0) removes the graph
tokens. Each stage after stage 1 reports the val loss both with and without
the graph tokens.

The original encoder-decoder version (T5/LongT5) is still in the repository.
See [Legacy encoder-decoder model](#legacy-encoder-decoder-model).
[docs/DECODER_ONLY_MIGRATION.md](docs/DECODER_ONLY_MIGRATION.md) explains why
the project moved to a decoder-only design.

## Installation

```bash
# Install torch for your CUDA build first (or use a RunPod PyTorch image)
pip install -r requirements.txt
```

Install vLLM (local LLM QA generation) in a separate environment from
`requirements-vllm.txt`, because vLLM pins its own torch.

## Workflow

### 1. Curate the data

Stage 1 data: documents annotated by
[GLiNER2](https://huggingface.co/fastino/gliner2-base-v1) with entities,
spans, entity types, concepts and relations. No LLM is needed.

```bash
python data/curate_dataset.py --annotator gliner \
    --sources allenai/c4 wikipedia --source-ratios 0.6 0.4 \
    --target-samples 50000 --output-dir data/curated
```

Stage 2/3 data: question/answer pairs on the curated documents, made through the
OpenAI Batch API. The Batch API costs half the normal price, and the job needs no GPU.

```bash
python data/curate_qa_batch.py prepare --corpus data/curated/curated.jsonl \
    --out-dir data/curated_qa --qa-docs 20000 --should-respond-ratio 0.7
python data/curate_qa_batch.py run --out-dir data/curated_qa
```

Other options:
- `curate_dataset.py --annotator hybrid` makes QA pairs with vLLM, Anthropic or
  OpenAI in the same pass.
- `data/curate_traces.py` makes tool-calling traces for the chat phase.

Each command writes `<name>.jsonl`, `<name>_val.jsonl` and
`<name>_vocab.json`. [DATASET_FORMAT.md](DATASET_FORMAT.md) gives the schema.

### 2. Train

```bash
python train_causal.py --preset qwen3-0.6b \
    --dataset data/curated/curated.jsonl \
    --stage2-dataset data/curated_qa/curated.jsonl \
    --epochs 2 --batch-size 4 \
    --output-dir ckpt/qwen3-0.6b --log-dir runs/qwen3-0.6b
```

| Stage | Trains | Frozen | Data | LR |
|---|---|---|---|---|
| 1. `symbolic` | adapter, heads, GNN | backbone, injector | `--dataset` | `--symbolic-lr` (1e-4) |
| 2. `decoder` | backbone, injector | symbolic modules | `--stage2-dataset` | `--lr` x 0.5 |
| 3. `joint` | all modules | none | `--stage2-dataset` | `--lr` x 0.1 |
| chat (`--phase chat`) | all modules | none | `--chat-dataset` | `--chat-lr` |

How the stages run:
- Each stage uses `<file>_val.jsonl` next to its training file for validation, and stops early on val loss (`--patience`).
- Stage 1 reports entity F1 and relation F1.
- Stages 2 and 3 report `val` and `val_gate0`. `val_gate0` is the val loss with the graph tokens off.
- The injection gate has its own learning rate (`--gate-lr`, 1e-3) in every stage.

Useful options:
- `--resume ckpt/.../latest.pt` continues after a crash. Add `--stages` to name the stages that remain.
- `--hub-repo user/repo --hub-path name` pushes each stage checkpoint, the final model and the tokenizer to a private Hugging Face repository.
- `torchrun --nproc_per_node N train_causal.py ...` trains on several GPUs. `--batch-size` is per GPU.
- `--preset causal-testing` uses a tiny random backbone with the real Qwen3 tokenizer, for smoke runs on a laptop.

Outputs in `--output-dir`:
- `stage_<name>.pt`: the checkpoint at the end of each stage
- `latest.pt`: the last epoch of the stage that is running
- `final_causal_model.pt`: the final model
- `tokenizer/`

Each checkpoint holds the weights, the config and the label maps. To load one,
use `train_causal.load_causal_checkpoint(path)`.

### 3. Evaluate

`eval_harness.py` wraps a checkpoint for
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
It supports multiple-choice, perplexity and generation tasks.

```bash
# This model, graph tokens on
python eval_harness.py --checkpoint ckpt/qwen3-0.6b/final_causal_model.pt \
    --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq \
    --output-dir results/ours

# Same checkpoint, graph tokens off
python eval_harness.py ... --no-soft-nodes --output-dir results/nonodes

# Baseline: the plain backbone
lm_eval --model hf --model_args pretrained=Qwen/Qwen3-0.6B-Base \
    --tasks arc_easy,... --output_path results/base
```

With the graph tokens off, the wrapper gives the same scores as lm-eval's `hf`
backend on the same weights. A test on 20 ARC-Easy and 20 HellaSwag questions
gave a largest log-likelihood difference of 9e-5.

### 4. Run on RunPod

[docs/RUNPOD.md](docs/RUNPOD.md) is the runbook for curation and training on a
RunPod GPU pod. It covers these topics:
- GPU size for each preset
- `scripts/runpod_setup.sh`
- runs that continue after you disconnect, with a network volume
- the smoke test
- the log values to monitor

## Presets

| Preset | Backbone | Notes |
|---|---|---|
| `qwen3-0.6b` | Qwen/Qwen3-0.6B-Base | full fine-tune. 22.7 GB peak at batch 4 in the v1 run (A40) |
| `qwen3-1.7b` | Qwen/Qwen3-1.7B-Base | full fine-tune, gradient checkpointing |
| `qwen3-1.7b-lora` | Qwen/Qwen3-1.7B-Base | LoRA on the backbone, fits a 24 GB card |
| `qwen3-4b-lora` | Qwen/Qwen3-4B-Base | LoRA on the backbone |
| `gemma3-1b` | google/gemma-3-1b-pt | full fine-tune |
| `llama3.2-1b` | meta-llama/Llama-3.2-1B | full fine-tune |
| `causal-testing` | tiny random Llama | offline smoke runs and tests |

All causal presets use 4,096 prompt tokens and 512 response tokens. Change
these limits with `--max-input-length` and `--max-output-length`.

## Dataset format

Training data is JSONL. [DATASET_FORMAT.md](DATASET_FORMAT.md) gives the full specification.

```json
{
    "text": "Paris is the capital of France.\n\nWhat is the capital of France?",
    "entities": ["Paris", "France"],
    "entity_spans": [[0, 5], [24, 30]],
    "entity_types": ["location", "location"],
    "concepts": [["city"], ["country"]],
    "relations": [[0, 1, "capital_of"]],
    "should_respond": 1,
    "response": "Paris."
}
```

## Project structure

```
train_causal.py              Staged training for the decoder-only model
eval_harness.py              lm-evaluation-harness wrapper
config.py                    Model presets (CausalModelConfig) and training configs
DATASET_FORMAT.md            Dataset specification

model/
    neurosymbolic_causal.py  NeuroSymbolicCausalLM (read / extract / write)
    extraction_adapter.py    Bidirectional adapter on the tapped layer
    injection.py             Gated soft-token injector
    entity.py                Entity heads, concept bank, node scoring
    gnn.py                   Graph neural networks
    logic.py                 Soft-logic constraints
    neurosymbolic.py         Legacy encoder-decoder model (T5/LongT5)

data/
    curate_dataset.py        GLiNER2 / hybrid LLM curation
    curate_qa_batch.py       QA generation through the OpenAI Batch API
    curate_traces.py         Tool-calling traces for the chat phase
    curate/                  Annotators, taxonomy, LLM backends, Hub upload
    collator_causal.py       Batching for the causal model
    collator.py              Shared prompt-side batching
    chat_template.py         Chat format and special tokens

training/                    Stage trainers and evaluation utilities
continual_learning/          EWC, SI, LwF, episodic memory (legacy model)
scripts/runpod_setup.sh      Pod setup
docs/                        Model card, RunPod runbook, migration notes
tests/                       pytest suite (offline, tiny backbone)
```

## Tests

```bash
pytest tests/
```

The tests run offline. They use a tiny random backbone with the real Qwen3
tokenizer from the Hugging Face cache.

## Legacy encoder-decoder model

The first version used a T5/LongT5 encoder-decoder (`model/neurosymbolic.py`,
`train.py`). The decoder cross-attended to the encoder states and the graph
nodes. This version is still in the repository, but it is not developed now.

```bash
python train.py --preset testing --epochs 2 --debug
```

`continual_learning/` (EWC, SI, LwF, episodic replay) uses the legacy model.
`examples/continual_learning_example.py` shows how to use it.

## License

No license is chosen yet. Model weights trained from Qwen3 follow the base
model's Apache 2.0 license.
