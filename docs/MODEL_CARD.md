---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B-Base
language:
  - en
library_name: pytorch
pipeline_tag: text-generation
tags:
  - neurosymbolic
  - knowledge-graph
  - graph-neural-network
  - entity-extraction
  - relation-extraction
  - extractive-qa
  - abstention
---

# NeuroSymbolic Causal LM: qwen3-0.6b-v1

A 0.6B-parameter decoder-only language model. It extracts the entities and
relations in its prompt and builds a small graph from them. It then puts the
graph back into its own context before it writes the answer.

The base model is [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base).
This version comes from the first full training run. It is a research
checkpoint, not a product.

| | |
|---|---|
| Base model | Qwen/Qwen3-0.6B-Base (pretrained weights) |
| New modules | extraction adapter, entity / concept / relation heads, GNN, gated injector, soft-logic constraints (random init) |
| Parameters | 618.7M total (Qwen3-0.6B-Base plus 22.4M in the new modules) |
| Context | 4,096 prompt tokens, 512 response tokens |
| Training data | 50k annotated documents (C4 + Wikipedia), 20k extractive QA pairs |
| Training hardware | 1x NVIDIA A40 48 GB, about 20 h |
| Code | train_causal.py in the project repository |

## How the model works

The model has three steps for each prompt:

1. **Read.** The backbone processes the prompt. The model taps the hidden
   states of layer 18 of 28. A small bidirectional adapter (1 transformer
   layer) reads these states.
2. **Extract.** Heads on the adapter output predict:
   - an entity type for each token
   - concept assignments
   - relations between entity pairs

   The heads hold 24 entity types, 1,024 concepts and 128 relation types. The
   training data uses 17 entity types, 132 concepts and 52 relation types.

   The model selects the top 16 entity positions as graph nodes. A GNN refines
   the node features. Soft-logic constraints penalize relations that the type
   taxonomy does not allow.
3. **Write.** A projection maps the 16 node features into the token-embedding
   space. A learned scalar gate scales them. The model inserts them as 16 soft
   tokens between the prompt and the response:

   ```
   [prompt tokens] [16 gated node tokens] [response tokens]
   ```

The backbone then predicts the response as a normal causal LM.

**Abstention.** If the prompt is a document with no question, the model emits
EOS as its first token. The training data has no unanswerable questions, so the
model did not learn to refuse a question that the document cannot answer.

**Ablation.** Set the gate to 0 (`use_soft_nodes=False`) to remove the graph
signal. The difference between the two settings shows how much the graph helps.

## Training data

| Set | Rows (train / val) | How it was made |
|---|---|---|
| Symbolic corpus | 49,009 / 991 | 50k documents, 60% from `allenai/c4` and 40% from Wikipedia. Each document has at most 16,000 characters. [GLiNER2](https://huggingface.co/fastino/gliner2-base-v1) (`fastino/gliner2-base-v1`) annotated the entities, entity types and relations. |
| QA set | 19,598 / 402 | 20k extractive question/answer pairs on curated documents. `gpt-6-luna` wrote them through the OpenAI Batch API. 13,766 rows (respond) are a document plus a question, with the answer as target. 6,234 rows (abstain) are a document with no question, and the target is EOS. A check found 0 answer spans outside their documents. |

Both sets are in a private dataset repository (`sbartlett97/neurosymbolic-curated`).

**QA prompt format.** The prompt is the document, then a blank line, then the question.
The target is a short answer, then EOS:

```
<document text>

What is the approximate distance between the airports in Vienna and East 34th St Heliport?
```
→ `About 6,798 km.<EOS>`

## Training procedure

There are three stages, with 2 epochs each. Batch size is 4. The optimizer is AdamW with
linear warmup (10%) and linear decay. The injection gate has its own learning
rate (1e-3) and no weight decay.

| Stage | Trains | Frozen | Trainable params | Data | LR | Steps |
|---|---|---|---|---|---|---|
| 1. Symbolic | adapter, heads, GNN | backbone, injector | 22.4M | symbolic corpus | 1e-4 | 24,504 |
| 2. Decoder | backbone, injector | symbolic modules | 596.3M | QA set | 2.5e-5 | 9,798 |
| 3. Joint | all modules | none | 618.7M | QA set | 5e-6 | 9,798 |

**Compute:**
- Hardware: 1x A40 48 GB. fp32 weights and AdamW states, bf16 autocast.
- Peak GPU memory: 22.7 GB.
- Speed: about 1.1 steps/s in stage 1, about 0.42 steps/s in stages 2 and 3.
- Wall time: about 20 h. Cost: about $12 at $0.59/hr.

## Results

### Validation during training

Each validation set holds at most 500 rows of the stage's val file.

| Stage | Epoch | Train loss | Val loss | Val loss, gate = 0 | Other |
|---|---|---|---|---|---|
| Symbolic | 1 | 1.2132 | 0.9221 | n/a | entity F1 0.508, relation F1 0.192 |
| Symbolic | 2 | 0.8190 | 0.8610 | n/a | entity F1 0.522, relation F1 0.222 |
| Decoder | 1 | 0.6225 | 0.5544 | 0.5657 | abstain acc 1.000, respond rate 1.000 |
| Decoder | 2 | 0.2014 | 0.5837 | 0.5988 | abstain acc 1.000, respond rate 1.000 |
| Joint | 1 | 0.8744 | 1.5296 | 1.5521 | abstain acc 1.000, respond rate 1.000 |
| Joint | 2 | 0.6714 | 1.7103 | 1.7472 | n/a |

How to read the table:

- **Gate = 0** removes the graph tokens. The val loss is lower with the graph in
  every stage after stage 1. The gap grows from 0.011 (decoder epoch 1) to 0.037
  (joint epoch 2). The model uses the graph signal, but the gain is small.
- **Joint val loss** is the sum of the extraction, relation, logic and response
  losses. Do not compare it with the decoder-stage val loss.
- **Abstain acc / respond rate** only examine the first generated token: EOS or not
  EOS. They do not measure answer quality. A score of 1.000 shows that the
  model emits EOS when the prompt has no question, and answers when it has one.
  This is an easy cue: the prompt ends with a question or it does not.

### Benchmarks

These runs use lm-evaluation-harness through `eval_harness.py`. Each
benchmark compares the base model (Qwen/Qwen3-0.6B-Base) with v1 and v2
(see [v2](#v2-lora-run-with-unanswerable-questions)), with and without the
graph tokens. "Off" means gate = 0 (`--no-soft-nodes`).

| Task | Shots | Metric | Base | v1 | v1, off | v2 | v2, off |
|---|---|---|---|---|---|---|---|
| MMLU | 5 | acc | **52.5** | 35.3 | 39.8 | 49.1 | 50.8 |
| ARC-Easy | 0 | acc | 65.3 | 57.6 | 59.2 | 64.1 | **66.5** |
| ARC-Easy | 0 | acc_norm | 58.0 | 51.3 | 55.0 | 55.5 | **64.8** |
| ARC-Challenge | 0 | acc_norm | **38.1** | 33.5 | 34.0 | 37.4 | 38.0 |
| HellaSwag | 0 | acc_norm | **53.8** | 39.2 | 40.5 | 50.8 | 52.3 |
| PIQA | 0 | acc_norm | **69.8** | 63.0 | 64.4 | 69.1 | 69.6 |
| WinoGrande | 0 | acc | 58.7 | 49.9 | 51.9 | 57.6 | **59.4** |
| BoolQ | 0 | acc | **70.0** | 64.0 | 61.9 | 63.1 | 61.4 |

MMLU by group (5-shot acc):

| Group | Base | v1 | v1, off | v2 | v2, off |
|---|---|---|---|---|---|
| Humanities | 45.0 | 32.1 | 35.8 | 43.0 | 44.7 |
| STEM | 49.6 | 34.8 | 39.0 | 47.2 | 47.6 |
| Social sciences | 62.1 | 38.9 | 44.3 | 56.3 | 59.5 |
| Other | 57.3 | 36.9 | 42.2 | 53.1 | 54.4 |

SQuAD v2 (0-shot, the first 2,000 validation questions):

| Metric | Base | v1 | v1, off | v2 | v2, off |
|---|---|---|---|---|---|
| exact | 22.75 | 14.15 | 14.05 | **23.75** | 21.20 |
| f1 | 29.21 | 26.65 | 26.68 | **36.55** | 34.28 |
| HasAns_f1 | 58.25 | 52.55 | 52.99 | 65.21 | **65.46** |
| NoAns_exact | 0.00 | 0.60 | 0.20 | **7.72** | 2.91 |
| best_f1 | 50.07 | 50.34 | 51.17 | **54.14** | 53.32 |

How to read the tables:

- **Precision.** The base and v1 runs use fp32. The v2 runs use bf16. In a
  check, bf16 moved scores by about 1 point or less.
- **SQuAD generation limit.** The v1 and v2 runs stop after 256 new tokens.
  The base run used the lm-eval default of 2,048 new tokens: the hf backend
  ignored `max_gen_toks=256`. The score uses only the answer text, so the
  results stay comparable. A long answer is wrong with either limit.
- **v1** lost 5–17 points against the base model. Full fine-tuning on a narrow
  QA set damaged the general ability of the backbone.
- **v2** is within 2 points of the base model on 6 of 7 tasks with the graph
  off. BoolQ is the exception: it is 8.6 points lower, and v2 did not improve
  it over v1.
- **Graph on vs off.** The graph lowers most general benchmarks by 0.5–9
  points. It raises BoolQ by 1.7 points and SQuAD f1 by 2.3 points. The SQuAD
  gain comes from unanswerable questions: the model abstains on 7.7% of them
  with the graph and on 2.9% without it.

## v2: LoRA run with unanswerable questions

v2 keeps the v1 architecture and the v1 stage 1 weights. It changes the
backbone training and the stage 2/3 data, to keep the general ability of the
base model.

**Changes from v1:**

1. **LoRA backbone.** Preset `qwen3-0.6b-lora`: bf16 base weights, LoRA rank
   16, alpha 32, on all linear layers. The backbone weights stay frozen.
2. **Stage 1 from v1.** `--init-from` loads the v1 `stage_symbolic.pt` heads,
   adapter and GNN. Stage 1 does not run again.
3. **1 epoch per stage.** v1 overfit in its second decoder epoch.
4. **New stage 2/3 data** (`curated_v2`, built by `data/mix_stage2.py`):

   | Kind | Train rows | Target |
   |---|---|---|
   | respond | 13,766 | the v1 QA answer |
   | unanswerable | 6,117 | EOS. The prompt is a v1 abstain document plus a question from another document. The borrowed answer does not occur in the document. |
   | general | 1,715 | the next 64–128 tokens of a corpus document that the QA set does not use |

   The train set has 21,598 rows. The val set has 443 rows.

| Stage | Trains | Trainable params | Peak LR | Optimizer steps |
|---|---|---|---|---|
| 2. Decoder | LoRA, injector | 10.4M | 1e-4 | 5,400 |
| 3. Joint | LoRA, injector, symbolic modules | 32.7M | 2e-5 | 5,400 |

**Compute:** 1x A40 48 GB. Batch 2 with gradient accumulation 2 (effective
batch 4), no gradient checkpointing. Peak GPU memory 39.7 GB. Speed about 0.68
optimizer steps/s. Wall time about 4.5 h for stages 2 and 3.

**Validation:**

| Stage | Train loss | Val loss | Val loss, gate = 0 | Other |
|---|---|---|---|---|
| Decoder | 0.7887 | 0.7923 | 0.9892 | abstain acc 1.000, respond rate 1.000 |
| Joint | 1.4081 | 1.6577 | 1.8495 | abstain acc 1.000, respond rate 1.000 |

- The val loss gap between gate on and gate off is 0.197 (decoder) and 0.192
  (joint). In v1 it was 0.011–0.037. The graph tokens now carry more of the
  signal. The final gate value is 0.127.
- Abstain acc now tests unanswerable questions, not documents with no
  question. The val set gives 1.000, but SQuAD v2 gives a NoAns exact score of
  only 7.7%. A possible cause: the val questions come from the same generator
  as the training questions. This run does not test that cause.

## Use

The model is a custom PyTorch module, not a `transformers` model. To load it, use
the project code:

```python
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from train_causal import load_causal_checkpoint

d = snapshot_download("sbartlett97/neurosymbolic-causal-lm",
                      allow_patterns=["qwen3-0.6b-v1/final_causal_model.pt",
                                      "qwen3-0.6b-v1/tokenizer/*"])
model, tokenizer, ckpt = load_causal_checkpoint(
    f"{d}/qwen3-0.6b-v1/final_causal_model.pt", device="cuda")

prompt = document + "\n\n" + question
# As in training: a long document loses its start, not the question
tokenizer.truncation_side = "left"
enc = tokenizer(prompt, truncation=True, max_length=4096,
                return_tensors="pt").to("cuda")
out = model.generate(enc.input_ids, enc.attention_mask, max_length=64,
                     do_sample=False, eos_token_id=tokenizer.eos_token_id,
                     pad_token_id=tokenizer.pad_token_id)
answer = tokenizer.decode(out[0], skip_special_tokens=True)  # "" = abstain
```

To read the extracted graph, call `model(input_ids, attention_mask)` without
`y_ids`. The output has `token_ent_logits`, `rel_logits_matrix`,
`node_entity_type_probs` and `concept_probs`.

### Files in this repository

| Path | Contents |
|---|---|
| `qwen3-0.6b-v1/final_causal_model.pt` | final weights, config and label maps |
| `qwen3-0.6b-v1/stage_symbolic.pt` | weights after stage 1 |
| `qwen3-0.6b-v1/stage_decoder.pt` | weights after stage 2 |
| `qwen3-0.6b-v1/stage_joint.pt` | weights after stage 3 |
| `qwen3-0.6b-v1/tokenizer/` | Qwen3 tokenizer with the added chat tokens |
| `qwen3-0.6b-v1/logs/`, `tensorboard/` | training log, GPU log, TensorBoard events |
| `qwen3-0.6b-v1/benchmarks/` | lm-eval results: v1 (`ours`, `nonodes`) and the base model (`base`) |
| `qwen3-0.6b-v2/final_causal_model.pt` | v2 final weights (after the joint stage) |
| `qwen3-0.6b-v2/stage_decoder.pt`, `stage_joint.pt` | v2 weights after stages 2 and 3 |
| `qwen3-0.6b-v2/tokenizer/` | tokenizer for v2 |
| `qwen3-0.6b-v2/logs/`, `tensorboard/` | v2 training logs and TensorBoard events |
| `qwen3-0.6b-v2/benchmarks/` | lm-eval results for v2 (`ours`, `nonodes`) |

## Intended use

- Research on neurosymbolic language models: graph extraction plus graph injection
  in one decoder-only model.
- Extractive QA over one document.
- Entity and relation extraction from the same forward pass.

## Out of scope

- Chat or instruction following. The chat stage did not run, so this is a
  base-style model.
- Open-domain questions without a source document.
- Production use, or any high-stakes decision.

## Limitations

- **Decoder overfitting.** In stage 2, val loss rose from 0.5544 (epoch 1) to
  0.5837 (epoch 2), while train loss fell from 0.62 to 0.20. `stage_decoder.pt`
  holds the epoch 2 weights. The next run uses 1 decoder epoch or early
  stopping.
- **Small graph gain.** The graph tokens lower val loss by about 2–3%. This
  run does not show that the graph improves answer accuracy.
- **Silver labels.** GLiNER2 made the entity and relation labels automatically,
  without human review. The F1 scores above measure agreement with GLiNER2,
  not with human labels.
- **Synthetic QA.** An LLM wrote all the questions and answers. Their style is narrower
  than real user questions.
- **No unanswerable questions in training.** Abstain rows have no question, so
  abstention is perfect on the val set for an easy reason. The model did not
  learn to refuse a question that its document cannot answer. SQuAD v2 tests
  this case. The next data version adds questions with no answer in the
  document.
- **English only.** All training text is English web and Wikipedia text.
- **Inherited bias.** The model keeps the biases of Qwen3-0.6B-Base, C4,
  Wikipedia and the QA generator.

## License

The weights follow the Apache 2.0 license of Qwen3-0.6B-Base. The QA set came from
OpenAI model outputs. Before any use beyond research, read the OpenAI terms for
model training on API outputs.

## Citation

```bibtex
@misc{bartlett2026neurosymbolic,
  title  = {NeuroSymbolic Causal LM},
  author = {Bartlett, Samuel},
  year   = {2026},
  note   = {qwen3-0.6b-v1}
}
```
