# Readiness: dataset generation + RunPod training (2026-10-16)

Source: two audits (training path, curation pipeline) + advisor review.
Scope: close gaps that crash a run or train it wrong. Prep only — no pod creation.

## A. Training — silent wrong-training
- [x] A1 Precision: fp32 backbone for full FT on CUDA, bf16 autocast, no GradScaler (fp16 scaler only on explicit opt-in). LoRA keeps bf16 base.
- [x] A2 Train entity class 0 ("none") on real tokens, -100 on padding; top-K node scoring excludes class 0 (both models); down-weight class 0.
- [x] A3 Pass gold spans to symbolic/joint losses (relation indices match nodes).
- [x] A4 NaN-safe entity loss (no valid labels -> skip term); extend entity-type maps on label merge.
- [x] A5 tokenizer.padding_side="right"; backbone.config.pad_token_id = tokenizer.pad_token_id.
- [x] A6 Test: EOS label is not masked when pad == EOS (answer + abstention rows).
- [x] A7 Error on missing user-given data paths.
- [x] A8 Collator drops spans past truncation instead of clamping.

## B. Training — usable runs + checkpoints
- [x] B1 Full config + label maps + tokenizer in saves; stage_<name>.pt kept per stage + latest.pt overwritten (no rotation); load_causal_checkpoint(); resume reports key mismatches.
- [x] B2 logits_to_keep in write pass (memory).
- [x] B3 Gradient checkpointing use_reentrant=False.
- [x] B4 _check_model_health only with --debug.
- [x] B5 --preset causal-testing runs from CLI + end-to-end test.
- [x] B6 Validation loss from _val.jsonl for early stopping.

## A9 (found during verification)
- [x] A9 ToyCognitiveDataset rewrote curated QA rows into heuristic question/abstain pairs; now toy data only.

## C. Curation (subagent)
- [x] C1 Refuse non-empty output dir without --overwrite/--resume.
- [x] C2 Resume skips consumed docs + truncates JSONL (or refuse --resume).
- [x] C3 Apply --max-doc-length to doc.text; pass max_len to GLiNER2.
- [x] C4 Error on zero-doc source; --text-field.
- [x] C5 GLiNER auto-CUDA, loaded before vLLM; vLLM mem 0.6 in hybrid.
- [x] C7 QA parse: no stop strings that eat JSON, chat template, log qa_failed rate.
- [x] C8 --quantization default none.
- [x] C10 Delete RedPajama source.
- [x] C13 Hash dedup + deterministic _val.jsonl split.
- [x] C14 --output-name honored.

## D. RunPod prep
- [x] D1 Split requirements (train vs curate/vLLM); pin transformers 5.13.*; add peft, accelerate.
- [x] D2 Py3.11 compatibility check.
- [x] D3 scripts/runpod_setup.sh + docs/RUNPOD.md runbook (HF_HOME on /workspace, HF_TOKEN, tmux, smoke runs, VRAM table).

## Verification
- [x] Full test suite green (92 passed); all files compile on Python 3.11.
- [x] Real Qwen3-0.6B-Base on MPS, all stages + chat phase, on real curated output.
- [x] Bf16 weight-update probe shows updates land (bf16 master 19% vs fp32 100% of weights changed in 10 steps).

## Deferred (listed in final report)
- Per-turn tool-call targets; glaive JSON normalization.
- API LLM backend (OpenAI/Anthropic).
- DDP + gradient accumulation.
- Soft-logic rules config (relation loss is now a mean, with negatives).
- Entity-type merge across label batches; GLiNER/QA batching (C6).

## Review (2026-10-16)
- Training: fixes A1-A9 and B1-B6 verified by 10 new tests, a real Qwen3-0.6B-Base
  run on MPS through all stages + chat, resume, and reload.
- Curation: C1-C16 by subagent; 19 new tests; real c4 + glaive runs (0 bad spans,
  resume gives the same output as an uninterrupted run).
- Advisor final review, fixed:
  - bf16 base + fp32 heads with adapter_layers=0 crashed eval/generate; _read now casts (test fails without the fix).
  - One val set served every stage (abstain-only corpus for decoder early stopping); now per-stage `_val.jsonl`.
  - Embedding resize shrank Qwen's padded vocab, so reload failed on shape; now grow-only. LoRA + full FT
    train-save-reload verified exact on real Qwen3-0.6B (chat-token rows train, tied head included).
  - Setup script stops on torch < 2.6, a swapped venv torch, or no CUDA.
- Not verified here (no CUDA/vLLM on this Mac): bf16 autocast on GPU, VRAM
  per preset, vLLM QA parsing. The runbook smoke test (docs/RUNPOD.md §5)
  covers the first two on the pod.
- TransformersBackend still defaults to 4-bit (needs bitsandbytes); vLLM is the
  documented backend.

# Round 2: deferred gaps (2026-10-16)

## Training (me)
- [x] E1 Gradient accumulation (--grad-accum), scheduler counts optimizer steps.
- [x] E2 DDP via torchrun (per-stage re-wrap after freezing, DistributedSampler, rank-0 save/log/eval, synced early stopping). Test: 2-proc gloo on CPU.
- [x] E3 Soft-logic type constraints from taxonomy relation domain/range (vectorized; node mask from gold spans; --soft-logic-weight).
- [x] E4 train.py/NeuroSymbolicLM: fp32 master weights (same bf16 lost-update bug).

## Curation (subagents)
- [x] F1 Traces: one sample per assistant turn (tool calls become targets); glaive <functioncall> -> real JSON; keep function/ipython/tool_calls roles.
- [x] F2 API LLM backends (OpenAI, Anthropic) for hybrid QA.
- [x] F3 GLiNER2: merge entity types by surface text across label batches; skip code/digital label batches for corpus prose.
- [x] F4 Batching: GLiNER2 batch_extract_*, QA prompts batched per backend call.
- [x] F5 TransformersBackend: 4-bit off by default.

## Round 2 review
- 145 tests pass; all files compile on Python 3.11.
- E1 accumulation equals full batch (max diff < 1e-5). E2 DDP: 2-proc gloo run of all
  stages with grad checkpointing + accumulation; rank-local NaN keeps ranks in sync
  (without the skip sync the run aborts). E3 4552 disallowed triples from taxonomy.
  E4 encoder-decoder trainers run on a tiny offline T5 under bf16 autocast.
- F1 per-turn targets: 22% of glaive targets are tool calls; 8/8 decode exactly.
- F2 API backends mocked only (no paid calls); gpt-6-luna ID checked on the web.
- F4 batched GLiNER2 output identical (0/8 docs differ) but slower on MPS; default 1.
- Existing data/curated/curated.jsonl predates fixes (27/100 rows > 4000 chars, no
  vocab file): regenerate before training.
- Not verified: CUDA/NCCL DDP, GLiNER batching speed on CUDA, live API calls.

## Round 3: relax limits for RunPod hardware
- [x] max_doc_length 4000 -> 16000 chars (~3.7k Qwen tokens at 4.3 chars/token).
- [x] GLiNER2 max_len 1024 words -> uncapped (a cap silently drops late entities; tested).
- [x] Training max_input_length 2048 -> 4096 for all causal presets.
- [x] max_entities 24 -> 128 per document, new --max-entities (config value was never wired).
- [x] Runbook GPU table updated for the longer context.

# Round 4: first full training run, qwen3-0.6b (2026-10-20)

## Smoke run (done)
- [x] A40 48 GB pod in CA-MTL-1, with a 150 GB network volume at /workspace
- [x] Setup on the pod: 153 tests pass
- [x] 400 rows, 1 epoch per stage, batch 4: SMOKE_EXIT=0, no skipped batches
- [x] Hub push to sbartlett97/neurosymbolic-causal-lm/smoke: all stages and final
- [x] Peak GPU memory 22.1 GB (joint stage, 619M trainable)
- [x] Gate LR fix: `--gate-lr` (default 1e-3), with its own optimizer group.
      At the stage LR, the gate moved only 0.1000 -> 0.1001 in 100 steps

Smoke speed at batch 4: symbolic 4.8, decoder 1.8, joint 1.6 samples/s.

## Full run plan
- [ ] Commit and push the gate LR change. Pull it on the pod
- [ ] Run detached (setsid), with the log on the volume:
      python -u train_causal.py --preset qwen3-0.6b \
        --dataset /workspace/data/curated/curated.jsonl \
        --stage2-dataset /workspace/data/curated_qa/curated.jsonl \
        --epochs 2 --batch-size 4 --max-eval-samples 500 --log-every 100 \
        --output-dir /workspace/ckpt/qwen3-0.6b-v1 --log-dir /workspace/runs/qwen3-0.6b-v1 \
        --hub-repo sbartlett97/neurosymbolic-causal-lm --hub-path qwen3-0.6b-v1
- [ ] Time estimate for each epoch: symbolic 2.8 h, decoder 3.0 h, joint 3.4 h.
      2 epochs: about 19 h, about $11 at $0.59/hr
- [ ] Checks during the run (stop the run if one fails):
  1. Symbolic epoch 1: entity_f1 above the smoke value 0.318. relation_f1 above 0
  2. Decoder: the gate moves off 0.1. val is lower than val_gate0, and the gap grows
  3. Joint: the response val loss does not rise above the decoder stage value
- [ ] Recovery: the volume keeps latest.pt. Continue with --resume and the remaining --stages
- [ ] After the run: delete the pod. Keep the volume until the Hub has the final model

## Benchmarks and model card (2026-10-21)
- [x] `eval_harness.py`: lm-eval wrapper (loglikelihood, rolling, generate_until). 8 tests on the tiny model.
      End-to-end check through lm-eval on arc_easy, gsm8k, wikitext (limit 3)
- [x] Fix: top-K nodes of a short prompt in a padded batch took padding positions.
      Now those slots are empty nodes, as when the prompt runs alone. Regression test added
- [x] `docs/MODEL_CARD.md` draft. Joint epoch 2 and benchmark rows are pending
- [ ] After the run: fill in joint epoch 2. Run the benchmarks on a new A40 pod with volume 9y5kxdqc86
      (base model, this model, this model with gate = 0). Fill in the benchmark table
- [ ] Decoder-only val loss for stage_decoder.pt and stage_joint.pt (the joint check)
- [ ] Push the card to the Hub as README.md (after approval)

# Round 5: v2 run plan, qwen3-0.6b-lora (2026-10-21)

Goal: keep the base model's general ability (v1 lost 4-15 points on every
zero-shot task) and keep the graph's gain on document QA.

## Changes
- [ ] Preset `qwen3-0.6b-lora` in config.py: Qwen3-0.6B-Base, use_lora=True
      (r 16, alpha 32, all-linear, chat token rows trainable). Base in bf16
- [ ] `--epochs 1` for the decoder and joint stages
- [ ] `--lr 2e-4` (LoRA range): decoder 1e-4, joint 2e-5. Gate LR stays 1e-3
- [ ] Reuse the v1 symbolic stage: new `--init-from <ckpt>` loads every weight
      except `backbone.*`. It refuses if the label maps differ. Valid because
      stage 1 froze the backbone, so v1 stage_symbolic.pt = Qwen base + heads.
      Saves the 6 h symbolic stage. Run `--stages decoder joint`
- [ ] New `data/mix_stage2.py`: builds the v2 stage 2/3 set from the v1 files
      (no new API calls):
  1. respond: the 13,766 v1 QA rows, unchanged
  2. unanswerable (replaces the v1 no-question abstain rows): a document plus
     a question from another document that shares an entity type.
     should_respond=0, target EOS
  3. general text: a prefix of a corpus document not used in QA. The target is
     the next 64-128 tokens. Entity spans and relations are kept only if they
     sit inside the prefix. About 4,000 rows (15% of rows, about half of the
     response tokens: the loss is a per-token mean)
  - Train/val split follows the source document's split
  - Writes curated_v2.jsonl, curated_v2_val.jsonl, curated_v2_vocab.json
- [ ] Tests: preset, --init-from (loads, refuses a map mismatch), mix script
      (span filtering, relation reindex, no answer leaks, split kept)

## Why the abstain rows change
v1 abstain rows were documents with no question (target EOS). General-text
rows are also documents with no question, but their target is a continuation.
The two kinds would contradict each other. Unanswerable questions fix this,
and they teach real abstention, which v1 did not learn.

## Run
- Train rows: about 13.8k respond + 6.2k unanswerable + 4k general = 24k.
  About 6,000 steps per stage at batch 4
- Time: decoder + joint about 7-8 h (LoRA, bf16 base). Cost about $4.50 at $0.59/hr
- Benchmarks: the same 6 zero-shot tasks, MMLU 5-shot and SQuAD v2, on v2
  (graph on and graph off). The base model results from v1 stay valid.
  About 3 h, about $2

## Checks
1. Zero-shot scores with graph off: within 2 points of Qwen3-0.6B-Base
2. Decoder and joint: val < val_gate0
3. abstain_acc (now on unanswerable questions) > 0.8 and respond_rate > 0.9
4. SQuAD v2: better than v1, mainly on the no-answer questions

# Round 6: v3 plan (2026-10-22)

Goals:
1. Abstain on questions that the document cannot answer, also when the
   question is on the same topic (SQuAD v2 NoAns: v2 7.7%, val 100%).
2. Keep general language modelling and text generation at the base level,
   also with the graph on (v2 graph on: ARC-Easy acc_norm -9.3 vs graph off).
3. Keep the graph's gain on document QA (v2: val loss -0.19 with the graph).

## Step 0: find the cause of the graph-on loss (decision point)
v2 graph on vs off: ARC-E acc -2.4 but acc_norm -9.3; HellaSwag acc +0.2 but
acc_norm -1.5. A constant log-prob shift on every answer choice cancels in
acc but not in acc_norm. Hypothesis: the 16 node slots between the prompt and
the answer cause the loss (position), not the node content.
- [x] ARC-Easy, limit 500, v2 final model, fp32, local MPS run. Result:
      | Mode                          | acc  | acc_norm |
      | graph off                     | 64.0 | 62.8     |
      | graph on                      | 60.4 | 53.8     |
      | 16 zero slots (gate = 0)      | 56.8 | 52.6     |
      | nodes from another question   | 61.8 | 54.0     |
      Wrong-question nodes score the same as the right nodes. The node
      content does not cause the loss. The 16 slots between the prompt and
      the answer cause it (position). The standard error at 500 questions is
      about 2.1 points: on vs other-question nodes is noise, off vs on (-9.0
      acc_norm) is about 4 standard errors.
      Recommendation: Option 2. Approved by the user (2026-10-22)
- [ ] If gate 0 loses about as much as graph on: the cause is the position.
      Then a routing gate does not fix the loss when it routes "on"
- [ ] If gate 0 is close to graph off: the cause is the content

## Part A: data (data/mix_stage2.py), no API calls
- [x] Same-document negatives: a respond row with the answer sentences
      deleted. Same question, target EOS. Key = answer words not in the
      question. Skip a row if the key is weak or longer than 6 words, if more
      than 3 sentences match, or if the key survives the deletion. Only
      factual questions (who, when, where, which, what, how many/much/...).
      Yield: 7,528 of 13,481 respond rows. Hand check of 50 samples: no clear
      error, 3 that the diff display did not let me judge
- [x] Entity-swap negatives: the document stays the same. A question name
      (person, organization, location) changes to a same-type name that the
      document does not contain. Dates and quantities gave nonsense questions
- [x] Span fix after a deletion: drop spans in the cut, shift later spans,
      reindex relations. Test text[s:e] == entity for every span
- [x] Counts are CLI options. Keep negatives at about 30-35% of rows
      (v2: 28%), so respond_rate stays high
- [x] A "kind" field on each row (respond, general, unanswerable_cross,
      unanswerable_same, unanswerable_swap)
- [x] Tests for each new row kind. Default options still give the v2
      files byte for byte (checked on the Hub data)
- [ ] Later (needs the OpenAI Batch API, ask first): a few thousand yes/no
      questions, for BoolQ (v2: -8.6 vs base)
- [ ] More general text: v2 used 2,000 general rows (`--general-rows 2000`),
      and 29,411 free corpus documents are left. Try 4,000-6,000. A new source
      is needed only beyond that. Rows without labels need a check of what the
      collator does with an empty entity list

## Part B: graph injection (after Step 0, ask before the work starts)
Option 1, routing gate (user's idea): a small head predicts if the graph
helps. If not, the model adds no node tokens (= v2 graph off).
- Good: the "off" path is exactly v2 graph off (within 2 points of base)
- Bad: it needs labels (loss with vs without nodes, an extra forward pass),
  and the yes/no choice is not differentiable. It must route prompt types it
  never saw (MMLU). Each misrouted general prompt pays the full position
  loss (about 9 acc_norm on ARC-Easy). On document QA the slots cost nothing
  measurable (SQuAD HasAns_f1 65.2 on vs 65.5 off)
Option 2, gated cross-attention (Flamingo style): the token sequence stays
the same as in the base model. A few added cross-attention layers let
response tokens read the node features. Each layer has a tanh gate that
starts at 0, so at init the model is exactly the base model.
- Good: the use of the graph is per token and depends on the content. The LM
  loss trains it, with no extra labels. No position loss
- Bad: more code (wrap Qwen3 layers under peft, mask prompt positions from
  the nodes, KV cache in generate). Stages 2 and 3 train again (v1
  stage_symbolic.pt stays usable)
Both: node dropout in stages 2 and 3 (per sample), so the backbone works
with and without the graph.
Option 3, nodes before the prompt: keeps "Answer:" next to the answer, with
a small code change. Needs a retrain to test. The first position is an
attention sink in Qwen, so prefix tokens can change the whole prompt.
Recommendation: Option 2 if Step 0 shows a position cause. Option 1 if it
shows a content cause.
Main risk of Option 2: Step 0 does not show that cross-attention keeps the
QA gain (val loss -0.19, SQuAD NoAns 7.7% vs 2.9%). Only Run B shows it.
No option helps pure language modelling: prompt positions cannot read nodes
built from the whole prompt. The goal there is no loss.

## Part B status (Option 2, approved 2026-10-22)
- [x] NodeCrossAttention (model/injection.py): pre-norm, q from hidden,
      k/v from nodes, out proj, tanh gate at 0. Empty node slots masked
- [x] injection="xattn": hooks on every 2nd decoder layer from the tap
      layer on (Qwen3-0.6B: 18, 20, 22, 24, 26; 13.1M params). Positions
      from the last prompt token on read the nodes. generate: the newest
      position only. Context stays set after the write pass for gradient
      checkpointing
- [x] node_dropout (training only); preset qwen3-0.6b-lora-xattn (0.1)
- [x] train_causal: gate log, val_gate0 zeroes all gates, optimizer gate
      group, stage freezing, --init-from a prefix checkpoint (its injector
      is skipped, xattn starts fresh)
- [x] Val generation check: up to 20 rows of each kind, abstain_acc_<kind>,
      each with the graph on and off (*_nograph)
- [x] generate refuses use_cache=False in xattn mode (the newest-position
      rule needs the cache)
- [x] v2 val response loss (443 rows, fp32, local MPS), mean per row:
      | Kind               | real nodes | other-question nodes | no slots |
      | all                | 0.5255     | 0.5263               | 1.2130   |
      | respond (285)      | 0.3565     | 0.3579               | 1.4079   |
      | general (41)       | 3.2000     | 3.1994               | 3.3167   |
      | unanswerable (117) | 0.0000     | 0.0000               | 0.0010   |
      The node content is worth 0.0008. The large gap to "no slots" comes
      from the slots: v2 learned them as an "answer starts here" marker.
      So v2 shows no evidence that the graph content helps. The model card
      claim "the graph tokens now carry more of the signal" is wrong
- [x] Run B criterion: content, not slot presence. In xattn mode a gate of
      0 is exactly no graph, so val_gate0 measures content. Added
      val_shuffled: each row gets the nodes of the next row in its batch
      (only the injected nodes; the heads keep the right ones). Needs a val
      batch of 2 or more
- [x] Model card (PR #3, Hub README): the v2 gain comes from the slots
- User decision (2026-10-22): skip Run A. Run B only, after a memory test
- [ ] Before Run B on a pod: a memory test in a separate process (13.1M
      more trainable params; v2 peaked at 39.7 GB of 48 GB)
- [x] Tests (tests/test_xattn.py): init = backbone, prompt positions
      unchanged, generate logits = training-pass logits, gradients (gate
      first, then attention), checkpointing gives the same gradients,
      init-from. Mutation checks on the masks fail the tests
- [x] Real backbone smoke run (CPU): hooks attach through peft, init equals
      the backbone, 3 steps train, generate works

## Runs
- Run A: v2 recipe + Part A data. Compare with v2 to measure the data change
- Run B: Part A data + Part B change. Compare with Run A
- About $5 each (A40, $0.59/hr)

## Evaluation additions
- [ ] Val abstain_acc for each kind (same-doc negatives separately)
- [ ] Perplexity on wikitext. Check loglikelihood_rolling in eval_harness.py

## Run B results (2026-10-22, pod 2j9grlmwelqch4, deleted)
Training: 12:47-18:08 UTC, 0.67 steps/s, no errors. Benchmarks in bf16.
Hub: qwen3-0.6b-v3/ (stage_decoder.pt, stage_joint.pt,
final_causal_model.pt, 8 benchmark JSONs, logs, TensorBoard).
base wikitext failed on the pod (fp32 batch 2, CUDA OOM). Run again locally.

Validation:
| Stage   | val    | val_gate0 | val_shuffled | mean gate |
| Decoder | 0.9863 | 0.9865    | 0.9867       | ~0.004    |
| Joint   | 1.8550 | 1.8543    | 1.8549       | ~0.05     |
(joint val includes the symbolic losses: compare within a row only)
Generation (joint), graph on = graph off: abstain_acc 0.933, respond_rate
0.925, unanswerable_cross 1.000, unanswerable_same 0.800 (about 20 rows),
unanswerable_swap 1.000.

Benchmarks (acc_norm for ARC, HellaSwag, PIQA; acc for the others):
| Task          | Base  | v2    | v2 off | v3    | v3 off |
| MMLU 5-shot   | 52.5  | 49.1  | 50.8   | 50.9  | 51.0   |
| ARC-Easy      | 58.0  | 55.5  | 64.8   | 61.7  | 61.7   |
| ARC-Challenge | 38.1  | 37.4  | 38.0   | 38.8  | 39.3   |
| HellaSwag     | 53.8  | 50.8  | 52.3   | 52.5  | 52.4   |
| PIQA          | 69.8  | 69.1  | 69.6   | 69.3  | 69.6   |
| WinoGrande    | 58.7  | 57.6  | 59.4   | 59.5  | 59.4   |
| BoolQ         | 70.0  | 63.1  | 61.4   | 62.4  | 62.6   |
| SQuAD f1      | 29.21 | 36.55 | 34.28  | 48.73 | 48.54  |
| SQuAD HasAns_f1 | 58.25 | 65.21 | 65.46 | 60.88 | 60.99 |
| SQuAD NoAns   | 0.00  | 7.72  | 2.91   | 36.51 | 36.01  |
| wikitext word ppl | BASE_WIKI | - | - | 20.27 | 20.19 |

Checks:
- [ ] val < val_gate0 and val < val_shuffled: fails. The graph content
      adds nothing in both stages (differences of 0.0007 or less)
- [ ] unanswerable_same, graph on > graph off: fails (0.800 = 0.800)
- [x] respond_rate > 0.9 (0.925)
- [x] Graph on within 2 points of base: MMLU, ARC, HellaSwag, PIQA,
      WinoGrande pass. BoolQ fails (-7.6; v2 had the same loss)
Conclusion:
- Cross-attention removes the v2 slot-position cost (ARC-E +6.2,
  HellaSwag +1.7, MMLU +1.8 against v2 graph on). Graph on = graph off.
- The new data gives the large SQuAD gain: NoAns 7.7 -> 36.5, f1 36.6 ->
  48.7. It costs 4.3 points of HasAns_f1 against v2 (more abstention).
- The graph is not used. The nodes come from the passage that is already
  in the prompt, so the LoRA learns the task from the text alone. A graph
  gain needs a task where the graph holds information the prompt lacks.

# Round 7: can the graph path carry an answer? (plan, 2026-10-23)

Run B: the graph adds nothing when the prompt already holds the passage.
Before a companion knowledge graph, test the channel itself: build the
graph from the passage, but keep the passage out of the prompt.

## Step 1a: separate graph input (code)
- [x] Model: forward/generate take graph_input_ids + graph_attention_mask.
      The read pass (tap, heads, nodes, GNN) runs on the graph input. The
      write pass runs on input_ids. Default (None): the graph input is the
      prompt, so the old behaviour stays the same
- [x] Collator: a row with "graph_text" tokenizes it for the read pass.
      Entity spans and labels align to graph_text. "text" is the prompt
- [x] Trainers (symbolic, decoder, joint), symbolic_metrics, val loss and
      val generation pass the graph input (training.trainers.graph_kwargs).
      Entity labels use the graph mask. eval_harness: no change (its
      tasks have no graph text)
- [x] Tests (tests/test_graph_input.py, 13; suite 208 pass): default = old outputs exactly; graph input changes only the
      nodes; spans align to graph_text; generate agrees with the training
      pass; shuffle control works with a graph input

## Step 1b: graph-only data (mix_stage2)
- [x] make_graph_only: graph_text = the document, text = the question,
      kind "graph_only_<kind>". Spans that start past the document are
      dropped with their relations (none in the v3 val set: 0 of 9,375).
      Real v3 val: 440 question rows, 0 span mismatches
- [x] --graph-only-frac: that share of the question rows (respond and all
      unanswerable kinds) in graph-only form, in train and val. 0 (default)
      keeps the old output exactly. A graph-only negative can only be
      recognised through the graph (the prompt is the bare question)
- [x] eval_graph_only.py: 4 modes on the graph-only val rows (closed_book,
      graph, graph_shuffled, passage). Respond rows: answer loss with top-K
      and gold-span nodes, greedy token F1 (value / other answers), key-word
      recall. Unanswerable rows: abstain rate per kind
- [x] Tests: 4 in tests/test_v2_run.py, 2 in tests/test_graph_input.py.
      The JSON holds one record per row (metrics per mode) and the paired
      bootstrap contrasts
- Baseline, v3 final model, 8 rows (local smoke run): on a bare question
  v3 abstains every time, with or without the graph (loss 2.65 in
  closed_book, graph and graph_shuffled). Passage mode: F1 0.75

## Step 1c: run (A40, about $5)
- [ ] Init from v1 stage_symbolic.pt (as Run B), not from v3: v3 trained
      on these train questions with their answers, so it can answer a bare
      train question from memory. That lowers the train loss without the
      graph and gives the gate less reason to open. With the same init and
      schedule as Run B, the only change is --graph-only-frac, and Run B is
      the control
- [ ] Data: the v3 mix with --graph-only-frac 0.5 (about 10k graph-only
      rows in train, about 220 in val). node_dropout 0.1 also trains the
      closed-book case a little, so closed_book is a fair baseline
- [x] Pod scripts rebuilt (scratchpad r7/pod/): setup_r7.sh (branch
      checkout, curated_r7 data with the Run B counts + --graph-only-frac
      0.5, data push), memtest_r7.sh (24 longest rows, both stages, then
      eval_graph_only.py on 8 rows), run_r7.sh (training, eval on the r7
      and v3 models, push, R7_EXIT), pod_reaper_r7.sh (watcher)
- [ ] Before launch: push claude/r7-graph-input to GitHub (the pod fetches
      it). Then the memory test, then run_r7.sh
- [x] (was) Pod tooling: /private/tmp/claude-501/v3/ is gone. Rebuild the run
      script from qwen3-0.6b-v3/logs/run_v3.sh (Hub). The run script runs
      eval_graph_only.py and pushes its JSON (with per-row records) before
      the exit marker. The watcher deletes the pod only after that push
- [ ] After training: eval_graph_only.py on the val set (bf16, all rows)
- [ ] Measures on val (respond rows, answer tokens only):
      closed book loss vs graph-only loss vs passage loss; exact match /
      token F1 of generated answers in the 3 modes; val_shuffled in
      graph-only mode
- Decision (eval_graph_only.py contrasts, paired bootstrap 95% CI over
  the graph-only respond val rows):
  - Main metrics: key_recall (answer words not in the question) and
    f1 on value answers. loss_topk is secondary: it averages over answer
    tokens that repeat the question
  - The channel works if the lower bound of BOTH
    key_recall(graph) - key_recall(closed_book) and
    key_recall(graph) - key_recall(graph_shuffled) is above 0
  - Abstention: report abstain_unanswerable_same in graph mode together
    with respond_rate (always answering gets about 64% of the rows right)
  - In-training val_shuffled mixes graph-only and passage rows: use it as a
    weak signal only
  - graph only clearly better than closed book (and better than
    val_shuffled): the channel works. Go to Step 2 (companion KG, with a
    text-retrieval baseline)
  - graph only = closed book: the channel is the bottleneck. Fix the node
    encoder first (span pooling at inference, more nodes, value nodes),
    then test again

- Report graph-only loss and EM/F1 twice: with gold-span nodes (channel
  capacity; the joint val loss uses gold spans) and with top-K nodes (the
  real pipeline; generate uses top-K). The split shows which part fails:
  the channel or the node selection
- [x] Before Step 1b: local coverage check on the v3 model. With val passages
  as the graph input, how often is an answer token among the 16 top-K
  node positions (entity answers vs value answers)? Low coverage: fix the
  node selection before the run
  Result (2026-10-23, v3 final, 279 val respond rows with the answer key
  in the passage, mean passage 425 tokens; key = answer words not in the
  question; a word counts if a token of any of its occurrences is a node):
  | Nodes          | rows with a key word | key words | value rows (119) | other rows (160) |
  | top-K (16)     | 69.5%                | 36.8%     | 69.7%            | 64.4%            |
  | gold spans     | 78.1%                | 50.0%     | 73.1%            | 77.5%            |
  | 16 random      | 38.4%                | 15.1%     | 22.7%            | 42.5%            |
  The question as part of the graph input changes nothing (69.5%).
  Conclusion: the node selection is good enough to run Step 1b/1c. The
  graph-only gain has a ceiling: about 30% of rows have no answer word
  among the nodes, and a node is one token (Qwen splits numbers into
  digits), so the run must show that a node carries the whole value.
  Script: scratchpad r7/coverage.py (not in the repo)
- Later, only if Step 1c is null with good coverage: let the question
  tokens read the graph too (no leak with a separate graph input)

## Risks
- The closed-book model can learn some answers from the question alone
  (memorised facts). The closed-book baseline measures that
- 16 single-token nodes may hold the entity but not a date or number
  answer. Report the gain per answer type (entity vs value)

## Step 1c results (2026-10-24, pod s9samuiea6x6ik, deleted)
Training 19:14-01:42 UTC (gradient checkpointing on, 0.56 steps/s), eval
11 min. Hub: qwen3-0.6b-r7/ (stage_decoder.pt, stage_joint.pt,
final_causal_model.pt, eval/graph_only_{r7,v3}.json, logs, TensorBoard).
Gate (mean |tanh|): 0.010 after the decoder stage, 0.124 after the joint
stage (Run B: 0.05).

In-training val (mixed rows):
| Stage   | val    | val_gate0 | val_shuffled |
| Decoder | 1.5362 | 1.5366    | 1.5359       |
| Joint   | 2.4151 | 2.4125    | 2.4163       |
Joint generation, graph on = graph off: respond_rate 0.617 (0.633 off),
all graph_only_unanswerable kinds 1.000.

eval_graph_only.py, 220 graph-only val rows (139 answerable):
| Model | Mode           | key_recall | f1    | loss_topk | loss_gold | respond |
| r7    | closed_book    | 0.004      | 0.008 | 1.977     | -         | 0.022   |
| r7    | graph          | 0.004      | 0.008 | 1.979     | 1.984     | 0.022   |
| r7    | graph_shuffled | 0.004      | 0.008 | 1.985     | 1.984     | 0.022   |
| r7    | passage        | 0.739      | 0.710 | 0.449     | -         | 0.928   |
| v3    | graph          | 0.003      | 0.013 | 2.941     | 2.941     | 0.043   |
| v3    | passage        | 0.767      | 0.714 | 0.398     | -         | 0.935   |
Abstain on graph-only unanswerable rows: 1.000 in every no-passage mode
(the model abstains on almost every bare question, so this is no skill).
Paired 95% CIs (r7):
- key_recall graph - closed_book: 0.000 [0.000, 0.000] -> FAIL
- key_recall graph - graph_shuffled: 0.000 [0.000, 0.000] -> FAIL
- loss_topk graph - closed_book: +0.002 [-0.003, +0.007]
- loss_topk graph - graph_shuffled: -0.005 [-0.011, +0.000]
- loss_gold graph - graph_shuffled: +0.000 [-0.005, +0.005]
- loss_topk passage - graph: -1.53 [-1.71, -1.35]

Decision: the graph path does not carry answers, also with gold-span nodes
and an open gate. The channel is the bottleneck.
Note on the metrics: greedy decoding abstains on 97.8% of the graph-only
answerable rows (a bare question is unanswerable 36% of the time in train,
and EOS wins the first-token argmax), so key_recall cannot show a partial
gain. The answer loss can, and it shows none either.

Next (recommendation, not started): find which half fails.
1. Encoder probe (local, no pod): train a linear probe from the r7 node
   features (the cross-attention keys/values) to the answer key tokens.
   No answer information in the nodes: the encoder (16 single-token
   states -> node_proj -> GNN) loses it.
2. Reader control (pod run): cross-attention to all layer-14 passage token
   states (no top-K, no node_proj). If that answers graph-only questions,
   the 16-node compression is the bottleneck; if not, the reader cannot
   learn retrieval from 6.7k rows.

## Step 1d: locate the failure (local, r7 model; plan 2026-10-24)
Encoder probe (token identity per node; the answer-word ceiling is
coverage x identity accuracy, and the coverage is known: 69.5%):
- Features at the 16 top-K node positions of graph-only respond rows
  (train subset for fitting, the 139 val rows for scoring): raw tap state
  (1024), adapter output (1024), node features after node_proj + GNN (256,
  what the cross-attention reads)
- Linear probe per level into the frozen LM head; target = the token id at
  the node position; top-1 / top-5 accuracy on val
- Chance: the same probe trained on mismatched rows (targets of another row)
Reader checks (teacher forcing, graph-only respond val rows):
- Content sensitivity per xattn layer at the answer-reading positions:
  |out(own graph) - out(other row's graph)| / |out(own graph)|, and the
  size of the xattn term relative to the hidden state
- Attention at the last prompt token (it predicts the first answer token):
  share of rows whose top node holds an answer word, against chance
  (answer nodes / valid nodes); attention entropy over the nodes
Decision map:
- Identity kept at tap and adapter, lost at node features: node_proj / GNN
  is the bottleneck. Feed 1024-d states to the xattn, or widen node_dim
- Identity kept, reader content-insensitive or attention at chance: the
  reader did not learn retrieval. Look at its training budget first (new
  module, joint stage lr 2e-5, 1 epoch) before a run that also changes the
  encoder
- Identity kept and attention finds the answer node, but no loss change:
  compare the xattn term with the residual norm at those positions

## Step 1d results (2026-10-26, local, r7 final model)
Encoder probe (linear map into the frozen LM head; target = token id at
each of the 16 top-K node positions; 1,500 graph-only train rows = 24,000
nodes; scored on the 139 val rows = 2,224 nodes; CPU fp32):
| Level                          | top-1 | top-5 | chance top-1 |
| raw tap state (1024)           | 90.5% | 96.3% | 2.1%         |
| adapter output (1024)          | 90.1% | 96.2% | 2.6%         |
| node features (256, xattn reads)| 49.4% | 67.1% | 4.6%         |
Val rows with an answer word among the nodes: 71.2%.
Reader checks (teacher forcing, 139 graph-only respond val rows, MPS):
| Layer | content sensitivity | xattn / hidden | top node = answer | chance | entropy |
| 18    | 1.02                | 2.9%           | 15%               | 19%    | 0.88    |
| 20    | 0.77                | 0.7%           | 20%               | 19%    | 0.84    |
| 22    | 0.80                | 0.7%           | 21%               | 19%    | 0.83    |
| 24    | 0.75                | 1.0%           | 21%               | 19%    | 0.92    |
| 26    | 1.04                | 0.5%           | 16%               | 19%    | 0.82    |
Conclusion: the reader is the main failure. Its attention is near uniform
and finds the answer node at chance rate; it adds a small content-
dependent average of the nodes. The encoder keeps most of the identity
(node_proj + GNN lose about half: 90% -> 49%, still 10x chance).
Likely cause: mixed training gives the backbone a shortcut (the LoRA fits
"bare question -> abstain"), so nothing forces retrieval. The zero gate
start is not the main cause (AdamW rescales the small gradients).
Scripts: scratchpad r7/probe.py (not in the repo).

## Step 1e: two-arm reader test (plan, 2026-10-26)
Backbone + LoRA + adapter + heads frozen (from the r7 final model). Only
the cross-attention trains, fresh (gate 0). Data: graph-only respond rows
only (6,726 train), so the graph is the only way to lower the loss.
- Arm A: the reader reads the current 256-d node features (node_proj + GNN
  frozen)
- Arm B: the reader reads the 1024-d adapter states at the same positions
  (xattn_source="adapter")
Same steps, lr and data for both arms. eval_graph_only.py on each.
Outcome map:
- Both null: this reader design cannot learn retrieval here. A KG needs
  another reader
- B works, A null: widen the node path, then go back to mixed training
- Both work: the shortcut in mixed training was the problem. Fix the mix /
  curriculum (e.g. a reader warm-up stage before mixed training)
Step 1e build (2026-10-26):
- [x] Model: xattn_source="node" | "adapter" (adapter: kv reads the
      adapter states at the node positions, 1024 wide; _select_node_states)
- [x] train_causal: stage "reader" (injection modules only, lr = --lr),
      --init-full (also the backbone + LoRA), --fresh-injection,
      --xattn-source. Tests: 5 new in tests/test_xattn.py; suite 220 pass
- [x] Local smoke run on the real model (CPU, 6 rows): 745 tensors loaded
      with the backbone, 30 fresh xattn tensors, 21.0M trainable (adapter
      arm), checkpoint reloads as adapter
- [x] Pod scripts (scratchpad r7/pod/run_r7e.sh, pod_reaper_r7e.sh): both
      arms 2 epochs, lr 2e-4, batch 2 x accum 2, gradient checkpointing,
      then eval_graph_only.py per arm; R7E_EXIT. No separate memory test:
      the backbone is frozen and r7 used about 7 GB with checkpointing

## Step 1e results (2026-10-26, pod gqv2kerslecwg4, A40 EU-SE-1)
Both arms: 2 epochs, 3,364 steps, rc 0. Pod deleted 15:43 GMT. Hub:
qwen3-0.6b-r7e-{node,adapter}/ (models), qwen3-0.6b-r7e/{eval,logs,tensorboard}/.
Training (val loss, 139 graph-only val rows):
| Arm     | epoch | val    | val_gate0 | val_shuffled | final gate |
| node    | 1     | 1.8141 | 1.9110    | 1.8179       |            |
| node    | 2     | 1.8234 | 1.9110    | 1.8380       | 0.133      |
| adapter | 1     | 1.8106 | 1.9110    | 1.8143       |            |
| adapter | 2     | 1.8365 | 1.9110    | 1.8497       | 0.041      |
Eval (eval_graph_only.py, 139 respond rows of 220):
| Arm     | Mode           | key_recall | f1    | f1_value | loss_topk | loss_gold | respond |
| both    | closed_book    | 0.004      | 0.008 | 0.023    | 1.977     | -         | 0.02    |
| node    | graph          | 0.130      | 0.317 | 0.377    | 1.881     | 1.884     | 1.00    |
| node    | graph_shuffled | 0.116      | 0.305 | 0.378    | 1.888     | 1.882     | 1.00    |
| node    | passage        | 0.780      | 0.731 | 0.744    | 0.433     | -         | 0.99    |
| adapter | graph          | 0.108      | 0.309 | 0.359    | 1.891     | 1.900     | 1.00    |
| adapter | graph_shuffled | 0.108      | 0.301 | 0.367    | 1.912     | 1.913     | 1.00    |
| adapter | passage        | 0.786      | 0.745 | 0.771    | 0.442     | -         | 0.99    |
Contrasts (mean [95% paired bootstrap CI]):
| Contrast                           | node                     | adapter                  |
| key_recall graph - closed_book     | +0.126 [0.089, 0.168]    | +0.105 [0.071, 0.141]    |
| key_recall graph - graph_shuffled  | +0.014 [-0.002, 0.031]   | +0.000 [-0.007, 0.008]   |
| f1_value graph - graph_shuffled    | -0.001 [-0.008, 0.006]   | -0.008 [-0.022, 0.002]   |
| loss_topk graph - closed_book      | -0.096 [-0.116, -0.075]  | -0.086 [-0.108, -0.064]  |
| loss_topk graph - graph_shuffled   | -0.006 [-0.016, 0.004]   | -0.020 [-0.037, -0.004]  |
| loss_gold graph - graph_shuffled   | +0.002 [-0.007, 0.012]   | -0.013 [-0.025, -0.001]  |
| loss_topk passage - graph          | -1.448 [-1.62, -1.28]    | -1.449 [-1.63, -1.27]    |
Outcome: BOTH NULL. In both arms the key_recall lower bound for
graph - graph_shuffled is below 0. The gain over closed_book is the answer
form only: the reader makes the model respond (closed_book abstains on
98%), and shuffled nodes give the same gain. The adapter arm has a content
effect on the loss (-0.020 nats), but it is about 1.4% of the passage gap.
Side effect: in the graph modes the model never abstains on unanswerable
rows (the training used respond rows only).
Conclusion: this cross-attention reader cannot learn retrieval from the
node states with this budget (6,726 rows x 2 epochs, lr 2e-4), even with
1,024-d states and no shortcut. The 256-d node path is not the cause. A
companion KG through this reader is not useful.

## Step 2: node-text test (plan, 2026-10-26)
Question: can the words that the symbolic branch selects help the LM when
they reach it as text? The probe says the content is there; Step 1e says
the vector reader cannot use it. The LM reads text well (passage mode).
Model: r7 final (local, MPS fp32; bf16 on MPS breaks generation). Rows:
the 220 graph-only val rows. All modes in one run, so contrasts are paired.
Text modes (eval_graph_only.py --modes, no graph input, nodes off):
- node_text: the whole words at the top-K (16) node positions of the
  passage, in passage order, adjacent words joined, repeats once
- node_text_shuffled: the partner row's node text (control)
- random_text: words at 16 random passage tokens (control)
- gold_text: the first 16 gold entity spans (ceiling for the extractor)
Plus closed_book and passage. --force-answer: a respond row that abstains
generates again with EOS blocked at the first step (the r7 LoRA abstains
on prompts that do not look like a passage).
Read-out (set before the run):
- Main metric: loss_topk (it does not depend on abstaining)
- node_text passes if the 95% upper bound of loss_topk node_text -
  node_text_shuffled is below 0 (lower loss); key_recall_forced as support
- Gap closed = (closed_book - X) / (closed_book - passage), on loss
- node_text vs random_text: does the extractor select better than chance?
- gold_text vs node_text: room for a better extractor
Build:
- [x] eval_graph_only.py: TEXT_MODES, word_spans, add_text_fields,
      --modes, --force-answer, gap_closed, TEXT_CONTRASTS. Tests: 4 new in
      tests/test_graph_input.py; suite 224 pass
- [x] Smoke runs (4 rows): node text looks right; fp32 needed on MPS
- [x] Full run on pod sci1vqnfg7tvtd (RTX PRO 4000, cuda fp32), 17:19-17:34
      GMT. Pod deleted 17:34. Hub: qwen3-0.6b-r7/eval/node_text_r7.json,
      qwen3-0.6b-r7/logs/s2.log

### Step 2 results (220 rows; 139 respond rows for loss, 138 for recall)
| mode | loss_topk | key_recall | key_recall_forced | f1_forced | respond | gap closed |
|---|---|---|---|---|---|---|
| closed_book | 1.978 | 0.004 | 0.116 | 0.305 | 0.022 | 0 |
| passage | 0.452 | 0.745 | 0.787 | 0.747 | 0.928 | 1.00 |
| node_text | 2.174 | 0.066 | 0.268 | 0.377 | 0.094 | -0.13 |
| node_text_shuffled | 3.152 | 0.000 | 0.051 | 0.165 | 0.000 | -0.77 |
| random_text | 2.693 | 0.014 | 0.131 | 0.247 | 0.029 | -0.47 |
| gold_text | 1.791 | 0.136 | 0.398 | 0.420 | 0.180 | 0.12 |

Paired contrasts (mean [95% CI]):
- loss_topk node_text - node_text_shuffled: -0.978 [-1.177, -0.780]
- loss_topk node_text - closed_book: +0.196 [-0.002, +0.406]
- loss_topk node_text - random_text: -0.519 [-0.711, -0.327]
- loss_topk gold_text - node_text: -0.383 [-0.560, -0.222]
- loss_topk random_text - closed_book: +0.715 [+0.566, +0.865]
- loss_topk node_text_shuffled - closed_book: +1.174 [+1.040, +1.316]
- key_recall_forced node_text - closed_book: +0.152 [+0.085, +0.216]
- key_recall_forced node_text - node_text_shuffled: +0.217 [+0.154, +0.279]
- key_recall_forced node_text - random_text: +0.136 [+0.079, +0.196]
- key_recall_forced gold_text - node_text: +0.130 [+0.062, +0.199]

Read-out:
- The pre-registered test passes: node_text has lower loss than
  node_text_shuffled, and key_recall_forced supports it.
- The control is weak. Any extra text that does not match the passage
  raises the loss above closed_book (shuffled +1.17, random +0.72).
  node_text is also no better than closed_book on loss (gap closed -13%).
- The selection is better than chance: node_text beats random_text on
  loss and on key_recall_forced. The node words carry answer content
  (forced key recall 0.27 against 0.12 closed book).
- A bag of words is the limit, not the extractor. Even gold entities
  close only 12% of the gap. The passage closes 100%. The answer needs
  the relations and context between the entities.
- Conclusion: the symbolic branch selects useful words. A list of entities
  without relations does not give the LM the answer.

