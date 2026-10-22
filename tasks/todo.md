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
- [ ] Before Run A: v2 val response loss with real nodes, nodes from
      another question, and no slots (local). A prefix-mode gate of 0 keeps
      16 zero slots, which hurt by themselves, so v2's val_gate0 gap (0.19)
      overstates the graph gain. If real = other-question, the Run B
      criterion is abstention (graph on vs off), not val_gate0
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
