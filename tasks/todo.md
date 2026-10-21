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
