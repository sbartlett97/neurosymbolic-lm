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
