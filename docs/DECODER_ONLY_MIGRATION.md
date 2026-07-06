# Migration Plan: Decoder-Only Backbone with Bolted-On Symbolic Modules

Status: **planned** (mechanics validated against transformers 5.13 — see
"Validated assumptions" at the end).

## Goal

Port the neurosymbolic architecture from T5/T5Gemma encoder-decoder to
modern decoder-only checkpoints (Qwen3, Gemma 3, Llama 3.2 class), keeping
its two defining mechanisms:

1. **Intermediate extraction heads** — entity classifier, concept bank,
   GNN, relation scorer, soft logic — reading representations of the input.
2. **Attention broadcasting** — the extracted symbolic state is something
   the generator *attends to*, not just an auxiliary loss.

The dataset (span-anchored JSONL, per-message trace annotations, taxonomy
vocab) ports **unchanged**. That was the point of keeping annotations
template-independent.

## Architecture mapping

| Encoder-decoder (today) | Decoder-only (target) |
|---|---|
| Encoder hidden states | Backbone hidden states at a tapped mid-layer, **prompt positions only** |
| Heads read encoder output | Heads read tapped states through a small bidirectional **extraction adapter** |
| Decoder cross-attends over `[enc ‖ node features]` | Node features become **soft tokens** inserted between prompt and response; response tokens attend to them causally |
| `decoder_input_ids` / `labels` | Single sequence `prompt + [nodes] + response`, labels `-100` over prompt and node slots |
| EOS-based abstention | Identical: response = immediate EOS when `should_respond=0` |

### Sequence layout

```
[prompt tokens .................] [node_1 ... node_K] [response tokens ... EOS]
      heads read here                soft embeddings         LM loss here
      (tap layer L_t)             (projected node feats)
```

- Causal order `prompt < nodes < response` means node slots can attend to
  the whole prompt and the response can attend to both — the decoder-only
  equivalent of the current `torch.cat([enc, memory_nodes], dim=1)` memory.
- Injection is at the **embedding layer** (`inputs_embeds`), the
  GraphToken/prefix-tuning pattern. Per-layer KV injection is a later
  experiment, not v1 (more powerful, much more invasive, cache-fragile).

### Forward pass (training)

1. **Read pass**: backbone over prompt tokens with
   `output_hidden_states=True`; take layer `L_t` (default ≈ 2/3 depth).
2. **Extraction adapter**: 1–2 transformer layers with *bidirectional*
   attention over the tapped prompt states. This is the fix for the causal
   handicap (a mention's representation only sees left context). The
   adapter is not in the generation path, so bidirectionality here is
   legitimate — it restores encoder-quality extraction for the heads.
3. **Heads**: existing modules, unchanged — token entity logits, span
   pooling → node features → GNN → relation scorer → soft logic; concept
   bank from pooled prompt representation.
4. **Write pass**: build
   `inputs_embeds = [embed(prompt) ‖ project(nodes) ‖ embed(response)]`,
   run the full backbone once for the LM loss. Node projection =
   `node_dim → hidden_size` linear + LayerNorm + learned scalar gate
   (init small, e.g. 0.1) so injection starts near-invisible and the
   model grows into it.

Cost: ~1.5–2× a plain forward (prompt is encoded twice). A single-pass
variant (tap and inject at different layers of one forward via hooks) is a
phase-4 optimization, not v1 — it couples layers and complicates caching.

### Generation

`encode prompt → heads → nodes → generate(inputs_embeds=[prompt ‖ nodes])`.
HF `generate()` from `inputs_embeds` with KV cache is supported (validated).
The symbolic state is computed once per request and cached — inference
overhead is one extra prompt pass plus K extra KV slots.

## What stays, what's new, what's deprecated

**Unchanged (the moat):**
- Entire curation stack: `data/curate/*` (GLiNER2, taxonomy, trace
  annotator, code annotator, loaders), JSONL schema, vocab files.
- Head modules: `model/entity.py`, `model/gnn.py`, `model/logic.py`,
  `model/pooling.py`.
- Continual learning: designed against the model's *output dict* keys
  (`entity_logits`, `concept_probs`, `rel_logits_matrix`, `logits`) and a
  `model(input_ids, attention_mask, spans, y_ids)` call shape — the new
  model keeps both, so `continual_learning/*` ports with no changes.

**New files:**

| File | Contents |
|---|---|
| `model/neurosymbolic_causal.py` | `NeuroSymbolicCausalLM`: backbone loading (`AutoModelForCausalLM`), read pass, write pass, `generate()`; returns the same output dict as `NeuroSymbolicLM` |
| `model/extraction_adapter.py` | Small bidirectional transformer over tapped prompt states |
| `model/injection.py` | `NodePrefixInjector`: node→hidden projection, LayerNorm, gate; builds combined `inputs_embeds` + masks |
| `data/collator_causal.py` | `CausalCognitiveCollator`: single-sequence packing, prompt-masked labels, `prompt_length`; **reuses** span/entity/relation/flattening logic from `CognitiveCollator` (refactor those into shared functions first) |
| `train_causal.py` | Staged training CLI (or `train.py --arch causal`) |
| `tests/test_causal_*.py` | Offline tests on a tiny from-config Llama (see Testing) |

**Config additions** (`config.py`):
`CausalModelConfig` — `model_name`, `tap_layer_ratio: float = 0.66`,
`n_soft_tokens = max_nodes`, `adapter_layers: int = 1`,
`use_lora: bool`, `lora_r/alpha/targets`, plus presets:
`qwen3-0.6b`, `qwen3-1.7b`, `gemma3-1b`, `llama3.2-1b`, `causal-testing`
(tiny from-config model, no download).

**Deprecated (kept, not deleted, until parity):** `model/neurosymbolic.py`
and `train.py` stay as the encoder-decoder reference implementation for
A/B comparison. Decide their fate after Phase 4 parity numbers.

## Chat formatting decision

Keep our flat `<system>/<user>/<assistant>/<tool>` `ChatTemplate` for v1,
added as special tokens to the backbone tokenizer (resize embeddings, as
today). Rationale: the entire per-message span-offset machinery
(`format_messages_with_offsets` → collator flattening) works unchanged, and
BPE fast tokenizers give better offset mappings than T5's sentencepiece.
A `HFChatTemplateAdapter` that computes content offsets under a model's
*native* chat template is a v2 item — worth having before comparing against
instruct checkpoints on their own template, irrelevant before then.

## Training curriculum (unchanged in spirit)

| Stage | Trains | Frozen | Data |
|---|---|---|---|
| 1 — Symbolic | adapter + heads + node/concept projections | backbone entirely | GLiNER2-curated corpus (`should_respond=0`) |
| 2 — Response | injector + backbone (full FT ≤1B, LoRA above) | heads + adapter | hybrid QA + traces |
| 3 — Joint | everything (LoRA or low LR) | — | mixed |

- Stage 1 is now *very* cheap: backbone frozen, only adapter/head grads.
- Abstention works identically (labels = EOS immediately after node slots).
- On a 24 GB 4090: full fine-tune is fine at ≤1B with gradient
  checkpointing; use LoRA (peft) for 3–4B. `peft` becomes an optional
  dependency, exercised in Stage 2/3 presets.

## Phases

**Phase 0 — prep (small):**
- Refactor `CognitiveCollator`'s span/entity/type/flattening logic into
  shared module-level functions both collators import.
- Add `causal-testing` tiny-model preset and test harness.

**Phase 1 — extraction on causal states:**
- `NeuroSymbolicCausalLM` read pass + adapter + heads, Stage-1 training.
- **Gate:** entity/relation F1 on held-out curated data within ~2 points of
  the encoder-decoder Stage-1 baseline. If the causal tap underperforms,
  ablate `tap_layer_ratio` ∈ {0.5, 0.66, 1.0} and adapter depth {0, 1, 2}
  before touching anything else. (Adapter depth 0 = the pure-causal
  baseline that quantifies the bidirectionality gap.)

**Phase 2 — injection + response training:**
- Injector, write pass, `generate()`, Stage-2 training, abstention check.
- **Gate:** (a) gated injection ≠ no-op — ablate by zeroing the gate at
  eval and measuring response-quality delta; (b) abstention rate on
  `should_respond=0` held-out ≥ encoder-decoder baseline.

**Phase 3 — joint + traces:**
- Stage 3, chat/trace phase on `curate_traces.py` output, LoRA option.

**Phase 4 — parity, CL port, cleanup:**
- Side-by-side eval vs T5Gemma on the same data (entity F1, BLEU/response
  quality, abstention accuracy, tool-call format accuracy on traces).
- Run `ContinuousLearner` against the causal model (should be zero-change;
  verify uncertainty MC-dropout behaves with the two-pass forward).
- Single-pass tap/inject optimization; decide encoder-decoder deprecation.

## Testing strategy (the big win)

Decoder-only models can be constructed **offline from config**
(`LlamaForCausalLM(LlamaConfig(...))` at hidden_size 32–64), so unlike the
T5 side, every migration test runs real forwards/backwards in CI:

- collator: packing, label masking, span→token alignment in prompt region
- read pass: tap shapes, adapter output, head logits
- injection: combined embeds shape, gate=0 ⇒ logits identical to baseline
  (exact no-op test), gradient flow into node projections from LM loss
- generation from `inputs_embeds` incl. abstention (immediate EOS)
- one full micro-training-step per stage (loss decreases on a toy batch)

## Risks / open decisions

| Risk | Mitigation |
|---|---|
| Causal states weaker for extraction | Bidirectional adapter (built-in); tap-layer ablation; Phase-1 gate before further investment |
| Injection ignored (gate → 0) or destabilizing | Gated init small; monitor gate value + zero-gate ablation as standing eval |
| `generate(inputs_embeds=...)` quirks across HF versions | Pin transformers; covered by offline tests |
| Instruct checkpoints resist our flat chat template | Start from *base* checkpoints for stages 1–2; native-template adapter (v2) before instruct comparisons |
| Embedding resize (4 special tokens) on tied-embedding models | Standard `resize_token_embeddings`; covered by tests |
| 2× prompt compute | Accept for v1; single-pass hook variant in Phase 4 |

## Validated assumptions (ran against installed transformers 5.13)

- `output_hidden_states=True` exposes per-layer states for mid-stack taps ✓
- `forward(inputs_embeds=[embeds ‖ soft tokens])` works ✓
- `generate(inputs_embeds=..., attention_mask=...)` works with KV cache ✓
- Tiny from-config `LlamaForCausalLM` runs offline (CI testability) ✓
