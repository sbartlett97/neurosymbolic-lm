# Dataset Format Specification

Training data is JSONL: one JSON object per line. The same schema serves all
three training stages; which fields are *used* differs per stage.

## Sample schema

```json
{
    "text": "Marie Curie won the Nobel Prize in 1903.\n\nWho won the Nobel Prize in 1903?",
    "entities": ["Marie Curie", "Nobel Prize", "1903"],
    "entity_spans": [[0, 11], [20, 31], [35, 39]],
    "entity_types": ["person", "creative_work", "temporal"],
    "concepts": [["scientist"], ["document"], ["year", "date"]],
    "relations": [[0, 1, "created"], [1, 2, "happened_on"]],
    "should_respond": 1,
    "response": "Marie Curie won the Nobel Prize in Physics in 1903."
}
```

| Field | Type | Notes |
|---|---|---|
| `text` | str | Encoder input. For QA samples the question is appended to the document, so document spans stay valid. |
| `entities` | list[str] | Entity surface forms, in span order. |
| `entity_spans` | list[[int,int]] | Character offsets into `text`, **end-exclusive** (`text[start:end]` is the entity). |
| `entity_types` | list[str] | Coarse type per entity, from the 15-type inventory in `data/curate/taxonomy.py` (index 0 is reserved for none/padding, giving `n_entity_types=16`). |
| `concepts` | list[list[str]] | 1–3 fine-grained concept labels per entity, from the taxonomy. Coarse type names are also valid (top-level) concepts. |
| `relations` | list[[int,int,str]] | `[head_idx, tail_idx, relation]` where indices point into `entities` and `relation` is a taxonomy relation label. |
| `should_respond` | 0 or 1 | Whether the decoder should produce a response (1) or learn to emit EOS immediately / abstain (0). |
| `response` | str | Target response; empty when `should_respond=0`. |

Legacy fields still accepted: `messages` (chat mode), `kg_relations`,
`kg_paths`, `image`. `entity_types` is optional for legacy data — the
collator falls back to deriving types from concepts via
`concept_to_entity_type_map`.

## Trace samples (assistant conversations)

Multi-turn traces keep `messages` intact and carry symbolic supervision
**per message**, so annotations stay valid regardless of how the chat
template renders the conversation:

```json
{
    "messages": [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Fix the bug in src/app/main.py"},
        {"role": "assistant", "content": "{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/app/main.py\"}}"},
        {"role": "tool", "content": "def load(path): ..."},
        {"role": "assistant", "content": "Fixed - load() now validates its input."}
    ],
    "message_annotations": [
        {"message_idx": 1, "entities": ["src/app/main.py"], "entity_spans": [[15, 30]],
         "concepts": [["file_path"]], "entity_types": ["digital_artifact"], "relations": []}
    ],
    "should_respond": 1,
    "response": "Fixed - load() now validates its input."
}
```

Rules:

- Roles: `system`, `user`, `assistant`, `tool` (tool/function results).
  Assistant tool calls are inline JSON content; the final assistant turn is
  the decoder target and is **never** annotated.
- `entity_spans` in `message_annotations` are relative to that message's
  **stripped** content (`content.strip()`); relation indices are local to
  the message's own entity list.
- At collate time, `ChatTemplate.format_messages_with_offsets` reports where
  each message landed in the encoder input and the collator projects spans
  and re-bases relation indices automatically
  (`CognitiveCollator._flatten_message_annotations`).

Produce trace data with:

```bash
# Deterministic annotation only (regex artifacts + AST code parsing)
python data/curate_traces.py --source glaiveai/glaive-function-calling-v2 \
    --target-samples 20000 --no-gliner

# With GLiNER2 prose extraction as well
python data/curate_traces.py --source glaiveai/glaive-function-calling-v2 \
    --target-samples 20000
```

Per message, fenced code blocks are annotated by the AST-based
`CodeAnnotator` (functions, classes, imports, calls, raises — exact spans,
Python via stdlib `ast`, other languages pluggable), digital artifacts
(URLs, file paths, emails, versions, env vars) by exact regex, and the
remaining prose by GLiNER2.

## Vocabulary file

The curation pipeline writes `<name>_vocab.json` next to the dataset:

```json
{
    "concepts": {"animal": 1, "artist": 2, "...": 3},
    "relations": {"acquired": 1, "born_in": 2, "...": 3},
    "entity_types": {"person": 1, "organization": 2, "...": 3},
    "concept_to_entity_type": {"scientist": 1, "company": 2, "...": 3},
    "statistics": {"num_concepts": 105, "num_relations": 41, "num_samples": 50000}
}
```

All maps are 1-indexed; **0 is reserved** for unknown/padding everywhere.
`train.py` automatically picks this file up (`load_vocab_file`) so label
indices and head sizes stay stable across runs and shards. Without it,
vocab is re-derived from the dataset contents (legacy behavior — avoid for
real training runs, since index assignment then depends on file contents).

## Stage usage

| Stage | Fields used | Typical data |
|---|---|---|
| 1 — Symbolic | `text`, `entities`, `entity_spans`, `entity_types`, `concepts`, `relations` | GLiNER2-annotated corpus text (`--annotator gliner`), `should_respond=0` |
| 2 — Decoder | `text`, `should_respond`, `response` | Hybrid QA samples (`--annotator hybrid`) and/or instruction data |
| 3 — Joint | all fields | Mix of both; hybrid samples carry full symbolic + response supervision |

A recommended recipe: one GLiNER2 pass over corpus text for a large Stage-1
set, then a hybrid pass (GLiNER2 + LLM QA) over a subset for Stages 2–3,
with `--should-respond-ratio` controlling the answer/abstain balance.

## Producing data

```bash
# Stage-1 symbolic data (GLiNER2 only, no LLM needed)
python data/curate_dataset.py --annotator gliner \
    --sources allenai/c4 wikipedia --source-ratios 0.6 0.4 \
    --target-samples 50000 --output-dir data/curated

# Stage-2/3 data (GLiNER2 + LLM QA generation)
python data/curate_dataset.py --annotator hybrid \
    --backend vllm --llm meta-llama/Llama-3.1-8B-Instruct \
    --should-respond-ratio 0.7 --target-samples 20000 \
    --output-dir data/curated_qa
```

### Annotation semantics (GLiNER2)

GLiNER2 is a *zero-shot* extractor: it selects from the label inventory
supplied per pass — it does not invent labels. Label diversity therefore
lives in `data/curate/taxonomy.py` (~100 described concept labels across 15
coarse types, ~40 relation types). To extend coverage for a domain, add
labels there; the vocab file, QC validation, and entity-type mapping all
follow automatically. Keep totals within the model budgets
(`n_concepts`, `n_relations` in `config.ModelConfig`).
