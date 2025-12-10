# Neuro-Symbolic Multimodal Language Model (NS-MLM)

A research prototype implementing a unified neuro‑symbolic architecture designed to move language models closer to human‑aligned cognition. This system integrates:

* Transformer-based neural text encoding
* Entity extraction
* Concept grounding
* Relation inference
* Differentiable soft‑logic rule enforcement
* Cognitive fusion for controlled response generation
* Tiered (curriculum) training across symbolic and neural modules

This document describes the entire system, training pipeline, data structures, and implementation layout.

---

## 1. Project Objectives

### Cognitive Aims

* Ground language in **entities**, **concepts**, and **relationships**.
* Create a latent space aligned with structured semantics rather than pure pattern prediction.
* Enable **neuro-symbolic reasoning** through differentiable rule constraints.
* Introduce a gating controller so the model only produces output when cognitively appropriate.

### Engineering Aims

* Start with text-only experiments.
* Modular design enabling later multimodal extensions.
* Fully-trainable custom LM architecture (no dependency on pretrained LLMs).
* Clean separation between neural encoding and symbolic reasoning layers.

---

## 2. Architecture Overview

### 2.1 High‑Level Flow

```
Input Text
   ↓
Neural Transformer Encoder
   ↓
Entity Extractor ---------------------┐
   ↓                                  │
Concept Mapper ------------------------┤
   ↓                                  │
Relation Inferencer (pairwise) --------┤
   ↓                                  │
Soft‑Logic Constraints (rule loss) ----┘
                ↓
Cognitive Fusion Layer (merges all signals)
                ↓
Response Controller (semantic gating)
                ↓
Transformer Decoder
                ↓
Output Text (or abstain)
```

### 2.2 Components

* **Custom Transformer Encoder**: learns token-level semantics.
* **Entity Extractor**: identifies latent entity vectors.
* **Concept Mapper**: maps entities → concept embeddings.
* **Relation Inferencer**: predicts relations between all entity pairs.
* **Pair Mapping Module**: converts sparse pair logits to dense relation matrices.
* **Soft-Logic Engine**: enforces symbolic rule consistency.
* **Cognitive Fusion**: merges encoder output with symbolic signals.
* **Response Controller**: decides whether the model should answer.
* **Decoder**: generates the text answer.

---

## 3. Dataset Structure

### 3.1 Sample Format

Each sample consists of:

* `input_text`: raw sequence
* `target_text`: correct model output
* `entities`: list of entity spans
* `entity_types`: per-entity type IDs
* `concept_ids`: concept grounding indices
* `relations`: list of `(entity_i, entity_j, relation_id)` tuples
* `rules`: optional symbolic constraints (optional in data; rules can also be global)

### 3.2 Modalities

Current version: **text-only**.
Future expansion: audio and image embeddings feeding into the same semantic backbone.

---

## 4. Training Pipeline (Tiered / Curriculum)

### Stage 1: Train Entity Extractor

* Loss: entity classification, span alignment.

### Stage 2: Train Concept Mapper

* Maps entity embeddings to concept IDs.
* Loss: cross-entropy over concept vocabulary.

### Stage 3: Train Relation Inferencer

* Predicts relation logits between each entity pair.
* Uses `pair_logits_to_matrix` for supervision.

### Stage 4: Soft-Logic Constraint Training

* Apply `SoftLogicConstraints` to enforce rules.
* Auxiliary loss blended with normal training.

### Stage 5: Cognitive Fusion Training

* Train fusion layer to merge semantic streams.

### Stage 6: End-to-End Joint Training

* Unfreeze everything.
* Optimize neural + symbolic losses together.

---

## 5. Implementation Details

### 5.1 Source Files

```
nsmlm/
  model/
    encoder.py
    decoder.py
    entity_extractor.py
    concept_mapper.py
    relation_inferencer.py
    fusion.py
    response_controller.py
    nsmlm_model.py

  logic/
    pair_mapping_and_soft_logic.py

  data/
    toy_dataset.jsonl
    dataset_loader.py

  train/
    train_stage1.py
    train_stage2.py
    train_stage3.py
    train_stage4.py
    train_stage5.py
    train_full.py
```

### 5.2 Losses

```
L_total =
  L_lm * w_lm +
  L_entity * w_e +
  L_concept * w_c +
  L_relation * w_r +
  L_softlogic * w_s +
  L_gate * w_g
```

---

## 6. Soft Logic Integration

The symbolic system integrates directly into the forward pass:

1. Extract entity type probabilities.
2. Produce sparse pair logits.
3. Convert to dense matrix: `pair_logits_to_matrix`.
4. Apply rules through `SoftLogicConstraints`.
5. Add resulting penalty to total loss.
6. Cognitive fusion receives entity embeddings, concept embeddings, and relation matrices.

---

## 7. Toy Dataset Template

Contains example 10–20 fully-specified samples demonstrating the expected structure.
Useful as a scaffold for constructing the initial 10k dataset.

---

## 8. Example Usage

Run:

```
python train/train_stage1.py
python train/train_stage2.py
python train/train_stage3.py
...
python train/train_full.py
```

All scripts produce checkpoints and logs.

---

## 9. Roadmap

* Integrate knowledge graph embeddings.
* Add multi-hop differentiable reasoning.
* Add audio/image encoders.
* Implement distributed training.

---

## 10. License

Specify your license terms here.
