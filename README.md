# Neuro-Symbolic Language Model (NS-LM)

A research prototype implementing a unified neuro-symbolic architecture designed to move language models closer to human-aligned cognition. This system integrates:

* Transformer-based neural text encoding (supports custom or pre-trained encoders like BERT)
* Entity extraction and classification
* Multi-label concept grounding
* Relation inference via Graph Neural Networks
* Differentiable soft-logic rule enforcement
* Cognitive fusion for controlled response generation
* Autoregressive sequence generation with transformer decoder
* Tiered (curriculum) training across symbolic and neural modules

This document describes the entire system, training pipeline, data structures, and implementation layout.

---

## 1. Project Objectives

### Cognitive Aims

* Ground language in **entities**, **concepts**, and **relationships**.
* Create a latent space aligned with structured semantics rather than pure pattern prediction.
* Enable **neuro-symbolic reasoning** through differentiable rule constraints.
* Introduce a gating controller so the model only produces output when cognitively appropriate.
* Generate coherent text responses using autoregressive sequence generation.

### Engineering Aims

* Start with text-only experiments.
* Modular design enabling later multimodal extensions.
* Support for both custom and pre-trained transformer architectures (BERT, T5, BART).
* Clean separation between neural encoding and symbolic reasoning layers.
* Full sequence generation capabilities with configurable sampling strategies.

---

## 2. Architecture Overview

### 2.1 High-Level Flow

```
Input Text
   ↓
Neural Transformer Encoder (Custom or Pre-trained BERT)
   ↓
Entity Extractor (Token-level Classification) ────────┐
   ↓                                                   │
Concept Mapper (Multi-label Concept Bank) ────────────┤
   ↓                                                   │
Node Representation (Top-K Entity Tokens)             │
   ↓                                                   │
Graph Neural Network (Relation Reasoning) ────────────┤
   ↓                                                   │
Pairwise Relation Scorer ─────────────────────────────┤
   ↓                                                   │
Soft-Logic Constraints (Rule Loss) ───────────────────┘
                ↓
Cognitive Fusion Layer (merges encoder + node representations)
                ↓
Response Controller (semantic gating: answer/abstain/ask)
                ↓
Transformer Decoder (Custom or Pre-trained T5/BART)
                ↓
Autoregressive Sequence Generation
                ↓
Output Text (or abstain)
```

### 2.2 Components

* **Transformer Encoder**: 
  - Custom `SimpleTransformerEncoder` or pre-trained BERT via `PretrainedEncoderWrapper`
  - Learns token-level semantics
  - Supports frozen or fine-tunable pre-trained models

* **Entity Extractor** (`TokenEntityClassifier`): 
  - Token-level entity type classification
  - Outputs logits over entity types for each token position

* **Concept Mapper** (`ConceptBank` + `concept_proj`): 
  - Maps entities to concept embeddings via learnable concept bank
  - Supports multi-label concept assignment (entities can have multiple concepts)
  - Uses soft assignment for differentiable concept grounding

* **Node Representation**:
  - Extracts top-K entity tokens as nodes
  - Projects to node dimension for graph processing

* **Graph Neural Network** (`SimpleGNN`): 
  - Refines node representations through message passing
  - Enables relation reasoning between entities

* **Relation Inferencer** (`rel_scorer`): 
  - Predicts relations between all entity pairs
  - Outputs sparse pair logits converted to dense relation matrices

* **Soft-Logic Engine** (`SoftLogicConstraints`): 
  - Enforces symbolic rule consistency as differentiable constraints
  - Rules specify expected relations between entity type pairs
  - Adds auxiliary loss to guide training

* **Response Controller** (`Controller`): 
  - Decides whether the model should answer, abstain, or ask for clarification
  - Uses pooled token and node representations

* **Transformer Decoder**: 
  - Custom `SimpleTransformerDecoder` or pre-trained T5/BART via `PretrainedDecoderWrapper`
  - Generates sequences autoregressively
  - Uses encoder outputs and node representations as memory
  - Supports causal masking for autoregressive generation

* **Generation Method** (`generate()`): 
  - Autoregressive sequence generation with configurable sampling
  - Supports greedy decoding, temperature sampling, top-k, and top-p (nucleus) sampling
  - Early stopping on EOS tokens

---

## 3. Dataset Structure

### 3.1 Sample Format

Each sample in the dataset consists of:

```python
{
    "text": "The cat chased the mouse.",
    "entities": ["cat", "mouse"],
    "entity_spans": [(4, 7), (20, 25)],  # Character-level spans
    "concepts": [["animal", "predator"], ["animal", "prey"]],  # Multi-label concepts per entity
    "relations": [(0, 1, "chases")],  # (entity_i, entity_j, relation_name)
    "should_respond": 1,  # 1 if model should generate response, 0 otherwise
    "response": "Yes, the cat chases the mouse."  # Target response text (optional)
}
```

**Key Features:**
- **Multiple concepts per entity**: Each entity can have multiple concept labels (e.g., `["person", "professional", "medical"]`)
- **Multiple relationships**: Entities can participate in multiple relations with different entities
- **Backward compatible**: Single concept strings are automatically converted to lists
- **Question/Statement pairs**: The dataset automatically creates question versions of statements for training

### 3.2 Modalities

Current version: **text-only**.
Future expansion: audio and image embeddings feeding into the same semantic backbone.

---

## 4. Training Pipeline (Tiered / Curriculum)

The training is organized into 4 main stages plus an intermediate stage:

### Stage 1: MLM Pretraining (Optional)

* **Purpose**: Pre-train the encoder on masked language modeling
* **Loss**: Cross-entropy on masked token prediction
* **Skip condition**: Automatically skipped if using a pre-trained encoder (e.g., BERT)
* **Trainable components**: Encoder only (if not using pre-trained)

### Stage 2: Entity & Concept Extraction

* **Purpose**: Train entity classification and concept mapping
* **Losses**: 
  - Entity classification loss (token-level)
  - Multi-label concept classification loss (BCE)
  - Soft-logic constraint loss (optional, weighted)
* **Trainable components**: 
  - Entity classifier (`token_ent`)
  - Concept head (`concept_head`)
  - GNN (`gnn`)
  - Relation scorer (`rel_scorer`)
  - Node projection (`node_proj`)
  - Encoder (if not frozen)

### Stage 3: Response Controller

* **Purpose**: Train the controller to decide when to respond
* **Loss**: Binary cross-entropy on `should_respond` prediction
* **Trainable components**: Controller only

### Stage 3.5: Decoder Response Generation

* **Purpose**: Train the decoder to generate response sequences
* **Loss**: Cross-entropy on next-token prediction (teacher forcing)
* **Trainable components**: Decoder only
* **Evaluation**: Automatic generation evaluation after training

### Stage 4: Joint End-to-End Training

* **Purpose**: Fine-tune all components together
* **Losses**: Combined loss from all previous stages:
  - Entity classification loss
  - Concept classification loss
  - Response controller loss
  - Decoder language modeling loss
  - Soft-logic constraint loss (weighted)
* **Trainable components**: All trainable parameters (encoder, decoder, heads, GNN, etc.)
* **Evaluation**: Periodic generation evaluation (every 5 epochs)

---

## 5. Implementation Details

### 5.1 Source Files

```
neurosymbolic_model/
  model.py              # Complete model architecture (NeuroSymbolicLM)
  main.py               # Training pipeline, dataset, and utilities
  comprehensive_dataset.jsonl  # Extended dataset examples
  requirements.txt      # Python dependencies
  README.md            # This file
```

### 5.2 Key Classes and Functions

**Model (`model.py`):**
- `NeuroSymbolicLM`: Main model class
- `SimpleTransformerEncoder`: Custom encoder implementation
- `PretrainedEncoderWrapper`: Wrapper for pre-trained encoders (BERT)
- `SimpleTransformerDecoder`: Custom decoder implementation
- `PretrainedDecoderWrapper`: Wrapper for pre-trained decoders (T5, BART)
- `TokenEntityClassifier`: Entity type classification head
- `ConceptBank`: Learnable concept embeddings
- `SimpleGNN`: Graph neural network for relation reasoning
- `SoftLogicConstraints`: Differentiable rule enforcement
- `Controller`: Response gating mechanism

**Training (`main.py`):**
- `ToyCognitiveDataset`: In-memory dataset with question/statement pairs
- `CognitiveCollator`: Batch collation with multi-label concept encoding
- `Stage1_MLM_Trainer`: MLM pretraining
- `Stage2_Symbolic_Trainer`: Entity and concept training
- `Stage3_Control_Trainer`: Controller training
- `Stage3_Decoder_Trainer`: Decoder training
- `Stage4_Joint_Trainer`: Joint end-to-end training
- `run_training()`: Main training orchestration function
- `evaluate_generation()`: Generation evaluation function
- `generate_response()`: Inference utility function

### 5.3 Loss Components

```
L_total =
  L_entity * w_e +           # Entity classification
  L_concept * w_c +          # Multi-label concept classification
  L_controller * w_ctrl +    # Response controller
  L_decoder * w_dec +        # Decoder language modeling
  L_softlogic * w_s          # Soft-logic constraints (default w_s=0.1)
```

---

## 6. Soft Logic Integration

The symbolic system integrates directly into the forward pass:

1. Extract entity type probabilities from token-level predictions
2. Produce sparse pair logits for all entity pairs
3. Convert to dense matrix: `pair_logits_to_matrix`
4. Apply rules through `SoftLogicConstraints`
5. Add resulting penalty to total loss
6. Rules specify expected relations between entity type pairs (e.g., "animals can chase other animals")

**Example Rule Configuration:**
```python
soft_logic_rules = {
    "concept_to_entity_type_map": {
        "animal": 0, "person": 1, "location": 2, "object": 3, "attribute": 4
    },
    "rules": [
        {"concept_a": "animal", "concept_b": "animal", "relation": "chases", 
         "weight": 1.0, "polarity": 1},  # Encourage
        {"concept_a": "attribute", "concept_b": "location", "relation": "capital_of", 
         "weight": 0.5, "polarity": -1},  # Discourage
    ]
}
```

---

## 7. Generation Capabilities

The model supports autoregressive sequence generation with the `generate()` method:

**Features:**
- Greedy decoding (argmax) or sampling
- Temperature scaling
- Top-k filtering
- Top-p (nucleus) sampling
- Early stopping on EOS tokens
- Causal masking for autoregressive generation

**Usage:**
```python
generated_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=128,
    bos_token_id=1,
    eos_token_id=2,
    temperature=1.0,
    do_sample=False,  # Use greedy decoding
    top_k=None,
    top_p=None
)
```

**Inference Utility:**
```python
from main import generate_response

response = generate_response(
    model, 
    tokenizer, 
    "Is Paris the capital of France?",
    device="cuda",
    max_length=128,
    do_sample=False
)
```

---

## 8. Example Usage

### Basic Training

```python
from transformers import AutoTokenizer
from model import NeuroSymbolicLM
from main import run_training

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token

# Initialize model with pre-trained encoder
model = NeuroSymbolicLM(
    vocab_size=len(tokenizer),
    d_model=768,  # BERT-base hidden size
    n_entity_types=8,
    n_concepts=512,
    concept_dim=256,
    node_dim=256,
    max_nodes=16,
    use_pretrained_encoder=True,
    pretrained_model_name="bert-base-uncased",
    freeze_encoder=True,  # Or False to fine-tune
    use_pretrained_decoder=True,
    pretrained_decoder_name="t5-small",
    freeze_decoder=False
)

# Configure soft logic rules (optional)
soft_logic_rules = {
    "concept_to_entity_type_map": {
        "animal": 0, "person": 1, "location": 2, "object": 3, "attribute": 4
    },
    "rules": [
        {"concept_a": "animal", "concept_b": "animal", "relation": "chases", 
         "weight": 1.0, "polarity": 1},
    ]
}

# Run training
device = "cuda" if torch.cuda.is_available() else "cpu"
trained_model = run_training(
    tokenizer, 
    model, 
    device=device, 
    epochs_per_stage=10,
    skip_stage1_if_pretrained=True,
    soft_logic_weight=0.1,
    soft_logic_rules=soft_logic_rules
)
```

### Generation and Evaluation

```python
from main import generate_response, evaluate_generation, print_generation_results

# Generate a response
response = generate_response(
    trained_model,
    tokenizer,
    "Is Paris the capital of France?",
    device=device,
    max_length=128
)
print(f"Generated: {response}")

# Evaluate on dataset
results = evaluate_generation(
    trained_model,
    tokenizer,
    dataset,
    device=device,
    num_samples=10
)
print_generation_results(results, num_to_print=5)
```

### Running the Training Script

```bash
python main.py
```

The script will:
1. Initialize the model with pre-trained BERT encoder and T5 decoder
2. Run all training stages automatically
3. Evaluate generation after Stage 3.5 and periodically during Stage 4
4. Perform final evaluation and show example generations

---

## 9. Model Configuration Options

### Encoder Options

- **Custom encoder**: Set `use_pretrained_encoder=False` (default)
- **Pre-trained BERT**: Set `use_pretrained_encoder=True`, `pretrained_model_name="bert-base-uncased"`
- **Freeze encoder**: Set `freeze_encoder=True` to keep pre-trained weights frozen

### Decoder Options

- **Custom decoder**: Set `use_pretrained_decoder=False` (default)
- **Pre-trained T5**: Set `use_pretrained_decoder=True`, `pretrained_decoder_name="t5-small"`
- **Pre-trained BART**: Set `pretrained_decoder_name="facebook/bart-base"`
- **Freeze decoder**: Set `freeze_decoder=True` to keep pre-trained weights frozen

### Architecture Parameters

- `d_model`: Model dimension (768 for BERT-base, 512 for custom)
- `n_entity_types`: Number of entity type classes (default: 8)
- `n_concepts`: Size of concept bank (default: 512)
- `concept_dim`: Dimension of concept embeddings (default: 256)
- `node_dim`: Dimension of node representations (default: 256)
- `max_nodes`: Maximum number of entity nodes (default: 16)

---

## 10. Evaluation and Monitoring

The training pipeline includes automatic evaluation:

- **After Stage 3.5**: Generation evaluation on 5 samples
- **During Stage 4**: Periodic evaluation every 5 epochs
- **After training**: Final evaluation on 10 samples

Evaluation shows:
- Input text
- Target response
- Generated response

This helps monitor generation quality throughout training.

---

## 11. Roadmap

* [ ] **Integrate knowledge graph embeddings** - See `KG_EMBEDDINGS_GUIDE.md` for detailed implementation plan
  - Initialize concept/entity embeddings from pre-trained KG embeddings (ConceptNet, Wikidata)
  - Add KG regularization to maintain semantic consistency
  - Enable entity linking to leverage external knowledge
  - Expected benefits: Better generalization, handling of rare entities, improved relation understanding
* [ ] Add multi-hop differentiable reasoning
* [ ] Add audio/image encoders for multimodal support
* [ ] Implement distributed training
* [ ] Add beam search for generation
* [ ] Implement more sophisticated evaluation metrics (BLEU, ROUGE)
* [ ] Support for larger pre-trained models (BERT-large, T5-base/large)

---

## 12. Additional Documentation

- **`KG_EMBEDDINGS_GUIDE.md`**: Comprehensive guide on integrating knowledge graph embeddings
- **`SOFT_LOGIC_GUIDE.md`**: Detailed explanation of soft-logic constraints
- **`PRETRAINED_ENCODER_GUIDE.md`**: Guide for using pre-trained encoders
- **`PRETRAINED_DECODER_RECOMMENDATIONS.md`**: Recommendations for pre-trained decoders
- **`DECODER_TRAINING_GUIDE.md`**: Guide for training the decoder component

---

## 13. Dependencies

See `requirements.txt` for full list. Key dependencies:

- `torch`: PyTorch deep learning framework
- `transformers`: HuggingFace transformers library for pre-trained models
- `numpy`: Numerical computations

---

## 14. License

Specify your license terms here.
