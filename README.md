# NeuroSymbolic Language Model

A neurosymbolic language model that combines transformer-based neural encoding with explicit symbolic reasoning through entity extraction, concept grounding, and relation inference.

## Architecture

```
Input Text
    |
    v
T5/LongT5 Encoder
    |
    +---> Token Entity Classifier (entity type per token)
    |
    +---> Concept Bank (soft concept assignment)
    |
    +---> Node Extraction (top-K entity tokens)
              |
              v
         Graph Neural Network (relation reasoning)
              |
              v
         Pairwise Relation Scorer
              |
              v
         Soft Logic Constraints (differentiable rules)
              |
    +---------+
    |
    v
Combined Memory (encoder + node features)
    |
    v
T5/LongT5 Decoder
    |
    v
Generated Response
```

### Key Components

- **T5/LongT5 Backbone**: Unified encoder-decoder for long context (up to 16k tokens with LongT5)
- **Token Entity Classifier**: Per-token entity type classification
- **Concept Bank**: Learnable concept embeddings with soft assignment
- **Graph Neural Network**: Message passing for relational reasoning
- **Relation Scorer**: Pairwise relation classification between entities
- **Soft Logic Constraints**: Differentiable rule enforcement

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training with Local Data

```bash
# Basic training with default dataset
python train.py --dataset comprehensive_dataset.jsonl --epochs 5

# Quick test run
python train.py --preset testing --epochs 2 --debug
```

### Training with HuggingFace Datasets

```bash
# Prepare data from HuggingFace and train
python train.py --prepare-data --data-sources rebel dolly --epochs 5

# Use specific stages
python train.py --dataset mydata.jsonl --stages symbolic decoder joint
```

### Data Labelling

Create custom training data with the Streamlit labelling interface:

```bash
streamlit run labelling_app.py
```

## Training Stages

Training follows a three-stage curriculum:

### Stage 1: Symbolic Training
- Trains entity classifier, concept bank, GNN, relation scorer
- Decoder is frozen
- Uses entity/relation extraction loss

### Stage 2: Decoder Training  
- Trains decoder for response generation
- Symbolic heads are frozen
- Uses language modeling loss

### Stage 3: Joint Training
- Fine-tunes all components together
- Lower learning rate for stability
- Combined loss from all components

## Dataset Format

Training data uses JSONL format. See `DATASET_FORMAT.md` for full specification.

```json
{
    "text": "Paris is the capital of France.",
    "entities": ["Paris", "France"],
    "concepts": [["city", "location"], ["country", "location"]],
    "relations": [[0, 1, "capital_of"]],
    "should_respond": 1,
    "response": "Yes, Paris is the capital of France."
}
```

## Project Structure

```
neurosymbolic_model/
    config.py                    # Model and training configurations
    train.py                     # Main training script
    labelling_app.py             # Streamlit data labelling UI
    comprehensive_dataset.jsonl  # Example training data
    DATASET_FORMAT.md            # Dataset format specification
    
    model/                       # Model architecture
        neurosymbolic.py         # Main NeuroSymbolicLM class
        entity.py                # Entity classifier, concept bank
        gnn.py                   # Graph neural networks
        logic.py                 # Soft logic constraints
        pooling.py               # Pooling utilities
        encoders.py              # Standalone encoders (optional)
        decoders.py              # Standalone decoders (optional)
    
    data/                        # Data handling
        dataset.py               # ToyCognitiveDataset
        collator.py              # CognitiveCollator for batching
        pipeline.py              # Data download/conversion pipeline
        staged_pipeline.py       # Staged data preparation
    
    training/                    # Training utilities
        trainers.py              # Stage trainers
        evaluation.py            # Generation evaluation
        utils.py                 # Checkpointing, logging, etc.
    
    continual_learning/          # Online learning components
        learner.py               # ContinualLearner class
        memory.py                # Episodic memory
        regularization.py        # EWC, SI, LwF
        safety.py                # Content filtering
        uncertainty.py           # Uncertainty estimation
```

## Configuration

### Model Presets

```python
# RTX 4090 with 8k context (recommended)
python train.py --preset 4090-8k

# RTX 4090 with 16k context
python train.py --preset 4090-16k

# Testing/development
python train.py --preset testing
```

### Command Line Options

```
Data:
  --dataset PATH          Training data file (JSONL)
  --stage2-dataset PATH   Optional separate decoder training data
  --max-samples N         Limit number of samples
  --prepare-data          Download datasets from HuggingFace
  --data-sources          Sources: rebel, dolly, alpaca

Training:
  --stages                Stages to run: symbolic, decoder, joint
  --epochs N              Epochs per stage
  --batch-size N          Training batch size
  --lr FLOAT              Learning rate
  --patience N            Early stopping patience

Hardware:
  --device                cuda or cpu
  --no-amp                Disable mixed precision
  
Output:
  --output-dir PATH       Checkpoint directory
  --resume PATH           Resume from checkpoint
```

## Continual Learning

The model supports online learning with:
- Elastic Weight Consolidation (EWC)
- Synaptic Intelligence (SI)
- Learning without Forgetting (LwF)
- Episodic memory replay
- Uncertainty-based sample selection

See `examples/continual_learning_example.py` for usage.

## Evaluation

Training includes automatic evaluation:
- Generation quality assessment
- BLEU score computation
- Entity F1 metrics

## Hardware Requirements

| Preset | GPU | VRAM | Max Context |
|--------|-----|------|-------------|
| testing | Any | 4GB+ | 512 tokens |
| 4090-8k | RTX 4090 | 24GB | 8k tokens |
| 4090-16k | RTX 4090 | 24GB | 16k tokens |

## License

[Specify your license]
