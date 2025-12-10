"""
Neurosymbolic Language Model Package

A modular implementation combining neural networks with symbolic reasoning:
- Transformer encoder (custom or pre-trained BERT)
- Entity/concept extraction and classification
- Graph neural network for relational reasoning
- Soft logic constraints for rule-based guidance
- Transformer decoder for text generation
- Optional knowledge graph integration

Example usage:
    from neurosymbolic_model import NeuroSymbolicLM
    from neurosymbolic_model.data import ToyCognitiveDataset
    from neurosymbolic_model.training import run_training
    
    model = NeuroSymbolicLM(vocab_size=30522, use_pretrained_encoder=True)
    dataset = ToyCognitiveDataset()
"""

from .model import NeuroSymbolicLM, compute_losses
from .config import ModelConfig, TrainingConfig, KGConfig, SoftLogicConfig

__version__ = "0.1.0"

__all__ = [
    "NeuroSymbolicLM",
    "compute_losses",
    "ModelConfig",
    "TrainingConfig",
    "KGConfig",
    "SoftLogicConfig",
]
