"""Configuration dataclasses for the neurosymbolic model."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class ModelConfig:
    """Configuration for the NeuroSymbolicLM model."""
    
    vocab_size: int = 30522  # BERT default
    d_model: int = 768  # BERT-base hidden size
    n_entity_types: int = 8
    n_relations: int = 32
    n_concepts: int = 512  # Can be set dynamically based on dataset
    concept_dim: int = 256
    node_dim: int = 256
    max_nodes: int = 16
    dropout: float = 0.1
    
    # Encoder settings
    use_pretrained_encoder: bool = True
    pretrained_model_name: str = "bert-base-uncased"
    freeze_encoder: bool = True
    encoder_nhead: int = 8
    encoder_nlayer: int = 6
    
    # Decoder settings
    use_pretrained_decoder: bool = True
    pretrained_decoder_name: str = "t5-small"
    freeze_decoder: bool = False
    decoder_nhead: int = 8
    decoder_nlayer: int = 6
    
    # Knowledge graph settings
    use_kg: bool = False
    kg_embed_dim: int = 300
    use_kg_gnn: bool = False
    use_path_reasoning: bool = False
    max_path_length: int = 3


@dataclass
class KGConfig:
    """Configuration for knowledge graph integration."""
    
    kg_type: str = "conceptnet"  # Options: "conceptnet", "wikidata", "wordnet", "generic"
    embedding_path: Optional[str] = None
    triples_path: Optional[str] = None
    entity_mapping: Optional[Dict[str, str]] = None
    max_triples: int = 50000


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    device: str = "cpu"
    epochs_per_stage: int = 10
    batch_size: int = 8
    num_workers: int = 0
    learning_rate: float = 1e-4
    joint_learning_rate: float = 1e-5
    soft_logic_weight: float = 0.1
    skip_stage1_if_pretrained: bool = True
    
    # Gradient clipping
    grad_clip_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = False
    
    # Learning rate scheduling
    use_scheduler: bool = True
    warmup_epochs: int = 2
    
    # Checkpointing
    checkpoint_dir: Optional[str] = "checkpoints"
    max_checkpoints: int = 3
    save_best_only: bool = True
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    
    # Logging
    enable_tensorboard: bool = True
    log_dir: str = "runs"
    eval_every_n_epochs: int = 5
    
    # Dataset
    dataset_file_path: Optional[str] = None
    
    # Soft logic rule generation
    use_dynamic_rules: bool = True
    min_rule_frequency: int = 2
    min_rule_confidence: float = 0.2
    max_rules: int = 50
    include_negative_rules: bool = True


@dataclass
class SoftLogicRule:
    """A single soft logic rule."""
    
    concept_a: str
    concept_b: str
    relation: str
    weight: float = 1.0
    polarity: int = 1  # 1 = encourage, -1 = discourage


@dataclass
class SoftLogicConfig:
    """Configuration for soft logic rules."""
    
    concept_to_entity_type_map: Dict[str, int] = field(default_factory=dict)
    rules: List[SoftLogicRule] = field(default_factory=list)
    
    def to_dict_list(self) -> List[Dict]:
        """Convert rules to list of dicts for compatibility."""
        return [
            {
                "concept_a": r.concept_a,
                "concept_b": r.concept_b,
                "relation": r.relation,
                "weight": r.weight,
                "polarity": r.polarity
            }
            for r in self.rules
        ]


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    
    num_eval_samples: int = 10
    max_generation_length: int = 128
    compute_bleu: bool = True
    compute_entity_f1: bool = True
    bleu_max_n: int = 4
