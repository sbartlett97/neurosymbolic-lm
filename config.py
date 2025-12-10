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


@dataclass
class ContinualLearningConfig:
    """Configuration for continual/online learning."""
    
    # Uncertainty estimation
    uncertainty_threshold: float = 0.5
    uncertainty_method: str = "mc_dropout"  # 'mc_dropout', 'ensemble', 'combined'
    mc_samples: int = 10
    
    # Episodic memory
    memory_size: int = 1000
    memory_strategy: str = "hybrid"  # 'reservoir', 'uncertainty', 'diversity', 'balanced', 'hybrid'
    diversity_weight: float = 0.3
    
    # Experience replay
    replay_ratio: float = 0.3
    replay_strategy: str = "random"  # 'random', 'weighted', 'uncertain', 'recent'
    
    # Regularization (anti-forgetting)
    use_ewc: bool = True
    use_si: bool = False
    use_lwf: bool = True
    ewc_weight: float = 1000.0
    si_weight: float = 100.0
    lwf_alpha: float = 0.5
    lwf_temperature: float = 2.0
    
    # Safety filtering
    enable_safety_filter: bool = True
    safety_strictness: str = "medium"  # 'low', 'medium', 'high', 'maximum'
    safety_log_path: Optional[str] = "safety_audit.jsonl"
    
    # Learning parameters
    online_learning_rate: float = 1e-4
    max_steps_per_event: int = 10
    online_batch_size: int = 8
    
    # Knowledge consolidation
    consolidate_every_n_events: int = 10
    min_samples_for_consolidation: int = 50
    
    # Component freezing during online learning
    freeze_encoder_online: bool = True
    freeze_decoder_online: bool = False
    
    # Symbolic updates
    enable_concept_expansion: bool = True
    enable_rule_learning: bool = True
    enable_kg_updates: bool = True
    max_concepts: int = 2048
    max_rules: int = 200


@dataclass
class SafetyConfig:
    """Configuration for safety and content filtering."""
    
    # Strictness level
    strictness: str = "medium"  # 'low', 'medium', 'high', 'maximum'
    
    # Component toggles
    enable_keyword_filter: bool = True
    enable_semantic_filter: bool = True
    enable_ethical_filter: bool = True
    enable_audit_logging: bool = True
    
    # Semantic filter settings
    semantic_similarity_threshold: float = 0.7
    
    # Audit settings
    log_path: Optional[str] = "safety_audit.jsonl"
    log_all_checks: bool = False  # Log safe content too
    
    # Blocklist/allowlist paths (for loading from files)
    blocklist_path: Optional[str] = None
    allowlist_path: Optional[str] = None
