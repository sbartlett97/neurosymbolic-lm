"""Configuration dataclasses for the neurosymbolic model."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class ModelConfig:
    """Configuration for the NeuroSymbolicLM model.
    
    Supports both standard T5 and LongT5 for extended context windows.
    
    For RTX 4090 (24GB VRAM) recommendations:
    - Standard T5-base: up to ~4k context with gradient checkpointing
    - LongT5-base (TGlobal): up to ~16k context with gradient checkpointing
    - LongT5-large: up to ~8k context with gradient checkpointing
    """
    
    # Model backbone
    model_name: str = "google/long-t5-tglobal-base"  # Long context by default
    
    # Context lengths
    max_input_length: int = 4096  # Start conservative, scale up
    max_output_length: int = 1024  # Output typically shorter
    
    # Architecture settings
    d_model: int = 768  # Will be set from model config
    n_entity_types: int = 24  # Headroom over the ~17 coarse taxonomy types (0 = none)
    n_relations: int = 128  # Expanded for production
    n_concepts: int = 1024  # Expanded for production
    concept_dim: int = 256
    node_dim: int = 256
    max_nodes: int = 32  # More nodes for longer documents
    dropout: float = 0.1
    
    # Freeze options
    freeze_encoder: bool = False
    freeze_decoder: bool = False
    
    # Chat mode
    use_chat_mode: bool = False
    default_system_prompt: str = "You are a helpful assistant."

    # Memory efficiency
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True  # If available

    # T5Gemma 2
    use_vision: bool = False
    vision_max_tokens: int = 256

    # Soft entity selection
    use_soft_entity_selection: bool = False
    entity_selection_initial_temp: float = 2.0
    entity_selection_min_temp: float = 0.1

    # Linear graph transformer
    use_linear_graph_transformer: bool = False
    linear_attn_n_random_features: int = 64

    # Global workspace
    use_global_workspace: bool = False
    workspace_n_slots: int = 16
    workspace_n_cycles: int = 1
    
    @classmethod
    def for_4090_16k(cls) -> "ModelConfig":
        """Optimized config for RTX 4090 with 16k context."""
        return cls(
            model_name="google/long-t5-tglobal-base",
            max_input_length=16384,
            max_output_length=2048,
            n_entity_types=24,
            n_relations=128,
            n_concepts=1024,
            max_nodes=48,
            gradient_checkpointing=True,
            use_flash_attention=True,
        )
    
    @classmethod
    def for_4090_8k(cls) -> "ModelConfig":
        """Conservative config for RTX 4090 with 8k context."""
        return cls(
            model_name="google/long-t5-tglobal-base",
            max_input_length=8192,
            max_output_length=2048,
            n_entity_types=24,
            n_relations=128,
            n_concepts=1024,
            max_nodes=32,
            gradient_checkpointing=True,
            use_flash_attention=True,
        )
    
    @classmethod
    def for_4090_chat(cls) -> "ModelConfig":
        """Optimized config for chat fine-tuning on RTX 4090.

        Uses 4K input to accommodate multi-turn conversations
        with gradient checkpointing for memory efficiency.
        """
        return cls(
            model_name="google/long-t5-tglobal-base",
            max_input_length=4096,
            max_output_length=1024,
            n_entity_types=24,
            n_relations=128,
            n_concepts=1024,
            max_nodes=32,
            gradient_checkpointing=True,
            use_flash_attention=True,
            use_chat_mode=True,
        )

    @classmethod
    def for_testing(cls) -> "ModelConfig":
        """Small config for testing and development."""
        return cls(
            model_name="google/long-t5-tglobal-base",
            max_input_length=512,
            max_output_length=256,
            n_entity_types=8,
            n_relations=32,
            n_concepts=256,
            max_nodes=16,
            gradient_checkpointing=False,
        )

    @classmethod
    def for_t5gemma_1b(cls) -> "ModelConfig":
        """T5Gemma 2 1B-1B multimodal config."""
        return cls(
            model_name="google/t5gemma-2-1b-1b",
            max_input_length=8192,
            max_output_length=2048,
            n_entity_types=24,
            n_relations=128,
            n_concepts=1024,
            max_nodes=64,
            gradient_checkpointing=True,
            use_vision=True,
        )

    @classmethod
    def for_t5gemma_270m(cls) -> "ModelConfig":
        """T5Gemma 2 270M lightweight multimodal config."""
        return cls(
            model_name="google/t5gemma-2-270m-270m",
            max_input_length=4096,
            max_output_length=1024,
            n_entity_types=24,
            n_relations=128,
            n_concepts=512,
            max_nodes=32,
            gradient_checkpointing=False,
            use_vision=True,
        )


# Preset configurations for common hardware
MODEL_PRESETS = {
    "4090-16k": ModelConfig.for_4090_16k,
    "4090-8k": ModelConfig.for_4090_8k,
    "4090-chat": ModelConfig.for_4090_chat,
    "testing": ModelConfig.for_testing,
    "t5gemma-1b": ModelConfig.for_t5gemma_1b,
    "t5gemma-270m": ModelConfig.for_t5gemma_270m,
}


@dataclass
class CausalModelConfig:
    """Configuration for the decoder-only NeuroSymbolicCausalLM.

    See docs/DECODER_ONLY_MIGRATION.md. The symbolic heads read a tapped
    mid-layer of the backbone through a small bidirectional extraction
    adapter; GNN node features are injected back as gated soft tokens
    between prompt and response.
    """

    # Backbone
    model_name: str = "Qwen/Qwen3-0.6B-Base"

    # Where the heads read from: fraction of decoder depth (0 < r <= 1)
    tap_layer_ratio: float = 0.66

    # Bidirectional extraction adapter over tapped prompt states
    # (0 layers = pure-causal baseline for the bidirectionality ablation)
    adapter_layers: int = 1
    adapter_heads: int = 4
    adapter_ff_mult: int = 4

    # Symbolic head budgets (match the taxonomy, same as ModelConfig)
    n_entity_types: int = 24
    n_relations: int = 128
    n_concepts: int = 1024
    concept_dim: int = 256
    node_dim: int = 256
    max_nodes: int = 16  # == number of injected soft tokens
    dropout: float = 0.1

    # Soft-token injection
    injection_gate_init: float = 0.1

    # Context lengths
    max_input_length: int = 2048
    max_output_length: int = 512

    # Memory efficiency
    gradient_checkpointing: bool = True

    # LoRA (stage 2/3 on backbones too large for full fine-tuning)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    @classmethod
    def for_qwen3_0_6b(cls) -> "CausalModelConfig":
        return cls(model_name="Qwen/Qwen3-0.6B-Base")

    @classmethod
    def for_qwen3_1_7b(cls) -> "CausalModelConfig":
        return cls(model_name="Qwen/Qwen3-1.7B-Base", gradient_checkpointing=True)

    @classmethod
    def for_gemma3_1b(cls) -> "CausalModelConfig":
        return cls(model_name="google/gemma-3-1b-pt")

    @classmethod
    def for_llama32_1b(cls) -> "CausalModelConfig":
        return cls(model_name="meta-llama/Llama-3.2-1B")

    @classmethod
    def for_qwen3_4b_lora(cls) -> "CausalModelConfig":
        return cls(
            model_name="Qwen/Qwen3-4B-Base",
            use_lora=True,
            max_input_length=4096,
        )

    @classmethod
    def for_testing(cls) -> "CausalModelConfig":
        """Tiny offline backbone built from config — no downloads needed."""
        return cls(
            model_name="__tiny__",
            n_entity_types=8,
            n_relations=32,
            n_concepts=64,
            concept_dim=32,
            node_dim=32,
            max_nodes=4,
            adapter_layers=1,
            adapter_heads=2,
            max_input_length=128,
            max_output_length=64,
            gradient_checkpointing=False,
        )


CAUSAL_MODEL_PRESETS = {
    "qwen3-0.6b": CausalModelConfig.for_qwen3_0_6b,
    "qwen3-1.7b": CausalModelConfig.for_qwen3_1_7b,
    "qwen3-4b-lora": CausalModelConfig.for_qwen3_4b_lora,
    "gemma3-1b": CausalModelConfig.for_gemma3_1b,
    "llama3.2-1b": CausalModelConfig.for_llama32_1b,
    "causal-testing": CausalModelConfig.for_testing,
}


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

    # Two-phase training
    training_phase: str = "pretrain"  # "pretrain", "chat", or "both"
    chat_lr: float = 1e-5
    chat_epochs: int = 3
    
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
class ProductionTrainingConfig:
    """Configuration for production-scale training with memory efficiency.
    
    Optimized for single RTX 4090 (24GB VRAM) with gradient accumulation
    and mixed precision training.
    """
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    
    # Batch settings - effective batch = batch_size * gradient_accumulation
    batch_size: int = 2  # Per-device batch size
    gradient_accumulation_steps: int = 8  # Effective batch size = 16
    
    # Training duration
    epochs_per_stage: int = 10
    max_steps: Optional[int] = None  # Override epochs if set
    
    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Memory efficiency
    use_amp: bool = True  # Mixed precision
    gradient_checkpointing: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 1
    max_checkpoints: int = 5
    
    # Evaluation
    eval_every_n_epochs: int = 2
    num_eval_samples: int = 10
    
    # Logging
    log_dir: str = "runs"
    enable_tensorboard: bool = True
    
    # Early stopping
    early_stopping_patience: int = 3
    
    # Data
    data_dir: str = "data/processed"
    max_samples: Optional[int] = None
    
    @classmethod
    def for_4090(cls) -> "ProductionTrainingConfig":
        """Optimized settings for RTX 4090."""
        return cls(
            batch_size=2,
            gradient_accumulation_steps=8,
            use_amp=True,
            gradient_checkpointing=True,
            num_workers=4,
        )
    
    @classmethod
    def for_a100(cls) -> "ProductionTrainingConfig":
        """Optimized settings for A100 (40GB/80GB)."""
        return cls(
            batch_size=8,
            gradient_accumulation_steps=4,
            use_amp=True,
            gradient_checkpointing=True,
            num_workers=8,
        )
    
    @classmethod
    def for_testing(cls) -> "ProductionTrainingConfig":
        """Quick testing configuration."""
        return cls(
            batch_size=2,
            gradient_accumulation_steps=1,
            epochs_per_stage=2,
            eval_every_n_epochs=1,
            max_samples=100,
            use_amp=False,
            gradient_checkpointing=False,
        )


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
