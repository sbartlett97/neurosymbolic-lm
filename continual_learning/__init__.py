"""
Continual Learning module for the Neurosymbolic Model.

This module provides comprehensive support for online/continual learning:
- Uncertainty estimation for identifying what the model doesn't know
- Episodic memory and experience replay
- Regularization to prevent catastrophic forgetting (EWC, SI, LwF)
- Safety filtering to prevent learning harmful content
- Symbolic component updates (concepts, rules, KG)

Main classes:
- ContinuousLearner: Main orchestrator for continual learning
- UncertaintyEstimator: Quantify model uncertainty
- EpisodicMemory: Store and replay experiences
- SafetyRegulator: Filter harmful content
- SymbolicUpdateManager: Update symbolic components

Example usage:
    from continual_learning import ContinuousLearner, ContinuousLearnerConfig
    
    config = ContinuousLearnerConfig(
        uncertainty_threshold=0.5,
        memory_size=1000,
        use_ewc=True,
        enable_safety_filter=True
    )
    
    learner = ContinuousLearner(model, tokenizer, config, device="cuda")
    
    # Process new data
    stats = learner.process_stream(new_samples)
    
    # Check statistics
    print(learner.get_statistics())
"""

from .uncertainty import (
    UncertaintyEstimator,
    UncertaintyMetrics,
    MonteCarloDropout,
    EnsembleUncertainty,
)

from .memory import (
    EpisodicMemory,
    ReplayBuffer,
    MemoryEntry,
)

from .regularization import (
    EWCRegularizer,
    SynapticIntelligence,
    LearningWithoutForgetting,
    CombinedRegularizer,
)

from .safety import (
    SafetyRegulator,
    SafetyVerdict,
    ContentCategory,
    KeywordFilter,
    SemanticSafetyChecker,
    EthicalConstraintChecker,
    SafetyAuditLog,
)

from .symbolic_updates import (
    SymbolicUpdateManager,
    ConceptBankUpdater,
    SoftLogicRuleUpdater,
    KnowledgeGraphUpdater,
    ConceptUpdate,
    RuleUpdate,
)

from .inference import (
    ProductionInference,
    SelfLabelingPipeline,
    EntityExtractor,
    ConceptExtractor,
    RelationExtractor,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelation,
)

from .learner import (
    ContinuousLearner,
    ContinuousLearnerConfig,
    LearningDecision,
    LearningEvent,
    OnlineLearningLoop,
)

__all__ = [
    # Main classes
    "ContinuousLearner",
    "ContinuousLearnerConfig",
    "LearningDecision",
    "LearningEvent",
    "OnlineLearningLoop",
    
    # Uncertainty
    "UncertaintyEstimator",
    "UncertaintyMetrics",
    "MonteCarloDropout",
    "EnsembleUncertainty",
    
    # Memory
    "EpisodicMemory",
    "ReplayBuffer",
    "MemoryEntry",
    
    # Regularization
    "EWCRegularizer",
    "SynapticIntelligence",
    "LearningWithoutForgetting",
    "CombinedRegularizer",
    
    # Safety
    "SafetyRegulator",
    "SafetyVerdict",
    "ContentCategory",
    "KeywordFilter",
    "SemanticSafetyChecker",
    "EthicalConstraintChecker",
    "SafetyAuditLog",
    
    # Symbolic updates
    "SymbolicUpdateManager",
    "ConceptBankUpdater",
    "SoftLogicRuleUpdater",
    "KnowledgeGraphUpdater",
    "ConceptUpdate",
    "RuleUpdate",
    
    # Production inference & self-labeling
    "ProductionInference",
    "SelfLabelingPipeline",
    "EntityExtractor",
    "ConceptExtractor",
    "RelationExtractor",
    "ExtractionResult",
    "ExtractedEntity",
    "ExtractedRelation",
]
