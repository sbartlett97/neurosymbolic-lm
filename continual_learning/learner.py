"""
Main Continuous Learner orchestrating all continual learning components.

Provides a unified interface for:
- Processing new data with safety filtering
- Uncertainty-based learning decisions
- Experience replay with regularization
- Symbolic component updates
- Knowledge consolidation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
from datetime import datetime
import json
from pathlib import Path

from .uncertainty import UncertaintyEstimator, UncertaintyMetrics
from .memory import EpisodicMemory, ReplayBuffer, MemoryEntry
from .regularization import CombinedRegularizer, EWCRegularizer
from .safety import SafetyRegulator, SafetyVerdict, ContentCategory
from .symbolic_updates import SymbolicUpdateManager
from .inference import ProductionInference, SelfLabelingPipeline


class LearningDecision(Enum):
    """Decision on whether to learn from a sample."""
    LEARN = "learn"
    SKIP_CONFIDENT = "skip_confident"
    SKIP_UNSAFE = "skip_unsafe"
    DEFER = "defer"  # Store for later review


@dataclass
class LearningEvent:
    """Record of a learning event."""
    timestamp: str
    num_samples: int
    num_learned: int
    num_skipped_confident: int
    num_skipped_unsafe: int
    avg_loss: float
    avg_uncertainty: float
    triggered_consolidation: bool = False


@dataclass
class ContinuousLearnerConfig:
    """Configuration for the continuous learner."""
    # Uncertainty settings
    uncertainty_threshold: float = 0.5
    uncertainty_method: str = "mc_dropout"
    mc_samples: int = 10
    
    # Memory settings
    memory_size: int = 1000
    memory_strategy: str = "hybrid"
    replay_ratio: float = 0.3
    replay_strategy: str = "random"
    
    # Regularization settings
    use_ewc: bool = True
    use_lwf: bool = True
    ewc_weight: float = 1000.0
    lwf_alpha: float = 0.5
    
    # Safety settings
    safety_strictness: str = "medium"
    enable_safety_filter: bool = True
    safety_log_path: str = "safety_audit.jsonl"
    
    # Learning settings
    learning_rate: float = 1e-4
    max_steps_per_event: int = 10
    batch_size: int = 8
    grad_clip_norm: float = 1.0
    
    # Consolidation settings
    consolidate_every_n_events: int = 10
    min_samples_for_consolidation: int = 50
    
    # Freezing settings
    freeze_encoder: bool = True
    freeze_decoder: bool = False
    
    # Logging
    log_dir: str = "continual_learning_logs"
    save_checkpoints: bool = True
    
    # Self-labeling settings (for text-only input)
    enable_self_labeling: bool = True
    self_label_entity_threshold: float = 0.7
    self_label_concept_threshold: float = 0.5
    self_label_relation_threshold: float = 0.5
    self_label_min_confidence: float = 0.6
    min_entities_for_learning: int = 1


class ContinuousLearner:
    """
    Main orchestrator for continuous online learning.
    
    Combines uncertainty estimation, safety filtering, experience replay,
    regularization, and symbolic updates for safe, effective online learning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[ContinuousLearnerConfig] = None,
        device: str = "cpu"
    ):
        """
        Args:
            model: The neurosymbolic model to train
            tokenizer: Tokenizer for text processing
            config: Configuration options
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ContinuousLearnerConfig()
        self.device = device
        
        # Initialize components
        self._init_components()
        
        # Statistics
        self.total_samples_seen = 0
        self.total_samples_learned = 0
        self.learning_events: List[LearningEvent] = []
        self.samples_since_consolidation = 0
        
        # Logging
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_components(self):
        """Initialize all sub-components."""
        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(
            self.model,
            method=self.config.uncertainty_method,
            num_samples=self.config.mc_samples
        )
        
        # Episodic memory
        self.memory = EpisodicMemory(
            max_size=self.config.memory_size,
            selection_strategy=self.config.memory_strategy
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            self.memory,
            replay_ratio=self.config.replay_ratio,
            replay_strategy=self.config.replay_strategy
        )
        
        # Regularization
        self.regularizer = CombinedRegularizer(
            self.model,
            use_ewc=self.config.use_ewc,
            use_si=False,
            use_lwf=self.config.use_lwf,
            ewc_weight=self.config.ewc_weight,
            lwf_alpha=self.config.lwf_alpha
        )
        
        # Safety regulator
        self.safety_regulator = SafetyRegulator(
            strictness=self.config.safety_strictness,
            enable_logging=True,
            log_path=str(self.log_dir / self.config.safety_log_path)
        ) if self.config.enable_safety_filter else None
        
        # Symbolic updater
        self.symbolic_updater = SymbolicUpdateManager(self.model)
        
        # Production inference and self-labeling
        self.inference = ProductionInference(
            self.model,
            self.tokenizer,
            entity_confidence=self.config.self_label_entity_threshold,
            device=self.device
        )
        
        self.self_labeler = SelfLabelingPipeline(
            self.inference,
            entity_threshold=self.config.self_label_entity_threshold,
            concept_threshold=self.config.self_label_concept_threshold,
            relation_threshold=self.config.self_label_relation_threshold,
            min_entities=self.config.min_entities_for_learning
        ) if self.config.enable_self_labeling else None
        
        # Optimizer (created on first learning event)
        self.optimizer = None
    
    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Get or create optimizer."""
        if self.optimizer is None:
            # Determine which parameters to train
            params = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                if self.config.freeze_encoder and "encoder" in name:
                    continue
                
                if self.config.freeze_decoder and "decoder" in name:
                    continue
                
                params.append(param)
            
            self.optimizer = Adam(params, lr=self.config.learning_rate)
        
        return self.optimizer
    
    def process_stream(
        self,
        samples: List[Dict],
        force_learn: bool = False,
        callback: Optional[Callable[[Dict, LearningDecision], None]] = None
    ) -> Dict[str, Any]:
        """
        Process a stream of new samples.
        
        This is the main entry point for continuous learning.
        Supports both labeled samples and text-only samples (via self-labeling).
        
        Args:
            samples: List of new samples to process. Each sample can be:
                - Full dict with 'text', 'entities', 'concepts', etc.
                - Text-only dict with just 'text' key (will be self-labeled)
                - Plain string (will be converted to {'text': string})
            force_learn: Force learning even for confident predictions
            callback: Optional callback for each sample decision
        
        Returns:
            Statistics about the processing
        """
        stats = {
            "total": len(samples),
            "learned": 0,
            "skipped_confident": 0,
            "skipped_unsafe": 0,
            "skipped_low_confidence": 0,
            "deferred": 0,
            "self_labeled": 0,
            "uncertainties": [],
            "safety_violations": []
        }
        
        samples_to_learn = []
        
        for sample in samples:
            self.total_samples_seen += 1
            
            # Normalize sample format
            sample = self._normalize_sample(sample)
            
            # Step 1: Safety check (on raw text)
            if self.safety_regulator:
                verdict = self._check_safety(sample)
                if not verdict.is_safe:
                    stats["skipped_unsafe"] += 1
                    stats["safety_violations"].append(verdict.category.value)
                    if callback:
                        callback(sample, LearningDecision.SKIP_UNSAFE)
                    continue
            
            # Step 2: Self-labeling if needed (text-only input)
            if self._needs_labeling(sample):
                if self.self_labeler:
                    labeled_sample, confidence = self.self_labeler.generate_labels(sample["text"])
                    if labeled_sample and confidence >= self.config.self_label_min_confidence:
                        sample = labeled_sample
                        stats["self_labeled"] += 1
                    else:
                        # Confidence too low for self-labeling
                        stats["skipped_low_confidence"] += 1
                        if callback:
                            callback(sample, LearningDecision.DEFER)
                        continue
                else:
                    # No self-labeler and no labels - skip
                    stats["skipped_low_confidence"] += 1
                    if callback:
                        callback(sample, LearningDecision.DEFER)
                    continue
            
            # Step 3: Uncertainty estimation
            uncertainty = self._estimate_uncertainty(sample)
            stats["uncertainties"].append(uncertainty.overall_score)
            
            # Step 4: Decision
            decision = self._make_learning_decision(uncertainty, force_learn)
            
            if decision == LearningDecision.LEARN:
                samples_to_learn.append((sample, uncertainty))
                stats["learned"] += 1
            elif decision == LearningDecision.SKIP_CONFIDENT:
                stats["skipped_confident"] += 1
            elif decision == LearningDecision.DEFER:
                stats["deferred"] += 1
            
            # Add to memory regardless of learning decision
            self.memory.add(
                sample,
                uncertainty=uncertainty.overall_score,
                concepts=sample.get("concepts", [])
            )
            
            # Update symbolic components (only if labeled)
            if not self._needs_labeling(sample):
                self.symbolic_updater.process_sample(sample)
            
            if callback:
                callback(sample, decision)
        
        # Step 5: Learn from uncertain samples
        if samples_to_learn:
            avg_loss = self._learn_batch(samples_to_learn)
            stats["avg_loss"] = avg_loss
            self.total_samples_learned += len(samples_to_learn)
            self.samples_since_consolidation += len(samples_to_learn)
            
            # Record learning event
            event = LearningEvent(
                timestamp=datetime.now().isoformat(),
                num_samples=len(samples),
                num_learned=len(samples_to_learn),
                num_skipped_confident=stats["skipped_confident"],
                num_skipped_unsafe=stats["skipped_unsafe"],
                avg_loss=avg_loss,
                avg_uncertainty=sum(u.overall_score for _, u in samples_to_learn) / len(samples_to_learn)
            )
            self.learning_events.append(event)
        
        # Step 6: Check for consolidation
        if self._should_consolidate():
            self._consolidate()
            if self.learning_events:
                self.learning_events[-1].triggered_consolidation = True
        
        return stats
    
    def process_text_stream(
        self,
        texts: List[str],
        force_learn: bool = False,
        callback: Optional[Callable[[Dict, LearningDecision], None]] = None
    ) -> Dict[str, Any]:
        """
        Process a stream of raw text (production mode).
        
        This is the recommended entry point for production systems
        where only text is available (no pre-extracted labels).
        
        Args:
            texts: List of text strings to process
            force_learn: Force learning even for confident predictions
            callback: Optional callback for each sample decision
        
        Returns:
            Statistics about the processing
        """
        # Convert texts to sample dicts
        samples = [{"text": text} for text in texts]
        return self.process_stream(samples, force_learn, callback)
    
    def _normalize_sample(self, sample) -> Dict:
        """Normalize sample to standard dict format."""
        if isinstance(sample, str):
            return {"text": sample}
        return sample
    
    def _needs_labeling(self, sample: Dict) -> bool:
        """Check if sample needs self-labeling."""
        # Needs labeling if missing key structural elements
        has_entities = bool(sample.get("entities"))
        has_entity_types = bool(sample.get("entity_type_labels"))
        return not (has_entities and has_entity_types)
    
    def _check_safety(self, sample: Dict) -> SafetyVerdict:
        """Check if a sample is safe to learn from."""
        text = sample.get("text", "")
        entities = sample.get("entities", [])
        
        return self.safety_regulator.check(
            text,
            entities=entities,
            source="continuous_learning"
        )
    
    def _estimate_uncertainty(self, sample: Dict) -> UncertaintyMetrics:
        """Estimate model's uncertainty about a sample."""
        tokenized = self.tokenizer(
            sample["text"],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        
        return self.uncertainty_estimator.estimate(input_ids, attention_mask)
    
    def _make_learning_decision(
        self,
        uncertainty: UncertaintyMetrics,
        force_learn: bool
    ) -> LearningDecision:
        """Decide whether to learn from a sample based on uncertainty."""
        if force_learn:
            return LearningDecision.LEARN
        
        if uncertainty.is_uncertain(self.config.uncertainty_threshold):
            return LearningDecision.LEARN
        else:
            return LearningDecision.SKIP_CONFIDENT
    
    def _learn_batch(
        self,
        samples_with_uncertainty: List[Tuple[Dict, UncertaintyMetrics]]
    ) -> float:
        """Perform incremental learning on a batch of samples."""
        self.model.train()
        optimizer = self._get_optimizer()
        
        new_samples = [s for s, _ in samples_with_uncertainty]
        total_loss = 0.0
        num_steps = 0
        
        for step in range(self.config.max_steps_per_event):
            optimizer.zero_grad()
            
            # Create mixed batch with replay
            mixed_batch, num_new, num_replay = self.replay_buffer.create_mixed_batch(
                new_samples,
                self.config.batch_size
            )
            
            if not mixed_batch:
                break
            
            # Forward pass
            loss, student_outputs = self._compute_loss(mixed_batch)
            
            # Add regularization penalty
            if student_outputs:
                sample_batch = mixed_batch[0]
                tokenized = self.tokenizer(
                    sample_batch["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                reg_penalty = self.regularizer.penalty(
                    input_ids=tokenized["input_ids"].to(self.device),
                    attention_mask=tokenized["attention_mask"].to(self.device),
                    student_outputs=student_outputs
                )
                loss = loss + reg_penalty
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip_norm
            )
            optimizer.step()
            
            # Update SI if using
            self.regularizer.update_step()
            
            total_loss += loss.item()
            num_steps += 1
        
        return total_loss / max(num_steps, 1)
    
    def _compute_loss(self, batch: List[Dict]) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Compute training loss for a batch."""
        texts = [s["text"] for s in batch]
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids, attention_mask, spans=None, y_ids=None)
        
        # Compute losses
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Entity classification loss (if labels available)
        if any("entity_type_labels" in s for s in batch):
            entity_logits = outputs["entity_logits"]
            # Simplified: use first token predictions
            # In practice, would use proper span alignment
            
        # Concept loss (if labels available)
        if any("concepts" in s for s in batch):
            concept_probs = outputs["concept_probs"]
            # Simplified concept loss
        
        # Use a simple self-supervised objective
        # Encourage consistent predictions
        entity_probs = F.softmax(outputs["entity_logits"], dim=-1)
        consistency_loss = -torch.mean(entity_probs * torch.log(entity_probs + 1e-10))
        total_loss = total_loss + 0.1 * consistency_loss
        
        return total_loss, outputs
    
    def _should_consolidate(self) -> bool:
        """Check if knowledge should be consolidated."""
        events_since = len(self.learning_events) % self.config.consolidate_every_n_events
        enough_samples = self.samples_since_consolidation >= self.config.min_samples_for_consolidation
        
        return events_since == 0 and enough_samples
    
    def _consolidate(self):
        """Consolidate knowledge (update EWC Fisher, snapshot for LwF, etc.)."""
        print("Consolidating knowledge...")
        
        # Create a dataloader from memory for Fisher computation
        memory_samples = self.memory.sample(
            min(100, len(self.memory.memory)),
            strategy="random"
        )
        
        if not memory_samples:
            return
        
        # Simple dataloader
        class SimpleDataset:
            def __init__(self, samples, tokenizer, device):
                self.samples = samples
                self.tokenizer = tokenizer
                self.device = device
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                sample = self.samples[idx]
                tokenized = self.tokenizer(
                    sample["text"],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=128,
                    truncation=True
                )
                return {
                    "input_ids": tokenized["input_ids"].squeeze(0),
                    "attention_mask": tokenized["attention_mask"].squeeze(0)
                }
        
        dataset = SimpleDataset(memory_samples, self.tokenizer, self.device)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Move batch to device
        def collate_to_device(batch):
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]).to(self.device),
                "attention_mask": torch.stack([b["attention_mask"] for b in batch]).to(self.device)
            }
        
        # Consolidate regularizers
        self.regularizer.consolidate(dataloader)
        
        self.samples_since_consolidation = 0
        print(f"Knowledge consolidated. EWC Fisher updated.")
    
    def save_state(self, path: str):
        """Save the continuous learner state."""
        state = {
            "config": self.config.__dict__,
            "total_samples_seen": self.total_samples_seen,
            "total_samples_learned": self.total_samples_learned,
            "samples_since_consolidation": self.samples_since_consolidation,
            "learning_events": [
                {
                    "timestamp": e.timestamp,
                    "num_samples": e.num_samples,
                    "num_learned": e.num_learned,
                    "avg_loss": e.avg_loss,
                    "avg_uncertainty": e.avg_uncertainty
                }
                for e in self.learning_events
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save memory separately
        memory_path = path.replace('.json', '_memory.json')
        self.memory.save(memory_path)
    
    def load_state(self, path: str):
        """Load the continuous learner state."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.total_samples_seen = state["total_samples_seen"]
        self.total_samples_learned = state["total_samples_learned"]
        self.samples_since_consolidation = state["samples_since_consolidation"]
        
        # Load memory
        memory_path = path.replace('.json', '_memory.json')
        if Path(memory_path).exists():
            self.memory.load(memory_path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "total_samples_seen": self.total_samples_seen,
            "total_samples_learned": self.total_samples_learned,
            "learning_rate": self.total_samples_learned / max(self.total_samples_seen, 1),
            "num_learning_events": len(self.learning_events),
            "memory_stats": self.memory.get_statistics(),
            "safety_stats": self.safety_regulator.get_statistics() if self.safety_regulator else {},
            "symbolic_stats": self.symbolic_updater.get_statistics(),
            "samples_since_consolidation": self.samples_since_consolidation
        }
    
    def add_safety_blocklist(self, terms: List[str]):
        """Add terms to the safety blocklist."""
        if self.safety_regulator:
            self.safety_regulator.add_blocklist_terms(terms)
    
    def add_safety_allowlist(self, terms: List[str]):
        """Add terms to the safety allowlist."""
        if self.safety_regulator:
            self.safety_regulator.add_allowlist_terms(terms)
    
    def add_harmful_example(
        self,
        text: str,
        category: ContentCategory,
        embedding: Optional[torch.Tensor] = None
    ):
        """Add a harmful example for semantic safety filtering."""
        if self.safety_regulator:
            self.safety_regulator.add_harmful_example(text, category, embedding)


class OnlineLearningLoop:
    """
    High-level online learning loop for production use.
    
    Handles data streaming, batching, and periodic evaluation.
    """
    
    def __init__(
        self,
        learner: ContinuousLearner,
        eval_fn: Optional[Callable] = None,
        eval_every_n_samples: int = 100
    ):
        self.learner = learner
        self.eval_fn = eval_fn
        self.eval_every_n_samples = eval_every_n_samples
        
        self.samples_since_eval = 0
        self.eval_results: List[Dict] = []
    
    def process_sample(self, sample: Dict) -> Dict[str, Any]:
        """Process a single sample."""
        result = self.learner.process_stream([sample])
        
        self.samples_since_eval += 1
        
        if self.eval_fn and self.samples_since_eval >= self.eval_every_n_samples:
            eval_result = self.eval_fn(self.learner.model)
            self.eval_results.append({
                "timestamp": datetime.now().isoformat(),
                "samples_seen": self.learner.total_samples_seen,
                **eval_result
            })
            self.samples_since_eval = 0
        
        return result
    
    def process_batch(self, samples: List[Dict]) -> Dict[str, Any]:
        """Process a batch of samples."""
        result = self.learner.process_stream(samples)
        
        self.samples_since_eval += len(samples)
        
        if self.eval_fn and self.samples_since_eval >= self.eval_every_n_samples:
            eval_result = self.eval_fn(self.learner.model)
            self.eval_results.append({
                "timestamp": datetime.now().isoformat(),
                "samples_seen": self.learner.total_samples_seen,
                **eval_result
            })
            self.samples_since_eval = 0
        
        return result
