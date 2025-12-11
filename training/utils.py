"""Training utilities for extraction and configuration."""

from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import torch

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class CheckpointManager:
    """Manage model checkpoints during training."""
    
    def __init__(
        self,
        save_dir: str = "checkpoints",
        max_checkpoints: int = 3,
        save_best_only: bool = False
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.best_score = None
        self.checkpoints: List[Path] = []
    
    def save(
        self,
        model,
        optimizer,
        epoch: int,
        stage: str,
        score: float,
        config: Optional[Dict] = None
    ) -> Optional[Path]:
        """Save a checkpoint.
        
        Args:
            model: The model to save
            optimizer: The optimizer (can be None)
            epoch: Current epoch number
            stage: Training stage name
            score: Current score/loss value
            config: Optional config dict to save
        
        Returns:
            Path to saved checkpoint or None if not saved
        """
        if self.save_best_only:
            if self.best_score is not None and score >= self.best_score:
                return None
            self.best_score = score
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{stage}_epoch{epoch}_{timestamp}.pt"
        filepath = self.save_dir / filename
        
        checkpoint = {
            "epoch": epoch,
            "stage": stage,
            "model_state_dict": model.state_dict(),
            "score": score,
            "config": config
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        self.checkpoints.append(filepath)
        
        # Remove old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        print(f"Saved checkpoint: {filepath}")
        return filepath
    
    def load_latest(self, model, optimizer=None) -> Optional[Dict]:
        """Load the most recent checkpoint."""
        if not self.checkpoints:
            existing = sorted(self.save_dir.glob("checkpoint_*.pt"))
            if existing:
                self.checkpoints = existing
        
        if not self.checkpoints:
            return None
        
        latest = self.checkpoints[-1]
        return self.load(latest, model, optimizer)
    
    def load(self, filepath: Path, model, optimizer=None) -> Dict:
        """Load a specific checkpoint."""
        checkpoint = torch.load(filepath, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        print(f"Loaded checkpoint: {filepath}")
        return checkpoint


class TrainingLogger:
    """TensorBoard logging wrapper with fallback."""
    
    def __init__(
        self,
        log_dir: str = "runs",
        experiment_name: Optional[str] = None,
        enable_tensorboard: bool = True
    ):
        self.enabled = enable_tensorboard and TENSORBOARD_AVAILABLE
        self.writer = None
        
        if self.enabled:
            if experiment_name:
                log_path = f"{log_dir}/{experiment_name}"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_path = f"{log_dir}/{timestamp}"
            
            self.writer = SummaryWriter(log_dir=log_path)
            print(f"TensorBoard logging to: {log_path}")
        elif enable_tensorboard:
            print("TensorBoard not available. Install with: pip install tensorboard")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars."""
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram."""
        if self.enabled and self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text."""
        if self.enabled and self.writer:
            self.writer.add_text(tag, text, step)
    
    def close(self):
        """Close the logger."""
        if self.writer:
            self.writer.close()


def extract_concept_to_entity_type_map(
    dataset, 
    n_entity_types: Optional[int] = None, 
    base_concept_priority: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Dynamically extract concept-to-entity-type mapping from dataset.
    
    Args:
        dataset: ToyCognitiveDataset instance
        n_entity_types: Maximum number of entity types. None = use all found.
        base_concept_priority: Ordered list of base concept names to prioritize.
    
    Returns:
        Mapping from concept names to entity type indices (0-indexed)
    """
    if base_concept_priority is None:
        base_concept_priority = ["animal", "person", "location", "object", "attribute"]
    
    # Extract all concepts
    all_concepts = set()
    concept_frequencies = {}
    
    for sample in dataset:
        concepts = sample.get("concepts", [])
        for concept_list in concepts:
            if isinstance(concept_list, str):
                concept_list = [concept_list]
            elif not isinstance(concept_list, list):
                concept_list = [str(concept_list)]
            
            for concept in concept_list:
                all_concepts.add(concept)
                concept_frequencies[concept] = concept_frequencies.get(concept, 0) + 1
    
    # Build mapping
    concept_to_entity_type_map = {}
    entity_type_idx = 0
    
    # Priority concepts first
    for base_concept in base_concept_priority:
        if base_concept in all_concepts:
            if n_entity_types is None or entity_type_idx < n_entity_types:
                concept_to_entity_type_map[base_concept] = entity_type_idx
                entity_type_idx += 1
    
    # Add remaining by frequency
    remaining_concepts = sorted(
        [(c, f) for c, f in concept_frequencies.items() if c not in concept_to_entity_type_map],
        key=lambda x: x[1],
        reverse=True
    )
    
    for concept, freq in remaining_concepts:
        if n_entity_types is None or entity_type_idx < n_entity_types:
            concept_to_entity_type_map[concept] = entity_type_idx
            entity_type_idx += 1
        else:
            break
    
    if len(concept_to_entity_type_map) == 0:
        print("Warning: No concepts found in dataset. Using empty mapping.")
    
    return concept_to_entity_type_map


def extract_relation_map_from_dataset(dataset) -> Dict[str, int]:
    """
    Dynamically extract relation map from dataset.
    
    Args:
        dataset: ToyCognitiveDataset instance
    
    Returns:
        Mapping from relation names to indices (1-indexed, 0 is padding)
    """
    all_relations = set()
    
    for sample in dataset:
        relations = sample.get("relations", [])
        for rel in relations:
            if len(rel) >= 3:
                rel_name = rel[2]
                all_relations.add(rel_name)
    
    return {rel: idx + 1 for idx, rel in enumerate(sorted(all_relations))}


def generate_soft_logic_rules_from_dataset(
    dataset, 
    concept_to_entity_type_map: Dict[str, int], 
    relation_map: Dict[str, int],
    min_frequency: int = 2, 
    min_confidence: float = 0.3, 
    max_rules: int = 50, 
    include_negative_rules: bool = True
) -> List[Dict]:
    """
    Dynamically generate soft logic rules from dataset patterns.
    
    Args:
        dataset: ToyCognitiveDataset instance
        concept_to_entity_type_map: Mapping from concepts to entity types
        relation_map: Mapping from relations to indices
        min_frequency: Minimum pattern occurrences for positive rules
        min_confidence: Minimum confidence threshold
        max_rules: Maximum number of rules to generate
        include_negative_rules: Whether to generate negative rules
    
    Returns:
        List of rule dicts with concept_a, concept_b, relation, weight, polarity
    """
    # Count patterns
    pattern_counts = {}
    relation_totals = {}
    concept_pair_totals = {}
    
    for sample in dataset:
        entities = sample.get("entities", [])
        concepts = sample.get("concepts", [])
        relations = sample.get("relations", [])
        
        # Normalize concepts
        normalized_concepts = []
        for concept_list in concepts:
            if isinstance(concept_list, str):
                normalized_concepts.append([concept_list])
            elif isinstance(concept_list, list):
                normalized_concepts.append(concept_list)
            else:
                normalized_concepts.append([str(concept_list)])
        
        # Process relations
        for rel in relations:
            if len(rel) >= 3:
                head_idx, tail_idx, rel_name = rel[0], rel[1], rel[2]
                
                if head_idx >= len(normalized_concepts) or tail_idx >= len(normalized_concepts):
                    continue
                
                head_concepts = normalized_concepts[head_idx]
                tail_concepts = normalized_concepts[tail_idx]
                
                for head_concept in head_concepts:
                    for tail_concept in tail_concepts:
                        if head_concept in concept_to_entity_type_map and tail_concept in concept_to_entity_type_map:
                            pattern = (head_concept, tail_concept, rel_name)
                            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                            relation_totals[rel_name] = relation_totals.get(rel_name, 0) + 1
                            concept_pair = (head_concept, tail_concept)
                            concept_pair_totals[concept_pair] = concept_pair_totals.get(concept_pair, 0) + 1
    
    # Generate positive rules
    rules = []
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    
    for (concept_a, concept_b, relation), count in sorted_patterns:
        if len(rules) >= max_rules:
            break
        
        if relation not in relation_map:
            continue
        
        rel_total = relation_totals.get(relation, 1)
        confidence = count / rel_total if rel_total > 0 else 0
        
        if count >= min_frequency and confidence >= min_confidence:
            weight = min(1.0, count / max(1, min_frequency * 2)) * confidence
            
            rules.append({
                "concept_a": concept_a,
                "concept_b": concept_b,
                "relation": relation,
                "weight": float(weight),
                "polarity": 1
            })
    
    # Generate negative rules
    if include_negative_rules and len(rules) < max_rules:
        all_concept_pairs = set(concept_pair_totals.keys())
        all_relations = set(relation_map.keys())
        
        for concept_a, concept_b in list(all_concept_pairs)[:20]:
            if len(rules) >= max_rules:
                break
            
            pair_total = concept_pair_totals.get((concept_a, concept_b), 0)
            if pair_total < 2:
                continue
            
            for relation in all_relations:
                if len(rules) >= max_rules:
                    break
                
                pattern = (concept_a, concept_b, relation)
                count = pattern_counts.get(pattern, 0)
                
                if count == 0 or (count == 1 and pair_total >= 3):
                    has_other_relations = any(
                        pattern_counts.get((concept_a, concept_b, r), 0) > 0
                        for r in all_relations if r != relation
                    )
                    
                    if has_other_relations:
                        rules.append({
                            "concept_a": concept_a,
                            "concept_b": concept_b,
                            "relation": relation,
                            "weight": 0.3,
                            "polarity": -1
                        })
    
    return rules


def configure_soft_logic_rules(
    model, 
    concept_to_entity_type_map: Dict[str, int], 
    relation_map: Dict[str, int], 
    rules_config: List[Dict]
):
    """
    Configure soft logic rules for the model.
    
    Args:
        model: NeuroSymbolicLM model
        concept_to_entity_type_map: Mapping from concepts to entity types
        relation_map: Mapping from relations to indices
        rules_config: List of rule dicts with concept_a, concept_b, relation, weight, polarity
    """
    for rule in rules_config:
        etype_a = concept_to_entity_type_map.get(rule["concept_a"])
        etype_b = concept_to_entity_type_map.get(rule["concept_b"])
        rel_idx_1indexed = relation_map.get(rule["relation"])
        
        if etype_a is None or etype_b is None or rel_idx_1indexed is None:
            print(f"Warning: Skipping rule {rule} - invalid concept or relation mapping")
            continue
        
        rel_idx = rel_idx_1indexed - 1  # Convert to 0-indexed
        
        if rel_idx < 0 or rel_idx >= model.softlogic.n_relations:
            print(f"Warning: Skipping rule {rule} - relation index {rel_idx} out of bounds")
            continue
        
        weight = rule.get("weight", 1.0)
        polarity = rule.get("polarity", 1)
        
        model.softlogic.add_rule(etype_a, etype_b, rel_idx, weight, polarity)
    
    print(f"Configured {len(model.softlogic.rules)} soft logic rules")
