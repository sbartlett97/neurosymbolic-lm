"""
Episodic memory and experience replay for continual learning.

Implements various memory management strategies:
- Reservoir sampling
- Uncertainty-based selection
- Diversity-based selection (using embeddings)
- Class-balanced sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import random
import numpy as np
from collections import defaultdict
import json
from pathlib import Path
from datetime import datetime


@dataclass
class MemoryEntry:
    """A single entry in episodic memory."""
    sample: Dict[str, Any]
    uncertainty: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    embedding: Optional[torch.Tensor] = None
    concepts: List[str] = field(default_factory=list)
    importance: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            "sample": self.sample,
            "uncertainty": self.uncertainty,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "concepts": self.concepts,
            "importance": self.importance
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryEntry':
        return cls(
            sample=data["sample"],
            uncertainty=data.get("uncertainty", 0.0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            access_count=data.get("access_count", 0),
            concepts=data.get("concepts", []),
            importance=data.get("importance", 1.0)
        )


class EpisodicMemory:
    """
    Episodic memory buffer for experience replay.
    
    Supports multiple selection strategies for managing limited memory.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        selection_strategy: str = "reservoir",
        diversity_weight: float = 0.3
    ):
        """
        Args:
            max_size: Maximum number of entries to store
            selection_strategy: How to select entries to keep
                - 'reservoir': Random reservoir sampling
                - 'uncertainty': Keep high-uncertainty samples
                - 'diversity': Maximize embedding diversity
                - 'balanced': Balance across concepts
                - 'hybrid': Combine uncertainty and diversity
            diversity_weight: Weight for diversity in hybrid mode
        """
        self.max_size = max_size
        self.selection_strategy = selection_strategy
        self.diversity_weight = diversity_weight
        
        self.memory: List[MemoryEntry] = []
        self.concept_counts: Dict[str, int] = defaultdict(int)
        self.seen_count = 0
        
        # Statistics
        self.total_added = 0
        self.total_replaced = 0
    
    def add(
        self,
        sample: Dict,
        uncertainty: float = 0.0,
        embedding: Optional[torch.Tensor] = None,
        concepts: Optional[List[str]] = None,
        importance: float = 1.0
    ) -> bool:
        """
        Add a sample to memory.
        
        Args:
            sample: The sample to store
            uncertainty: Uncertainty score for this sample
            embedding: Optional embedding for diversity calculations
            concepts: Concepts present in this sample
            importance: Importance weight
        
        Returns:
            True if sample was added, False if rejected
        """
        self.seen_count += 1
        concepts = concepts or sample.get("concepts", [])
        if isinstance(concepts[0], list) if concepts else False:
            concepts = [c for clist in concepts for c in clist]
        
        entry = MemoryEntry(
            sample=sample,
            uncertainty=uncertainty,
            embedding=embedding.detach().cpu() if embedding is not None else None,
            concepts=concepts,
            importance=importance
        )
        
        if len(self.memory) < self.max_size:
            self._add_entry(entry)
            return True
        
        # Memory full - use selection strategy
        if self.selection_strategy == "reservoir":
            return self._reservoir_add(entry)
        elif self.selection_strategy == "uncertainty":
            return self._uncertainty_add(entry)
        elif self.selection_strategy == "diversity":
            return self._diversity_add(entry)
        elif self.selection_strategy == "balanced":
            return self._balanced_add(entry)
        elif self.selection_strategy == "hybrid":
            return self._hybrid_add(entry)
        else:
            return self._reservoir_add(entry)
    
    def _add_entry(self, entry: MemoryEntry):
        """Add entry and update statistics."""
        self.memory.append(entry)
        for concept in entry.concepts:
            self.concept_counts[concept] += 1
        self.total_added += 1
    
    def _remove_entry(self, idx: int):
        """Remove entry at index and update statistics."""
        entry = self.memory[idx]
        for concept in entry.concepts:
            self.concept_counts[concept] = max(0, self.concept_counts[concept] - 1)
        self.memory.pop(idx)
        self.total_replaced += 1
    
    def _reservoir_add(self, entry: MemoryEntry) -> bool:
        """Reservoir sampling: equal probability of keeping any sample."""
        idx = random.randint(0, self.seen_count - 1)
        if idx < self.max_size:
            self._remove_entry(idx)
            self._add_entry(entry)
            return True
        return False
    
    def _uncertainty_add(self, entry: MemoryEntry) -> bool:
        """Keep samples with highest uncertainty."""
        min_idx = min(
            range(len(self.memory)),
            key=lambda i: self.memory[i].uncertainty
        )
        
        if entry.uncertainty > self.memory[min_idx].uncertainty:
            self._remove_entry(min_idx)
            self._add_entry(entry)
            return True
        return False
    
    def _diversity_add(self, entry: MemoryEntry) -> bool:
        """Maximize embedding diversity using farthest point sampling."""
        if entry.embedding is None:
            return self._reservoir_add(entry)
        
        # Find most similar existing entry
        min_dist = float('inf')
        most_similar_idx = 0
        
        for i, mem_entry in enumerate(self.memory):
            if mem_entry.embedding is not None:
                dist = torch.norm(entry.embedding - mem_entry.embedding).item()
                if dist < min_dist:
                    min_dist = dist
                    most_similar_idx = i
        
        # Replace if new entry is more different from its nearest neighbor
        # than the most similar pair in memory
        threshold = self._compute_diversity_threshold()
        if min_dist > threshold:
            self._remove_entry(most_similar_idx)
            self._add_entry(entry)
            return True
        return False
    
    def _balanced_add(self, entry: MemoryEntry) -> bool:
        """Balance samples across concepts."""
        if not entry.concepts:
            return self._reservoir_add(entry)
        
        # Find overrepresented concept
        max_concept = max(self.concept_counts.keys(), key=lambda c: self.concept_counts[c], default=None)
        
        if max_concept and self.concept_counts[max_concept] > self.max_size // len(self.concept_counts):
            # Remove sample from overrepresented concept
            for i, mem_entry in enumerate(self.memory):
                if max_concept in mem_entry.concepts:
                    self._remove_entry(i)
                    self._add_entry(entry)
                    return True
        
        return self._reservoir_add(entry)
    
    def _hybrid_add(self, entry: MemoryEntry) -> bool:
        """Combine uncertainty and diversity scores."""
        # Compute combined score for each memory entry
        scores = []
        for i, mem_entry in enumerate(self.memory):
            uncertainty_score = mem_entry.uncertainty
            
            diversity_score = 0.0
            if entry.embedding is not None and mem_entry.embedding is not None:
                diversity_score = torch.norm(entry.embedding - mem_entry.embedding).item()
            
            combined = (1 - self.diversity_weight) * uncertainty_score + self.diversity_weight * diversity_score
            scores.append((i, combined))
        
        # Replace entry with lowest combined score if new entry is better
        min_idx, min_score = min(scores, key=lambda x: x[1])
        
        new_score = entry.uncertainty
        if new_score > min_score:
            self._remove_entry(min_idx)
            self._add_entry(entry)
            return True
        return False
    
    def _compute_diversity_threshold(self) -> float:
        """Compute adaptive threshold for diversity-based selection."""
        if len(self.memory) < 2:
            return 0.0
        
        # Sample pairs and compute distances
        distances = []
        sample_size = min(50, len(self.memory))
        indices = random.sample(range(len(self.memory)), sample_size)
        
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if self.memory[indices[i]].embedding is not None and self.memory[indices[j]].embedding is not None:
                    dist = torch.norm(
                        self.memory[indices[i]].embedding - self.memory[indices[j]].embedding
                    ).item()
                    distances.append(dist)
        
        if distances:
            return np.percentile(distances, 25)  # 25th percentile as threshold
        return 0.0
    
    def sample(
        self,
        batch_size: int,
        strategy: str = "random"
    ) -> List[Dict]:
        """
        Sample from memory for replay.
        
        Args:
            batch_size: Number of samples to return
            strategy: Sampling strategy
                - 'random': Uniform random
                - 'weighted': Weight by importance
                - 'uncertain': Prioritize uncertain samples
                - 'recent': Prioritize recent samples
        
        Returns:
            List of sample dictionaries
        """
        if len(self.memory) == 0:
            return []
        
        batch_size = min(batch_size, len(self.memory))
        
        if strategy == "random":
            selected = random.sample(self.memory, batch_size)
        elif strategy == "weighted":
            weights = [e.importance for e in self.memory]
            total = sum(weights)
            probs = [w / total for w in weights]
            indices = np.random.choice(len(self.memory), size=batch_size, replace=False, p=probs)
            selected = [self.memory[i] for i in indices]
        elif strategy == "uncertain":
            sorted_memory = sorted(self.memory, key=lambda e: e.uncertainty, reverse=True)
            selected = sorted_memory[:batch_size]
        elif strategy == "recent":
            sorted_memory = sorted(self.memory, key=lambda e: e.timestamp, reverse=True)
            selected = sorted_memory[:batch_size]
        else:
            selected = random.sample(self.memory, batch_size)
        
        # Update access counts
        for entry in selected:
            entry.access_count += 1
        
        return [entry.sample for entry in selected]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "size": len(self.memory),
            "max_size": self.max_size,
            "seen_count": self.seen_count,
            "total_added": self.total_added,
            "total_replaced": self.total_replaced,
            "concept_distribution": dict(self.concept_counts),
            "avg_uncertainty": np.mean([e.uncertainty for e in self.memory]) if self.memory else 0.0,
            "avg_access_count": np.mean([e.access_count for e in self.memory]) if self.memory else 0.0
        }
    
    def save(self, filepath: str):
        """Save memory to file."""
        data = {
            "entries": [e.to_dict() for e in self.memory],
            "config": {
                "max_size": self.max_size,
                "selection_strategy": self.selection_strategy,
                "diversity_weight": self.diversity_weight
            },
            "statistics": {
                "seen_count": self.seen_count,
                "total_added": self.total_added,
                "total_replaced": self.total_replaced
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self, filepath: str):
        """Load memory from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.memory = [MemoryEntry.from_dict(e) for e in data["entries"]]
        self.max_size = data["config"]["max_size"]
        self.selection_strategy = data["config"]["selection_strategy"]
        self.diversity_weight = data["config"]["diversity_weight"]
        self.seen_count = data["statistics"]["seen_count"]
        self.total_added = data["statistics"]["total_added"]
        self.total_replaced = data["statistics"]["total_replaced"]
        
        # Rebuild concept counts
        self.concept_counts = defaultdict(int)
        for entry in self.memory:
            for concept in entry.concepts:
                self.concept_counts[concept] += 1
    
    def clear(self):
        """Clear all memory."""
        self.memory = []
        self.concept_counts = defaultdict(int)
        self.seen_count = 0
        self.total_added = 0
        self.total_replaced = 0


class ReplayBuffer:
    """
    High-level replay buffer for training.
    
    Manages mixing of new samples with replayed memories.
    """
    
    def __init__(
        self,
        memory: EpisodicMemory,
        replay_ratio: float = 0.3,
        replay_strategy: str = "random"
    ):
        """
        Args:
            memory: EpisodicMemory instance
            replay_ratio: Fraction of batch that should be replayed samples
            replay_strategy: How to sample from memory
        """
        self.memory = memory
        self.replay_ratio = replay_ratio
        self.replay_strategy = replay_strategy
    
    def create_mixed_batch(
        self,
        new_samples: List[Dict],
        target_batch_size: int
    ) -> Tuple[List[Dict], int, int]:
        """
        Create a batch mixing new samples and replayed memories.
        
        Args:
            new_samples: New samples to include
            target_batch_size: Target total batch size
        
        Returns:
            (mixed_batch, num_new, num_replayed)
        """
        num_new = min(len(new_samples), target_batch_size)
        num_replay = int(target_batch_size * self.replay_ratio)
        
        # Adjust if not enough new samples
        if num_new < target_batch_size - num_replay:
            num_replay = target_batch_size - num_new
        
        new_batch = random.sample(new_samples, num_new) if len(new_samples) > num_new else new_samples
        replay_batch = self.memory.sample(num_replay, self.replay_strategy)
        
        mixed = new_batch + replay_batch
        random.shuffle(mixed)
        
        return mixed, len(new_batch), len(replay_batch)
