"""
Symbolic component updaters for the neurosymbolic model.

Provides mechanisms to update symbolic components without full retraining:
- Concept bank expansion
- Soft logic rule updates
- Knowledge graph integration
- Entity type discovery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
import numpy as np
from dataclasses import dataclass


@dataclass
class ConceptUpdate:
    """Represents an update to the concept bank."""
    concept_name: str
    embedding: torch.Tensor
    source: str = "learned"
    confidence: float = 1.0
    related_concepts: List[str] = None
    
    def __post_init__(self):
        if self.related_concepts is None:
            self.related_concepts = []


@dataclass
class RuleUpdate:
    """Represents an update to soft logic rules."""
    concept_a: str
    concept_b: str
    relation: str
    weight: float = 1.0
    polarity: int = 1  # 1 = encourage, -1 = discourage
    frequency: int = 1
    confidence: float = 1.0


class ConceptBankUpdater:
    """
    Manages updates to the concept bank.
    
    Supports:
    - Adding new concepts
    - Updating existing concept embeddings
    - Concept merging and pruning
    - Hierarchical concept organization
    """
    
    def __init__(
        self,
        model: nn.Module,
        similarity_threshold: float = 0.9,
        max_concepts: int = 2048
    ):
        """
        Args:
            model: The neurosymbolic model
            similarity_threshold: Threshold for concept merging
            max_concepts: Maximum number of concepts
        """
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.max_concepts = max_concepts
        
        # Track concept metadata
        self.concept_metadata: Dict[int, Dict] = {}
        self.concept_names: Dict[str, int] = {}
        self.concept_hierarchy: Dict[str, List[str]] = defaultdict(list)
    
    def add_concept(
        self,
        name: str,
        embedding: Optional[torch.Tensor] = None,
        parent_concept: Optional[str] = None,
        source: str = "manual"
    ) -> int:
        """
        Add a new concept to the concept bank.
        
        Args:
            name: Name of the concept
            embedding: Optional initial embedding
            parent_concept: Optional parent for hierarchical organization
            source: Source of the concept
        
        Returns:
            Index of the new concept
        """
        if name in self.concept_names:
            return self.concept_names[name]
        
        concept_bank = self.model.concept_bank
        current_size = concept_bank.embeddings.shape[0]
        
        if current_size >= self.max_concepts:
            # Try to merge similar concepts first
            self._merge_similar_concepts()
            current_size = concept_bank.embeddings.shape[0]
            
            if current_size >= self.max_concepts:
                raise ValueError("Concept bank full and cannot merge")
        
        # Initialize embedding
        if embedding is None:
            embedding = torch.randn(concept_bank.embeddings.shape[1])
            embedding = F.normalize(embedding, dim=0)
        
        embedding = embedding.to(concept_bank.embeddings.device)
        
        # Expand concept bank
        new_embeddings = torch.cat([
            concept_bank.embeddings.data,
            embedding.unsqueeze(0)
        ], dim=0)
        
        concept_bank.embeddings = nn.Parameter(new_embeddings)
        
        # Update metadata
        new_idx = current_size
        self.concept_names[name] = new_idx
        self.concept_metadata[new_idx] = {
            "name": name,
            "source": source,
            "created_at": None,  # Would use datetime in practice
            "update_count": 0
        }
        
        # Update hierarchy
        if parent_concept:
            self.concept_hierarchy[parent_concept].append(name)
        
        # Update concept head if needed
        if hasattr(self.model, 'concept_head'):
            self._expand_concept_head(new_idx + 1)
        
        return new_idx
    
    def _expand_concept_head(self, new_size: int):
        """Expand the concept head to accommodate new concepts."""
        concept_head = self.model.concept_head
        old_size = concept_head.out_features
        
        if new_size <= old_size:
            return
        
        # Create new larger head
        new_head = nn.Linear(
            concept_head.in_features,
            new_size,
            bias=concept_head.bias is not None
        ).to(concept_head.weight.device)
        
        # Copy old weights
        with torch.no_grad():
            new_head.weight[:old_size] = concept_head.weight
            if concept_head.bias is not None:
                new_head.bias[:old_size] = concept_head.bias
            
            # Initialize new weights
            nn.init.xavier_uniform_(new_head.weight[old_size:])
            if new_head.bias is not None:
                new_head.bias[old_size:] = 0
        
        self.model.concept_head = new_head
    
    def update_concept_embedding(
        self,
        name: str,
        new_embedding: torch.Tensor,
        learning_rate: float = 0.1
    ):
        """Update an existing concept's embedding with exponential moving average."""
        if name not in self.concept_names:
            return
        
        idx = self.concept_names[name]
        concept_bank = self.model.concept_bank
        
        new_embedding = new_embedding.to(concept_bank.embeddings.device)
        
        with torch.no_grad():
            # EMA update
            concept_bank.embeddings.data[idx] = (
                (1 - learning_rate) * concept_bank.embeddings.data[idx] +
                learning_rate * new_embedding
            )
            # Normalize
            concept_bank.embeddings.data[idx] = F.normalize(
                concept_bank.embeddings.data[idx], dim=0
            )
        
        self.concept_metadata[idx]["update_count"] += 1
    
    def _merge_similar_concepts(self):
        """Merge concepts that are too similar."""
        concept_bank = self.model.concept_bank
        embeddings = concept_bank.embeddings.data
        
        # Compute pairwise similarities
        similarities = F.cosine_similarity(
            embeddings.unsqueeze(0),
            embeddings.unsqueeze(1),
            dim=2
        )
        
        # Find pairs to merge
        to_merge = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                if similarities[i, j] > self.similarity_threshold:
                    to_merge.append((i, j))
        
        # Merge (keep first, remove second)
        indices_to_remove = set()
        for i, j in to_merge:
            if j not in indices_to_remove:
                indices_to_remove.add(j)
                # Merge embedding (average)
                with torch.no_grad():
                    embeddings[i] = F.normalize(
                        (embeddings[i] + embeddings[j]) / 2, dim=0
                    )
        
        if indices_to_remove:
            # Remove merged concepts
            keep_mask = torch.ones(len(embeddings), dtype=torch.bool)
            for idx in indices_to_remove:
                keep_mask[idx] = False
            
            concept_bank.embeddings = nn.Parameter(embeddings[keep_mask])
    
    def get_concept_similarities(self, name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get most similar concepts to a given concept."""
        if name not in self.concept_names:
            return []
        
        idx = self.concept_names[name]
        concept_bank = self.model.concept_bank
        
        similarities = F.cosine_similarity(
            concept_bank.embeddings[idx].unsqueeze(0),
            concept_bank.embeddings,
            dim=1
        )
        
        # Get top-k (excluding self)
        values, indices = torch.topk(similarities, min(top_k + 1, len(similarities)))
        
        results = []
        for val, idx in zip(values.tolist(), indices.tolist()):
            for concept_name, concept_idx in self.concept_names.items():
                if concept_idx == idx and concept_name != name:
                    results.append((concept_name, val))
                    break
        
        return results[:top_k]


class SoftLogicRuleUpdater:
    """
    Manages updates to soft logic rules.
    
    Supports:
    - Adding new rules from observed patterns
    - Updating rule weights based on evidence
    - Rule pruning and consolidation
    - Conflict detection
    """
    
    def __init__(
        self,
        model: nn.Module,
        min_frequency: int = 3,
        min_confidence: float = 0.5,
        max_rules: int = 200
    ):
        """
        Args:
            model: The neurosymbolic model
            min_frequency: Minimum observations before adding rule
            min_confidence: Minimum confidence for rule
            max_rules: Maximum number of rules
        """
        self.model = model
        self.min_frequency = min_frequency
        self.min_confidence = min_confidence
        self.max_rules = max_rules
        
        # Track observed patterns
        self.pattern_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self.pattern_positive: Dict[Tuple[str, str, str], int] = defaultdict(int)
        
        # Active rules
        self.active_rules: List[RuleUpdate] = []
    
    def observe_pattern(
        self,
        concept_a: str,
        concept_b: str,
        relation: str,
        positive: bool = True
    ):
        """
        Record an observation of a concept-relation pattern.
        
        Args:
            concept_a: First concept
            concept_b: Second concept
            relation: Relation between concepts
            positive: Whether this is a positive example
        """
        key = (concept_a, concept_b, relation)
        self.pattern_counts[key] += 1
        
        if positive:
            self.pattern_positive[key] += 1
        
        # Check if we should add/update a rule
        self._check_and_update_rule(key)
    
    def _check_and_update_rule(self, pattern_key: Tuple[str, str, str]):
        """Check if pattern should become a rule."""
        concept_a, concept_b, relation = pattern_key
        
        count = self.pattern_counts[pattern_key]
        positive = self.pattern_positive[pattern_key]
        
        if count < self.min_frequency:
            return
        
        confidence = positive / count
        
        if confidence < self.min_confidence and (1 - confidence) < self.min_confidence:
            return  # Not confident enough either way
        
        # Determine polarity
        polarity = 1 if confidence >= 0.5 else -1
        effective_confidence = confidence if polarity == 1 else (1 - confidence)
        
        # Create or update rule
        rule = RuleUpdate(
            concept_a=concept_a,
            concept_b=concept_b,
            relation=relation,
            weight=effective_confidence,
            polarity=polarity,
            frequency=count,
            confidence=effective_confidence
        )
        
        self._add_or_update_rule(rule)
    
    def _add_or_update_rule(self, rule: RuleUpdate):
        """Add a new rule or update existing one."""
        # Check for existing rule
        for i, existing in enumerate(self.active_rules):
            if (existing.concept_a == rule.concept_a and
                existing.concept_b == rule.concept_b and
                existing.relation == rule.relation):
                
                # Update existing rule
                self.active_rules[i] = rule
                self._sync_to_model()
                return
        
        # Add new rule
        if len(self.active_rules) >= self.max_rules:
            self._prune_rules()
        
        self.active_rules.append(rule)
        self._sync_to_model()
    
    def _prune_rules(self):
        """Remove lowest-confidence rules."""
        # Sort by confidence and keep top rules
        self.active_rules.sort(key=lambda r: r.confidence, reverse=True)
        self.active_rules = self.active_rules[:self.max_rules - 10]  # Leave room
    
    def _sync_to_model(self):
        """Sync rules to the model's soft logic component."""
        if not hasattr(self.model, 'softlogic'):
            return
        
        # Convert rules to model format
        rules_dict = []
        for rule in self.active_rules:
            rules_dict.append({
                "concept_a": rule.concept_a,
                "concept_b": rule.concept_b,
                "relation": rule.relation,
                "weight": rule.weight,
                "polarity": rule.polarity
            })
        
        # Update model
        self.model.softlogic.rules = rules_dict
    
    def detect_conflicts(self) -> List[Tuple[RuleUpdate, RuleUpdate]]:
        """Detect conflicting rules."""
        conflicts = []
        
        for i, rule1 in enumerate(self.active_rules):
            for rule2 in self.active_rules[i+1:]:
                # Same concepts and relation but different polarity
                if (rule1.concept_a == rule2.concept_a and
                    rule1.concept_b == rule2.concept_b and
                    rule1.relation == rule2.relation and
                    rule1.polarity != rule2.polarity):
                    conflicts.append((rule1, rule2))
        
        return conflicts
    
    def get_rules_for_concepts(self, concepts: List[str]) -> List[RuleUpdate]:
        """Get all rules involving given concepts."""
        relevant = []
        concepts_set = set(concepts)
        
        for rule in self.active_rules:
            if rule.concept_a in concepts_set or rule.concept_b in concepts_set:
                relevant.append(rule)
        
        return relevant


class KnowledgeGraphUpdater:
    """
    Manages updates to the knowledge graph.
    
    Supports:
    - Adding new entities and relations
    - Updating entity embeddings
    - Path caching for reasoning
    - Fact verification
    """
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: The neurosymbolic model with KG components
        """
        self.model = model
        
        # Local cache of KG updates
        self.pending_entities: List[Dict] = []
        self.pending_relations: List[Tuple[str, str, str]] = []
        self.entity_update_counts: Dict[str, int] = defaultdict(int)
    
    def add_entity(
        self,
        entity_text: str,
        entity_type: Optional[str] = None,
        embedding: Optional[torch.Tensor] = None
    ):
        """Add a new entity to the knowledge graph."""
        self.pending_entities.append({
            "text": entity_text,
            "type": entity_type,
            "embedding": embedding
        })
    
    def add_relation(
        self,
        subject: str,
        relation: str,
        obj: str,
        confidence: float = 1.0
    ):
        """Add a new relation to the knowledge graph."""
        self.pending_relations.append((subject, relation, obj))
        
        # Also update the KG graph if available
        if hasattr(self.model, 'kg_graph') and self.model.kg_graph is not None:
            self.model.kg_graph.add_edge(subject, relation, obj)
    
    def commit_updates(self):
        """Commit pending updates to the model."""
        # Update entity mappings
        if hasattr(self.model, 'entity_mapping'):
            for entity in self.pending_entities:
                # Add to mapping (implementation depends on KG format)
                pass
        
        self.pending_entities = []
        self.pending_relations = []
    
    def query_paths(
        self,
        source: str,
        target: str,
        max_length: int = 3
    ) -> List[List[Tuple[str, str]]]:
        """Query paths between entities in the KG."""
        if not hasattr(self.model, 'kg_graph') or self.model.kg_graph is None:
            return []
        
        return self.model.kg_graph.find_paths(source, target, max_length)
    
    def verify_fact(
        self,
        subject: str,
        relation: str,
        obj: str
    ) -> Tuple[bool, float]:
        """
        Verify if a fact exists in the KG.
        
        Returns:
            (exists, confidence)
        """
        if not hasattr(self.model, 'kg_graph') or self.model.kg_graph is None:
            return False, 0.0
        
        # Check direct edge
        neighbors = self.model.kg_graph.get_neighbors(subject)
        for rel, target in neighbors:
            if rel == relation and target == obj:
                return True, 1.0
        
        # Check reverse
        neighbors = self.model.kg_graph.get_neighbors(obj)
        for rel, target in neighbors:
            if f"~{rel}" == relation and target == subject:
                return True, 0.9
        
        # Check paths
        paths = self.query_paths(subject, obj, max_length=2)
        if paths:
            return True, 0.5  # Indirect evidence
        
        return False, 0.0


class SymbolicUpdateManager:
    """
    Unified manager for all symbolic updates.
    
    Coordinates updates across concept bank, rules, and KG.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        
        self.concept_updater = ConceptBankUpdater(model)
        self.rule_updater = SoftLogicRuleUpdater(model)
        self.kg_updater = KnowledgeGraphUpdater(model)
    
    def process_sample(self, sample: Dict):
        """
        Extract and apply symbolic updates from a sample.
        
        Args:
            sample: Sample dictionary with entities, concepts, relations
        """
        entities = sample.get("entities", [])
        concepts = sample.get("concepts", [])
        relations = sample.get("relations", [])
        
        # Update concepts
        for i, entity_concepts in enumerate(concepts):
            if isinstance(entity_concepts, str):
                entity_concepts = [entity_concepts]
            
            for concept in entity_concepts:
                if concept not in self.concept_updater.concept_names:
                    try:
                        self.concept_updater.add_concept(concept, source="observed")
                    except ValueError:
                        pass  # Concept bank full
        
        # Update rules from relations
        for head_idx, tail_idx, relation in relations:
            if head_idx < len(concepts) and tail_idx < len(concepts):
                head_concepts = concepts[head_idx]
                tail_concepts = concepts[tail_idx]
                
                if isinstance(head_concepts, str):
                    head_concepts = [head_concepts]
                if isinstance(tail_concepts, str):
                    tail_concepts = [tail_concepts]
                
                for hc in head_concepts:
                    for tc in tail_concepts:
                        self.rule_updater.observe_pattern(hc, tc, relation, positive=True)
        
        # Update KG
        for head_idx, tail_idx, relation in relations:
            if head_idx < len(entities) and tail_idx < len(entities):
                self.kg_updater.add_relation(
                    entities[head_idx],
                    relation,
                    entities[tail_idx]
                )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about symbolic components."""
        return {
            "num_concepts": len(self.concept_updater.concept_names),
            "num_rules": len(self.rule_updater.active_rules),
            "pending_kg_entities": len(self.kg_updater.pending_entities),
            "pending_kg_relations": len(self.kg_updater.pending_relations)
        }
