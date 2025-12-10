"""
Production inference module for text-only input.

Provides automatic extraction of:
- Entity spans and types
- Concepts
- Relations between entities

This allows the model to work without pre-annotated data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import re


@dataclass
class ExtractedEntity:
    """An extracted entity from text."""
    text: str
    start_char: int
    end_char: int
    start_token: int
    end_token: int
    entity_type: int
    confidence: float


@dataclass
class ExtractedRelation:
    """An extracted relation between entities."""
    head_idx: int
    tail_idx: int
    relation_type: int
    confidence: float


@dataclass
class ExtractionResult:
    """Complete extraction result from text."""
    text: str
    entities: List[ExtractedEntity]
    concepts: List[List[int]]  # Concept IDs per entity
    concept_confidences: List[List[float]]
    relations: List[ExtractedRelation]
    controller_decision: str  # 'answer', 'abstain', 'ask_clarify'
    controller_confidence: float
    
    def to_sample_dict(self) -> Dict[str, Any]:
        """Convert to sample dictionary format for training."""
        return {
            "text": self.text,
            "entities": [e.text for e in self.entities],
            "entity_type_labels": [e.entity_type for e in self.entities],
            "entity_spans": [(e.start_char, e.end_char) for e in self.entities],
            "entity_token_spans": [(e.start_token, e.end_token) for e in self.entities],
            "entity_confidences": [e.confidence for e in self.entities],
            "concepts": self.concepts,
            "concept_confidences": self.concept_confidences,
            "relations": [(r.head_idx, r.tail_idx, r.relation_type) for r in self.relations],
            "relation_confidences": [r.confidence for r in self.relations],
            "should_respond": 1 if self.controller_decision == "answer" else 0,
        }


class EntityExtractor:
    """
    Extracts entity spans from model predictions.
    
    Uses BIO tagging style: identifies contiguous spans of high-confidence
    entity predictions.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        min_span_length: int = 1,
        max_span_length: int = 10,
        merge_adjacent: bool = True
    ):
        """
        Args:
            confidence_threshold: Minimum confidence to consider a token as entity
            min_span_length: Minimum entity span length in tokens
            max_span_length: Maximum entity span length in tokens
            merge_adjacent: Merge adjacent spans of same type
        """
        self.confidence_threshold = confidence_threshold
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        self.merge_adjacent = merge_adjacent
    
    def extract(
        self,
        entity_logits: torch.Tensor,
        input_ids: torch.Tensor,
        tokenizer,
        attention_mask: Optional[torch.Tensor] = None
    ) -> List[List[ExtractedEntity]]:
        """
        Extract entities from model's entity logits.
        
        Args:
            entity_logits: (B, L, n_entity_types) entity classification logits
            input_ids: (B, L) input token IDs
            tokenizer: Tokenizer for decoding
            attention_mask: Optional (B, L) attention mask
        
        Returns:
            List of entity lists, one per batch item
        """
        B, L, n_types = entity_logits.shape
        
        # Get probabilities and predictions
        probs = F.softmax(entity_logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)  # (B, L)
        
        all_entities = []
        
        for b in range(B):
            batch_entities = []
            
            # Get valid sequence length
            if attention_mask is not None:
                seq_len = attention_mask[b].sum().item()
            else:
                seq_len = L
            
            # Find entity spans (non-zero predictions with high confidence)
            # Type 0 is typically "O" (outside/no entity)
            current_span_start = None
            current_span_type = None
            current_confidences = []
            
            for t in range(int(seq_len)):
                pred_type = predictions[b, t].item()
                conf = confidences[b, t].item()
                
                # Check if this is an entity token
                is_entity = pred_type > 0 and conf >= self.confidence_threshold
                
                if is_entity:
                    if current_span_start is None:
                        # Start new span
                        current_span_start = t
                        current_span_type = pred_type
                        current_confidences = [conf]
                    elif pred_type == current_span_type and self.merge_adjacent:
                        # Continue span
                        current_confidences.append(conf)
                        # Check max length
                        if t - current_span_start + 1 >= self.max_span_length:
                            # End span due to length
                            batch_entities.append(self._create_entity(
                                b, current_span_start, t, current_span_type,
                                current_confidences, input_ids, tokenizer
                            ))
                            current_span_start = None
                            current_span_type = None
                            current_confidences = []
                    else:
                        # Different type - end current and start new
                        if current_span_start is not None:
                            batch_entities.append(self._create_entity(
                                b, current_span_start, t - 1, current_span_type,
                                current_confidences, input_ids, tokenizer
                            ))
                        current_span_start = t
                        current_span_type = pred_type
                        current_confidences = [conf]
                else:
                    # Not an entity - end current span if any
                    if current_span_start is not None:
                        batch_entities.append(self._create_entity(
                            b, current_span_start, t - 1, current_span_type,
                            current_confidences, input_ids, tokenizer
                        ))
                        current_span_start = None
                        current_span_type = None
                        current_confidences = []
            
            # Handle span at end of sequence
            if current_span_start is not None:
                batch_entities.append(self._create_entity(
                    b, current_span_start, int(seq_len) - 1, current_span_type,
                    current_confidences, input_ids, tokenizer
                ))
            
            # Filter by minimum span length
            batch_entities = [
                e for e in batch_entities
                if e.end_token - e.start_token + 1 >= self.min_span_length
            ]
            
            all_entities.append(batch_entities)
        
        return all_entities
    
    def _create_entity(
        self,
        batch_idx: int,
        start_token: int,
        end_token: int,
        entity_type: int,
        confidences: List[float],
        input_ids: torch.Tensor,
        tokenizer
    ) -> ExtractedEntity:
        """Create an ExtractedEntity from token span."""
        # Decode tokens to get text
        token_ids = input_ids[batch_idx, start_token:end_token + 1].tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        
        # Estimate character positions (approximate)
        # In practice, would need offset mapping from tokenizer
        prefix_ids = input_ids[batch_idx, :start_token].tolist()
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
        start_char = len(prefix_text)
        end_char = start_char + len(text)
        
        return ExtractedEntity(
            text=text,
            start_char=start_char,
            end_char=end_char,
            start_token=start_token,
            end_token=end_token,
            entity_type=entity_type,
            confidence=sum(confidences) / len(confidences)
        )


class ConceptExtractor:
    """Extracts concepts for entities from model predictions."""
    
    def __init__(
        self,
        top_k: int = 3,
        confidence_threshold: float = 0.1
    ):
        """
        Args:
            top_k: Number of top concepts per entity
            confidence_threshold: Minimum concept probability
        """
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
    
    def extract(
        self,
        concept_probs: torch.Tensor,
        n_entities: int
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Extract concepts from model's concept probabilities.
        
        Args:
            concept_probs: (B, n_concepts) concept probabilities
            n_entities: Number of entities to assign concepts to
        
        Returns:
            (concepts, confidences) - lists per entity
        """
        # Get top-k concepts
        B = concept_probs.shape[0]
        
        topk_probs, topk_indices = torch.topk(
            concept_probs, min(self.top_k, concept_probs.shape[-1]), dim=-1
        )
        
        # For each entity, assign concepts based on overall concept distribution
        # In a more sophisticated version, would compute per-entity concepts
        all_concepts = []
        all_confidences = []
        
        for b in range(B):
            # Filter by threshold
            mask = topk_probs[b] >= self.confidence_threshold
            concepts = topk_indices[b][mask].tolist()
            confidences = topk_probs[b][mask].tolist()
            
            if not concepts:
                # Fall back to top concept
                concepts = [topk_indices[b, 0].item()]
                confidences = [topk_probs[b, 0].item()]
            
            # Assign same concepts to all entities (simplified)
            # Could be improved with entity-specific concept mapping
            for _ in range(n_entities):
                all_concepts.append(concepts)
                all_confidences.append(confidences)
        
        return all_concepts, all_confidences


class RelationExtractor:
    """Extracts relations between entities from model predictions."""
    
    def __init__(
        self,
        confidence_threshold: float = 0.3,
        max_relations_per_pair: int = 1
    ):
        """
        Args:
            confidence_threshold: Minimum relation probability
            max_relations_per_pair: Maximum relations to extract per entity pair
        """
        self.confidence_threshold = confidence_threshold
        self.max_relations_per_pair = max_relations_per_pair
    
    def extract(
        self,
        rel_logits_matrix: torch.Tensor,
        n_entities: int
    ) -> List[List[ExtractedRelation]]:
        """
        Extract relations from model's relation logits.
        
        Args:
            rel_logits_matrix: (B, N, N, n_relations) relation logits
            n_entities: Number of actual entities (may be < N due to padding)
        
        Returns:
            List of relation lists per batch item
        """
        B, N, _, n_rel = rel_logits_matrix.shape
        
        # Get probabilities
        probs = F.softmax(rel_logits_matrix, dim=-1)
        
        all_relations = []
        
        for b in range(B):
            batch_relations = []
            
            # Only consider actual entities
            for i in range(min(n_entities, N)):
                for j in range(i + 1, min(n_entities, N)):
                    # Get top relation for this pair
                    pair_probs = probs[b, i, j]
                    
                    # Skip if relation type 0 (no relation) has highest prob
                    max_prob, max_rel = pair_probs.max(dim=-1)
                    
                    if max_rel.item() > 0 and max_prob.item() >= self.confidence_threshold:
                        batch_relations.append(ExtractedRelation(
                            head_idx=i,
                            tail_idx=j,
                            relation_type=max_rel.item(),
                            confidence=max_prob.item()
                        ))
            
            all_relations.append(batch_relations)
        
        return all_relations


class ProductionInference:
    """
    Production inference pipeline for text-only input.
    
    Combines entity, concept, and relation extraction to provide
    complete structured output from raw text.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        entity_confidence: float = 0.5,
        concept_confidence: float = 0.1,
        relation_confidence: float = 0.3,
        device: str = "cpu"
    ):
        """
        Args:
            model: The neurosymbolic model
            tokenizer: Tokenizer for text processing
            entity_confidence: Threshold for entity extraction
            concept_confidence: Threshold for concept extraction
            relation_confidence: Threshold for relation extraction
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize extractors
        self.entity_extractor = EntityExtractor(confidence_threshold=entity_confidence)
        self.concept_extractor = ConceptExtractor(confidence_threshold=concept_confidence)
        self.relation_extractor = RelationExtractor(confidence_threshold=relation_confidence)
    
    @torch.no_grad()
    def extract(self, text: str) -> ExtractionResult:
        """
        Extract all structured information from text.
        
        Args:
            text: Input text
        
        Returns:
            ExtractionResult with entities, concepts, relations
        """
        self.model.eval()
        
        # Tokenize
        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids, attention_mask, spans=None, y_ids=None)
        
        # Extract entities
        entities_batch = self.entity_extractor.extract(
            outputs["entity_logits"],
            input_ids,
            self.tokenizer,
            attention_mask
        )
        entities = entities_batch[0]  # Single text
        
        # Extract concepts
        n_entities = len(entities)
        if n_entities > 0:
            concepts, concept_confs = self.concept_extractor.extract(
                outputs["concept_probs"],
                n_entities
            )
        else:
            concepts, concept_confs = [], []
        
        # Extract relations
        if "rel_logits_matrix" in outputs and n_entities > 0:
            relations_batch = self.relation_extractor.extract(
                outputs["rel_logits_matrix"],
                n_entities
            )
            relations = relations_batch[0]
        else:
            relations = []
        
        # Get controller decision
        controller_logits = outputs["controller_logits"]
        controller_probs = F.softmax(controller_logits, dim=-1)
        controller_decision_idx = controller_probs[0].argmax().item()
        controller_decisions = ["answer", "abstain", "ask_clarify"]
        controller_decision = controller_decisions[min(controller_decision_idx, 2)]
        controller_confidence = controller_probs[0, controller_decision_idx].item()
        
        return ExtractionResult(
            text=text,
            entities=entities,
            concepts=concepts,
            concept_confidences=concept_confs,
            relations=relations,
            controller_decision=controller_decision,
            controller_confidence=controller_confidence
        )
    
    @torch.no_grad()
    def extract_batch(self, texts: List[str]) -> List[ExtractionResult]:
        """Extract from multiple texts."""
        results = []
        for text in texts:
            results.append(self.extract(text))
        return results
    
    @torch.no_grad()
    def generate_response(
        self,
        text: str,
        max_length: int = 128,
        **generation_kwargs
    ) -> Tuple[str, ExtractionResult]:
        """
        Generate a response and return extractions.
        
        Args:
            text: Input text
            max_length: Maximum generation length
            **generation_kwargs: Additional generation parameters
        
        Returns:
            (generated_text, extraction_result)
        """
        # First extract structure
        extraction = self.extract(text)
        
        # Tokenize for generation
        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        
        # Generate
        generated_ids = self.model.generate(
            input_ids,
            attention_mask,
            max_length=max_length,
            bos_token_id=self.tokenizer.cls_token_id or 101,
            eos_token_id=self.tokenizer.sep_token_id or 102,
            pad_token_id=self.tokenizer.pad_token_id or 0,
            **generation_kwargs
        )
        
        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        
        return generated_text, extraction


class SelfLabelingPipeline:
    """
    Pipeline for generating pseudo-labels from model predictions.
    
    Used for self-supervised online learning when no ground truth is available.
    """
    
    def __init__(
        self,
        inference: ProductionInference,
        entity_threshold: float = 0.7,
        concept_threshold: float = 0.5,
        relation_threshold: float = 0.5,
        min_entities: int = 1
    ):
        """
        Args:
            inference: ProductionInference instance
            entity_threshold: High-confidence threshold for entity pseudo-labels
            concept_threshold: High-confidence threshold for concept pseudo-labels
            relation_threshold: High-confidence threshold for relation pseudo-labels
            min_entities: Minimum entities required to generate labels
        """
        self.inference = inference
        self.entity_threshold = entity_threshold
        self.concept_threshold = concept_threshold
        self.relation_threshold = relation_threshold
        self.min_entities = min_entities
    
    def generate_labels(self, text: str) -> Tuple[Optional[Dict], float]:
        """
        Generate pseudo-labels for text using high-confidence predictions.
        
        Args:
            text: Input text
        
        Returns:
            (pseudo_labels, confidence) - None if confidence too low
        """
        extraction = self.inference.extract(text)
        
        # Filter to high-confidence entities
        high_conf_entities = [
            e for e in extraction.entities
            if e.confidence >= self.entity_threshold
        ]
        
        if len(high_conf_entities) < self.min_entities:
            return None, 0.0
        
        # Filter to high-confidence relations
        entity_indices = {
            extraction.entities.index(e): i
            for i, e in enumerate(high_conf_entities)
            if e in extraction.entities
        }
        
        high_conf_relations = []
        for r in extraction.relations:
            if r.confidence >= self.relation_threshold:
                # Remap indices to filtered entities
                if r.head_idx in entity_indices and r.tail_idx in entity_indices:
                    high_conf_relations.append(ExtractedRelation(
                        head_idx=entity_indices[r.head_idx],
                        tail_idx=entity_indices[r.tail_idx],
                        relation_type=r.relation_type,
                        confidence=r.confidence
                    ))
        
        # Compute overall confidence
        entity_confs = [e.confidence for e in high_conf_entities]
        overall_confidence = sum(entity_confs) / len(entity_confs)
        
        # Build pseudo-label dictionary
        pseudo_labels = {
            "text": text,
            "entities": [e.text for e in high_conf_entities],
            "entity_type_labels": [e.entity_type for e in high_conf_entities],
            "entity_spans": [(e.start_char, e.end_char) for e in high_conf_entities],
            "entity_token_spans": [(e.start_token, e.end_token) for e in high_conf_entities],
            "concepts": extraction.concepts[:len(high_conf_entities)],
            "relations": [
                (r.head_idx, r.tail_idx, r.relation_type)
                for r in high_conf_relations
            ],
            "should_respond": 1 if extraction.controller_decision == "answer" else 0,
            # Metadata for tracking
            "_pseudo_labeled": True,
            "_entity_confidences": entity_confs,
            "_overall_confidence": overall_confidence,
        }
        
        return pseudo_labels, overall_confidence
    
    def generate_batch_labels(
        self,
        texts: List[str],
        min_confidence: float = 0.6
    ) -> Tuple[List[Dict], List[str]]:
        """
        Generate pseudo-labels for a batch of texts.
        
        Args:
            texts: List of input texts
            min_confidence: Minimum overall confidence to include
        
        Returns:
            (labeled_samples, rejected_texts)
        """
        labeled = []
        rejected = []
        
        for text in texts:
            labels, confidence = self.generate_labels(text)
            
            if labels is not None and confidence >= min_confidence:
                labeled.append(labels)
            else:
                rejected.append(text)
        
        return labeled, rejected
