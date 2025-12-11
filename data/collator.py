"""Data collator for batching cognitive dataset samples."""

from typing import List, Dict, Optional, Tuple
import torch


class CognitiveCollator:
    """
    Collator for batching cognitive dataset samples.
    
    Handles tokenization, entity encoding, concept mapping, and relation extraction.
    Uses a single tokenizer for both encoder inputs and decoder targets.
    """
    
    def __init__(
        self, 
        tokenizer, 
        concept_map: Dict[str, int], 
        relation_map: Dict[str, int], 
        include_responses: bool = False, 
        concept_to_entity_type_map: Optional[Dict[str, int]] = None,
        max_length: int = 512,
        max_output_length: int = 256
    ):
        """
        Initialize the collator.
        
        Args:
            tokenizer: HuggingFace tokenizer (e.g., T5 tokenizer)
            concept_map: Mapping from concept names to indices (1-indexed)
            relation_map: Mapping from relation names to indices (1-indexed)
            include_responses: Whether to include decoder inputs/labels
            concept_to_entity_type_map: Mapping from concepts to entity type indices
            max_length: Maximum input sequence length
            max_output_length: Maximum output/response sequence length
        """
        self.tokenizer = tokenizer
        self.concept_map = concept_map
        self.relation_map = relation_map
        self.include_responses = include_responses
        self.concept_to_entity_type_map = concept_to_entity_type_map or {}
        self.n_concepts = max(concept_map.values()) if concept_map else 0
        self.max_length = max_length
        self.max_output_length = max_output_length
    
    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of sample dictionaries
        
        Returns:
            Dictionary with batched tensors
        """
        texts = [x["text"] for x in batch]
        tok = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        max_entities = max(len(x["entities"]) for x in batch) if batch else 1
        entity_ids = torch.zeros(len(batch), max_entities, dtype=torch.long)
        entity_type_labels = torch.zeros(len(batch), max_entities, dtype=torch.long)
        concept_labels = torch.zeros(len(batch), max_entities, self.n_concepts, dtype=torch.float)
        
        relation_triplets = []
        should_respond_values = [int(x.get("should_respond", 0)) for x in batch]
        should_respond = torch.tensor(should_respond_values, dtype=torch.long)
        
        # Entity token spans for proper alignment
        entity_token_spans = []
        
        # Process decoder inputs if needed
        decoder_input_ids = None
        decoder_labels = None
        if self.include_responses:
            decoder_input_ids, decoder_labels = self._process_responses(batch)
        
        # Process each sample
        for i, sample in enumerate(batch):
            self._process_entities(
                sample, i, entity_ids, entity_type_labels, concept_labels, max_entities
            )
            relation_triplets.append(self._process_relations(sample))
            
            # Compute token-level spans for entity alignment
            token_spans = self._compute_entity_token_spans(
                sample, texts[i], tok["input_ids"][i]
            )
            entity_token_spans.append(token_spans)
        
        result = {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "entity_ids": entity_ids,
            "entity_type_labels": entity_type_labels,
            "concept_labels": concept_labels,
            "relations": relation_triplets,
            "should_respond": should_respond,
            "entity_token_spans": entity_token_spans,
        }
        
        if decoder_input_ids is not None:
            result["decoder_input_ids"] = decoder_input_ids
            result["decoder_labels"] = decoder_labels
        
        return result
    
    def _process_responses(self, batch: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process response texts for decoder training.
        
        For T5:
        - decoder_input_ids starts with pad_token_id (decoder_start_token_id)
        - labels are the target tokens (response + eos)
        
        Abstention Learning (EOS-based):
        For samples with should_respond=0, the labels are set to [eos_token, -100, ...].
        This teaches the decoder to output EOS immediately, effectively learning to
        "abstain" or "stay silent" without needing a separate decision head.
        
        The decoder learns:
        - should_respond=1: Generate full response ending with EOS
        - should_respond=0: Generate EOS immediately (abstain)
        """
        responses = []
        should_respond_mask = []
        
        for x in batch:
            if x.get("should_respond", 0) == 1 and "response" in x and x["response"].strip():
                responses.append(x["response"])
                should_respond_mask.append(True)
            else:
                # Placeholder - will be overridden
                responses.append("")
                should_respond_mask.append(False)
        
        # Get special token IDs
        pad_token_id = self.tokenizer.pad_token_id or 0
        eos_token_id = self.tokenizer.eos_token_id or 1
        decoder_start_token_id = getattr(self.tokenizer, 'decoder_start_token_id', pad_token_id)
        
        # Tokenize responses
        resp_tok = self.tokenizer(
            responses, 
            padding=True, 
            truncation=True, 
            max_length=self.max_output_length,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        # Labels are the tokenized responses
        decoder_labels = resp_tok["input_ids"].clone()
        
        # Create decoder_input_ids: shift right and prepend start token
        batch_size, seq_len = decoder_labels.shape
        decoder_input_ids = torch.full_like(decoder_labels, pad_token_id)
        decoder_input_ids[:, 0] = decoder_start_token_id
        decoder_input_ids[:, 1:] = decoder_labels[:, :-1]
        
        # Set padding tokens to -100 in labels (ignore in loss)
        decoder_labels[decoder_labels == pad_token_id] = -100
        
        # Handle non-response samples: teach model to output EOS immediately
        for i, should_respond in enumerate(should_respond_mask):
            if not should_respond:
                # decoder_input_ids: [start_token, pad, pad, ...]
                decoder_input_ids[i] = pad_token_id
                decoder_input_ids[i, 0] = decoder_start_token_id
                
                # labels: [eos_token, -100, -100, ...] - only predict EOS
                decoder_labels[i] = -100
                decoder_labels[i, 0] = eos_token_id
        
        return decoder_input_ids, decoder_labels
    
    def _process_entities(
        self, 
        sample: dict, 
        batch_idx: int,
        entity_ids: torch.Tensor,
        entity_type_labels: torch.Tensor,
        concept_labels: torch.Tensor,
        max_entities: int
    ):
        """Process entities from a sample."""
        entities = sample.get("entities", [])
        concepts = sample.get("concepts", [])
        
        for j, ent in enumerate(entities):
            if j >= max_entities:
                break
            
            entity_ids[batch_idx, j] = j + 1
            
            # Get concepts for this entity
            if j < len(concepts):
                ent_concepts = concepts[j]
                if isinstance(ent_concepts, str):
                    ent_concepts = [ent_concepts]
                elif not isinstance(ent_concepts, list):
                    ent_concepts = [str(ent_concepts)]
                
                # Map concepts to entity types
                for concept in ent_concepts:
                    concept_idx = self.concept_map.get(concept, 0)
                    if concept_idx > 0 and concept_idx <= self.n_concepts:
                        concept_labels[batch_idx, j, concept_idx - 1] = 1.0
                    
                    if concept in self.concept_to_entity_type_map:
                        entity_type_labels[batch_idx, j] = self.concept_to_entity_type_map[concept]
    
    def _process_relations(self, sample: dict) -> List[Tuple[int, int, int]]:
        """Extract relation triplets from a sample.
        
        Supports two relation formats:
        1. Index-based: (head_idx, tail_idx, relation_type) - e.g., [0, 1, "lives_in"]
        2. Name-based: (head_name, relation_type, tail_name) - e.g., ["cat", "chases", "mouse"]
        """
        triplets = []
        relations = sample.get("relations", [])
        entities = sample.get("entities", [])
        
        for rel in relations:
            if len(rel) != 3:
                continue
            
            # Determine format based on first element type
            if isinstance(rel[0], int):
                # Index-based format: (head_idx, tail_idx, relation_type)
                head_idx, tail_idx, rel_type = rel
            else:
                # Name-based format: (head_name, relation_type, tail_name)
                head, rel_type, tail = rel
                head_idx = entities.index(head) if head in entities else -1
                tail_idx = entities.index(tail) if tail in entities else -1
            
            if head_idx >= 0 and tail_idx >= 0 and head_idx < len(entities) and tail_idx < len(entities):
                rel_type_idx = self.relation_map.get(rel_type, 0)
                if rel_type_idx > 0:
                    triplets.append((head_idx, tail_idx, rel_type_idx))
        
        return triplets
    
    def _compute_entity_token_spans(
        self,
        sample: dict,
        text: str,
        input_ids: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """
        Compute token-level spans for entities.
        
        Returns list of (start_token_idx, end_token_idx) for each entity.
        """
        entities = sample.get("entities", [])
        token_spans = []
        
        for entity in entities:
            # Find character span in original text
            try:
                char_start = text.lower().find(entity.lower())
                if char_start == -1:
                    token_spans.append((0, 0))
                    continue
                char_end = char_start + len(entity)
            except:
                token_spans.append((0, 0))
                continue
            
            # Convert to token span
            token_span = self._char_span_to_token_span(
                text, char_start, char_end, input_ids
            )
            token_spans.append(token_span)
        
        return token_spans
    
    def _char_span_to_token_span(
        self,
        text: str,
        char_start: int,
        char_end: int,
        input_ids: torch.Tensor
    ) -> Tuple[int, int]:
        """Convert character span to token span."""
        # Use tokenizer's char_to_token if available
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
        offsets = encoding.get("offset_mapping", [])
        
        if not offsets:
            # Fallback: approximate token positions
            tokens_before = len(self.tokenizer.encode(text[:char_start], add_special_tokens=False))
            tokens_in_span = len(self.tokenizer.encode(text[char_start:char_end], add_special_tokens=False))
            return (tokens_before + 1, tokens_before + tokens_in_span)
        
        start_token = None
        end_token = None
        
        for idx, (offset_start, offset_end) in enumerate(offsets):
            if offset_start is None or offset_end is None:
                continue
            
            # Find first token that overlaps with char_start
            if start_token is None and offset_end > char_start:
                start_token = idx
            
            # Find last token that overlaps with char_end
            if offset_start < char_end:
                end_token = idx
        
        if start_token is None:
            start_token = 1  # Skip special tokens
        if end_token is None:
            end_token = start_token
        
        # Clamp to valid range
        max_idx = len(input_ids) - 1
        start_token = min(start_token, max_idx)
        end_token = min(end_token, max_idx)
        
        return (start_token, end_token)
