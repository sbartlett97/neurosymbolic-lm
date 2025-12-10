"""Data collator for batching cognitive dataset samples."""

from typing import List, Dict, Optional, Tuple
import torch


class CognitiveCollator:
    """
    Collator for batching cognitive dataset samples.
    
    Handles tokenization, entity encoding, concept mapping, and relation extraction.
    Includes entity token span alignment for proper entity classification.
    """
    
    def __init__(
        self, 
        tokenizer, 
        concept_map: Dict[str, int], 
        relation_map: Dict[str, int], 
        include_responses: bool = False, 
        concept_to_entity_type_map: Optional[Dict[str, int]] = None,
        decoder_tokenizer=None
    ):
        """
        Initialize the collator.
        
        Args:
            tokenizer: HuggingFace tokenizer for encoder
            concept_map: Mapping from concept names to indices (1-indexed)
            relation_map: Mapping from relation names to indices (1-indexed)
            include_responses: Whether to include decoder inputs/labels
            concept_to_entity_type_map: Mapping from concepts to entity type indices
            decoder_tokenizer: Optional separate tokenizer for decoder (e.g., T5 tokenizer)
                              If None, uses the same tokenizer as encoder
        """
        self.tokenizer = tokenizer
        self.decoder_tokenizer = decoder_tokenizer if decoder_tokenizer is not None else tokenizer
        self.concept_map = concept_map
        self.relation_map = relation_map
        self.include_responses = include_responses
        self.concept_to_entity_type_map = concept_to_entity_type_map or {}
        self.n_concepts = max(concept_map.values()) if concept_map else 0
    
    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of sample dictionaries
        
        Returns:
            Dictionary with batched tensors
        """
        texts = [x["text"] for x in batch]
        tok = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
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
    
    def _compute_entity_token_spans(
        self,
        sample: dict,
        text: str,
        input_ids: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """Compute token-level spans for entities.
        
        Args:
            sample: Sample dictionary with entities and entity_spans
            text: Original text
            input_ids: Tokenized input IDs
        
        Returns:
            List of (start_token_idx, end_token_idx) tuples
        """
        token_spans = []
        entities = sample.get("entities", [])
        char_spans = sample.get("entity_spans", [])
        
        if not char_spans:
            # Fallback: try to find entity by string matching
            for entity in entities:
                entity_lower = entity.lower()
                text_lower = text.lower()
                start_char = text_lower.find(entity_lower)
                if start_char >= 0:
                    char_spans.append((start_char, start_char + len(entity)))
                else:
                    char_spans.append(None)
        
        for i, entity in enumerate(entities):
            if i >= len(char_spans) or char_spans[i] is None:
                # Default to first token if we can't find the span
                token_spans.append((1, 1))  # Skip [CLS]
                continue
            
            char_start, char_end = char_spans[i]
            
            # Convert character spans to token spans
            token_start, token_end = self._char_to_token_span(
                text, char_start, char_end, input_ids
            )
            token_spans.append((token_start, token_end))
        
        return token_spans
    
    def _char_to_token_span(
        self,
        text: str,
        char_start: int,
        char_end: int,
        input_ids: torch.Tensor
    ) -> Tuple[int, int]:
        """Convert character-level span to token-level span.
        
        Args:
            text: Original text
            char_start: Character start index
            char_end: Character end index
            input_ids: Tokenized input IDs
        
        Returns:
            (token_start, token_end) tuple
        """
        # Tokenize with return_offsets_mapping if available
        try:
            encoded = self.tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                add_special_tokens=True
            )
            offsets = encoded.get("offset_mapping", [])
            
            if offsets:
                token_start = None
                token_end = None
                
                for idx, (start, end) in enumerate(offsets):
                    if start is None or end is None:
                        continue
                    # Token overlaps with entity span
                    if start < char_end and end > char_start:
                        if token_start is None:
                            token_start = idx
                        token_end = idx
                
                if token_start is not None and token_end is not None:
                    return (token_start, token_end)
        except Exception:
            pass
        
        # Fallback: estimate based on character position
        # Rough approximation: average 4 chars per token
        approx_token_start = max(1, char_start // 4)  # Skip [CLS]
        approx_token_end = max(approx_token_start, char_end // 4)
        
        # Clamp to valid range
        max_idx = len(input_ids) - 2  # Account for [CLS] and [SEP]
        approx_token_start = min(approx_token_start, max_idx)
        approx_token_end = min(approx_token_end, max_idx)
        
        return (approx_token_start, approx_token_end)
    
    def _process_responses(self, batch: List[dict]):
        """Process response texts for decoder training.
        
        Uses decoder_tokenizer which may be different from encoder tokenizer
        (e.g., T5 tokenizer for T5 decoder, BERT tokenizer for encoder).
        
        For T5:
        - decoder_input_ids should start with pad_token_id (decoder_start_token_id)
        - labels are the actual target tokens
        """
        responses = []
        should_respond_mask = []
        
        for x in batch:
            if x.get("should_respond", 0) == 1 and "response" in x and x["response"].strip():
                responses.append(x["response"])
                should_respond_mask.append(True)
            else:
                # For non-response samples, use a minimal placeholder
                responses.append("")
                should_respond_mask.append(False)
        
        # Use decoder_tokenizer for response tokenization
        resp_tok = self.decoder_tokenizer(
            responses, 
            padding=True, 
            truncation=True, 
            max_length=128,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        # Labels are the tokenized responses
        decoder_labels = resp_tok["input_ids"].clone()
        
        # For T5-style models, decoder_input_ids should be shifted right
        # with decoder_start_token_id (usually pad_token_id) prepended
        decoder_start_token_id = getattr(
            self.decoder_tokenizer, 'decoder_start_token_id', 
            self.decoder_tokenizer.pad_token_id
        )
        if decoder_start_token_id is None:
            decoder_start_token_id = 0
        
        # Create decoder_input_ids by shifting right and prepending start token
        batch_size, seq_len = decoder_labels.shape
        decoder_input_ids = torch.zeros_like(decoder_labels)
        decoder_input_ids[:, 0] = decoder_start_token_id
        decoder_input_ids[:, 1:] = decoder_labels[:, :-1]
        
        # Set padding tokens to -100 in labels (ignore in loss)
        pad_token_id = self.decoder_tokenizer.pad_token_id
        if pad_token_id is not None:
            decoder_labels[decoder_labels == pad_token_id] = -100
        
        # Handle non-response samples - set all labels to -100
        for i, should_respond in enumerate(should_respond_mask):
            if not should_respond:
                decoder_labels[i] = -100
        
        return decoder_input_ids, decoder_labels
    
    def _get_eos_token(self) -> str:
        """Get the EOS token string from decoder tokenizer."""
        eos_token = self.decoder_tokenizer.eos_token
        if eos_token is None:
            eos_token = self.decoder_tokenizer.sep_token
        if eos_token is None:
            eos_token = self.decoder_tokenizer.pad_token
        if eos_token is None:
            eos_token = "</s>"
        return eos_token
    
    def _get_eos_token_id(self) -> Optional[int]:
        """Get the EOS token ID from decoder tokenizer."""
        eos_token_id = self.decoder_tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.decoder_tokenizer.sep_token_id
        if eos_token_id is None:
            eos_token_id = self.decoder_tokenizer.pad_token_id
        return eos_token_id
    
    def _process_entities(
        self, 
        sample: dict, 
        batch_idx: int,
        entity_ids: torch.Tensor,
        entity_type_labels: torch.Tensor,
        concept_labels: torch.Tensor,
        max_entities: int
    ):
        """Process entity information for a single sample."""
        ents = sample["entities"]
        
        for j, ent in enumerate(ents):
            entity_ids[batch_idx, j] = j + 1  # 1-indexed
            
            # Get concepts for this entity
            entity_concepts = sample["concepts"][j]
            if isinstance(entity_concepts, str):
                entity_concepts = [entity_concepts]
            elif not isinstance(entity_concepts, list):
                entity_concepts = [entity_concepts]
            
            # Map to entity type
            entity_type = 0
            for concept_name in entity_concepts:
                if concept_name in self.concept_to_entity_type_map:
                    entity_type = self.concept_to_entity_type_map[concept_name]
                    break
            entity_type_labels[batch_idx, j] = entity_type
            
            # Set binary concept labels
            for concept_name in entity_concepts:
                concept_val = self.concept_map.get(concept_name, 0)
                if concept_val > 0:
                    concept_idx = concept_val - 1  # Convert to 0-indexed
                    concept_labels[batch_idx, j, concept_idx] = 1.0
        
        # Pad remaining slots
        for j in range(len(ents), max_entities):
            entity_ids[batch_idx, j] = 0
            entity_type_labels[batch_idx, j] = 0
    
    def _process_relations(self, sample: dict) -> torch.Tensor:
        """Process relations for a single sample."""
        rels = sample["relations"]
        rel_tensor = torch.zeros((len(rels), 3), dtype=torch.long)
        
        for k, (h, t, r) in enumerate(rels):
            rel_tensor[k] = torch.tensor([h, t, self.relation_map.get(r, 0)])
        
        return rel_tensor
