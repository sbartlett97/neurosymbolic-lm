"""Dataset classes and text processing utilities."""

import json
from typing import List, Tuple, Optional
from torch.utils.data import Dataset


def statement_to_question(text: str) -> str:
    """
    Convert a statement to a question form.
    
    Handles common patterns like:
    - "X is Y" -> "Is X Y?"
    - "X verb Y" -> "Did X verb Y?" or "Does X verb Y?"
    - "The X verb Y" -> "Did the X verb Y?" or "Does the X verb Y?"
    
    If text is already a question, returns it unchanged.
    """
    text = text.strip()
    if not text:
        return text
    
    # Check if already a question - return as-is
    question_starters = (
        'Is ', 'Are ', 'Was ', 'Were ', 'Do ', 'Does ', 'Did ', 
        'Can ', 'Could ', 'Would ', 'Should ', 'Will ', 'Have ', 'Has ', 'Had ',
        'Where ', 'What ', 'Who ', 'When ', 'Why ', 'How ', 'Which '
    )
    if text.startswith(question_starters):
        # Already a question - ensure it ends with ?
        if not text.endswith('?'):
            text = text.rstrip('.') + '?'
        return text
    
    # Remove trailing period
    if text.endswith('.'):
        text = text[:-1]
    
    # Handle "is" statements: "X is Y" -> "Is X Y?"
    if text.startswith(('The ', 'A ', 'An ')):
        # Check for "is" pattern
        if ' is ' in text or ' is the ' in text or ' is a ' in text or ' is an ' in text:
            parts = text.split(' is ', 1)
            if len(parts) == 2:
                return f"Is {parts[0]} {parts[1]}?"
    
    # Past tense indicators
    past_tense_indicators = [
        'ed ', 'ed.', ' chased', ' barked', ' flew', ' hunted', ' swam', 
        ' lived', ' hopped', ' howled', ' galloped', ' collected', ' spun',
        ' met', ' wrote', ' discovered', ' played', ' helped', ' read',
        ' trained', ' decided', ' fixed', ' captured', ' designed',
        ' painted', ' cooked', ' studied', ' grew', ' flew', ' made',
        ' published', ' fell', ' ran', ' stood', ' stretched', ' appeared',
        ' twinkled', ' damaged', ' burned', ' blew', ' shines', ' lights',
        ' reflects', ' hit', ' cut', ' tied', ' echoed', ' powers'
    ]
    
    # Present tense indicators
    present_tense_indicators = [
        's ', 's.', ' chases', ' barks', ' flies', ' hunts', ' swims',
        ' lives', ' hops', ' howls', ' gallops', ' collects', ' spins',
        ' meets', ' writes', ' discovers', ' plays', ' helps', ' reads',
        ' trains', ' decides', ' fixes', ' captures', ' designs',
        ' paints', ' cooks', ' studies', ' grows', ' makes',
        ' publishes', ' falls', ' runs', ' stands', ' stretches', ' appears',
        ' twinkles', ' damages', ' burns', ' blows', ' lights',
        ' reflects', ' hits', ' cuts', ' ties', ' echoes', ' powers',
        ' opens', ' writes', ' shows', ' faces', ' protects', ' crosses',
        ' leads', ' contains', ' displays', ' teaches', ' owns'
    ]
    
    is_past_tense = any(indicator in text for indicator in past_tense_indicators)
    is_present_tense = any(indicator in text for indicator in present_tense_indicators)
    
    # Handle "The X verb Y" or "A X verb Y" patterns
    if text.startswith(('The ', 'A ', 'An ')):
        if is_past_tense:
            return f"Did {text.lower()}?"
        elif is_present_tense or ' is ' in text:
            if ' is ' in text:
                parts = text.split(' is ', 1)
                if len(parts) == 2:
                    return f"Is {parts[0]} {parts[1]}?"
            else:
                return f"Does {text.lower()}?"
        else:
            return f"Did {text.lower()}?"
    
    # Handle sentences starting with capital letter (proper nouns or other)
    if text[0].isupper() and ' is ' in text:
        parts = text.split(' is ', 1)
        if len(parts) == 2:
            return f"Is {parts[0]} {parts[1]}?"
    
    # Fallback
    if is_past_tense:
        return f"Did {text.lower()}?"
    else:
        return f"Does {text.lower()}?"


def recalculate_entity_spans(
    question_text: str, 
    entities: List[str], 
    original_spans: Optional[List[Tuple[int, int]]] = None
) -> List[Tuple[int, int]]:
    """
    Recalculate entity spans for question text by finding entities in the question.
    
    Args:
        question_text: The question text to search in
        entities: List of entity strings to find
        original_spans: Optional original spans for fallback estimation
    
    Returns:
        List of (start, end) tuples for each entity
    """
    spans = []
    question_lower = question_text.lower()
    used_positions = set()
    
    for i, entity in enumerate(entities):
        entity_lower = entity.lower()
        start_idx = -1
        search_start = 0
        
        while True:
            found_idx = question_lower.find(entity_lower, search_start)
            if found_idx == -1:
                break
            
            # Check for overlap with already used positions
            overlap = False
            for pos in range(found_idx, found_idx + len(entity)):
                if pos in used_positions:
                    overlap = True
                    break
            
            if not overlap:
                start_idx = found_idx
                break
            
            search_start = found_idx + 1
        
        if start_idx != -1:
            end_idx = start_idx + len(entity) - 1
            spans.append((start_idx, end_idx))
            for pos in range(start_idx, start_idx + len(entity)):
                used_positions.add(pos)
        else:
            # Fallback estimation
            if original_spans and i < len(original_spans):
                orig_start, orig_end = original_spans[i]
                estimated_start = orig_start + 4
                estimated_end = estimated_start + len(entity) - 1
                spans.append((estimated_start, estimated_end))
            else:
                spans.append((0, len(entity) - 1))
    
    return spans


def generate_response_text(
    original_text: str, 
    entities: List[str], 
    relations: List[Tuple]
) -> str:
    """Generate a response text for a question based on the original statement."""
    return f"Yes, {original_text.lower().rstrip('.')}."


class ToyCognitiveDataset(Dataset):
    """
    Dataset supporting multiple concepts and relationships per entity.
    
    Key features:
    - Multiple concepts per entity: Each entity can have multiple concept labels
      (e.g., ["animal", "predator"] for "cat")
    - Multiple relationships: Entities can participate in multiple relations
    - Backward compatible: Single concept strings are automatically converted to lists
    
    Example with multiple concepts:
        {
            "text": "The doctor treated the patient.",
            "entities": ["doctor", "patient"],
            "concepts": [["person", "professional", "medical"], ["person", "medical"]],
            "relations": [(0, 1, "treats")]
        }
    """
    
    def __init__(self, jsonl_file_path: Optional[str] = None):
        """
        Initialize dataset from JSONL file or use hardcoded toy data.
        
        Args:
            jsonl_file_path: Path to JSONL file. If None, uses hardcoded toy dataset.
        """
        if jsonl_file_path is not None:
            self.data = self._load_from_jsonl(jsonl_file_path)
        else:
            self.data = self._get_default_data()
        
        self._normalize_concepts()
        self._process_entries()
    
    def _load_from_jsonl(self, filepath: str) -> List[dict]:
        """Load dataset from JSONL file."""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    # Convert lists to tuples for spans and relations
                    if "entity_spans" in entry and entry["entity_spans"]:
                        if len(entry["entity_spans"]) > 0 and isinstance(entry["entity_spans"][0], list):
                            entry["entity_spans"] = [
                                tuple(span) if isinstance(span, list) else span 
                                for span in entry["entity_spans"]
                            ]
                    if "relations" in entry and entry["relations"]:
                        if len(entry["relations"]) > 0 and isinstance(entry["relations"][0], list):
                            entry["relations"] = [
                                tuple(rel) if isinstance(rel, list) else rel 
                                for rel in entry["relations"]
                            ]
                    data.append(entry)
        return data
    
    def _get_default_data(self) -> List[dict]:
        """Return hardcoded toy dataset for testing."""
        return [
            {
                "text": "The cat chased the mouse.",
                "entities": ["cat", "mouse"],
                "entity_spans": [(4, 7), (20, 25)],
                "concepts": [["animal", "predator"], ["animal", "prey"]],
                "relations": [(0, 1, "chases")],
                "should_respond": 1,
                "response": "Yes, the cat chases the mouse.",
            },
            {
                "text": "Blue is a colour.",
                "entities": ["Blue"],
                "entity_spans": [(0, 4)],
                "concepts": ["attribute"],
                "relations": [],
                "should_respond": 0,
            },
            {
                "text": "Paris is the capital of France.",
                "entities": ["Paris", "France"],
                "entity_spans": [(0, 5), (24, 30)],
                "concepts": [["location", "city"], ["location", "country"]],
                "relations": [(0, 1, "capital_of")],
                "should_respond": 1,
                "response": "Yes, Paris is the capital of France.",
            },
            {
                "text": "The dog barked at the mailman.",
                "entities": ["dog", "mailman"],
                "entity_spans": [(4, 7), (20, 27)],
                "concepts": ["animal", "person"],
                "relations": [(0, 1, "barks_at")],
                "should_respond": 1,
                "response": "Yes, the dog barks at the mailman.",
            },
            {
                "text": "Alice loves Bob.",
                "entities": ["Alice", "Bob"],
                "entity_spans": [(0, 5), (12, 15)],
                "concepts": ["person", "person"],
                "relations": [(0, 1, "loves")],
                "should_respond": 1,
            },
        ]
    
    def _normalize_concepts(self):
        """Convert single concept strings to lists for backward compatibility."""
        for entry in self.data:
            if "concepts" in entry:
                normalized = []
                for concept in entry["concepts"]:
                    if isinstance(concept, str):
                        normalized.append([concept])
                    elif isinstance(concept, list):
                        normalized.append(concept)
                    else:
                        normalized.append([str(concept)])
                entry["concepts"] = normalized
    
    def _is_question(self, text: str) -> bool:
        """Check if text is already a question."""
        text = text.strip()
        question_starters = (
            'Is ', 'Are ', 'Was ', 'Were ', 'Do ', 'Does ', 'Did ',
            'Can ', 'Could ', 'Would ', 'Should ', 'Will ', 'Have ', 'Has ', 'Had ',
            'Where ', 'What ', 'Who ', 'When ', 'Why ', 'How ', 'Which '
        )
        return text.startswith(question_starters) or text.endswith('?')
    
    def _process_entries(self):
        """Process entries: handle questions and statements appropriately.
        
        - If text is already a question with should_respond=1: use as-is
        - If text is a statement with should_respond=1: create both statement 
          (should_respond=0) and question (should_respond=1) versions
        - If should_respond=0: use as-is
        """
        processed_data = []
        for entry in self.data:
            if entry.get("should_respond", 0) == 1:
                text = entry["text"]
                
                if self._is_question(text):
                    # Text is already a question - use as-is
                    processed_data.append(entry)
                else:
                    # Text is a statement - create both versions
                    # Create statement version with should_respond=0
                    statement_entry = entry.copy()
                    statement_entry["should_respond"] = 0
                    if "response" in statement_entry:
                        del statement_entry["response"]
                    processed_data.append(statement_entry)
                    
                    # Create question version with should_respond=1
                    question_text = statement_to_question(text)
                    question_entry = entry.copy()
                    question_entry["text"] = question_text
                    question_entry["entity_spans"] = recalculate_entity_spans(
                        question_text, entry["entities"], entry.get("entity_spans")
                    )
                    
                    if "response" not in question_entry or not question_entry.get("response"):
                        question_entry["response"] = generate_response_text(
                            entry["text"], entry["entities"], entry["relations"]
                        )
                    
                    question_entry["should_respond"] = 1
                    processed_data.append(question_entry)
            else:
                processed_data.append(entry)
        
        self.data = processed_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]
