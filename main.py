# Toy dataset template and tiered training scaffold

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Optional, Dict

# Schema:
# sample = {
#   "text": "The cat chased the mouse.",
#   "entities": ["cat", "mouse"],
#   "entity_spans": [(4,7),(20,25)],
#   "concepts": [["animal", "predator"], ["animal", "prey"]],  # Multiple concepts per entity (list of lists)
#   "relations": [(0,1,"chases")],  # Can have multiple relations per entity pair
#   "should_respond": 1,
#   "response": "Yes, the cat chases the mouse."  # Optional: response text for decoder training
# }
# 
# For backward compatibility, single concepts can still be provided as strings:
#   "concepts": ["animal", "animal"]  # Will be converted to [["animal"], ["animal"]]

def statement_to_question(text):
    """
    Convert a statement to a question form.
    Handles common patterns like:
    - "X is Y" -> "Is X Y?"
    - "X verb Y" -> "Did X verb Y?" or "Does X verb Y?"
    - "The X verb Y" -> "Did the X verb Y?" or "Does the X verb Y?"
    """
    text = text.strip()
    if not text:
        return text
    
    # Remove trailing period
    if text.endswith('.'):
        text = text[:-1]
    
    # Handle "is" statements: "X is Y" -> "Is X Y?"
    if text.startswith(('The ', 'A ', 'An ')):
        # Check for "is" pattern
        if ' is ' in text or ' is the ' in text or ' is a ' in text or ' is an ' in text:
            # "The X is Y" -> "Is the X Y?"
            parts = text.split(' is ', 1)
            if len(parts) == 2:
                return f"Is {parts[0]} {parts[1]}?"
    
    # Handle past tense verbs (simple heuristic)
    past_tense_indicators = ['ed ', 'ed.', ' chased', ' barked', ' flew', ' hunted', ' swam', 
                            ' lived', ' hopped', ' howled', ' galloped', ' collected', ' spun',
                            ' met', ' wrote', ' discovered', ' played', ' helped', ' read',
                            ' trained', ' decided', ' fixed', ' captured', ' designed',
                            ' painted', ' cooked', ' studied', ' grew', ' flew', ' made',
                            ' published', ' fell', ' ran', ' stood', ' stretched', ' appeared',
                            ' twinkled', ' damaged', ' burned', ' blew', ' shines', ' lights',
                            ' reflects', ' hit', ' cut', ' tied', ' echoed', ' powers']
    
    # Handle present tense verbs
    present_tense_indicators = ['s ', 's.', ' chases', ' barks', ' flies', ' hunts', ' swims',
                               ' lives', ' hops', ' howls', ' gallops', ' collects', ' spins',
                               ' meets', ' writes', ' discovers', ' plays', ' helps', ' reads',
                               ' trains', ' decides', ' fixes', ' captures', ' designs',
                               ' paints', ' cooks', ' studies', ' grows', ' makes',
                               ' publishes', ' falls', ' runs', ' stands', ' stretches', ' appears',
                               ' twinkles', ' damages', ' burns', ' blows', ' lights',
                               ' reflects', ' hits', ' cuts', ' ties', ' echoes', ' powers',
                               ' opens', ' writes', ' shows', ' faces', ' protects', ' crosses',
                               ' leads', ' contains', ' displays', ' teaches', ' owns']
    
    # Check for past tense
    is_past_tense = any(indicator in text for indicator in past_tense_indicators)
    is_present_tense = any(indicator in text for indicator in present_tense_indicators)
    
    # Handle "The X verb Y" or "A X verb Y" patterns
    if text.startswith(('The ', 'A ', 'An ')):
        if is_past_tense:
            # "The X verbed Y" -> "Did the X verb Y?"
            # Simple approach: add "Did" at the beginning
            return f"Did {text.lower()}?"
        elif is_present_tense or ' is ' in text:
            # For present tense or "is" statements
            if ' is ' in text:
                parts = text.split(' is ', 1)
                if len(parts) == 2:
                    return f"Is {parts[0]} {parts[1]}?"
            else:
                # "The X verbs Y" -> "Does the X verb Y?"
                return f"Does {text.lower()}?"
        else:
            # Default: try to make it a question
            return f"Did {text.lower()}?"
    
    # Handle sentences starting with capital letter (proper nouns or other)
    if text[0].isupper() and ' is ' in text:
        parts = text.split(' is ', 1)
        if len(parts) == 2:
            return f"Is {parts[0]} {parts[1]}?"
    
    # Fallback: add "Did" or "Does" at the beginning
    if is_past_tense:
        return f"Did {text.lower()}?"
    else:
        return f"Does {text.lower()}?"


def recalculate_entity_spans(question_text, entities, original_spans=None):
    """
    Recalculate entity spans for question text by finding entities in the question.
    Returns list of (start, end) tuples.
    """
    spans = []
    question_lower = question_text.lower()
    used_positions = set()  # Track used character positions to avoid overlaps
    
    for i, entity in enumerate(entities):
        entity_lower = entity.lower()
        # Try to find entity in question (case-insensitive)
        # Search from the beginning, but skip already used positions
        start_idx = -1
        search_start = 0
        
        while True:
            found_idx = question_lower.find(entity_lower, search_start)
            if found_idx == -1:
                break
            
            # Check if this position overlaps with already used positions
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
            # Mark these positions as used
            for pos in range(start_idx, start_idx + len(entity)):
                used_positions.add(pos)
        else:
            # Fallback: if entity not found, try to estimate based on original spans
            # or use a simple heuristic
            if original_spans and i < len(original_spans):
                # Try to adjust original span by question prefix length
                # This is a rough estimate
                orig_start, orig_end = original_spans[i]
                # Estimate: question usually adds 3-4 chars at start ("Did ", "Is ", etc.)
                estimated_start = orig_start + 4
                estimated_end = estimated_start + len(entity) - 1
                spans.append((estimated_start, estimated_end))
            else:
                # Last resort: place at beginning (will be wrong but won't crash)
                spans.append((0, len(entity) - 1))
    
    return spans


def generate_response_text(original_text, entities, relations):
    """
    Generate a response text for a question based on the original statement.
    """
    # Simple heuristic: if we have a response pattern, use it
    # Otherwise, generate "Yes, [original statement]"
    if relations:
        return f"Yes, {original_text.lower().rstrip('.')}."
    else:
        return f"Yes, {original_text.lower().rstrip('.')}."


class ToyCognitiveDataset(Dataset):
    """
    Dataset supporting multiple concepts and relationships per entity.
    
    Key features:
    - Multiple concepts per entity: Each entity can have multiple concept labels
      (e.g., ["animal", "predator"] for "cat")
    - Multiple relationships: Entities can participate in multiple relations
      with different entities or even the same entity pair
    - Backward compatible: Single concept strings are automatically converted
      to lists for compatibility with existing data
    
    Example with multiple concepts:
        {
            "text": "The doctor treated the patient.",
            "entities": ["doctor", "patient"],
            "concepts": [["person", "professional", "medical"], ["person", "medical"]],
            "relations": [(0, 1, "treats")]
        }
    
    This allows the model to learn richer semantic representations where entities
    can belong to multiple categories simultaneously (e.g., a "doctor" is both
    a "person", a "professional", and has "medical" attributes).
    """
    def __init__(self):
        self.data = [
            # Original samples
            {
                "text": "The cat chased the mouse.",
                "entities": ["cat", "mouse"],
                "entity_spans": [(4, 7), (20, 25)],
                "concepts": [["animal", "predator"], ["animal", "prey"]],  # Multiple concepts per entity
                "relations": [(0, 1, "chases")],
                "should_respond": 1,
                "response": "Yes, the cat chases the mouse.",
            },
            {
                "text": "Blue is a colour.",
                "entities": ["Blue"],
                "entity_spans": [(0,4)],
                "concepts": ["attribute"],
                "relations": [],
                "should_respond": 0,
            },
            {
                "text": "Paris is the capital of France.",
                "entities": ["Paris", "France"],
                "entity_spans": [(0,5),(24,30)],
                "concepts": [["location", "city"], ["location", "country"]],  # Multiple concepts: city and location
                "relations": [(0,1,"capital_of")],
                "should_respond": 1,
                "response": "Yes, Paris is the capital of France.",
            },
            # Animals and actions
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
                "text": "A bird flew over the tree.",
                "entities": ["bird", "tree"],
                "entity_spans": [(2, 6), (21, 25)],
                "concepts": ["animal", "object"],
                "relations": [(0, 1, "flies_over")],
                "should_respond": 1,
            },
            {
                "text": "The lion hunts the zebra.",
                "entities": ["lion", "zebra"],
                "entity_spans": [(4, 8), (20, 25)],
                "concepts": ["animal", "animal"],
                "relations": [(0, 1, "hunts")],
                "should_respond": 1,
            },
            {
                "text": "Fish swim in water.",
                "entities": ["Fish", "water"],
                "entity_spans": [(0, 4), (15, 20)],
                "concepts": ["animal", "object"],
                "relations": [(0, 1, "swims_in")],
                "should_respond": 1,
            },
            {
                "text": "The elephant is large.",
                "entities": ["elephant"],
                "entity_spans": [(4, 12)],
                "concepts": ["animal"],
                "relations": [],
                "should_respond": 0,
            },
            # Locations and geography
            {
                "text": "London is the capital of England.",
                "entities": ["London", "England"],
                "entity_spans": [(0, 6), (28, 35)],
                "concepts": ["location", "location"],
                "relations": [(0, 1, "capital_of")],
                "should_respond": 1,
                "response": "Yes, London is the capital of England.",
            },
            {
                "text": "Tokyo is in Japan.",
                "entities": ["Tokyo", "Japan"],
                "entity_spans": [(0, 5), (12, 17)],
                "concepts": ["location", "location"],
                "relations": [(0, 1, "located_in")],
                "should_respond": 1,
            },
            {
                "text": "New York is a city in America.",
                "entities": ["New York", "America"],
                "entity_spans": [(0, 8), (25, 32)],
                "concepts": ["location", "location"],
                "relations": [(0, 1, "located_in")],
                "should_respond": 1,
            },
            {
                "text": "The river flows through the valley.",
                "entities": ["river", "valley"],
                "entity_spans": [(4, 9), (25, 31)],
                "concepts": ["location", "location"],
                "relations": [(0, 1, "flows_through")],
                "should_respond": 1,
            },
            {
                "text": "Mount Everest is the highest peak.",
                "entities": ["Mount Everest"],
                "entity_spans": [(0, 13)],
                "concepts": ["location"],
                "relations": [],
                "should_respond": 0,
            },
            {
                "text": "Berlin is the capital of Germany.",
                "entities": ["Berlin", "Germany"],
                "entity_spans": [(0, 6), (28, 35)],
                "concepts": ["location", "location"],
                "relations": [(0, 1, "capital_of")],
                "should_respond": 1,
            },
            {
                "text": "The ocean covers most of Earth.",
                "entities": ["ocean", "Earth"],
                "entity_spans": [(4, 9), (24, 29)],
                "concepts": ["location", "location"],
                "relations": [(0, 1, "covers")],
                "should_respond": 1,
            },
            # People and relationships
            {
                "text": "Alice loves Bob.",
                "entities": ["Alice", "Bob"],
                "entity_spans": [(0, 5), (12, 15)],
                "concepts": ["person", "person"],
                "relations": [(0, 1, "loves")],
                "should_respond": 1,
            },
            {
                "text": "The teacher taught the students.",
                "entities": ["teacher", "students"],
                "entity_spans": [(4, 11), (22, 30)],
                "concepts": ["person", "person"],
                "relations": [(0, 1, "teaches")],
                "should_respond": 1,
            },
            {
                "text": "John owns a car.",
                "entities": ["John", "car"],
                "entity_spans": [(0, 4), (11, 14)],
                "concepts": ["person", "object"],
                "relations": [(0, 1, "owns")],
                "should_respond": 1,
            },
            {
                "text": "Mary is a doctor.",
                "entities": ["Mary"],
                "entity_spans": [(0, 4)],
                "concepts": ["person"],
                "relations": [],
                "should_respond": 0,
            },
            {
                "text": "The chef cooked the meal.",
                "entities": ["chef", "meal"],
                "entity_spans": [(4, 8), (20, 24)],
                "concepts": ["person", "object"],
                "relations": [(0, 1, "cooks")],
                "should_respond": 1,
            },
            {
                "text": "Tom met Sarah at the park.",
                "entities": ["Tom", "Sarah", "park"],
                "entity_spans": [(0, 3), (8, 13), (20, 24)],
                "concepts": [["person", "male"], ["person", "female"], ["location", "public_space"]],  # Multiple concepts with attributes
                "relations": [(0, 1, "met"), (0, 2, "at")],  # Multiple relations from same entity
                "should_respond": 1,
            },
            # Objects and properties
            {
                "text": "The book is on the table.",
                "entities": ["book", "table"],
                "entity_spans": [(4, 8), (18, 23)],
                "concepts": ["object", "object"],
                "relations": [(0, 1, "on")],
                "should_respond": 1,
            },
            {
                "text": "Red is a primary color.",
                "entities": ["Red"],
                "entity_spans": [(0, 3)],
                "concepts": ["attribute"],
                "relations": [],
                "should_respond": 0,
            },
            {
                "text": "The car is fast.",
                "entities": ["car"],
                "entity_spans": [(4, 7)],
                "concepts": ["object"],
                "relations": [],
                "should_respond": 0,
            },
            {
                "text": "The apple fell from the tree.",
                "entities": ["apple", "tree"],
                "entity_spans": [(4, 9), (20, 24)],
                "concepts": ["object", "object"],
                "relations": [(0, 1, "fell_from")],
                "should_respond": 1,
            },
            {
                "text": "The computer runs software.",
                "entities": ["computer", "software"],
                "entity_spans": [(4, 12), (18, 26)],
                "concepts": ["object", "object"],
                "relations": [(0, 1, "runs")],
                "should_respond": 1,
            },
            {
                "text": "Green is my favorite color.",
                "entities": ["Green"],
                "entity_spans": [(0, 5)],
                "concepts": ["attribute"],
                "relations": [],
                "should_respond": 0,
            },
            {
                "text": "The phone rang loudly.",
                "entities": ["phone"],
                "entity_spans": [(4, 9)],
                "concepts": ["object"],
                "relations": [],
                "should_respond": 0,
            },
            # Time and events
            {
                "text": "Monday comes before Tuesday.",
                "entities": ["Monday", "Tuesday"],
                "entity_spans": [(0, 6), (22, 29)],
                "concepts": ["attribute", "attribute"],
                "relations": [(0, 1, "before")],
                "should_respond": 1,
            },
            {
                "text": "Spring follows winter.",
                "entities": ["Spring", "winter"],
                "entity_spans": [(0, 6), (14, 20)],
                "concepts": ["attribute", "attribute"],
                "relations": [(0, 1, "follows")],
                "should_respond": 1,
            },
            {
                "text": "The meeting starts at noon.",
                "entities": ["meeting", "noon"],
                "entity_spans": [(4, 11), (23, 27)],
                "concepts": ["object", "attribute"],
                "relations": [(0, 1, "starts_at")],
                "should_respond": 1,
            },
            # Abstract concepts
            {
                "text": "Happiness brings joy.",
                "entities": ["Happiness", "joy"],
                "entity_spans": [(0, 9), (17, 20)],
                "concepts": ["attribute", "attribute"],
                "relations": [(0, 1, "brings")],
                "should_respond": 1,
            },
            {
                "text": "Knowledge leads to wisdom.",
                "entities": ["Knowledge", "wisdom"],
                "entity_spans": [(0, 9), (20, 26)],
                "concepts": ["attribute", "attribute"],
                "relations": [(0, 1, "leads_to")],
                "should_respond": 1,
            },
            {
                "text": "Freedom is important.",
                "entities": ["Freedom"],
                "entity_spans": [(0, 7)],
                "concepts": ["attribute"],
                "relations": [],
                "should_respond": 0,
            },
            # More animals
            {
                "text": "The horse galloped across the field.",
                "entities": ["horse", "field"],
                "entity_spans": [(4, 9), (27, 32)],
                "concepts": ["animal", "location"],
                "relations": [(0, 1, "galloped_across")],
                "should_respond": 1,
            },
            {
                "text": "A bear lives in the forest.",
                "entities": ["bear", "forest"],
                "entity_spans": [(2, 6), (20, 26)],
                "concepts": ["animal", "location"],
                "relations": [(0, 1, "lives_in")],
                "should_respond": 1,
            },
            {
                "text": "The rabbit hopped into the garden.",
                "entities": ["rabbit", "garden"],
                "entity_spans": [(4, 10), (24, 30)],
                "concepts": ["animal", "location"],
                "relations": [(0, 1, "hopped_into")],
                "should_respond": 1,
            },
            {
                "text": "Dolphins are intelligent creatures.",
                "entities": ["Dolphins"],
                "entity_spans": [(0, 8)],
                "concepts": ["animal"],
                "relations": [],
                "should_respond": 0,
            },
            {
                "text": "The wolf howled at the moon.",
                "entities": ["wolf", "moon"],
                "entity_spans": [(4, 8), (22, 26)],
                "concepts": ["animal", "object"],
                "relations": [(0, 1, "howled_at")],
                "should_respond": 1,
            },
            # More locations
            {
                "text": "Rome is the capital of Italy.",
                "entities": ["Rome", "Italy"],
                "entity_spans": [(0, 4), (28, 33)],
                "concepts": ["location", "location"],
                "relations": [(0, 1, "capital_of")],
                "should_respond": 1,
            },
            {
                "text": "Madrid is in Spain.",
                "entities": ["Madrid", "Spain"],
                "entity_spans": [(0, 6), (13, 18)],
                "concepts": ["location", "location"],
                "relations": [(0, 1, "located_in")],
                "should_respond": 1,
            },
            {
                "text": "The bridge crosses the river.",
                "entities": ["bridge", "river"],
                "entity_spans": [(4, 10), (22, 27)],
                "concepts": ["object", "location"],
                "relations": [(0, 1, "crosses")],
                "should_respond": 1,
            },
            {
                "text": "The road leads to the mountain.",
                "entities": ["road", "mountain"],
                "entity_spans": [(4, 8), (22, 30)],
                "concepts": ["object", "location"],
                "relations": [(0, 1, "leads_to")],
                "should_respond": 1,
            },
            {
                "text": "The desert is very hot.",
                "entities": ["desert"],
                "entity_spans": [(4, 10)],
                "concepts": ["location"],
                "relations": [],
                "should_respond": 0,
            },
            {
                "text": "The island is surrounded by water.",
                "entities": ["island", "water"],
                "entity_spans": [(4, 10), (28, 33)],
                "concepts": ["location", "object"],
                "relations": [(0, 1, "surrounded_by")],
                "should_respond": 1,
            },
            # More people
            {
                "text": "The artist painted a picture.",
                "entities": ["artist", "picture"],
                "entity_spans": [(4, 10), (22, 29)],
                "concepts": ["person", "object"],
                "relations": [(0, 1, "painted")],
                "should_respond": 1,
            },
            {
                "text": "Emma wrote a letter to her friend.",
                "entities": ["Emma", "letter", "friend"],
                "entity_spans": [(0, 4), (14, 20), (27, 33)],
                "concepts": ["person", "object", "person"],
                "relations": [(0, 1, "wrote"), (1, 2, "to")],
                "should_respond": 1,
            },
            {
                "text": "The scientist discovered a new element.",
                "entities": ["scientist", "element"],
                "entity_spans": [(4, 13), (30, 37)],
                "concepts": ["person", "object"],
                "relations": [(0, 1, "discovered")],
                "should_respond": 1,
            },
            {
                "text": "David plays the guitar.",
                "entities": ["David", "guitar"],
                "entity_spans": [(0, 5), (18, 24)],
                "concepts": ["person", "object"],
                "relations": [(0, 1, "plays")],
                "should_respond": 1,
            },
            {
                "text": "The nurse helped the patient.",
                "entities": ["nurse", "patient"],
                "entity_spans": [(4, 9), (20, 27)],
                "concepts": ["person", "person"],
                "relations": [(0, 1, "helped")],
                "should_respond": 1,
            },
            {
                "text": "Lisa is a talented musician.",
                "entities": ["Lisa"],
                "entity_spans": [(0, 4)],
                "concepts": ["person"],
                "relations": [],
                "should_respond": 0,
            },
            # More objects
            {
                "text": "The key opens the door.",
                "entities": ["key", "door"],
                "entity_spans": [(4, 7), (18, 22)],
                "concepts": ["object", "object"],
                "relations": [(0, 1, "opens")],
                "should_respond": 1,
            },
            {
                "text": "The pen writes on paper.",
                "entities": ["pen", "paper"],
                "entity_spans": [(4, 7), (20, 25)],
                "concepts": ["object", "object"],
                "relations": [(0, 1, "writes_on")],
                "should_respond": 1,
            },
            {
                "text": "The lamp lights the room.",
                "entities": ["lamp", "room"],
                "entity_spans": [(4, 8), (20, 24)],
                "concepts": ["object", "location"],
                "relations": [(0, 1, "lights")],
                "should_respond": 1,
            },
            {
                "text": "The clock shows the time.",
                "entities": ["clock", "time"],
                "entity_spans": [(4, 9), (20, 24)],
                "concepts": ["object", "attribute"],
                "relations": [(0, 1, "shows")],
                "should_respond": 1,
            },
            {
                "text": "The window faces the street.",
                "entities": ["window", "street"],
                "entity_spans": [(4, 10), (22, 28)],
                "concepts": ["object", "location"],
                "relations": [(0, 1, "faces")],
                "should_respond": 1,
            },
            {
                "text": "The mirror reflects light.",
                "entities": ["mirror", "light"],
                "entity_spans": [(4, 10), (20, 25)],
                "concepts": ["object", "object"],
                "relations": [(0, 1, "reflects")],
                "should_respond": 1,
            },
            # Attributes and properties
            {
                "text": "Yellow is bright and cheerful.",
                "entities": ["Yellow"],
                "entity_spans": [(0, 6)],
                "concepts": ["attribute"],
                "relations": [],
                "should_respond": 0,
            },
            {
                "text": "The weather is sunny today.",
                "entities": ["weather"],
                "entity_spans": [(4, 11)],
                "concepts": ["attribute"],
                "relations": [],
                "should_respond": 0,
            },
            {
                "text": "Speed is measured in kilometers per hour.",
                "entities": ["Speed"],
                "entity_spans": [(0, 5)],
                "concepts": ["attribute"],
                "relations": [],
                "should_respond": 0,
            },
            # Complex relations
            {
                "text": "The student studied for the exam.",
                "entities": ["student", "exam"],
                "entity_spans": [(4, 11), (26, 30)],
                "concepts": ["person", "object"],
                "relations": [(0, 1, "studied_for")],
                "should_respond": 1,
            },
            {
                "text": "The farmer grows crops in the field.",
                "entities": ["farmer", "crops", "field"],
                "entity_spans": [(4, 10), (17, 22), (30, 35)],
                "concepts": ["person", "object", "location"],
                "relations": [(0, 1, "grows"), (1, 2, "in")],
                "should_respond": 1,
            },
            {
                "text": "The pilot flew the plane to Paris.",
                "entities": ["pilot", "plane", "Paris"],
                "entity_spans": [(4, 9), (18, 23), (27, 32)],
                "concepts": ["person", "object", "location"],
                "relations": [(0, 1, "flew"), (1, 2, "to")],
                "should_respond": 1,
            },
            {
                "text": "The baker made bread in the kitchen.",
                "entities": ["baker", "bread", "kitchen"],
                "entity_spans": [(4, 9), (14, 19), (27, 34)],
                "concepts": ["person", "object", "location"],
                "relations": [(0, 1, "made"), (1, 2, "in")],
                "should_respond": 1,
            },
            {
                "text": "The writer published a book about history.",
                "entities": ["writer", "book", "history"],
                "entity_spans": [(4, 10), (22, 26), (34, 41)],
                "concepts": ["person", "object", "attribute"],
                "relations": [(0, 1, "published"), (1, 2, "about")],
                "should_respond": 1,
            },
            # More animals with actions
            {
                "text": "The spider spun a web.",
                "entities": ["spider", "web"],
                "entity_spans": [(4, 10), (18, 21)],
                "concepts": ["animal", "object"],
                "relations": [(0, 1, "spun")],
                "should_respond": 1,
            },
            {
                "text": "The bee collected nectar from flowers.",
                "entities": ["bee", "nectar", "flowers"],
                "entity_spans": [(4, 7), (16, 22), (28, 35)],
                "concepts": ["animal", "object", "object"],
                "relations": [(0, 1, "collected"), (1, 2, "from")],
                "should_respond": 1,
            },
            {
                "text": "The penguin lives in Antarctica.",
                "entities": ["penguin", "Antarctica"],
                "entity_spans": [(4, 11), (22, 32)],
                "concepts": ["animal", "location"],
                "relations": [(0, 1, "lives_in")],
                "should_respond": 1,
            },
            {
                "text": "The tiger is an endangered species.",
                "entities": ["tiger"],
                "entity_spans": [(4, 9)],
                "concepts": ["animal"],
                "relations": [],
                "should_respond": 0,
            },
            {
                "text": "The whale swam in the ocean.",
                "entities": ["whale", "ocean"],
                "entity_spans": [(4, 9), (22, 27)],
                "concepts": ["animal", "location"],
                "relations": [(0, 1, "swam_in")],
                "should_respond": 1,
            },
            # More locations
            {
                "text": "The village is near the mountain.",
                "entities": ["village", "mountain"],
                "entity_spans": [(4, 11), (24, 32)],
                "concepts": ["location", "location"],
                "relations": [(0, 1, "near")],
                "should_respond": 1,
            },
            {
                "text": "The castle stands on the hill.",
                "entities": ["castle", "hill"],
                "entity_spans": [(4, 10), (24, 28)],
                "concepts": ["location", "location"],
                "relations": [(0, 1, "stands_on")],
                "should_respond": 1,
            },
            {
                "text": "The airport is outside the city.",
                "entities": ["airport", "city"],
                "entity_spans": [(4, 11), (26, 30)],
                "concepts": ["location", "location"],
                "relations": [(0, 1, "outside")],
                "should_respond": 1,
            },
            {
                "text": "The library contains many books.",
                "entities": ["library", "books"],
                "entity_spans": [(4, 11), (22, 27)],
                "concepts": ["location", "object"],
                "relations": [(0, 1, "contains")],
                "should_respond": 1,
            },
            {
                "text": "The museum displays ancient artifacts.",
                "entities": ["museum", "artifacts"],
                "entity_spans": [(4, 10), (24, 33)],
                "concepts": ["location", "object"],
                "relations": [(0, 1, "displays")],
                "should_respond": 1,
            },
            {
                "text": "The beach stretches along the coast.",
                "entities": ["beach", "coast"],
                "entity_spans": [(4, 9), (26, 31)],
                "concepts": ["location", "location"],
                "relations": [(0, 1, "stretches_along")],
                "should_respond": 1,
            },
            # More people interactions
            {
                "text": "The parent read a story to the child.",
                "entities": ["parent", "story", "child"],
                "entity_spans": [(4, 10), (16, 21), (28, 33)],
                "concepts": ["person", "object", "person"],
                "relations": [(0, 1, "read"), (1, 2, "to")],
                "should_respond": 1,
            },
            {
                "text": "The coach trained the athletes.",
                "entities": ["coach", "athletes"],
                "entity_spans": [(4, 9), (20, 28)],
                "concepts": ["person", "person"],
                "relations": [(0, 1, "trained")],
                "should_respond": 1,
            },
            {
                "text": "The judge decided the case.",
                "entities": ["judge", "case"],
                "entity_spans": [(4, 9), (20, 24)],
                "concepts": ["person", "object"],
                "relations": [(0, 1, "decided")],
                "should_respond": 1,
            },
            {
                "text": "The mechanic fixed the car engine.",
                "entities": ["mechanic", "car", "engine"],
                "entity_spans": [(4, 12), (20, 23), (24, 30)],
                "concepts": ["person", "object", "object"],
                "relations": [(0, 1, "fixed"), (1, 2, "has")],
                "should_respond": 1,
            },
            {
                "text": "The photographer captured the moment.",
                "entities": ["photographer", "moment"],
                "entity_spans": [(4, 16), (28, 34)],
                "concepts": ["person", "attribute"],
                "relations": [(0, 1, "captured")],
                "should_respond": 1,
            },
            {
                "text": "The architect designed the building.",
                "entities": ["architect", "building"],
                "entity_spans": [(4, 13), (25, 33)],
                "concepts": ["person", "object"],
                "relations": [(0, 1, "designed")],
                "should_respond": 1,
            },
            # More objects and interactions
            {
                "text": "The hammer hit the nail.",
                "entities": ["hammer", "nail"],
                "entity_spans": [(4, 10), (18, 22)],
                "concepts": ["object", "object"],
                "relations": [(0, 1, "hit")],
                "should_respond": 1,
            },
            {
                "text": "The scissors cut the paper.",
                "entities": ["scissors", "paper"],
                "entity_spans": [(4, 11), (20, 25)],
                "concepts": ["object", "object"],
                "relations": [(0, 1, "cut")],
                "should_respond": 1,
            },
            {
                "text": "The brush painted the wall.",
                "entities": ["brush", "wall"],
                "entity_spans": [(4, 9), (22, 26)],
                "concepts": ["object", "object"],
                "relations": [(0, 1, "painted")],
                "should_respond": 1,
            },
            {
                "text": "The umbrella protects from rain.",
                "entities": ["umbrella", "rain"],
                "entity_spans": [(4, 12), (28, 32)],
                "concepts": ["object", "attribute"],
                "relations": [(0, 1, "protects_from")],
                "should_respond": 1,
            },
            {
                "text": "The battery powers the device.",
                "entities": ["battery", "device"],
                "entity_spans": [(4, 11), (22, 28)],
                "concepts": ["object", "object"],
                "relations": [(0, 1, "powers")],
                "should_respond": 1,
            },
            {
                "text": "The rope tied the boat to the dock.",
                "entities": ["rope", "boat", "dock"],
                "entity_spans": [(4, 8), (18, 22), (29, 33)],
                "concepts": ["object", "object", "location"],
                "relations": [(0, 1, "tied"), (1, 2, "to")],
                "should_respond": 1,
            },
            # Attributes and abstract
            {
                "text": "Purple is a royal color.",
                "entities": ["Purple"],
                "entity_spans": [(0, 6)],
                "concepts": ["attribute"],
                "relations": [],
                "should_respond": 0,
            },
            {
                "text": "The temperature dropped significantly.",
                "entities": ["temperature"],
                "entity_spans": [(4, 15)],
                "concepts": ["attribute"],
                "relations": [],
                "should_respond": 0,
            },
            {
                "text": "Beauty is in the eye of the beholder.",
                "entities": ["Beauty"],
                "entity_spans": [(0, 6)],
                "concepts": ["attribute"],
                "relations": [],
                "should_respond": 0,
            },
            {
                "text": "Patience is a virtue.",
                "entities": ["Patience"],
                "entity_spans": [(0, 8)],
                "concepts": ["attribute"],
                "relations": [],
                "should_respond": 0,
            },
            {
                "text": "The sound echoed through the cave.",
                "entities": ["sound", "cave"],
                "entity_spans": [(4, 9), (25, 29)],
                "concepts": ["attribute", "location"],
                "relations": [(0, 1, "echoed_through")],
                "should_respond": 1,
            },
            # Final diverse samples
            {
                "text": "The storm damaged the roof.",
                "entities": ["storm", "roof"],
                "entity_spans": [(4, 9), (20, 24)],
                "concepts": ["attribute", "object"],
                "relations": [(0, 1, "damaged")],
                "should_respond": 1,
            },
            {
                "text": "The fire burned the wood.",
                "entities": ["fire", "wood"],
                "entity_spans": [(4, 8), (20, 24)],
                "concepts": ["attribute", "object"],
                "relations": [(0, 1, "burned")],
                "should_respond": 1,
            },
            {
                "text": "The wind blew the leaves.",
                "entities": ["wind", "leaves"],
                "entity_spans": [(4, 8), (18, 24)],
                "concepts": ["attribute", "object"],
                "relations": [(0, 1, "blew")],
                "should_respond": 1,
            },
            {
                "text": "The sun shines during the day.",
                "entities": ["sun", "day"],
                "entity_spans": [(4, 7), (24, 27)],
                "concepts": ["object", "attribute"],
                "relations": [(0, 1, "shines_during")],
                "should_respond": 1,
            },
            {
                "text": "The moon appears at night.",
                "entities": ["moon", "night"],
                "entity_spans": [(4, 8), (20, 25)],
                "concepts": ["object", "attribute"],
                "relations": [(0, 1, "appears_at")],
                "should_respond": 1,
            },
            {
                "text": "The star twinkled in the sky.",
                "entities": ["star", "sky"],
                "entity_spans": [(4, 8), (25, 28)],
                "concepts": ["object", "location"],
                "relations": [(0, 1, "twinkled_in")],
                "should_respond": 1,
            },
            # Examples with multiple concepts per entity
            {
                "text": "The doctor treated the patient in the hospital.",
                "entities": ["doctor", "patient", "hospital"],
                "entity_spans": [(4, 10), (19, 26), (34, 42)],
                "concepts": [["person", "professional", "medical"], ["person", "medical"], ["location", "medical", "building"]],
                "relations": [(0, 1, "treats"), (0, 2, "works_in"), (1, 2, "located_in")],
                "should_respond": 1,
            },
            {
                "text": "The teacher taught mathematics to students.",
                "entities": ["teacher", "mathematics", "students"],
                "entity_spans": [(4, 11), (18, 28), (32, 40)],
                "concepts": [["person", "professional", "educator"], ["attribute", "subject", "academic"], ["person", "learner"]],
                "relations": [(0, 1, "teaches"), (0, 2, "teaches")],
                "should_respond": 1,
            },
            {
                "text": "The library contains books and computers.",
                "entities": ["library", "books", "computers"],
                "entity_spans": [(4, 11), (20, 25), (30, 39)],
                "concepts": [["location", "building", "public_space"], ["object", "knowledge"], ["object", "technology", "electronic"]],
                "relations": [(0, 1, "contains"), (0, 2, "contains")],
                "should_respond": 1,
            },
        ]
        
        # Normalize concepts: convert single strings to lists for backward compatibility
        for entry in self.data:
            if "concepts" in entry:
                normalized_concepts = []
                for concept in entry["concepts"]:
                    if isinstance(concept, str):
                        normalized_concepts.append([concept])
                    elif isinstance(concept, list):
                        normalized_concepts.append(concept)
                    else:
                        normalized_concepts.append([str(concept)])
                entry["concepts"] = normalized_concepts
        
        # Process entries: convert should_respond=1 statements to should_respond=0
        # and create question versions with should_respond=1
        processed_data = []
        for entry in self.data:
            if entry.get("should_respond", 0) == 1:
                # Create statement version with should_respond=0 (remove response)
                statement_entry = entry.copy()
                statement_entry["should_respond"] = 0
                if "response" in statement_entry:
                    del statement_entry["response"]
                processed_data.append(statement_entry)
                
                # Create question version with should_respond=1
                question_text = statement_to_question(entry["text"])
                question_entry = entry.copy()
                question_entry["text"] = question_text
                
                # Recalculate entity spans for question
                question_entry["entity_spans"] = recalculate_entity_spans(
                    question_text, entry["entities"], entry.get("entity_spans")
                )
                
                # Ensure response exists (use existing or generate)
                if "response" not in question_entry or not question_entry.get("response"):
                    question_entry["response"] = generate_response_text(entry["text"], entry["entities"], entry["relations"])
                
                question_entry["should_respond"] = 1
                processed_data.append(question_entry)
            else:
                # Keep entries with should_respond=0 as-is
                processed_data.append(entry)
        
        self.data = processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ------------------------
# Collator
# ------------------------

class CognitiveCollator:
    def __init__(self, tokenizer, concept_map, relation_map, include_responses=False):
        self.tokenizer = tokenizer
        self.concept_map = concept_map
        self.relation_map = relation_map
        self.include_responses = include_responses
        # Get number of concepts for multi-label encoding
        self.n_concepts = max(concept_map.values()) if concept_map else 0

    def __call__(self, batch):
        texts = [x["text"] for x in batch]
        tok = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        max_entities = max(len(x["entities"]) for x in batch)
        entity_ids = torch.zeros(len(batch), max_entities, dtype=torch.long)
        # Multi-label concept encoding: (B, max_entities, n_concepts) binary matrix
        concept_labels = torch.zeros(len(batch), max_entities, self.n_concepts, dtype=torch.float)

        relation_triplets = []
        # Create should_respond tensor, ensuring it's always a tensor
        should_respond_values = [int(x.get("should_respond", 0)) for x in batch]
        should_respond = torch.tensor(should_respond_values, dtype=torch.long)

        # Tokenize response texts for decoder training
        decoder_input_ids = None
        decoder_labels = None
        if self.include_responses:
            responses = []
            should_respond_mask = []
            for x in batch:
                if x.get("should_respond", 0) == 1 and "response" in x:
                    responses.append(x["response"])
                    should_respond_mask.append(True)
                else:
                    # For samples that don't require a response, use single EOS token
                    # Get EOS token - try eos_token first, fallback to sep_token or pad_token
                    eos_token = self.tokenizer.eos_token
                    if eos_token is None:
                        eos_token = self.tokenizer.sep_token
                    if eos_token is None:
                        eos_token = self.tokenizer.pad_token
                    if eos_token is None:
                        # Last resort: use a special token string that should exist
                        eos_token = "</s>"
                    responses.append(eos_token)
                    should_respond_mask.append(False)
            
            # Tokenize responses
            resp_tok = self.tokenizer(
                responses, 
                padding=True, 
                truncation=True, 
                max_length=128,  # Limit response length
                return_tensors="pt"
            )
            decoder_input_ids = resp_tok["input_ids"]
            decoder_labels = resp_tok["input_ids"].clone()
            
            # Get EOS token ID for handling non-response samples
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is None:
                eos_token_id = self.tokenizer.sep_token_id
            if eos_token_id is None:
                eos_token_id = self.tokenizer.pad_token_id
            
            # Set padding tokens to -100 (ignore in loss)
            decoder_labels[resp_tok["attention_mask"] == 0] = -100
            
            # For samples that don't require a response, set labels so only EOS token is target
            # With next-token prediction (shift_labels = labels[:, 1:]), we need:
            # - labels at position 0: ignored (we predict what comes after position 0)
            # - labels at position 1: EOS (we want to predict EOS at position 0 after shifting)
            # So labels should be: [-100, EOS, -100, ...]
            for i, should_respond in enumerate(should_respond_mask):
                if not should_respond:
                    # Set all labels to -100 first
                    decoder_labels[i] = -100
                    # Find EOS token in the sequence
                    if eos_token_id is not None:
                        # Find all EOS token positions
                        eos_positions = (decoder_input_ids[i] == eos_token_id).nonzero(as_tuple=True)[0]
                        if len(eos_positions) > 0:
                            # We want to predict EOS, so set label at position 1 to EOS
                            # After shifting (labels[:, 1:]), this becomes the target at position 0
                            if decoder_labels.shape[1] > 1:
                                decoder_labels[i, 1] = eos_token_id
                            else:
                                # If sequence is too short, set at position 0
                                decoder_labels[i, 0] = eos_token_id
                        else:
                            # If no EOS token found, find the first non-padding token and use it
                            non_pad_mask = resp_tok["attention_mask"][i] == 1
                            if non_pad_mask.any():
                                first_token_pos = non_pad_mask.nonzero(as_tuple=True)[0][0]
                                first_token_id = decoder_input_ids[i, first_token_pos]
                                # Set label at position 1 to predict the first token
                                if decoder_labels.shape[1] > 1:
                                    decoder_labels[i, 1] = first_token_id
                                else:
                                    decoder_labels[i, 0] = first_token_id

        for i, sample in enumerate(batch):
            ents = sample["entities"]
            # Fix entity ID assignment: use actual entity indices (1-indexed, 0 is padding)
            for j, ent in enumerate(ents):
                entity_ids[i, j] = j + 1  # Entity index within sample (1-indexed)
                
                # Handle multiple concepts per entity
                # Support both old format (single string) and new format (list of strings)
                entity_concepts = sample["concepts"][j]
                if isinstance(entity_concepts, str):
                    # Backward compatibility: single concept as string
                    entity_concepts = [entity_concepts]
                elif not isinstance(entity_concepts, list):
                    # Fallback: convert to list
                    entity_concepts = [entity_concepts]
                
                # Set binary labels for all concepts associated with this entity
                for concept_name in entity_concepts:
                    concept_val = self.concept_map.get(concept_name, 0)
                    if concept_val > 0:
                        # concept_map is 1-indexed, convert to 0-indexed for tensor indexing
                        concept_idx = concept_val - 1
                        concept_labels[i, j, concept_idx] = 1.0
                
            # Pad remaining slots with 0 (padding)
            for j in range(len(ents), max_entities):
                entity_ids[i, j] = 0
                # concept_labels already initialized to zeros
            
            rels = sample["relations"]
            rel_tensor = torch.zeros((len(rels), 3), dtype=torch.long)
            for k, (h, t, r) in enumerate(rels):
                rel_tensor[k] = torch.tensor([h, t, self.relation_map.get(r, 0)])
            relation_triplets.append(rel_tensor)

        result = {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "entity_ids": entity_ids,
            "concept_labels": concept_labels,  # Multi-label binary matrix (B, max_entities, n_concepts)
            "relations": relation_triplets,
            "should_respond": should_respond,
        }
        
        if decoder_input_ids is not None:
            result["decoder_input_ids"] = decoder_input_ids
            result["decoder_labels"] = decoder_labels
        
        return result


# ------------------------
# Tiered Training Scaffold
# ------------------------

# Stages:
# 1. Pretrain encoder on MLM
# 2. Train entity extractor, concept mapper, relation head
# 3. Train cognitive fusion and response controller
# 4. Joint finetune full stack

from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

class Stage1_MLM_Trainer:
    def __init__(self, model, tokenizer, optimizer):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.loss_fct = nn.CrossEntropyLoss()

    def train_step(self, batch):
        # Skip if using pre-trained encoder (no MLM head)
        if self.model.mlm_head is None:
            return 0.0
            
        self.optimizer.zero_grad()
        labels = batch["input_ids"].clone()
        masked = batch["input_ids"].clone()
        attention_mask = batch["attention_mask"]
        
        # Create random mask positions (15% masking)
        rand = torch.rand(masked.shape, device=masked.device)
        mask_positions = (rand < 0.15) & (attention_mask == 1)
        labels[~mask_positions] = -100  # Ignore non-masked tokens in loss
        
        masked[mask_positions] = self.tokenizer.mask_token_id
        
        # Forward through encoder and MLM head
        enc = self.model.encode(masked, attention_mask)
        logits = self.model.mlm_head(enc)  # (B, L, vocab_size)
        
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        self.optimizer.step()
        return loss.item()


def configure_soft_logic_rules(model, concept_to_entity_type_map, relation_map, rules_config):
    """
    Configure soft logic rules for the model.
    
    Args:
        model: NeuroSymbolicLM model
        concept_to_entity_type_map: dict mapping concept names to entity type indices
            e.g., {"animal": 0, "person": 1, "location": 2, "object": 3, "attribute": 4}
        relation_map: dict mapping relation names to relation indices
            e.g., {"chases": 0, "capital_of": 1, "loves": 2, ...}
        rules_config: list of rule dicts, each with:
            - "concept_a": concept name for first entity type
            - "concept_b": concept name for second entity type  
            - "relation": relation name
            - "weight": float weight for the rule (default 1.0)
            - "polarity": 1 to encourage relation, -1 to discourage (default 1)
    
    Example:
        rules_config = [
            {"concept_a": "animal", "concept_b": "animal", "relation": "chases", "weight": 1.0, "polarity": 1},
            {"concept_a": "location", "concept_b": "location", "relation": "capital_of", "weight": 1.0, "polarity": 1},
        ]
    """
    for rule in rules_config:
        etype_a = concept_to_entity_type_map.get(rule["concept_a"])
        etype_b = concept_to_entity_type_map.get(rule["concept_b"])
        rel_idx = relation_map.get(rule["relation"])
        
        if etype_a is None or etype_b is None or rel_idx is None:
            print(f"Warning: Skipping rule {rule} - invalid concept or relation mapping")
            continue
        
        weight = rule.get("weight", 1.0)
        polarity = rule.get("polarity", 1)
        
        model.softlogic.add_rule(etype_a, etype_b, rel_idx, weight, polarity)
    
    print(f"Configured {len(model.softlogic.rules)} soft logic rules")


class Stage2_Symbolic_Trainer:
    def __init__(self, model, optimizer, soft_logic_weight=0.1):
        self.model = model
        self.optimizer = optimizer
        self.ce = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding (0)
        self.bce = nn.BCEWithLogitsLoss()  # For multi-label concept classification
        self.soft_logic_weight = soft_logic_weight

    def train_step(self, batch):
        self.optimizer.zero_grad()
        # Forward through full model to get all outputs including soft logic inputs
        out = self.model(
            batch["input_ids"], 
            batch["attention_mask"], 
            spans=None,
            y_ids=None
        )
        
        enc = self.model.encode(batch["input_ids"], batch["attention_mask"])
        
        # Entity classification: token-level predictions
        ent_logits = out["token_ent_logits"]  # (B, L, n_entity_types)
        # For entity loss, we need to map entity_ids to token positions
        # Since entity_ids are per-entity (not per-token), we'll use a simplified approach:
        # Use the first token of each entity span for supervision
        # For now, use a pooled representation approach
        B, L, E = ent_logits.shape
        max_entities = batch["entity_ids"].shape[1]
        
        # Create token-level entity labels (simplified: use entity_ids for first entity tokens)
        # This is a simplified approach - in production, you'd use actual span alignments
        entity_token_labels = torch.zeros(B, L, dtype=torch.long, device=enc.device)
        for i in range(B):
            num_ents = (batch["entity_ids"][i] > 0).sum().item()
            # Assign entity labels to first few tokens (simplified)
            for j in range(min(num_ents, L)):
                entity_token_labels[i, j] = batch["entity_ids"][i, j]
        
        ent_loss = self.ce(ent_logits.view(-1, E), entity_token_labels.view(-1))
        
        # Concept classification: multi-label prediction per entity
        # Get concept labels: (B, max_entities, n_concept_types) binary matrix
        concept_labels = batch["concept_labels"]  # (B, max_entities, n_concept_types)
        n_concept_types = concept_labels.shape[2]
        
        # For each entity, predict concepts using entity representations
        # Use entity-level representations: pool encoder outputs for each entity
        # Simplified: use mean pooling of encoder outputs as entity representation
        pooled_enc = enc.mean(dim=1)  # (B, d_model)
        concept_logits = self.model.concept_head(pooled_enc)  # (B, n_concepts_bank) where n_concepts_bank=512
        n_concepts_bank = concept_logits.shape[1]
        
        # For multi-label loss, we need to aggregate concept labels across entities
        # Option 1: Use first entity's concepts (simplified for now)
        # Option 2: Average concept labels across all entities in the sample
        # Option 3: Predict concepts for each entity separately (requires entity-level pooling)
        # Using Option 2: average concept labels across entities (weighted by entity presence)
        entity_mask = (batch["entity_ids"] > 0).float()  # (B, max_entities)
        # Average concept labels across entities (only for valid entities)
        num_valid_entities = entity_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        aggregated_concept_labels = (concept_labels * entity_mask.unsqueeze(-1)).sum(dim=1) / num_valid_entities  # (B, n_concept_types)
        
        # Map concept labels to concept bank size
        if n_concept_types == n_concepts_bank:
            target_labels = aggregated_concept_labels
        elif n_concept_types < n_concepts_bank:
            # Pad with zeros
            padding = torch.zeros(aggregated_concept_labels.shape[0], n_concepts_bank - n_concept_types, 
                                 device=aggregated_concept_labels.device, dtype=aggregated_concept_labels.dtype)
            target_labels = torch.cat([aggregated_concept_labels, padding], dim=1)
        else:
            # Truncate
            target_labels = aggregated_concept_labels[:, :n_concepts_bank]
        
        # Multi-label binary cross-entropy loss
        con_loss = self.bce(concept_logits, target_labels)
        
        # Soft logic constraint loss
        soft_logic_loss = torch.tensor(0.0, device=enc.device)
        if len(self.model.softlogic.rules) > 0 and "node_entity_type_probs" in out and "rel_logits_matrix" in out:
            soft_logic_loss, _ = self.model.softlogic(
                out["node_entity_type_probs"], 
                out["rel_logits_matrix"]
            )
        
        loss = ent_loss + con_loss + self.soft_logic_weight * soft_logic_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Stage3_Decoder_Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def train_step(self, batch):
        self.optimizer.zero_grad()
        # Forward through model with decoder inputs for teacher forcing
        if "decoder_input_ids" not in batch:
            # Skip if no decoder inputs available
            return 0.0
        
        out = self.model(
            batch["input_ids"], 
            batch["attention_mask"], 
            spans=None,
            y_ids=batch["decoder_input_ids"]  # Teacher forcing
        )
        
        # Decoder language modeling loss
        logits = out["logits"]  # (B, T, vocab_size)
        labels = batch["decoder_labels"]  # (B, T), -100 for padding/ignore
        
        # Shift labels for next-token prediction (standard LM training)
        # The decoder already receives the full sequence, so we predict next tokens
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, vocab_size)
        shift_labels = labels[:, 1:].contiguous()  # (B, T-1)
        
        decoder_loss = self.ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        decoder_loss.backward()
        self.optimizer.step()
        return decoder_loss.item()


class Stage3_Control_Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.bce = nn.BCEWithLogitsLoss()

    def train_step(self, batch):
        self.optimizer.zero_grad()
        # Forward through full model to get controller logits
        out = self.model(
            batch["input_ids"], 
            batch["attention_mask"], 
            spans=None,  # No gold spans in this stage
            y_ids=None
        )
        # Controller outputs 3 logits: [answer, abstain, ask_clarify]
        # Use answer logit (index 0) for should_respond prediction
        answer_logit = out["controller_logits"][:, 0]  # (B,)
        # Ensure should_respond is a tensor
        should_respond = batch["should_respond"]
        if not isinstance(should_respond, torch.Tensor):
            should_respond = torch.tensor(should_respond, dtype=torch.float, device=answer_logit.device)
        else:
            should_respond = should_respond.float()
        loss = self.bce(answer_logit, should_respond)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def evaluate_generation(model, tokenizer, dataset, device="cpu", num_samples=5, max_length=128):
    """
    Evaluate model's generation capabilities on a subset of the dataset.
    
    Args:
        model: Trained NeuroSymbolicLM model
        tokenizer: Tokenizer for encoding/decoding
        dataset: Dataset to evaluate on
        device: Device to run evaluation on
        num_samples: Number of samples to evaluate
        max_length: Maximum generation length
        
    Returns:
        results: List of dicts with input, generated, and target texts
    """
    model.eval()
    results = []
    
    # Get samples that should have responses
    samples_with_responses = [s for s in dataset if s.get("should_respond", 0) == 1 and "response" in s]
    eval_samples = samples_with_responses[:num_samples]
    
    if len(eval_samples) == 0:
        print("  No samples with responses found for evaluation")
        return results
    
    # Get token IDs for special tokens
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    if bos_token_id is None:
        bos_token_id = 1  # Default fallback
    if eos_token_id is None:
        eos_token_id = 2  # Default fallback
    
    with torch.no_grad():
        for sample in eval_samples:
            # Tokenize input
            input_text = sample["text"]
            tokenized = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            
            # Generate response
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                temperature=1.0,
                do_sample=False  # Use greedy decoding for evaluation
            )
            
            # Decode generated text
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            target_text = sample.get("response", "")
            
            results.append({
                "input": input_text,
                "generated": generated_text,
                "target": target_text
            })
    
    model.train()
    return results


def print_generation_results(results, num_to_print=3):
    """Print generation evaluation results."""
    if len(results) == 0:
        return
    
    print(f"\n  Generation Evaluation (showing {min(num_to_print, len(results))} samples):")
    print("  " + "-" * 58)
    for i, result in enumerate(results[:num_to_print]):
        print(f"  Sample {i+1}:")
        print(f"    Input:    {result['input'][:60]}...")
        print(f"    Target:   {result['target'][:60]}...")
        print(f"    Generated: {result['generated'][:60]}...")
        print()


class Stage4_Joint_Trainer:
    def __init__(self, model, optimizer, soft_logic_weight=0.1):
        self.model = model
        self.optimizer = optimizer
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.ce_ignore_neg100 = nn.CrossEntropyLoss(ignore_index=-100)
        self.bce = nn.BCEWithLogitsLoss()
        self.soft_logic_weight = soft_logic_weight

    def train_step(self, batch):
        self.optimizer.zero_grad()
        
        # Use decoder inputs if available, otherwise None
        decoder_input_ids = batch.get("decoder_input_ids", None)
        
        out = self.model(
            batch["input_ids"], 
            batch["attention_mask"], 
            spans=None,  # Could use gold spans if available
            y_ids=decoder_input_ids
        )
        
        # Entity loss: token-level classification
        B, L, E = out["entity_logits"].shape
        entity_token_labels = torch.zeros(B, L, dtype=torch.long, device=out["entity_logits"].device)
        for i in range(B):
            num_ents = (batch["entity_ids"][i] > 0).sum().item()
            for j in range(min(num_ents, L)):
                entity_token_labels[i, j] = batch["entity_ids"][i, j]
        ent_loss = self.ce(out["entity_logits"].view(-1, E), entity_token_labels.view(-1))
        
        # Concept loss: multi-label classification
        # concept_probs is (B, n_concepts) from softmax over concept bank
        # The concept bank has n_concepts=512, but concept_labels uses concept_map indices
        # We need to map concept_labels to match the concept bank size
        concept_probs = out["concept_probs"]  # (B, n_concepts_bank) where n_concepts_bank=512
        n_concepts_bank = concept_probs.shape[1]
        
        # Get aggregated concept labels (average across entities)
        concept_labels = batch["concept_labels"]  # (B, max_entities, n_concept_types)
        n_concept_types = concept_labels.shape[2]
        
        # Aggregate across entities first
        entity_mask = (batch["entity_ids"] > 0).float()  # (B, max_entities)
        num_valid_entities = entity_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        aggregated_concept_labels = (concept_labels * entity_mask.unsqueeze(-1)).sum(dim=1) / num_valid_entities  # (B, n_concept_types)
        
        # Map concept labels to concept bank indices
        # For now, we'll use a simplified approach: create a mapping tensor
        # Since concept_labels uses concept_map indices (1-indexed, converted to 0-indexed in tensor)
        # and concept_probs uses concept bank indices (0-511), we need to align them
        # Simplified: use aggregated_concept_labels as-is if dimensions match, otherwise pad/truncate
        if n_concept_types == n_concepts_bank:
            target_labels = aggregated_concept_labels
        elif n_concept_types < n_concepts_bank:
            # Pad with zeros
            padding = torch.zeros(aggregated_concept_labels.shape[0], n_concepts_bank - n_concept_types, 
                                 device=aggregated_concept_labels.device, dtype=aggregated_concept_labels.dtype)
            target_labels = torch.cat([aggregated_concept_labels, padding], dim=1)
        else:
            # Truncate
            target_labels = aggregated_concept_labels[:, :n_concepts_bank]
        
        # Convert probabilities to logits for BCE loss
        eps = 1e-8
        concept_logits = torch.log(concept_probs + eps) - torch.log(1 - concept_probs + eps)
        
        # Multi-label binary cross-entropy loss
        con_loss = self.bce(concept_logits, target_labels)
        
        # Response controller loss
        # Ensure should_respond is a tensor
        should_respond = batch["should_respond"]
        if not isinstance(should_respond, torch.Tensor):
            should_respond = torch.tensor(should_respond, dtype=torch.float, device=out["response_logit"].device)
        else:
            should_respond = should_respond.float()
        resp_loss = self.bce(out["response_logit"].squeeze(-1), should_respond)
        
        # Decoder language modeling loss (if decoder was used)
        decoder_loss = torch.tensor(0.0, device=out["entity_logits"].device)
        if "logits" in out and decoder_input_ids is not None and "decoder_labels" in batch:
            logits = out["logits"]  # (B, T, vocab_size)
            labels = batch["decoder_labels"]  # (B, T)
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            decoder_loss = self.ce_ignore_neg100(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
        
        # Soft logic constraint loss
        soft_logic_loss = torch.tensor(0.0, device=out["entity_logits"].device)
        if len(self.model.softlogic.rules) > 0 and "node_entity_type_probs" in out and "rel_logits_matrix" in out:
            soft_logic_loss, _ = self.model.softlogic(
                out["node_entity_type_probs"], 
                out["rel_logits_matrix"]
            )
        
        loss = ent_loss + con_loss + resp_loss + decoder_loss + self.soft_logic_weight * soft_logic_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()


def load_kg_data_for_training(kg_embedding_path: Optional[str] = None,
                              kg_triples_path: Optional[str] = None,
                              entity_mapping: Optional[Dict[str, str]] = None,
                              kg_type: str = "conceptnet"):
    """
    Load knowledge graph data for training.
    
    Args:
        kg_embedding_path: Path to KG embeddings file
            - ConceptNet: Path to ConceptNet Numberbatch embeddings (.txt)
            - Wikidata: Path to Wikidata2Vec embeddings (.txt)
            - Other: Should follow same format (entity <space-separated-vector>)
        kg_triples_path: Path to KG triples file (for path reasoning)
            - ConceptNet: Path to ConceptNet assertions (.jsonl)
            - Wikidata: Path to Wikidata triples (format depends on source)
            - Other: Should be list of (source, relation, target) tuples
        entity_mapping: Optional manual mapping from text entities to KG entities
        kg_type: Type of knowledge graph ("conceptnet", "wikidata", "wordnet", or "generic")
    
    Returns:
        Tuple of (kg_loader, entity_linker, kg_graph) or (None, None, None) if KG unavailable
    """
    try:
        from kg_utils import KGEmbeddingLoader, EntityLinker, KGGraph, load_conceptnet_triples
    except ImportError:
        print("Warning: kg_utils not available. KG integration disabled.")
        return None, None, None
    
    if kg_embedding_path is None:
        print("No KG embedding path provided. KG integration disabled.")
        return None, None, None
    
    try:
        # Load KG embeddings
        kg_loader = KGEmbeddingLoader(kg_type=kg_type)
        
        # Load embeddings based on KG type
        if kg_type.lower() == "conceptnet":
            kg_embed_dim = kg_loader.load_conceptnet_embeddings(kg_embedding_path)
        elif kg_type.lower() == "wikidata":
            kg_embed_dim = kg_loader.load_wikidata_embeddings(kg_embedding_path)
        else:
            # Generic format: assume same format as ConceptNet
            kg_embed_dim = kg_loader.load_conceptnet_embeddings(kg_embedding_path)
        
        print(f"Loaded {kg_type} KG embeddings: {len(kg_loader.entity_embeddings)} entities, dim={kg_embed_dim}")
        
        # Create entity linker
        entity_linker = EntityLinker(kg_loader, entity_mapping)
        
        # Load KG graph if triples path provided
        kg_graph = None
        if kg_triples_path is not None:
            try:
                if kg_type.lower() == "conceptnet":
                    triples = load_conceptnet_triples(kg_triples_path, max_triples=50000)
                else:
                    # For other KG types, assume generic format or implement custom loader
                    print(f"Warning: Generic triple loading not yet implemented for {kg_type}")
                    print("Path reasoning will be disabled.")
                    triples = []
                
                if triples:
                    kg_graph = KGGraph()
                    kg_graph.load_from_triples(triples)
                    print(f"Loaded KG graph: {len(triples)} triples")
            except Exception as e:
                print(f"Warning: Could not load KG triples: {e}")
                print("Path reasoning will be disabled.")
        
        return kg_loader, entity_linker, kg_graph
    
    except Exception as e:
        print(f"Warning: Could not load KG data: {e}")
        print("Continuing without KG integration.")
        return None, None, None


def run_training(tokenizer, model, device="cpu", epochs_per_stage=1, skip_stage1_if_pretrained=True, 
                 soft_logic_weight=0.1, soft_logic_rules=None,
                 kg_embedding_path: Optional[str] = None,
                 kg_triples_path: Optional[str] = None,
                 kg_entity_mapping: Optional[Dict[str, str]] = None,
                 kg_type: str = "conceptnet"):
    """
    Run tiered training across 4 stages.
    
    Args:
        tokenizer: HuggingFace tokenizer
        model: NeuroSymbolicLM model
        device: torch device
        epochs_per_stage: number of epochs to train each stage
        skip_stage1_if_pretrained: If True, skip Stage 1 when using pre-trained encoder
        soft_logic_weight: weight for soft logic constraint loss (default 0.1)
        soft_logic_rules: optional dict with:
            - "concept_to_entity_type_map": dict mapping concept names to entity type indices
            - "relation_map": dict mapping relation names to relation indices (should match collator)
            - "rules": list of rule dicts (see configure_soft_logic_rules docstring)
        kg_embedding_path: Optional path to KG embeddings file (enables KG integration)
            - ConceptNet: Path to ConceptNet Numberbatch embeddings
            - Wikidata: Path to Wikidata2Vec embeddings
            - Other: Should follow same format
        kg_triples_path: Optional path to KG triples file (enables path reasoning)
            - ConceptNet: Path to ConceptNet assertions (.jsonl)
            - Wikidata: Path to Wikidata triples
        kg_entity_mapping: Optional mapping from text entities to KG entities
        kg_type: Type of knowledge graph ("conceptnet", "wikidata", "wordnet", or "generic")
    """
    model = model.to(device)
    model.train()
    
    # Load KG data if model supports it and paths are provided
    if model.use_kg and (kg_embedding_path is not None):
        print("\n" + "=" * 60)
        print("Loading Knowledge Graph Data")
        print("=" * 60)
        kg_loader, entity_linker, kg_graph = load_kg_data_for_training(
            kg_embedding_path=kg_embedding_path,
            kg_triples_path=kg_triples_path,
            entity_mapping=kg_entity_mapping,
            kg_type=kg_type
        )
        
        if kg_loader is not None:
            # Create entity mapping from dataset if not provided
            if kg_entity_mapping is None:
                ds_temp = ToyCognitiveDataset()
                all_entities = []
                for sample in ds_temp:
                    all_entities.extend(sample.get("entities", []))
                unique_entities = list(set(all_entities))
                
                try:
                    from kg_utils import create_entity_mapping_from_dataset
                    kg_entity_mapping = create_entity_mapping_from_dataset(unique_entities, kg_loader)
                    print(f"Auto-generated entity mapping for {len(kg_entity_mapping)} entities")
                except:
                    kg_entity_mapping = {}
            
            # Load KG data into model
            model.load_kg_data(
                kg_loader=kg_loader,
                entity_linker=entity_linker,
                kg_graph=kg_graph,
                entity_mapping=kg_entity_mapping
            )
            print("KG data loaded into model successfully")
        else:
            print("Warning: KG integration requested but data loading failed.")
            print("Continuing without KG integration.")
    elif model.use_kg:
        print("\nWarning: Model has KG integration enabled but no KG data paths provided.")
        print("Continuing without KG integration.")
    
    ds = ToyCognitiveDataset()
    # Expanded concept map to support multiple concepts per entity
    # When scaling, add all concepts that appear in your dataset here
    concept_map = {
        "animal": 1, "attribute": 2, "location": 3, "person": 4, "object": 5,
        # Additional concepts for richer semantic representation
        "predator": 6, "prey": 7, "city": 8, "country": 9,
        "male": 10, "female": 11, "public_space": 12,
        "professional": 13, "medical": 14, "educator": 15,
        "subject": 16, "academic": 17, "learner": 18,
        "building": 19, "knowledge": 20, "technology": 21, "electronic": 22
    }
    # Expand relation map to include all relations in dataset
    relation_map = {
        "chases": 1, "capital_of": 2, "barks_at": 3, "flies_over": 4, "hunts": 5,
        "swims_in": 6, "located_in": 7, "flows_through": 8, "loves": 9, "teaches": 10,
        "owns": 11, "cooks": 12, "met": 13, "on": 14, "fell_from": 15, "runs": 16,
        "before": 17, "follows": 18, "starts_at": 19, "brings": 20, "leads_to": 21,
        "galloped_across": 22, "lives_in": 23, "hopped_into": 24, "howled_at": 25,
        "near": 26, "stands_on": 27, "outside": 28, "contains": 29, "displays": 30,
        "stretches_along": 31
    }
    
    # Configure soft logic rules if provided
    if soft_logic_rules is not None:
        configure_soft_logic_rules(
            model,
            soft_logic_rules.get("concept_to_entity_type_map", {}),
            relation_map,  # Use the relation_map from collator
            soft_logic_rules.get("rules", [])
        )
    
    # Create collators: one without responses for early stages, one with for decoder training
    collator_basic = CognitiveCollator(tokenizer, concept_map, relation_map, include_responses=False)
    collator_with_responses = CognitiveCollator(tokenizer, concept_map, relation_map, include_responses=True)
    
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collator_basic)
    dl_with_responses = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collator_with_responses)
    
    # Move batches to device helper
    def to_device(batch):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Stage 1: MLM Pretraining (skip if using pre-trained encoder)
    use_pretrained = model.use_pretrained_encoder
    skip_stage1 = skip_stage1_if_pretrained and use_pretrained
    
    if not skip_stage1:
        print("=" * 60)
        print("Stage 1: MLM Pretraining")
        print("=" * 60)
        # Only train encoder if not using pre-trained (or if pre-trained is not frozen)
        if use_pretrained:
            # If using pre-trained but not frozen, we could fine-tune it here
            # For now, skip if using pre-trained
            print("  Skipping Stage 1: Using pre-trained encoder")
        else:
            optimizer1 = Adam(model.encoder.parameters(), lr=1e-4)
            s1 = Stage1_MLM_Trainer(model, tokenizer, optimizer1)
            for epoch in range(epochs_per_stage):
                total_loss = 0.0
                for batch in dl:
                    batch = to_device(batch)
                    loss = s1.train_step(batch)
                    total_loss += loss
                print(f"  Epoch {epoch+1}/{epochs_per_stage}, Avg Loss: {total_loss/len(dl):.4f}")
    else:
        print("=" * 60)
        print("Stage 1: MLM Pretraining (SKIPPED - Using pre-trained encoder)")
        print("=" * 60)

    print("\n" + "=" * 60)
    print("Stage 2: Entity & Concept Extraction")
    print("=" * 60)
    # Train entity and concept heads, plus GNN and relation scorer for soft logic
    # If using pre-trained encoder, check if it's frozen
    if use_pretrained:
        # Check if encoder is frozen by checking if any parameters require grad
        encoder_frozen = not any(p.requires_grad for p in model.encoder.parameters())
        if encoder_frozen:
            # Encoder is frozen, only train heads, GNN, and relation scorer
            stage2_params = (list(model.token_ent.parameters()) + 
                           list(model.concept_head.parameters()) +
                           list(model.gnn.parameters()) +
                           list(model.rel_scorer.parameters()) +
                           list(model.node_proj.parameters()))
        else:
            # Include encoder for fine-tuning
            stage2_params = (list(model.encoder.parameters()) + 
                           list(model.token_ent.parameters()) + 
                           list(model.concept_head.parameters()) +
                           list(model.gnn.parameters()) +
                           list(model.rel_scorer.parameters()) +
                           list(model.node_proj.parameters()))
    else:
        # Custom encoder, include it
        stage2_params = (list(model.encoder.parameters()) + 
                        list(model.token_ent.parameters()) + 
                        list(model.concept_head.parameters()) +
                        list(model.gnn.parameters()) +
                        list(model.rel_scorer.parameters()) +
                        list(model.node_proj.parameters()))
    optimizer2 = Adam(stage2_params, lr=1e-4)
    s2 = Stage2_Symbolic_Trainer(model, optimizer2, soft_logic_weight=soft_logic_weight)
    for epoch in range(epochs_per_stage):
        total_loss = 0.0
        for batch in dl:
            batch = to_device(batch)
            loss = s2.train_step(batch)
            total_loss += loss
        print(f"  Epoch {epoch+1}/{epochs_per_stage}, Avg Loss: {total_loss/len(dl):.4f}")

    print("\n" + "=" * 60)
    print("Stage 3: Response Controller")
    print("=" * 60)
    optimizer3 = Adam(model.controller.parameters(), lr=1e-4)
    s3 = Stage3_Control_Trainer(model, optimizer3)
    for epoch in range(epochs_per_stage):
        total_loss = 0.0
        for batch in dl:
            batch = to_device(batch)
            loss = s3.train_step(batch)
            total_loss += loss
        print(f"  Epoch {epoch+1}/{epochs_per_stage}, Avg Loss: {total_loss/len(dl):.4f}")

    print("\n" + "=" * 60)
    print("Stage 3.5: Decoder Response Generation")
    print("=" * 60)
    optimizer3_5 = Adam(model.decoder.parameters(), lr=1e-4)
    s3_5 = Stage3_Decoder_Trainer(model, optimizer3_5)
    for epoch in range(epochs_per_stage):
        total_loss = 0.0
        for batch in dl_with_responses:
            batch = to_device(batch)
            loss = s3_5.train_step(batch)
            total_loss += loss
        print(f"  Epoch {epoch+1}/{epochs_per_stage}, Avg Loss: {total_loss/len(dl_with_responses):.4f}")
    
    # Evaluate generation after Stage 3.5
    if epochs_per_stage > 0:
        print("\n  Evaluating generation after Stage 3.5...")
        gen_results = evaluate_generation(model, tokenizer, ds, device=device, num_samples=5)
        print_generation_results(gen_results, num_to_print=3)

    print("\n" + "=" * 60)
    print("Stage 4: Joint End-to-End Training")
    print("=" * 60)
    # For joint training, include all trainable parameters
    # If encoder is frozen, it won't be included in optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer4 = Adam(trainable_params, lr=1e-5)  # Lower LR for fine-tuning
    s4 = Stage4_Joint_Trainer(model, optimizer4, soft_logic_weight=soft_logic_weight)
    for epoch in range(epochs_per_stage):
        total_loss = 0.0
        for batch in dl_with_responses:  # Use dataloader with responses for joint training
            batch = to_device(batch)
            loss = s4.train_step(batch)
            total_loss += loss
        print(f"  Epoch {epoch+1}/{epochs_per_stage}, Avg Loss: {total_loss/len(dl_with_responses):.4f}")
        
        # Evaluate generation periodically (every 5 epochs or on last epoch)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs_per_stage:
            print(f"\n  Evaluating generation after epoch {epoch+1}...")
            gen_results = evaluate_generation(model, tokenizer, ds, device=device, num_samples=5)
            print_generation_results(gen_results, num_to_print=3)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Generation Evaluation")
    print("=" * 60)
    final_results = evaluate_generation(model, tokenizer, ds, device=device, num_samples=10)
    print_generation_results(final_results, num_to_print=5)
    
    return model


def generate_response(model, tokenizer, input_text, device="cpu", max_length=128, 
                      temperature=1.0, do_sample=False, top_k=None, top_p=None):
    """
    Generate a response for a given input text using the trained model.
    
    Args:
        model: Trained NeuroSymbolicLM model
        tokenizer: Tokenizer for encoding/decoding
        input_text: Input text to generate response for
        device: Device to run generation on
        max_length: Maximum generation length
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        
    Returns:
        generated_text: Generated response text
    """
    model.eval()
    
    # Get token IDs for special tokens
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    if bos_token_id is None:
        bos_token_id = 1
    if eos_token_id is None:
        eos_token_id = 2
    
    # Tokenize input
    tokenized = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    model.train()
    return generated_text


if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # BERT tokenizer should have pad_token, but ensure it's set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    # Initialize model with pre-trained encoder
    from model import NeuroSymbolicLM
    vocab_size = len(tokenizer)
    
    # ========================================================================
    # Knowledge Graph Integration Configuration
    # ========================================================================
    # To enable KG-aware GNN and multi-hop reasoning:
    # 
    # Supported KG types:
    #   - "conceptnet": ConceptNet Numberbatch embeddings
    #     Download from: https://github.com/commonsense/conceptnet-numberbatch
    #     Triples from: https://github.com/commonsense/conceptnet5/wiki/Downloads
    #   
    #   - "wikidata": Wikidata2Vec embeddings
    #     Download from: https://github.com/dice-group/wikidata2vec
    #   
    #   - "generic": Any KG with format: "entity <space-separated-vector>"
    # 
    # 1. Download KG embeddings for your chosen KG type
    # 2. (Optional) Download KG triples for path reasoning
    # 3. Set the paths below and enable the flags
    # ========================================================================
    USE_KG = False  # Set to True to enable KG integration
    KG_TYPE = "conceptnet"  # Options: "conceptnet", "wikidata", "wordnet", "generic"
    KG_EMBEDDING_PATH = None  # Path to KG embeddings file
    KG_TRIPLES_PATH = None  # Path to KG triples file (for path reasoning)
    USE_KG_GNN = False  # Use KG-aware GNN instead of simple GNN
    USE_PATH_REASONING = False  # Enable multi-hop path reasoning
    MAX_PATH_LENGTH = 3  # Maximum path length for reasoning
    
    # Example configuration for ConceptNet (uncomment and set to actual paths):
    # KG_TYPE = "conceptnet"
    # KG_EMBEDDING_PATH = "data/conceptnet-vectors-numberbatch-17.06.txt"
    # KG_TRIPLES_PATH = "data/conceptnet_assertions.jsonl"
    # USE_KG = True
    # USE_KG_GNN = True  # Recommended: enables KG-guided message passing
    # USE_PATH_REASONING = True  # Optional: enables multi-hop reasoning
    # 
    # Example configuration for Wikidata:
    # KG_TYPE = "wikidata"
    # KG_EMBEDDING_PATH = "data/wikidata2vec_embeddings.txt"
    # KG_TRIPLES_PATH = "data/wikidata_triples.nt"  # May need custom loader
    # USE_KG = True
    # USE_KG_GNN = True
    # ========================================================================
    
    # Use pre-trained BERT encoder (hidden_size=768)
    # Set d_model to match BERT's hidden size
    # Optionally use pre-trained T5 decoder for better response generation
    model = NeuroSymbolicLM(
        vocab_size=vocab_size,
        d_model=768,  # BERT-base hidden size
        n_entity_types=8,
        n_concepts=512,
        concept_dim=256,
        node_dim=256,
        max_nodes=16,
        use_pretrained_encoder=True,  # Use pre-trained BERT
        pretrained_model_name="bert-base-uncased",
        freeze_encoder=True,  # Set to True to freeze encoder, False to fine-tune
        use_pretrained_decoder=True,  # Set to True to use T5 decoder
        pretrained_decoder_name="t5-small",  # Options: "t5-small", "t5-base", "facebook/bart-base"
        freeze_decoder=False,  # Set to True to freeze decoder, False to fine-tune
        # KG integration parameters
        use_kg=USE_KG,
        kg_embed_dim=300,  # ConceptNet embedding dimension
        use_kg_gnn=USE_KG_GNN,
        use_path_reasoning=USE_PATH_REASONING,
        max_path_length=MAX_PATH_LENGTH
    )
    
    # Configure soft logic rules
    # Map concepts to entity types (0-indexed entity type IDs)
    # Note: entity_types in model are 0-7, concepts in dataset are: animal, person, location, object, attribute
    concept_to_entity_type_map = {
        "animal": 0,
        "person": 1, 
        "location": 2,
        "object": 3,
        "attribute": 4
    }
    
    # Define soft logic rules
    # Rules specify: when entity type A and entity type B appear, relation R should (or shouldn't) be present
    soft_logic_rules = {
        "concept_to_entity_type_map": concept_to_entity_type_map,
        "rules": [
            # Encourage: animals can chase other animals
            {"concept_a": "animal", "concept_b": "animal", "relation": "chases", "weight": 1.0, "polarity": 1},
            # Encourage: locations can have capital_of relation
            {"concept_a": "location", "concept_b": "location", "relation": "capital_of", "weight": 1.0, "polarity": 1},
            # Encourage: animals can live in locations
            {"concept_a": "animal", "concept_b": "location", "relation": "lives_in", "weight": 1.0, "polarity": 1},
            # Discourage: attributes shouldn't have capital_of relation
            {"concept_a": "attribute", "concept_b": "location", "relation": "capital_of", "weight": 0.5, "polarity": -1},
        ]
    }
    
    print(f"Model initialized with pre-trained encoder")
    print(f"Encoder hidden size: {model.d_model}")
    print(f"Encoder frozen: {not any(p.requires_grad for p in model.encoder.parameters())}")
    if model.use_pretrained_decoder:
        print(f"Using pre-trained decoder: {model.decoder.pretrained_model.config.name_or_path}")
        print(f"Decoder frozen: {model.decoder.freeze_decoder}")
    
    # Print KG integration status
    if model.use_kg:
        print(f"\nKG Integration: ENABLED")
        print(f"  - KG-aware GNN: {model.use_kg_gnn}")
        print(f"  - Path reasoning: {model.use_path_reasoning}")
        print(f"  - Max path length: {model.max_path_length}")
        print(f"  - KG embedding dim: {model.kg_embed_dim}")
    else:
        print(f"\nKG Integration: DISABLED")
        print("  To enable: Set USE_KG=True and provide KG_EMBEDDING_PATH")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run training (Stage 1 will be skipped automatically)
    # soft_logic_weight controls how much the soft logic constraint loss contributes (default 0.1)
    trained_model = run_training(
        tokenizer, 
        model, 
        device=device, 
        epochs_per_stage=10, 
        skip_stage1_if_pretrained=True,
        soft_logic_weight=0.1,  # Adjust this to balance soft logic vs other losses
        soft_logic_rules=soft_logic_rules,
        # KG integration parameters
        kg_embedding_path=KG_EMBEDDING_PATH,
        kg_triples_path=KG_TRIPLES_PATH,
        kg_entity_mapping=None,  # Can provide manual mapping here if needed
        kg_type=KG_TYPE
    )
    
    # Example: Generate responses for some test inputs
    print("\n" + "=" * 60)
    print("Example Generation")
    print("=" * 60)
    test_inputs = [
        "Is Paris the capital of France?",
        "Does the cat chase the mouse?",
        "Did the teacher teach the students?"
    ]
    
    for test_input in test_inputs:
        generated = generate_response(
            trained_model, 
            tokenizer, 
            test_input, 
            device=device,
            max_length=128,
            do_sample=False  # Use greedy decoding
        )
        print(f"\nInput:    {test_input}")
        print(f"Generated: {generated}")
