"""Prompt templates for LLM annotation."""

# System prompt for annotation
SYSTEM_PROMPT = """You are an expert annotator for training a neurosymbolic language model. The model combines a transformer backbone with symbolic reasoning components (entity recognition, concept grounding, relation extraction, and logic constraints) to deeply understand input text and decide whether a response is warranted.

The model should ONLY respond when:
1. The input contains a question (explicit or implied) that warrants an answer.
2. The input contains a factual error or misconception that should be corrected.
3. The input requests information, clarification, or action.
4. The input contains genuinely novel information that should be acknowledged or integrated into the model's knowledge.

The model should NOT respond when:
- The input is a simple statement of fact with no question or error.
- The input is casual/phatic (e.g. greetings, small talk) with no informational need.
- The input is already correct and complete, requiring no elaboration.

Your job is to annotate text so the model learns both rich contextual understanding AND appropriate response behavior.

You must output ONLY valid JSON with NO additional text, explanation, or markdown formatting.

Guidelines:
- Extract named entities (people, organizations, locations, dates, etc.)
- Provide character-level spans for each entity
- Assign 1-3 concepts per entity from the concept hierarchy
- Extract meaningful relations between entities
- Critically assess whether the input requires a response, and why
- If responding, provide a focused, factual response

Be precise with character spans - they must match the exact positions in the text."""

# Main annotation prompt template
ANNOTATION_PROMPT = """Analyze the following text. Extract structured contextual information and determine whether a response is needed.

TEXT:
{text}

Provide a JSON response with exactly this structure:
{{
  "entities": ["entity1", "entity2", ...],
  "entity_spans": [[start1, end1], [start2, end2], ...],
  "concepts": [["concept1", "concept2"], ["concept1"], ...],
  "relations": [[head_idx, tail_idx, "relation_type"], ...],
  "should_respond": 0 or 1,
  "response_reason": "question" | "correction" | "request" | "novelty" | "none",
  "response": "A response if warranted, otherwise empty string."
}}

IMPORTANT:
- entity_spans: [start, end] are character indices where the entity appears in TEXT
- concepts: For each entity, provide 1-3 concepts from this hierarchy:
  - person, organization, location, date, time, quantity, object, event
  - More specific: scientist, politician, company, city, country, year, money
- relations: Use indices into the entities list, with relation types like:
  - born_in, located_in, works_for, founded, member_of, part_of, capital_of
  - created_by, directed_by, written_by, owned_by, married_to
- should_respond: Decide carefully:
  - 1 if the text asks a question (explicit or implied)
  - 1 if the text contains a factual error that should be corrected
  - 1 if the text requests information or action
  - 0 if the text is a correct factual statement, casual remark, or otherwise needs no reply
- response_reason: Why a response is needed:
  - "question" - the input asks or implies a question
  - "correction" - the input contains a factual error or misconception
  - "request" - the input requests information, help, or action
  - "novelty" - the input contains genuinely new/uncommon information worth acknowledging
  - "none" - no response needed (should_respond must be 0)
- response: If should_respond=1, provide a focused, factual response. If 0, use ""

Output ONLY the JSON object:"""

# Prompt for statement-only (no response expected - used for known benign inputs)
STATEMENT_PROMPT = """Analyze the following text and extract entities, concepts, and relations. This text is a factual statement that does not require a response.

TEXT:
{text}

Provide a JSON response:
{{
  "entities": ["entity1", "entity2", ...],
  "entity_spans": [[start1, end1], [start2, end2], ...],
  "concepts": [["concept1", "concept2"], ["concept1"], ...],
  "relations": [[head_idx, tail_idx, "relation_type"], ...],
  "should_respond": 0,
  "response_reason": "none",
  "response": ""
}}

Output ONLY the JSON object:"""

# Prompt for QA generation - creates training pairs where a response IS warranted
QA_PROMPT = """Based on the following text and extracted information, generate a question-answer pair that the model should learn to respond to.

TEXT:
{text}

ENTITIES: {entities}
RELATIONS: {relations}

Generate a factual question about this text and its correct answer. The question should be one where a response is clearly warranted.

Provide a JSON response:
{{
  "question": "A factual question about the text",
  "answer": "The correct, focused answer based on the text",
  "response_reason": "question"
}}

Output ONLY the JSON object:"""

# Concepts hierarchy for reference
CONCEPT_HIERARCHY = {
    "person": ["scientist", "politician", "artist", "athlete", "writer", "actor", "musician", "leader"],
    "organization": ["company", "government", "university", "non_profit", "sports_team", "band"],
    "location": ["city", "country", "region", "continent", "building", "landmark", "address"],
    "temporal": ["date", "time", "year", "month", "day", "period", "era"],
    "quantity": ["money", "percentage", "distance", "weight", "count", "age"],
    "object": ["product", "vehicle", "document", "artwork", "food", "weapon", "tool"],
    "event": ["war", "election", "disaster", "ceremony", "meeting", "competition"],
    "concept": ["theory", "law", "disease", "technology", "language", "religion"],
}

# Relation types
RELATION_TYPES = [
    # Person relations
    "born_in", "died_in", "lived_in", "nationality", "educated_at",
    "works_for", "employed_by", "founded", "created", "wrote", "directed",
    "married_to", "child_of", "parent_of", "sibling_of", "member_of",
    # Organization relations
    "headquartered_in", "subsidiary_of", "parent_company", "acquired",
    "merged_with", "partner_of", "competitor_of",
    # Location relations
    "located_in", "capital_of", "part_of", "borders", "near",
    # Event relations
    "occurred_in", "started_on", "ended_on", "participant_in",
    # Object relations
    "owned_by", "made_by", "used_by", "contains", "made_of",
    # Generic relations
    "related_to", "instance_of", "subclass_of", "same_as",
]


def get_annotation_prompt(text: str, include_response: bool = True) -> str:
    """Get the annotation prompt for a text.

    Args:
        text: The text to annotate
        include_response: Whether to include QA response generation

    Returns:
        Formatted prompt string
    """
    if include_response:
        return ANNOTATION_PROMPT.format(text=text)
    else:
        return STATEMENT_PROMPT.format(text=text)


def get_qa_prompt(text: str, entities: list, relations: list) -> str:
    """Get prompt for QA generation.

    Args:
        text: Original text
        entities: Extracted entities
        relations: Extracted relations

    Returns:
        Formatted QA prompt
    """
    return QA_PROMPT.format(
        text=text,
        entities=", ".join(entities) if entities else "None",
        relations=str(relations) if relations else "None",
    )


def get_concept_suggestions(entity_type: str) -> list:
    """Get suggested concepts for an entity type.

    Args:
        entity_type: The general entity type

    Returns:
        List of suggested specific concepts
    """
    return CONCEPT_HIERARCHY.get(entity_type, ["object"])
