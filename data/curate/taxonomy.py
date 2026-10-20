"""Label taxonomy for GLiNER2-based annotation.

GLiNER2 is a zero-shot extractor: it selects from labels *we* supply rather
than inventing its own. Diversity therefore comes from this inventory — a
wide set of fine-grained concept labels, each with a natural-language
description that GLiNER2 matches against the text.

The taxonomy is deliberately aligned with the model architecture budgets in
``config.ModelConfig``:

- 16 coarse entity types  -> ``n_entity_types=16`` (index 0 reserved for
  padding/none, so 15 real types + none)
- fine-grained concept labels -> rows of the ConceptBank (``n_concepts``)
- relation labels -> ``n_relations`` (index 0 reserved for unknown)

Every fine-grained label maps to exactly one coarse entity type, so a single
GLiNER2 entity pass yields both the concept labels (all fine labels that
matched a span) and the entity type (coarse parent of the best-scoring
label) — no separate concept-assignment step and no vocabulary drift between
the two heads.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Coarse entity types (index 0 = none/padding, matching the collator/trainer
# convention of ignore_index=0)
# ---------------------------------------------------------------------------

ENTITY_TYPES: List[str] = [
    "person",
    "organization",
    "location",
    "temporal",
    "quantity",
    "creative_work",
    "product",
    "event",
    "biological",
    "substance",
    "abstract_concept",
    "technology",
    "activity",
    "attribute",
    "language_or_group",
    # Assistant-trace / coding domains
    "digital_artifact",
    "code_construct",
]

ENTITY_TYPE_TO_ID: Dict[str, int] = {t: i + 1 for i, t in enumerate(ENTITY_TYPES)}


# ---------------------------------------------------------------------------
# Fine-grained concept labels.
#
# Keyed by concept name; value is (coarse entity type, GLiNER2 description).
# Descriptions matter: GLiNER2 matches spans against them, so they should
# describe what the span *looks like in text*, not encyclopedic definitions.
# ---------------------------------------------------------------------------

CONCEPT_LABELS: Dict[str, Tuple[str, str]] = {
    # --- person ---
    "scientist": ("person", "A scientist, researcher, or academic mentioned by name"),
    "politician": ("person", "A politician, head of state, or government official mentioned by name"),
    "artist": ("person", "A visual artist, painter, sculptor, or designer mentioned by name"),
    "musician": ("person", "A musician, singer, composer, or band member mentioned by name"),
    "actor": ("person", "An actor, actress, or performer mentioned by name"),
    "writer": ("person", "An author, journalist, poet, or writer mentioned by name"),
    "athlete": ("person", "An athlete or sports player mentioned by name"),
    "business_person": ("person", "An executive, entrepreneur, founder, or investor mentioned by name"),
    "military_figure": ("person", "A military officer, soldier, or commander mentioned by name"),
    "religious_figure": ("person", "A religious leader, cleric, or saint mentioned by name"),
    "historical_figure": ("person", "A historical person such as a monarch, explorer, or philosopher"),
    "fictional_character": ("person", "A fictional or mythological character"),
    "generic_person": ("person", "Any other named person or personal role such as a doctor or teacher"),
    # --- organization ---
    "company": ("organization", "A commercial company, corporation, or business"),
    "government_body": ("organization", "A government agency, ministry, parliament, or public institution"),
    "university": ("organization", "A university, school, college, or research institute"),
    "nonprofit": ("organization", "A non-profit organization, NGO, charity, or foundation"),
    "sports_team": ("organization", "A sports team, club, or league"),
    "band": ("organization", "A musical band, orchestra, or ensemble"),
    "media_outlet": ("organization", "A newspaper, broadcaster, publisher, or media organization"),
    "political_party": ("organization", "A political party or political movement"),
    "military_unit": ("organization", "An army, military branch, or armed group"),
    "international_org": ("organization", "An international or intergovernmental organization such as the UN or EU"),
    # --- location ---
    "city": ("location", "A city, town, or village"),
    "country": ("location", "A country or sovereign state"),
    "region": ("location", "A state, province, county, or named region"),
    "continent": ("location", "A continent or major world region"),
    "body_of_water": ("location", "A river, lake, sea, or ocean"),
    "mountain": ("location", "A mountain, mountain range, or volcano"),
    "building": ("location", "A building, stadium, airport, or man-made structure"),
    "landmark": ("location", "A monument, landmark, park, or tourist attraction"),
    "street_or_address": ("location", "A street name, road, or address"),
    "astronomical_object": ("location", "A planet, star, galaxy, or other astronomical object"),
    # --- temporal ---
    "date": ("temporal", "A specific calendar date such as 'March 5, 1997'"),
    "year": ("temporal", "A year or decade such as '1984' or 'the 1960s'"),
    "time_of_day": ("temporal", "A clock time or part of the day"),
    "duration": ("temporal", "A length or span of time such as 'three weeks'"),
    "era": ("temporal", "A named historical period or era such as 'the Renaissance'"),
    "recurring_time": ("temporal", "A day of the week, month name, season, or holiday"),
    # --- quantity ---
    "money": ("quantity", "A monetary amount or price"),
    "percentage": ("quantity", "A percentage or ratio"),
    "physical_measure": ("quantity", "A physical measurement such as distance, weight, speed, or temperature"),
    "count": ("quantity", "A count or cardinal number of things"),
    "age": ("quantity", "A person's or thing's age"),
    "ordinal": ("quantity", "An ordinal or ranking such as 'first' or '3rd'"),
    # --- creative_work ---
    "book": ("creative_work", "A book, novel, or written publication title"),
    "film": ("creative_work", "A film or movie title"),
    "tv_show": ("creative_work", "A television show or series title"),
    "song_or_album": ("creative_work", "A song, album, or musical work title"),
    "artwork": ("creative_work", "A painting, sculpture, or other artwork title"),
    "video_game": ("creative_work", "A video game title"),
    "periodical": ("creative_work", "A newspaper, magazine, or journal title"),
    "document": ("creative_work", "A named document, treaty, law, report, or standard"),
    # --- product ---
    "vehicle": ("product", "A vehicle model such as a car, aircraft, ship, or spacecraft"),
    "device": ("product", "An electronic device, gadget, or machine model"),
    "food_or_drink": ("product", "A food item, dish, beverage, or ingredient"),
    "clothing": ("product", "A clothing item, fashion product, or accessory"),
    "weapon": ("product", "A weapon or weapons system"),
    "generic_product": ("product", "Any other commercial product, brand, or model name"),
    # --- event ---
    "war_or_conflict": ("event", "A war, battle, conflict, or military operation"),
    "election": ("event", "An election, referendum, or vote"),
    "disaster": ("event", "A natural disaster, accident, or catastrophe"),
    "sports_event": ("event", "A sports competition, match, tournament, or olympics"),
    "festival_or_ceremony": ("event", "A festival, ceremony, award show, or celebration"),
    "meeting_or_conference": ("event", "A summit, conference, meeting, or negotiation"),
    "historical_event": ("event", "Any other named historical event, movement, or incident"),
    # --- biological ---
    "animal": ("biological", "An animal or animal species"),
    "plant": ("biological", "A plant, tree, or plant species"),
    "microorganism": ("biological", "A microorganism such as a bacterium or virus"),
    "body_part": ("biological", "A body part, organ, or anatomical structure"),
    "disease": ("biological", "A disease, medical condition, symptom, or injury"),
    "gene_or_protein": ("biological", "A gene, protein, or biological molecule"),
    # --- substance ---
    "chemical": ("substance", "A chemical element, compound, or substance"),
    "drug": ("substance", "A medication, drug, or pharmaceutical substance"),
    "material": ("substance", "A material such as wood, steel, plastic, or fabric"),
    "natural_resource": ("substance", "A natural resource such as oil, coal, or minerals"),
    # --- abstract_concept ---
    "theory_or_law": ("abstract_concept", "A scientific theory, law, principle, or theorem"),
    "field_of_study": ("abstract_concept", "An academic discipline or field of study"),
    "ideology_or_religion": ("abstract_concept", "A religion, belief system, ideology, or philosophy"),
    "emotion_or_state": ("abstract_concept", "An emotion, feeling, or mental state"),
    "social_construct": ("abstract_concept", "A social or legal concept such as democracy, marriage, or copyright"),
    # --- technology ---
    "software": ("technology", "A software application, operating system, or app"),
    "programming_language": ("technology", "A programming language, framework, or library"),
    "website_or_platform": ("technology", "A website, online platform, or social network"),
    "technical_method": ("technology", "A technology, technique, algorithm, or process"),
    "unit_or_standard": ("technology", "A technical standard, protocol, unit, or file format"),
    # --- activity ---
    "sport": ("activity", "A sport or physical game"),
    "occupation": ("activity", "A job title, profession, or occupation"),
    "hobby_or_practice": ("activity", "A hobby, craft, or cultural practice"),
    "industry": ("activity", "An industry or economic sector"),
    # --- attribute ---
    "color": ("attribute", "A color"),
    "shape_or_size": ("attribute", "A shape, size, or physical property"),
    "quality": ("attribute", "A descriptive quality or characteristic"),
    # --- language_or_group ---
    "language": ("language_or_group", "A natural language or dialect"),
    "nationality_or_ethnicity": ("language_or_group", "A nationality, ethnic group, or demonym"),
    "community": ("language_or_group", "A named community, tribe, or social group"),
    # --- digital_artifact (assistant traces, tool use) ---
    "file_path": ("digital_artifact", "A file or directory path such as 'src/main.py' or '/etc/hosts'"),
    "url": ("digital_artifact", "A URL or web address"),
    "email_address": ("digital_artifact", "An email address"),
    "identifier": ("digital_artifact", "An identifier such as a UUID, hash, ticket number, or account ID"),
    "version_number": ("digital_artifact", "A software version number such as 'v2.1.0' or 'Python 3.11'"),
    "environment_variable": ("digital_artifact", "An environment variable or configuration key such as 'API_KEY' or 'PATH'"),
    "cli_command": ("digital_artifact", "A shell or command-line command such as 'git commit' or 'pip install'"),
    "error_message": ("digital_artifact", "An error message, exception text, or status code from a program"),
    "tool_name": ("digital_artifact", "The name of a callable tool, function API, or service an assistant can invoke"),
    "api_endpoint": ("digital_artifact", "An API endpoint, route, or HTTP method such as 'GET /users'"),
    "database_object": ("digital_artifact", "A database, table, column, or query object name"),
    "ui_element": ("digital_artifact", "A user interface element such as a button, menu, or settings field"),
    # --- code_construct (source code) ---
    "function": ("code_construct", "A function or method name in source code"),
    "class_name": ("code_construct", "A class, struct, or interface name in source code"),
    "variable": ("code_construct", "A variable, constant, or parameter name in source code"),
    "module_or_package": ("code_construct", "A module, package, or namespace name in source code"),
    "data_structure": ("code_construct", "A data structure or type such as a list, dict, array, or tensor"),
    "exception_type": ("code_construct", "An exception or error type name in code such as 'ValueError'"),
    # --- technology additions ---
    "library_or_framework": ("technology", "A software library, framework, or SDK such as PyTorch or React"),
}


# ---------------------------------------------------------------------------
# Relation labels with descriptions for GLiNER2 relation extraction.
#
# Keyed by relation name; value is (description, set of plausible head coarse
# types, set of plausible tail coarse types). Head/tail types are used to
# prune which relations get asked about for a given document (schema-aware
# pruning keeps each GLiNER2 pass small and precise) and to sanity-check
# extracted pairs.  Empty tuple = any type allowed.
# ---------------------------------------------------------------------------

RELATION_LABELS: Dict[str, Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = {
    # person <-> location
    "born_in": ("person was born in location", ("person",), ("location",)),
    "died_in": ("person died in location", ("person",), ("location",)),
    "lived_in": ("person lived or resides in location", ("person",), ("location",)),
    "nationality": ("person has nationality or citizenship", ("person",), ("location", "language_or_group")),
    # person <-> organization
    "works_for": ("person works for or is employed by organization", ("person",), ("organization",)),
    "leader_of": ("person leads, heads, or is CEO/president of organization or location", ("person",), ("organization", "location")),
    "founded": ("person founded or established organization", ("person",), ("organization",)),
    "member_of": ("person or organization is a member of organization or group", ("person", "organization"), ("organization", "language_or_group")),
    "educated_at": ("person studied at or graduated from institution", ("person",), ("organization",)),
    # person <-> person
    "married_to": ("person is or was married to person", ("person",), ("person",)),
    "child_of": ("person is the child of person", ("person",), ("person",)),
    "parent_of": ("person is the parent of person", ("person",), ("person",)),
    "sibling_of": ("person is the sibling of person", ("person",), ("person",)),
    "collaborated_with": ("person worked or collaborated with person", ("person",), ("person",)),
    # person <-> creative_work / product
    "created": ("person or organization created, invented, or developed something", ("person", "organization"), ("creative_work", "product", "technology", "abstract_concept")),
    "wrote": ("person wrote book, article, or document", ("person",), ("creative_work",)),
    "directed": ("person directed film or show", ("person",), ("creative_work",)),
    "performed_in": ("person acted or performed in creative work or event", ("person",), ("creative_work", "event")),
    "composed": ("person composed or performed song or musical work", ("person",), ("creative_work",)),
    # organization <-> organization / location
    "subsidiary_of": ("organization is a subsidiary or division of organization", ("organization",), ("organization",)),
    "acquired": ("organization acquired or bought organization", ("organization",), ("organization",)),
    "headquartered_in": ("organization is headquartered or based in location", ("organization",), ("location",)),
    "competitor_of": ("organization competes with organization", ("organization",), ("organization",)),
    "partner_of": ("organization partners or is allied with organization", ("organization", "person"), ("organization", "person")),
    # location <-> location
    "located_in": ("place or thing is located in or within location", (), ("location",)),
    "capital_of": ("city is the capital of country or region", ("location",), ("location",)),
    "part_of": ("thing is a part or component of a larger thing", (), ()),
    "borders": ("location shares a border with location", ("location",), ("location",)),
    # events
    "occurred_in": ("event occurred or took place in location or time", ("event",), ("location", "temporal")),
    "participant_in": ("person or organization participated in event", ("person", "organization", "location"), ("event",)),
    "caused": ("something caused or led to something else", (), ()),
    "happened_on": ("event happened on date or during period", ("event",), ("temporal",)),
    # products / works
    "made_by": ("product or work was made or manufactured by organization or person", ("product", "creative_work", "technology"), ("organization", "person")),
    "owned_by": ("thing is owned or held by person or organization", (), ("person", "organization")),
    "used_for": ("thing is used for purpose or activity", (), ()),
    "made_of": ("thing is made of material or substance", (), ("substance",)),
    "released_in": ("work or product was released or published in time or place", ("creative_work", "product", "technology"), ("temporal", "location")),
    # biological / medical
    "treats": ("drug or treatment treats disease or condition", ("substance", "technology"), ("biological",)),
    "causes_condition": ("agent or factor causes disease or condition", (), ("biological",)),
    "habitat_of": ("location is the habitat of animal or plant", ("location",), ("biological",)),
    # code / trace relations
    "calls": ("function, code, or assistant calls a function or tool", ("code_construct", "digital_artifact"), ("code_construct", "digital_artifact", "technology")),
    "defined_in": ("code construct is defined in a class, module, or file", ("code_construct",), ("code_construct", "digital_artifact")),
    "imports": ("module or file imports a module or library", ("code_construct", "digital_artifact"), ("code_construct", "technology")),
    "returns_value": ("function returns a value, object, or type", ("code_construct",), ()),
    "raises": ("code raises or produces an exception or error", ("code_construct", "digital_artifact"), ("code_construct", "digital_artifact")),
    "depends_on": ("software depends on a library, service, or version", ("technology", "code_construct", "digital_artifact", "product"), ("technology", "code_construct", "digital_artifact")),
    "argument_of": ("value is passed as an argument to a tool, command, or function", (), ("code_construct", "digital_artifact", "technology")),
    "produced_by": ("output or artifact was produced by a tool, command, or function", ("digital_artifact",), ("code_construct", "digital_artifact", "technology")),
    "configures": ("setting or variable configures software or a system", ("digital_artifact",), ("technology", "digital_artifact", "product")),
    "located_at": ("resource or content is located at a path or URL", (), ("digital_artifact",)),
    # generic fallbacks
    "instance_of": ("thing is an instance or type of category", (), ()),
    "related_to": ("two things are otherwise related", (), ()),
}


DEFAULT_GLINER_MODEL = "fastino/gliner2-base-v1"

# GLiNER-family models degrade when asked about too many labels at once;
# passes are chunked to stay well inside a reliable range.
MAX_LABELS_PER_PASS = 24
MAX_RELATIONS_PER_PASS = 20
MAX_CONCEPTS_PER_ENTITY = 3


@dataclass
class Taxonomy:
    """Runtime view over the label inventory with helper lookups.

    A custom (e.g. domain-specific) taxonomy can be constructed by passing
    different dicts; the defaults reproduce the module-level inventory.
    """

    concept_labels: Dict[str, Tuple[str, str]] = field(
        default_factory=lambda: dict(CONCEPT_LABELS)
    )
    relation_labels: Dict[str, Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = field(
        default_factory=lambda: dict(RELATION_LABELS)
    )
    entity_types: List[str] = field(default_factory=lambda: list(ENTITY_TYPES))

    def __post_init__(self):
        self.entity_type_to_id = {t: i + 1 for i, t in enumerate(self.entity_types)}
        self.concept_to_type = {c: parent for c, (parent, _) in self.concept_labels.items()}

    # -- entity label helpers -------------------------------------------------

    def entity_label_batches(
        self, batch_size: int = MAX_LABELS_PER_PASS
    ) -> List[Dict[str, str]]:
        """Concept labels chunked into {name: description} dicts per GLiNER2 pass.

        Labels are grouped so each pass contains contiguous coarse-type groups,
        which keeps semantically-confusable labels (e.g. all person subtypes)
        in the same pass where GLiNER2 can arbitrate between them.
        """
        items = [(name, desc) for name, (_, desc) in self.concept_labels.items()]
        return [
            dict(items[i : i + batch_size]) for i in range(0, len(items), batch_size)
        ]

    def coarse_type_of(self, concept: str) -> Optional[str]:
        return self.concept_to_type.get(concept)

    def entity_type_id(self, coarse_type: str) -> int:
        return self.entity_type_to_id.get(coarse_type, 0)

    # -- relation helpers -----------------------------------------------------

    def relations_for_types(self, present_types: set) -> Dict[str, str]:
        """Relation {name: description} restricted to plausible ones.

        Given the set of coarse entity types present in a document, drop
        relations whose head or tail type constraints cannot be satisfied.
        """
        out = {}
        for name, (desc, heads, tails) in self.relation_labels.items():
            head_ok = not heads or any(t in present_types for t in heads)
            tail_ok = not tails or any(t in present_types for t in tails)
            if head_ok and tail_ok:
                out[name] = desc
        return out

    def relation_batches(
        self, relations: Dict[str, str], batch_size: int = MAX_RELATIONS_PER_PASS
    ) -> List[Dict[str, str]]:
        items = list(relations.items())
        return [
            dict(items[i : i + batch_size]) for i in range(0, len(items), batch_size)
        ]

    def relation_type_plausible(
        self, relation: str, head_type: Optional[str], tail_type: Optional[str]
    ) -> bool:
        """Check an extracted pair against the relation's type constraints."""
        if relation not in self.relation_labels:
            return False
        _, heads, tails = self.relation_labels[relation]
        if heads and head_type is not None and head_type not in heads:
            return False
        if tails and tail_type is not None and tail_type not in tails:
            return False
        return True

    # -- vocab export ----------------------------------------------------------

    def vocab(self) -> Dict[str, Dict[str, int]]:
        """1-indexed vocab maps in the format CognitiveCollator expects.

        Coarse entity type names are included as (top-level) concepts, so an
        entity whose fine-grained labels fail validation can still carry its
        coarse type as a concept.
        """
        all_concepts = sorted(set(self.concept_labels) | set(self.entity_types))
        return {
            "concepts": {c: i + 1 for i, c in enumerate(all_concepts)},
            "relations": {r: i + 1 for i, r in enumerate(sorted(self.relation_labels))},
            "entity_types": dict(self.entity_type_to_id),
            "concept_to_entity_type": self.concept_to_entity_type_map(),
        }

    def concept_to_entity_type_map(self) -> Dict[str, int]:
        """Concept name -> entity type index (for SoftLogicConfig / collator)."""
        out = {
            c: self.entity_type_to_id[parent]
            for c, parent in self.concept_to_type.items()
        }
        out.update(self.entity_type_to_id)
        return out


def get_default_taxonomy() -> Taxonomy:
    return Taxonomy()
