"""Data pipeline for downloading, converting, and preparing production datasets.

This module provides utilities for:
1. Downloading standard NLP datasets (DocRED, MetaQA, Dolly, etc.)
2. Converting them to the neurosymbolic format
3. Creating unified training datasets with proper splits

Supported datasets:
- DocRED: Document-level relation extraction
- MetaQA: Knowledge graph QA
- Dolly 2.0: Instruction following
- OASST1: Multi-turn conversations
- Natural Questions: Factoid QA
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    source: str  # 'huggingface', 'url', 'local'
    hf_path: Optional[str] = None
    hf_subset: Optional[str] = None
    url: Optional[str] = None
    local_path: Optional[str] = None
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)  # train, val, test
    max_samples: Optional[int] = None
    seed: int = 42


@dataclass
class ConvertedSample:
    """A sample in the neurosymbolic format."""
    text: str
    entities: List[str]
    concepts: List[List[str]]
    relations: List[Tuple[int, int, str]]
    should_respond: int
    response: str = ""
    entity_spans: Optional[List[Tuple[int, int]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "text": self.text,
            "entities": self.entities,
            "concepts": self.concepts,
            "relations": [[r[0], r[1], r[2]] for r in self.relations],
            "should_respond": self.should_respond,
            "response": self.response,
        }
        if self.entity_spans:
            d["entity_spans"] = self.entity_spans
        if self.metadata:
            d["_metadata"] = self.metadata
        return d


class DatasetConverter(ABC):
    """Abstract base class for dataset converters."""
    
    @abstractmethod
    def convert(self, data: Any) -> List[ConvertedSample]:
        """Convert raw data to neurosymbolic format."""
        pass
    
    @abstractmethod
    def get_relation_types(self) -> List[str]:
        """Get all relation types in this dataset."""
        pass
    
    @abstractmethod
    def get_entity_types(self) -> List[str]:
        """Get all entity types in this dataset."""
        pass


class DocREDConverter(DatasetConverter):
    """Converter for DocRED dataset."""
    
    # DocRED relation to natural language mapping
    RELATION_NAMES = {
        "P6": "head_of_government",
        "P17": "country",
        "P19": "place_of_birth",
        "P20": "place_of_death",
        "P22": "father",
        "P25": "mother",
        "P26": "spouse",
        "P27": "country_of_citizenship",
        "P30": "continent",
        "P31": "instance_of",
        "P35": "head_of_state",
        "P36": "capital",
        "P37": "official_language",
        "P39": "position_held",
        "P40": "child",
        "P50": "author",
        "P54": "member_of_sports_team",
        "P57": "director",
        "P58": "screenwriter",
        "P69": "educated_at",
        "P86": "composer",
        "P102": "member_of_political_party",
        "P108": "employer",
        "P112": "founded_by",
        "P118": "league",
        "P123": "publisher",
        "P127": "owned_by",
        "P131": "located_in_admin_territory",
        "P136": "genre",
        "P137": "operator",
        "P140": "religion",
        "P150": "contains_admin_territory",
        "P155": "follows",
        "P156": "followed_by",
        "P159": "headquarters_location",
        "P161": "cast_member",
        "P162": "producer",
        "P166": "award_received",
        "P170": "creator",
        "P171": "parent_taxon",
        "P172": "ethnic_group",
        "P175": "performer",
        "P176": "manufacturer",
        "P178": "developer",
        "P179": "series",
        "P190": "sister_city",
        "P194": "legislative_body",
        "P205": "basin_country",
        "P206": "located_in_body_of_water",
        "P241": "military_branch",
        "P264": "record_label",
        "P272": "production_company",
        "P276": "location",
        "P279": "subclass_of",
        "P355": "subsidiary",
        "P361": "part_of",
        "P364": "original_language",
        "P400": "platform",
        "P403": "mouth_of_watercourse",
        "P449": "original_network",
        "P463": "member_of",
        "P488": "chairperson",
        "P495": "country_of_origin",
        "P527": "has_part",
        "P551": "residence",
        "P569": "date_of_birth",
        "P570": "date_of_death",
        "P571": "inception",
        "P576": "dissolved",
        "P577": "publication_date",
        "P580": "start_time",
        "P582": "end_time",
        "P607": "conflict",
        "P674": "characters",
        "P676": "lyrics_by",
        "P706": "located_on_terrain_feature",
        "P710": "participant",
        "P737": "influenced_by",
        "P740": "location_of_formation",
        "P749": "parent_organization",
        "P800": "notable_work",
        "P807": "separated_from",
        "P840": "narrative_location",
        "P937": "work_location",
        "P1001": "jurisdiction",
        "P1056": "product",
        "P1198": "unemployment_rate",
        "P1336": "territory_claimed_by",
        "P1344": "participant_of",
        "P1365": "replaces",
        "P1366": "replaced_by",
        "P1376": "capital_of",
        "P1412": "languages_spoken",
        "P3373": "sibling",
    }
    
    # Entity type mapping
    ENTITY_TYPES = {
        "PER": "person",
        "ORG": "organization",
        "LOC": "location",
        "TIME": "temporal",
        "NUM": "quantity",
        "MISC": "object",
    }
    
    # Question templates for generating QA pairs
    QUESTION_TEMPLATES = {
        "P17": ("What country is {head} in?", "{head} is in {tail}."),
        "P19": ("Where was {head} born?", "{head} was born in {tail}."),
        "P20": ("Where did {head} die?", "{head} died in {tail}."),
        "P26": ("Who is the spouse of {head}?", "{tail} is the spouse of {head}."),
        "P27": ("What is the nationality of {head}?", "{head} is from {tail}."),
        "P36": ("What is the capital of {head}?", "The capital of {head} is {tail}."),
        "P50": ("Who wrote {head}?", "{head} was written by {tail}."),
        "P57": ("Who directed {head}?", "{head} was directed by {tail}."),
        "P69": ("Where was {head} educated?", "{head} was educated at {tail}."),
        "P108": ("Who does {head} work for?", "{head} works for {tail}."),
        "P112": ("Who founded {head}?", "{head} was founded by {tail}."),
        "P127": ("Who owns {head}?", "{head} is owned by {tail}."),
        "P131": ("Where is {head} located?", "{head} is located in {tail}."),
        "P159": ("Where is the headquarters of {head}?", "The headquarters of {head} is in {tail}."),
        "P161": ("Who starred in {head}?", "{tail} starred in {head}."),
        "P170": ("Who created {head}?", "{head} was created by {tail}."),
        "P175": ("Who performed {head}?", "{head} was performed by {tail}."),
        "P176": ("Who manufactured {head}?", "{head} was manufactured by {tail}."),
        "P178": ("Who developed {head}?", "{head} was developed by {tail}."),
        "P276": ("Where is {head} located?", "{head} is located at {tail}."),
        "P495": ("What country is {head} from?", "{head} is from {tail}."),
        "P569": ("When was {head} born?", "{head} was born on {tail}."),
        "P570": ("When did {head} die?", "{head} died on {tail}."),
        "P571": ("When was {head} established?", "{head} was established in {tail}."),
        "P577": ("When was {head} published?", "{head} was published on {tail}."),
        "P740": ("Where was {head} formed?", "{head} was formed in {tail}."),
        "P800": ("What is a notable work by {head}?", "{tail} is a notable work by {head}."),
    }
    
    def __init__(self, include_qa: bool = True, max_qa_per_doc: int = 3):
        self.include_qa = include_qa
        self.max_qa_per_doc = max_qa_per_doc
    
    def convert(self, data: List[Dict]) -> List[ConvertedSample]:
        samples = []
        for doc in data:
            samples.extend(self._convert_document(doc))
        return samples
    
    def _convert_document(self, doc: Dict) -> List[ConvertedSample]:
        samples = []
        
        # Extract entities
        entities = []
        entity_types = []
        entity_spans = []
        
        for ent_idx, entity in enumerate(doc.get("vertexSet", [])):
            name = entity[0]["name"]
            etype = entity[0].get("type", "MISC")
            entities.append(name)
            entity_types.append(self.ENTITY_TYPES.get(etype, "object"))
            
            # Get span from first mention
            first_mention = entity[0]
            sent_idx = first_mention.get("sent_id", 0)
            pos = first_mention.get("pos", [0, 1])
            
            # Calculate absolute position (approximate)
            offset = sum(len(doc["sents"][i]) for i in range(sent_idx))
            entity_spans.append((offset + pos[0], offset + pos[1] - 1))
        
        # Build concepts from entity types
        concepts = [[etype] for etype in entity_types]
        
        # Extract relations
        relations = []
        for label in doc.get("labels", []):
            head_idx = label["h"]
            tail_idx = label["t"]
            rel_type = label["r"]
            rel_name = self.RELATION_NAMES.get(rel_type, rel_type)
            relations.append((head_idx, tail_idx, rel_name))
        
        # Combine sentences into text
        text = " ".join([" ".join(sent) for sent in doc.get("sents", [])])
        
        # Create statement sample (should_respond=0)
        statement = ConvertedSample(
            text=text,
            entities=entities,
            concepts=concepts,
            relations=relations,
            should_respond=0,
            response="",
            entity_spans=entity_spans,
            metadata={"source": "docred", "doc_title": doc.get("title", "")}
        )
        samples.append(statement)
        
        # Generate QA samples
        if self.include_qa:
            qa_count = 0
            for head_idx, tail_idx, rel_name in relations:
                if qa_count >= self.max_qa_per_doc:
                    break
                
                # Find original relation ID for template lookup
                rel_id = None
                for rid, rname in self.RELATION_NAMES.items():
                    if rname == rel_name:
                        rel_id = rid
                        break
                
                if rel_id and rel_id in self.QUESTION_TEMPLATES:
                    q_template, a_template = self.QUESTION_TEMPLATES[rel_id]
                    head_name = entities[head_idx]
                    tail_name = entities[tail_idx]
                    
                    question = q_template.format(head=head_name, tail=tail_name)
                    answer = a_template.format(head=head_name, tail=tail_name)
                    
                    qa_sample = ConvertedSample(
                        text=question,
                        entities=entities,
                        concepts=concepts,
                        relations=relations,
                        should_respond=1,
                        response=answer,
                        metadata={"source": "docred_qa", "relation": rel_name}
                    )
                    samples.append(qa_sample)
                    qa_count += 1
        
        return samples
    
    def get_relation_types(self) -> List[str]:
        return list(self.RELATION_NAMES.values())
    
    def get_entity_types(self) -> List[str]:
        return list(set(self.ENTITY_TYPES.values()))


class MetaQAConverter(DatasetConverter):
    """Converter for MetaQA dataset."""
    
    ENTITY_TYPES = ["movie", "actor", "director", "writer", "genre", "year", "language"]
    RELATION_TYPES = [
        "directed_by", "written_by", "starred_actors", "has_genre",
        "release_year", "in_language", "has_tags", "has_imdb_rating", "has_imdb_votes"
    ]
    
    def __init__(self, include_kg_context: bool = True):
        self.include_kg_context = include_kg_context
    
    def convert(self, data: Dict) -> List[ConvertedSample]:
        """Convert MetaQA data.
        
        Expects data with 'questions' and 'kb' keys.
        """
        samples = []
        questions = data.get("questions", [])
        kb = data.get("kb", {})
        
        for q_data in questions:
            question = q_data.get("question", "")
            answers = q_data.get("answers", [])
            entities_in_q = q_data.get("entities", [])
            
            if not question or not answers:
                continue
            
            # Build entity list from question entities and answers
            all_entities = list(entities_in_q) + [a for a in answers if a not in entities_in_q]
            
            # Infer concepts (simplified)
            concepts = []
            for ent in all_entities:
                if ent in kb.get("movies", []):
                    concepts.append(["movie"])
                elif ent in kb.get("actors", []):
                    concepts.append(["actor", "person"])
                elif ent in kb.get("directors", []):
                    concepts.append(["director", "person"])
                else:
                    concepts.append(["object"])
            
            # Build relations from KB if available
            relations = []
            for i, ent1 in enumerate(all_entities):
                for j, ent2 in enumerate(all_entities):
                    if i != j:
                        rel = self._find_relation(ent1, ent2, kb)
                        if rel:
                            relations.append((i, j, rel))
            
            # Format answer
            if len(answers) == 1:
                response = answers[0]
            else:
                response = ", ".join(answers[:-1]) + " and " + answers[-1]
            
            sample = ConvertedSample(
                text=question,
                entities=all_entities,
                concepts=concepts,
                relations=relations,
                should_respond=1,
                response=response,
                metadata={"source": "metaqa", "hop": q_data.get("hop", 1)}
            )
            samples.append(sample)
        
        return samples
    
    def _find_relation(self, ent1: str, ent2: str, kb: Dict) -> Optional[str]:
        """Find relation between two entities in KB."""
        triples = kb.get("triples", [])
        for h, r, t in triples:
            if h == ent1 and t == ent2:
                return r
        return None
    
    def get_relation_types(self) -> List[str]:
        return self.RELATION_TYPES
    
    def get_entity_types(self) -> List[str]:
        return self.ENTITY_TYPES


class InstructionConverter(DatasetConverter):
    """Converter for instruction-following datasets (Dolly, Alpaca, OASST)."""
    
    ENTITY_TYPES = ["person", "organization", "location", "object", "concept", "action"]
    RELATION_TYPES = ["related_to", "part_of", "causes", "describes", "example_of"]
    
    def __init__(self, extract_entities: bool = True):
        self.extract_entities = extract_entities
    
    def convert(self, data: List[Dict]) -> List[ConvertedSample]:
        samples = []
        for item in data:
            sample = self._convert_item(item)
            if sample:
                samples.append(sample)
        return samples
    
    def _convert_item(self, item: Dict) -> Optional[ConvertedSample]:
        """Convert a single instruction item."""
        # Handle different formats
        instruction = item.get("instruction") or item.get("prompt") or item.get("input", "")
        context = item.get("context", "") or item.get("input", "")
        response = item.get("response") or item.get("output") or item.get("completion", "")
        
        if not instruction or not response:
            return None
        
        # Combine instruction and context
        if context and context != instruction:
            text = f"{instruction}\n\nContext: {context}"
        else:
            text = instruction
        
        # Extract entities (simple NER-like extraction)
        entities = []
        concepts = []
        if self.extract_entities:
            entities, concepts = self._simple_entity_extraction(text + " " + response)
        
        return ConvertedSample(
            text=text,
            entities=entities,
            concepts=concepts,
            relations=[],  # No explicit relations in instruction data
            should_respond=1,
            response=response,
            metadata={
                "source": item.get("source", "instruction"),
                "category": item.get("category", "general")
            }
        )
    
    def _simple_entity_extraction(self, text: str) -> Tuple[List[str], List[List[str]]]:
        """Simple entity extraction using capitalization heuristics."""
        import re
        
        entities = []
        concepts = []
        
        # Find capitalized phrases (simple NER approximation)
        # This is a placeholder - in production, use spaCy or similar
        words = text.split()
        i = 0
        while i < len(words):
            word = words[i]
            # Skip common sentence starters
            if i == 0 or words[i-1].endswith(('.', '?', '!')):
                i += 1
                continue
            
            # Check for capitalized word that might be an entity
            if word[0].isupper() and len(word) > 1:
                entity_words = [word]
                j = i + 1
                # Collect consecutive capitalized words
                while j < len(words) and words[j][0].isupper():
                    entity_words.append(words[j])
                    j += 1
                
                entity = " ".join(entity_words)
                entity = re.sub(r'[^\w\s]', '', entity).strip()
                
                if entity and len(entity) > 1 and entity not in entities:
                    entities.append(entity)
                    concepts.append(["object"])  # Default concept
                
                i = j
            else:
                i += 1
        
        return entities[:20], concepts[:20]  # Limit entities
    
    def get_relation_types(self) -> List[str]:
        return self.RELATION_TYPES
    
    def get_entity_types(self) -> List[str]:
        return self.ENTITY_TYPES


class DataPipeline:
    """Main data pipeline for preparing training data."""
    
    SUPPORTED_DATASETS = {
        "docred": {
            "source": "github",
            "urls": {
                "train": "https://raw.githubusercontent.com/thunlp/DocRED/master/data/train_annotated.json",
                "dev": "https://raw.githubusercontent.com/thunlp/DocRED/master/data/dev.json",
            },
            "converter": DocREDConverter,
        },
        "dolly": {
            "source": "huggingface",
            "hf_path": "databricks/databricks-dolly-15k",
            "converter": InstructionConverter,
        },
        "oasst1": {
            "source": "huggingface",
            "hf_path": "OpenAssistant/oasst1",
            "converter": InstructionConverter,
        },
        "alpaca": {
            "source": "huggingface",
            "hf_path": "tatsu-lab/alpaca",
            "converter": InstructionConverter,
        },
    }
    
    def __init__(
        self,
        output_dir: str = "data/processed",
        cache_dir: str = "data/cache",
        seed: int = 42
    ):
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.seed = seed
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        random.seed(seed)
    
    def download_dataset(self, name: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """Download a dataset."""
        if name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Supported: {list(self.SUPPORTED_DATASETS.keys())}")
        
        config = self.SUPPORTED_DATASETS[name]
        
        if config["source"] == "huggingface":
            return self._download_hf(config["hf_path"], max_samples)
        elif config["source"] == "github":
            return self._download_github(name, config["urls"], max_samples)
        else:
            raise ValueError(f"Unknown source: {config['source']}")
    
    def _download_github(self, name: str, urls: Dict[str, str], max_samples: Optional[int] = None) -> Dict[str, Any]:
        """Download dataset directly from GitHub URLs."""
        import urllib.request
        
        result = {}
        
        for split, url in urls.items():
            print(f"Downloading {name}/{split} from {url}...")
            
            cache_file = self.cache_dir / f"{name}_{split}.json"
            
            # Check if cached
            if cache_file.exists():
                print(f"  Using cached file: {cache_file}")
                with open(cache_file) as f:
                    data = json.load(f)
            else:
                # Download
                try:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        content = response.read().decode('utf-8')
                        data = json.loads(content)
                        
                        # Cache it
                        with open(cache_file, 'w') as f:
                            f.write(content)
                        print(f"  Downloaded and cached: {len(data)} items")
                except Exception as e:
                    print(f"  Error downloading {split}: {e}")
                    continue
            
            # Apply max_samples limit
            if max_samples and len(data) > max_samples:
                data = data[:max_samples]
            
            result[split] = data
        
        return result
    
    def _download_hf(self, path: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """Download from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        print(f"Downloading {path} from HuggingFace...")
        
        try:
            # Try without trust_remote_code first (newer datasets format)
            dataset = load_dataset(path, cache_dir=str(self.cache_dir))
        except Exception as e:
            print(f"Standard load failed: {e}")
            try:
                # Try streaming mode
                print("Trying streaming mode...")
                dataset = load_dataset(path, streaming=True)
                # Convert streaming dataset to regular
                result = {}
                for split in dataset.keys():
                    print(f"  Loading {split}...")
                    data = []
                    for i, item in enumerate(dataset[split]):
                        data.append(item)
                        if max_samples and i + 1 >= max_samples:
                            break
                    result[split] = data
                return result
            except Exception as e2:
                print(f"Streaming also failed: {e2}")
                raise RuntimeError(f"Could not load dataset {path}: {e}, {e2}")
        
        result = {}
        for split in dataset.keys():
            data = list(dataset[split])
            if max_samples:
                data = data[:max_samples]
            result[split] = data
            print(f"  Loaded {split}: {len(data)} samples")
        
        return result
    
    def convert_dataset(
        self,
        name: str,
        data: Dict[str, Any],
        converter_kwargs: Optional[Dict] = None
    ) -> Dict[str, List[ConvertedSample]]:
        """Convert a dataset to neurosymbolic format."""
        config = self.SUPPORTED_DATASETS[name]
        converter_cls = config["converter"]
        converter = converter_cls(**(converter_kwargs or {}))
        
        result = {}
        for split, split_data in data.items():
            print(f"Converting {name}/{split}: {len(split_data)} samples...")
            converted = converter.convert(split_data)
            result[split] = converted
            print(f"  -> {len(converted)} samples after conversion")
        
        return result
    
    def save_dataset(
        self,
        name: str,
        data: Dict[str, List[ConvertedSample]],
        format: str = "jsonl"
    ) -> Dict[str, Path]:
        """Save converted dataset to disk."""
        paths = {}
        
        for split, samples in data.items():
            filename = f"{name}_{split}.{format}"
            path = self.output_dir / filename
            
            with open(path, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample.to_dict()) + "\n")
            
            paths[split] = path
            print(f"Saved {len(samples)} samples to {path}")
        
        return paths
    
    def prepare_dataset(
        self,
        name: str,
        max_samples: Optional[int] = None,
        converter_kwargs: Optional[Dict] = None
    ) -> Dict[str, Path]:
        """Full pipeline: download, convert, save."""
        data = self.download_dataset(name, max_samples)
        converted = self.convert_dataset(name, data, converter_kwargs)
        paths = self.save_dataset(name, converted)
        return paths
    
    def merge_datasets(
        self,
        datasets: List[Path],
        output_name: str,
        shuffle: bool = True,
        ratios: Optional[List[float]] = None
    ) -> Path:
        """Merge multiple datasets into one."""
        all_samples = []
        
        for path in datasets:
            with open(path) as f:
                for line in f:
                    if line.strip():
                        all_samples.append(json.loads(line))
        
        if shuffle:
            random.shuffle(all_samples)
        
        # Apply ratios if specified (for balancing)
        if ratios:
            # Group by source
            by_source = {}
            for sample in all_samples:
                source = sample.get("_metadata", {}).get("source", "unknown")
                if source not in by_source:
                    by_source[source] = []
                by_source[source].append(sample)
            
            # Sample according to ratios
            total = len(all_samples)
            sampled = []
            sources = list(by_source.keys())
            for i, source in enumerate(sources):
                if i < len(ratios):
                    n = int(total * ratios[i])
                    sampled.extend(random.sample(by_source[source], min(n, len(by_source[source]))))
            all_samples = sampled
            
            if shuffle:
                random.shuffle(all_samples)
        
        output_path = self.output_dir / f"{output_name}.jsonl"
        with open(output_path, "w") as f:
            for sample in all_samples:
                f.write(json.dumps(sample) + "\n")
        
        print(f"Merged {len(all_samples)} samples to {output_path}")
        return output_path
    
    def create_splits(
        self,
        input_path: Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Dict[str, Path]:
        """Create train/val/test splits from a single file."""
        with open(input_path) as f:
            samples = [json.loads(line) for line in f if line.strip()]
        
        random.shuffle(samples)
        
        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        splits = {
            "train": samples[:train_end],
            "val": samples[train_end:val_end],
            "test": samples[val_end:]
        }
        
        paths = {}
        base_name = input_path.stem
        
        for split_name, split_data in splits.items():
            path = self.output_dir / f"{base_name}_{split_name}.jsonl"
            with open(path, "w") as f:
                for sample in split_data:
                    f.write(json.dumps(sample) + "\n")
            paths[split_name] = path
            print(f"Created {split_name} split: {len(split_data)} samples -> {path}")
        
        return paths


def build_vocab_from_datasets(
    dataset_paths: List[Path],
    output_dir: Path
) -> Dict[str, Dict[str, int]]:
    """Build vocabulary mappings from converted datasets."""
    all_entities = set()
    all_concepts = set()
    all_relations = set()
    
    for path in dataset_paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                
                # Collect entity types from concepts
                for concept_list in sample.get("concepts", []):
                    for c in concept_list:
                        all_concepts.add(c)
                
                # Collect relations
                for rel in sample.get("relations", []):
                    if len(rel) >= 3:
                        all_relations.add(rel[2])
    
    # Build mappings
    concept_map = {c: i for i, c in enumerate(sorted(all_concepts))}
    relation_map = {r: i for i, r in enumerate(sorted(all_relations))}
    
    # Entity type mapping (derived from concepts)
    entity_types = [
        "person", "organization", "location", "temporal", "quantity",
        "object", "event", "concept", "attribute", "action"
    ]
    entity_type_map = {e: i for i, e in enumerate(entity_types)}
    
    # Save mappings
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mappings = {
        "concepts": concept_map,
        "relations": relation_map,
        "entity_types": entity_type_map
    }
    
    with open(output_dir / "vocab_mappings.json", "w") as f:
        json.dump(mappings, f, indent=2)
    
    print(f"Built vocabulary: {len(concept_map)} concepts, {len(relation_map)} relations")
    return mappings


if __name__ == "__main__":
    # Example usage
    pipeline = DataPipeline(output_dir="data/processed", cache_dir="data/cache")
    
    # Prepare Dolly dataset
    print("Preparing Dolly dataset...")
    dolly_paths = pipeline.prepare_dataset("dolly", max_samples=5000)
    
    print("\nDone!")
