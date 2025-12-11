"""Staged data pipeline for multi-stage training.

This module provides separate data formats and loaders for each training stage:
- Stage 1: Entity/Relation extraction (DocRED, TACRED format)
- Stage 2: Instruction tuning (simple input/output pairs)
- Stage 3: Joint fine-tuning (combined format)

This follows the standard LLM training paradigm:
1. Pre-train backbone (we use pre-trained T5/LongT5)
2. Train task-specific heads on labeled data
3. Instruction-tune for response generation
4. Joint fine-tune to integrate all components
"""

import json
import random
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from torch.utils.data import Dataset


# =============================================================================
# Stage 1: Entity/Relation Dataset
# =============================================================================

@dataclass
class EntityRelationSample:
    """Sample for entity/relation training (Stage 1)."""
    text: str
    entities: List[str]
    entity_types: List[str]  # e.g., ["person", "location", "organization"]
    entity_spans: List[Tuple[int, int]]  # character spans
    relations: List[Tuple[int, int, str]]  # (head_idx, tail_idx, relation_type)
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "entities": self.entities,
            "entity_types": self.entity_types,
            "entity_spans": self.entity_spans,
            "relations": self.relations,
            "should_respond": 0,  # Stage 1 doesn't need responses
            "response": "",
        }


class EntityRelationDataset(Dataset):
    """Dataset for entity/relation training."""
    
    def __init__(self, samples: List[EntityRelationSample]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict:
        sample = self.samples[idx]
        return {
            "text": sample.text,
            "entities": sample.entities,
            "concepts": [[t] for t in sample.entity_types],  # Convert to concept format
            "entity_spans": sample.entity_spans,
            "relations": sample.relations,
            "should_respond": 0,
            "response": "",
        }


# =============================================================================
# Stage 2: Instruction Dataset  
# =============================================================================

@dataclass
class InstructionSample:
    """Sample for instruction tuning (Stage 2)."""
    instruction: str
    response: str
    context: Optional[str] = None
    
    def to_dict(self) -> Dict:
        text = self.instruction
        if self.context:
            text = f"{self.instruction}\n\nContext: {self.context}"
        return {
            "text": text,
            "entities": [],
            "concepts": [],
            "relations": [],
            "should_respond": 1,
            "response": self.response,
        }


class InstructionDataset(Dataset):
    """Dataset for instruction tuning - simple input/output pairs."""
    
    def __init__(self, samples: List[InstructionSample]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict:
        return self.samples[idx].to_dict()


# =============================================================================
# Data Loaders
# =============================================================================

class REBELLoader:
    """Load REBEL dataset for entity/relation training.
    
    REBEL (Relation Extraction By End-to-end Language generation) is a large-scale
    dataset linking Wikipedia text to Wikidata relations. It's cleaner and easier
    to load than DocRED.
    
    Format: Each sample has context text and linearized triplets like:
        <triplet> Subject <subj> Relation <obj> Object <triplet> ...
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load(
        self, 
        split: str = "train", 
        max_samples: Optional[int] = None
    ) -> List[EntityRelationSample]:
        """Load REBEL and convert to EntityRelationSample format."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        print(f"Loading REBEL dataset ({split}) from HuggingFace...")
        
        dataset = load_dataset(
            "Babelscape/rebel-dataset",
            split=split,
            cache_dir=str(self.cache_dir)
        )
        
        samples = []
        skipped = 0
        
        for item in dataset:
            sample = self._convert_item(item)
            if sample:
                samples.append(sample)
                if max_samples and len(samples) >= max_samples:
                    break
            else:
                skipped += 1
        
        print(f"Loaded {len(samples)} samples from REBEL {split} (skipped {skipped})")
        return samples
    
    def _parse_triplets(self, triplet_str: str) -> List[Tuple[str, str, str]]:
        """Parse linearized triplet string into (subject, relation, object) tuples.
        
        Format: <triplet> Subject <subj> Relation <obj> Object <triplet> ...
        """
        triplets = []
        
        # Split by <triplet> marker
        parts = triplet_str.split("<triplet>")
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Parse: Subject <subj> Relation <obj> Object
            if "<subj>" in part and "<obj>" in part:
                try:
                    # Split at <subj>
                    subj_split = part.split("<subj>")
                    if len(subj_split) < 2:
                        continue
                    
                    subject = subj_split[0].strip()
                    rest = subj_split[1]
                    
                    # Split at <obj>
                    obj_split = rest.split("<obj>")
                    if len(obj_split) < 2:
                        continue
                    
                    relation = obj_split[0].strip()
                    obj = obj_split[1].strip()
                    
                    # Clean up any trailing markers
                    obj = obj.split("<")[0].strip()
                    
                    if subject and relation and obj:
                        triplets.append((subject, relation, obj))
                        
                except Exception:
                    continue
        
        return triplets
    
    def _find_entity_span(self, text: str, entity: str) -> Tuple[int, int]:
        """Find character span of entity in text."""
        text_lower = text.lower()
        entity_lower = entity.lower()
        
        idx = text_lower.find(entity_lower)
        if idx >= 0:
            return (idx, idx + len(entity))
        
        # Fallback: return placeholder span
        return (0, min(len(entity), len(text)))
    
    def _convert_item(self, item: Dict) -> Optional[EntityRelationSample]:
        """Convert a REBEL item to EntityRelationSample."""
        context = item.get("context", "")
        triplet_str = item.get("triplets", "")
        
        if not context or not triplet_str:
            return None
        
        # Parse triplets
        triplets = self._parse_triplets(triplet_str)
        
        if not triplets:
            return None
        
        # Extract unique entities and build index
        entity_to_idx = {}
        entities = []
        entity_types = []
        entity_spans = []
        
        for subj, rel, obj in triplets:
            for ent in [subj, obj]:
                if ent not in entity_to_idx:
                    entity_to_idx[ent] = len(entities)
                    entities.append(ent)
                    # Infer basic entity type (REBEL doesn't provide types)
                    entity_types.append("entity")
                    entity_spans.append(self._find_entity_span(context, ent))
        
        # Build relations with entity indices
        relations = []
        for subj, rel, obj in triplets:
            head_idx = entity_to_idx.get(subj, -1)
            tail_idx = entity_to_idx.get(obj, -1)
            if head_idx >= 0 and tail_idx >= 0:
                relations.append((head_idx, tail_idx, rel))
        
        if not entities or not relations:
            return None
        
        return EntityRelationSample(
            text=context,
            entities=entities,
            entity_types=entity_types,
            entity_spans=entity_spans,
            relations=relations
        )


class DollyLoader:
    """Load Dolly dataset for instruction tuning."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load(self, max_samples: Optional[int] = None) -> List[InstructionSample]:
        """Load Dolly and convert to InstructionSample format."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        print("Loading Dolly dataset from HuggingFace...")
        
        dataset = load_dataset(
            "databricks/databricks-dolly-15k",
            cache_dir=str(self.cache_dir)
        )
        
        samples = []
        for item in dataset["train"]:
            instruction = item.get("instruction", "")
            response = item.get("response", "")
            context = item.get("context", "")
            
            if instruction and response:
                samples.append(InstructionSample(
                    instruction=instruction,
                    response=response,
                    context=context if context else None
                ))
                
                if max_samples and len(samples) >= max_samples:
                    break
        
        print(f"Loaded {len(samples)} instruction samples from Dolly")
        return samples


class AlpacaLoader:
    """Load Alpaca dataset for instruction tuning."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load(self, max_samples: Optional[int] = None) -> List[InstructionSample]:
        """Load Alpaca and convert to InstructionSample format."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        print("Loading Alpaca dataset from HuggingFace...")
        
        dataset = load_dataset(
            "tatsu-lab/alpaca",
            cache_dir=str(self.cache_dir)
        )
        
        samples = []
        for item in dataset["train"]:
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            input_text = item.get("input", "")
            
            if instruction and output:
                samples.append(InstructionSample(
                    instruction=instruction,
                    response=output,
                    context=input_text if input_text else None
                ))
                
                if max_samples and len(samples) >= max_samples:
                    break
        
        print(f"Loaded {len(samples)} instruction samples from Alpaca")
        return samples


# =============================================================================
# Staged Pipeline
# =============================================================================

class StagedDataPipeline:
    """Pipeline for preparing stage-specific training data."""
    
    def __init__(self, cache_dir: str = "data/cache", output_dir: str = "data/staged"):
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rebel_loader = REBELLoader(str(self.cache_dir))
        self.dolly_loader = DollyLoader(str(self.cache_dir))
        self.alpaca_loader = AlpacaLoader(str(self.cache_dir))
    
    def prepare_stage1_data(
        self,
        max_samples: Optional[int] = None,
        split: str = "train",
        save: bool = True
    ) -> EntityRelationDataset:
        """Prepare data for Stage 1: Entity/Relation training.
        
        Uses REBEL dataset for entity and relation extraction training.
        REBEL is cleaner and easier to load than DocRED.
        """
        print("\n" + "=" * 60)
        print("Preparing Stage 1: Entity/Relation Dataset (REBEL)")
        print("=" * 60)
        
        samples = self.rebel_loader.load(split, max_samples)
        
        if save:
            path = self.output_dir / "stage1_entity_relation.jsonl"
            with open(path, 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample.to_dict()) + "\n")
            print(f"Saved to {path}")
        
        return EntityRelationDataset(samples)
    
    def prepare_stage2_data(
        self,
        sources: List[str] = ["dolly"],
        max_samples_per_source: Optional[int] = None,
        save: bool = True
    ) -> InstructionDataset:
        """Prepare data for Stage 2: Instruction tuning.
        
        Uses instruction datasets (Dolly, Alpaca) for decoder training.
        """
        print("\n" + "=" * 60)
        print("Preparing Stage 2: Instruction Dataset")
        print("=" * 60)
        
        all_samples = []
        
        for source in sources:
            if source == "dolly":
                samples = self.dolly_loader.load(max_samples_per_source)
            elif source == "alpaca":
                samples = self.alpaca_loader.load(max_samples_per_source)
            else:
                print(f"Unknown source: {source}, skipping")
                continue
            
            all_samples.extend(samples)
        
        # Shuffle
        random.shuffle(all_samples)
        
        if save:
            path = self.output_dir / "stage2_instruction.jsonl"
            with open(path, 'w') as f:
                for sample in all_samples:
                    f.write(json.dumps(sample.to_dict()) + "\n")
            print(f"Saved to {path}")
        
        return InstructionDataset(all_samples)
    
    def prepare_all(
        self,
        max_stage1_samples: Optional[int] = None,
        max_stage2_samples_per_source: Optional[int] = None,
        stage2_sources: List[str] = ["dolly", "alpaca"]
    ) -> Tuple[EntityRelationDataset, InstructionDataset]:
        """Prepare all training data."""
        
        stage1_data = self.prepare_stage1_data(max_stage1_samples)
        stage2_data = self.prepare_stage2_data(stage2_sources, max_stage2_samples_per_source)
        
        print("\n" + "=" * 60)
        print("Data Preparation Complete")
        print("=" * 60)
        print(f"Stage 1 (Entity/Relation): {len(stage1_data)} samples")
        print(f"Stage 2 (Instruction): {len(stage2_data)} samples")
        
        return stage1_data, stage2_data
    
    def get_stage1_path(self) -> Path:
        return self.output_dir / "stage1_entity_relation.jsonl"
    
    def get_stage2_path(self) -> Path:
        return self.output_dir / "stage2_instruction.jsonl"


def get_entity_types_from_dataset(dataset) -> List[str]:
    """Extract unique entity types from dataset."""
    types = set()
    for sample in dataset:
        concepts = sample.get("concepts", [])
        for concept_list in concepts:
            if isinstance(concept_list, list):
                types.update(concept_list)
            else:
                types.add(concept_list)
    return sorted(types)


def get_relations_from_dataset(dataset) -> List[str]:
    """Extract unique relation types from dataset."""
    relations = set()
    for sample in dataset:
        for rel in sample.get("relations", []):
            if len(rel) >= 3:
                relations.add(rel[2])
    return sorted(relations)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare staged training data")
    parser.add_argument("--stage", type=str, choices=["1", "2", "all"], default="all",
                       help="Which stage to prepare")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Max samples per dataset")
    parser.add_argument("--output-dir", type=str, default="data/staged",
                       help="Output directory")
    args = parser.parse_args()
    
    pipeline = StagedDataPipeline(output_dir=args.output_dir)
    
    if args.stage == "1":
        pipeline.prepare_stage1_data(args.max_samples)
    elif args.stage == "2":
        pipeline.prepare_stage2_data(["dolly", "alpaca"], args.max_samples)
    else:
        pipeline.prepare_all(args.max_samples, args.max_samples)
