"""Training utilities for extraction and configuration."""

from typing import Dict, List, Optional


def extract_concept_to_entity_type_map(
    dataset, 
    n_entity_types: Optional[int] = None, 
    base_concept_priority: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Dynamically extract concept-to-entity-type mapping from dataset.
    
    Args:
        dataset: ToyCognitiveDataset instance
        n_entity_types: Maximum number of entity types. None = use all found.
        base_concept_priority: Ordered list of base concept names to prioritize.
    
    Returns:
        Mapping from concept names to entity type indices (0-indexed)
    """
    if base_concept_priority is None:
        base_concept_priority = ["animal", "person", "location", "object", "attribute"]
    
    # Extract all concepts
    all_concepts = set()
    concept_frequencies = {}
    
    for sample in dataset:
        concepts = sample.get("concepts", [])
        for concept_list in concepts:
            if isinstance(concept_list, str):
                concept_list = [concept_list]
            elif not isinstance(concept_list, list):
                concept_list = [str(concept_list)]
            
            for concept in concept_list:
                all_concepts.add(concept)
                concept_frequencies[concept] = concept_frequencies.get(concept, 0) + 1
    
    # Build mapping
    concept_to_entity_type_map = {}
    entity_type_idx = 0
    
    # Priority concepts first
    for base_concept in base_concept_priority:
        if base_concept in all_concepts:
            if n_entity_types is None or entity_type_idx < n_entity_types:
                concept_to_entity_type_map[base_concept] = entity_type_idx
                entity_type_idx += 1
    
    # Add remaining by frequency
    remaining_concepts = sorted(
        [(c, f) for c, f in concept_frequencies.items() if c not in concept_to_entity_type_map],
        key=lambda x: x[1],
        reverse=True
    )
    
    for concept, freq in remaining_concepts:
        if n_entity_types is None or entity_type_idx < n_entity_types:
            concept_to_entity_type_map[concept] = entity_type_idx
            entity_type_idx += 1
        else:
            break
    
    if len(concept_to_entity_type_map) == 0:
        print("Warning: No concepts found in dataset. Using empty mapping.")
    
    return concept_to_entity_type_map


def extract_relation_map_from_dataset(dataset) -> Dict[str, int]:
    """
    Dynamically extract relation map from dataset.
    
    Args:
        dataset: ToyCognitiveDataset instance
    
    Returns:
        Mapping from relation names to indices (1-indexed, 0 is padding)
    """
    all_relations = set()
    
    for sample in dataset:
        relations = sample.get("relations", [])
        for rel in relations:
            if len(rel) >= 3:
                rel_name = rel[2]
                all_relations.add(rel_name)
    
    return {rel: idx + 1 for idx, rel in enumerate(sorted(all_relations))}


def generate_soft_logic_rules_from_dataset(
    dataset, 
    concept_to_entity_type_map: Dict[str, int], 
    relation_map: Dict[str, int],
    min_frequency: int = 2, 
    min_confidence: float = 0.3, 
    max_rules: int = 50, 
    include_negative_rules: bool = True
) -> List[Dict]:
    """
    Dynamically generate soft logic rules from dataset patterns.
    
    Args:
        dataset: ToyCognitiveDataset instance
        concept_to_entity_type_map: Mapping from concepts to entity types
        relation_map: Mapping from relations to indices
        min_frequency: Minimum pattern occurrences for positive rules
        min_confidence: Minimum confidence threshold
        max_rules: Maximum number of rules to generate
        include_negative_rules: Whether to generate negative rules
    
    Returns:
        List of rule dicts with concept_a, concept_b, relation, weight, polarity
    """
    # Count patterns
    pattern_counts = {}
    relation_totals = {}
    concept_pair_totals = {}
    
    for sample in dataset:
        entities = sample.get("entities", [])
        concepts = sample.get("concepts", [])
        relations = sample.get("relations", [])
        
        # Normalize concepts
        normalized_concepts = []
        for concept_list in concepts:
            if isinstance(concept_list, str):
                normalized_concepts.append([concept_list])
            elif isinstance(concept_list, list):
                normalized_concepts.append(concept_list)
            else:
                normalized_concepts.append([str(concept_list)])
        
        # Process relations
        for rel in relations:
            if len(rel) >= 3:
                head_idx, tail_idx, rel_name = rel[0], rel[1], rel[2]
                
                if head_idx >= len(normalized_concepts) or tail_idx >= len(normalized_concepts):
                    continue
                
                head_concepts = normalized_concepts[head_idx]
                tail_concepts = normalized_concepts[tail_idx]
                
                for head_concept in head_concepts:
                    for tail_concept in tail_concepts:
                        if head_concept in concept_to_entity_type_map and tail_concept in concept_to_entity_type_map:
                            pattern = (head_concept, tail_concept, rel_name)
                            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                            relation_totals[rel_name] = relation_totals.get(rel_name, 0) + 1
                            concept_pair = (head_concept, tail_concept)
                            concept_pair_totals[concept_pair] = concept_pair_totals.get(concept_pair, 0) + 1
    
    # Generate positive rules
    rules = []
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    
    for (concept_a, concept_b, relation), count in sorted_patterns:
        if len(rules) >= max_rules:
            break
        
        if relation not in relation_map:
            continue
        
        rel_total = relation_totals.get(relation, 1)
        confidence = count / rel_total if rel_total > 0 else 0
        
        if count >= min_frequency and confidence >= min_confidence:
            weight = min(1.0, count / max(1, min_frequency * 2)) * confidence
            
            rules.append({
                "concept_a": concept_a,
                "concept_b": concept_b,
                "relation": relation,
                "weight": float(weight),
                "polarity": 1
            })
    
    # Generate negative rules
    if include_negative_rules and len(rules) < max_rules:
        all_concept_pairs = set(concept_pair_totals.keys())
        all_relations = set(relation_map.keys())
        
        for concept_a, concept_b in list(all_concept_pairs)[:20]:
            if len(rules) >= max_rules:
                break
            
            pair_total = concept_pair_totals.get((concept_a, concept_b), 0)
            if pair_total < 2:
                continue
            
            for relation in all_relations:
                if len(rules) >= max_rules:
                    break
                
                pattern = (concept_a, concept_b, relation)
                count = pattern_counts.get(pattern, 0)
                
                if count == 0 or (count == 1 and pair_total >= 3):
                    has_other_relations = any(
                        pattern_counts.get((concept_a, concept_b, r), 0) > 0
                        for r in all_relations if r != relation
                    )
                    
                    if has_other_relations:
                        rules.append({
                            "concept_a": concept_a,
                            "concept_b": concept_b,
                            "relation": relation,
                            "weight": 0.3,
                            "polarity": -1
                        })
    
    return rules


def configure_soft_logic_rules(
    model, 
    concept_to_entity_type_map: Dict[str, int], 
    relation_map: Dict[str, int], 
    rules_config: List[Dict]
):
    """
    Configure soft logic rules for the model.
    
    Args:
        model: NeuroSymbolicLM model
        concept_to_entity_type_map: Mapping from concepts to entity types
        relation_map: Mapping from relations to indices
        rules_config: List of rule dicts with concept_a, concept_b, relation, weight, polarity
    """
    for rule in rules_config:
        etype_a = concept_to_entity_type_map.get(rule["concept_a"])
        etype_b = concept_to_entity_type_map.get(rule["concept_b"])
        rel_idx_1indexed = relation_map.get(rule["relation"])
        
        if etype_a is None or etype_b is None or rel_idx_1indexed is None:
            print(f"Warning: Skipping rule {rule} - invalid concept or relation mapping")
            continue
        
        rel_idx = rel_idx_1indexed - 1  # Convert to 0-indexed
        
        if rel_idx < 0 or rel_idx >= model.softlogic.n_relations:
            print(f"Warning: Skipping rule {rule} - relation index {rel_idx} out of bounds")
            continue
        
        weight = rule.get("weight", 1.0)
        polarity = rule.get("polarity", 1)
        
        model.softlogic.add_rule(etype_a, etype_b, rel_idx, weight, polarity)
    
    print(f"Configured {len(model.softlogic.rules)} soft logic rules")
