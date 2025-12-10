"""
Knowledge Graph utilities for loading embeddings, entity linking, and graph operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import json

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    # Create a dummy nx module for type hints
    class DummyNX:
        DiGraph = None
    nx = DummyNX()


class KGEmbeddingLoader:
    """Load and manage knowledge graph embeddings."""
    
    def __init__(self, kg_type: str = "conceptnet"):
        """
        Args:
            kg_type: Type of KG ("conceptnet", "wikidata", "wordnet")
        """
        self.kg_type = kg_type
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self.relation_embeddings: Dict[str, np.ndarray] = {}
        self.entity_to_id: Dict[str, int] = {}
        self.relation_to_id: Dict[str, int] = {}
        self.id_to_entity: Dict[int, str] = {}
        self.id_to_relation: Dict[int, str] = {}
    
    def load_conceptnet_embeddings(self, embedding_path: str, vocab_path: Optional[str] = None):
        """
        Load ConceptNet Numberbatch embeddings.
        
        Format: Each line is "entity <space-separated-vector>"
        """
        self.entity_embeddings = {}
        self.entity_to_id = {}
        self.id_to_entity = {}
        
        with open(embedding_path, 'r', encoding='utf-8') as f:
            # Skip header line
            header = f.readline().strip().split()
            dim = int(header[1])
            
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < dim + 1:
                    continue
                
                entity = parts[0]
                vector = np.array([float(x) for x in parts[1:dim+1]])
                
                self.entity_embeddings[entity] = vector
                self.entity_to_id[entity] = idx
                self.id_to_entity[idx] = entity
        
        print(f"Loaded {len(self.entity_embeddings)} ConceptNet embeddings (dim={dim})")
        return dim
    
    def load_wikidata_embeddings(self, embedding_path: str):
        """Load Wikidata2Vec embeddings."""
        # Similar structure to ConceptNet
        return self.load_conceptnet_embeddings(embedding_path)
    
    def get_embedding(self, entity: str) -> Optional[np.ndarray]:
        """Get embedding for an entity."""
        return self.entity_embeddings.get(entity)
    
    def get_embedding_tensor(self, entities: List[str], device: str = "cpu") -> torch.Tensor:
        """
        Get embeddings for a list of entities as a tensor.
        
        Args:
            entities: List of entity strings
            device: Device to place tensor on
        
        Returns:
            embeddings: (len(entities), embed_dim) tensor, zero-padded for missing entities
        """
        if len(entities) == 0:
            return torch.zeros((0, 300), device=device)  # Default dim
        
        # Get dimension from first available embedding
        dim = None
        for entity in entities:
            emb = self.get_embedding(entity)
            if emb is not None:
                dim = len(emb)
                break
        
        if dim is None:
            dim = 300  # Default
        
        embeddings = []
        for entity in entities:
            emb = self.get_embedding(entity)
            if emb is not None:
                embeddings.append(torch.tensor(emb, dtype=torch.float32))
            else:
                embeddings.append(torch.zeros(dim, dtype=torch.float32))
        
        return torch.stack(embeddings).to(device)


class EntityLinker:
    """Link text entities to knowledge graph entities."""
    
    def __init__(self, kg_loader: KGEmbeddingLoader, entity_mapping: Optional[Dict[str, str]] = None):
        """
        Args:
            kg_loader: KGEmbeddingLoader instance
            entity_mapping: Optional manual mapping from text entities to KG entities
        """
        self.kg_loader = kg_loader
        self.entity_mapping = entity_mapping or {}
        self.cache: Dict[str, str] = {}
    
    def normalize_entity_name(self, entity: str) -> str:
        """Normalize entity name for matching."""
        # Convert to lowercase, remove special chars
        normalized = entity.lower().strip()
        # Remove common prefixes/suffixes
        normalized = normalized.replace("the ", "").replace(" a ", " ").replace(" an ", " ")
        return normalized
    
    def link_entity(self, entity_text: str) -> Optional[str]:
        """
        Link a text entity to a KG entity.
        
        Args:
            entity_text: Text mention of entity
        
        Returns:
            KG entity string or None if not found
        """
        # Check cache
        if entity_text in self.cache:
            return self.cache[entity_text]
        
        # Check manual mapping
        if entity_text in self.entity_mapping:
            kg_entity = self.entity_mapping[entity_text]
            self.cache[entity_text] = kg_entity
            return kg_entity
        
        # Try normalized name
        normalized = self.normalize_entity_name(entity_text)
        if normalized in self.entity_mapping:
            kg_entity = self.entity_mapping[normalized]
            self.cache[entity_text] = kg_entity
            return kg_entity
        
        # Try ConceptNet format
        conceptnet_formats = [
            f"/c/en/{normalized}",
            f"/c/en/{normalized}/n",
            f"/c/en/{normalized.replace(' ', '_')}",
        ]
        
        for fmt in conceptnet_formats:
            if fmt in self.kg_loader.entity_embeddings:
                self.cache[entity_text] = fmt
                return fmt
        
        # Fuzzy matching: find closest entity by name similarity
        best_match = None
        best_score = 0.0
        
        normalized_lower = normalized.lower()
        for kg_entity in self.kg_loader.entity_embeddings.keys():
            if kg_entity.startswith("/c/en/"):
                kg_name = kg_entity.replace("/c/en/", "").replace("/n", "").replace("_", " ")
                if normalized_lower in kg_name or kg_name in normalized_lower:
                    score = len(set(normalized_lower.split()) & set(kg_name.split())) / max(len(normalized_lower.split()), len(kg_name.split()))
                    if score > best_score and score > 0.5:
                        best_score = score
                        best_match = kg_entity
        
        if best_match:
            self.cache[entity_text] = best_match
            return best_match
        
        return None
    
    def link_entities_batch(self, entities: List[str]) -> List[Optional[str]]:
        """Link a batch of entities."""
        return [self.link_entity(e) for e in entities]


class KGGraph:
    """Representation of knowledge graph structure for path finding."""
    
    def __init__(self):
        self.edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # entity -> [(relation, target_entity)]
        self.reverse_edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # For bidirectional search
    
    def add_edge(self, source: str, relation: str, target: str):
        """Add an edge to the graph."""
        self.edges[source].append((relation, target))
        self.reverse_edges[target].append((relation, source))
    
    def load_from_triples(self, triples: List[Tuple[str, str, str]]):
        """
        Load graph from list of (source, relation, target) triples.
        
        Args:
            triples: List of (source_entity, relation, target_entity) tuples
        """
        for source, relation, target in triples:
            self.add_edge(source, relation, target)
    
    def find_paths(self, source: str, target: str, max_length: int = 3, max_paths: int = 10) -> List[List[Tuple[str, str]]]:
        """
        Find paths between two entities.
        
        Args:
            source: Source entity
            target: Target entity
            max_length: Maximum path length
            max_paths: Maximum number of paths to return
        
        Returns:
            List of paths, each path is list of (relation, entity) tuples
        """
        if source == target:
            return [[("self", source)]]
        
        paths = []
        visited = set()
        
        def dfs(current: str, path: List[Tuple[str, str]], length: int):
            if len(paths) >= max_paths:
                return
            
            if current == target:
                paths.append(path[:])
                return
            
            if length >= max_length:
                return
            
            if current in visited:
                return
            
            visited.add(current)
            
            # Explore forward edges
            for relation, next_entity in self.edges.get(current, []):
                path.append((relation, next_entity))
                dfs(next_entity, path, length + 1)
                path.pop()
            
            # Explore reverse edges (bidirectional)
            for relation, next_entity in self.reverse_edges.get(current, []):
                path.append((f"~{relation}", next_entity))  # Mark reverse relations
                dfs(next_entity, path, length + 1)
                path.pop()
            
            visited.remove(current)
        
        dfs(source, [], 0)
        return paths[:max_paths]
    
    def get_neighbors(self, entity: str, max_neighbors: int = 50) -> List[Tuple[str, str]]:
        """Get neighboring entities and relations."""
        neighbors = self.edges.get(entity, [])
        if len(neighbors) > max_neighbors:
            neighbors = neighbors[:max_neighbors]
        return neighbors


def load_conceptnet_triples(triples_path: str, max_triples: Optional[int] = None) -> List[Tuple[str, str, str]]:
    """
    Load ConceptNet triples from a file.
    
    Format: Each line is JSON with "start", "rel", "end" fields
    """
    triples = []
    with open(triples_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if max_triples and idx >= max_triples:
                break
            
            try:
                data = json.loads(line.strip())
                start = data.get("start", {}).get("uri", "")
                rel = data.get("rel", {}).get("uri", "")
                end = data.get("end", {}).get("uri", "")
                
                if start and rel and end:
                    triples.append((start, rel, end))
            except:
                continue
    
    return triples


def create_entity_mapping_from_dataset(dataset_entities: List[str], kg_loader: KGEmbeddingLoader) -> Dict[str, str]:
    """
    Create entity mapping from dataset entities to KG entities.
    
    Args:
        dataset_entities: List of entity strings from dataset
        kg_loader: KGEmbeddingLoader instance
    
    Returns:
        Mapping from dataset entity to KG entity
    """
    linker = EntityLinker(kg_loader)
    mapping = {}
    
    for entity in dataset_entities:
        kg_entity = linker.link_entity(entity)
        if kg_entity:
            mapping[entity] = kg_entity
    
    return mapping
