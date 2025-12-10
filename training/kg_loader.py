"""Knowledge graph loading utilities."""

from typing import Optional, Tuple, Dict


def load_kg_data_for_training(
    kg_embedding_path: Optional[str] = None,
    kg_triples_path: Optional[str] = None,
    entity_mapping: Optional[Dict[str, str]] = None,
    kg_type: str = "conceptnet"
) -> Tuple:
    """
    Load knowledge graph data for training.
    
    Args:
        kg_embedding_path: Path to KG embeddings file
        kg_triples_path: Path to KG triples file (for path reasoning)
        entity_mapping: Optional manual mapping from text entities to KG entities
        kg_type: Type of knowledge graph ("conceptnet", "wikidata", "wordnet", "generic")
    
    Returns:
        Tuple of (kg_loader, entity_linker, kg_graph) or (None, None, None) if unavailable
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
        kg_loader = KGEmbeddingLoader(kg_type=kg_type)
        
        # Load embeddings
        if kg_type.lower() == "conceptnet":
            kg_embed_dim = kg_loader.load_conceptnet_embeddings(kg_embedding_path)
        elif kg_type.lower() == "wikidata":
            kg_embed_dim = kg_loader.load_wikidata_embeddings(kg_embedding_path)
        else:
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
                    print(f"Warning: Generic triple loading not implemented for {kg_type}")
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
