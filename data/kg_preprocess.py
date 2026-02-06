"""Offline KG entity linking and path extraction for ConceptNet.

Reads training JSONL, enriches each sample with:
  - kg_entity_uris: ConceptNet URIs for each entity
  - kg_relations:   (head_idx, tail_idx, relation_uri) triplets from ConceptNet
  - kg_paths:       Multi-hop paths between entity pairs

Usage:
    python data/kg_preprocess.py \
        --input data.jsonl \
        --output data_kg.jsonl \
        --kg-embeddings numberbatch-en-19.08.txt \
        --kg-triples triples.jsonl
"""

import argparse
import gzip
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kg_utils import KGEmbeddingLoader, EntityLinker, KGGraph, load_conceptnet_triples

# ConceptNet download URLs
NUMBERBATCH_URL = (
    "https://conceptnet.s3.amazonaws.com/downloads/2019/"
    "numberbatch/numberbatch-en-19.08.txt.gz"
)


def download_file(url: str, dest: Path, description: str = "") -> Path:
    """Download a file if it doesn't already exist."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {description or url} ...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  Saved to {dest}")
    return dest


def download_numberbatch(cache_dir: Path) -> Path:
    """Download and decompress ConceptNet Numberbatch embeddings."""
    gz_path = cache_dir / "numberbatch-en-19.08.txt.gz"
    txt_path = cache_dir / "numberbatch-en-19.08.txt"

    if txt_path.exists():
        print(f"  Numberbatch already decompressed: {txt_path}")
        return txt_path

    download_file(NUMBERBATCH_URL, gz_path, "Numberbatch embeddings")

    print("  Decompressing...")
    with gzip.open(str(gz_path), "rb") as f_in:
        with open(str(txt_path), "wb") as f_out:
            while True:
                chunk = f_in.read(1024 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)
    print(f"  Decompressed to {txt_path}")
    return txt_path


def enrich_sample(
    sample: dict,
    linker: EntityLinker,
    graph: Optional[KGGraph],
    max_path_length: int = 3,
    max_paths_per_pair: int = 5,
) -> dict:
    """Enrich a single sample with KG entity URIs, relations, and paths."""
    entities = sample.get("entities", [])
    if not entities:
        return sample

    # Link entities to ConceptNet URIs
    uris = linker.link_entities_batch(entities)
    sample["kg_entity_uris"] = [u if u else "" for u in uris]

    # Find direct relations between entity pairs from the graph
    kg_relations = []
    kg_paths = []

    if graph is not None:
        valid_uris = [(i, u) for i, u in enumerate(uris) if u]
        for idx_a in range(len(valid_uris)):
            for idx_b in range(idx_a + 1, len(valid_uris)):
                i, uri_a = valid_uris[idx_a]
                j, uri_b = valid_uris[idx_b]

                # Direct neighbors
                for rel, target in graph.edges.get(uri_a, []):
                    if target == uri_b:
                        kg_relations.append([i, j, rel])
                for rel, target in graph.edges.get(uri_b, []):
                    if target == uri_a:
                        kg_relations.append([j, i, rel])

                # Multi-hop paths
                paths = graph.find_paths(
                    uri_a, uri_b,
                    max_length=max_path_length,
                    max_paths=max_paths_per_pair,
                )
                if paths:
                    # Convert tuples to lists for JSON
                    kg_paths.append(
                        [[list(step) for step in path] for path in paths]
                    )

    sample["kg_relations"] = kg_relations
    sample["kg_paths"] = kg_paths
    return sample


def main():
    parser = argparse.ArgumentParser(
        description="Enrich training JSONL with ConceptNet KG data"
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL")
    parser.add_argument(
        "--kg-embeddings",
        type=str,
        default=None,
        help="Path to Numberbatch embeddings (auto-downloads if missing)",
    )
    parser.add_argument(
        "--kg-triples",
        type=str,
        default=None,
        help="Path to ConceptNet triples JSONL",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache/kg",
        help="Cache directory for downloads",
    )
    parser.add_argument(
        "--max-triples",
        type=int,
        default=100000,
        help="Max number of KG triples to load",
    )
    parser.add_argument(
        "--max-path-length",
        type=int,
        default=3,
        help="Max multi-hop path length",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load Numberbatch embeddings
    emb_path = args.kg_embeddings
    if emb_path is None:
        emb_path = str(download_numberbatch(cache_dir))
    elif not Path(emb_path).exists():
        print(f"Embeddings not found at {emb_path}, downloading...")
        emb_path = str(download_numberbatch(cache_dir))

    print("Loading ConceptNet Numberbatch embeddings...")
    loader = KGEmbeddingLoader(kg_type="conceptnet")
    loader.load_conceptnet_embeddings(emb_path)
    linker = EntityLinker(loader)

    # Optionally load triples for path finding
    graph = None
    if args.kg_triples and Path(args.kg_triples).exists():
        print(f"Loading KG triples from {args.kg_triples}...")
        triples = load_conceptnet_triples(
            args.kg_triples, max_triples=args.max_triples
        )
        graph = KGGraph()
        graph.load_from_triples(triples)
        print(f"  Loaded {len(triples)} triples, {len(graph.edges)} source nodes")

    # Process samples
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Enriching {input_path} -> {output_path}")
    count = 0
    linked = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample = enrich_sample(
                sample, linker, graph, max_path_length=args.max_path_length
            )
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
            if any(u for u in sample.get("kg_entity_uris", [])):
                linked += 1
            if count % 1000 == 0:
                print(f"  Processed {count} samples ({linked} with KG links)...")

    print(f"Done. {count} samples processed, {linked} with KG entity links.")


if __name__ == "__main__":
    main()
