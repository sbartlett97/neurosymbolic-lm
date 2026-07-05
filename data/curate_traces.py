#!/usr/bin/env python3
"""Curate assistant-trace training data with per-message symbolic annotation.

Pipeline:
1. Stream conversations from a HuggingFace dataset (glaive/ShareGPT/messages)
2. Normalize to the internal messages format (system/user/assistant/tool)
3. Annotate each encoder-side message: GLiNER2 on prose, AST parsing on
   fenced code blocks, regex extraction for digital artifacts
4. Write JSONL trace samples (messages + message_annotations) plus the
   taxonomy vocabulary file

Usage:
    # Deterministic-only annotation (no GLiNER2 model needed)
    python data/curate_traces.py \
        --source glaiveai/glaive-function-calling-v2 \
        --target-samples 20000 --no-gliner

    # Full annotation with GLiNER2 prose extraction
    python data/curate_traces.py \
        --source glaiveai/glaive-function-calling-v2 \
        --target-samples 20000 --gliner-model fastino/gliner2-base-v1
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.curate.taxonomy import get_default_taxonomy
from data.curate.trace_annotator import TraceAnnotator
from data.curate.trace_loader import TraceSourceLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curate assistant-trace data with symbolic annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="glaiveai/glaive-function-calling-v2",
        help="HuggingFace dataset path (default: glaiveai/glaive-function-calling-v2)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["glaive", "sharegpt", "messages"],
        default=None,
        help="Source format (default: auto for known sources, else 'messages')",
    )
    parser.add_argument("--split", type=str, default=None, help="Dataset split")
    parser.add_argument(
        "--target-samples", type=int, default=20000,
        help="Number of trace samples to produce (default: 20000)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/curated_traces",
        help="Output directory (default: data/curated_traces)",
    )
    parser.add_argument(
        "--output-name", type=str, default="traces",
        help="Output file base name (default: traces)",
    )
    parser.add_argument(
        "--no-gliner", action="store_true",
        help="Skip GLiNER2 prose extraction (deterministic annotators only)",
    )
    parser.add_argument(
        "--gliner-model", type=str, default="fastino/gliner2-base-v1",
        help="GLiNER2 model (default: fastino/gliner2-base-v1)",
    )
    parser.add_argument("--entity-threshold", type=float, default=0.5)
    parser.add_argument("--relation-threshold", type=float, default=0.4)
    parser.add_argument("--gliner-device", type=str, default=None)
    parser.add_argument(
        "--max-entities-per-message", type=int, default=16,
        help="Entity cap per message (default: 16)",
    )
    parser.add_argument(
        "--min-messages", type=int, default=2,
        help="Skip conversations shorter than this (default: 2)",
    )
    parser.add_argument(
        "--max-messages", type=int, default=24,
        help="Truncate conversations longer than this (default: 24)",
    )
    parser.add_argument(
        "--min-annotated-messages", type=int, default=1,
        help="Skip samples with fewer annotated messages (default: 1)",
    )
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_annotator(args) -> TraceAnnotator:
    taxonomy = get_default_taxonomy()
    gliner = None
    if not args.no_gliner:
        from data.curate.gliner_annotator import GLiNER2Annotator

        print(f"Loading GLiNER2: {args.gliner_model}...")
        gliner = GLiNER2Annotator(
            model_name=args.gliner_model,
            taxonomy=taxonomy,
            entity_threshold=args.entity_threshold,
            relation_threshold=args.relation_threshold,
            device=args.gliner_device,
            max_entities=args.max_entities_per_message,
        )
    return TraceAnnotator(
        gliner_annotator=gliner,
        taxonomy=taxonomy,
        max_entities_per_message=args.max_entities_per_message,
    )


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.output_name}.jsonl"
    vocab_path = out_dir / f"{args.output_name}_vocab.json"

    print("=" * 60)
    print("Trace Curation Pipeline")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Target samples: {args.target_samples}")
    print(f"GLiNER2 prose extraction: {'off' if args.no_gliner else args.gliner_model}")
    print(f"Output: {out_path}")
    print("=" * 60)

    annotator = build_annotator(args)
    loader = TraceSourceLoader(
        source=args.source, fmt=args.format, split=args.split, seed=args.seed
    )

    taxonomy = get_default_taxonomy()
    vocab = taxonomy.vocab()

    written = 0
    skipped = 0
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
        for doc in loader:
            if written >= args.target_samples:
                break

            messages = doc.messages[: args.max_messages]
            if len(messages) < args.min_messages:
                skipped += 1
                continue

            sample = annotator.annotate_trace(messages)
            if len(sample["message_annotations"]) < args.min_annotated_messages:
                skipped += 1
                continue
            if not sample["should_respond"]:
                skipped += 1
                continue

            sample["source"] = doc.source
            f.write(json.dumps(sample) + "\n")
            written += 1

            if written % args.checkpoint_every == 0:
                f.flush()
                elapsed = time.time() - start_time
                rate = written / elapsed if elapsed > 0 else 0
                eta = (args.target_samples - written) / rate if rate > 0 else 0
                print(
                    f"Progress: {written}/{args.target_samples} | "
                    f"Rate: {rate:.1f} samples/sec | ETA: {eta / 60:.1f} min"
                )

    vocab["statistics"] = {
        "num_concepts": len(vocab["concepts"]),
        "num_relations": len(vocab["relations"]),
        "num_samples": written,
        **annotator.get_statistics(),
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Trace Curation Complete")
    print("=" * 60)
    print(f"Time elapsed: {elapsed / 60:.1f} minutes")
    print(f"Samples written: {written} (skipped {skipped})")
    print(f"Annotator stats: {annotator.get_statistics()}")
    print(f"Loader stats: {loader.get_statistics()}")
    print(f"Output: {out_path}")
    print(f"Vocabulary: {vocab_path}")


if __name__ == "__main__":
    main()
