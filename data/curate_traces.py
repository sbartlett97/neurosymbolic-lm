#!/usr/bin/env python3
"""Curate assistant-trace training data with per-message symbolic annotation.

Pipeline:
1. Stream conversations from a HuggingFace dataset (glaive/ShareGPT/messages)
2. Normalize to the internal messages format (system/user/assistant/tool).
   Tool calls become canonical JSON: {"name": ..., "arguments": {...}}.
   A malformed call cuts the conversation before that turn.
3. Skip duplicate conversations (hash of normalized message text) and pick
   the train/val split per conversation (deterministic hash, --val-ratio)
4. Annotate each encoder-side message once per conversation: GLiNER2 on
   prose, AST parsing on fenced code blocks, regex for digital artifacts
5. Expand the conversation into one sample per assistant turn (at most
   --max-turns-per-conversation). Every sample of a conversation goes to
   the same split, so no conversation prefix leaks into validation.
6. Write JSONL trace samples (messages + message_annotations) to
   NAME.jsonl and NAME_val.jsonl, plus the taxonomy vocabulary file
   NAME_vocab.json (written at start)

Prompts grow with each turn. The script does not cap prompt length: the
causal collator truncates the encoder input from the left, so the end of
the prompt next to the target stays.

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

from data.curate.dedup import is_validation, text_hash
from data.curate.taxonomy import get_default_taxonomy
from data.curate.trace_annotator import (
    TraceAnnotator,
    assistant_turns,
    expand_trace,
    select_target_turns,
)
from data.curate.trace_loader import TraceSourceLoader, is_tool_call


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
        help="Number of training samples (train + val) to produce, counted "
             "after per-turn expansion. The run stops at a conversation "
             "boundary, so it can write up to "
             "--max-turns-per-conversation - 1 extra samples (default: 20000)",
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
        "--overwrite", action="store_true",
        help="Replace existing output files with the same name",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.02,
        help="Fraction of samples written to NAME_val.jsonl (default: 0.02)",
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
        help="Skip conversations with fewer annotated messages. The check is "
             "per conversation, not per expanded sample, so first-turn samples "
             "with few annotations stay (default: 1)",
    )
    parser.add_argument(
        "--max-turns-per-conversation", type=int, default=8,
        help="Cap on samples per conversation. Turns are picked evenly and "
             "always hold the first and last assistant turn; 0 keeps all "
             "(default: 8)",
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


def trace_hash(messages) -> str:
    """Stable hash of a conversation's roles and message text."""
    return text_hash("\n".join(f"{m['role']}: {m['content']}" for m in messages))


def write_vocab(vocab_path: Path, vocab: dict, statistics: dict):
    """Write the taxonomy vocabulary with run statistics."""
    vocab = dict(vocab)
    vocab["statistics"] = {
        "num_concepts": len(vocab["concepts"]),
        "num_relations": len(vocab["relations"]),
        **statistics,
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.output_name}.jsonl"
    val_path = out_dir / f"{args.output_name}_val.jsonl"
    vocab_path = out_dir / f"{args.output_name}_vocab.json"

    existing = [
        p for p in (out_path, val_path, vocab_path)
        if p.exists() and p.stat().st_size > 0
    ]
    if existing and not args.overwrite:
        names = ", ".join(p.name for p in existing)
        print(
            f"Error: output files already exist in {out_dir}: {names}. "
            f"Use --overwrite to replace them."
        )
        sys.exit(1)

    print("=" * 60)
    print("Trace Curation Pipeline")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Target samples: {args.target_samples}")
    print(f"GLiNER2 prose extraction: {'off' if args.no_gliner else args.gliner_model}")
    print(f"Output: {out_path} (val: {val_path}, ratio {args.val_ratio})")
    print("=" * 60)

    # The vocabulary comes from the taxonomy. Write it now so a crash
    # still leaves a usable vocab file next to the partial output.
    taxonomy = get_default_taxonomy()
    vocab = taxonomy.vocab()
    write_vocab(vocab_path, vocab, {"num_samples": 0})

    annotator = build_annotator(args)
    loader = TraceSourceLoader(
        source=args.source, fmt=args.format, split=args.split, seed=args.seed
    )

    written = 0
    val_written = 0
    tool_call_targets = 0
    conversations_read = 0
    conversations_used = 0
    skipped = 0
    duplicates = 0
    seen_hashes = set()
    next_checkpoint = args.checkpoint_every
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f, \
            open(val_path, "w", encoding="utf-8") as f_val:
        for doc in loader:
            # Stop only between conversations
            if written >= args.target_samples:
                break
            conversations_read += 1

            # No sample reads past the last assistant turn, so drop the tail
            messages = doc.messages[: args.max_messages]
            turns = assistant_turns(messages)
            if turns:
                messages = messages[: turns[-1] + 1]
            if not turns or len(messages) < args.min_messages:
                skipped += 1
                continue

            # Skip exact duplicates before the annotation cost. The hash
            # also picks the split, so a whole conversation goes to one side.
            hash_hex = trace_hash(messages)
            if hash_hex in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(hash_hex)
            is_val = is_validation(hash_hex, args.val_ratio)

            # Annotate once; every expanded sample re-uses the annotations
            sample = annotator.annotate_trace(messages)
            if len(sample["message_annotations"]) < args.min_annotated_messages:
                skipped += 1
                continue

            targets = select_target_turns(messages, args.max_turns_per_conversation)
            conversations_used += 1
            out_f = f_val if is_val else f
            for turn_sample in expand_trace(sample, targets):
                turn_sample["source"] = doc.source
                turn_sample["conversation_id"] = hash_hex
                out_f.write(json.dumps(turn_sample, ensure_ascii=False) + "\n")
                written += 1
                val_written += int(is_val)
                tool_call_targets += int(is_tool_call(turn_sample["response"]))

            if written >= next_checkpoint:
                next_checkpoint += args.checkpoint_every
                f.flush()
                f_val.flush()
                elapsed = time.time() - start_time
                rate = written / elapsed if elapsed > 0 else 0
                eta = max(args.target_samples - written, 0) / rate if rate > 0 else 0
                print(
                    f"Progress: {written}/{args.target_samples} | "
                    f"Rate: {rate:.1f} samples/sec | ETA: {eta / 60:.1f} min"
                )

    write_vocab(
        vocab_path,
        vocab,
        {
            "num_samples": written,
            "num_val_samples": val_written,
            "num_conversations": conversations_used,
            "num_tool_call_targets": tool_call_targets,
            **annotator.get_statistics(),
        },
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Trace Curation Complete")
    print("=" * 60)
    print(f"Time elapsed: {elapsed / 60:.1f} minutes")
    loader_stats = loader.get_statistics()
    share = tool_call_targets / written if written else 0.0
    print(
        f"Conversations read: {conversations_read} (used {conversations_used}, "
        f"skipped {skipped}, duplicates {duplicates})"
    )
    print(
        f"Samples written: {written} (train {written - val_written}, "
        f"val {val_written})"
    )
    print(f"Tool-call targets written: {tool_call_targets} ({share:.1%})")
    print(
        f"Malformed tool calls dropped: {loader_stats['malformed_tool_calls']} "
        f"(conversations cut before that turn; the loader skipped "
        f"{loader_stats['skipped']} rows with no assistant turn left)"
    )
    print(f"Annotator stats: {annotator.get_statistics()}")
    print(f"Loader stats: {loader_stats}")
    print(f"Output: {out_path}")
    print(f"Validation: {val_path}")
    print(f"Vocabulary: {vocab_path}")


if __name__ == "__main__":
    main()
