#!/usr/bin/env python3
"""Dataset curation CLI for generating neurosymbolic training data.

This script creates training data by:
1. Loading documents from HuggingFace datasets (Wikipedia, C4, FineWeb)
2. Filtering for quality (length, language, content)
3. Annotating with GLiNER2 (entities, spans, concepts, entity types,
   relations) and/or an LLM (QA pairs)
4. Quality checking and fixing annotations
5. Deduplicating by normalized-text hash and assigning a deterministic
   train/validation split
6. Writing to JSONL format compatible with CognitiveCollator

Outputs (in --output-dir, with --output-name NAME):
    NAME.jsonl             train samples
    NAME_val.jsonl         validation samples (--val-ratio, hash-based)
    NAME_vocab.json        taxonomy vocabulary (written at start)
    NAME_checkpoint.json   resume state

Usage:
    # Stage-1 symbolic data with GLiNER2 (default annotator)
    python data/curate_dataset.py \
        --target-samples 50000 \
        --annotator gliner \
        --sources allenai/c4 wikipedia \
        --source-ratios 0.6 0.4 \
        --output-dir data/curated

    # GLiNER2 extraction + LLM QA generation (stage-2/3 data)
    python data/curate_dataset.py \
        --annotator hybrid \
        --backend vllm \
        --llm meta-llama/Llama-3.1-8B-Instruct \
        --should-respond-ratio 0.7

    # Hybrid QA with a hosted API (key from ANTHROPIC_API_KEY / OPENAI_API_KEY)
    python data/curate_dataset.py --annotator hybrid --backend anthropic
    python data/curate_dataset.py --annotator hybrid --backend openai \
        --api-concurrency 16

    # Test run without any models
    python data/curate_dataset.py --target-samples 100 --annotator llm --backend mock

    # Resume from checkpoint (skips source rows already consumed)
    python data/curate_dataset.py --resume-from data/curated/curated_checkpoint.json

    # Replace an earlier run with the same output name
    python data/curate_dataset.py --target-samples 100 --overwrite
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.curate import (
    CurationConfig,
    SourceLoader,
    DocumentFilter,
    LLMAnnotator,
    QualityControl,
    OutputWriter,
)
from data.curate.dedup import text_hash
from data.curate.document_filter import ContentCleaner, truncate_text
from data.curate.source_loader import parse_text_field_args


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Curate dataset with LLM annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Target and sources
    parser.add_argument(
        "--target-samples",
        type=int,
        default=50000,
        help="Target number of curated samples (default: 50000)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["allenai/c4", "wikipedia"],
        help="Dataset sources (default: allenai/c4 wikipedia)",
    )
    parser.add_argument(
        "--source-ratios",
        nargs="+",
        type=float,
        default=None,
        help="Sampling ratios for sources (default: equal)",
    )
    parser.add_argument(
        "--text-field",
        nargs="+",
        default=None,
        metavar="[SOURCE=]FIELD",
        help=(
            "Text column name. A bare FIELD applies to all sources; "
            "SOURCE=FIELD applies to one source (default: 'text')"
        ),
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/curated",
        help="Output directory (default: data/curated)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="curated",
        help="Output file base name (default: curated)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files with the same name",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.02,
        help="Fraction of samples written to NAME_val.jsonl (default: 0.02)",
    )

    # Annotator selection
    parser.add_argument(
        "--annotator",
        type=str,
        choices=["gliner", "llm", "hybrid"],
        default="gliner",
        help=(
            "Annotation strategy: 'gliner' (GLiNER2 schema-driven extraction), "
            "'llm' (legacy free-form LLM JSON), 'hybrid' (GLiNER2 extraction "
            "+ LLM QA generation) (default: gliner)"
        ),
    )

    # GLiNER2 settings
    parser.add_argument(
        "--gliner-model",
        type=str,
        default="fastino/gliner2-base-v1",
        help="GLiNER2 model (default: fastino/gliner2-base-v1)",
    )
    parser.add_argument(
        "--entity-threshold",
        type=float,
        default=0.5,
        help="GLiNER2 entity confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--relation-threshold",
        type=float,
        default=0.4,
        help="GLiNER2 relation confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--gliner-device",
        type=str,
        default=None,
        help="Device for GLiNER2 (e.g. cuda, mps, cpu; default: cuda > mps > cpu)",
    )
    parser.add_argument(
        "--gliner-max-len",
        type=int,
        default=0,
        help="Max word tokens GLiNER2 encodes per text; 0 = no limit (default). "
             "A cap below the document length leaves late entities unlabeled",
    )
    parser.add_argument(
        "--gliner-batch-size",
        type=int,
        default=None,
        help="Texts per GLiNER2 forward pass (default 1; output is identical, measure docs/s before raising it)",
    )

    # LLM settings
    parser.add_argument(
        "--llm",
        type=str,
        default=None,
        help=(
            "LLM model name (default per backend: meta-llama/Llama-3.1-8B-Instruct "
            "for vllm/transformers, gpt-6-luna for openai, claude-haiku-4-5 "
            "for anthropic)"
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "transformers", "openai", "anthropic", "mock"],
        default="mock",
        help=(
            "LLM backend for 'llm'/'hybrid' annotators. openai/anthropic read "
            "the key from OPENAI_API_KEY/ANTHROPIC_API_KEY (default: mock for testing)"
        ),
    )
    parser.add_argument(
        "--api-concurrency",
        type=int,
        default=8,
        help="Parallel requests per batch for openai/anthropic (default: 8)",
    )
    parser.add_argument(
        "--api-effort",
        type=str,
        default=None,
        help=(
            "Effort level for openai (reasoning_effort, default: low) or "
            "anthropic (output_config.effort, default: not sent; claude-haiku-4-5 "
            "does not accept it)"
        ),
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["awq", "gptq", "none"],
        default="none",
        help="Quantization method; awq/gptq need a pre-quantized checkpoint (default: none)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help=(
            "vLLM GPU memory fraction (default: 0.6 for hybrid, where GLiNER2 "
            "shares the GPU; 0.9 otherwise)"
        ),
    )

    # Batch settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for LLM inference (default: 16)",
    )

    # Content settings
    parser.add_argument(
        "--should-respond-ratio",
        type=float,
        default=0.7,
        help="Ratio of samples with QA pairs (default: 0.7)",
    )
    parser.add_argument(
        "--min-doc-length",
        type=int,
        default=100,
        help="Minimum document length (default: 100)",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=128,
        help="Most confident entities kept per document (default: 128). "
             "Entities past the cap are unlabeled and train as 'none'",
    )
    parser.add_argument(
        "--max-doc-length",
        type=int,
        default=16000,
        help="Maximum document length in characters (default: 16000, about "
             "3.7k tokens; keep it within the training --max-input-length)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="Save checkpoint every N samples (default: 1000)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Safety
    parser.add_argument(
        "--enable-safety-filter",
        action="store_true",
        help="Enable safety content filtering",
    )
    parser.add_argument(
        "--safety-strictness",
        type=str,
        choices=["low", "medium", "high", "maximum"],
        default="medium",
        help="Safety filter strictness (default: medium)",
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def create_config(args: argparse.Namespace) -> CurationConfig:
    """Create CurationConfig from command line arguments."""
    # Normalize source ratios
    if args.source_ratios is None:
        source_ratios = [1.0 / len(args.sources)] * len(args.sources)
    else:
        source_ratios = args.source_ratios

    return CurationConfig(
        target_samples=args.target_samples,
        sources=args.sources,
        source_ratios=source_ratios,
        output_dir=args.output_dir,
        output_name=args.output_name,
        overwrite=args.overwrite,
        val_ratio=args.val_ratio,
        text_field=parse_text_field_args(args.text_field),
        annotator=args.annotator,
        gliner_model=args.gliner_model,
        gliner_entity_threshold=args.entity_threshold,
        gliner_relation_threshold=args.relation_threshold,
        gliner_device=args.gliner_device,
        gliner_max_len=args.gliner_max_len or None,
        gliner_batch_size=args.gliner_batch_size,
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        llm_model=args.llm,
        llm_backend=args.backend,
        api_concurrency=args.api_concurrency,
        api_effort=args.api_effort,
        llm_quantization=None if args.quantization == "none" else args.quantization,
        batch_size=args.batch_size,
        should_respond_ratio=args.should_respond_ratio,
        min_doc_length=args.min_doc_length,
        max_doc_length=args.max_doc_length,
        max_entities=args.max_entities,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
        enable_safety_filter=args.enable_safety_filter,
        safety_strictness=args.safety_strictness,
        seed=args.seed,
    )


def run_curation(config: CurationConfig, verbose: bool = False):
    """Run the curation pipeline.

    Args:
        config: CurationConfig with pipeline settings
        verbose: Whether to print verbose output
    """
    print("=" * 60)
    print("Dataset Curation Pipeline")
    print("=" * 60)
    print(f"Target samples: {config.target_samples}")
    print(f"Sources: {config.sources}")
    print(f"Source ratios: {config.source_ratios}")
    print(f"Annotator: {config.annotator}")
    if config.annotator in ("gliner", "hybrid"):
        print(f"GLiNER2: {config.gliner_model}")
    if config.annotator in ("llm", "hybrid"):
        print(f"LLM: {config.llm_model} ({config.llm_backend})")
    print(f"Output: {config.output_dir}/{config.output_name}.jsonl")
    print(f"Validation ratio: {config.val_ratio}")
    print("=" * 60)

    taxonomy = None
    if config.annotator != "llm":
        from data.curate import get_default_taxonomy

        taxonomy = get_default_taxonomy()

    # Output writer. Check the output files before any model loads.
    writer = OutputWriter(
        output_dir=config.output_dir,
        output_name=config.output_name,
        checkpoint_every=config.checkpoint_every,
        taxonomy=taxonomy,
        val_ratio=config.val_ratio,
    )
    start_offsets = None
    if config.resume_from:
        checkpoint_path = Path(config.resume_from)
        if checkpoint_path.resolve() != writer.checkpoint_path.resolve():
            # The checkpoint line counts only describe its own output files
            print(
                f"Error: checkpoint {checkpoint_path} does not belong to output "
                f"{writer.output_path}. Pass the matching --output-dir and --output-name."
            )
            sys.exit(1)
        if not checkpoint_path.exists() or not writer.resume(str(checkpoint_path)):
            print(f"Error: cannot load checkpoint {checkpoint_path}")
            sys.exit(1)
        start_offsets = writer.state.source_offsets
        print(f"Resuming from {writer.state.samples_written} samples")
    else:
        try:
            writer.start(overwrite=config.overwrite)
        except FileExistsError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Initialize components
    print("\nInitializing components...")

    # Source loader
    loader = SourceLoader(
        sources=config.sources,
        ratios=config.source_ratios,
        seed=config.seed,
        text_field=config.text_field,
        start_offsets=start_offsets,
    )

    # Document filter
    doc_filter = DocumentFilter(
        min_length=config.min_doc_length,
        max_length=config.max_doc_length,
    )

    # Content cleaner
    cleaner = ContentCleaner()

    # vLLM shares the GPU with GLiNER2 in hybrid mode
    gpu_memory_utilization = config.vllm_gpu_memory_utilization
    if gpu_memory_utilization is None:
        gpu_memory_utilization = 0.6 if config.annotator == "hybrid" else 0.9

    # Annotator
    if config.annotator == "llm":
        print(f"Loading LLM backend: {config.llm_backend}...")
        annotator = LLMAnnotator(
            model_name=config.llm_model,
            backend=config.llm_backend,
            quantization=config.llm_quantization,
            should_respond_ratio=config.should_respond_ratio,
            gpu_memory_utilization=gpu_memory_utilization,
            api_concurrency=config.api_concurrency,
            api_effort=config.api_effort,
        )
    else:
        from data.curate import GLiNER2Annotator

        # Load GLiNER2 first. vLLM then sizes its cache from the GPU
        # memory that is left.
        print(f"Loading GLiNER2: {config.gliner_model}...")
        annotator = GLiNER2Annotator(
            model_name=config.gliner_model,
            taxonomy=taxonomy,
            entity_threshold=config.gliner_entity_threshold,
            relation_threshold=config.gliner_relation_threshold,
            device=config.gliner_device,
            should_respond_ratio=config.should_respond_ratio,
            max_len=config.gliner_max_len,
            batch_size=config.gliner_batch_size,
            max_entities=config.max_entities,
            # Corpus prose: code and tool-use labels only add wrong types
            include_code_labels=False,
        )

        if config.annotator == "hybrid":
            print(f"Loading LLM backend for QA generation: {config.llm_backend}...")
            llm = LLMAnnotator(
                model_name=config.llm_model,
                backend=config.llm_backend,
                quantization=config.llm_quantization,
                gpu_memory_utilization=gpu_memory_utilization,
                api_concurrency=config.api_concurrency,
                api_effort=config.api_effort,
            )
            annotator.response_backend = llm.backend
            print(f"QA model: {llm.model_name}")

    # Safety regulator (optional)
    safety_regulator = None
    if config.enable_safety_filter:
        try:
            from continual_learning.safety import SafetyRegulator

            safety_regulator = SafetyRegulator(
                strictness=config.safety_strictness,
                log_path=config.safety_log_path,
            )
            print(f"Safety filter enabled: {config.safety_strictness}")
        except ImportError:
            print("Warning: SafetyRegulator not available, skipping safety filtering")

    # Quality control
    qc = QualityControl(safety_regulator=safety_regulator, taxonomy=taxonomy)

    # Main curation loop
    print("\nStarting curation...")
    start_time = time.time()

    batch_count = 0
    docs_processed = 0
    duplicates = 0
    load_error = None

    doc_iterator = iter(loader)

    while writer.state.samples_written < config.target_samples:
        # Get batch of documents. Do not fetch more than the target needs.
        batch_docs = []
        batch_hashes = []
        batch_limit = min(
            config.batch_size, config.target_samples - writer.state.samples_written
        )
        while len(batch_docs) < batch_limit:
            try:
                doc = next(doc_iterator)
            except StopIteration:
                break
            except (ValueError, RuntimeError) as e:
                # Bad source config (missing text field, empty source)
                load_error = e
                break
            docs_processed += 1

            # Filter document (cuts doc.text to max_doc_length)
            filter_result = doc_filter.filter(doc)
            if not filter_result.passed:
                continue

            # Clean content. Placeholders such as [EMAIL] can make the
            # text longer, so cut it again.
            doc.text = truncate_text(cleaner.clean(doc.text), config.max_doc_length)

            # Skip exact duplicates of normalized text
            hash_hex = text_hash(doc.text)
            if writer.is_duplicate(hash_hex):
                duplicates += 1
                continue
            writer.mark_seen(hash_hex)

            batch_docs.append(doc)
            batch_hashes.append(hash_hex)

        if load_error is not None:
            print(f"Error: {load_error}")
            break
        if not batch_docs:
            print("No more documents available")
            break

        # Annotate batch
        results = annotator.annotate_batch(batch_docs)

        # Quality control. Keep each result with its own document.
        passed_results, sources, hashes = [], [], []
        for doc, hash_hex, result in zip(batch_docs, batch_hashes, results):
            qc_result = qc.validate(result)
            if qc_result.passed:
                passed_results.append(qc_result.fixed_result or result)
                sources.append(doc.source)
                hashes.append(hash_hex)

        # Write results, but do not go past the target
        remaining = config.target_samples - writer.state.samples_written
        writer.write_batch(passed_results[:remaining], sources[:remaining], hashes[:remaining])

        # Checkpoint only between batches, so the source offsets match
        # the rows on disk
        writer.maybe_checkpoint(loader.get_offsets())

        batch_count += 1

        # Progress update
        if batch_count % 10 == 0 or verbose:
            elapsed = time.time() - start_time
            rate = writer.state.samples_written / elapsed if elapsed > 0 else 0
            eta = (config.target_samples - writer.state.samples_written) / rate if rate > 0 else 0

            print(
                f"Progress: {writer.state.samples_written}/{config.target_samples} "
                f"({100 * writer.state.samples_written / config.target_samples:.1f}%) | "
                f"Rate: {rate:.1f} samples/sec | "
                f"ETA: {eta / 60:.1f} min"
            )

    # Finalize
    writer.state.source_offsets = loader.get_offsets()
    writer.finalize()

    # Print final statistics
    print("\n" + "=" * 60)
    print("Curation Complete" if load_error is None else "Curation Stopped")
    print("=" * 60)

    elapsed = time.time() - start_time
    annotator_stats = annotator.get_statistics()
    print(f"\nTime elapsed: {elapsed / 60:.1f} minutes")
    print(f"Documents processed: {docs_processed}")
    print(f"Duplicates skipped: {duplicates}")
    print(f"Filter pass rate: {doc_filter.get_statistics()['pass_rate']:.2%}")
    print(f"Annotation success rate: {annotator_stats['success_rate']:.2%}")
    if annotator_stats.get("qa_generated", 0) or annotator_stats.get("qa_failed", 0):
        print(
            f"QA generated: {annotator_stats['qa_generated']} | "
            f"QA failed: {annotator_stats['qa_failed']} "
            f"({annotator_stats['qa_failed_rate']:.2%})"
        )
    print(f"QC pass rate: {qc.get_statistics()['pass_rate']:.2%}")

    writer_stats = writer.get_statistics()
    print(f"\nFinal output:")
    print(f"  Samples: {writer_stats['samples_written']}")
    print(f"  Train: {writer_stats['train_written']} | Val: {writer_stats['val_written']}")
    print(f"  Concepts: {writer_stats['num_concepts']}")
    print(f"  Relations: {writer_stats['num_relations']}")
    print(f"  Source distribution: {writer_stats['source_distribution']}")

    if load_error is not None:
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()
    config = create_config(args)
    run_curation(config, verbose=args.verbose)


if __name__ == "__main__":
    main()
