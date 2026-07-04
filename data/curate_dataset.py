#!/usr/bin/env python3
"""Dataset curation CLI for generating neurosymbolic training data.

This script creates training data by:
1. Loading documents from HuggingFace datasets (Wikipedia, C4, RedPajama)
2. Filtering for quality (length, language, content)
3. Annotating with GLiNER2 (entities, spans, concepts, entity types,
   relations) and/or an LLM (QA pairs)
4. Quality checking and fixing annotations
5. Writing to JSONL format compatible with CognitiveCollator

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

    # Test run without any models
    python data/curate_dataset.py --target-samples 100 --annotator llm --backend mock

    # Resume from checkpoint
    python data/curate_dataset.py --resume-from data/curated/curated_checkpoint.json
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

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
from data.curate.document_filter import ContentCleaner


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
        help="Device for GLiNER2 (e.g. cuda, cpu; default: auto)",
    )

    # LLM settings
    parser.add_argument(
        "--llm",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="LLM model name (default: meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "transformers", "mock"],
        default="mock",
        help="LLM backend for 'llm'/'hybrid' annotators (default: mock for testing)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["awq", "gptq", "none"],
        default="awq",
        help="Quantization method (default: awq)",
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
        "--max-doc-length",
        type=int,
        default=4000,
        help="Maximum document length (default: 4000)",
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
        annotator=args.annotator,
        gliner_model=args.gliner_model,
        gliner_entity_threshold=args.entity_threshold,
        gliner_relation_threshold=args.relation_threshold,
        gliner_device=args.gliner_device,
        llm_model=args.llm,
        llm_backend=args.backend,
        llm_quantization=None if args.quantization == "none" else args.quantization,
        batch_size=args.batch_size,
        should_respond_ratio=args.should_respond_ratio,
        min_doc_length=args.min_doc_length,
        max_doc_length=args.max_doc_length,
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
    print(f"Output: {config.output_dir}")
    print("=" * 60)

    # Initialize components
    print("\nInitializing components...")

    # Source loader
    loader = SourceLoader(
        sources=config.sources,
        ratios=config.source_ratios,
        seed=config.seed,
    )

    # Document filter
    doc_filter = DocumentFilter(
        min_length=config.min_doc_length,
        max_length=config.max_doc_length,
    )

    # Content cleaner
    cleaner = ContentCleaner()

    # Annotator
    taxonomy = None
    if config.annotator == "llm":
        print(f"Loading LLM backend: {config.llm_backend}...")
        annotator = LLMAnnotator(
            model_name=config.llm_model,
            backend=config.llm_backend,
            quantization=config.llm_quantization,
            should_respond_ratio=config.should_respond_ratio,
        )
    else:
        from data.curate import GLiNER2Annotator, get_default_taxonomy

        taxonomy = get_default_taxonomy()

        response_backend = None
        if config.annotator == "hybrid":
            print(f"Loading LLM backend for QA generation: {config.llm_backend}...")
            llm = LLMAnnotator(
                model_name=config.llm_model,
                backend=config.llm_backend,
                quantization=config.llm_quantization,
            )
            response_backend = llm.backend

        print(f"Loading GLiNER2: {config.gliner_model}...")
        annotator = GLiNER2Annotator(
            model_name=config.gliner_model,
            taxonomy=taxonomy,
            entity_threshold=config.gliner_entity_threshold,
            relation_threshold=config.gliner_relation_threshold,
            device=config.gliner_device,
            response_backend=response_backend,
            should_respond_ratio=config.should_respond_ratio,
        )

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

    # Output writer
    writer = OutputWriter(
        output_dir=config.output_dir,
        output_name="curated",
        checkpoint_every=config.checkpoint_every,
        taxonomy=taxonomy,
    )

    # Resume from checkpoint if specified
    start_count = 0
    if config.resume_from:
        checkpoint_path = Path(config.resume_from)
        if checkpoint_path.exists():
            writer.checkpoint_path = checkpoint_path
            if writer.load_checkpoint():
                start_count = writer.state.samples_written
                print(f"Resuming from {start_count} samples")

    # Main curation loop
    print("\nStarting curation...")
    start_time = time.time()

    samples_needed = config.target_samples - start_count
    batch_count = 0
    docs_processed = 0

    doc_iterator = iter(loader)

    while writer.state.samples_written < config.target_samples:
        # Get batch of documents
        batch_docs = []
        while len(batch_docs) < config.batch_size:
            try:
                doc = next(doc_iterator)
            except StopIteration:
                break

            # Filter document
            filter_result = doc_filter.filter(doc)
            if filter_result.passed:
                # Clean content
                doc.text = cleaner.clean(doc.text)
                batch_docs.append(doc)

            docs_processed += 1

        if not batch_docs:
            print("No more documents available")
            break

        # Annotate batch
        results = annotator.annotate_batch(batch_docs)

        # Quality control
        passed_results = []
        for result in results:
            qc_result = qc.validate(result)
            if qc_result.passed:
                passed_results.append(qc_result.fixed_result or result)

        # Write results
        sources = [doc.source for doc in batch_docs[: len(passed_results)]]
        writer.write_batch(passed_results, sources)

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
    writer.finalize()

    # Print final statistics
    print("\n" + "=" * 60)
    print("Curation Complete")
    print("=" * 60)

    elapsed = time.time() - start_time
    print(f"\nTime elapsed: {elapsed / 60:.1f} minutes")
    print(f"Documents processed: {docs_processed}")
    print(f"Filter pass rate: {doc_filter.get_statistics()['pass_rate']:.2%}")
    print(f"Annotation success rate: {annotator.get_statistics()['success_rate']:.2%}")
    print(f"QC pass rate: {qc.get_statistics()['pass_rate']:.2%}")

    writer_stats = writer.get_statistics()
    print(f"\nFinal output:")
    print(f"  Samples: {writer_stats['samples_written']}")
    print(f"  Concepts: {writer_stats['num_concepts']}")
    print(f"  Relations: {writer_stats['num_relations']}")
    print(f"  Source distribution: {writer_stats['source_distribution']}")


def main():
    """Main entry point."""
    args = parse_args()
    config = create_config(args)
    run_curation(config, verbose=args.verbose)


if __name__ == "__main__":
    main()
