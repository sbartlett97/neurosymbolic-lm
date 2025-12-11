#!/usr/bin/env python3
"""Unified training script for NeuroSymbolicLM.

This script consolidates training logic into a single, configurable pipeline that supports:
1. Multi-stage training (entity/relation -> decoder -> joint)
2. Custom datasets in JSONL format
3. Memory-efficient training with gradient checkpointing and AMP
4. Automatic dataset preparation from HuggingFace
5. Proper evaluation on held-out eval dataset

Usage:
    # Quick test with local data
    python train.py --dataset comprehensive_dataset.jsonl --epochs 3
    
    # Training with separate eval dataset
    python train.py --dataset comprehensive_dataset.jsonl --eval-dataset eval_dataset.jsonl
    
    # Prepare data from HuggingFace and train
    python train.py --prepare-data --data-sources rebel dolly
    
    # Full training with all stages
    python train.py --dataset data/staged/stage1_entity_relation.jsonl \
                    --stage2-dataset data/staged/stage2_instruction.jsonl
    
    # Resume from checkpoint
    python train.py --dataset mydata.jsonl --resume checkpoints/checkpoint.pt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import ModelConfig, MODEL_PRESETS
from model import NeuroSymbolicLM
from data.collator import CognitiveCollator
from data.dataset import ToyCognitiveDataset
from training import (
    Stage2_Symbolic_Trainer,
    Stage3_Decoder_Trainer,
    Stage4_Joint_Trainer,
    CheckpointManager,
    EarlyStopping,
    TrainingLogger,
    evaluate_generation,
    compute_aggregate_metrics,
    print_generation_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified training for NeuroSymbolicLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--dataset", type=str, default="comprehensive_dataset.jsonl",
                           help="Primary training dataset (JSONL format)")
    data_group.add_argument("--eval-dataset", type=str, default="eval_dataset.jsonl",
                           help="Evaluation dataset (JSONL format)")
    data_group.add_argument("--stage2-dataset", type=str, default=None,
                           help="Optional separate dataset for decoder training")
    data_group.add_argument("--max-samples", type=int, default=None,
                           help="Limit number of training samples")
    data_group.add_argument("--prepare-data", action="store_true",
                           help="Download and prepare datasets from HuggingFace")
    data_group.add_argument("--data-sources", type=str, nargs="+", 
                           default=["rebel", "dolly"],
                           help="Data sources to prepare: rebel, dolly, alpaca")
    
    # Model
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--preset", type=str, default="testing",
                            choices=list(MODEL_PRESETS.keys()),
                            help="Hardware preset configuration")
    model_group.add_argument("--model-name", type=str, default=None,
                            help="Override model backbone")
    
    # Training stages
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--stages", type=str, nargs="+", 
                            default=["symbolic", "decoder", "joint"],
                            choices=["symbolic", "decoder", "joint"],
                            help="Training stages to run")
    train_group.add_argument("--epochs", type=int, default=5,
                            help="Epochs per stage")
    train_group.add_argument("--batch-size", type=int, default=4,
                            help="Training batch size")
    train_group.add_argument("--lr", type=float, default=1e-4,
                            help="Learning rate")
    train_group.add_argument("--warmup-ratio", type=float, default=0.1,
                            help="Warmup ratio")
    train_group.add_argument("--patience", type=int, default=3,
                            help="Early stopping patience")
    
    # Hardware
    hw_group = parser.add_argument_group("Hardware")
    hw_group.add_argument("--device", type=str, default="cuda",
                         help="Device (cuda/cpu)")
    hw_group.add_argument("--no-amp", action="store_true",
                         help="Disable mixed precision training")
    hw_group.add_argument("--num-workers", type=int, default=2,
                         help="DataLoader workers")
    
    # Checkpointing
    ckpt_group = parser.add_argument_group("Checkpointing")
    ckpt_group.add_argument("--output-dir", type=str, default="checkpoints",
                           help="Output directory for checkpoints")
    ckpt_group.add_argument("--resume", type=str, default=None,
                           help="Resume from checkpoint")
    ckpt_group.add_argument("--save-every", type=int, default=1,
                           help="Save checkpoint every N epochs")
    
    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--log-dir", type=str, default="runs",
                          help="TensorBoard log directory")
    log_group.add_argument("--eval-every", type=int, default=2,
                          help="Evaluate generation every N epochs")
    log_group.add_argument("--debug", action="store_true",
                          help="Enable debug output")
    
    return parser.parse_args()


def prepare_datasets(args) -> Dict[str, Path]:
    """Prepare datasets from HuggingFace sources."""
    from data.staged_pipeline import StagedDataPipeline
    
    pipeline = StagedDataPipeline()
    paths = {}
    
    if "rebel" in args.data_sources:
        print("\nPreparing Stage 1 data (REBEL - entity/relation)...")
        pipeline.prepare_stage1_data(max_samples=args.max_samples)
        paths["stage1"] = pipeline.get_stage1_path()
    
    if "dolly" in args.data_sources or "alpaca" in args.data_sources:
        sources = [s for s in args.data_sources if s in ["dolly", "alpaca"]]
        print(f"\nPreparing Stage 2 data ({sources})...")
        pipeline.prepare_stage2_data(
            sources=sources,
            max_samples_per_source=args.max_samples
        )
        paths["stage2"] = pipeline.get_stage2_path()
    
    return paths


def load_dataset(path: Path, max_samples: Optional[int] = None) -> ToyCognitiveDataset:
    """Load dataset from JSONL file."""
    dataset = ToyCognitiveDataset(str(path))
    if max_samples and len(dataset) > max_samples:
        dataset.data = dataset.data[:max_samples]
    return dataset


def extract_vocab_from_dataset(dataset: ToyCognitiveDataset) -> Tuple[Dict, Dict, int, int, int]:
    """Extract vocabulary mappings from dataset."""
    concepts = set()
    relations = set()
    entity_types = set()
    
    for sample in dataset:
        for concept_list in sample.get("concepts", []):
            if isinstance(concept_list, list):
                concepts.update(concept_list)
                entity_types.update(concept_list)
            else:
                concepts.add(concept_list)
                entity_types.add(concept_list)
        
        for rel in sample.get("relations", []):
            if len(rel) >= 3:
                relations.add(rel[2])
    
    # Build maps (1-indexed, 0 is padding)
    concept_map = {c: i + 1 for i, c in enumerate(sorted(concepts))}
    relation_map = {r: i + 1 for i, r in enumerate(sorted(relations))}
    
    n_entity_types = max(16, len(entity_types) + 4)
    n_relations = max(64, len(relations) + 10)
    n_concepts = max(256, len(concepts) + 50)
    
    return concept_map, relation_map, n_entity_types, n_relations, n_concepts


def create_dataloader(
    dataset: ToyCognitiveDataset,
    tokenizer,
    concept_map: Dict,
    relation_map: Dict,
    model_config: ModelConfig,
    batch_size: int,
    num_workers: int,
    include_responses: bool
) -> DataLoader:
    """Create dataloader with collator."""
    collator = CognitiveCollator(
        tokenizer=tokenizer,
        concept_map=concept_map,
        relation_map=relation_map,
        include_responses=include_responses,
        max_length=model_config.max_input_length,
        max_output_length=model_config.max_output_length
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=len(dataset) > batch_size
    )


def run_evaluation(
    model: NeuroSymbolicLM,
    tokenizer,
    eval_dataset: ToyCognitiveDataset,
    device: str,
    model_config: ModelConfig,
    stage_name: str = "Eval",
    num_samples: Optional[int] = None,
    logger: Optional[TrainingLogger] = None,
    global_step: int = 0
) -> Dict[str, float]:
    """
    Run comprehensive evaluation on the eval dataset.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for encoding/decoding
        eval_dataset: Dataset to evaluate on
        device: Device to run on
        model_config: Model configuration
        stage_name: Name for logging
        num_samples: Number of samples to evaluate (None = all)
        logger: Optional logger for metrics
        global_step: Current training step for logging
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Count samples with responses for evaluation
    samples_with_responses = [
        s for s in eval_dataset 
        if s.get("should_respond", 0) == 1 and s.get("response", "").strip()
    ]
    
    if num_samples is None:
        num_samples = len(samples_with_responses)
    else:
        num_samples = min(num_samples, len(samples_with_responses))
    
    if num_samples == 0:
        print(f"  No samples with responses found in eval dataset")
        return {}
    
    print(f"\n  Running evaluation on {num_samples} samples...")
    
    results = evaluate_generation(
        model, tokenizer, eval_dataset,
        device=device,
        num_samples=num_samples,
        max_length=model_config.max_output_length,
        compute_metrics=True
    )
    
    # Compute aggregate metrics
    metrics = compute_aggregate_metrics(results)
    
    # Print results
    print_generation_results(results, num_to_print=min(5, len(results)), show_metrics=True)
    
    # Log metrics
    if logger and metrics:
        for name, value in metrics.items():
            logger.log_scalar(f"{stage_name}/{name}", value, global_step)
    
    model.train()
    return metrics


def run_final_evaluation(
    model: NeuroSymbolicLM,
    tokenizer,
    train_dataset: ToyCognitiveDataset,
    eval_dataset: Optional[ToyCognitiveDataset],
    device: str,
    model_config: ModelConfig,
    logger: Optional[TrainingLogger] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run final comprehensive evaluation on both train and eval datasets.
    
    Returns metrics for both datasets.
    """
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    all_metrics = {}
    
    # Evaluate on training set (subset)
    print("\n--- Training Set Evaluation (sample) ---")
    train_metrics = run_evaluation(
        model, tokenizer, train_dataset, device, model_config,
        stage_name="Final/Train", num_samples=10, logger=logger
    )
    all_metrics["train"] = train_metrics
    
    # Evaluate on eval set (full)
    if eval_dataset:
        print("\n--- Evaluation Set (held-out) ---")
        eval_metrics = run_evaluation(
            model, tokenizer, eval_dataset, device, model_config,
            stage_name="Final/Eval", num_samples=None, logger=logger
        )
        all_metrics["eval"] = eval_metrics
        
        # Print comparison
        if train_metrics and eval_metrics:
            print("\n" + "-" * 40)
            print("Train vs Eval Comparison:")
            print("-" * 40)
            for metric in ["avg_bleu", "avg_entity_f1"]:
                train_val = train_metrics.get(metric, 0)
                eval_val = eval_metrics.get(metric, 0)
                diff = eval_val - train_val
                print(f"  {metric}: Train={train_val:.4f}, Eval={eval_val:.4f} (diff={diff:+.4f})")
    
    return all_metrics


def train_stage(
    stage_name: str,
    model: NeuroSymbolicLM,
    trainer,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    num_epochs: int,
    device: str,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
    early_stopping: EarlyStopping,
    tokenizer=None,
    train_dataset=None,
    eval_dataset=None,
    model_config=None,
    eval_every: int = 2,
    save_every: int = 1,
    debug: bool = False
) -> Tuple[float, int]:
    """Train a single stage with periodic evaluation."""
    print(f"\n{'='*60}")
    print(f"Training Stage: {stage_name}")
    print(f"{'='*60}")
    
    model.train()
    global_step = 0
    best_loss = float('inf')
    zero_loss_count = 0
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        epoch_zero_losses = 0
        
        for step, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Debug first batch of first epoch
            if debug and epoch == 0 and step == 0:
                print(f"\n  DEBUG - First batch contents:")
                print(f"    input_ids shape: {batch['input_ids'].shape}")
                print(f"    entity_ids shape: {batch['entity_ids'].shape}")
                print(f"    entity_ids sample: {batch['entity_ids'][0][:5]}")
                print(f"    entity_type_labels shape: {batch['entity_type_labels'].shape}")
                print(f"    entity_type_labels sample: {batch['entity_type_labels'][0][:5]}")
                print(f"    concept_labels shape: {batch['concept_labels'].shape}")
                print(f"    concept_labels sum per sample: {batch['concept_labels'].sum(dim=(1,2))}")
                print(f"    relations sample: {batch['relations'][0][:3] if batch['relations'][0] else 'empty'}")
                if 'decoder_input_ids' in batch:
                    print(f"    decoder_input_ids shape: {batch['decoder_input_ids'].shape}")
                    print(f"    decoder_labels shape: {batch['decoder_labels'].shape}")
                    valid_labels = (batch['decoder_labels'] != -100).sum()
                    print(f"    valid decoder labels: {valid_labels}")
            
            loss = trainer.train_step(batch)
            
            # Track zero losses
            if loss == 0.0 or loss != loss:
                epoch_zero_losses += 1
                if epoch == 0 and step < 3:
                    print(f"    WARNING: Step {step} returned loss=0.0")
                continue
            
            total_loss += loss
            num_batches += 1
            global_step += 1
            
            if scheduler:
                scheduler.step()
            
            logger.log_scalar(f"{stage_name}/Loss", loss, global_step)
            
            if debug and step % 50 == 0:
                print(f"  Step {step}: loss={loss:.4f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f} (valid batches: {num_batches}/{len(train_loader)}, zero losses: {epoch_zero_losses})")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_manager.save(model, optimizer, epoch + 1, stage_name, avg_loss)
        
        # Track best
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Evaluate on eval dataset (for decoder-related stages)
        if tokenizer and model_config and stage_name.lower() in ["decoder", "joint"]:
            if (epoch + 1) % eval_every == 0:
                # Quick eval on training data
                if train_dataset:
                    print("  Quick train eval:")
                    model.eval()
                    results = evaluate_generation(
                        model, tokenizer, train_dataset,
                        device=device, num_samples=3,
                        max_length=model_config.max_output_length
                    )
                    for r in results[:2]:
                        print(f"    Q: {r['input'][:50]}...")
                        print(f"    A: {r['generated'][:50]}...")
                    model.train()
                
                # Eval on held-out eval dataset
                if eval_dataset:
                    print("  Eval dataset metrics:")
                    run_evaluation(
                        model, tokenizer, eval_dataset, device, model_config,
                        stage_name=f"{stage_name}/Epoch{epoch+1}",
                        num_samples=5,
                        logger=logger,
                        global_step=global_step
                    )
        
        # Early stopping
        if early_stopping(avg_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    return best_loss, global_step


def main():
    args = parse_args()
    
    print("=" * 70)
    print("NeuroSymbolicLM Unified Training")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Preset: {args.preset}")
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Prepare data if requested
    if args.prepare_data:
        prepared_paths = prepare_datasets(args)
        if not args.dataset or not Path(args.dataset).exists():
            if "stage1" in prepared_paths:
                args.dataset = str(prepared_paths["stage1"])
            if "stage2" in prepared_paths and not args.stage2_dataset:
                args.stage2_dataset = str(prepared_paths["stage2"])
    
    # Load model config
    model_config = MODEL_PRESETS[args.preset]()
    if args.model_name:
        model_config.model_name = args.model_name
    
    print(f"\nModel: {model_config.model_name}")
    print(f"Max input length: {model_config.max_input_length}")
    print(f"Max output length: {model_config.max_output_length}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load primary dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"\nError: Dataset not found: {dataset_path}")
        print("Run with --prepare-data to download datasets, or provide a valid --dataset path")
        sys.exit(1)
    
    print(f"\nLoading dataset: {dataset_path}")
    train_dataset = load_dataset(dataset_path, args.max_samples)
    print(f"  Loaded {len(train_dataset)} samples")
    
    # Load stage 2 dataset if provided
    stage2_dataset = None
    if args.stage2_dataset and Path(args.stage2_dataset).exists():
        print(f"Loading stage 2 dataset: {args.stage2_dataset}")
        stage2_dataset = load_dataset(Path(args.stage2_dataset), args.max_samples)
        print(f"  Loaded {len(stage2_dataset)} samples")
    
    # Load evaluation dataset
    eval_dataset = None
    eval_path = Path(args.eval_dataset)
    if eval_path.exists():
        print(f"\nLoading eval dataset: {eval_path}")
        eval_dataset = load_dataset(eval_path)
        eval_samples_with_response = sum(1 for s in eval_dataset if s.get("should_respond", 0) == 1)
        print(f"  Loaded {len(eval_dataset)} samples ({eval_samples_with_response} with responses)")
    else:
        print(f"\nNo eval dataset found at {eval_path} - will skip held-out evaluation")
    
    # Extract vocabulary
    concept_map, relation_map, n_entity_types, n_relations, n_concepts = \
        extract_vocab_from_dataset(train_dataset)
    
    if stage2_dataset:
        c2, r2, ne2, nr2, nc2 = extract_vocab_from_dataset(stage2_dataset)
        concept_map.update(c2)
        relation_map.update(r2)
        n_entity_types = max(n_entity_types, ne2)
        n_relations = max(n_relations, nr2)
        n_concepts = max(n_concepts, nc2)
    
    print(f"  Entity types: {n_entity_types}")
    print(f"  Relations: {len(relation_map)} -> capacity {n_relations}")
    print(f"  Concepts: {len(concept_map)} -> capacity {n_concepts}")
    
    # Create model
    print(f"\nCreating model...")
    model = NeuroSymbolicLM(
        model_name=model_config.model_name,
        n_entity_types=n_entity_types,
        n_relations=n_relations,
        n_concepts=n_concepts,
        concept_dim=model_config.concept_dim,
        node_dim=model_config.node_dim,
        max_nodes=model_config.max_nodes,
        gradient_checkpointing=model_config.gradient_checkpointing,
        max_input_length=model_config.max_input_length,
        max_output_length=model_config.max_output_length,
    )
    
    # Resume from checkpoint
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Setup logging and checkpointing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TrainingLogger(log_dir=args.log_dir, experiment_name=f"unified_{timestamp}")
    checkpoint_manager = CheckpointManager(save_dir=args.output_dir, max_checkpoints=5)
    
    use_amp = not args.no_amp and args.device == "cuda"
    print(f"  Mixed precision (AMP): {use_amp}")
    
    # Training stages
    print(f"\nStages to run: {args.stages}")
    
    # Stage 1: Symbolic (Entity/Relation)
    if "symbolic" in args.stages:
        print("\n" + "="*60)
        print("Stage 1: Symbolic Training (Entity/Relation)")
        print("="*60)
        
        # Freeze decoder
        for p in model.t5.decoder.parameters():
            p.requires_grad = False
        for p in model.t5.lm_head.parameters():
            p.requires_grad = False
        
        train_loader = create_dataloader(
            train_dataset, tokenizer, concept_map, relation_map,
            model_config, args.batch_size, args.num_workers,
            include_responses=False
        )
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=args.lr)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * args.warmup_ratio), total_steps
        )
        
        trainer = Stage2_Symbolic_Trainer(
            model, optimizer, use_amp=use_amp, device=args.device
        )
        
        train_stage(
            "Symbolic", model, trainer, train_loader,
            optimizer, scheduler, args.epochs, args.device,
            logger, checkpoint_manager, EarlyStopping(patience=args.patience),
            tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset,
            model_config=model_config, debug=args.debug, save_every=args.save_every
        )
    
    # Stage 2: Decoder
    if "decoder" in args.stages:
        print("\n" + "="*60)
        print("Stage 2: Decoder Training")
        print("="*60)
        
        # Unfreeze decoder, freeze symbolic heads
        for p in model.t5.decoder.parameters():
            p.requires_grad = True
        for p in model.t5.lm_head.parameters():
            p.requires_grad = True
        
        # Freeze symbolic components
        for p in model.token_ent.parameters():
            p.requires_grad = False
        for p in model.concept_bank.parameters():
            p.requires_grad = False
        for p in model.rel_scorer.parameters():
            p.requires_grad = False
        for p in model.gnn.parameters():
            p.requires_grad = False
        
        # Use stage2 dataset if available, otherwise use primary
        decoder_dataset = stage2_dataset if stage2_dataset else train_dataset
        
        train_loader = create_dataloader(
            decoder_dataset, tokenizer, concept_map, relation_map,
            model_config, args.batch_size, args.num_workers,
            include_responses=True
        )
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=args.lr * 0.5)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * args.warmup_ratio), total_steps
        )
        
        trainer = Stage3_Decoder_Trainer(
            model, optimizer, use_amp=use_amp, device=args.device
        )
        
        train_stage(
            "Decoder", model, trainer, train_loader,
            optimizer, scheduler, args.epochs, args.device,
            logger, checkpoint_manager, EarlyStopping(patience=args.patience),
            tokenizer=tokenizer, train_dataset=decoder_dataset, eval_dataset=eval_dataset,
            model_config=model_config, eval_every=args.eval_every, 
            debug=args.debug, save_every=args.save_every
        )
    
    # Stage 3: Joint
    if "joint" in args.stages:
        print("\n" + "="*60)
        print("Stage 3: Joint Training")
        print("="*60)
        
        # Unfreeze everything
        for p in model.parameters():
            p.requires_grad = True
        
        # Use the dataset with responses
        joint_dataset = stage2_dataset if stage2_dataset else train_dataset
        
        train_loader = create_dataloader(
            joint_dataset, tokenizer, concept_map, relation_map,
            model_config, args.batch_size, args.num_workers,
            include_responses=True
        )
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=args.lr * 0.1)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * args.warmup_ratio), total_steps
        )
        
        trainer = Stage4_Joint_Trainer(
            model, optimizer, use_amp=use_amp, device=args.device
        )
        
        train_stage(
            "Joint", model, trainer, train_loader,
            optimizer, scheduler, args.epochs, args.device,
            logger, checkpoint_manager, EarlyStopping(patience=args.patience + 2),
            tokenizer=tokenizer, train_dataset=joint_dataset, eval_dataset=eval_dataset,
            model_config=model_config, eval_every=args.eval_every, 
            debug=args.debug, save_every=args.save_every
        )
    
    # Run final comprehensive evaluation
    final_metrics = run_final_evaluation(
        model, tokenizer, train_dataset, eval_dataset,
        args.device, model_config, logger
    )
    
    # Save final model
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    final_path = Path(args.output_dir) / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "model_name": model_config.model_name,
            "n_entity_types": n_entity_types,
            "n_relations": n_relations,
            "n_concepts": n_concepts,
        },
        "concept_map": concept_map,
        "relation_map": relation_map,
        "final_metrics": final_metrics,
    }, final_path)
    print(f"Saved final model to {final_path}")
    
    # Print final metrics summary
    if final_metrics:
        print("\nFinal Metrics Summary:")
        for split, metrics in final_metrics.items():
            if metrics:
                print(f"  {split.upper()}:")
                for name, value in metrics.items():
                    print(f"    {name}: {value:.4f}")
    
    logger.close()


if __name__ == "__main__":
    main()
