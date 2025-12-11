#!/usr/bin/env python3
"""Production training script for NeuroSymbolicLM.

This script implements a multi-stage training pipeline optimized for:
- Long context (up to 16k tokens with LongT5)
- Memory efficiency (gradient checkpointing, mixed precision)
- Single RTX 4090 (24GB VRAM)

Training Stages:
1. Entity/Relation Understanding (DocRED)
2. Knowledge-Grounded QA (instruction data)
3. Full Joint Training

Usage:
    # Quick test run
    python train_production.py --preset testing --max-samples 100
    
    # Full training on RTX 4090 with 8k context
    python train_production.py --preset 4090-8k
    
    # Full training with 16k context (memory intensive)
    python train_production.py --preset 4090-16k --gradient-accumulation 4
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import random

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Local imports
from config import ModelConfig, TrainingConfig, MODEL_PRESETS
from model import NeuroSymbolicLM
from data.collator import CognitiveCollator
from data.dataset import ToyCognitiveDataset
from data.pipeline import DataPipeline, build_vocab_from_datasets
from training import (
    Stage2_Symbolic_Trainer,
    Stage3_Decoder_Trainer,
    Stage4_Joint_Trainer,
    CheckpointManager,
    EarlyStopping,
    TrainingLogger,
)
from training.evaluation import evaluate_generation


def parse_args():
    parser = argparse.ArgumentParser(description="Production training for NeuroSymbolicLM")
    
    # Preset configurations
    parser.add_argument(
        "--preset", 
        type=str, 
        default="4090-8k",
        choices=list(MODEL_PRESETS.keys()),
        help="Hardware preset configuration"
    )
    
    # Model settings
    parser.add_argument("--model-name", type=str, default=None, help="Override model name")
    parser.add_argument("--max-input-length", type=int, default=None, help="Override max input length")
    parser.add_argument("--max-output-length", type=int, default=None, help="Override max output length")
    
    # Data settings
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--dataset", type=str, default="comprehensive_dataset.jsonl", help="Training dataset file to use")
    parser.add_argument("--eval-dataset", type=str, default=None, help="Evaluation/validation dataset file (same format as training dataset)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of training samples")
    parser.add_argument("--prepare-data", action="store_true", help="Download and prepare datasets")
    parser.add_argument("--data-sources", type=str, nargs="+", default=["dolly"], 
                       help="Data sources to prepare (dolly, docred, alpaca, oasst1)")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per training stage")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    
    # Memory efficiency
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--no-gradient-checkpointing", action="store_true", help="Disable gradient checkpointing")
    
    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    
    # Logging
    parser.add_argument("--log-dir", type=str, default="runs", help="Tensorboard log directory")
    parser.add_argument("--eval-every", type=int, default=2, help="Evaluate every N epochs")
    parser.add_argument("--num-eval-samples", type=int, default=5, help="Number of evaluation samples")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    
    # Debugging
    parser.add_argument("--debug", action="store_true", help="Debug mode (verbose output)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (setup only)")
    
    return parser.parse_args()


def setup_model_config(args) -> ModelConfig:
    """Create model configuration from args and preset."""
    config = MODEL_PRESETS[args.preset]()
    
    # Override with command line args
    if args.model_name:
        config.model_name = args.model_name
    if args.max_input_length:
        config.max_input_length = args.max_input_length
    if args.max_output_length:
        config.max_output_length = args.max_output_length
    if args.no_gradient_checkpointing:
        config.gradient_checkpointing = False
    
    return config


def prepare_datasets(args) -> Dict[str, Path]:
    """Download and prepare training datasets."""
    pipeline = DataPipeline(
        output_dir=args.data_dir,
        cache_dir="data/cache"
    )
    
    all_paths = {}
    
    for source in args.data_sources:
        print(f"\n{'='*60}")
        print(f"Preparing {source} dataset...")
        print('='*60)
        
        try:
            paths = pipeline.prepare_dataset(
                source,
                max_samples=args.max_samples
            )
            all_paths[source] = paths
        except Exception as e:
            print(f"Error preparing {source}: {e}")
            continue
    
    # Merge all training data
    if len(all_paths) > 1:
        train_paths = [p.get("train") or p.get("training") for p in all_paths.values() if p]
        train_paths = [p for p in train_paths if p and p.exists()]
        
        if train_paths:
            merged = pipeline.merge_datasets(
                train_paths,
                "merged_train",
                shuffle=True
            )
            all_paths["merged"] = {"train": merged}
    
    return all_paths


def load_dataset(
    path: Path,
    tokenizer,
    model_config: ModelConfig,
    max_samples: Optional[int] = None
) -> ToyCognitiveDataset:
    """Load and prepare dataset."""
    dataset = ToyCognitiveDataset(jsonl_file_path=str(path))
    
    # Limit samples if requested
    if max_samples is not None and len(dataset) > max_samples:
        dataset.data = dataset.data[:max_samples]
    
    return dataset


def create_dataloaders(
    dataset: Dataset,
    tokenizer,
    model_config: ModelConfig,
    batch_size: int,
    num_workers: int,
    include_responses: bool = True
) -> DataLoader:
    """Create dataloader with appropriate collator."""
    # Build concept and relation maps from dataset
    concept_map = {}
    relation_map = {}
    
    for sample in dataset:
        for concept_list in sample.get("concepts", []):
            for c in concept_list:
                if c not in concept_map:
                    concept_map[c] = len(concept_map)
        
        for rel in sample.get("relations", []):
            if len(rel) >= 3:
                r = rel[2]
                if r not in relation_map:
                    relation_map[r] = len(relation_map)
    
    collator = CognitiveCollator(
        tokenizer=tokenizer,
        concept_map=concept_map,
        relation_map=relation_map,
        include_responses=include_responses,
        max_length=model_config.max_input_length
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True  # For stable gradient accumulation
    )


def train_epoch(
    model: nn.Module,
    trainer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    epoch: int,
    logger: TrainingLogger,
    global_step: int,
    debug: bool = False
) -> tuple:
    """Train for one epoch - matches train.py approach exactly."""
    model.train()
    
    epoch_losses = []
    
    for step, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Call trainer.train_step - it handles everything internally (backward, clipping, step)
        loss = trainer.train_step(batch)
        
        if loss > 0:  # Skip batches with no valid labels
            epoch_losses.append(loss)
        
        global_step += 1
        logger.log_scalar("Train/Loss", loss, global_step)
        logger.log_scalar("Train/LR", optimizer.param_groups[0]["lr"], global_step)
        
        if debug and step % 100 == 0:
            print(f"  Step {step}: loss={loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")
    
    # Step scheduler after epoch (matches train.py)
    if scheduler:
        scheduler.step()
    
    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    return avg_loss, global_step


def run_training(
    model: NeuroSymbolicLM,
    tokenizer,
    train_dataset: Dataset,
    model_config: ModelConfig,
    args,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
    eval_dataset: Optional[Dataset] = None
):
    """Run the full training pipeline."""
    device = args.device
    model = model.to(device)
    
    # Create dataloader
    train_loader = create_dataloaders(
        train_dataset,
        tokenizer,
        model_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_responses=True
    )
    
    # Calculate total steps
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    print(f"\nTraining configuration:")
    print(f"  Total samples: {len(train_dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max input length: {model_config.max_input_length}")
    print(f"  Max output length: {model_config.max_output_length}")
    
    # Setup mixed precision (handled internally by trainers)
    use_amp = not args.no_amp and device == "cuda"
    
    global_step = 0
    
    # ========================================================================
    # Stage 1: Entity/Relation Training
    # ========================================================================
    print("\n" + "=" * 60)
    print("Stage 1: Entity/Relation Training")
    print("=" * 60)
    
    # Freeze decoder for this stage
    for param in model.t5.decoder.parameters():
        param.requires_grad = False
    for param in model.t5.lm_head.parameters():
        param.requires_grad = False
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps // 3,
        num_training_steps=total_steps // 3
    )
    
    trainer = Stage2_Symbolic_Trainer(
        model, optimizer,
        grad_clip_norm=args.max_grad_norm,
        use_amp=use_amp,
        device=device
    )
    
    early_stop = EarlyStopping(patience=3)
    
    for epoch in range(args.epochs):
        avg_loss, global_step = train_epoch(
            model, trainer, train_loader, optimizer, scheduler,
            device, epoch, logger, global_step, args.debug
        )
        
        print(f"Stage 1 Epoch {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % args.save_every == 0:
            checkpoint_manager.save(model, optimizer, epoch + 1, "stage1", avg_loss)
        
        if early_stop(avg_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # ========================================================================
    # Stage 2: Decoder Training
    # ========================================================================
    print("\n" + "=" * 60)
    print("Stage 2: Decoder Training")
    print("=" * 60)
    
    # Unfreeze decoder
    for param in model.t5.decoder.parameters():
        param.requires_grad = True
    for param in model.t5.lm_head.parameters():
        param.requires_grad = True
    
    # Optionally freeze encoder
    if model_config.freeze_encoder:
        for param in model.t5.encoder.parameters():
            param.requires_grad = False
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate * 0.5, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps // 3,
        num_training_steps=total_steps // 3
    )
    
    trainer = Stage3_Decoder_Trainer(
        model, optimizer,
        grad_clip_norm=args.max_grad_norm,
        use_amp=use_amp,
        device=device
    )
    
    early_stop = EarlyStopping(patience=3)
    
    for epoch in range(args.epochs):
        avg_loss, global_step = train_epoch(
            model, trainer, train_loader, optimizer, scheduler,
            device, epoch, logger, global_step, args.debug
        )
        
        print(f"Stage 2 Epoch {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")
        
        # Evaluation
        if (epoch + 1) % args.eval_every == 0:
            eval_ds = eval_dataset if eval_dataset is not None else train_dataset
            print(f"\n  Evaluating generation on {'validation' if eval_dataset else 'training'} set...")
            results = evaluate_generation(
                model, tokenizer, eval_ds,
                device=device, num_samples=args.num_eval_samples,
                max_length=model_config.max_output_length
            )
            for r in results[:3]:
                print(f"  Input: {r['input'][:80]}...")
                print(f"  Generated: {r['generated'][:80]}...")
                print()
        
        if (epoch + 1) % args.save_every == 0:
            checkpoint_manager.save(model, optimizer, epoch + 1, "stage2", avg_loss)
        
        if early_stop(avg_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # ========================================================================
    # Stage 3: Joint Training
    # ========================================================================
    print("\n" + "=" * 60)
    print("Stage 3: Joint End-to-End Training")
    print("=" * 60)
    
    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate * 0.1, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps // 3,
        num_training_steps=total_steps // 3
    )
    
    trainer = Stage4_Joint_Trainer(
        model, optimizer,
        grad_clip_norm=args.max_grad_norm,
        use_amp=use_amp,
        device=device
    )
    
    early_stop = EarlyStopping(patience=5)
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        avg_loss, global_step = train_epoch(
            model, trainer, train_loader, optimizer, scheduler,
            device, epoch, logger, global_step, args.debug
        )
        
        print(f"Stage 3 Epoch {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")
        
        # Evaluation
        if (epoch + 1) % args.eval_every == 0:
            eval_ds = eval_dataset if eval_dataset is not None else train_dataset
            print(f"\n  Evaluating generation on {'validation' if eval_dataset else 'training'} set...")
            results = evaluate_generation(
                model, tokenizer, eval_ds,
                device=device, num_samples=args.num_eval_samples,
                max_length=model_config.max_output_length
            )
            for r in results[:3]:
                print(f"  Input: {r['input'][:80]}...")
                print(f"  Generated: {r['generated'][:80]}...")
                print()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_manager.save(model, optimizer, epoch + 1, "best", avg_loss)
        
        if (epoch + 1) % args.save_every == 0:
            checkpoint_manager.save(model, optimizer, epoch + 1, "stage3", avg_loss)
        
        if early_stop(avg_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Save final model
    checkpoint_manager.save(model, optimizer, args.epochs, "final", avg_loss)
    
    return model


def main():
    args = parse_args()
    
    print("=" * 70)
    print("NeuroSymbolicLM Production Training")
    print("=" * 70)
    print(f"Preset: {args.preset}")
    print(f"Device: {args.device}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Setup model config
    model_config = setup_model_config(args)
    
    print(f"\nModel configuration:")
    print(f"  Model: {model_config.model_name}")
    print(f"  Max input length: {model_config.max_input_length}")
    print(f"  Max output length: {model_config.max_output_length}")
    print(f"  Gradient checkpointing: {model_config.gradient_checkpointing}")
    
    # Prepare data if requested
    if args.prepare_data:
        print("\n" + "=" * 60)
        print("Preparing Datasets")
        print("=" * 60)
        prepare_datasets(args)
        print("\nData preparation complete!")
        if args.dry_run:
            return
    
    # Find training data
    train_path = Path(args.dataset)
    if not train_path.exists():
        # Try relative to data_dir if not found
        train_path = Path(args.data_dir) / args.dataset
        if not train_path.exists():
            # Try current directory
            train_path = Path(args.dataset)
            if not train_path.exists():
                print(f"\nTraining dataset not found: {args.dataset}")
                print("Please specify a valid dataset file with --dataset")
                sys.exit(1)
    
    print(f"\nTraining data: {train_path}")
    
    # Find eval data if specified
    eval_path = None
    if args.eval_dataset:
        eval_path = Path(args.eval_dataset)
        if not eval_path.exists():
            # Try relative to data_dir if not found
            eval_path = Path(args.data_dir) / args.eval_dataset
            if not eval_path.exists():
                # Try current directory
                eval_path = Path(args.eval_dataset)
                if not eval_path.exists():
                    print(f"\nWarning: Evaluation dataset not found: {args.eval_dataset}")
                    print("Continuing without evaluation dataset")
                    eval_path = None
        
        if eval_path:
            print(f"Evaluation data: {eval_path}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {model_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load training dataset
    print(f"Loading training dataset...")
    train_dataset = load_dataset(train_path, tokenizer, model_config, args.max_samples)
    print(f"  Loaded {len(train_dataset)} training samples")
    
    # Load eval dataset if provided
    eval_dataset = None
    if eval_path:
        print(f"Loading evaluation dataset...")
        eval_dataset = load_dataset(eval_path, tokenizer, model_config, None)
        print(f"  Loaded {len(eval_dataset)} evaluation samples")
    
    # Infer dimensions from dataset
    n_entity_types = model_config.n_entity_types
    n_relations = model_config.n_relations
    n_concepts = model_config.n_concepts
    
    # Count actual values in dataset (include both train and eval)
    all_concepts = set()
    all_relations = set()
    for sample in train_dataset:
        for concept_list in sample.get("concepts", []):
            for c in concept_list:
                all_concepts.add(c)
        for rel in sample.get("relations", []):
            if len(rel) >= 3:
                all_relations.add(rel[2])
    
    # Also count from eval dataset if provided
    if eval_dataset:
        for sample in eval_dataset:
            for concept_list in sample.get("concepts", []):
                for c in concept_list:
                    all_concepts.add(c)
            for rel in sample.get("relations", []):
                if len(rel) >= 3:
                    all_relations.add(rel[2])
    
    n_concepts = max(n_concepts, len(all_concepts) + 100)
    n_relations = max(n_relations, len(all_relations) + 50)
    
    print(f"  Concepts: {len(all_concepts)} (capacity: {n_concepts})")
    print(f"  Relations: {len(all_relations)} (capacity: {n_relations})")
    
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
        freeze_encoder=model_config.freeze_encoder,
        freeze_decoder=model_config.freeze_decoder,
        gradient_checkpointing=model_config.gradient_checkpointing,
        max_input_length=model_config.max_input_length,
        max_output_length=model_config.max_output_length,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup logging and checkpointing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.preset}_{timestamp}"
    
    logger = TrainingLogger(
        log_dir=args.log_dir,
        experiment_name=run_name,
        enable_tensorboard=True
    )
    
    checkpoint_manager = CheckpointManager(
        save_dir=args.checkpoint_dir,
        max_checkpoints=5
    )
    
    if args.dry_run:
        print("\nDry run complete. Model and data are ready.")
        return
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Run training
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    try:
        model = run_training(
            model, tokenizer, train_dataset,
            model_config, args, logger, checkpoint_manager,
            eval_dataset=eval_dataset
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        checkpoint_manager.save(model, None, 0, "interrupted", 0.0)
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        checkpoint_manager.save(model, None, 0, "error", 0.0)
        raise
    finally:
        logger.close()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
