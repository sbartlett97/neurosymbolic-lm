#!/usr/bin/env python3
"""Staged training script following modern LLM training paradigm.

Training Stages:
1. Entity/Relation Training - Uses REBEL for entity extraction, relation 
   classification, concept mapping, and GNN training
2. Instruction Tuning - Uses Dolly/Alpaca for decoder training with simple
   input/output pairs (no entity annotations needed)
3. Joint Fine-tuning - Optional stage to integrate all components

This follows the standard approach:
- Pre-training: We use pre-trained T5/LongT5 (skip)
- Task-specific training: Entity/relation heads
- Instruction tuning: Response generation
- Integration: Joint fine-tuning

Usage:
    # Prepare data
    python train_staged.py --prepare-data
    
    # Run all stages
    python train_staged.py --preset 4090-8k
    
    # Run specific stage
    python train_staged.py --stage 1 --dataset data/staged/stage1_entity_relation.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Dict, List, Any
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
from data.staged_pipeline import (
    StagedDataPipeline,
    get_entity_types_from_dataset,
    get_relations_from_dataset
)
from training import (
    Stage2_Symbolic_Trainer,
    Stage3_Decoder_Trainer,
    Stage4_Joint_Trainer,
    CheckpointManager,
    EarlyStopping,
    TrainingLogger,
    evaluate_generation,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Staged training for NeuroSymbolicLM")
    
    # Data preparation
    parser.add_argument("--prepare-data", action="store_true", help="Prepare training data")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per dataset")
    
    # Stage selection
    parser.add_argument("--stage", type=str, choices=["1", "2", "3", "all"], default="all",
                       help="Training stage: 1=entity/relation, 2=instruction, 3=joint, all=all stages")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset path")
    
    # Model
    parser.add_argument("--preset", type=str, default="4090-8k", choices=list(MODEL_PRESETS.keys()))
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    
    # Training
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per stage")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--debug", action="store_true")
    
    return parser.parse_args()


def create_collator(tokenizer, dataset, model_config: ModelConfig, include_responses: bool):
    """Create collator with proper mappings from dataset."""
    # Extract concepts and relations from dataset
    concept_set = set()
    relation_set = set()
    
    for sample in dataset:
        for concept_list in sample.get("concepts", []):
            if isinstance(concept_list, list):
                concept_set.update(concept_list)
            else:
                concept_set.add(concept_list)
        for rel in sample.get("relations", []):
            if len(rel) >= 3:
                relation_set.add(rel[2])
    
    concept_map = {c: i + 1 for i, c in enumerate(sorted(concept_set))}
    relation_map = {r: i + 1 for i, r in enumerate(sorted(relation_set))}
    
    return CognitiveCollator(
        tokenizer=tokenizer,
        concept_map=concept_map,
        relation_map=relation_map,
        include_responses=include_responses,
        max_length=model_config.max_input_length,
        max_output_length=model_config.max_output_length
    )


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
    tokenizer=None,
    dataset=None,
    model_config=None,
    debug: bool = False
):
    """Train a single stage."""
    print(f"\n{'='*60}")
    print(f"Training: {stage_name}")
    print(f"{'='*60}")
    
    model.train()
    global_step = 0
    early_stop = EarlyStopping(patience=3)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for step, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            loss = trainer.train_step(batch)
            
            if loss == 0.0 or loss != loss:  # Skip invalid losses
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
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_manager.save(model, optimizer, epoch + 1, stage_name, avg_loss)
        
        # Evaluation for decoder stages
        if tokenizer and dataset and model_config and "Instruction" in stage_name:
            if (epoch + 1) % 2 == 0:
                print("  Evaluating generation...")
                results = evaluate_generation(
                    model, tokenizer, dataset,
                    device=device, num_samples=3,
                    max_length=model_config.max_output_length
                )
                for r in results[:2]:
                    print(f"  Q: {r['input'][:60]}...")
                    print(f"  A: {r['generated'][:60]}...")
        
        if early_stop(avg_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    return avg_loss


def main():
    args = parse_args()
    
    print("=" * 70)
    print("NeuroSymbolicLM Staged Training")
    print("=" * 70)
    
    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Prepare data if requested
    if args.prepare_data:
        print("\nPreparing training data...")
        pipeline = StagedDataPipeline()
        pipeline.prepare_all(
            max_stage1_samples=args.max_samples,
            max_stage2_samples_per_source=args.max_samples
        )
        print("\nData preparation complete!")
        if args.stage == "all" and not args.dataset:
            pass  # Continue to training
        else:
            return
    
    # Model config
    model_config = MODEL_PRESETS[args.preset]()
    
    # Tokenizer
    print(f"\nLoading tokenizer: {model_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Paths for staged data
    stage1_path = Path("data/staged/stage1_entity_relation.jsonl")
    stage2_path = Path("data/staged/stage2_instruction.jsonl")
    
    # Determine which datasets exist
    has_stage1 = stage1_path.exists()
    has_stage2 = stage2_path.exists()
    
    if not has_stage1 and not has_stage2 and not args.dataset:
        print("\nNo training data found. Run with --prepare-data first.")
        return
    
    # Load appropriate dataset based on stage
    if args.dataset:
        dataset_path = Path(args.dataset)
    elif args.stage == "1" and has_stage1:
        dataset_path = stage1_path
    elif args.stage == "2" and has_stage2:
        dataset_path = stage2_path
    elif args.stage == "all":
        # Will load both datasets
        dataset_path = None
    else:
        print(f"No dataset found for stage {args.stage}")
        return
    
    # Get entity types and relations for model dimensions
    n_entity_types = 16
    n_relations = 64
    n_concepts = 256
    
    if has_stage1:
        stage1_dataset = ToyCognitiveDataset(str(stage1_path))
        entity_types = get_entity_types_from_dataset(stage1_dataset)
        relations = get_relations_from_dataset(stage1_dataset)
        n_entity_types = max(n_entity_types, len(entity_types) + 4)
        n_relations = max(n_relations, len(relations) + 10)
        n_concepts = max(n_concepts, len(entity_types) + 100)
        print(f"Stage 1 dataset: {len(stage1_dataset)} samples")
        print(f"  Entity types: {len(entity_types)}, Relations: {len(relations)}")
    
    if has_stage2:
        stage2_dataset = ToyCognitiveDataset(str(stage2_path))
        print(f"Stage 2 dataset: {len(stage2_dataset)} samples")
    
    # Create model
    print(f"\nCreating model: {model_config.model_name}")
    model = NeuroSymbolicLM(
        model_name=model_config.model_name,
        n_entity_types=n_entity_types,
        n_relations=n_relations,
        n_concepts=n_concepts,
        gradient_checkpointing=model_config.gradient_checkpointing,
        max_input_length=model_config.max_input_length,
        max_output_length=model_config.max_output_length,
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Setup logging and checkpointing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TrainingLogger(log_dir="runs", experiment_name=f"staged_{timestamp}")
    checkpoint_manager = CheckpointManager(save_dir=args.output_dir, max_checkpoints=5)
    
    use_amp = not args.no_amp and args.device == "cuda"
    
    # ========================================================================
    # Stage 1: Entity/Relation Training
    # ========================================================================
    if args.stage in ["1", "all"] and has_stage1:
        print("\n" + "=" * 60)
        print("Stage 1: Entity/Relation Training (REBEL)")
        print("=" * 60)
        print("Training: Entity classifier, Concept bank, GNN, Relation scorer")
        
        # Freeze decoder
        for p in model.t5.decoder.parameters():
            p.requires_grad = False
        for p in model.t5.lm_head.parameters():
            p.requires_grad = False
        
        collator = create_collator(tokenizer, stage1_dataset, model_config, include_responses=False)
        train_loader = DataLoader(
            stage1_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collator, num_workers=2, pin_memory=True
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
            "Stage1_EntityRelation", model, trainer, train_loader,
            optimizer, scheduler, args.epochs, args.device,
            logger, checkpoint_manager, debug=args.debug
        )
    
    # ========================================================================
    # Stage 2: Instruction Tuning
    # ========================================================================
    if args.stage in ["2", "all"] and has_stage2:
        print("\n" + "=" * 60)
        print("Stage 2: Instruction Tuning (Dolly/Alpaca)")
        print("=" * 60)
        print("Training: Decoder for response generation")
        
        # Unfreeze decoder, optionally freeze encoder
        for p in model.t5.decoder.parameters():
            p.requires_grad = True
        for p in model.t5.lm_head.parameters():
            p.requires_grad = True
        
        # Freeze entity/relation heads (keep what we learned)
        for p in model.token_ent.parameters():
            p.requires_grad = False
        for p in model.concept_bank.parameters():
            p.requires_grad = False
        for p in model.rel_scorer.parameters():
            p.requires_grad = False
        for p in model.gnn.parameters():
            p.requires_grad = False
        
        collator = create_collator(tokenizer, stage2_dataset, model_config, include_responses=True)
        train_loader = DataLoader(
            stage2_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collator, num_workers=2, pin_memory=True
        )
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=args.lr * 0.5)  # Lower LR for fine-tuning
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * args.warmup_ratio), total_steps
        )
        
        trainer = Stage3_Decoder_Trainer(
            model, optimizer, use_amp=use_amp, device=args.device
        )
        
        train_stage(
            "Stage2_Instruction", model, trainer, train_loader,
            optimizer, scheduler, args.epochs, args.device,
            logger, checkpoint_manager,
            tokenizer=tokenizer, dataset=stage2_dataset, model_config=model_config,
            debug=args.debug
        )
    
    # ========================================================================
    # Stage 3: Joint Fine-tuning (Optional)
    # ========================================================================
    if args.stage in ["3", "all"] and has_stage1 and has_stage2:
        print("\n" + "=" * 60)
        print("Stage 3: Joint Fine-tuning")
        print("=" * 60)
        print("Training: All components with lower learning rate")
        
        # Unfreeze everything
        for p in model.parameters():
            p.requires_grad = True
        
        # Use stage 2 data for joint training (has both entities and responses)
        collator = create_collator(tokenizer, stage2_dataset, model_config, include_responses=True)
        train_loader = DataLoader(
            stage2_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collator, num_workers=2, pin_memory=True
        )
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=args.lr * 0.1)  # Very low LR
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * args.warmup_ratio), total_steps
        )
        
        trainer = Stage4_Joint_Trainer(
            model, optimizer, use_amp=use_amp, device=args.device
        )
        
        train_stage(
            "Stage3_Joint", model, trainer, train_loader,
            optimizer, scheduler, args.epochs, args.device,
            logger, checkpoint_manager,
            tokenizer=tokenizer, dataset=stage2_dataset, model_config=model_config,
            debug=args.debug
        )
    
    # Save final model
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    final_path = Path(args.output_dir) / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "model_name": model_config.model_name,
            "n_entity_types": n_entity_types,
            "n_relations": n_relations,
            "n_concepts": n_concepts,
        }
    }, final_path)
    print(f"Saved final model to {final_path}")
    
    logger.close()


if __name__ == "__main__":
    main()
