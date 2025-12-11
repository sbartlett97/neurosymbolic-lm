#!/usr/bin/env python3
"""
Main training script for the Neurosymbolic Language Model.

This script handles:
- Model initialization with T5 backbone
- Dataset loading and preprocessing  
- Tiered training across multiple stages
- Learning rate scheduling
- Model checkpointing
- Early stopping
- TensorBoard logging
- Generation evaluation with metrics

Usage:
    python train.py
    
Configure settings in the main() function.
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from transformers import AutoTokenizer
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime

from data import ToyCognitiveDataset, CognitiveCollator
from model import NeuroSymbolicLM
from training import (
    Stage2_Symbolic_Trainer,
    Stage3_Control_Trainer,
    Stage3_Decoder_Trainer,
    Stage4_Joint_Trainer,
    extract_concept_to_entity_type_map,
    extract_relation_map_from_dataset,
    generate_soft_logic_rules_from_dataset,
    configure_soft_logic_rules,
    evaluate_generation,
    print_generation_results,
    generate_response,
    compute_aggregate_metrics,
)

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class CheckpointManager:
    """Manage model checkpoints during training."""
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 3,
        save_best_only: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.best_score = None
        self.checkpoints: List[Path] = []
    
    def save(
        self,
        model,
        optimizer,
        epoch: int,
        stage: str,
        score: float,
        config: Optional[Dict] = None
    ) -> Optional[Path]:
        if self.save_best_only:
            if self.best_score is not None and score >= self.best_score:
                return None
            self.best_score = score
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{stage}_epoch{epoch}_{timestamp}.pt"
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {
            "epoch": epoch,
            "stage": stage,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "score": score,
            "config": config
        }
        
        torch.save(checkpoint, filepath)
        self.checkpoints.append(filepath)
        
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        return filepath
    
    def load_latest(self, model, optimizer=None) -> Optional[Dict]:
        if not self.checkpoints:
            existing = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
            if existing:
                self.checkpoints = existing
        
        if not self.checkpoints:
            return None
        
        latest = self.checkpoints[-1]
        return self.load(latest, model, optimizer)
    
    def load(self, filepath: Path, model, optimizer=None) -> Dict:
        checkpoint = torch.load(filepath, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint


class TensorBoardLogger:
    """TensorBoard logging wrapper."""
    
    def __init__(self, log_dir: str = "runs", enabled: bool = True):
        self.enabled = enabled and TENSORBOARD_AVAILABLE
        self.writer = None
        
        if self.enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(log_dir=f"{log_dir}/{timestamp}")
    
    def log_scalar(self, tag: str, value: float, step: int):
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def close(self):
        if self.writer:
            self.writer.close()


def create_scheduler(optimizer, num_epochs: int, warmup_epochs: int = 2):
    """Create a learning rate scheduler with warmup."""
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    main_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=num_epochs - warmup_epochs,
        T_mult=1,
        eta_min=1e-6
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )
    
    return scheduler


def run_training(
    tokenizer,
    model,
    device: str = "cpu",
    epochs_per_stage: int = 10,
    soft_logic_weight: float = 0.1,
    soft_logic_rules: Optional[Dict] = None,
    dataset_file_path: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    concept_to_entity_type_map: Optional[Dict[str, int]] = None,
    use_amp: bool = False,
    grad_clip_norm: float = 1.0,
    use_scheduler: bool = True,
    checkpoint_dir: Optional[str] = "checkpoints",
    enable_tensorboard: bool = True,
    early_stopping_patience: int = 5,
    eval_every_n_epochs: int = 5
):
    """
    Run tiered training.
    
    With T5 backbone, we skip Stage 1 (MLM) since T5 is already pretrained.
    Training stages:
    - Stage 2: Entity/Concept/Relation training
    - Stage 3: Response controller training  
    - Stage 3.5: Decoder training
    - Stage 4: Joint end-to-end training
    """
    model = model.to(device)
    model.train()
    
    # Initialize logging and checkpointing
    logger = TensorBoardLogger(enabled=enable_tensorboard)
    checkpoint_manager = CheckpointManager(checkpoint_dir) if checkpoint_dir else None
    global_step = 0
    
    # Load dataset
    if dataset_file_path:
        print(f"\nLoading dataset from: {dataset_file_path}")
    else:
        print("\nUsing hardcoded toy dataset")
    ds = ToyCognitiveDataset(jsonl_file_path=dataset_file_path)
    
    # Extract mappings
    all_concepts = set()
    for sample in ds:
        concepts = sample.get("concepts", [])
        for concept_list in concepts:
            if isinstance(concept_list, str):
                concept_list = [concept_list]
            elif not isinstance(concept_list, list):
                concept_list = [str(concept_list)]
            for concept in concept_list:
                all_concepts.add(concept)
    
    concept_map = {concept: idx + 1 for idx, concept in enumerate(sorted(all_concepts))}
    print(f"Extracted concept map: {len(concept_map)} concepts")
    
    relation_map = extract_relation_map_from_dataset(ds)
    print(f"Extracted relation map: {len(relation_map)} relations")
    
    if concept_to_entity_type_map is None:
        concept_to_entity_type_map = extract_concept_to_entity_type_map(
            ds,
            n_entity_types=model.n_entity_types,
            base_concept_priority=["animal", "person", "location", "object", "attribute"]
        )
        print(f"Extracted concept-to-entity-type mapping: {concept_to_entity_type_map}")
    
    # Configure soft logic rules
    if soft_logic_rules is not None:
        configure_soft_logic_rules(
            model,
            soft_logic_rules.get("concept_to_entity_type_map", {}),
            relation_map,
            soft_logic_rules.get("rules", [])
        )
    
    # Create collators (single tokenizer for everything)
    collator_basic = CognitiveCollator(
        tokenizer, concept_map, relation_map,
        include_responses=False,
        concept_to_entity_type_map=concept_to_entity_type_map or {}
    )
    collator_with_responses = CognitiveCollator(
        tokenizer, concept_map, relation_map,
        include_responses=True,
        concept_to_entity_type_map=concept_to_entity_type_map or {}
    )
    
    # Create DataLoaders
    use_pin_memory = device == "cuda"
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True, collate_fn=collator_basic,
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    dl_with_responses = DataLoader(
        ds, batch_size=batch_size, shuffle=True, collate_fn=collator_with_responses,
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    
    print(f"DataLoader configured: batch_size={batch_size}, dataset_size={len(ds)}")
    
    def to_device(batch):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # ========================================================================
    # Stage 2: Entity/Concept/Relation Training
    # ========================================================================
    print("\n" + "=" * 60)
    print("Stage 2: Entity/Concept/Relation Training")
    print("=" * 60)
    
    # Only train non-frozen parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer2 = Adam(trainable_params, lr=1e-4)
    scheduler2 = create_scheduler(optimizer2, epochs_per_stage) if use_scheduler else None
    early_stop2 = EarlyStopping(patience=early_stopping_patience) if early_stopping_patience > 0 else None
    
    s2 = Stage2_Symbolic_Trainer(
        model, optimizer2,
        grad_clip_norm=grad_clip_norm, use_amp=use_amp, device=device
    )
    
    for epoch in range(epochs_per_stage):
        epoch_losses = []
        for batch in dl:
            batch = to_device(batch)
            loss = s2.train_step(batch)
            epoch_losses.append(loss)
            global_step += 1
            logger.log_scalar("Train/Stage2_Loss", loss, global_step)
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Stage 2 Epoch {epoch+1}/{epochs_per_stage}, Avg Loss: {avg_loss:.4f}")
        
        if scheduler2:
            scheduler2.step()
        
        # Checkpoint
        if checkpoint_manager and (epoch + 1) % eval_every_n_epochs == 0:
            checkpoint_manager.save(model, optimizer2, epoch + 1, "stage2", avg_loss)
        
        # Early stopping
        if early_stop2 and early_stop2(avg_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # ========================================================================
    # Stage 3: Response Controller Training
    # ========================================================================
    print("\n" + "=" * 60)
    print("Stage 3: Response Controller Training")
    print("=" * 60)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer3 = Adam(trainable_params, lr=5e-5)
    scheduler3 = create_scheduler(optimizer3, epochs_per_stage) if use_scheduler else None
    early_stop3 = EarlyStopping(patience=early_stopping_patience) if early_stopping_patience > 0 else None
    
    s3 = Stage3_Control_Trainer(
        model, optimizer3,
        grad_clip_norm=grad_clip_norm, use_amp=use_amp, device=device
    )
    
    for epoch in range(epochs_per_stage):
        epoch_losses = []
        for batch in dl_with_responses:
            batch = to_device(batch)
            loss = s3.train_step(batch)
            epoch_losses.append(loss)
            global_step += 1
            logger.log_scalar("Train/Stage3_Loss", loss, global_step)
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Stage 3 Epoch {epoch+1}/{epochs_per_stage}, Avg Loss: {avg_loss:.4f}")
        
        if scheduler3:
            scheduler3.step()
        
        if checkpoint_manager and (epoch + 1) % eval_every_n_epochs == 0:
            checkpoint_manager.save(model, optimizer3, epoch + 1, "stage3", avg_loss)
        
        if early_stop3 and early_stop3(avg_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # ========================================================================
    # Stage 3.5: Decoder Response Generation
    # ========================================================================
    print("\n" + "=" * 60)
    print("Stage 3.5: Decoder Response Generation")
    print("=" * 60)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer3_5 = Adam(trainable_params, lr=5e-5)
    scheduler3_5 = create_scheduler(optimizer3_5, epochs_per_stage) if use_scheduler else None
    early_stop3_5 = EarlyStopping(patience=early_stopping_patience) if early_stopping_patience > 0 else None
    
    s3_5 = Stage3_Decoder_Trainer(
        model, optimizer3_5,
        grad_clip_norm=grad_clip_norm, use_amp=use_amp, device=device
    )
    
    for epoch in range(epochs_per_stage):
        epoch_losses = []
        for batch in dl_with_responses:
            batch = to_device(batch)
            loss = s3_5.train_step(batch)
            if loss > 0:  # Skip batches with no valid labels
                epoch_losses.append(loss)
            global_step += 1
            logger.log_scalar("Train/Stage3.5_Loss", loss, global_step)
        
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        print(f"Stage 3.5 Epoch {epoch+1}/{epochs_per_stage}, Avg Loss: {avg_loss:.4f}")
        
        if scheduler3_5:
            scheduler3_5.step()
        
        # Evaluate generation periodically
        if (epoch + 1) % eval_every_n_epochs == 0:
            print(f"\nGeneration evaluation at epoch {epoch + 1}:")
            try:
                results = evaluate_generation(
                    model, tokenizer, ds, device=device, num_samples=3
                )
                print_generation_results(results, num_to_print=2)
            except Exception as e:
                print(f"Generation evaluation failed: {e}")
            
            if checkpoint_manager:
                checkpoint_manager.save(model, optimizer3_5, epoch + 1, "stage3.5", avg_loss)
        
        if early_stop3_5 and early_stop3_5(avg_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # ========================================================================
    # Stage 4: Joint End-to-End Training
    # ========================================================================
    print("\n" + "=" * 60)
    print("Stage 4: Joint End-to-End Training")
    print("=" * 60)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer4 = Adam(trainable_params, lr=2e-5)
    scheduler4 = create_scheduler(optimizer4, epochs_per_stage) if use_scheduler else None
    early_stop4 = EarlyStopping(patience=early_stopping_patience) if early_stopping_patience > 0 else None
    
    s4 = Stage4_Joint_Trainer(
        model, optimizer4,
        soft_logic_weight=soft_logic_weight,
        grad_clip_norm=grad_clip_norm, use_amp=use_amp, device=device
    )
    
    for epoch in range(epochs_per_stage):
        epoch_losses = []
        for batch in dl_with_responses:
            batch = to_device(batch)
            loss = s4.train_step(batch)
            epoch_losses.append(loss)
            global_step += 1
            logger.log_scalar("Train/Stage4_Loss", loss, global_step)
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Stage 4 Epoch {epoch+1}/{epochs_per_stage}, Avg Loss: {avg_loss:.4f}")
        
        if scheduler4:
            scheduler4.step()
        
        # Evaluate generation periodically
        if (epoch + 1) % eval_every_n_epochs == 0:
            print(f"\nGeneration evaluation at epoch {epoch + 1}:")
            try:
                results = evaluate_generation(
                    model, tokenizer, ds, device=device, num_samples=3
                )
                print_generation_results(results, num_to_print=2)
                
                metrics = compute_aggregate_metrics(results)
                for name, value in metrics.items():
                    logger.log_scalar(f"Eval/{name}", value, global_step)
            except Exception as e:
                print(f"Generation evaluation failed: {e}")
            
            if checkpoint_manager:
                checkpoint_manager.save(model, optimizer4, epoch + 1, "stage4", avg_loss)
        
        if early_stop4 and early_stop4(avg_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Generation Evaluation")
    print("=" * 60)
    try:
        final_results = evaluate_generation(
            model, tokenizer, ds, device=device, num_samples=10
        )
        print_generation_results(final_results, num_to_print=5)
        
        final_metrics = compute_aggregate_metrics(final_results)
        for name, value in final_metrics.items():
            logger.log_scalar(f"Final/{name}", value, 0)
    except Exception as e:
        print(f"Final evaluation failed: {e}")
    
    logger.close()
    
    return model


def main():
    """Main entry point for training."""
    # ========================================================================
    # Model Configuration - Using T5 as unified backbone
    # ========================================================================
    MODEL_NAME = "t5-small"  # Options: t5-small, t5-base, google/flan-t5-small, etc.
    FREEZE_ENCODER = False  # Whether to freeze T5 encoder
    FREEZE_DECODER = False  # Whether to freeze T5 decoder
    
    print(f"\nInitializing NeuroSymbolicLM with {MODEL_NAME} backbone")
    
    # Initialize tokenizer (T5 tokenizer for everything)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # ========================================================================
    # Dataset Configuration
    # ========================================================================
    dataset_file_path = "comprehensive_dataset.jsonl"
    
    # Load dataset to infer model parameters
    print(f"\nLoading dataset to infer model parameters from: {dataset_file_path}")
    ds_temp = ToyCognitiveDataset(jsonl_file_path=dataset_file_path)
    
    # Extract all concepts for dynamic n_concepts
    all_concepts = set()
    for sample in ds_temp:
        concepts = sample.get("concepts", [])
        for concept_list in concepts:
            if isinstance(concept_list, str):
                concept_list = [concept_list]
            elif not isinstance(concept_list, list):
                concept_list = [str(concept_list)]
            for concept in concept_list:
                all_concepts.add(concept)
    
    n_concepts = max(len(all_concepts) * 2, 64)
    print(f"Dynamic n_concepts: {n_concepts} (based on {len(all_concepts)} unique concepts)")
    
    # Extract mappings
    concept_to_entity_type_map = extract_concept_to_entity_type_map(
        ds_temp,
        n_entity_types=None,
        base_concept_priority=["animal", "person", "location", "object", "attribute"]
    )
    
    n_entity_types = len(concept_to_entity_type_map)
    if n_entity_types == 0:
        print("Warning: No entity types found. Using default n_entity_types=8")
        n_entity_types = 8
    else:
        print(f"Inferred n_entity_types={n_entity_types} from dataset")
    
    relation_map = extract_relation_map_from_dataset(ds_temp)
    n_relations = len(relation_map)
    if n_relations == 0:
        print("Warning: No relations found. Using default n_relations=32")
        n_relations = 32
    else:
        print(f"Inferred n_relations={n_relations} from dataset")
    
    # ========================================================================
    # Create Model
    # ========================================================================
    model = NeuroSymbolicLM(
        model_name=MODEL_NAME,
        n_entity_types=n_entity_types,
        n_relations=n_relations,
        n_concepts=n_concepts,
        concept_dim=256,
        node_dim=256,
        max_nodes=16,
        freeze_encoder=FREEZE_ENCODER,
        freeze_decoder=FREEZE_DECODER,
    )
    
    print(f"\nModel initialized:")
    print(f"  - Backbone: {MODEL_NAME}")
    print(f"  - Hidden size: {model.d_model}")
    print(f"  - Vocab size: {model.vocab_size}")
    print(f"  - Entity types: {n_entity_types}")
    print(f"  - Relations: {n_relations}")
    print(f"  - Concepts: {n_concepts}")
    
    # ========================================================================
    # Soft Logic Rules Configuration
    # ========================================================================
    print("\nGenerating soft logic rules dynamically from dataset patterns...")
    generated_rules = generate_soft_logic_rules_from_dataset(
        ds_temp,
        concept_to_entity_type_map,
        relation_map,
        min_frequency=2,
        min_confidence=0.2,
        max_rules=50,
        include_negative_rules=True
    )
    print(f"Generated {len(generated_rules)} soft logic rules from dataset")
    
    if len(generated_rules) > 0:
        print("Sample generated rules:")
        for rule in generated_rules[:5]:
            polarity_str = "ENCOURAGE" if rule["polarity"] == 1 else "DISCOURAGE"
            print(f"  {polarity_str}: {rule['concept_a']} + {rule['concept_b']} -> {rule['relation']} (weight={rule['weight']:.2f})")
    
    soft_logic_rules = {
        "concept_to_entity_type_map": concept_to_entity_type_map,
        "rules": generated_rules
    }
    
    # ========================================================================
    # Training Configuration
    # ========================================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    EPOCHS_PER_STAGE = 10
    USE_AMP = device == "cuda"
    
    # Run training
    trained_model = run_training(
        tokenizer,
        model,
        device=device,
        epochs_per_stage=EPOCHS_PER_STAGE,
        soft_logic_weight=0.1,
        soft_logic_rules=soft_logic_rules,
        dataset_file_path=dataset_file_path,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        concept_to_entity_type_map=concept_to_entity_type_map,
        use_amp=USE_AMP,
        grad_clip_norm=1.0,
        use_scheduler=True,
        checkpoint_dir="checkpoints",
        enable_tensorboard=True,
        early_stopping_patience=5,
        eval_every_n_epochs=5
    )
    
    # ========================================================================
    # Example Generation
    # ========================================================================
    print("\n" + "=" * 60)
    print("Example Generation")
    print("=" * 60)
    
    test_inputs = [
        "Is Paris the capital of France?",
        "Does the cat chase the mouse?",
        "Did the teacher teach the students?"
    ]
    
    for test_input in test_inputs:
        generated = generate_response(
            trained_model,
            tokenizer,
            test_input,
            device=device,
            max_length=128,
            do_sample=False
        )
        print(f"\nInput:     {test_input}")
        print(f"Generated: {generated}")


if __name__ == "__main__":
    main()
