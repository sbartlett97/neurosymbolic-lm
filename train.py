#!/usr/bin/env python3
"""
Main training script for the Neurosymbolic Language Model.

This script handles:
- Model initialization with configurable components
- Dataset loading and preprocessing  
- Tiered training across 4 stages
- Learning rate scheduling
- Model checkpointing
- Early stopping
- TensorBoard logging
- Generation evaluation with metrics

Usage:
    python train.py
    
Configure settings in the main() function or via command line arguments.
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from transformers import AutoTokenizer
from typing import Optional, Dict, List
from pathlib import Path
import json
from datetime import datetime

from data import ToyCognitiveDataset, CognitiveCollator
from model import NeuroSymbolicLM
from training import (
    Stage1_MLM_Trainer,
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
    load_kg_data_for_training,
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
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
        
        Returns:
            True if training should stop
        """
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
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Only save when metric improves
        """
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
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state to save
            epoch: Current epoch
            stage: Current training stage
            score: Current metric score
            config: Optional configuration to save
        
        Returns:
            Path to saved checkpoint or None if not saved
        """
        # Check if we should save
        if self.save_best_only:
            if self.best_score is not None and score >= self.best_score:
                return None
            self.best_score = score
        
        # Create checkpoint
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
        
        # Remove old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        return filepath
    
    def load_latest(self, model, optimizer=None) -> Optional[Dict]:
        """Load the latest checkpoint."""
        if not self.checkpoints:
            # Try to find existing checkpoints
            existing = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
            if existing:
                self.checkpoints = existing
        
        if not self.checkpoints:
            return None
        
        latest = self.checkpoints[-1]
        return self.load(latest, model, optimizer)
    
    def load(self, filepath: Path, model, optimizer=None) -> Dict:
        """Load a specific checkpoint."""
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
        """Log a scalar value."""
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars."""
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram."""
        if self.enabled and self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """Close the writer."""
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
        T_0=max(1, num_epochs - warmup_epochs),
        T_mult=1
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
    skip_stage1_if_pretrained: bool = True,
    soft_logic_weight: float = 0.1,
    soft_logic_rules: Optional[Dict] = None,
    kg_embedding_path: Optional[str] = None,
    kg_triples_path: Optional[str] = None,
    kg_entity_mapping: Optional[Dict[str, str]] = None,
    kg_type: str = "conceptnet",
    dataset_file_path: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    concept_to_entity_type_map: Optional[Dict[str, int]] = None,
    # New parameters
    use_amp: bool = False,
    grad_clip_norm: float = 1.0,
    use_scheduler: bool = True,
    checkpoint_dir: Optional[str] = "checkpoints",
    enable_tensorboard: bool = True,
    early_stopping_patience: int = 5,
    eval_every_n_epochs: int = 5
):
    """
    Run tiered training across 4 stages with advanced features.
    
    Args:
        tokenizer: HuggingFace tokenizer
        model: NeuroSymbolicLM model
        device: torch device
        epochs_per_stage: number of epochs per stage
        skip_stage1_if_pretrained: Skip Stage 1 when using pre-trained encoder
        soft_logic_weight: Weight for soft logic constraint loss
        soft_logic_rules: Dict with concept_to_entity_type_map and rules
        kg_embedding_path: Path to KG embeddings file
        kg_triples_path: Path to KG triples file
        kg_entity_mapping: Mapping from text entities to KG entities
        kg_type: Type of knowledge graph
        dataset_file_path: Path to JSONL dataset file
        batch_size: Training batch size
        num_workers: DataLoader workers
        concept_to_entity_type_map: Mapping from concepts to entity types
        use_amp: Enable mixed precision training
        grad_clip_norm: Gradient clipping norm
        use_scheduler: Use learning rate scheduler
        checkpoint_dir: Directory for checkpoints (None to disable)
        enable_tensorboard: Enable TensorBoard logging
        early_stopping_patience: Patience for early stopping (0 to disable)
        eval_every_n_epochs: Evaluate every N epochs
    """
    model = model.to(device)
    model.train()
    
    # Initialize logging and checkpointing
    logger = TensorBoardLogger(enabled=enable_tensorboard)
    checkpoint_manager = CheckpointManager(checkpoint_dir) if checkpoint_dir else None
    global_step = 0
    
    # Load KG data if enabled
    if model.use_kg and kg_embedding_path is not None:
        print("\n" + "=" * 60)
        print("Loading Knowledge Graph Data")
        print("=" * 60)
        kg_loader, entity_linker, kg_graph = load_kg_data_for_training(
            kg_embedding_path=kg_embedding_path,
            kg_triples_path=kg_triples_path,
            entity_mapping=kg_entity_mapping,
            kg_type=kg_type
        )
        
        if kg_loader is not None:
            if kg_entity_mapping is None:
                ds_temp = ToyCognitiveDataset(jsonl_file_path=dataset_file_path)
                all_entities = []
                for sample in ds_temp:
                    all_entities.extend(sample.get("entities", []))
                unique_entities = list(set(all_entities))
                
                try:
                    from kg_utils import create_entity_mapping_from_dataset
                    kg_entity_mapping = create_entity_mapping_from_dataset(unique_entities, kg_loader)
                    print(f"Auto-generated entity mapping for {len(kg_entity_mapping)} entities")
                except Exception:
                    kg_entity_mapping = {}
            
            model.load_kg_data(
                kg_loader=kg_loader,
                entity_linker=entity_linker,
                kg_graph=kg_graph,
                entity_mapping=kg_entity_mapping
            )
            print("KG data loaded into model successfully")
        else:
            print("Warning: KG integration requested but data loading failed.")
    
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
    
    # Create collators
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
    use_pin_memory = device == "cuda" or (isinstance(device, torch.device) and device.type == 'cuda')
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collator_basic,
                    num_workers=num_workers, pin_memory=use_pin_memory)
    dl_with_responses = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collator_with_responses,
                                   num_workers=num_workers, pin_memory=use_pin_memory)
    
    print(f"DataLoader configured: batch_size={batch_size}, num_workers={num_workers}, dataset_size={len(ds)}")
    
    def to_device(batch):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Stage 1: MLM Pretraining
    use_pretrained = model.use_pretrained_encoder
    skip_stage1 = skip_stage1_if_pretrained and use_pretrained
    
    if not skip_stage1:
        print("=" * 60)
        print("Stage 1: MLM Pretraining")
        print("=" * 60)
        if use_pretrained:
            print("  Skipping Stage 1: Using pre-trained encoder")
        else:
            optimizer1 = Adam(model.encoder.parameters(), lr=1e-4)
            scheduler1 = create_scheduler(optimizer1, epochs_per_stage) if use_scheduler else None
            s1 = Stage1_MLM_Trainer(
                model, tokenizer, optimizer1,
                grad_clip_norm=grad_clip_norm, use_amp=use_amp, device=device
            )
            
            for epoch in range(epochs_per_stage):
                total_loss = 0.0
                for batch in dl:
                    batch = to_device(batch)
                    loss = s1.train_step(batch)
                    total_loss += loss
                    global_step += 1
                
                avg_loss = total_loss / len(dl)
                print(f"  Epoch {epoch+1}/{epochs_per_stage}, Avg Loss: {avg_loss:.4f}")
                logger.log_scalar("Stage1/loss", avg_loss, epoch)
                
                if scheduler1:
                    scheduler1.step()
                    logger.log_scalar("Stage1/lr", optimizer1.param_groups[0]["lr"], epoch)
    else:
        print("=" * 60)
        print("Stage 1: MLM Pretraining (SKIPPED - Using pre-trained encoder)")
        print("=" * 60)
    
    # Stage 2: Entity & Concept Extraction
    print("\n" + "=" * 60)
    print("Stage 2: Entity & Concept Extraction")
    print("=" * 60)
    
    if use_pretrained:
        encoder_frozen = not any(p.requires_grad for p in model.encoder.parameters())
        if encoder_frozen:
            stage2_params = (list(model.token_ent.parameters()) +
                            list(model.concept_head.parameters()) +
                            list(model.gnn.parameters()) +
                            list(model.rel_scorer.parameters()) +
                            list(model.node_proj.parameters()))
        else:
            stage2_params = (list(model.encoder.parameters()) +
                            list(model.token_ent.parameters()) +
                            list(model.concept_head.parameters()) +
                            list(model.gnn.parameters()) +
                            list(model.rel_scorer.parameters()) +
                            list(model.node_proj.parameters()))
    else:
        stage2_params = (list(model.encoder.parameters()) +
                        list(model.token_ent.parameters()) +
                        list(model.concept_head.parameters()) +
                        list(model.gnn.parameters()) +
                        list(model.rel_scorer.parameters()) +
                        list(model.node_proj.parameters()))
    
    optimizer2 = Adam(stage2_params, lr=1e-4)
    scheduler2 = create_scheduler(optimizer2, epochs_per_stage) if use_scheduler else None
    s2 = Stage2_Symbolic_Trainer(
        model, optimizer2, soft_logic_weight=soft_logic_weight,
        grad_clip_norm=grad_clip_norm, use_amp=use_amp, device=device
    )
    
    for epoch in range(epochs_per_stage):
        total_loss = 0.0
        for batch in dl:
            batch = to_device(batch)
            loss = s2.train_step(batch)
            total_loss += loss
            global_step += 1
        
        avg_loss = total_loss / len(dl)
        print(f"  Epoch {epoch+1}/{epochs_per_stage}, Avg Loss: {avg_loss:.4f}")
        logger.log_scalar("Stage2/loss", avg_loss, epoch)
        
        if scheduler2:
            scheduler2.step()
            logger.log_scalar("Stage2/lr", optimizer2.param_groups[0]["lr"], epoch)
    
    # Stage 3: Response Controller
    print("\n" + "=" * 60)
    print("Stage 3: Response Controller")
    print("=" * 60)
    
    optimizer3 = Adam(model.controller.parameters(), lr=1e-4)
    scheduler3 = create_scheduler(optimizer3, epochs_per_stage) if use_scheduler else None
    s3 = Stage3_Control_Trainer(
        model, optimizer3,
        grad_clip_norm=grad_clip_norm, use_amp=use_amp, device=device
    )
    
    for epoch in range(epochs_per_stage):
        total_loss = 0.0
        for batch in dl:
            batch = to_device(batch)
            loss = s3.train_step(batch)
            total_loss += loss
            global_step += 1
        
        avg_loss = total_loss / len(dl)
        print(f"  Epoch {epoch+1}/{epochs_per_stage}, Avg Loss: {avg_loss:.4f}")
        logger.log_scalar("Stage3/loss", avg_loss, epoch)
        
        if scheduler3:
            scheduler3.step()
    
    # Stage 3.5: Decoder Response Generation
    print("\n" + "=" * 60)
    print("Stage 3.5: Decoder Response Generation")
    print("=" * 60)
    
    optimizer3_5 = Adam(model.decoder.parameters(), lr=1e-4)
    scheduler3_5 = create_scheduler(optimizer3_5, epochs_per_stage) if use_scheduler else None
    s3_5 = Stage3_Decoder_Trainer(
        model, optimizer3_5,
        grad_clip_norm=grad_clip_norm, use_amp=use_amp, device=device
    )
    
    for epoch in range(epochs_per_stage):
        total_loss = 0.0
        for batch in dl_with_responses:
            batch = to_device(batch)
            loss = s3_5.train_step(batch)
            total_loss += loss
            global_step += 1
        
        avg_loss = total_loss / len(dl_with_responses)
        print(f"  Epoch {epoch+1}/{epochs_per_stage}, Avg Loss: {avg_loss:.4f}")
        logger.log_scalar("Stage3.5/loss", avg_loss, epoch)
        
        if scheduler3_5:
            scheduler3_5.step()
    
    if epochs_per_stage > 0:
        print("\n  Evaluating generation after Stage 3.5...")
        gen_results = evaluate_generation(model, tokenizer, ds, device=device, num_samples=5)
        print_generation_results(gen_results, num_to_print=3)
        
        metrics = compute_aggregate_metrics(gen_results)
        for name, value in metrics.items():
            logger.log_scalar(f"Stage3.5/{name}", value, 0)
    
    # Stage 4: Joint End-to-End Training
    print("\n" + "=" * 60)
    print("Stage 4: Joint End-to-End Training")
    print("=" * 60)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer4 = Adam(trainable_params, lr=1e-5)
    scheduler4 = create_scheduler(optimizer4, epochs_per_stage) if use_scheduler else None
    s4 = Stage4_Joint_Trainer(
        model, optimizer4, soft_logic_weight=soft_logic_weight,
        grad_clip_norm=grad_clip_norm, use_amp=use_amp, device=device
    )
    
    early_stopper = EarlyStopping(patience=early_stopping_patience) if early_stopping_patience > 0 else None
    best_loss = float("inf")
    
    for epoch in range(epochs_per_stage):
        total_loss = 0.0
        for batch in dl_with_responses:
            batch = to_device(batch)
            loss = s4.train_step(batch)
            total_loss += loss
            global_step += 1
        
        avg_loss = total_loss / len(dl_with_responses)
        print(f"  Epoch {epoch+1}/{epochs_per_stage}, Avg Loss: {avg_loss:.4f}")
        logger.log_scalar("Stage4/loss", avg_loss, epoch)
        
        if scheduler4:
            scheduler4.step()
            logger.log_scalar("Stage4/lr", optimizer4.param_groups[0]["lr"], epoch)
        
        # Checkpointing
        if checkpoint_manager and avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_manager.save(model, optimizer4, epoch, "stage4", avg_loss)
            print(f"    Saved checkpoint (loss: {avg_loss:.4f})")
        
        # Early stopping check
        if early_stopper and early_stopper(avg_loss):
            print(f"  Early stopping triggered at epoch {epoch+1}")
            break
        
        # Periodic evaluation
        if (epoch + 1) % eval_every_n_epochs == 0 or (epoch + 1) == epochs_per_stage:
            print(f"\n  Evaluating generation after epoch {epoch+1}...")
            gen_results = evaluate_generation(model, tokenizer, ds, device=device, num_samples=5)
            print_generation_results(gen_results, num_to_print=3)
            
            metrics = compute_aggregate_metrics(gen_results)
            for name, value in metrics.items():
                logger.log_scalar(f"Stage4/{name}", value, epoch)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Generation Evaluation")
    print("=" * 60)
    final_results = evaluate_generation(model, tokenizer, ds, device=device, num_samples=10)
    print_generation_results(final_results, num_to_print=5)
    
    # Log final metrics
    final_metrics = compute_aggregate_metrics(final_results)
    for name, value in final_metrics.items():
        logger.log_scalar(f"Final/{name}", value, 0)
    
    logger.close()
    
    return model


def main():
    """Main entry point for training."""
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    vocab_size = len(tokenizer)
    
    # ========================================================================
    # Knowledge Graph Configuration
    # ========================================================================
    USE_KG = False  # Disabled by default, enable with KG embeddings
    KG_TYPE = "conceptnet"
    KG_EMBEDDING_PATH = None  # Set to path for KG embeddings
    KG_TRIPLES_PATH = None    # Set to path for KG triples
    USE_KG_GNN = False
    USE_PATH_REASONING = False
    MAX_PATH_LENGTH = 3
    
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
    
    # Dynamic n_concepts based on dataset (with buffer for generalization)
    n_concepts = max(len(all_concepts) * 2, 64)  # At least 64, or 2x unique concepts
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
    # Model Configuration
    # ========================================================================
    model = NeuroSymbolicLM(
        vocab_size=vocab_size,
        d_model=768,
        n_entity_types=n_entity_types,
        n_relations=n_relations,
        n_concepts=n_concepts,  # Now dynamic
        concept_dim=256,
        node_dim=256,
        max_nodes=16,
        use_pretrained_encoder=True,
        pretrained_model_name="answerdotai/ModernBERT-base",
        freeze_encoder=True,
        use_pretrained_decoder=True,
        pretrained_decoder_name="t5-small",
        freeze_decoder=False,
        use_kg=USE_KG,
        kg_embed_dim=300,
        use_kg_gnn=USE_KG_GNN,
        use_path_reasoning=USE_PATH_REASONING,
        max_path_length=MAX_PATH_LENGTH
    )
    
    print(f"Extracted concept-to-entity-type mapping: {concept_to_entity_type_map}")
    
    # ========================================================================
    # Soft Logic Rules Configuration
    # ========================================================================
    USE_DYNAMIC_RULES = True
    
    if USE_DYNAMIC_RULES:
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
    else:
        # Manual rules
        soft_logic_rules = {
            "concept_to_entity_type_map": concept_to_entity_type_map,
            "rules": [
                {"concept_a": "animal", "concept_b": "animal", "relation": "chases", "weight": 1.0, "polarity": 1},
                {"concept_a": "location", "concept_b": "location", "relation": "capital_of", "weight": 1.0, "polarity": 1},
                {"concept_a": "animal", "concept_b": "location", "relation": "lives_in", "weight": 1.0, "polarity": 1},
            ]
        }
    
    print(f"\nModel initialized with pre-trained encoder")
    print(f"Encoder hidden size: {model.d_model}")
    print(f"Encoder frozen: {not any(p.requires_grad for p in model.encoder.parameters())}")
    if model.use_pretrained_decoder:
        print(f"Using pre-trained decoder: {model.decoder.pretrained_model.config.name_or_path}")
        print(f"Decoder frozen: {model.decoder.freeze_decoder}")
    
    if model.use_kg:
        print(f"\nKG Integration: ENABLED")
        print(f"  - KG-aware GNN: {model.use_kg_gnn}")
        print(f"  - Path reasoning: {model.use_path_reasoning}")
    else:
        print(f"\nKG Integration: DISABLED")
    
    # ========================================================================
    # Training Configuration
    # ========================================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    EPOCHS_PER_STAGE = 10
    USE_AMP = device == "cuda"  # Enable AMP only on GPU
    
    # Run training
    trained_model = run_training(
        tokenizer,
        model,
        device=device,
        epochs_per_stage=EPOCHS_PER_STAGE,
        skip_stage1_if_pretrained=True,
        soft_logic_weight=0.1,
        soft_logic_rules=soft_logic_rules,
        kg_embedding_path=KG_EMBEDDING_PATH,
        kg_triples_path=KG_TRIPLES_PATH,
        kg_entity_mapping=None,
        kg_type=KG_TYPE,
        dataset_file_path=dataset_file_path,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        concept_to_entity_type_map=concept_to_entity_type_map,
        # New features
        use_amp=USE_AMP,
        grad_clip_norm=1.0,
        use_scheduler=True,
        checkpoint_dir="checkpoints",
        enable_tensorboard=True,
        early_stopping_patience=5,
        eval_every_n_epochs=5
    )
    
    # Example generation
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
