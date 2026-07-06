#!/usr/bin/env python3
"""Staged training for the decoder-only NeuroSymbolicCausalLM.

Same curriculum as train.py, adapted to the causal architecture
(docs/DECODER_ONLY_MIGRATION.md):

  Stage 1 (symbolic): backbone frozen; trains the extraction adapter and
      symbolic heads on curated corpus data — very cheap.
  Stage 2 (decoder):  heads/adapter frozen; trains the soft-token injector
      and the backbone (full fine-tune, or LoRA with --preset *-lora).
  Stage 3 (joint):    everything, low LR.
  Chat phase:         trace/chat data via CausalCognitiveCollator.

Usage:
    python train_causal.py --preset qwen3-0.6b \
        --dataset data/curated/curated.jsonl --stages symbolic decoder joint

    python train_causal.py --preset qwen3-0.6b --phase chat \
        --chat-dataset data/curated_traces/traces.jsonl
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import CausalModelConfig, CAUSAL_MODEL_PRESETS
from data.chat_template import ChatTemplate
from data.collator_causal import CausalCognitiveCollator
from data.dataset import ToyCognitiveDataset
from model.neurosymbolic_causal import NeuroSymbolicCausalLM
from train import (
    extract_vocab_from_dataset,
    load_dataset,
    load_vocab_file,
    merge_new_labels,
    resolve_device,
    run_evaluation,
    train_stage,
)
from training import (
    CheckpointManager,
    EarlyStopping,
    Stage2_Symbolic_Trainer,
    Stage3_Decoder_Trainer,
    Stage4_Joint_Trainer,
    TrainingLogger,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Staged training for NeuroSymbolicCausalLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--dataset", type=str, default="data/curated/curated.jsonl")
    data_group.add_argument("--eval-dataset", type=str, default="eval_dataset.jsonl")
    data_group.add_argument("--stage2-dataset", type=str, default=None)
    data_group.add_argument("--chat-dataset", type=str, default=None)
    data_group.add_argument("--max-samples", type=int, default=None)

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--preset", type=str, default="qwen3-0.6b",
                             choices=sorted(CAUSAL_MODEL_PRESETS.keys()))
    model_group.add_argument("--model-name", type=str, default=None,
                             help="Override backbone checkpoint")
    model_group.add_argument("--tap-layer-ratio", type=float, default=None)
    model_group.add_argument("--adapter-layers", type=int, default=None,
                             help="0 = pure-causal extraction baseline")

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--phase", type=str, default="pretrain",
                             choices=["pretrain", "chat", "both"])
    train_group.add_argument("--stages", type=str, nargs="+",
                             default=["symbolic", "decoder", "joint"],
                             choices=["symbolic", "decoder", "joint"])
    train_group.add_argument("--epochs", type=int, default=5)
    train_group.add_argument("--chat-epochs", type=int, default=3)
    train_group.add_argument("--batch-size", type=int, default=4)
    train_group.add_argument("--lr", type=float, default=5e-5)
    train_group.add_argument("--chat-lr", type=float, default=1e-5)
    train_group.add_argument("--warmup-ratio", type=float, default=0.1)
    train_group.add_argument("--patience", type=int, default=3)
    train_group.add_argument("--system-prompt", type=str,
                             default="You are a helpful assistant.")

    hw_group = parser.add_argument_group("Hardware")
    hw_group.add_argument("--device", type=str, default="auto",
                          help="auto/cuda/mps/cpu")
    hw_group.add_argument("--no-amp", action="store_true")
    hw_group.add_argument("--num-workers", type=int, default=2)

    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--output-dir", type=str, default="checkpoints_causal")
    out_group.add_argument("--resume", type=str, default=None)
    out_group.add_argument("--log-dir", type=str, default="runs")
    out_group.add_argument("--eval-every", type=int, default=2)
    out_group.add_argument("--save-every", type=int, default=1)
    out_group.add_argument("--debug", action="store_true")

    return parser.parse_args()


def build_model_config(args) -> CausalModelConfig:
    config = CAUSAL_MODEL_PRESETS[args.preset]()
    if args.model_name:
        config.model_name = args.model_name
    if args.tap_layer_ratio is not None:
        config.tap_layer_ratio = args.tap_layer_ratio
    if args.adapter_layers is not None:
        config.adapter_layers = args.adapter_layers
    return config


def build_model(config: CausalModelConfig) -> NeuroSymbolicCausalLM:
    if config.model_name == "__tiny__":
        raise SystemExit(
            "The causal-testing preset builds a random tiny backbone and is "
            "meant for the unit tests, not CLI training. Pick a real preset."
        )
    model = NeuroSymbolicCausalLM(
        model_name=config.model_name,
        tap_layer_ratio=config.tap_layer_ratio,
        adapter_layers=config.adapter_layers,
        adapter_heads=config.adapter_heads,
        adapter_ff_mult=config.adapter_ff_mult,
        n_entity_types=config.n_entity_types,
        n_relations=config.n_relations,
        n_concepts=config.n_concepts,
        concept_dim=config.concept_dim,
        node_dim=config.node_dim,
        max_nodes=config.max_nodes,
        dropout=config.dropout,
        injection_gate_init=config.injection_gate_init,
        gradient_checkpointing=config.gradient_checkpointing,
        max_input_length=config.max_input_length,
        max_output_length=config.max_output_length,
    )
    if config.use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise SystemExit("--preset *-lora requires peft: pip install peft")
        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model.backbone = get_peft_model(model.backbone, lora_cfg)
        print("LoRA adapters attached to backbone")
    return model


def unfreeze_backbone(model: NeuroSymbolicCausalLM, use_lora: bool):
    """Unfreeze the backbone for stages 2/3.

    With LoRA, only adapter weights train; the base weights stay frozen.
    """
    if use_lora:
        for name, p in model.backbone.named_parameters():
            p.requires_grad = "lora_" in name
    else:
        model.freeze_backbone(False)


def create_dataloader(
    dataset,
    tokenizer,
    concept_map: Dict,
    relation_map: Dict,
    entity_type_map: Dict,
    config: CausalModelConfig,
    batch_size: int,
    num_workers: int,
    include_responses: bool,
    chat_mode: bool = False,
    chat_template: Optional[ChatTemplate] = None,
    entity_type_name_map: Optional[Dict] = None,
    pin_memory: bool = False,
) -> DataLoader:
    collator = CausalCognitiveCollator(
        tokenizer=tokenizer,
        concept_map=concept_map,
        relation_map=relation_map,
        include_responses=include_responses,
        concept_to_entity_type_map=entity_type_map,
        max_length=config.max_input_length,
        max_output_length=config.max_output_length,
        chat_mode=chat_mode,
        chat_template=chat_template,
        max_nodes=config.max_nodes,
        entity_type_map=entity_type_name_map,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        drop_last=len(dataset) > batch_size,
    )


def main():
    args = parse_args()
    config = build_model_config(args)

    print("=" * 70)
    print("NeuroSymbolicCausalLM Training (decoder-only)")
    print("=" * 70)
    print(f"Preset: {args.preset} | Backbone: {config.model_name}")

    args.device = resolve_device(args.device)
    print(f"Device: {args.device}")
    use_amp = not args.no_amp and args.device == "cuda"
    pin = args.device == "cuda"

    # --- data ---------------------------------------------------------
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: dataset not found: {dataset_path}")
        sys.exit(1)
    train_dataset = load_dataset(dataset_path, args.max_samples)
    print(f"Loaded {len(train_dataset)} samples")

    stage2_dataset = None
    if args.stage2_dataset and Path(args.stage2_dataset).exists():
        stage2_dataset = load_dataset(Path(args.stage2_dataset), args.max_samples)

    eval_dataset = None
    if Path(args.eval_dataset).exists():
        eval_dataset = load_dataset(Path(args.eval_dataset))

    # --- vocabulary (same rules as train.py) ---------------------------
    vocab_file = load_vocab_file(dataset_path)
    entity_type_name_map: Dict[str, int] = {}
    if vocab_file is not None:
        concept_map = dict(vocab_file["concepts"])
        relation_map = dict(vocab_file["relations"])
        entity_type_name_map = dict(vocab_file.get("entity_types", {}))
        entity_type_map = dict(vocab_file.get("concept_to_entity_type", {}))
        n_entity_types = max(config.n_entity_types,
                             max(entity_type_name_map.values(), default=0) + 1)
        n_relations = max(config.n_relations, len(relation_map) + 10)
        n_concepts = max(config.n_concepts, len(concept_map) + 50)
    else:
        concept_map, relation_map, entity_type_map, n_entity_types, \
            n_relations, n_concepts = extract_vocab_from_dataset(train_dataset)
        n_entity_types = max(n_entity_types, config.n_entity_types)

    if stage2_dataset:
        c2, r2, _, _, _, _ = extract_vocab_from_dataset(stage2_dataset)
        merge_new_labels(concept_map, c2.keys())
        merge_new_labels(relation_map, r2.keys())

    config.n_entity_types = n_entity_types
    config.n_relations = max(n_relations, config.n_relations)
    config.n_concepts = max(n_concepts, config.n_concepts)
    print(f"Vocab: {len(concept_map)} concepts, {len(relation_map)} relations, "
          f"{len(entity_type_name_map) or len(entity_type_map)} entity types")

    # --- model + tokenizer ---------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(config)

    chat_template = ChatTemplate(default_system_prompt=args.system_prompt)
    added = ChatTemplate.add_special_tokens(tokenizer)
    if added > 0:
        model.backbone.resize_token_embeddings(len(tokenizer))
        print(f"Added {added} chat special tokens; embeddings resized")

    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model_state_dict"], strict=False)
        print(f"Resumed from {args.resume}")

    model = model.to(args.device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,} (tap layer {model.tap_layer})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TrainingLogger(log_dir=args.log_dir, experiment_name=f"causal_{timestamp}")
    checkpoint_manager = CheckpointManager(save_dir=args.output_dir, max_checkpoints=5)

    def loader_for(dataset, include_responses, chat=False):
        return create_dataloader(
            dataset, tokenizer, concept_map, relation_map, entity_type_map,
            config, args.batch_size, args.num_workers,
            include_responses=include_responses,
            chat_mode=chat, chat_template=chat_template if chat else None,
            entity_type_name_map=entity_type_name_map, pin_memory=pin,
        )

    def run(stage_name, trainer_cls, dataset, include_responses, lr,
            epochs, chat=False, patience_bonus=0):
        train_loader = loader_for(dataset, include_responses, chat)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=lr)
        total_steps = max(1, len(train_loader) * epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * args.warmup_ratio), total_steps
        )
        trainer = trainer_cls(model, optimizer, use_amp=use_amp, device=args.device)
        train_stage(
            stage_name, model, trainer, train_loader, optimizer, scheduler,
            epochs, args.device, logger, checkpoint_manager,
            EarlyStopping(patience=args.patience + patience_bonus),
            tokenizer=tokenizer, train_dataset=dataset, eval_dataset=eval_dataset,
            model_config=config, eval_every=args.eval_every,
            debug=args.debug, save_every=args.save_every,
        )

    # =====================================================================
    if args.phase in ("pretrain", "both"):
        if "symbolic" in args.stages:
            print("\nStage 1: Symbolic (backbone frozen)")
            model.freeze_backbone(True)
            model.freeze_symbolic(False)
            for p in model.injector.parameters():
                p.requires_grad = False
            run("Symbolic", Stage2_Symbolic_Trainer, train_dataset,
                include_responses=False, lr=args.lr, epochs=args.epochs)

        if "decoder" in args.stages:
            print("\nStage 2: Response (heads frozen, injector + backbone)")
            model.freeze_symbolic(True)
            unfreeze_backbone(model, config.use_lora)
            for p in model.injector.parameters():
                p.requires_grad = True
            decoder_ds = stage2_dataset if stage2_dataset else train_dataset
            run("Decoder", Stage3_Decoder_Trainer, decoder_ds,
                include_responses=True, lr=args.lr * 0.5, epochs=args.epochs)

        if "joint" in args.stages:
            print("\nStage 3: Joint")
            model.freeze_symbolic(False)
            unfreeze_backbone(model, config.use_lora)
            for p in model.injector.parameters():
                p.requires_grad = True
            joint_ds = stage2_dataset if stage2_dataset else train_dataset
            run("Joint", Stage4_Joint_Trainer, joint_ds,
                include_responses=True, lr=args.lr * 0.1, epochs=args.epochs,
                patience_bonus=2)

    if args.phase in ("chat", "both"):
        print("\nChat/Trace Phase")
        chat_path = args.chat_dataset
        chat_dataset = (
            load_dataset(Path(chat_path), args.max_samples)
            if chat_path and Path(chat_path).exists() else train_dataset
        )
        c_chat, r_chat, _, _, _, _ = extract_vocab_from_dataset(chat_dataset)
        merge_new_labels(concept_map, c_chat.keys())
        merge_new_labels(relation_map, r_chat.keys())

        model.freeze_symbolic(False)
        unfreeze_backbone(model, config.use_lora)
        for p in model.injector.parameters():
            p.requires_grad = True
        run("Chat", Stage4_Joint_Trainer, chat_dataset,
            include_responses=True, lr=args.chat_lr, epochs=args.chat_epochs,
            chat=True)

    # --- final save ------------------------------------------------------
    final_path = Path(args.output_dir) / "final_causal_model.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "model_name": config.model_name,
            "tap_layer_ratio": config.tap_layer_ratio,
            "adapter_layers": config.adapter_layers,
            "n_entity_types": config.n_entity_types,
            "n_relations": config.n_relations,
            "n_concepts": config.n_concepts,
            "max_nodes": config.max_nodes,
        },
        "concept_map": concept_map,
        "relation_map": relation_map,
    }, final_path)
    print(f"\nSaved final model to {final_path}")

    if eval_dataset is not None:
        run_evaluation(model, tokenizer, eval_dataset, args.device, config,
                       stage_name="Final", logger=logger)


if __name__ == "__main__":
    main()
