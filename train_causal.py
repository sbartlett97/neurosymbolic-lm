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

Each stage logs train loss and a validation loss, and early-stops on the
validation loss. Stage 1 also reports token-entity and relation F1 (the
Phase-1 gate). Decoder/joint/chat also report the validation loss with the
injection gate zeroed (the Phase-2 "is injection a no-op" gate) and the
abstention accuracy on should_respond=0 samples.

Outputs in --output-dir:
  tokenizer/            tokenizer incl. the added chat tokens
  stage_<name>.pt       end-of-stage checkpoint (kept, one per stage)
  latest.pt             last epoch of the running stage (overwritten)
  final_causal_model.pt end of run
Load any of them with load_causal_checkpoint().

Usage:
    python train_causal.py --preset qwen3-0.6b \
        --dataset data/curated/curated.jsonl --stages symbolic decoder joint

    python train_causal.py --preset qwen3-0.6b --phase chat \
        --chat-dataset data/curated_traces/traces.jsonl

    # Several GPUs on one node (--batch-size is per GPU)
    torchrun --nproc_per_node 4 train_causal.py --preset qwen3-0.6b ...

    # Offline-model smoke run (tiny random backbone, real tokenizer)
    python train_causal.py --preset causal-testing --dataset tiny.jsonl \
        --epochs 1 --max-samples 16
"""

import argparse
import gc
import math
import os
import random
import sys
import time
from dataclasses import asdict, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import CausalModelConfig, CAUSAL_MODEL_PRESETS
from data.chat_template import CHAT_SPECIAL_TOKENS, ChatTemplate
from data.collator_causal import CausalCognitiveCollator
from model.neurosymbolic_causal import NeuroSymbolicCausalLM, build_tiny_backbone
from model.entity import entity_node_scores
from train import (
    extract_vocab_from_dataset,
    load_dataset,
    load_vocab_file,
    merge_new_labels,
    resolve_device,
)
from training import (
    EarlyStopping,
    Stage2_Symbolic_Trainer,
    Stage3_Decoder_Trainer,
    Stage4_Joint_Trainer,
    TrainingLogger,
)
from training.trainers import create_entity_token_labels


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Staged training for NeuroSymbolicCausalLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--dataset", type=str, default="data/curated/curated.jsonl")
    data_group.add_argument("--eval-dataset", type=str, default=None,
                            help="One validation JSONL for every stage. Default: "
                                 "each stage uses <file>_val.jsonl next to its "
                                 "own training file (--dataset, --stage2-dataset, "
                                 "--chat-dataset)")
    data_group.add_argument("--stage2-dataset", type=str, default=None)
    data_group.add_argument("--chat-dataset", type=str, default=None)
    data_group.add_argument("--max-samples", type=int, default=None)
    data_group.add_argument("--max-eval-samples", type=int, default=500)

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--preset", type=str, default="qwen3-0.6b",
                             choices=sorted(CAUSAL_MODEL_PRESETS.keys()))
    model_group.add_argument("--model-name", type=str, default=None,
                             help="Override backbone checkpoint")
    model_group.add_argument("--tap-layer-ratio", type=float, default=None)
    model_group.add_argument("--adapter-layers", type=int, default=None,
                             help="0 = pure-causal extraction baseline")
    model_group.add_argument("--max-input-length", type=int, default=None)
    model_group.add_argument("--max-output-length", type=int, default=None)
    model_group.add_argument("--no-gradient-checkpointing", action="store_true")
    model_group.add_argument("--gradient-checkpointing", action="store_true",
                             help="Force it on (presets set their own default)")

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--phase", type=str, default="pretrain",
                             choices=["pretrain", "chat", "both"])
    train_group.add_argument("--stages", type=str, nargs="+",
                             default=["symbolic", "decoder", "joint"],
                             choices=["symbolic", "decoder", "joint"])
    train_group.add_argument("--epochs", type=int, default=5)
    train_group.add_argument("--chat-epochs", type=int, default=3)
    train_group.add_argument("--batch-size", type=int, default=4,
                             help="Per GPU. Effective batch = batch-size x "
                                  "grad-accum x number of GPUs")
    train_group.add_argument("--grad-accum", type=int, default=1,
                             help="Micro-batches per optimizer step")
    train_group.add_argument("--soft-logic-weight", type=float, default=0.1,
                             help="Weight of the taxonomy relation domain/range "
                                  "constraint (0 = off)")
    train_group.add_argument("--symbolic-lr", type=float, default=1e-4,
                             help="Stage 1 LR (heads only, backbone frozen)")
    train_group.add_argument("--lr", type=float, default=5e-5,
                             help="Stage 2 LR is lr*0.5, joint lr*0.1")
    train_group.add_argument("--chat-lr", type=float, default=1e-5)
    train_group.add_argument("--weight-decay", type=float, default=0.01)
    train_group.add_argument("--warmup-ratio", type=float, default=0.1)
    train_group.add_argument("--patience", type=int, default=3)
    train_group.add_argument("--seed", type=int, default=42)
    train_group.add_argument("--system-prompt", type=str,
                             default="You are a helpful assistant.")

    hw_group = parser.add_argument_group("Hardware")
    hw_group.add_argument("--device", type=str, default="auto",
                          help="auto/cuda/mps/cpu")
    hw_group.add_argument("--no-amp", action="store_true",
                          help="Disable autocast (fp32 compute; not valid "
                               "with a bf16 LoRA base)")
    hw_group.add_argument("--amp-dtype", type=str, default="bf16",
                          choices=["bf16", "fp16"],
                          help="fp16 only for GPUs without bf16 (pre-Ampere)")
    hw_group.add_argument("--num-workers", type=int, default=2)

    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--output-dir", type=str, default="checkpoints_causal")
    out_group.add_argument("--resume", type=str, default=None,
                           help="Checkpoint to start from. Its config and label "
                                "maps are used; pick the remaining --stages.")
    out_group.add_argument("--log-dir", type=str, default="runs")
    out_group.add_argument("--log-every", type=int, default=50,
                           help="Print train loss every N steps")
    out_group.add_argument("--debug", action="store_true",
                           help="Per-step NaN weight scan and extra prints")

    return parser.parse_args(argv)


# ----------------------------------------------------------------------
# Config, vocabulary, checkpoints
# ----------------------------------------------------------------------

def build_model_config(args) -> CausalModelConfig:
    config = CAUSAL_MODEL_PRESETS[args.preset]()
    if args.model_name:
        config.model_name = args.model_name
    if args.tap_layer_ratio is not None:
        config.tap_layer_ratio = args.tap_layer_ratio
    if args.adapter_layers is not None:
        config.adapter_layers = args.adapter_layers
    if args.max_input_length is not None:
        config.max_input_length = args.max_input_length
    if args.max_output_length is not None:
        config.max_output_length = args.max_output_length
    if args.no_gradient_checkpointing:
        config.gradient_checkpointing = False
    if args.gradient_checkpointing:
        config.gradient_checkpointing = True
    return config


def config_from_dict(d: Dict) -> CausalModelConfig:
    known = {f.name for f in fields(CausalModelConfig)}
    return CausalModelConfig(**{k: v for k, v in d.items() if k in known})


def require_path(path: Optional[str], flag: str) -> Optional[Path]:
    """A path the user passed must exist. A typo must not silently fall
    back to another dataset (e.g. stage 2 training on abstain-only data)."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Error: {flag} not found: {p}")
    return p


def build_vocab(datasets_with_paths, config: CausalModelConfig):
    """Label maps from the first vocab file found, extended by any labels
    in the other datasets. Head sizes are fixed from the result, so this
    runs before the model is built."""
    concept_map: Dict[str, int] = {}
    relation_map: Dict[str, int] = {}
    entity_type_name_map: Dict[str, int] = {}
    concept_to_type: Dict[str, int] = {}

    vocab_file = None
    for _, path in datasets_with_paths:
        vocab_file = load_vocab_file(path)
        if vocab_file is not None:
            break

    if vocab_file is not None:
        concept_map.update(vocab_file["concepts"])
        relation_map.update(vocab_file["relations"])
        entity_type_name_map.update(vocab_file.get("entity_types", {}))
        concept_to_type.update(vocab_file.get("concept_to_entity_type", {}))
    else:
        print("  Warning: no *_vocab.json found; deriving labels from data. "
              "Label ids then depend on file contents.")

    for dataset, _ in datasets_with_paths:
        c, r, t, _, _, _ = extract_vocab_from_dataset(dataset)
        merge_new_labels(concept_map, c.keys())
        merge_new_labels(relation_map, r.keys())
        if vocab_file is None:
            for name, idx in t.items():
                concept_to_type.setdefault(name, idx)
        names = set()
        for sample in dataset:
            names.update(x for x in sample.get("entity_types", []) if x)
            for ann in sample.get("message_annotations", []):
                names.update(x for x in ann.get("entity_types", []) if x)
        merge_new_labels(entity_type_name_map, sorted(names))

    max_type = max(
        list(entity_type_name_map.values()) + list(concept_to_type.values()),
        default=0,
    )
    config.n_entity_types = max(config.n_entity_types, max_type + 1)
    config.n_relations = max(config.n_relations, max(relation_map.values(), default=0) + 1)
    config.n_concepts = max(config.n_concepts, max(concept_map.values(), default=0))
    return concept_map, relation_map, entity_type_name_map, concept_to_type


def save_causal_checkpoint(path: Path, model, config: CausalModelConfig,
                           maps: Dict, extra: Optional[Dict] = None):
    """Everything needed to rebuild the model: weights, full config, label
    maps. The tokenizer lives in <output_dir>/tokenizer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "concept_map": maps["concept_map"],
        "relation_map": maps["relation_map"],
        "entity_type_map": maps["entity_type_map"],
        "concept_to_entity_type_map": maps["concept_to_entity_type_map"],
        "chat_system_prompt": maps.get("chat_system_prompt"),
    }
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: a crash mid-save keeps the old file
    print(f"  Saved {path}")


def build_backbone_and_tokenizer(config: CausalModelConfig, device: str,
                                 tokenizer_dir: Optional[Path] = None):
    tok_name = tokenizer_dir or config.tokenizer_name or config.model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The model packs prompt[:len] then nodes then response, so prompts
    # must be right-padded (Gemma tokenizers default to left)
    tokenizer.padding_side = "right"
    added = ChatTemplate.add_special_tokens(tokenizer)

    if config.model_name == "__tiny__":
        backbone = build_tiny_backbone(vocab_size=len(tokenizer))
    else:
        from transformers import AutoModelForCausalLM

        # fp32 master weights for full fine-tuning; bf16 autocast does the
        # fast math. A LoRA base never updates, so bf16 there saves memory.
        dtype = (
            torch.bfloat16 if config.use_lora and device.startswith("cuda")
            else torch.float32
        )
        print(f"Loading causal backbone: {config.model_name} ({dtype})")
        backbone = AutoModelForCausalLM.from_pretrained(
            config.model_name, dtype=dtype
        )
        # Grow only. Many checkpoints pad the vocab (Qwen3: 151936 rows for
        # 151669 tokens), so the new chat tokens often fit already. A shrink
        # here would not repeat on reload (the saved tokenizer already holds
        # the tokens), and the checkpoint shapes would then differ.
        n_rows = backbone.get_input_embeddings().num_embeddings
        if len(tokenizer) > n_rows:
            backbone.resize_token_embeddings(len(tokenizer))
            print(f"Added {added} chat special tokens; embeddings resized "
                  f"{n_rows} -> {len(tokenizer)}")
        elif added:
            print(f"Added {added} chat special tokens (fit in the existing "
                  f"{n_rows} embedding rows)")

    backbone.config.pad_token_id = tokenizer.pad_token_id
    return backbone, tokenizer


def taxonomy_type_constraints(maps: Dict, n_types: int, n_relations: int):
    """(E, E, R) bool mask of which relation may hold between which head
    and tail entity types, from the taxonomy's relation signatures.

    Only labels present in both the label maps and the taxonomy get
    constraints; everything else (head headroom, custom labels) stays
    allowed.
    """
    from data.curate.taxonomy import get_default_taxonomy

    tax = get_default_taxonomy()
    allowed = torch.ones(n_types, n_types, n_relations, dtype=torch.bool)
    type_ids = {
        name: idx for name, idx in maps.get("entity_type_map", {}).items()
        if name in tax.entity_types and 0 < idx < n_types
    }
    for rel, r in maps.get("relation_map", {}).items():
        if rel not in tax.relation_labels or not 0 < r < n_relations:
            continue
        _, heads, tails = tax.relation_labels[rel]
        for a_name, a in type_ids.items():
            for b_name, b in type_ids.items():
                ok = (not heads or a_name in heads) and (not tails or b_name in tails)
                allowed[a, b, r] = ok
    return allowed


def build_model(config: CausalModelConfig, backbone, tokenizer,
                maps: Optional[Dict] = None,
                soft_logic: bool = True) -> NeuroSymbolicCausalLM:
    if config.use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise SystemExit("--preset *-lora requires peft: pip install peft")
        # The chat tokens are new rows; with a frozen base they would keep
        # their random init, so train just those rows
        new_ids = tokenizer.convert_tokens_to_ids(CHAT_SPECIAL_TOKENS)
        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            trainable_token_indices={"embed_tokens": list(new_ids)},
        )
        backbone = get_peft_model(backbone, lora_cfg)
        print("LoRA adapters attached to backbone")

    model = NeuroSymbolicCausalLM(
        model_name=config.model_name,
        backbone=backbone,
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
    if maps is not None and soft_logic:
        allowed = taxonomy_type_constraints(
            maps, config.n_entity_types, config.n_relations
        )
        if not allowed.all():
            model.softlogic.set_type_constraints(allowed)
            print(f"Soft logic: {int((~allowed).sum())} disallowed "
                  "(head type, tail type, relation) triples")
    # Names of backbone params that train in stages 2/3: all of them for
    # full FT; adapters + new token rows for LoRA (peft set these)
    model._backbone_trainable = {
        n for n, p in model.backbone.named_parameters()
        if p.requires_grad or not config.use_lora
    }
    return model


def load_causal_checkpoint(path: str, device: str = "cpu"):
    """Rebuild model + tokenizer from a train_causal checkpoint.

    Returns (model, tokenizer, checkpoint_dict).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = config_from_dict(ckpt["config"])
    tok_dir = Path(path).parent / "tokenizer"
    backbone, tokenizer = build_backbone_and_tokenizer(
        config, device, tok_dir if tok_dir.exists() else None
    )
    model = build_model(config, backbone, tokenizer, maps=ckpt)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    report_key_mismatch(result)
    return model.to(device), tokenizer, ckpt


def report_key_mismatch(result):
    if result.missing_keys or result.unexpected_keys:
        print(f"  Checkpoint key mismatch: {len(result.missing_keys)} missing, "
              f"{len(result.unexpected_keys)} unexpected")
        for k in (result.missing_keys[:5] + result.unexpected_keys[:5]):
            print(f"    {k}")
        raise SystemExit(
            "Checkpoint does not match the model (LoRA vs full FT, or a "
            "different preset). Refusing to train on partially loaded weights."
        )


# ----------------------------------------------------------------------
# Freezing
# ----------------------------------------------------------------------

def set_stage_trainable(model: NeuroSymbolicCausalLM, symbolic: bool,
                        backbone: bool, injector: bool):
    model.freeze_symbolic(not symbolic)
    for n, p in model.backbone.named_parameters():
        p.requires_grad = backbone and n in model._backbone_trainable
    for p in model.injector.parameters():
        p.requires_grad = injector


def make_optimizer(model, lr: float, weight_decay: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # No decay on biases, norms, and the injection gate
        if p.ndim < 2 or "norm" in n.lower() or n.endswith("gate"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    n_params = sum(p.numel() for g in groups for p in g["params"])
    print(f"  Trainable parameters: {n_params:,}")
    return AdamW([g for g in groups if g["params"]], lr=lr)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

@torch.no_grad()
def symbolic_metrics(model, loader, device) -> Dict[str, float]:
    """Token-entity micro F1 (non-"none" classes) and relation F1 over
    gold entity pairs. These are the Phase-1 gate numbers."""
    was_training = model.training
    model.eval()
    ent_tp = ent_fp = ent_fn = 0
    rel_tp = rel_fp = rel_fn = 0
    for batch in loader:
        batch = to_device(batch, device)
        spans = batch["entity_token_spans"]
        out = model(batch["input_ids"], batch["attention_mask"], spans=spans)
        logits = out["token_ent_logits"]
        B, L, E = logits.shape
        labels = create_entity_token_labels(batch, B, L, E, logits.device)
        pred = logits.argmax(-1)
        valid = labels != -100
        ent_tp += int(((pred == labels) & (labels > 0) & valid).sum())
        ent_fp += int(((pred != labels) & (pred > 0) & valid).sum())
        ent_fn += int(((pred != labels) & (labels > 0) & valid).sum())

        rel_pred = out["rel_logits_matrix"].argmax(-1)  # (B, N, N)
        N = rel_pred.shape[1]
        for b in range(B):
            present = [j for j, (s, _) in enumerate(spans[b][:N]) if s >= 0]
            gold = {(h, t): r for (h, t, r) in batch["relations"][b]
                    if h < N and t < N}
            for h in present:
                for t in present:
                    if h == t:
                        continue
                    p, g = int(rel_pred[b, h, t]), gold.get((h, t), 0)
                    if p > 0 and p == g:
                        rel_tp += 1
                    else:
                        rel_fp += int(p > 0)
                        rel_fn += int(g > 0)
    model.train(was_training)

    def f1(tp, fp, fn):
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return 2 * prec * rec / max(prec + rec, 1e-9)

    return {"entity_f1": f1(ent_tp, ent_fp, ent_fn),
            "relation_f1": f1(rel_tp, rel_fp, rel_fn)}


@torch.no_grad()
def generation_report(model, tokenizer, collator, samples: List[Dict],
                      device, max_new_tokens: int, n_show: int = 3) -> Dict[str, float]:
    """Greedy generation on validation samples, with prompts built by the
    training collator (identical formatting). Reports abstention accuracy:
    should_respond=0 samples must emit EOS first."""
    eos = tokenizer.eos_token_id
    abstain_total = abstain_ok = 0
    respond_total = respond_nonempty = 0
    shown = 0
    for sample in samples:
        batch = to_device(collator([sample]), device)
        gen = model.generate(batch["input_ids"], batch["attention_mask"],
                             max_length=max_new_tokens, do_sample=False,
                             eos_token_id=eos, pad_token_id=tokenizer.pad_token_id)
        first = int(gen[0, 0]) if gen.shape[1] > 0 else eos
        if int(sample.get("should_respond", 0)) == 0:
            abstain_total += 1
            abstain_ok += int(first == eos)
        else:
            respond_total += 1
            respond_nonempty += int(first != eos)
            if shown < n_show:
                shown += 1
                text = tokenizer.decode(gen[0], skip_special_tokens=True)
                print(f"    target: {str(sample.get('response', ''))[:80]!r}")
                print(f"    output: {text[:80]!r}")
    return {
        "abstain_acc": abstain_ok / max(abstain_total, 1),
        "respond_rate": respond_nonempty / max(respond_total, 1),
    }


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


# ----------------------------------------------------------------------
# Stage loop
# ----------------------------------------------------------------------

def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def broadcast_flag(flag: bool, device) -> bool:
    """Rank 0's decision, for every rank (e.g. early stopping)."""
    if not dist.is_initialized():
        return flag
    t = torch.tensor([int(flag)], device=device)
    dist.broadcast(t, src=0)
    return bool(t.item())


def all_reduce_sum(value: float, device) -> float:
    if not dist.is_initialized():
        return value
    t = torch.tensor([float(value)], device=device)
    dist.all_reduce(t)
    return float(t.item())


def run_stage(name: str, model, trainer, train_loader, val_loader, epochs: int,
              scheduler, logger, early_stopping, args, save_fn,
              symbolic_eval: bool, gen_eval_fn=None) -> None:
    """One stage. `model` is the core module; `trainer` may hold a DDP
    wrapper. Only rank 0 validates, logs, and saves; it decides early
    stopping and broadcasts the decision."""
    accum = max(1, args.grad_accum)
    n_batches = len(train_loader)
    coll_device = args.device if dist.is_initialized() and dist.get_backend() == "nccl" else "cpu"
    print(f"\n{'=' * 60}\nStage: {name} ({epochs} epochs, {n_batches} batches/epoch, "
          f"{math.ceil(n_batches / accum)} optimizer steps/epoch)\n{'=' * 60}")
    step = 0
    best_val = math.inf
    for epoch in range(epochs):
        model.train()
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        t0 = time.time()
        total, n_valid, n_skipped = 0.0, 0, 0
        for i, batch in enumerate(train_loader):
            last = (i + 1) % accum == 0 or i + 1 == n_batches
            loss = trainer.train_step(
                to_device(batch, args.device), step=last, loss_scale=1.0 / accum
            )
            if trainer.stepped:
                scheduler.step()
                step += 1
                if step % args.log_every == 0:
                    rate = step / max(time.time() - t0, 1e-9)
                    print(f"  step {step}: loss={loss:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e} "
                          f"gate={float(model.injector.gate):.4f} ({rate:.2f} steps/s)")
            if loss == 0.0 or loss != loss:
                n_skipped += 1
                continue
            total += loss
            n_valid += 1
            logger.log_scalar(f"{name}/train_loss", loss, step)

        # Losses averaged over all ranks, so every rank sees the same value
        n_valid_all = all_reduce_sum(n_valid, coll_device)
        train_loss = all_reduce_sum(total, coll_device) / max(n_valid_all, 1)
        msg = (f"Epoch {epoch + 1}/{epochs}: train={train_loss:.4f} "
               f"(valid batches {n_valid}, skipped {n_skipped}, "
               f"{time.time() - t0:.0f}s)")
        if n_valid_all == 0:
            print(msg)
            raise SystemExit(
                f"Stage {name}: no batch produced a valid loss. Check that the "
                "dataset has the labels this stage needs."
            )

        stop = False
        if is_main_process():
            monitor = train_loss
            if val_loader is not None:
                val_loss = mean_val_loss(trainer, val_loader, args.device)
                msg += f" val={val_loss:.4f}"
                logger.log_scalar(f"{name}/val_loss", val_loss, step)
                if val_loss == val_loss:
                    monitor = val_loss
                if not symbolic_eval:
                    # Phase-2 gate: does the model use the injected nodes?
                    gate = model.injector.gate.data.clone()
                    model.injector.gate.data.zero_()
                    ablated = mean_val_loss(trainer, val_loader, args.device)
                    model.injector.gate.data.copy_(gate)
                    msg += f" val_gate0={ablated:.4f}"
                    logger.log_scalar(f"{name}/val_loss_gate0", ablated, step)
                else:
                    m = symbolic_metrics(model, val_loader, args.device)
                    msg += f" entity_f1={m['entity_f1']:.3f} relation_f1={m['relation_f1']:.3f}"
                    for k, v in m.items():
                        logger.log_scalar(f"{name}/{k}", v, step)
            print(msg)

            if gen_eval_fn is not None:
                m = gen_eval_fn()
                if m:
                    print(f"  abstain_acc={m['abstain_acc']:.3f} "
                          f"respond_rate={m['respond_rate']:.3f}")
                    for k, v in m.items():
                        logger.log_scalar(f"{name}/{k}", v, step)

            if monitor < best_val:
                best_val = monitor
            stop = early_stopping(monitor)

        save_fn("latest.pt", {"stage": name, "epoch": epoch + 1})
        if broadcast_flag(stop, coll_device):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    save_fn(f"stage_{name.lower()}.pt", {"stage": name, "best_monitor": best_val})


def mean_val_loss(trainer, loader, device) -> float:
    losses = [trainer.eval_loss(to_device(b, device)) for b in loader]
    losses = [x for x in losses if x is not None]
    return sum(losses) / len(losses) if losses else float("nan")


class _NullLogger:
    def log_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def setup_distributed(device: str):
    """torchrun sets WORLD_SIZE/RANK/LOCAL_RANK. Returns (device, world)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return device, 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # Ranks 1.. wait while rank 0 validates, generates, and saves a
    # multi-GB checkpoint. The 10-minute NCCL default is too short for that.
    timeout = timedelta(minutes=int(os.environ.get("NEUROSYM_DIST_TIMEOUT_MIN", "120")))
    if device == "cuda":
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timeout)
        device = f"cuda:{local_rank}"
    else:
        dist.init_process_group("gloo", timeout=timeout)
        device = "cpu"
    if dist.get_rank() != 0:
        # One copy of the console output
        sys.stdout = open(os.devnull, "w")
    return device, world


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.device, world = setup_distributed(resolve_device(args.device))
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if world > 1:
        print(f"Distributed: {world} processes")

    resume_ckpt = None
    if args.resume:
        require_path(args.resume, "--resume")
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        config = config_from_dict(resume_ckpt["config"])
        print(f"Resuming from {args.resume} (config and label maps from checkpoint)")
    else:
        config = build_model_config(args)

    print("=" * 70)
    print("NeuroSymbolicCausalLM Training (decoder-only)")
    print("=" * 70)
    print(f"Preset: {args.preset} | Backbone: {config.model_name} | Device: {args.device}")

    use_amp = not args.no_amp and args.device.startswith("cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    if use_amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("This GPU has no bf16; use --amp-dtype fp16")
    if config.use_lora and args.device.startswith("cuda") and not use_amp:
        raise SystemExit("A LoRA preset loads a bf16 base on CUDA; it needs "
                         "autocast (drop --no-amp)")
    pin = args.device.startswith("cuda")

    # --- data ---------------------------------------------------------
    dataset_path = require_path(args.dataset, "--dataset")
    stage2_path = require_path(args.stage2_dataset, "--stage2-dataset")
    chat_path = require_path(args.chat_dataset, "--chat-dataset")
    explicit_eval = require_path(args.eval_dataset, "--eval-dataset")

    def val_for(path: Optional[Path]):
        """Validation set for one training file: --eval-dataset if given,
        else <file>_val.jsonl next to it. Each stage validates on data like
        its own: the Stage-1 corpus is abstain-only, so decoder early
        stopping on it would reward always abstaining."""
        if path is None:
            return None
        vpath = explicit_eval or path.with_name(path.stem + "_val.jsonl")
        if not vpath.exists():
            print(f"  Warning: no validation set for {path.name} (write "
                  f"{vpath.name} or pass --eval-dataset); early stopping "
                  "uses train loss")
            return None
        ds = load_dataset(vpath, args.max_eval_samples)
        print(f"Validation for {path.name}: {len(ds)} samples from {vpath}")
        return ds

    train_dataset = load_dataset(dataset_path, args.max_samples)
    print(f"Loaded {len(train_dataset)} samples from {dataset_path}")
    stage2_dataset = load_dataset(stage2_path, args.max_samples) if stage2_path else None
    chat_dataset = load_dataset(chat_path, args.max_samples) if chat_path else None
    val_sets = {
        "symbolic": val_for(dataset_path),
        "response": val_for(stage2_path or dataset_path),
        "chat": val_for(chat_path),
    }
    if args.phase in ("chat", "both") and chat_dataset is None:
        raise SystemExit("--phase chat needs --chat-dataset")

    # --- vocabulary (fixed before the model is built) -------------------
    if resume_ckpt is not None:
        concept_map = resume_ckpt["concept_map"]
        relation_map = resume_ckpt["relation_map"]
        entity_type_name_map = resume_ckpt.get("entity_type_map", {})
        concept_to_type = resume_ckpt.get("concept_to_entity_type_map", {})
    else:
        sources = [(train_dataset, dataset_path)]
        if stage2_dataset is not None:
            sources.append((stage2_dataset, stage2_path))
        if chat_dataset is not None:
            sources.append((chat_dataset, chat_path))
        concept_map, relation_map, entity_type_name_map, concept_to_type = \
            build_vocab(sources, config)
    print(f"Vocab: {len(concept_map)} concepts, {len(relation_map)} relations, "
          f"{len(entity_type_name_map)} entity types | heads: "
          f"{config.n_entity_types} types, {config.n_relations} relations, "
          f"{config.n_concepts} concepts")
    maps = {
        "concept_map": concept_map,
        "relation_map": relation_map,
        "entity_type_map": entity_type_name_map,
        "concept_to_entity_type_map": concept_to_type,
        "chat_system_prompt": args.system_prompt,
    }

    # --- model + tokenizer ---------------------------------------------
    backbone, tokenizer = build_backbone_and_tokenizer(config, args.device)
    model = build_model(config, backbone, tokenizer, maps=maps,
                        soft_logic=args.soft_logic_weight > 0)
    if resume_ckpt is not None:
        report_key_mismatch(
            model.load_state_dict(resume_ckpt["model_state_dict"], strict=False)
        )
        del resume_ckpt
    model = model.to(args.device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,} (tap layer {model.tap_layer})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        tokenizer.save_pretrained(out_dir / "tokenizer")

    def save_fn(filename, extra):
        if is_main_process():
            save_causal_checkpoint(out_dir / filename, model, config, maps, extra)
        if dist.is_initialized():
            dist.barrier()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = (TrainingLogger(log_dir=args.log_dir, experiment_name=f"causal_{timestamp}")
              if is_main_process() else _NullLogger())
    chat_template = ChatTemplate(default_system_prompt=args.system_prompt)

    def make_collator(include_responses, chat=False):
        # The chat template is always passed: trace rows in any dataset
        # format through it
        return CausalCognitiveCollator(
            tokenizer=tokenizer,
            concept_map=concept_map,
            relation_map=relation_map,
            include_responses=include_responses,
            concept_to_entity_type_map=concept_to_type,
            max_length=config.max_input_length,
            max_output_length=config.max_output_length,
            chat_mode=chat,
            chat_template=chat_template,
            max_nodes=config.max_nodes,
            entity_type_map=entity_type_name_map,
        )

    def loader(dataset, include_responses, chat=False, shuffle=True):
        # Training batches are split across ranks; validation runs on
        # rank 0 over the whole set
        sampler = (DistributedSampler(dataset, shuffle=True, seed=args.seed, drop_last=True)
                   if shuffle and world > 1 else None)
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            collate_fn=make_collator(include_responses, chat),
            pin_memory=pin,
            drop_last=shuffle and len(dataset) > args.batch_size,
        )

    def run(name, trainer_cls, dataset, include_responses, lr, epochs,
            val_key, chat=False, symbolic_eval=False, patience_bonus=0):
        eval_dataset = val_sets[val_key]
        train_loader = loader(dataset, include_responses, chat)
        val_loader = (loader(eval_dataset, include_responses, chat, shuffle=False)
                      if eval_dataset is not None else None)
        optimizer = make_optimizer(model, lr, args.weight_decay)
        total_steps = max(1, math.ceil(len(train_loader) / args.grad_accum) * epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * args.warmup_ratio), total_steps
        )
        # DDP registers only the parameters that require grad when it is
        # built, so it is re-built per stage, after the stage's freezing
        train_model = model
        if world > 1:
            train_model = DDP(
                model,
                device_ids=[torch.cuda.current_device()] if args.device.startswith("cuda") else None,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )
        kwargs = {}
        if trainer_cls is not Stage3_Decoder_Trainer:
            kwargs["soft_logic_weight"] = args.soft_logic_weight
        trainer = trainer_cls(train_model, optimizer, use_amp=use_amp, device=args.device,
                              amp_dtype=amp_dtype, check_health=args.debug, **kwargs)

        gen_eval_fn = None
        if include_responses and eval_dataset is not None:
            gen_collator = make_collator(False, chat)
            # Both kinds: answers (respond_rate) and abstentions (abstain_acc)
            rows = list(eval_dataset)
            gen_samples = (
                [r for r in rows if int(r.get("should_respond", 0)) == 1][:10]
                + [r for r in rows if int(r.get("should_respond", 0)) == 0][:10]
            )
            gen_eval_fn = lambda: generation_report(  # noqa: E731
                model, tokenizer, gen_collator, gen_samples, args.device,
                max_new_tokens=min(64, config.max_output_length),
            )

        run_stage(name, model, trainer, train_loader, val_loader, epochs,
                  scheduler, logger, EarlyStopping(patience=args.patience + patience_bonus),
                  args, save_fn, symbolic_eval=symbolic_eval, gen_eval_fn=gen_eval_fn)
        del trainer, train_model
        # Drop the old DDP reducer and its hooks before the next stage wraps
        gc.collect()

    # =====================================================================
    if args.phase in ("pretrain", "both"):
        if "symbolic" in args.stages:
            print("\nStage 1: Symbolic (backbone frozen)")
            set_stage_trainable(model, symbolic=True, backbone=False, injector=False)
            run("Symbolic", Stage2_Symbolic_Trainer, train_dataset,
                include_responses=False, lr=args.symbolic_lr, epochs=args.epochs,
                val_key="symbolic", symbolic_eval=True)

        decoder_ds = stage2_dataset if stage2_dataset is not None else train_dataset
        if ("decoder" in args.stages or "joint" in args.stages):
            n_respond = sum(int(s.get("should_respond", 0)) for s in decoder_ds)
            print(f"\nResponse data: {n_respond}/{len(decoder_ds)} samples "
                  "with should_respond=1")
            if n_respond == 0:
                raise SystemExit(
                    "Decoder/joint stages have no should_respond=1 samples; the "
                    "model would only learn to abstain. Pass --stage2-dataset "
                    "with hybrid QA data."
                )

        if "decoder" in args.stages:
            print("\nStage 2: Response (heads frozen, injector + backbone)")
            set_stage_trainable(model, symbolic=False, backbone=True, injector=True)
            run("Decoder", Stage3_Decoder_Trainer, decoder_ds,
                include_responses=True, lr=args.lr * 0.5, epochs=args.epochs,
                val_key="response")

        if "joint" in args.stages:
            print("\nStage 3: Joint")
            set_stage_trainable(model, symbolic=True, backbone=True, injector=True)
            run("Joint", Stage4_Joint_Trainer, decoder_ds,
                include_responses=True, lr=args.lr * 0.1, epochs=args.epochs,
                val_key="response", patience_bonus=2)

    if args.phase in ("chat", "both"):
        print("\nChat/Trace Phase")
        set_stage_trainable(model, symbolic=True, backbone=True, injector=True)
        run("Chat", Stage4_Joint_Trainer, chat_dataset,
            include_responses=True, lr=args.chat_lr, epochs=args.chat_epochs,
            val_key="chat", chat=True)

    save_fn("final_causal_model.pt", {"stage": "final"})
    logger.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
