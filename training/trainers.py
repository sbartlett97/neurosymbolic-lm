"""Stage trainers for tiered training of the neurosymbolic model.

Includes:
- Gradient clipping for training stability
- Mixed precision training support (bf16 autocast by default)
- Proper entity span alignment

Both NeuroSymbolicLM and NeuroSymbolicCausalLM work with these trainers.

Label conventions (shared by every stage):
- Token entity labels: 0 = "none" on real tokens, 1.. = entity types,
  -100 = padding or an entity with unknown type (ignored).
- Relation labels: 0 = "no relation" between two gold entities, 1.. =
  relation types. Gold spans feed the nodes, so node j is entity j.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from contextlib import nullcontext


# ----------------------------------------------------------------------
# Shared label and loss helpers
# ----------------------------------------------------------------------

def graph_kwargs(batch: Dict[str, Any]) -> Dict[str, Any]:
    """The separate read-pass input of a batch (rows with "graph_text"),
    as model keyword arguments. Empty for other batches."""
    return {k: batch[k] for k in ("graph_input_ids", "graph_attention_mask")
            if k in batch}


def create_entity_token_labels(
    batch: Dict[str, Any],
    B: int,
    L: int,
    E: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-token entity labels aligned to the read-pass input positions
    (the graph input when the batch has one)."""
    attention_mask = batch.get("graph_attention_mask", batch.get("attention_mask"))
    if attention_mask is not None:
        mask = attention_mask[:, :L].to(device)
        labels = torch.where(
            mask > 0,
            torch.zeros(B, L, dtype=torch.long, device=device),
            torch.full((B, L), -100, dtype=torch.long, device=device),
        )
    else:
        labels = torch.zeros(B, L, dtype=torch.long, device=device)

    entity_token_spans = batch.get("entity_token_spans")
    entity_type_labels = batch["entity_type_labels"]

    if entity_token_spans is not None:
        for i in range(B):
            types = entity_type_labels[i]
            for j, (start, end) in enumerate(entity_token_spans[i]):
                if j >= len(types):
                    break
                if start < 0:
                    continue  # entity not in the (truncated) input
                etype = int(types[j].item())
                # Unknown type: the tokens are an entity, but not of a type
                # we can name. Do not teach "none" there.
                target = min(etype, E - 1) if etype > 0 else -100
                labels[i, start:min(end + 1, L)] = target
    else:
        # Fallback: assign entity types to first N token positions
        for i in range(B):
            num_ents = int((batch["entity_ids"][i] > 0).sum().item())
            for j in range(min(num_ents, L)):
                etype = int(entity_type_labels[i, j].item())
                labels[i, j] = min(etype, E - 1) if etype > 0 else -100

    return labels


def entity_loss(
    logits: torch.Tensor, labels: torch.Tensor, none_weight: float
) -> torch.Tensor:
    """Token entity cross-entropy, safe when a batch has no valid labels.

    Most tokens are "none", so class 0 is down-weighted.
    """
    E = logits.shape[-1]
    if (labels != -100).sum() == 0:
        return logits.sum() * 0.0
    weight = torch.ones(E, device=logits.device, dtype=torch.float32)
    weight[0] = none_weight
    return F.cross_entropy(
        logits.reshape(-1, E).float(),
        labels.reshape(-1),
        weight=weight,
        ignore_index=-100,
    )


def concept_loss(out: Dict[str, Any], batch: Dict[str, Any], bce: nn.Module) -> torch.Tensor:
    """BCE between pooled concept probabilities and averaged entity concepts."""
    concept_labels = batch["concept_labels"]
    n_concept_types = concept_labels.shape[2]

    entity_mask = (batch["entity_ids"] > 0).float()
    num_valid_entities = entity_mask.sum(dim=1, keepdim=True).clamp(min=1)
    aggregated = (concept_labels * entity_mask.unsqueeze(-1)).sum(dim=1) / num_valid_entities

    concept_probs = out["concept_probs"]
    n_concepts_bank = concept_probs.shape[1]

    if n_concept_types == n_concepts_bank:
        target = aggregated
    elif n_concept_types < n_concepts_bank:
        padding = torch.zeros(
            aggregated.shape[0], n_concepts_bank - n_concept_types,
            device=aggregated.device, dtype=aggregated.dtype,
        )
        target = torch.cat([aggregated, padding], dim=1)
    else:
        target = aggregated[:, :n_concepts_bank]

    # Convert probabilities to logits for BCEWithLogitsLoss, with clamping
    concept_logits = torch.logit(concept_probs.float().clamp(1e-6, 1 - 1e-6))
    return bce(concept_logits, target.to(concept_logits.device))


def relation_loss(
    rel_logits_matrix: torch.Tensor,
    relations: List[List],
    entity_token_spans: Optional[List[List]],
    negative_weight: float,
) -> torch.Tensor:
    """Directed relation cross-entropy over gold entity pairs.

    Node j is entity j (gold spans feed the nodes). Every ordered pair of
    two present entities is a training example: gold triples take their
    relation label, all other pairs take 0 ("no relation"). Without the
    negatives, the scorer predicts a relation for every pair at inference.
    """
    B, N, _, R = rel_logits_matrix.shape
    device = rel_logits_matrix.device
    heads, tails, targets, batch_idx = [], [], [], []

    for b in range(B):
        spans = entity_token_spans[b] if entity_token_spans is not None else []
        present = [
            j for j, (s, _) in enumerate(spans[:N]) if s >= 0
        ]
        if len(present) < 2:
            continue
        gold = {}
        for (h, t, r) in relations[b]:
            if h < N and t < N and h != t and 0 < r < R:
                gold[(h, t)] = r
        for h in present:
            for t in present:
                if h == t:
                    continue
                heads.append(h)
                tails.append(t)
                targets.append(gold.get((h, t), 0))
                batch_idx.append(b)

    if not targets:
        return rel_logits_matrix.sum() * 0.0

    b_t = torch.tensor(batch_idx, device=device)
    h_t = torch.tensor(heads, device=device)
    t_t = torch.tensor(tails, device=device)
    y = torch.tensor(targets, device=device)
    logits = rel_logits_matrix[b_t, h_t, t_t].float()

    weight = torch.ones(R, device=device, dtype=torch.float32)
    weight[0] = negative_weight
    return F.cross_entropy(logits, y, weight=weight)


def _ddp_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def node_mask_from_spans(
    entity_token_spans: Optional[List[List]], B: int, N: int, device
) -> Optional[torch.Tensor]:
    """(B, N) mask of nodes that hold a present gold entity."""
    if entity_token_spans is None:
        return None
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    for b in range(B):
        for j, (s, _) in enumerate(entity_token_spans[b][:N]):
            mask[b, j] = s >= 0
    return mask


def soft_logic_term(core, out: Dict[str, Any], spans, like: torch.Tensor) -> torch.Tensor:
    """Soft-logic loss, or 0 when the model has no rules or constraints."""
    softlogic = core.softlogic
    if not softlogic.active or "node_entity_type_probs" not in out:
        return like.new_zeros(())
    probs = out["node_entity_type_probs"]
    node_mask = node_mask_from_spans(spans, probs.shape[0], probs.shape[1], probs.device)
    loss, _ = softlogic(probs, out["rel_logits_matrix"], node_mask=node_mask)
    return loss


class BaseTrainer:
    """Base trainer class with common functionality."""

    def __init__(
        self,
        model,
        optimizer,
        grad_clip_norm: float = 1.0,
        use_amp: bool = False,
        device: str = "cpu",
        amp_dtype: torch.dtype = torch.bfloat16,
        check_health: bool = False,
    ):
        # `model` may be a DistributedDataParallel wrapper: training
        # forwards go through it (gradient sync); attribute access and
        # validation use the wrapped module
        self.model = model
        self.core = getattr(model, "module", model)
        self.optimizer = optimizer
        self.grad_clip_norm = grad_clip_norm
        self.use_amp = use_amp and torch.cuda.is_available()
        self.amp_dtype = amp_dtype
        self.device = device
        # Scanning every weight for NaN forces GPU syncs each step, so it
        # runs only on request (--debug)
        self.check_health = check_health

        # bf16 has the fp32 exponent range and needs no loss scaling.
        # Only fp16 needs a GradScaler.
        self.scaler = (
            torch.amp.GradScaler('cuda')
            if self.use_amp and amp_dtype == torch.float16 else None
        )
        # Grads left by an earlier stage (same parameters, new optimizer)
        # must not leak into this stage's first step or its clip norm
        self.model.zero_grad(set_to_none=True)
        self.pending_grads = False
        self.stepped = False

    def _check_model_health(self) -> bool:
        """Check if model weights contain NaN or Inf, and try to fix them."""
        if not self.check_health:
            return True
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                if not hasattr(self, '_nan_weight_printed'):
                    self._nan_weight_printed = True
                    print(f"    DEBUG: NaN found in {name}, resetting to zeros")
                param.data = torch.where(torch.isnan(param.data), torch.zeros_like(param.data), param.data)
            if torch.isinf(param).any():
                if not hasattr(self, '_inf_weight_printed'):
                    self._inf_weight_printed = True
                    print(f"    DEBUG: Inf found in {name}, clamping")
                param.data = param.data.clamp(-10, 10)
        return True  # Always return True after fixing

    def _get_amp_context(self):
        """Get appropriate autocast context for mixed precision."""
        if self.use_amp:
            return torch.amp.autocast('cuda', dtype=self.amp_dtype)
        return nullcontext()

    def _clip_params(self):
        return [p for p in self.model.parameters() if p.grad is not None]

    def _backward(self, loss: torch.Tensor):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        self.pending_grads = True

    def _optimizer_step(self):
        """Clip, step, and clear grads (optional fp16 loss scaling)."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self._clip_params(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self._clip_params(), self.grad_clip_norm)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.model.zero_grad(set_to_none=True)
        self.pending_grads = False

    def _manual_grad_sync(self):
        world = dist.get_world_size()
        for p in self.model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad.div_(world)

    def _sync_skip(self, skip: bool) -> bool:
        """Under DDP, all ranks skip together. One rank skipping backward
        while the others all-reduce gradients is a deadlock."""
        if not _ddp_active():
            return skip
        device = self.device if dist.get_backend() == "nccl" else "cpu"
        flag = torch.tensor([int(skip)], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    @torch.no_grad()
    def eval_loss(self, batch: Dict[str, Any]) -> Optional[float]:
        """Loss on a batch without an optimizer step (validation)."""
        was_training = self.core.training
        self.core.eval()
        try:
            with self._get_amp_context():
                loss = self.compute_loss(batch, model=self.core)
        finally:
            self.core.train(was_training)
        if loss is None or not torch.isfinite(loss):
            return None
        return loss.item()

    def compute_loss(self, batch: Dict[str, Any], model=None) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def train_step(
        self, batch: Dict[str, Any], step: bool = True, loss_scale: float = 1.0
    ) -> float:
        """Forward + backward on one (micro-)batch.

        With gradient accumulation the caller passes step=False for all
        but the last micro-batch, and loss_scale = 1 / accum_steps. The
        optimizer steps when step=True and any micro-batch since the last
        step produced gradients (self.stepped tells the caller).
        """
        self._check_model_health()
        self.stepped = False

        # Skip DDP gradient sync on non-final micro-batches
        no_sync = getattr(self.model, "no_sync", None)
        sync_ctx = no_sync() if (not step and no_sync is not None) else nullcontext()
        with sync_ctx:
            with self._get_amp_context():
                loss = self.compute_loss(batch)
            skip = loss is None or not bool(torch.isfinite(loss))
            if self._sync_skip(skip):
                skip = True
            else:
                self._backward(loss * loss_scale)

        if step and self.pending_grads:
            if skip and _ddp_active() and no_sync is not None:
                # The final micro-batch had no backward, so the grads from
                # earlier no_sync micro-batches were never all-reduced
                self._manual_grad_sync()
            self._optimizer_step()
            self.stepped = True

        if skip:
            # The stage loop counts this batch as invalid
            return 0.0 if loss is None else float("nan")
        return loss.item()


class Stage2_Symbolic_Trainer(BaseTrainer):
    """Stage 2: Entity extraction, concept mapping, and relation head training."""

    def __init__(
        self,
        model,
        optimizer,
        soft_logic_weight: float = 0.1,
        entity_weight: float = 1.0,
        concept_weight: float = 1.0,
        relation_weight: float = 1.0,
        none_weight: float = 0.1,
        relation_negative_weight: float = 0.1,
        grad_clip_norm: float = 1.0,
        use_amp: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(model, optimizer, grad_clip_norm, use_amp, device, **kwargs)
        # Use BCEWithLogitsLoss for concept classification (applies sigmoid internally)
        self.bce = nn.BCEWithLogitsLoss()
        self.soft_logic_weight = soft_logic_weight
        self.entity_weight = entity_weight
        self.concept_weight = concept_weight
        self.relation_weight = relation_weight
        self.none_weight = none_weight
        self.relation_negative_weight = relation_negative_weight

    def compute_loss(self, batch: Dict[str, Any], model=None) -> torch.Tensor:
        spans = batch.get("entity_token_spans")
        out = (model or self.model)(
            batch["input_ids"],
            batch["attention_mask"],
            spans=spans,
            y_ids=None,
            **graph_kwargs(batch),
        )

        ent_logits = out["token_ent_logits"]
        B, L, E = ent_logits.shape
        labels = create_entity_token_labels(batch, B, L, E, ent_logits.device)
        ent_loss = entity_loss(ent_logits, labels, self.none_weight)

        con_loss = concept_loss(out, batch, self.bce)

        rel_loss = relation_loss(
            out["rel_logits_matrix"], batch["relations"], spans,
            self.relation_negative_weight,
        )

        soft_logic_loss = soft_logic_term(self.core, out, spans, ent_logits)

        loss = (self.entity_weight * ent_loss +
                self.concept_weight * con_loss +
                self.relation_weight * rel_loss +
                self.soft_logic_weight * soft_logic_loss)

        # Debug: print component losses on first step
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"    DEBUG Stage2 losses: ent={ent_loss.item():.4f}, "
                  f"con={con_loss.item():.4f}, rel={rel_loss.item():.4f}, "
                  f"logic={float(soft_logic_loss):.4f}, total={loss.item():.4f}")
        return loss


class Stage3_Decoder_Trainer(BaseTrainer):
    """Stage 3.5: Decoder response generation training.

    Nodes come from top-K entity scores (spans=None), the same path
    generate() uses at inference.
    """

    def __init__(
        self,
        model,
        optimizer,
        grad_clip_norm: float = 1.0,
        use_amp: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(model, optimizer, grad_clip_norm, use_amp, device, **kwargs)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def compute_loss(self, batch: Dict[str, Any], model=None) -> Optional[torch.Tensor]:
        if "decoder_input_ids" not in batch:
            return None
        labels = batch["decoder_labels"]
        if (labels != -100).sum() == 0:
            return None

        out = (model or self.model)(
            batch["input_ids"],
            batch["attention_mask"],
            spans=None,
            y_ids=batch["decoder_input_ids"],
            **graph_kwargs(batch),
        )
        logits = out["logits"]
        # Collator and model align logits with labels; no shift needed
        decoder_loss = self.ce(
            logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1)
        )

        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"    DEBUG Stage3 decoder_loss={decoder_loss.item():.4f}")
        return decoder_loss


class Stage4_Joint_Trainer(BaseTrainer):
    """Stage 4: Joint end-to-end training of all components.

    Abstention is learned through the decoder generating EOS tokens for
    should_respond=0 samples.

    Gold spans feed the nodes here (teacher forcing), so the symbolic
    losses line up with gold entity indices. generate() still selects
    nodes by top-K entity score at inference.
    """

    def __init__(
        self,
        model,
        optimizer,
        soft_logic_weight: float = 0.1,
        entity_weight: float = 1.0,
        concept_weight: float = 1.0,
        relation_weight: float = 1.0,
        decoder_weight: float = 1.0,
        none_weight: float = 0.1,
        relation_negative_weight: float = 0.1,
        grad_clip_norm: float = 1.0,
        use_amp: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(model, optimizer, grad_clip_norm, use_amp, device, **kwargs)
        # Use ignore_index=-100 for decoder (standard HF convention)
        self.ce_ignore_neg100 = nn.CrossEntropyLoss(ignore_index=-100)
        # Use BCEWithLogitsLoss for concept classification
        self.bce = nn.BCEWithLogitsLoss()
        self.soft_logic_weight = soft_logic_weight
        self.entity_weight = entity_weight
        self.concept_weight = concept_weight
        self.relation_weight = relation_weight
        self.decoder_weight = decoder_weight
        self.none_weight = none_weight
        self.relation_negative_weight = relation_negative_weight

    def compute_loss(self, batch: Dict[str, Any], model=None) -> torch.Tensor:
        decoder_input_ids = batch.get("decoder_input_ids", None)
        spans = batch.get("entity_token_spans")

        out = (model or self.model)(
            batch["input_ids"],
            batch["attention_mask"],
            spans=spans,
            y_ids=decoder_input_ids,
            **graph_kwargs(batch),
        )

        ent_logits = out["entity_logits"]
        B, L, E = ent_logits.shape
        labels = create_entity_token_labels(batch, B, L, E, ent_logits.device)
        ent_loss = entity_loss(ent_logits, labels, self.none_weight)

        con_loss = concept_loss(out, batch, self.bce)

        rel_loss = relation_loss(
            out["rel_logits_matrix"], batch["relations"], spans,
            self.relation_negative_weight,
        )

        # Decoder loss (includes EOS-based abstention learning)
        decoder_loss = ent_logits.new_zeros(())
        if "logits" in out and decoder_input_ids is not None and "decoder_labels" in batch:
            logits = out["logits"]
            dec_labels = batch["decoder_labels"]
            if (dec_labels != -100).sum() > 0:
                decoder_loss = self.ce_ignore_neg100(
                    logits.reshape(-1, logits.size(-1)).float(),
                    dec_labels.reshape(-1)
                )

        soft_logic_loss = soft_logic_term(self.core, out, spans, ent_logits)

        loss = (self.entity_weight * ent_loss +
                self.concept_weight * con_loss +
                self.relation_weight * rel_loss +
                self.decoder_weight * decoder_loss +
                self.soft_logic_weight * soft_logic_loss)

        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"    DEBUG Stage4 losses: ent={ent_loss.item():.4f}, con={con_loss.item():.4f}, "
                  f"rel={rel_loss.item():.4f}, dec={decoder_loss.item():.4f}, "
                  f"logic={float(soft_logic_loss):.4f}, total={loss.item():.4f}")
        return loss
