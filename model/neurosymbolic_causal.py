"""NeuroSymbolicCausalLM: symbolic heads bolted onto a decoder-only backbone.

Architecture (see docs/DECODER_ONLY_MIGRATION.md):

    [prompt tokens ...........] [node_1 .. node_K] [response tokens .. EOS]
         |                            ^                      |
         | read pass                  | gated soft tokens    | LM loss
         v                            |                      v
    tap layer L_t -> bidirectional  GNN over span-pooled   causal LM over
    extraction adapter -> entity/   entity nodes           packed sequence
    concept/relation heads

Two passes per training step:

1. **Read**: backbone over the prompt with hidden states exposed; the
   tapped mid-layer goes through a small bidirectional adapter and feeds
   the symbolic heads (entity classifier, concept bank, GNN, relation
   scorer, soft logic) — the same modules the encoder-decoder model uses.
2. **Write**: the full packed sequence runs through the backbone once for
   the LM loss. The node features reach it in one of two ways
   (``injection``):

   - ``"prefix"``: gated soft tokens between the prompt and the response
     (``[prompt | K nodes | response]``). Used by v1 and v2.
   - ``"xattn"``: gated cross-attention in the decoder layers after the tap
     layer (``[prompt | response]``, as in the base model). Only the
     positions from the last prompt token on read the nodes: the nodes come
     from the whole prompt, and the last prompt token predicts the first
     response token (EOS = abstain).

   Returned ``logits`` are pre-aligned to the response tokens, so trainers
   compute plain cross-entropy against the collator's ``decoder_labels``
   with no shifting.

Output-dict keys and the ``forward(input_ids, attention_mask, spans,
y_ids)`` call shape match ``NeuroSymbolicLM``, so the stage trainers and
the continual-learning stack work with either model.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .entity import TokenEntityClassifier, ConceptBank, entity_node_scores
from .extraction_adapter import ExtractionAdapter
from .gnn import SimpleGNN
from .injection import NodeCrossAttention, NodePrefixInjector
from .logic import SoftLogicConstraints
from .pooling import MultiQueryPool


def build_tiny_backbone(vocab_size: int = 2048, hidden_size: int = 64):
    """Tiny from-config Llama for offline tests — no downloads required."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    return LlamaForCausalLM(cfg)


def _check_right_padded(attention_mask: torch.Tensor) -> None:
    """Packing copies prompt[:len], so prompts must be right-padded.

    Some tokenizers (e.g. Gemma) default to left padding. The training
    script sets padding_side="right"; this check catches other callers.
    """
    lens = attention_mask.sum(dim=1, keepdim=True)
    pos = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    if not torch.equal((pos < lens).to(attention_mask.dtype), attention_mask):
        raise ValueError(
            "NeuroSymbolicCausalLM expects right-padded prompts; set "
            "tokenizer.padding_side = 'right'"
        )


class NeuroSymbolicCausalLM(nn.Module):
    """Decoder-only neurosymbolic LM with soft-token broadcasting."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B-Base",
        backbone: Optional[nn.Module] = None,
        tap_layer_ratio: float = 0.66,
        adapter_layers: int = 1,
        adapter_heads: int = 4,
        adapter_ff_mult: int = 4,
        n_entity_types: int = 24,
        n_relations: int = 128,
        n_concepts: int = 1024,
        concept_dim: int = 256,
        node_dim: int = 256,
        max_nodes: int = 16,
        dropout: float = 0.1,
        injection_gate_init: float = 0.1,
        injection: str = "prefix",
        xattn_every: int = 2,
        xattn_heads: int = 8,
        node_dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        max_input_length: int = 2048,
        max_output_length: int = 512,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        # --- backbone -----------------------------------------------------
        if backbone is not None:
            self.backbone = backbone
        elif model_name == "__tiny__":
            self.backbone = build_tiny_backbone()
        else:
            from transformers import AutoModelForCausalLM

            # fp32 by default: AdamW on bf16 weights drops most small
            # updates. Mixed precision comes from bf16 autocast in the
            # trainers. Pass bf16 only for a frozen or LoRA-wrapped base.
            if torch_dtype is None:
                torch_dtype = torch.float32
            print(f"Loading causal backbone: {model_name} ({torch_dtype})")
            self.backbone = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch_dtype
            )

        if gradient_checkpointing:
            # Non-reentrant: works when the inputs do not require grad
            # (frozen embeddings, LoRA) and with the two-pass forward.
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("Gradient checkpointing enabled")

        self.hidden_size = self.backbone.config.hidden_size
        self.vocab_size = self.backbone.config.vocab_size
        n_layers = self.backbone.config.num_hidden_layers
        # hidden_states[0] is the embedding output; layer i is index i
        self.tap_layer = max(1, min(n_layers, round(n_layers * tap_layer_ratio)))

        # --- symbolic components (same modules as NeuroSymbolicLM) ---------
        self.n_entity_types = n_entity_types
        self.n_relations = n_relations
        self.n_concepts = n_concepts
        self.node_dim = node_dim
        self.max_nodes = max_nodes

        self.adapter = ExtractionAdapter(
            self.hidden_size,
            n_layers=adapter_layers,
            n_heads=adapter_heads,
            ff_mult=adapter_ff_mult,
            dropout=dropout,
        )
        self.token_ent = TokenEntityClassifier(self.hidden_size, n_entity_types)
        self.token_pool = MultiQueryPool(self.hidden_size, n_queries=6)
        self.concept_bank = ConceptBank(n_concepts, concept_dim)
        self.concept_proj = nn.Linear(self.hidden_size, concept_dim)
        self.node_proj = nn.Linear(self.hidden_size, node_dim)
        self.gnn = SimpleGNN(node_dim, n_layers=2)
        self.rel_scorer = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, n_relations),
        )
        self.softlogic = SoftLogicConstraints(n_entity_types, n_relations)

        # --- broadcasting -------------------------------------------------
        self.injection = injection
        # Training only, xattn only: the share of samples that get no graph,
        # so the backbone keeps working without it
        self.node_dropout = float(node_dropout)
        if injection == "prefix":
            self.injector = NodePrefixInjector(
                node_dim, self.hidden_size, gate_init=injection_gate_init
            )
        elif injection == "xattn":
            # Layers from the tap on: they do not change the tapped states,
            # so the read pass needs no graph
            self.xattn_layer_ids = list(range(self.tap_layer, n_layers, xattn_every))
            self.xattn = nn.ModuleList(
                NodeCrossAttention(self.hidden_size, node_dim, xattn_heads)
                for _ in self.xattn_layer_ids
            )
            # Set by _write / _generate, read by the layer hooks. _write
            # leaves it set: gradient checkpointing runs the layers again
            # in backward. The next read pass clears it.
            self._xattn_ctx: Optional[Dict] = None
            layers = self._decoder().layers
            for j, i in enumerate(self.xattn_layer_ids):
                layers[i].register_forward_hook(self._xattn_hook(j))
        else:
            raise ValueError(f"injection must be 'prefix' or 'xattn', not {injection!r}")

    # ------------------------------------------------------------------
    # Graph injection helpers
    # ------------------------------------------------------------------

    def _decoder(self) -> nn.Module:
        return (self.backbone.get_decoder()
                if hasattr(self.backbone, "get_decoder") else self.backbone)

    def _xattn_hook(self, j: int):
        def hook(module, args, output):
            ctx = self._xattn_ctx
            if ctx is None:
                return None
            hidden = output[0] if isinstance(output, tuple) else output
            mask = ctx["pos_mask"]
            if mask is None:  # generate: only the newest position predicts
                mask = hidden.new_zeros(hidden.shape[0], hidden.shape[1], 1)
                mask[:, -1] = 1
            elif mask.shape[1] != hidden.shape[1]:
                return None  # not the pass that set the context
            new = hidden + self.xattn[j](hidden, ctx["nodes"], ctx["valid"], mask)
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
        return hook

    def injection_modules(self) -> List[nn.Module]:
        """The modules that put the graph into the backbone."""
        return [self.injector] if self.injection == "prefix" else [self.xattn]

    def injection_gates(self) -> List[nn.Parameter]:
        """The scalar gates. All at 0 = no graph signal."""
        if self.injection == "prefix":
            return [self.injector.gate]
        return [m.gate for m in self.xattn]

    def injection_gate_value(self) -> float:
        """For logs: the prefix gate, or the mean |tanh(gate)| of the
        cross-attention layers."""
        if self.injection == "prefix":
            return float(self.injector.gate)
        return float(torch.stack([torch.tanh(g.detach()).abs() for g in self.injection_gates()]).mean())

    # ------------------------------------------------------------------
    # Read pass
    # ------------------------------------------------------------------

    def _read(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Backbone over the prompt; tapped mid-layer through the adapter."""
        _check_right_padded(attention_mask)
        if self.injection == "xattn":
            self._xattn_ctx = None
        out = self._decoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # A bf16 base (LoRA) feeds fp32 heads. Cast here so eval, generate
        # and continual learning work without autocast too.
        tapped = out.hidden_states[self.tap_layer].to(self.token_ent.fc.weight.dtype)
        return self.adapter(tapped, attention_mask)

    def _extract_node_features(
        self,
        enc: torch.Tensor,
        token_ent_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        spans: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> torch.Tensor:
        """Span-pool (gold spans) or top-K entity-score positions -> nodes."""
        B, L, _ = enc.shape

        if spans is not None:
            nodes = []
            for i in range(B):
                batch_nodes = []
                for (s, e) in (spans[i] if i < len(spans) else [])[: self.max_nodes]:
                    if int(s) < 0:
                        # Entity not in the (truncated) prompt: empty node
                        batch_nodes.append(
                            torch.zeros(self.node_dim, device=enc.device, dtype=enc.dtype)
                        )
                        continue
                    s = max(0, min(int(s), L - 1))
                    e = max(s, min(int(e), L - 1))
                    batch_nodes.append(self.node_proj(enc[i, s:e + 1].mean(dim=0)))
                while len(batch_nodes) < self.max_nodes:
                    batch_nodes.append(
                        torch.zeros(self.node_dim, device=enc.device, dtype=enc.dtype)
                    )
                nodes.append(torch.stack(batch_nodes[: self.max_nodes], dim=0))
            return torch.stack(nodes, dim=0)

        # Top-K by P(entity), padding positions excluded
        ent_scores = entity_node_scores(token_ent_logits, attention_mask)
        k = min(self.max_nodes, L)
        _, topk_idx = torch.topk(ent_scores, k=k, dim=-1)

        gathered = torch.gather(
            enc, 1, topk_idx.unsqueeze(-1).expand(-1, -1, enc.size(-1))
        )
        node_feats = self.node_proj(gathered)
        # A prompt shorter than k (padded to a longer one in the batch)
        # selects padding too. Those slots get empty nodes, as when the
        # prompt runs alone, so a score does not depend on its batch.
        real = torch.gather(attention_mask, 1, topk_idx).unsqueeze(-1)
        node_feats = node_feats * real.to(node_feats.dtype)
        if k < self.max_nodes:
            pad = node_feats.new_zeros(B, self.max_nodes - k, self.node_dim)
            node_feats = torch.cat([node_feats, pad], dim=1)
        return node_feats

    def _compute_node_entity_probs(
        self,
        token_ent_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        spans: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> torch.Tensor:
        B, L, E = token_ent_logits.shape
        N = self.max_nodes
        device = token_ent_logits.device

        if spans is not None:
            out = torch.zeros(B, N, E, device=device)
            for i in range(B):
                for j, (s, e) in enumerate((spans[i] if i < len(spans) else [])[:N]):
                    if int(s) < 0:
                        continue
                    s = max(0, min(int(s), L - 1))
                    e = max(s, min(int(e), L - 1))
                    out[i, j] = token_ent_logits[i, s:e + 1].mean(dim=0)
            return F.softmax(out, dim=-1)

        ent_scores = entity_node_scores(token_ent_logits, attention_mask)
        k = min(N, L)
        _, topk_idx = torch.topk(ent_scores, k=k, dim=-1)
        gathered = torch.gather(
            token_ent_logits, 1, topk_idx.unsqueeze(-1).expand(-1, -1, E)
        )
        if k < N:
            gathered = torch.cat(
                [gathered, gathered.new_zeros(B, N - k, E)], dim=1
            )
        return F.softmax(gathered, dim=-1)

    def _compute_pairwise_relations(
        self, node_feats: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        B, N, _ = node_feats.shape
        device = node_feats.device

        node_i = node_feats.unsqueeze(2).expand(-1, -1, N, -1)
        node_j = node_feats.unsqueeze(1).expand(-1, N, -1, -1)
        rel_logits_matrix = self.rel_scorer(
            torch.cat([node_i, node_j], dim=-1).clamp(-100, 100)
        ).clamp(-50, 50)

        triu = torch.triu_indices(N, N, offset=1, device=device)
        pair_logits = [
            rel_logits_matrix[b, triu[0], triu[1]] for b in range(B)
        ]
        return pair_logits, rel_logits_matrix

    # ------------------------------------------------------------------
    # Write pass
    # ------------------------------------------------------------------

    def _write(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        node_feats: Optional[torch.Tensor],
        y_ids: torch.Tensor,
        node_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pack [prompt | nodes | response] (prefix) or [prompt | response]
        (xattn) per sample and return logits aligned to the response
        tokens: logits[:, j] predicts y_ids[:, j].
        """
        B, R = y_ids.shape
        device = input_ids.device
        emb_layer = self.backbone.get_input_embeddings()

        prompt_emb = emb_layer(input_ids)
        resp_emb = emb_layer(y_ids)
        node_emb = None
        K = 0
        if node_feats is not None and self.injection == "prefix":
            node_emb = self.injector(node_feats).to(prompt_emb.dtype)
            K = node_emb.shape[1]

        # Response slots all attend. Padding sits after a sample's last
        # real token, and causal attention means it can only affect later
        # positions, whose labels are -100. This avoids any pad-id lookup
        # (pad often equals EOS, and a None pad id must not mask token 0).
        prompt_lens = attention_mask.sum(dim=1).tolist()
        total = [int(p) + K + R for p in prompt_lens]
        T = max(total)

        packed_emb = prompt_emb.new_zeros(B, T, self.hidden_size)
        packed_mask = torch.zeros(B, T, dtype=attention_mask.dtype, device=device)

        for i in range(B):
            p = int(prompt_lens[i])
            pos = 0
            packed_emb[i, :p] = prompt_emb[i, :p]
            packed_mask[i, :p] = 1
            pos = p
            if K > 0:
                packed_emb[i, pos:pos + K] = node_emb[i]
                packed_mask[i, pos:pos + K] = 1
                pos += K
            packed_emb[i, pos:pos + R] = resp_emb[i]
            packed_mask[i, pos:pos + R] = 1

        # y_ids[:, j] is predicted at packed position p + K + j - 1. Only
        # keep logits from the earliest such position on: a full B x T x V
        # tensor at a 150k vocab does not fit a 24 GB card.
        starts = torch.tensor(
            [int(p) + K - 1 for p in prompt_lens], device=device
        )
        if self.injection == "xattn":
            self._xattn_ctx = None
            if node_feats is not None:
                pos = torch.arange(T, device=device)
                pos_mask = (pos[None, :] >= starts[:, None]).unsqueeze(-1)
                if self.training and self.node_dropout > 0:
                    keep = torch.rand(B, device=device) >= self.node_dropout
                    pos_mask = pos_mask & keep[:, None, None]
                if node_valid is None:
                    node_valid = node_feats.new_ones(node_feats.shape[:2], dtype=torch.bool)
                self._xattn_ctx = {"nodes": node_feats, "valid": node_valid,
                                   "pos_mask": pos_mask}
        first = int(starts.min())
        out = self.backbone(
            inputs_embeds=packed_emb,
            attention_mask=packed_mask,
            logits_to_keep=T - first,
            return_dict=True,
        )
        tail_logits = out.logits  # (B, T - first, V)

        idx = (starts - first).unsqueeze(1) + torch.arange(R, device=device)
        return torch.gather(
            tail_logits, 1,
            idx.unsqueeze(-1).expand(-1, -1, tail_logits.shape[-1]),
        )

    # ------------------------------------------------------------------
    # Public API (parity with NeuroSymbolicLM)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        spans: Optional[List[List[Tuple[int, int]]]] = None,
        y_ids: Optional[torch.Tensor] = None,
        use_soft_nodes: bool = True,
        pixel_values: Optional[torch.Tensor] = None,  # API parity; unused
    ) -> Dict[str, torch.Tensor]:
        enc = self._read(input_ids, attention_mask)

        token_ent_logits = self.token_ent(enc)
        token_pool, _ = self.token_pool(enc, attention_mask)

        node_feats = self._extract_node_features(
            enc, token_ent_logits, attention_mask, spans
        )
        node_feats_refined = self.gnn(node_feats)
        # Empty slots are all-zero before the GNN (it can fill them in)
        node_valid = node_feats.abs().sum(dim=-1) > 0

        concept_query = self.concept_proj(token_pool)
        concept_vec, concept_probs = self.concept_bank.soft_assign(concept_query)

        pair_logits, rel_logits_matrix = self._compute_pairwise_relations(
            node_feats_refined
        )
        node_entity_type_probs = self._compute_node_entity_probs(
            token_ent_logits, attention_mask, spans
        )

        outputs: Dict[str, torch.Tensor] = {
            "entity_logits": token_ent_logits,
            "token_ent_logits": token_ent_logits,
            "concept_logits": concept_probs,
            "concept_probs": concept_probs,
            "pair_relation_logits": pair_logits,
            "node_entity_type_probs": node_entity_type_probs,
            "rel_logits_matrix": rel_logits_matrix,
            "enc": enc,
            "node_feats": node_feats_refined,
        }

        if y_ids is not None:
            outputs["logits"] = self._write(
                input_ids,
                attention_mask,
                node_feats_refined if use_soft_nodes else None,
                y_ids,
                node_valid,
            )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        use_soft_nodes: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Generate from [prompt | nodes] soft-token-augmented context.

        Prompts are left-packed so generation continues from the last real
        position for every sample. Returns only the generated tokens
        (HF behavior for ``inputs_embeds`` input).
        """
        was_training = self.training
        self.eval()
        try:
            return self._generate(
                input_ids, attention_mask, max_length, use_soft_nodes, **kwargs
            )
        finally:
            self.train(was_training)

    def _generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int,
        use_soft_nodes: bool,
        **kwargs,
    ) -> torch.Tensor:
        enc = self._read(input_ids, attention_mask)
        token_ent_logits = self.token_ent(enc)
        node_feats = self._extract_node_features(
            enc, token_ent_logits, attention_mask
        )
        node_feats_refined = self.gnn(node_feats)

        emb_layer = self.backbone.get_input_embeddings()
        prompt_emb = emb_layer(input_ids)
        node_emb = None
        K = 0
        if use_soft_nodes and self.injection == "xattn":
            # The hooks let only the newest position read the nodes. That
            # is right only if each step adds one cached position
            if kwargs.get("use_cache") is False:
                raise ValueError("injection='xattn' generation needs use_cache=True")
            self._xattn_ctx = {"nodes": node_feats_refined,
                               "valid": node_feats.abs().sum(dim=-1) > 0,
                               "pos_mask": None}
        elif use_soft_nodes:
            node_emb = self.injector(node_feats_refined).to(prompt_emb.dtype)
            K = node_emb.shape[1]

        B = input_ids.shape[0]
        prompt_lens = attention_mask.sum(dim=1).tolist()
        T = max(int(p) for p in prompt_lens) + K

        packed_emb = prompt_emb.new_zeros(B, T, self.hidden_size)
        packed_mask = torch.zeros(
            B, T, dtype=attention_mask.dtype, device=input_ids.device
        )
        for i in range(B):
            p = int(prompt_lens[i])
            # Left-pad: real content ends at the last position
            start = T - (p + K)
            packed_emb[i, start:start + p] = prompt_emb[i, :p]
            packed_mask[i, start:start + p] = 1
            if K > 0:
                packed_emb[i, start + p:start + p + K] = node_emb[i]
                packed_mask[i, start + p:start + p + K] = 1

        try:
            return self.backbone.generate(
                inputs_embeds=packed_emb,
                attention_mask=packed_mask,
                max_new_tokens=max_length,
                **kwargs,
            )
        finally:
            if self.injection == "xattn":
                self._xattn_ctx = None

    # -- convenience for staged freezing ---------------------------------

    def symbolic_modules(self) -> List[nn.Module]:
        """Modules trained in Stage 1 (backbone frozen)."""
        return [
            self.adapter, self.token_ent, self.token_pool, self.concept_bank,
            self.concept_proj, self.node_proj, self.gnn, self.rel_scorer,
            self.softlogic,
        ]

    def freeze_backbone(self, freeze: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def freeze_symbolic(self, freeze: bool = True):
        for m in self.symbolic_modules():
            for p in m.parameters():
                p.requires_grad = not freeze
