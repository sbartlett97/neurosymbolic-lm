
import math, random
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

def gelu(x): return F.gelu(x)


class PretrainedEncoderWrapper(nn.Module):
    """
    Wrapper for pre-trained transformer encoders (e.g., BERT) to match the interface
    of SimpleTransformerEncoder.
    """
    def __init__(self, model_name: str = "bert-base-uncased", freeze_encoder: bool = False):
        super().__init__()
        from transformers import AutoModel
        
        self.pretrained_model = AutoModel.from_pretrained(model_name)
        self.d_model = self.pretrained_model.config.hidden_size
        self.freeze_encoder = freeze_encoder
        
        if freeze_encoder:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass through pre-trained encoder.
        
        Args:
            input_ids: (B, L) token ids
            attention_mask: (B, L) attention mask (1 for real tokens, 0 for padding)
        
        Returns:
            hidden_states: (B, L, d_model) encoder outputs
        """
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state  # (B, L, hidden_size)


class PretrainedDecoderWrapper(nn.Module):
    """
    Wrapper for pre-trained decoder models (e.g., T5, BART) to match the interface
    of SimpleTransformerDecoder.
    
    Supports combining encoder outputs with node representations as memory.
    """
    def __init__(self, model_name: str = "t5-small", vocab_size: Optional[int] = None, 
                 freeze_decoder: bool = False, d_model: Optional[int] = None):
        super().__init__()
        from transformers import AutoModelForSeq2SeqLM, AutoConfig
        
        # Load model and config
        config = AutoConfig.from_pretrained(model_name)
        self.pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Extract decoder and lm_head
        # T5 uses decoder, BART also uses decoder
        self.decoder = self.pretrained_model.decoder
        self.lm_head = self.pretrained_model.lm_head
        
        # Get decoder dimension from config
        decoder_d_model = config.d_model if hasattr(config, 'd_model') else config.hidden_size
        
        # Get dimensions
        if d_model is None:
            self.d_model = decoder_d_model
            self.input_proj = nn.Identity()  # No projection needed
        else:
            self.d_model = d_model
            # Project encoder outputs to decoder dimension if needed
            if decoder_d_model != d_model:
                self.input_proj = nn.Linear(d_model, decoder_d_model)
            else:
                self.input_proj = nn.Identity()
        
        self.vocab_size = vocab_size if vocab_size else config.vocab_size
        self.freeze_decoder = freeze_decoder
        self.decoder_d_model = decoder_d_model
        
        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.lm_head.parameters():
                param.requires_grad = False
    
    def forward(self, y_ids: torch.Tensor, memory_tokens: torch.Tensor, 
                memory_nodes: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_token_mask: Optional[torch.Tensor] = None):
        """
        Forward pass through pre-trained decoder.
        
        Args:
            y_ids: (B, T) decoder input token ids
            memory_tokens: (B, L, d_model) encoder outputs
            memory_nodes: (B, N, d_model) node representations
            tgt_mask: Optional causal mask for decoder
            memory_token_mask: Optional mask for encoder outputs
        
        Returns:
            logits: (B, T, vocab_size) output logits
        """
        # Project memory to decoder dimension if needed
        memory_tokens_proj = self.input_proj(memory_tokens)  # (B, L, decoder_d_model)
        memory_nodes_proj = self.input_proj(memory_nodes)  # (B, N, decoder_d_model)
        
        # Concatenate encoder outputs and nodes along sequence dimension
        # This matches the current SimpleTransformerDecoder approach
        encoder_hidden_states = torch.cat([memory_tokens_proj, memory_nodes_proj], dim=1)  # (B, L+N, decoder_d_model)
        
        # Create attention mask for combined memory
        if memory_token_mask is not None:
            # Create mask for nodes (all ones, assuming all nodes are valid)
            B, L = memory_token_mask.shape
            N = memory_nodes.shape[1]
            node_mask = torch.ones(B, N, dtype=memory_token_mask.dtype, device=memory_token_mask.device)
            encoder_attention_mask = torch.cat([memory_token_mask, node_mask], dim=1)  # (B, L+N)
        else:
            encoder_attention_mask = None
        
        # Get decoder outputs
        # T5/BART decoders expect input_ids and encoder_hidden_states
        decoder_outputs = self.decoder(
            input_ids=y_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False
        )
        
        # Get hidden states
        hidden_states = decoder_outputs.last_hidden_state  # (B, T, decoder_d_model)
        
        # Get logits from language model head
        logits = self.lm_head(hidden_states)  # (B, T, vocab_size)
        
        return logits


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, d_model) * 0.01)
    def forward(self, x_len):
        return self.pe[:x_len].unsqueeze(0)  # (1, L, D)


class SimpleTransformerEncoder(nn.Module):
    def __init__(self, vocab_size:int, d_model:int=512, nhead:int=8, nlayer:int=6):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEmbedding(d_model, max_len=4096)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayer)
        self.d_model = d_model
    def forward(self, input_ids, attn_mask=None):
        x = self.tok_emb(input_ids) + self.pos(input_ids.size(1))
        # transformer expects src_key_padding_mask: True for padding — supply inverted mask if needed
        src_kpm = (attn_mask==0) if attn_mask is not None else None
        return self.enc(x, src_key_padding_mask=src_kpm)  # (B, L, D)


class SimpleTransformerDecoder(nn.Module):
    def __init__(self, vocab_size:int, d_model:int=512, nhead:int=8, nlayer:int=6):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEmbedding(d_model, max_len=2048)
        # build decoder layers that allow cross-attention to two KV sources (encoder tokens + nodes)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model, nhead, d_model*4, batch_first=True) for _ in range(nlayer)])
        self.to_vocab = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, y_ids, memory_tokens, memory_nodes, tgt_mask=None, memory_token_mask=None):
        # memory_nodes: (B, N_nodes, D) -> used as additional KV by concatenation (simple approach)
        # For cleaner control, we concatenate tokens and nodes into one memory stream.
        b, t = y_ids.shape
        y = self.tok_emb(y_ids) + self.pos(t)
        # build combined memory: concat token enc outputs and node KVs along sequence dim
        memory = torch.cat([memory_tokens, memory_nodes], dim=1)
        # Generate causal mask if not provided (for autoregressive generation)
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(t, y.device)
        memory_kpm = (memory_token_mask==0) if memory_token_mask is not None else None
        out = y
        for layer in self.layers:
            out = layer(out, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_kpm)
        logits = self.to_vocab(out)
        return logits  # (B, T, V)


class MultiQueryPool(nn.Module):
    def __init__(self, hidden_dim:int, n_queries:int=6):
        super().__init__()
        self.nq = n_queries
        self.q = nn.Parameter(torch.randn(n_queries, hidden_dim) * 0.02)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(n_queries * hidden_dim, hidden_dim)
    def forward(self, tokens:torch.Tensor, mask:Optional[torch.Tensor]=None):
        B, L, H = tokens.shape
        Q = self.q.unsqueeze(0).expand(B, -1, -1)  # (B, k, H)
        K = self.key(tokens)  # (B, L, H)
        V = self.value(tokens)
        scores = torch.matmul(Q, K.transpose(-1,-2)) / (H**0.5)  # (B, k, L)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1)==0, -1e9)
        attn = F.softmax(scores, dim=-1)
        pooled = torch.matmul(attn, V)  # (B, k, H)
        flat = pooled.view(B, -1)
        return self.out(flat), pooled  # (B,H), (B,k,H)

def span_mean_pool(tokens:torch.Tensor, spans:List[List[Tuple[int,int]]]):
    # tokens: (B,L,H), spans: per-batch list of (s,e) inclusive token indices
    B, L, H = tokens.shape
    out = tokens.new_zeros((B, H))
    for i in range(B):
        sps = spans[i]
        if len(sps)==0:
            out[i] = tokens[i].mean(dim=0)
            continue
        reps = []
        for (s,e) in sps:
            seg = tokens[i, s:e+1]
            reps.append(seg.mean(dim=0))
        out[i] = torch.stack(reps, dim=0).mean(dim=0)
    return out  # (B,H)


class TokenEntityClassifier(nn.Module):
    def __init__(self, d_model:int, n_entity_types:int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_entity_types)  # use BIO labels or entity-type per token
    def forward(self, token_hidden):
        return self.fc(token_hidden)  # (B,L,types)


class ConceptBank(nn.Module):
    def __init__(self, n_concepts:int=512, c_dim:int=256):
        super().__init__()
        self.n = n_concepts
        self.dim = c_dim
        self.emb = nn.Parameter(torch.randn(n_concepts, c_dim)*0.02)
    def lookup(self, concept_ids:torch.LongTensor):
        return F.embedding(concept_ids, self.emb)  # (B, M, c_dim)
    def soft_assign(self, query:torch.Tensor):
        # query: (B, c_dim) -> return probs over bank and soft vector
        qn = F.normalize(query, dim=-1)
        bankn = F.normalize(self.emb, dim=-1)
        sim = torch.matmul(qn, bankn.t())  # (B, n_concepts)
        p = F.softmax(sim, dim=-1)
        vec = torch.matmul(p, self.emb)  # (B, c_dim)
        return vec, p


class SimpleGNN(nn.Module):
    def __init__(self, node_dim:int, n_layers:int=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(node_dim*2, node_dim),
                nn.ReLU(),
                nn.Linear(node_dim, node_dim)
            ))
    def forward(self, node_feats:torch.Tensor, adj_mask:Optional[torch.Tensor]=None):
        # node_feats: (B, N, D)
        B, N, D = node_feats.shape
        h = node_feats
        for layer in self.layers:
            # compute pairwise messages
            hi = h.unsqueeze(2).expand(-1, -1, N, -1)  # (B,N,N,D)
            hj = h.unsqueeze(1).expand(-1, N, -1, -1)  # (B,N,N,D)
            pairs = torch.cat([hi, hj], dim=-1)  # (B,N,N,2D)
            m = layer(pairs)  # (B,N,N,D)
            if adj_mask is not None:
                m = m * adj_mask.unsqueeze(-1)  # mask messages
            msg = m.sum(dim=2)  # (B,N,D)
            h = h + msg  # residual update
        return h  # (B,N,D)


def pair_logits_to_matrix(pair_logits_list: List[torch.Tensor], pairs_index_map: List[torch.Tensor], num_nodes: int, n_rel: int, device: Optional[torch.device] = None):
    """
    Convert sparse pair logits (per-batch variable-length lists) into dense (B, N, N, R) tensors.

    Args:
      - pair_logits_list: list len B, each element tensor (P_b, R)
      - pairs_index_map: list len B, each tensor (P_b, 2) with (i,j) indices for each pair in same order as logits
      - num_nodes: fixed node count N used across batch (padding used if fewer nodes)
      - n_rel: number of relation types R
      - device: device for output

    Returns:
      - rel_tensor: (B, N, N, R) logits, diagonal zeros, symmetric if desired (we fill both [i,j] and [j,i])
    """
    B = len(pair_logits_list)
    rel_tensor = torch.zeros((B, num_nodes, num_nodes, n_rel), device=device)
    for b in range(B):
        pl = pair_logits_list[b]
        if pl is None or pl.numel() == 0:
            continue
        idxmap = pairs_index_map[b]  # (P_b, 2)
        assert idxmap.shape[0] == pl.shape[0]
        for p in range(pl.shape[0]):
            i, j = int(idxmap[p, 0].item()), int(idxmap[p, 1].item())
            rel_tensor[b, i, j] = pl[p]
            rel_tensor[b, j, i] = pl[p]  # symmetric assignment by default
    return rel_tensor


class SoftLogicConstraints(nn.Module):
    """
    Implements differentiable soft constraints as auxiliary losses.

    Rules are specified as tuples:
      (etype_a, etype_b, rel_idx, weight, polarity)
    where polarity=1 encourages relation presence, -1 discourages it.

    Loss computation strategy:
      - Given entity_type_probs: (B, N, E) distribution over entity types per node
      - Given rel_logits_matrix: (B, N, N, R) raw logits for relations between nodes
    The module computes expected relation probability between types (marginalizing over entity type distributions),
    evaluates whether the rule holds in expectation, and returns a BCE-style penalty scaled by weight.

    This provides a soft, differentiable constraint that pushes entity type distributions and relation logits toward
    being consistent with the rule set.
    """
    def __init__(self, n_entity_types: int, n_relations: int):
        super().__init__()
        self.rules = []
        self.n_entity_types = n_entity_types
        self.n_relations = n_relations

    def add_rule(self, etype_a: int, etype_b: int, rel_idx: int, weight: float = 1.0, polarity: int = 1):
        assert 0 <= etype_a < self.n_entity_types
        assert 0 <= etype_b < self.n_entity_types
        assert 0 <= rel_idx < self.n_relations
        assert polarity in (-1, 1)
        self.rules.append((etype_a, etype_b, rel_idx, float(weight), int(polarity)))

    def forward(self, entity_type_probs: torch.Tensor, rel_logits_matrix: torch.Tensor):
        """
        Args:
          - entity_type_probs: (B, N, E) probability distributions over entity types (softmax outputs)
          - rel_logits_matrix: (B, N, N, R) raw logits for relation predictions
        Returns:
          - total_loss: scalar tensor
          - details: dict with per-rule losses and optionally per-batch values for inspection
        """
        device = entity_type_probs.device
        B, N, E = entity_type_probs.shape
        _, _, _, R = rel_logits_matrix.shape
        assert E == self.n_entity_types and R == self.n_relations

        rel_probs = torch.sigmoid(rel_logits_matrix)  # use sigmoid for multi-label relations; if mutually-exclusive use softmax externally

        total_loss = torch.tensor(0.0, device=device)
        details = {"rules": []}

        if len(self.rules) == 0:
            return total_loss, details

        # Precompute expected pair-type marginals: for each (a,b) type pair, the expected probability that node i is type a and node j is type b.
        # This yields a (B, N, N, E, E) tensor if done naively; we'll compute per-rule to save memory.

        for (etype_a, etype_b, rel_idx, weight, polarity) in self.rules:
            # compute expected type product for every node pair: P(node_i type a) * P(node_j type b)
            # shape: (B, N, N)
            p_a = entity_type_probs[:, :, etype_a].unsqueeze(-1)  # (B, N, 1)
            p_b = entity_type_probs[:, :, etype_b].unsqueeze(1)  # (B, 1, N)
            type_pair_prob = (p_a * p_b).squeeze(-1)  # broadcasting -> (B, N, N)

            # Get relation probability for rel_idx
            rel_p = rel_probs[:, :, :, rel_idx]  # (B, N, N)

            # Expected relation probability conditioned on type pair: E[rel_p * I[type_pair]] over pairs
            # Compute mean over node pairs weighted by type_pair_prob
            weighted_rel = (type_pair_prob * rel_p).sum(dim=(1, 2))  # (B,)
            normalizer = type_pair_prob.sum(dim=(1, 2)).clamp(min=1e-6)  # (B,)
            expected_rel_given_types = weighted_rel / normalizer  # (B,)

            # If polarity == 1, we want expected_rel_given_types to be high; else low
            if polarity == 1:
                # loss term: BCE between expected_rel and 1
                target = torch.ones_like(expected_rel_given_types, device=device)
            else:
                target = torch.zeros_like(expected_rel_given_types, device=device)

            rule_loss = F.binary_cross_entropy(expected_rel_given_types, target)
            scaled = weight * rule_loss
            total_loss = total_loss + scaled
            details["rules"].append({"etype_a": etype_a, "etype_b": etype_b, "rel_idx": rel_idx, "weight": weight, "polarity": polarity, "loss": rule_loss.detach().cpu()})

        return total_loss, details


class Controller(nn.Module):
    def __init__(self, token_dim:int, node_dim:int):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(token_dim+node_dim, token_dim), nn.ReLU(), nn.Linear(token_dim, 3))
        # outputs logits for [answer, abstain, ask_clarify]
    def forward(self, token_pool:torch.Tensor, node_pool:torch.Tensor):
        x = torch.cat([token_pool, node_pool], dim=-1)
        return self.fc(x)  # (B,3)


class NeuroSymbolicLM(nn.Module):
    def __init__(self, vocab_size:int, d_model:int=512, n_entity_types:int=8, n_concepts:int=512, concept_dim:int=256, node_dim:int=256, max_nodes:int=16, 
                 encoder=None, use_pretrained_encoder:bool=False, pretrained_model_name:str="bert-base-uncased", freeze_encoder:bool=False,
                 decoder=None, use_pretrained_decoder:bool=False, pretrained_decoder_name:str="t5-small", freeze_decoder:bool=False):
        """
        Args:
            vocab_size: Vocabulary size (used for decoder, ignored if using pre-trained encoder/decoder)
            d_model: Model dimension (will be overridden if using pre-trained encoder)
            encoder: Optional custom encoder (if None, will create SimpleTransformerEncoder or use pre-trained)
            use_pretrained_encoder: If True, use pre-trained BERT encoder
            pretrained_model_name: Name of pre-trained encoder model to use
            freeze_encoder: If True, freeze pre-trained encoder parameters
            decoder: Optional custom decoder (if None, will create SimpleTransformerDecoder or use pre-trained)
            use_pretrained_decoder: If True, use pre-trained T5/BART decoder
            pretrained_decoder_name: Name of pre-trained decoder model to use
            freeze_decoder: If True, freeze pre-trained decoder parameters
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.n_entity_types = n_entity_types
        self.n_concepts = n_concepts
        self.concept_dim = concept_dim
        self.node_dim = node_dim
        self.max_nodes = max_nodes
        self.use_pretrained_encoder = use_pretrained_encoder
        self.use_pretrained_decoder = use_pretrained_decoder
        
        # Initialize encoder
        if encoder is not None:
            self.encoder = encoder
            self.d_model = encoder.d_model
        elif use_pretrained_encoder:
            self.encoder = PretrainedEncoderWrapper(pretrained_model_name, freeze_encoder=freeze_encoder)
            self.d_model = self.encoder.d_model
        else:
            self.encoder = SimpleTransformerEncoder(vocab_size, d_model, nhead=8, nlayer=6)
            self.d_model = d_model
        
        self.token_pool = MultiQueryPool(self.d_model, n_queries=6)
        self.token_ent = TokenEntityClassifier(self.d_model, n_entity_types)
        self.span_pool = MultiQueryPool(self.d_model, n_queries=4)
        self.concept_bank = ConceptBank(n_concepts, concept_dim)
        self.concept_proj = nn.Linear(self.d_model, concept_dim)
        self.node_proj = nn.Linear(self.d_model, node_dim)
        self.gnn = SimpleGNN(node_dim, n_layers=2)
        
        # Initialize decoder
        if decoder is not None:
            self.decoder = decoder
        elif use_pretrained_decoder:
            self.decoder = PretrainedDecoderWrapper(
                pretrained_decoder_name, 
                vocab_size=vocab_size,
                freeze_decoder=freeze_decoder,
                d_model=self.d_model
            )
        else:
            self.decoder = SimpleTransformerDecoder(vocab_size, self.d_model, nhead=8, nlayer=6)
        self.controller = Controller(d_model, node_dim)
        self.max_nodes = max_nodes
        # relation scorer for pairwise node relations: project pair -> R logits
        self.rel_scorer = nn.Sequential(nn.Linear(node_dim*2, node_dim), nn.ReLU(), nn.Linear(node_dim, 32))
        self.softlogic = SoftLogicConstraints(n_entity_types, 32)
        
        # MLM head for Stage 1 pretraining (only used if not using pre-trained encoder)
        if not use_pretrained_encoder:
            self.mlm_head = nn.Linear(self.d_model, vocab_size)
        else:
            self.mlm_head = None  # Not needed with pre-trained encoder
        
        # Concept head for Stage 2: maps entity embeddings to concept logits
        self.concept_head = nn.Linear(d_model, n_concepts)
        
        # Projection layer for node features to encoder dimension (fix dynamic layer bug)
        if node_dim != self.d_model:
            self.node_to_encoder_proj = nn.Linear(node_dim, self.d_model)
        else:
            self.node_to_encoder_proj = nn.Identity()

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Encode input tokens - used by Stage 2 trainer."""
        return self.encoder(input_ids, attention_mask)
    
    @property
    def entity_head(self):
        """Entity classification head - used by Stage 2 trainer."""
        return self.token_ent
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                 max_length: int = 128, bos_token_id: int = 1, eos_token_id: int = 2, 
                 pad_token_id: int = 0, temperature: float = 1.0, do_sample: bool = False,
                 top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
        """
        Generate a sequence autoregressively using the decoder.
        
        Args:
            input_ids: (B, L) input token IDs
            attention_mask: (B, L) attention mask for input
            max_length: Maximum length of generated sequence
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            temperature: Sampling temperature (1.0 = no scaling)
            do_sample: If True, use sampling; if False, use greedy decoding
            top_k: If set, only sample from top-k tokens
            top_p: If set, use nucleus sampling with this probability
            
        Returns:
            generated_ids: (B, T) generated token IDs
        """
        self.eval()
        device = input_ids.device
        B = input_ids.shape[0]
        
        # Encode input and prepare memory (similar to forward pass)
        with torch.no_grad():
            enc = self.encoder(input_ids, attention_mask)  # (B, L, D)
            
            # Entity token classification
            token_ent_logits = self.token_ent(enc)  # (B, L, E)
            
            # Create node features (heuristic top-K approach)
            ent_scores = token_ent_logits.max(dim=-1).values  # (B, L)
            topk_vals, topk_idx = torch.topk(ent_scores, k=min(self.max_nodes, enc.size(1)), dim=-1)
            nodes = []
            for i in range(B):
                idxs = topk_idx[i].tolist()
                reps = [enc[i, j] for j in idxs]
                reps = torch.stack(reps, dim=0)  # (N_i, D)
                # Pad to max_nodes
                pad = enc.new_zeros((self.max_nodes - reps.size(0), enc.size(-1)))
                reps_padded = torch.cat([reps, pad], dim=0)  # (max_nodes, D)
                nodes.append(reps_padded)
            node_feats = torch.stack(nodes, dim=0)  # (B, max_nodes, D)
            node_feats = self.node_proj(node_feats)  # (B, N, node_dim)
            
            # Refine node features with GNN
            node_feats_refined = self.gnn(node_feats)  # (B, N, node_dim)
            
            # Project to decoder dimension
            memory_nodes = self.node_to_encoder_proj(node_feats_refined)  # (B, N, d_model)
            
            # Initialize generation with BOS token
            generated_ids = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            
            # Autoregressive generation
            for step in range(max_length - 1):
                # Check if all sequences are finished
                if finished.all():
                    break
                
                # Get logits for current sequence
                logits = self.decoder(generated_ids, enc, memory_nodes, memory_token_mask=attention_mask)  # (B, T, vocab_size)
                
                # Get logits for next token (last position)
                next_token_logits = logits[:, -1, :] / temperature  # (B, vocab_size)
                
                # Apply top-k filtering if specified
                if top_k is not None and top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering if specified
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample or take argmax
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (B, 1)
                
                # Mark sequences that have reached EOS
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)  # (B, T+1)
        
        return generated_ids
    
    def forward(self, input_ids:torch.Tensor, attention_mask:torch.Tensor, spans:Optional[List[List[Tuple[int,int]]]]=None, y_ids:Optional[torch.Tensor]=None):
        """
        spans: optional gold entity spans for supervision, per batch item list of (s,e)
        y_ids: target decoder ids (for teacher forcing) or None for generation mode
        """
        B, L = input_ids.shape
        enc = self.encoder(input_ids, attention_mask)                # (B,L,D)
        token_pool, _ = self.token_pool(enc, attention_mask)         # (B,D)
        # Entity token classification
        token_ent_logits = self.token_ent(enc)                       # (B,L,E)
        # If gold spans given: construct node features from span pooling; else heuristic top-K
        if spans is not None:
            node_repr = span_mean_pool(enc, spans)                   # (B, node_dim_equiv -> reuse d_model)
            # project to node_dim
            node_feats = self.node_proj(node_repr).unsqueeze(1)      # (B,1,node_dim) expand later
        else:
            # heuristic: pick top-k tokens by max entity prob then use neighborhoods
            ent_scores = token_ent_logits.max(dim=-1).values        # (B,L)
            topk_vals, topk_idx = torch.topk(ent_scores, k=min(self.max_nodes, L), dim=-1)
            nodes = []
            for i in range(B):
                idxs = topk_idx[i].tolist()
                reps = [enc[i, j] for j in idxs]
                reps = torch.stack(reps, dim=0)                    # (N_i, D)
                pooled = reps.mean(dim=0)                          # coarse per-batch approach
                # create fixed N nodes by padding
                pad = enc.new_zeros((self.max_nodes - reps.size(0), enc.size(-1)))
                reps_padded = torch.cat([reps, pad], dim=0)        # (max_nodes, D)
                nodes.append(reps_padded)
            node_feats = torch.stack(nodes, dim=0)                 # (B, max_nodes, D)
            node_feats = self.node_proj(node_feats)                # (B, N, node_dim)

        # Concept mapping: map pooled token info to concept bank
        # Use token_pool which has d_model dimension (already computed above)
        concept_query = self.concept_proj(token_pool)               # (B, concept_dim)
        concept_vec, concept_probs = self.concept_bank.soft_assign(concept_query)  # (B, cdim), (B, n_concepts)

        # Relation reasoning: run GNN on nodes, produce refined node KVs
        node_feats_refined = self.gnn(node_feats)                   # (B, N, node_dim)

        # compute pairwise relation logits (sparse / pooled)
        B, N, Dn = node_feats_refined.shape
        n_rel = self.rel_scorer[-1].out_features
        pair_logits = []
        pairs_index_map = []
        for b in range(B):
            pairs = []
            indices = []
            for i in range(N):
                for j in range(i+1, N):
                    a = node_feats_refined[b, i]
                    bvec = node_feats_refined[b, j]
                    pair = torch.cat([a, bvec], dim=-1)
                    pairs.append(self.rel_scorer(pair))         # (R)
                    indices.append([i, j])
            if len(pairs)==0:
                pair_logits.append(torch.zeros((0, n_rel), device=node_feats_refined.device))
                pairs_index_map.append(torch.zeros((0, 2), dtype=torch.long, device=node_feats_refined.device))
            else:
                pair_logits.append(torch.stack(pairs, dim=0)) # variable len list per batch element
                pairs_index_map.append(torch.tensor(indices, dtype=torch.long, device=node_feats_refined.device))
        
        # Convert sparse pair logits to dense relation matrix for SoftLogicConstraints
        rel_logits_matrix = pair_logits_to_matrix(pair_logits, pairs_index_map, N, n_rel, device=node_feats_refined.device)  # (B, N, N, R)
        
        # Convert token-level entity logits to node-level entity type probabilities
        # For each node, pool entity logits from corresponding tokens
        # Since nodes are created from top-k tokens, we need to map back
        # Simplified approach: use mean pooling of entity logits for each node's token positions
        if spans is not None:
            # If we have gold spans, use them to pool entity logits per node
            node_entity_logits = []
            for i in range(B):
                node_logits = []
                for (s, e) in spans[i]:
                    # Pool entity logits over the span
                    span_logits = token_ent_logits[i, s:e+1].mean(dim=0)  # (E,)
                    node_logits.append(span_logits)
                # Pad to max_nodes
                while len(node_logits) < N:
                    node_logits.append(torch.zeros(self.n_entity_types, device=token_ent_logits.device))
                node_entity_logits.append(torch.stack(node_logits[:N], dim=0))
            node_entity_logits = torch.stack(node_entity_logits, dim=0)  # (B, N, E)
        else:
            # Use top-k token positions that were used to create nodes
            node_entity_logits = []
            ent_scores = token_ent_logits.max(dim=-1).values  # (B, L)
            topk_vals, topk_idx = torch.topk(ent_scores, k=min(self.max_nodes, L), dim=-1)
            for i in range(B):
                node_logits = []
                idxs = topk_idx[i].tolist()
                for j in idxs:
                    node_logits.append(token_ent_logits[i, j])  # (E,)
                # Pad to max_nodes
                while len(node_logits) < N:
                    node_logits.append(torch.zeros(self.n_entity_types, device=token_ent_logits.device))
                node_entity_logits.append(torch.stack(node_logits[:N], dim=0))
            node_entity_logits = torch.stack(node_entity_logits, dim=0)  # (B, N, E)
        
        # Convert to probabilities (softmax over entity types)
        node_entity_type_probs = F.softmax(node_entity_logits, dim=-1)  # (B, N, E)

        # Controller decision: pool tokens + pool nodes
        token_pool2, _ = self.token_pool(enc, attention_mask)
        node_pool = node_feats_refined.mean(dim=1)
        controller_logits = self.controller(token_pool2, node_pool)  # (B,3) -> [answer, abstain, ask]

        # Decode (teacher forcing if y_ids provided) — decoder memory is token enc + node KVs (node_feats_refined)
        # Project node_feats_refined to decoder d_model using the pre-initialized projection layer
        memory_nodes = self.node_to_encoder_proj(node_feats_refined)  # (B,N,d_model)

        if y_ids is not None:
            logits = self.decoder(y_ids, enc, memory_nodes, memory_token_mask=attention_mask)
            return {
                "logits": logits,
                "entity_logits": token_ent_logits,  # Aligned with Stage 4 expectations
                "concept_logits": concept_probs,  # Aligned with Stage 4 expectations
                "response_logit": controller_logits[:, 0:1],  # Extract answer logit for Stage 4
                "token_ent_logits": token_ent_logits,
                "concept_probs": concept_probs,
                "pair_relation_logits": pair_logits,
                "controller_logits": controller_logits,
                # SoftLogicConstraints inputs
                "node_entity_type_probs": node_entity_type_probs,  # (B, N, E)
                "rel_logits_matrix": rel_logits_matrix,  # (B, N, N, R)
            }
        else:
            return {
                "enc": enc,
                "node_feats": node_feats_refined,
                "entity_logits": token_ent_logits,  # Aligned with Stage 4 expectations
                "concept_logits": concept_probs,  # Aligned with Stage 4 expectations
                "response_logit": controller_logits[:, 0:1],  # Extract answer logit for Stage 4
                "token_ent_logits": token_ent_logits,
                "concept_probs": concept_probs,
                "pair_relation_logits": pair_logits,
                "controller_logits": controller_logits,
                # SoftLogicConstraints inputs
                "node_entity_type_probs": node_entity_type_probs,  # (B, N, E)
                "rel_logits_matrix": rel_logits_matrix,  # (B, N, N, R)
            }


def compute_losses(model_outputs:Dict, targets:Dict, lambda_weights:Dict):
    # targets: may contain token labels for entities, concept ids, relation labels (list of pairs), LM labels
    losses = {}
    # LM loss
    if "logits" in model_outputs and targets.get("lm_ids") is not None:
        logits = model_outputs["logits"]
        labels = targets["lm_ids"]
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        losses["lm"] = ce * lambda_weights.get("lm", 1.0)
    # entity token loss
    if targets.get("entity_token_labels") is not None:
        ent_logits = model_outputs["token_ent_logits"]  # (B,L,E)
        ent_labels = targets["entity_token_labels"]
        ent_loss = F.cross_entropy(ent_logits.view(-1, ent_logits.size(-1)), ent_labels.view(-1), ignore_index=-100)
        losses["entity"] = ent_loss * lambda_weights.get("entity", 1.0)
    # concept loss (if gold concept id available)
    if targets.get("concept_id") is not None:
        cp = model_outputs["concept_probs"]  # (B, n_concepts)
        cid = targets["concept_id"]
        c_loss = F.cross_entropy(cp.log(), cid)
        losses["concept"] = c_loss * lambda_weights.get("concept", 1.0)
    # relation loss: requires mapping from target pairs to pair_logits structure
    # abstention/controller loss: cross-entropy against provided action label
    if targets.get("controller_label") is not None:
        cl = model_outputs["controller_logits"]
        lab = targets["controller_label"]
        losses["controller"] = F.cross_entropy(cl, lab) * lambda_weights.get("controller", 1.0)
    # soft logic: if implemented, compute and add penalty
    # aggregate
    total = sum(losses.values()) if len(losses)>0 else torch.tensor(0.0)
    return total, losses

# ----------------------------
# Remarks and next steps (operational)
# ----------------------------
# - Use staged training: pretrain token encoder + token entity classifier + concept mapping using supervised spans and concept labels.
# - Pretrain GNN on relation-labeled mini datasets; integrate later.
# - Use contrastive objective for concept bank alignment if only weak labels: anchor queries vs positive concept nodes, InfoNCE.
# - Implement mapping from pair_logits list -> adjacency matrix for relation supervision and soft-logic constraints.
# - For production-scale experiments, replace simple transformer blocks with optimized implementations and use mixed precision + distributed training.

# End of prototype.
