"""Cross-attention graph injection (injection="xattn") on a tiny offline
backbone: init equals the backbone, prompt positions never read the nodes,
generate agrees with the training pass, and training reaches every part."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")

from model.injection import NodeCrossAttention  # noqa: E402
from model.neurosymbolic_causal import NeuroSymbolicCausalLM  # noqa: E402

VOCAB = 2048
HIDDEN = 64


def tiny_backbone(layers=4):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=HIDDEN * 2,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512))


def make_model(injection="xattn", gate=None, seed=0, **kw):
    backbone = tiny_backbone()
    torch.manual_seed(seed)
    model = NeuroSymbolicCausalLM(
        backbone=backbone, tap_layer_ratio=0.5, n_entity_types=8,
        n_relations=16, n_concepts=16, concept_dim=16, node_dim=32,
        max_nodes=4, adapter_layers=1, adapter_heads=2, injection=injection,
        xattn_every=1, xattn_heads=4, **kw)
    if gate is not None:
        with torch.no_grad():
            for g in model.injection_gates():
                g.fill_(gate)
    return model


def prompts(B=2, P=10):
    torch.manual_seed(1)
    ids = torch.randint(2, VOCAB, (B, P))
    mask = torch.ones(B, P, dtype=torch.long)
    mask[1, P - 3:] = 0  # the second prompt is shorter (right padding)
    ids[1, P - 3:] = 0
    return ids, mask


def test_layers_after_the_tap_only():
    model = make_model()
    assert model.tap_layer == 2
    assert model.xattn_layer_ids == [2, 3]
    assert not hasattr(model, "injector")


def test_init_is_exactly_the_backbone():
    model = make_model().eval()
    ids, mask = prompts()
    y = torch.randint(2, VOCAB, (2, 5))
    with torch.no_grad():
        on = model(ids, mask, y_ids=y)["logits"]
        off = model(ids, mask, y_ids=y, use_soft_nodes=False)["logits"]
    assert torch.equal(on, off)


def test_prompt_positions_never_read_the_nodes():
    model = make_model(gate=1.0).eval()
    ids, mask = prompts()
    y = torch.randint(2, VOCAB, (2, 5))
    seen = []
    hook = model._decoder().norm.register_forward_hook(
        lambda m, a, out: seen.append(out.detach().clone()))
    with torch.no_grad():
        enc_on = model(ids, mask, y_ids=y)["enc"]
        enc_off = model(ids, mask, y_ids=y, use_soft_nodes=False)["enc"]
    hook.remove()
    # seen: read on, write on, read off, write off
    write_on, write_off = seen[1], seen[3]
    assert torch.equal(enc_on, enc_off)  # the tapped read pass is the same
    for i, p in enumerate(mask.sum(1).tolist()):
        assert torch.equal(write_on[i, :p - 1], write_off[i, :p - 1])
        # The last prompt token predicts the first answer token: it reads
        assert not torch.allclose(write_on[i, p - 1], write_off[i, p - 1])
        assert not torch.allclose(write_on[i, p:p + 5], write_off[i, p:p + 5])


def test_generate_agrees_with_the_training_pass():
    """Same logits from generate (left-packed, KV cache, one position per
    step) and from the training pass (right-padded, teacher forcing)."""
    model = make_model(gate=1.0).eval()
    ids, mask = prompts()
    out = model.generate(ids, mask, max_length=4, do_sample=False,
                         pad_token_id=0, eos_token_id=None,
                         output_logits=True, return_dict_in_generate=True)
    gen = out.sequences
    assert gen.shape == (2, 4)
    gen_logits = torch.stack(out.logits, dim=1)
    with torch.no_grad():
        logits = model(ids, mask, y_ids=gen)["logits"]
        off = model(ids, mask, y_ids=gen, use_soft_nodes=False)["logits"]
    assert torch.allclose(gen_logits, logits, atol=1e-4)
    assert not torch.allclose(logits, off, atol=1e-3)
    assert model._xattn_ctx is None  # generate cleans up


def test_lm_loss_trains_gate_then_attention():
    model = make_model().train()
    ids, mask = prompts()
    y = torch.randint(2, VOCAB, (2, 5))

    def grads():
        model.zero_grad(set_to_none=True)
        out = model(ids, mask, y_ids=y)["logits"]
        torch.nn.functional.cross_entropy(out.reshape(-1, VOCAB), y.reshape(-1)).backward()
        return model.xattn[0]

    x = grads()
    # tanh(0) = 0: only the gates get a gradient at first (as in Flamingo)
    assert x.gate.grad is not None and x.gate.grad.abs() > 0
    assert x.q.weight.grad.abs().sum() == 0
    with torch.no_grad():
        for g in model.injection_gates():
            g.fill_(0.5)
    x = grads()
    assert x.q.weight.grad.abs().sum() > 0 and x.kv.weight.grad.abs().sum() > 0
    assert model.node_proj.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.gnn.parameters())


def test_node_dropout_one_is_no_graph():
    model = make_model(gate=1.0, node_dropout=1.0)
    ids, mask = prompts()
    y = torch.randint(2, VOCAB, (2, 5))
    model.train()
    with torch.no_grad():
        dropped = model(ids, mask, y_ids=y)["logits"]
        off = model(ids, mask, y_ids=y, use_soft_nodes=False)["logits"]
    model.eval()  # dropout is for training only
    with torch.no_grad():
        on = model(ids, mask, y_ids=y)["logits"]
    # No sample reads the nodes, so the backbone runs alone
    assert torch.allclose(dropped, off)
    assert not torch.allclose(on, off)


def test_gradient_checkpointing_gives_the_same_gradients():
    ids, mask = prompts()
    y = torch.randint(2, VOCAB, (2, 5))
    result = []
    for ckpt in (False, True):
        model = make_model(gate=0.5, gradient_checkpointing=ckpt)
        model.train()
        assert model.backbone.is_gradient_checkpointing == ckpt
        # Same dropout masks in both runs
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
        out = model(ids, mask, y_ids=y)["logits"]
        torch.nn.functional.cross_entropy(out.reshape(-1, VOCAB), y.reshape(-1)).backward()
        result.append([p.grad.clone() for p in model.xattn.parameters()])
    for a, b in zip(*result):
        assert torch.allclose(a, b, atol=1e-6)


def test_cross_attention_masks():
    torch.manual_seed(0)
    x = NodeCrossAttention(HIDDEN, 32, n_heads=4)
    with torch.no_grad():
        x.gate.fill_(1.0)
    hidden = torch.randn(2, 6, HIDDEN)
    nodes = torch.randn(2, 3, 32)
    valid = torch.tensor([[True, False, True], [False, False, False]])
    pos = torch.zeros(2, 6, 1)
    pos[:, 4:] = 1
    out = x(hidden, nodes, valid, pos)
    assert torch.isfinite(out).all()
    assert out[:, :4].abs().sum() == 0  # masked positions
    assert out[1].abs().sum() == 0  # a prompt with no nodes
    assert out[0, 4:].abs().sum() > 0
    # An empty slot changes nothing
    nodes2 = nodes.clone()
    nodes2[0, 1] = 100.0
    assert torch.allclose(x(hidden, nodes2, valid, pos), out)


def test_init_from_a_prefix_checkpoint(capsys):
    import train_causal
    prefix = make_model(injection="prefix", seed=1)
    model = make_model(seed=2)
    train_causal.load_init_weights(model, {"model_state_dict": prefix.state_dict()})
    assert "Injection mode differs" in capsys.readouterr().out
    assert torch.equal(model.node_proj.weight, prefix.node_proj.weight)
    assert all(float(g) == 0.0 for g in model.injection_gates())


def test_stage_setup_and_optimizer_groups():
    import train_causal
    model = make_model()
    model._backbone_trainable = set()
    train_causal.set_stage_trainable(model, symbolic=False, backbone=False, injector=True)
    assert all(p.requires_grad for p in model.xattn.parameters())
    opt = train_causal.make_optimizer(model, lr=1e-4, weight_decay=0.01, gate_lr=1e-3)
    gate_group = [g for g in opt.param_groups if g["lr"] == 1e-3]
    assert len(gate_group) == 1 and len(gate_group[0]["params"]) == len(model.xattn)
    train_causal.set_stage_trainable(model, symbolic=True, backbone=False, injector=False)
    assert not any(p.requires_grad for p in model.xattn.parameters())


def test_xattn_generate_needs_the_cache():
    model = make_model(gate=1.0).eval()
    ids, mask = prompts()
    with pytest.raises(ValueError, match="use_cache"):
        model.generate(ids, mask, max_length=2, use_cache=False, pad_token_id=0)
    assert model._xattn_ctx is None


@pytest.mark.parametrize("injection", ["prefix", "xattn"])
def test_shuffle_nodes_swaps_only_the_injected_content(injection):
    model = make_model(injection=injection, gate=1.0).eval()
    ids, mask = prompts()
    y = torch.randint(2, VOCAB, (2, 5))
    with torch.no_grad():
        right = model(ids, mask, y_ids=y)
        model.shuffle_nodes = True
        wrong = model(ids, mask, y_ids=y)
        model.shuffle_nodes = False
    assert not torch.allclose(right["logits"], wrong["logits"])
    # The heads still see each sample's own nodes
    assert torch.equal(right["rel_logits_matrix"], wrong["rel_logits_matrix"])
    assert torch.equal(right["node_feats"], wrong["node_feats"])


# --- xattn_source="adapter" and the reader stage (Round 7 Step 1e) ---------

def test_adapter_source_reads_the_wide_states():
    model = make_model(xattn_source="adapter").eval()
    assert model.xattn[0].kv.in_features == HIDDEN
    ids, mask = prompts()
    y = torch.randint(2, VOCAB, (2, 5))
    with torch.no_grad():
        on = model(ids, mask, y_ids=y)["logits"]
        off = model(ids, mask, y_ids=y, use_soft_nodes=False)["logits"]
    assert torch.equal(on, off)  # gate 0: exactly the backbone
    model = make_model(xattn_source="adapter", gate=1.0).train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    out = model(ids, mask, y_ids=y)["logits"]
    torch.nn.functional.cross_entropy(out.reshape(-1, VOCAB), y.reshape(-1)).backward()
    # The LM loss reaches the cross-attention, not node_proj / GNN
    assert model.xattn[0].kv.weight.grad.abs().sum() > 0
    assert model.node_proj.weight.grad is None or model.node_proj.weight.grad.abs().sum() == 0
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.gnn.parameters())


def test_adapter_source_generate_agrees_with_training_pass():
    model = make_model(xattn_source="adapter", gate=1.0).eval()
    ids, mask = prompts()
    out = model.generate(ids, mask, max_length=4, do_sample=False, pad_token_id=0,
                         eos_token_id=None, output_logits=True, return_dict_in_generate=True)
    gen_logits = torch.stack(out.logits, dim=1)
    with torch.no_grad():
        logits = model(ids, mask, y_ids=out.sequences)["logits"]
        node = make_model(gate=1.0).eval()(ids, mask, y_ids=out.sequences)["logits"]
    assert torch.allclose(gen_logits, logits, atol=1e-4)
    assert not torch.allclose(logits, node, atol=1e-3)


@pytest.mark.parametrize("use_spans", [False, True])
def test_node_states_sit_at_the_node_positions(use_spans):
    """node_proj of the selected adapter states = the node features."""
    model = make_model().eval()
    ids, mask = prompts()
    spans = [[(0, 1), (-1, -1), (3, 5)], [(2, 2)]] if use_spans else None
    with torch.no_grad():
        enc = model._read(ids, mask)
        logits = model.token_ent(enc)
        feats = model._extract_node_features(enc, logits, mask, spans)
        states = model._select_node_states(enc, logits, mask, spans)
        empty = states.abs().sum(-1, keepdim=True) == 0
        projected = model.node_proj(states).masked_fill(empty, 0.0)
    assert states.shape == (2, model.max_nodes, HIDDEN)
    assert torch.allclose(projected, feats, atol=1e-6)
    assert torch.equal(empty[..., 0], feats.abs().sum(-1) == 0)


def test_init_full_and_fresh_injection(capsys):
    import train_causal
    trained = make_model(seed=1, gate=0.5)
    with torch.no_grad():
        for p in trained.backbone.parameters():
            p.add_(0.01)
    ckpt = {"model_state_dict": trained.state_dict()}
    model = make_model(seed=2)
    train_causal.load_init_weights(model, ckpt, include_backbone=True, fresh_injection=True)
    assert "fresh injection" in capsys.readouterr().out
    first = next(iter(trained.backbone.state_dict()))
    assert torch.equal(model.backbone.state_dict()[first], trained.backbone.state_dict()[first])
    assert torch.equal(model.node_proj.weight, trained.node_proj.weight)
    assert all(float(g) == 0.0 for g in model.injection_gates())  # fresh
    assert not torch.equal(model.xattn[0].q.weight, trained.xattn[0].q.weight)
    # Default: no backbone, injection loaded (same mode)
    model = make_model(seed=2)
    train_causal.load_init_weights(model, ckpt)
    assert not torch.equal(model.backbone.state_dict()[first], trained.backbone.state_dict()[first])
    assert torch.equal(model.xattn[0].q.weight, trained.xattn[0].q.weight)


def test_reader_stage_trains_only_the_cross_attention():
    import train_causal
    model = make_model(xattn_source="adapter")
    model._backbone_trainable = {n for n, _ in model.backbone.named_parameters()}
    train_causal.set_stage_trainable(model, symbolic=False, backbone=False, injector=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("xattn.") for n in trainable)
