import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from custom_models.fullloop import FullLoopConfig
from custom_models.training import ParameterGroupedOptimizersContainer


ROOT = Path(__file__).resolve().parents[1]


def tiny_config(**kwargs):
    return FullLoopConfig(
        hidden_size=128, num_hidden_layers=2, num_heads=3, head_dim=32,
        expand_v=2, intermediate_size=352, vocab_size=256,
        fuse_norm=False, fuse_swiglu=False, fuse_cross_entropy=False,
        use_cache=False, **kwargs,
    )


def test_lt2_initialization_after_meta_materialization():
    config = tiny_config()
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config)
        model.apply(lambda module: setattr(module, '_is_hf_initialized', False))
    model.to_empty(device='cpu')
    with torch.no_grad():
        model.post_init()

    assert config.initializer_range == config.hidden_size ** -0.5
    assert torch.count_nonzero(model.model.residual_weight) == 0
    assert model.lm_head.weight is model.model.embeddings.weight
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            weights = module.weight.detach()
            assert weights.abs().max() <= 3 * config.initializer_range
            assert float(weights.std()) == pytest.approx(config.initializer_range, rel=0.1)
        elif isinstance(module, nn.Conv1d):
            assert module.weight.detach().abs().max() <= 1 / math.sqrt(config.conv_size)
    for layer in model.model.layers:
        assert torch.all((layer.mixer.A_log.exp() > 0) & (layer.mixer.A_log.exp() <= 16))
        dt = F.softplus(layer.mixer.dt_bias)
        assert torch.all((dt >= 0.001) & (dt <= 0.1))


class TracedMixer(nn.Module):
    def __init__(self, layer_idx, calls):
        super().__init__()
        self.layer_idx, self.calls = layer_idx, calls

    def forward(self, h, **kwargs):
        self.calls.append(('mixer', self.layer_idx))
        assert kwargs['use_cache'] is False
        return (self.layer_idx + 1) * h + 1, None, None


class TracedFFN(nn.Module):
    def __init__(self, layer_idx, calls):
        super().__init__()
        self.layer_idx, self.calls = layer_idx, calls

    def forward(self, h):
        self.calls.append(('ffn', self.layer_idx))
        return h * 0.25


@pytest.mark.parametrize('loop_count', [1, 4])
def test_full_stack_order_and_iteration_residual(loop_count):
    model = AutoModelForCausalLM.from_config(tiny_config(loop_count=loop_count)).model
    model.norm = nn.Identity()
    calls = []
    for i, layer in enumerate(model.layers):
        layer.attn_norm = layer.ffn_norm = nn.Identity()
        layer.mixer = TracedMixer(i, calls)
        layer.ffn = TracedFFN(i, calls)
    with torch.no_grad():
        for t in range(loop_count):
            model.residual_weight[t].fill_(0.1 * (t + 1))

    inputs = torch.ones(1, 3, model.config.hidden_size)
    actual = model(inputs_embeds=inputs).last_hidden_state
    expected = inputs
    for t in range(loop_count):
        h_input = expected
        for i in range(len(model.layers)):
            expected = (expected + (i + 1) * expected + 1) * 1.25
        expected = expected + 0.1 * (t + 1) * h_input
    torch.testing.assert_close(actual, expected)
    assert calls == [('mixer', 0), ('ffn', 0), ('mixer', 1), ('ffn', 1)] * loop_count
    assert [name for name, _ in model.named_parameters() if 'residual_weight' in name] == ['residual_weight']


@pytest.mark.parametrize('size, expected', [
    ('13m', 12_896_380), ('100m', 100_447_266),
    ('600m', 445_775_120), ('1p6b', 1_578_953_312),
])
def test_fullloop_configs(size, expected):
    config = AutoConfig.from_pretrained(ROOT / 'configs' / f'fullloop_{size}.json')
    matched = AutoConfig.from_pretrained(ROOT / 'configs' / f'mixerloop_{size}.json')
    for field in ('hidden_size', 'intermediate_size', 'head_dim', 'num_heads', 'num_hidden_layers',
                  'expand_v', 'vocab_size', 'max_position_embeddings', 'tie_word_embeddings'):
        assert getattr(config, field) == getattr(matched, field)
    assert config.loop_count == 4
    assert config.initializer_range == pytest.approx(config.hidden_size ** -0.5)
    # LT2 leaves the GDN gate norm at its default epsilon, distinct from outer norms.
    assert config.mixer_norm_eps == 1e-5
    assert config.allow_neg_eigval is (size == '1p6b')
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config)
    assert sum(p.numel() for p in model.parameters()) == expected
    assert model.model.layers[0].mixer.o_norm.eps == 1e-5
    assert model.model.layers[0].mixer.allow_neg_eigval == config.allow_neg_eigval


def test_fullloop_optimizer_contract():
    model = AutoModelForCausalLM.from_config(tiny_config())
    optimizers = ParameterGroupedOptimizersContainer(
        [model], torch.optim.AdamW,
        {'lr': 5e-4, 'betas': (0.9, 0.95), 'eps': 1e-8, 'weight_decay': 0.1},
    )
    for group in optimizers.optimizers[0].param_groups:
        ids = {id(p) for p in group['params']}
        for name, p in model.named_parameters():
            if id(p) in ids:
                no_decay = p.dim() < 2 or name.endswith(('A_log', 'dt_bias'))
                assert group['weight_decay'] == (0.0 if no_decay else 0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GDN kernels require CUDA')
def test_fullloop_forward_backward_and_gradient_checkpointing():
    config = tiny_config()
    model = AutoModelForCausalLM.from_config(config).cuda().train()
    tokens = torch.randint(0, config.vocab_size, (2, 128), device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(tokens, labels=tokens).loss
    loss.backward()
    assert torch.isfinite(loss)
    grads = {name: p.grad.clone() for name, p in model.named_parameters()}
    assert grads['model.residual_weight'].abs().sum() > 0
    assert all(torch.isfinite(grad).all() for grad in grads.values())

    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    with torch.autocast('cuda', dtype=torch.bfloat16):
        checkpointed_loss = model(tokens, labels=tokens).loss
    checkpointed_loss.backward()
    torch.testing.assert_close(checkpointed_loss, loss)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, grads[name])
