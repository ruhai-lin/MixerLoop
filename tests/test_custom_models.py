from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import custom_models  # noqa: F401
from custom_models.ffnloop import FFNLoopConfig
from custom_models.ffnloop.layers import FFNLoopBlock
from custom_models.fullloop import FullLoopConfig
from custom_models.mixerloop import MixerLoopConfig


def tiny_config(config_cls):
    return config_cls(
        hidden_size=32,
        num_hidden_layers=2,
        num_heads=4,
        head_dim=8,
        intermediate_size=64,
        loop_count=4,
        vocab_size=128,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        use_cache=False,
    )


def initialize_like_flame(config):
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config)
        model.apply(lambda module: setattr(module, '_is_hf_initialized', False))
    model.to_empty(device='cpu')
    torch.manual_seed(1234)
    with torch.no_grad():
        model.post_init()
    return model


@pytest.mark.parametrize('config_cls', [FullLoopConfig, MixerLoopConfig, FFNLoopConfig])
def test_meta_materialization_initializes_gdn_and_loop_weights(config_cls):
    config = tiny_config(config_cls)
    model = initialize_like_flame(config)
    base = model.model

    assert torch.count_nonzero(base.residual_weight) == 0
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())

    for layer in base.layers:
        mixer = layer.mixer
        decay = mixer.A_log.float().exp()
        step = F.softplus(mixer.dt_bias.float())
        assert torch.all((decay > 0) & (decay <= 16))
        assert torch.all((step >= 1e-3) & (step <= 0.1))


@pytest.mark.parametrize('config_cls', [FullLoopConfig, MixerLoopConfig])
def test_recurrent_mixer_projection_scaling_survives_meta_materialization(config_cls):
    config = tiny_config(config_cls)
    model = initialize_like_flame(config)
    expected = config.initializer_range / math.sqrt(
        2 * config.num_hidden_layers * config.loop_count
    )
    observed = float(model.model.layers[0].mixer.o_proj.weight.detach().std())
    assert observed == pytest.approx(expected, rel=0.2)


def test_recurrent_ffn_projection_scaling_survives_meta_materialization():
    config = tiny_config(FFNLoopConfig)
    model = initialize_like_flame(config)
    expected = config.initializer_range / math.sqrt(
        2 * config.num_hidden_layers * config.loop_count
    )
    observed = float(model.model.layers[0].ffn.down_proj.weight.detach().std())
    assert observed == pytest.approx(expected, rel=0.2)


class CountingMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, **kwargs):
        self.calls += 1
        return torch.zeros_like(hidden_states), None, None


class CountingFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        return torch.ones_like(hidden_states)


def test_ffn_loop_block_allocates_recurrence_to_ffn_only():
    config = tiny_config(FFNLoopConfig)
    block = FFNLoopBlock(config, layer_idx=0)
    block.attn_norm = nn.Identity()
    block.ffn_norm = nn.Identity()
    block.mixer = CountingMixer()
    block.ffn = CountingFFN()

    output = block(torch.zeros(1, 3, config.hidden_size))

    assert block.mixer.calls == 1
    assert block.ffn.calls == config.loop_count
    torch.testing.assert_close(output, torch.full_like(output, config.loop_count))
