from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

import custom_models  # noqa: F401
from custom_models.mixerloop import MixerLoopConfig
from custom_models.mixerloop.layers import MixerLoopBlock


@pytest.mark.parametrize(
    ("config_path", "expected_parameters"),
    [
        ("configs/mixerloop_15m.json", 15_594_112),
        ("configs/mixerloop_105m.json", 105_031_680),
        ("configs/mixerloop_328m.json", 327_969_664),
    ],
)
def test_shipped_config_parameter_counts(config_path, expected_parameters):
    config = AutoConfig.from_pretrained(config_path)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_parameters


def tiny_config(loop_count: int = 4):
    return MixerLoopConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_heads=4,
        head_dim=8,
        intermediate_size=64,
        loop_count=loop_count,
        vocab_size=128,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        use_cache=False,
    )


def initialize_like_flame(config):
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
        model.apply(lambda module: setattr(module, "_is_hf_initialized", False))
    model.to_empty(device="cpu")
    torch.manual_seed(1234)
    with torch.no_grad():
        model.post_init()
    return model


@pytest.mark.parametrize("loop_count", [1, 4])
def test_meta_materialization_initializes_gdn(loop_count):
    model = initialize_like_flame(tiny_config(loop_count))
    base = model.model

    assert not hasattr(base, "residual_weight")
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    for layer in base.layers:
        mixer = layer.mixer
        decay = mixer.A_log.float().exp()
        step = F.softplus(mixer.dt_bias.float())
        assert torch.all((decay > 0) & (decay <= 16))
        assert torch.all((step >= 1e-3) & (step <= 0.1))


@pytest.mark.parametrize("loop_count", [1, 4])
def test_recurrent_projection_scaling(loop_count):
    config = tiny_config(loop_count)
    model = initialize_like_flame(config)
    expected = config.initializer_range / math.sqrt(
        2 * config.num_hidden_layers * config.loop_count
    )
    observed = float(model.model.layers[0].mixer.o_proj.weight.detach().std())
    assert observed == pytest.approx(expected, rel=0.2)


class CountingMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, **kwargs):
        self.calls += 1
        return torch.ones_like(hidden_states), None, None


class CountingFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        return torch.ones_like(hidden_states)


@pytest.mark.parametrize("loop_count", [1, 4])
def test_mixerloop_repeats_only_the_mixer(loop_count):
    config = tiny_config(loop_count)
    block = MixerLoopBlock(config, layer_idx=0)
    block.attn_norm = nn.Identity()
    block.ffn_norm = nn.Identity()
    block.mixer = CountingMixer()
    block.ffn = CountingFFN()

    output = block(torch.zeros(1, 3, config.hidden_size))

    assert block.mixer.calls == loop_count
    assert block.ffn.calls == 1
    torch.testing.assert_close(output, torch.full_like(output, loop_count + 1))


def test_invalid_loop_count_is_rejected():
    with pytest.raises(ValueError, match="loop_count"):
        tiny_config(0)
