from __future__ import annotations

import hashlib
import struct

import pytest
import torch
from transformers import AutoModelForCausalLM

import custom_models  # noqa: F401
from custom_models.mixerloop import MixerLoopConfig
from flame.utils.convert_dcp_to_hf import (
    CHECKPOINT_MAGIC,
    CHECKPOINT_VERSION,
    export_q8_checkpoint,
    export_tokenizer_binary,
    quantize_q80
)


def deployment_config(loop_count: int):
    return MixerLoopConfig(
        hidden_size=256,
        num_hidden_layers=1,
        num_heads=8,
        head_dim=32,
        intermediate_size=768,
        loop_count=loop_count,
        max_position_embeddings=1024,
        vocab_size=32000,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        use_cache=False,
    )


def expected_q8_size(config) -> int:
    def q8(rows, cols):
        return rows * cols + rows * (cols // 32) * 4

    dim = config.hidden_size
    hidden = config.intermediate_size
    head = config.head_dim
    heads = config.num_heads
    conv = config.conv_size
    per_layer = (
        dim * 4
        + 4 * q8(dim, dim)
        + 2 * heads * dim * 4
        + 3 * dim * conv * 4
        + 2 * heads * 4
        + head * 4
        + q8(dim, dim)
        + dim * 4
        + 2 * q8(hidden, dim)
        + q8(dim, hidden)
    )
    residual = config.loop_count * dim * 4 if config.use_residual else 0
    return 256 + q8(config.vocab_size, dim) + config.num_hidden_layers * per_layer + dim * 4 + residual


def test_q80_zero_and_range():
    quantized, scales = quantize_q80(torch.zeros(2, 64))
    assert quantized.min() >= -127
    assert quantized.max() <= 127
    assert torch.equal(scales, torch.ones_like(scales))


def test_q80_rejects_bad_input_dimension():
    with pytest.raises(ValueError, match="not divisible"):
        quantize_q80(torch.zeros(2, 33))


@pytest.mark.parametrize("loop_count", [1, 4])
def test_q8_header_size_and_residual(loop_count, tmp_path):
    config = deployment_config(loop_count)
    model = AutoModelForCausalLM.from_config(config)
    with torch.no_grad():
        model.model.residual_weight.copy_(
            torch.arange(loop_count * config.hidden_size).reshape(loop_count, -1)
        )
    output = export_q8_checkpoint(model, tmp_path / "model.q8.bin")
    data = output.read_bytes()

    header = struct.unpack("<I15i", data[:64])
    assert header[0] == CHECKPOINT_MAGIC
    assert header[1] == CHECKPOINT_VERSION
    assert header[2:10] == (256, 768, 1, 8, 32, 32, 4, 32000)
    assert header[12:15] == (loop_count, 1, 32)
    assert len(data) == expected_q8_size(config)
    expected_residual = model.model.residual_weight.detach().float().numpy().tobytes()
    assert data.endswith(expected_residual)


def test_tokenizer_binary_is_stable(tmp_path):
    output = export_tokenizer_binary("assets/tokenizer/tokenizer.model", tmp_path / "tokenizer.bin")
    assert hashlib.sha256(output.read_bytes()).hexdigest() == (
        "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
    )
