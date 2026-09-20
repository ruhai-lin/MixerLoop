import struct

import pytest
import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoModelForCausalLM, AutoTokenizer
from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig

from custom_models.mixerloop import MixerLoopConfig
from custom_models.fullloop import FullLoopConfig
from flame.utils.convert_dcp_to_hf import save_pretrained
from hardware.quantization import (
    HEADER, HEADER_BYTES, HEADER_FIELDS, classify, export_checkpoint,
    quantize_q80, read_bundle, tensor_specs, write_bundle,
)


@pytest.mark.parametrize("config_class", [GatedDeltaNetConfig, MixerLoopConfig, FullLoopConfig])
def test_hf_export_roundtrip(tmp_path, config_class):
    config = config_class(
        hidden_size=32, num_hidden_layers=1, num_heads=1, head_dim=32,
        expand_v=2, intermediate_size=64, vocab_size=32000,
        fuse_norm=False, fuse_swiglu=False, fuse_cross_entropy=False,
        tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(config)
    if hasattr(model.model, "residual_weight"):
        with torch.no_grad():
            model.model.residual_weight.fill_(0.125)
    config.save_pretrained(tmp_path)
    dcp.save({"model": model.state_dict()}, checkpoint_id=tmp_path / "checkpoint/step-1")
    save_pretrained(str(tmp_path), 1, str(tmp_path), "assets/tokenizer")
    restored = AutoModelForCausalLM.from_pretrained(tmp_path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
    assert len(AutoTokenizer.from_pretrained(tmp_path)) == 32000
    assert (tmp_path / "model.safetensors").is_file()
    assert not (tmp_path / "tokenizer.bin").exists()
    assert not list(tmp_path.glob("*q8.bin"))


def test_q8_known_groups():
    w = torch.zeros(16, 64)
    w[:, :6] = torch.tensor([127., -127., 0.5, 1.5, 2.5, -1.5])
    q, s = quantize_q80(w)
    assert q.dtype == torch.int8 and s.dtype == torch.float32
    assert q[0, :6].tolist() == [127, -127, 0, 2, 2, -2]
    assert s.eq(1).all()  # includes all-zero groups
    assert q[:, 32:].eq(0).all()


@pytest.fixture(params=[GatedDeltaNetConfig, MixerLoopConfig])
def ptq_model(request):
    torch.manual_seed(1337)
    config = request.param(
        hidden_size=32, num_hidden_layers=1, num_heads=1, head_dim=32,
        expand_v=2, intermediate_size=64, vocab_size=64,
        fuse_norm=False, fuse_swiglu=False, fuse_cross_entropy=False,
        tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(config)
    if hasattr(model.model, "residual_weight"):
        with torch.no_grad():
            model.model.residual_weight.fill_(0.125)
    return model


def test_q8_binary_contract(tmp_path, ptq_model):
    header, tensors = classify(ptq_model)
    path = tmp_path / "model.bin"
    write_bundle(path, header, tensors)
    h, restored = read_bundle(path)
    assert h == header
    expected_size = HEADER_BYTES
    assert len(restored) == (21 if header["residual"] else 20)
    for name, shape, matrix in tensor_specs(header):
        count = int(np.prod(shape))
        expected_size += count + count // 32 * 4 if matrix else count * 4
        if matrix:
            q, s = quantize_q80(tensors[name])
            np.testing.assert_array_equal(restored[name][0], q.numpy())
            np.testing.assert_array_equal(restored[name][1], s.numpy())
        else:
            np.testing.assert_array_equal(restored[name], tensors[name].numpy())
    assert path.stat().st_size == expected_size
    raw = path.read_bytes()
    assert raw[HEADER.size:HEADER_BYTES] == bytes(HEADER_BYTES - HEADER.size)
    # Independently locate the embedding and first side tensor; LM is not duplicated.
    embedding_bytes = 64 * 32 + 64 * 4
    np.testing.assert_array_equal(
        np.frombuffer(raw, dtype="<f4", count=32, offset=HEADER_BYTES + embedding_bytes),
        tensors["model.layers.0.attn_norm.weight"].numpy())
    if header["residual"]:
        assert np.frombuffer(raw[-4 * 32 * 4:], dtype="<f4").tolist() == [0.125] * 128
    else:
        assert header["loop_count"] == 1
    write_bundle(path, header, tensors)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("sharded", [False, True])
def test_q8_hf_loading_and_overwrite(tmp_path, ptq_model, sharded):
    ptq_model.save_pretrained(tmp_path, safe_serialization=True,
                             max_shard_size="5KB" if sharded else "1GB")
    original = {p.name: p.read_bytes() for p in tmp_path.glob("*.safetensors")}
    output = export_checkpoint(tmp_path)
    assert output.name == f"{tmp_path.name}-q8.bin"
    raw = output.read_bytes()
    with pytest.raises(FileExistsError):
        export_checkpoint(tmp_path)
    export_checkpoint(tmp_path, force=True)
    assert output.read_bytes() == raw
    assert original == {p.name: p.read_bytes() for p in tmp_path.glob("*.safetensors")}


def test_q8_reject_unknown_or_untied(ptq_model):
    ptq_model.register_buffer("unknown", torch.zeros(1))
    with pytest.raises(ValueError, match="unclassified"):
        classify(ptq_model)
    del ptq_model.unknown
    ptq_model.lm_head.weight = torch.nn.Parameter(ptq_model.lm_head.weight.clone())
    with pytest.raises(ValueError, match="tied"):
        classify(ptq_model)


def test_q8_reject_corrupt_bundle(tmp_path, ptq_model):
    h, tensors = classify(ptq_model)
    path = tmp_path / "bad.bin"
    write_bundle(path, h, tensors)
    original = path.read_bytes()
    for raw in (original[:-1], original + b"x", original[:100],
                original[:255] + b"x" + original[256:]):
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            read_bundle(path)
    for field, value in (("version", 2), ("group_size", 16), ("state_dtype", 3),
                         ("residual", 1 - h["residual"]), ("loop_count", 0)):
        raw = bytearray(original)
        offset = HEADER_FIELDS.index(field) * 4
        raw[offset:offset + 4] = value.to_bytes(4, "little")
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            read_bundle(path)
    for offset, value in (
        (HEADER_BYTES, b"\x80"),  # symmetric INT8 excludes -128
        (HEADER_BYTES + 64 * 32, struct.pack("<f", 0)),
        (HEADER_BYTES + 64 * 32, struct.pack("<f", float("nan"))),
        (HEADER_BYTES + 64 * 36, struct.pack("<f", float("inf"))),
    ):
        raw = bytearray(original)
        raw[offset:offset + len(value)] = value
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            read_bundle(path)
