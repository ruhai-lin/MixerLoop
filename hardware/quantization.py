"""Local HF -> GDNe v3. Q8_0 group-32 weights; FP32 scales and side parameters.

The 256-byte little-endian header is HEADER_FIELDS uint32s, norm_eps float32,
then zero padding. Each matrix is row-major int8 followed by row-major FP32
scales [rows, cols/32]. Side tensors are contiguous FP32. No tensor padding.
tensor_specs defines the entire payload order; device stream packing stays in
weight.cpp. Version 3 adds explicit model/residual metadata, not new PTQ math.
"""

import argparse
import hashlib
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fla.models  # noqa: E402,F401
import custom_models  # noqa: E402,F401
from fla.layers import GatedDeltaNet  # noqa: E402
from fla.modules import GatedMLP  # noqa: E402

MAGIC, VERSION, HEADER_BYTES, GROUP_SIZE = 0x47444E65, 3, 256, 32
# Dtype IDs: 1=int8, 2=float32. Model IDs: 1=GDN, 2=MixerLoop.
HEADER_FIELDS = (
    "magic", "version", "dim", "hidden_dim", "n_layers", "num_heads",
    "head_k_dim", "head_v_dim", "conv_size", "vocab_size", "seq_len",
    "shared_classifier", "loop_count", "group_size", "model_type",
    "residual", "weight_dtype", "scale_dtype", "side_dtype", "state_dtype",
)
HEADER = struct.Struct("<" + "I" * len(HEADER_FIELDS) + "f")


def validate_header(h):
    expected = dict(magic=MAGIC, version=VERSION, shared_classifier=1,
                    group_size=32, weight_dtype=1, scale_dtype=2,
                    side_dtype=2, state_dtype=2)
    if any(h[k] != v for k, v in expected.items()):
        raise ValueError("unsupported GDNe format, sharing or dtype")
    if h["model_type"] not in (1, 2) or h["residual"] != (h["model_type"] == 2):
        raise ValueError("inconsistent model type / residual flag")
    if not 1 <= h["loop_count"] <= 4 or (h["model_type"] == 1 and h["loop_count"] != 1):
        raise ValueError("expected GDN T=1 or MixerLoop T=1..4")
    for key in ("dim", "hidden_dim", "n_layers", "num_heads", "head_k_dim",
                "head_v_dim", "conv_size", "vocab_size", "seq_len"):
        if h[key] < 1:
            raise ValueError(f"invalid {key}")
    if not np.isfinite(h["norm_eps"]) or h["norm_eps"] <= 0:
        raise ValueError("invalid norm epsilon")
    for name, shape, matrix in tensor_specs(h):
        if matrix and (shape[0] % 16 or shape[1] % GROUP_SIZE):
            raise ValueError(f"{name} {shape}: requires rows % 16 = cols % 32 = 0")


def tensor_specs(h):
    """Yield HF name, deployment shape, is_matrix in legacy v2 payload order."""
    d, f, heads = h["dim"], h["hidden_dim"], h["num_heads"]
    k, v = heads * h["head_k_dim"], heads * h["head_v_dim"]
    mixer, ffn, norm = ("mixer", "ffn", "ffn_norm") if h["model_type"] == 2 else ("attn", "mlp", "mlp_norm")
    yield "model.embeddings.weight", (h["vocab_size"], d), True
    for i in range(h["n_layers"]):
        p = f"model.layers.{i}."
        m = p + mixer + "."
        yield p + "attn_norm.weight", (d,), False
        for proj, width in (("q", k), ("k", k), ("v", v)):
            yield m + proj + "_proj.weight", (width, d), True
        for proj in ("a", "b"):
            yield m + proj + "_proj.weight", (heads, d), False
        yield m + "g_proj.weight", (v, d), True
        for proj, width in (("q", k), ("k", k), ("v", v)):
            yield m + proj + "_conv1d.weight", (width, h["conv_size"]), False
        yield m + "A_log", (heads,), False  # stored as -exp(A_log)
        yield m + "dt_bias", (heads,), False
        yield m + "o_norm.weight", (h["head_v_dim"],), False
        yield m + "o_proj.weight", (d, v), True
        yield p + norm + ".weight", (d,), False
        for proj, shape in (("gate", (f, d)), ("down", (d, f)), ("up", (f, d))):
            yield p + ffn + "." + proj + "_proj.weight", shape, True
    yield "model.norm.weight", (d,), False
    if h["residual"]:
        yield "model.residual_weight", (h["loop_count"], d), False


def classify(model):
    c = model.config
    if c.model_type not in ("gated_deltanet", "mixerloop"):
        raise ValueError(f"unsupported model type: {c.model_type}")
    is_loop = c.model_type == "mixerloop"
    if not c.tie_word_embeddings or model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError("deployment requires actually tied embedding / LM head")
    if (not getattr(c, "use_gate", True) or not getattr(c, "use_short_conv", True)
            or getattr(c, "allow_neg_eigval", False) or getattr(c, "attnres_block_size", None)
            or getattr(c, "num_v_heads", None) not in (None, c.num_heads)
            or c.hidden_act not in ("swish", "silu")):
        raise ValueError("unsupported GDN/MLP semantics")
    h = dict(
        magic=MAGIC, version=VERSION, dim=c.hidden_size, hidden_dim=c.intermediate_size,
        n_layers=c.num_hidden_layers, num_heads=c.num_heads, head_k_dim=c.head_dim,
        head_v_dim=int(c.head_dim * c.expand_v), conv_size=c.conv_size,
        vocab_size=c.vocab_size, seq_len=c.max_position_embeddings,
        shared_classifier=1, loop_count=c.loop_count if is_loop else 1,
        group_size=GROUP_SIZE, model_type=2 if is_loop else 1, residual=int(is_loop),
        weight_dtype=1, scale_dtype=2, side_dtype=2, state_dtype=2,
        norm_eps=float(np.float32(c.norm_eps)),
    )
    validate_header(h)
    for layer in model.model.layers:
        mixer = layer.mixer if is_loop else layer.attn
        ffn = layer.ffn if is_loop else layer.mlp
        if not isinstance(mixer, GatedDeltaNet) or not isinstance(ffn, GatedMLP):
            raise ValueError("expected GatedDeltaNet and GatedMLP modules")
    state = model.state_dict()
    names = {name for name, _, _ in tensor_specs(h)} | {"lm_head.weight"}
    if set(state) != names:
        raise ValueError(f"unclassified / missing tensors: {set(state) ^ names}")
    tensors = {}
    for name, shape, matrix in tensor_specs(h):
        value = state[name].detach().cpu().float()
        source_shape = (shape[0], 1, shape[1]) if "_conv1d.weight" in name else shape
        if tuple(value.shape) != source_shape:
            raise ValueError(f"{name}: expected {source_shape}, got {tuple(value.shape)}")
        value = value.reshape(shape)
        if name.endswith(".A_log"):
            value = -value.exp()
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite tensor: {name}")
        tensors[name] = value
        print(f"{'INT8_MATRIX' if matrix else 'FP32_SIDE':11} {name} {shape}")
    print("INT8_MATRIX lm_head.weight (alias of model.embeddings.weight; stored once)")
    return h, tensors


def quantize_q80(weight):
    groups = weight.detach().cpu().float().reshape(weight.shape[0], -1, GROUP_SIZE)
    maximum = groups.abs().amax(dim=-1)
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127)
    q = torch.round(groups / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q.reshape(weight.shape), scale


def read_bundle(path):
    """Strict reference reader, independent of the current fixed HLS geometry."""
    with open(path, "rb") as stream:
        raw = stream.read(HEADER_BYTES)
        if len(raw) != HEADER_BYTES or any(raw[HEADER.size:]):
            raise ValueError("truncated header or nonzero reserved bytes")
        values = HEADER.unpack(raw[:HEADER.size])
        h = dict(zip(HEADER_FIELDS, values[:-1]), norm_eps=values[-1])
        validate_header(h)
        tensors = {}
        for name, shape, matrix in tensor_specs(h):
            arrays = []
            formats = [("i1", shape), ("<f4", (shape[0], shape[1] // 32))] if matrix else [("<f4", shape)]
            for dtype, array_shape in formats:
                size = int(np.prod(array_shape)) * np.dtype(dtype).itemsize
                raw = stream.read(size)
                if len(raw) != size:
                    raise ValueError(f"truncated tensor: {name}")
                arrays.append(np.frombuffer(raw, dtype=dtype).reshape(array_shape))
            if matrix:
                if (arrays[0] == -128).any() or not np.isfinite(arrays[1]).all() or (arrays[1] <= 0).any():
                    raise ValueError(f"invalid Q8 tensor: {name}")
            elif not np.isfinite(arrays[0]).all():
                raise ValueError(f"nonfinite side tensor: {name}")
            tensors[name] = tuple(arrays) if matrix else arrays[0]
        if stream.read(1):
            raise ValueError("trailing bytes after GDNe payload")
    return h, tensors


def write_bundle(path, header, tensors):
    validate_header(header)
    with open(path, "wb") as stream:
        raw = HEADER.pack(*(header[k] for k in HEADER_FIELDS), header["norm_eps"])
        stream.write(raw + bytes(HEADER_BYTES - len(raw)))
        for name, _, matrix in tensor_specs(header):
            value = tensors[name]
            if matrix:
                q, scale = quantize_q80(value)
                stream.write(q.numpy().tobytes())
                stream.write(scale.numpy().astype("<f4").tobytes())
            else:
                stream.write(value.numpy().astype("<f4").tobytes())


@torch.inference_mode()
def export_checkpoint(checkpoint, force=False):
    checkpoint = Path(checkpoint).absolute()
    output = checkpoint / f"{checkpoint.name}-q8.bin"
    if output.exists() and not force:
        raise FileExistsError(f"{output} exists; use --force to replace it")
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    model, info = AutoModelForCausalLM.from_pretrained(
        checkpoint, config=config, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.float32, output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"checkpoint did not load exactly: {info}")
    header, tensors = classify(model)
    with tempfile.NamedTemporaryFile(dir=checkpoint, prefix=".q8-", delete=False) as f:
        temporary = Path(f.name)
    try:
        write_bundle(temporary, header, tensors)
        restored_header, restored = read_bundle(temporary)
        if restored_header != header:
            raise ValueError("header roundtrip failed")
        maximum, squared, count = 0.0, 0.0, 0
        for name, shape, matrix in tensor_specs(header):
            original = tensors[name].numpy()
            if matrix:
                q, scale = restored[name]
                expected_q, expected_scale = quantize_q80(tensors[name])
                np.testing.assert_array_equal(q, expected_q.numpy())
                np.testing.assert_array_equal(scale, expected_scale.numpy())
                error = q.reshape(shape[0], -1, 32) * scale[..., None] - original.reshape(shape[0], -1, 32)
                maximum = max(maximum, float(np.abs(error).max()))
                squared += float(np.square(error.astype(np.float64)).sum())
                count += error.size
            else:
                np.testing.assert_array_equal(restored[name], original)
        sha = hashlib.sha256()
        with temporary.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
        digest = sha.hexdigest()
        if force:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)  # Do not overwrite a concurrent export.
        print(f"matrix max_abs_error={maximum:.9g} mse={squared / count:.9g}")
        print(f"{output}\nbytes={output.stat().st_size} sha256={digest}")
    finally:
        temporary.unlink(missing_ok=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.force)
