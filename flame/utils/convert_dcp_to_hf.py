# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Convert the final Flame checkpoint to HF and MixerLoop deployment assets."""

from __future__ import annotations

import argparse
import io
import os
import struct
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import BinaryIO

import fla.models  # noqa: F401, registers FLA model types with Transformers
import sentencepiece as spm
import torch
import torch.serialization
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torchtitan.tools.logging import init_logger, logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import custom_models  # noqa: F401, registers MixerLoop with Transformers

CHECKPOINT_MAGIC = 0x47444E65  # "GDNe", little-endian
CHECKPOINT_VERSION = 2
CHECKPOINT_HEADER_BYTES = 256
QUANT_GROUP_SIZE = 32


def _write_fp32(handle: BinaryIO, tensor: torch.Tensor) -> None:
    values = tensor.detach().cpu().reshape(-1).to(torch.float32).contiguous()
    handle.write(values.numpy().tobytes())


def _write_int8(handle: BinaryIO, tensor: torch.Tensor) -> None:
    values = tensor.detach().cpu().reshape(-1).to(torch.int8).contiguous()
    handle.write(values.numpy().tobytes())


def quantize_q80(weight: torch.Tensor, group_size: int = QUANT_GROUP_SIZE):
    """Symmetric row-wise Q8_0 with one fp32 scale per input group."""

    if weight.ndim != 2:
        raise ValueError(f"Q8 tensors must be matrices, got shape {tuple(weight.shape)}")
    rows, cols = weight.shape
    if cols % group_size:
        raise ValueError(f"matrix input dimension {cols} is not divisible by group size {group_size}")
    grouped = weight.detach().cpu().float().reshape(rows, cols // group_size, group_size)
    maximum = grouped.abs().amax(dim=-1)
    scales = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127.0)
    quantized = torch.round(grouped / scales.unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
    return quantized.reshape(rows, cols).contiguous(), scales.contiguous()


def _write_q8(handle: BinaryIO, weight: torch.Tensor, group_size: int) -> None:
    quantized, scales = quantize_q80(weight, group_size)
    _write_int8(handle, quantized)
    _write_fp32(handle, scales)


def export_q8_checkpoint(
    model: AutoModelForCausalLM,
    output_path: str | os.PathLike[str],
    group_size: int = QUANT_GROUP_SIZE,
) -> Path:
    """Write the canonical GDNe v2 checkpoint consumed by ``hardware/``."""

    config = model.config
    if config.model_type != "mixerloop":
        raise ValueError(f"deployment export only supports mixerloop, got {config.model_type}")
    if group_size != QUANT_GROUP_SIZE:
        raise ValueError(f"the hardware profile requires group size {QUANT_GROUP_SIZE}")
    if config.hidden_size != config.num_heads * config.head_dim:
        raise ValueError("the hardware profile requires hidden_size == num_heads * head_dim")

    tied = bool(config.tie_word_embeddings)
    if not tied:
        raise ValueError("the hardware profile requires tied input and output embeddings")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        header = struct.pack(
            "<I13i",
            CHECKPOINT_MAGIC,
            CHECKPOINT_VERSION,
            int(config.hidden_size),
            int(config.intermediate_size),
            int(config.num_hidden_layers),
            int(config.num_heads),
            int(config.head_dim),
            int(config.head_dim),
            int(config.conv_size),
            int(config.vocab_size),
            int(config.max_position_embeddings),
            int(tied),
            0,
            group_size,
        )
        handle.write(header)
        handle.write(bytes(CHECKPOINT_HEADER_BYTES - len(header)))

        core = model.model
        _write_q8(handle, core.embeddings.weight, group_size)
        for layer in core.layers:
            mixer = layer.mixer
            ffn = layer.ffn
            _write_fp32(handle, layer.attn_norm.weight)
            _write_q8(handle, mixer.q_proj.weight, group_size)
            _write_q8(handle, mixer.k_proj.weight, group_size)
            _write_q8(handle, mixer.v_proj.weight, group_size)
            _write_fp32(handle, mixer.a_proj.weight)
            _write_fp32(handle, mixer.b_proj.weight)
            _write_q8(handle, mixer.g_proj.weight, group_size)
            _write_fp32(handle, mixer.q_conv1d.weight)
            _write_fp32(handle, mixer.k_conv1d.weight)
            _write_fp32(handle, mixer.v_conv1d.weight)
            _write_fp32(handle, -torch.exp(mixer.A_log.float()))
            _write_fp32(handle, mixer.dt_bias)
            _write_fp32(handle, mixer.o_norm.weight)
            _write_q8(handle, mixer.o_proj.weight, group_size)
            _write_fp32(handle, layer.ffn_norm.weight)
            _write_q8(handle, ffn.gate_proj.weight, group_size)
            _write_q8(handle, ffn.down_proj.weight, group_size)
            _write_q8(handle, ffn.up_proj.weight, group_size)
        _write_fp32(handle, core.norm.weight)

    logger.info(f"Saved hardware Q8 checkpoint to {output}")
    return output


def export_tokenizer_binary(tokenizer_model: str | os.PathLike[str], output_path: str | os.PathLike[str]) -> Path:
    """Write llama2.c-compatible tokenizer metadata for the hardware host."""

    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    tokens: list[bytes] = []
    scores: list[float] = []
    for index in range(processor.vocab_size()):
        piece = processor.id_to_piece(index)
        if index == processor.bos_id():
            piece = "\n<s>\n"
        elif index == processor.eos_id():
            piece = "\n</s>\n"
        tokens.append(piece.replace("▁", " ").encode("utf-8"))
        scores.append(processor.get_score(index))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(struct.pack("<I", max(map(len, tokens))))
        for token, score in zip(tokens, scores):
            handle.write(struct.pack("<fI", score, len(token)))
            handle.write(token)
    logger.info(f"Saved hardware tokenizer to {output}")
    return output


@torch.inference_mode()
def save_pretrained(path: str, step: int, config: str, tokenizer: str) -> None:
    output = Path(path)
    logger.info(f"Loading the resolved config from {config}")
    model_config = AutoConfig.from_pretrained(config, trust_remote_code=True)
    if model_config.model_type != "mixerloop":
        raise ValueError("the release export path only supports MixerLoop checkpoints")

    tokenizer_object = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    tokenizer_object.save_pretrained(output)

    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint = output / "checkpoint" / f"step-{step}"
        checkpoint_path = Path(temporary_directory) / "checkpoint.pt"
        logger.info(f"Converting distributed checkpoint {checkpoint}")
        dcp_to_torch_save(str(checkpoint), str(checkpoint_path))

        model = AutoModelForCausalLM.from_config(model_config)
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("model", payload)
        model.load_state_dict(state_dict)
        model.eval()

        model_config.save_pretrained(output)
        model.save_pretrained(output)
        export_q8_checkpoint(model, output / f"{output.name}_q8.bin")

    tokenizer_model = Path(tokenizer) / "tokenizer.model"
    if not tokenizer_model.is_file():
        raise FileNotFoundError(f"expected SentencePiece tokenizer model at {tokenizer_model}")
    export_tokenizer_binary(tokenizer_model, output / "tokenizer.bin")


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser("Convert the final DCP checkpoint and deployment artifacts.")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    arguments = parser.parse_args()
    save_pretrained(arguments.path, arguments.step, arguments.config, arguments.tokenizer)
