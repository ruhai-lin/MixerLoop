# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
"""Convert a completed FLAME DCP checkpoint to standard Hugging Face files."""

import argparse
import tempfile
from pathlib import Path

import fla.models  # noqa: F401
import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torchtitan.tools.logging import init_logger, logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import custom_models  # noqa: F401


@torch.inference_mode()
def save_pretrained(path: str, step: int, config: str, tokenizer: str) -> None:
    output = Path(path)
    model_config = AutoConfig.from_pretrained(config)
    tokenizer_object = AutoTokenizer.from_pretrained(tokenizer)
    checkpoint = output / "checkpoint" / f"step-{step}"
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "checkpoint.pt"
        logger.info(f"Loading final distributed checkpoint {checkpoint}")
        dcp_to_torch_save(str(checkpoint), str(checkpoint_path))
        # Full DCPs contain trusted local optimizer/dataloader Python state.
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("model", payload)
        model = AutoModelForCausalLM.from_config(model_config)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.save_pretrained(output, safe_serialization=True)
        tokenizer_object.save_pretrained(output)
    logger.info(f"HF export completed: {output}")


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    save_pretrained(args.path, args.step, args.config, args.tokenizer)
