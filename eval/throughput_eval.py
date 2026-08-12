#!/usr/bin/env python3
"""Measure BF16 prefill latency for matched MixerLoop checkpoints."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT = SCRIPT_DIR.parent
if str(RUN_ROOT) not in sys.path:
    sys.path.insert(0, str(RUN_ROOT))

import custom_models  # noqa: E402,F401


MODEL_LABELS = {
    "gated_deltanet": "NoLoop",
    "mixerloop": "MixerLoop",
    "ffnloop": "FFNLoop",
    "fullloop": "FullLoop",
}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark_model(
    model_path: str,
    *,
    batch_size: int,
    seq_len: int,
    warmup: int,
    iterations: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=model_dtype)
    model.to(device).eval()
    model_type = model.config.model_type
    if model_type not in MODEL_LABELS:
        raise ValueError(f"unsupported model type: {model_type}")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(
        0,
        int(model.config.vocab_size),
        (batch_size, seq_len),
        generator=generator,
        dtype=torch.long,
    ).to(device)

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    with autocast:
        for _ in range(warmup):
            model(input_ids=input_ids, logits_to_keep=1)
    synchronize(device)

    latencies_ms: list[float] = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for _ in range(iterations):
            synchronize(device)
            start = time.perf_counter()
            model(input_ids=input_ids, logits_to_keep=1)
            synchronize(device)
            latencies_ms.append(1_000.0 * (time.perf_counter() - start))

    median_ms = statistics.median(latencies_ms)
    result = {
        "model_path": str(Path(model_path).resolve()),
        "model_type": model_type,
        "label": MODEL_LABELS[model_type],
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "warmup_forwards": warmup,
        "timed_forwards": iterations,
        "dtype": "bfloat16" if device.type == "cuda" else "float32",
        "median_latency_ms": median_ms,
        "mean_latency_ms": statistics.fmean(latencies_ms),
        "stdev_latency_ms": statistics.stdev(latencies_ms) if iterations > 1 else 0.0,
        "prefill_tokens_per_second": batch_size * seq_len / (median_ms / 1_000.0),
        "latencies_ms": latencies_ms,
    }
    del model, input_ids
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_paths", nargs="+", required=True)
    parser.add_argument("--out_file", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if min(args.batch_size, args.seq_len, args.warmup, args.iterations) < 1:
        raise ValueError("batch size, sequence length, warmup, and iterations must be positive")
    device = torch.device(args.device)
    results = [
        benchmark_model(
            model_path,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            warmup=args.warmup,
            iterations=args.iterations,
            seed=args.seed,
            device=device,
        )
        for model_path in args.model_paths
    ]
    payload = {
        "protocol": "BF16 prefill; final-token logits only; synchronized wall-clock timing",
        "seed": args.seed,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "models": results,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(
            f"{result['label']}: {result['median_latency_ms']:.3f} ms, "
            f"{result['prefill_tokens_per_second']:.0f} tokens/s"
        )
    print(f"Wrote {args.out_file}")


if __name__ == "__main__":
    main()
