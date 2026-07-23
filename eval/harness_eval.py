#!/usr/bin/env python3
"""
lm-eval-harness runner for MixerLoop-compatible checkpoints.

Usage (from run snapshot created by train.sh):
  cd outputs/<run>/eval && python harness_eval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if RUN_ROOT not in sys.path:
    sys.path.insert(0, RUN_ROOT)

import custom_models  # noqa: F401
from lm_eval import simple_evaluate
from lm_eval.api.registry import get_model


DEFAULT_TASKS = [
    "hellaswag",
    "boolq",
    "piqa",
    "winogrande",
    "openbookqa",
    "arc_easy",
    "arc_challenge",
    "commonsense_qa",
    "copa",
]


def pick_primary_metric(metrics: dict[str, Any]) -> tuple[str, float]:
    preferred = [
        "acc_norm,none",
        "acc,none",
        "exact_match,none",
        "f1,none",
        "acc",
        "exact_match",
        "f1",
    ]
    for key in preferred:
        if key in metrics and isinstance(metrics[key], (int, float)):
            return key, float(metrics[key])
    for key, value in metrics.items():
        if "stderr" in key:
            continue
        if isinstance(value, (int, float)):
            return key, float(value)
    return "", float("nan")


def build_hf_lm(model_path: str, device: str, batch_size: str, tokenizer_path: str | None):
    model_args = [
        f"pretrained={model_path}",
        "trust_remote_code=True",
        "dtype=bfloat16" if device.startswith("cuda") else "dtype=float32",
    ]
    if tokenizer_path:
        model_args.append(f"tokenizer={tokenizer_path}")
    arg_string = ",".join(model_args)

    model_cls = get_model("hf")
    return model_cls.create_from_arg_string(
        arg_string,
        {
            "batch_size": batch_size,
            "device": device,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Run lm-eval-harness and export CSV.")
    parser.add_argument(
        "--model_path",
        default=None,
        help="HF model directory. Default: parent of this script (so run from eval/ directly).",
    )
    parser.add_argument(
        "--out_dir",
        default=".",
        help="Output directory. Default: current working directory.",
    )
    parser.add_argument("--tokenizer_path", default=None, help="Optional tokenizer path.")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="Comma-separated task list.")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=float, default=None, help="Optional lm-eval limit.")
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_samples", action="store_true")
    parser.add_argument("--verbosity", default="INFO")
    args = parser.parse_args()

    model_path = args.model_path or os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[harness] model={model_path}")
    print(f"[harness] tasks={tasks}")
    print(f"[harness] device={args.device}, batch_size={args.batch_size}")

    lm = build_hf_lm(
        model_path=model_path,
        device=args.device,
        batch_size=args.batch_size,
        tokenizer_path=args.tokenizer_path,
    )
    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        log_samples=args.log_samples,
        verbosity=args.verbosity,
    )

    json_path = os.path.join(out_dir, "harness_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(out_dir, "harness_eval.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "PrimaryMetric", "Score", "AllMetricsJSON"])
        for task, metrics in sorted(results.get("results", {}).items()):
            metric_name, score = pick_primary_metric(metrics)
            writer.writerow(
                [
                    task,
                    metric_name,
                    f"{score:.6f}" if score == score else "",
                    json.dumps(metrics, ensure_ascii=False),
                ]
            )

    print(f"[harness] wrote {json_path}")
    print(f"[harness] wrote {csv_path}")


if __name__ == "__main__":
    main()
