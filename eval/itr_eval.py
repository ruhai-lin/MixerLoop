#!/usr/bin/env python3
"""Finite, causal ITR monitoring at the language-model readout."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
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
    "fullloop": "FullLoop",
}


def _active_indices(gram: torch.Tensor, relative_threshold: float) -> list[int]:
    diagonal = gram.diag().double().clamp_min(0)
    if diagonal.numel() == 0 or float(diagonal.max()) == 0:
        return []
    cutoff = relative_threshold**2 * float(diagonal.max())
    return torch.nonzero(diagonal > cutoff, as_tuple=False).flatten().tolist()


def effective_rank(gram: torch.Tensor, relative_threshold: float = 1e-6) -> float:
    """Participation-ratio rank after unit-normalizing active trajectory vectors."""
    active = _active_indices(gram, relative_threshold)
    if not active:
        return 0.0
    selected = gram.double()[active][:, active]
    norms = selected.diag().clamp_min(1e-30).sqrt()
    normalized = selected / (norms[:, None] * norms[None, :])
    return float(normalized.trace().square() / normalized.square().sum().clamp_min(1e-30))


def marginal_itr(
    increment_gram: torch.Tensor,
    relative_threshold: float = 1e-6,
    pinv_rcond: float = 1e-8,
) -> list[float]:
    """Fraction of each finite readout increment outside preceding increments."""
    gram = 0.5 * (increment_gram.double() + increment_gram.double().T)
    active = set(_active_indices(gram, relative_threshold))
    values: list[float] = []
    for depth in range(gram.shape[0]):
        energy = float(gram[depth, depth].clamp_min(0))
        if depth not in active or energy == 0:
            values.append(0.0)
            continue
        prior = [index for index in range(depth) if index in active]
        if not prior:
            values.append(1.0)
            continue
        previous = gram[prior][:, prior]
        cross = gram[prior, depth]
        eigenvalues, eigenvectors = torch.linalg.eigh(previous)
        cutoff = pinv_rcond * float(eigenvalues.max().clamp_min(0))
        inverse = torch.where(eigenvalues > cutoff, eigenvalues.reciprocal(), 0)
        projected = float(((eigenvectors.T @ cross).square() * inverse).sum())
        residual = min(max(energy - projected, 0.0), energy)
        values.append(residual / energy)
    return values


def trajectory_summary(
    cumulative_gram: torch.Tensor,
    increment_gram: torch.Tensor,
    relative_threshold: float,
    pinv_rcond: float,
) -> dict[str, Any]:
    loops = cumulative_gram.shape[0]
    return {
        "itr": effective_rank(cumulative_gram, relative_threshold),
        "itr_by_depth": [
            effective_rank(cumulative_gram[:depth, :depth], relative_threshold)
            for depth in range(1, loops + 1)
        ],
        "marginal_itr": marginal_itr(increment_gram, relative_threshold, pinv_rcond),
        "cumulative_gram": cumulative_gram.double().cpu().tolist(),
        "increment_gram": increment_gram.double().cpu().tolist(),
    }


def load_windows(args: argparse.Namespace) -> torch.Tensor:
    data_dir = Path(args.data_dir).expanduser().resolve()
    if args.dataset_split == "validation":
        shards = [data_dir / "shard_06542.bin"]
    else:
        shards = [data_dir / f"shard_{index:05d}.bin" for index in range(170)]
    missing = [path for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"ClimbMix data is incomplete; first missing shard: {missing[0]}")

    rng = random.Random(args.data_seed)
    windows = []
    for _ in range(args.num_windows):
        shard = shards[rng.randrange(len(shards))]
        tokens = np.memmap(shard, dtype=np.uint16, mode="r")
        if len(tokens) < args.seq_len:
            raise ValueError(f"{shard} has fewer than {args.seq_len} tokens")
        start = rng.randrange(len(tokens) - args.seq_len + 1)
        windows.append(torch.from_numpy(tokens[start : start + args.seq_len].astype(np.int64)))
    return torch.stack(windows)


def prediction_positions(seq_len: int, count: int) -> torch.Tensor:
    if seq_len < 4:
        raise ValueError("seq_len must be at least 4")
    start = seq_len // 2
    positions = torch.linspace(start, seq_len - 2, steps=count).round().long().unique()
    if len(positions) != count:
        raise ValueError(f"cannot select {count} unique prediction positions from seq_len={seq_len}")
    return positions


def layer_mixer(layer: torch.nn.Module) -> torch.nn.Module:
    if hasattr(layer, "mixer"):
        return layer.mixer
    if hasattr(layer, "attn"):
        return layer.attn
    raise TypeError(f"{type(layer).__name__} has no supported token mixer")


class ContextPrefix:
    """Keep contextual mixing for a prefix of calls and isolate later calls by token."""

    def __init__(self, mixer: torch.nn.Module, native_prefix: int):
        self.mixer = mixer
        self.native_prefix = native_prefix
        self.calls = 0
        self._inside_local_call = False
        self._handle = None

    @staticmethod
    def _hidden_argument(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
        if args:
            return args[0]
        return kwargs["hidden_states"]

    @staticmethod
    def _first(output: Any) -> torch.Tensor:
        return output[0] if isinstance(output, (tuple, list)) else output

    @staticmethod
    def _replace_first(output: Any, value: torch.Tensor) -> Any:
        if isinstance(output, tuple):
            return (value, *output[1:])
        if isinstance(output, list):
            return [value, *output[1:]]
        return value

    def _hook(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        if self._inside_local_call:
            return output
        call_index = self.calls
        self.calls += 1
        if call_index < self.native_prefix:
            return output

        hidden = self._hidden_argument(args, kwargs)
        if hidden.shape[0] != 1:
            raise ValueError("causal ITR monitoring requires one window per forward")
        isolated = hidden.reshape(-1, 1, hidden.shape[-1])
        self._inside_local_call = True
        try:
            local_output = module(isolated, attention_mask=None)
        finally:
            self._inside_local_call = False
        local_hidden = self._first(local_output).reshape_as(self._first(output))
        return self._replace_first(output, local_hidden)

    def __enter__(self) -> "ContextPrefix":
        self._handle = self.mixer.register_forward_hook(self._hook, with_kwargs=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._handle.remove()


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def readout_state(
    model: torch.nn.Module,
    window: torch.Tensor,
    positions: torch.Tensor,
    target_layer: int | None,
    native_prefix: int | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    inputs = window.unsqueeze(0).to(device)
    if target_layer is None:
        context = nullcontext()
    else:
        context = ContextPrefix(layer_mixer(model.model.layers[target_layer]), int(native_prefix))

    with context as monitor, _autocast(device):
        logits = model(input_ids=inputs).logits[0, positions.to(device)].float()
    calls = 0 if target_layer is None else monitor.calls
    probabilities = logits.softmax(dim=-1)
    square_root_probabilities = probabilities.sqrt()
    targets = inputs[0, positions.to(device) + 1]
    target_log_probabilities = logits.log_softmax(dim=-1).gather(1, targets[:, None]).squeeze(1)
    predictions = logits.argmax(dim=-1)
    return square_root_probabilities, target_log_probabilities, predictions, calls


@torch.no_grad()
def evaluate_layer(
    model: torch.nn.Module,
    windows: torch.Tensor,
    positions: torch.Tensor,
    layer_index: int,
    loops: int,
    device: torch.device,
    relative_threshold: float,
    pinv_rcond: float,
) -> dict[str, Any]:
    cumulative_gram = torch.zeros((loops, loops), dtype=torch.float64, device=device)
    increment_gram = torch.zeros_like(cumulative_gram)
    step_hellinger = torch.zeros(loops, dtype=torch.float64, device=device)
    cumulative_hellinger = torch.zeros_like(step_hellinger)
    target_logprob_gain = torch.zeros_like(step_hellinger)
    top1_flip_rate = torch.zeros_like(step_hellinger)
    native_nll = 0.0
    context_off_nll = 0.0
    native_hook_error = 0.0

    for window_index, window in enumerate(windows):
        readouts = []
        log_probabilities = []
        predictions = []
        for prefix in range(loops + 1):
            readout, logp, predicted, calls = readout_state(
                model,
                window,
                positions,
                layer_index,
                prefix,
                device,
            )
            if calls != loops:
                raise RuntimeError(
                    f"layer {layer_index}: expected {loops} mixer calls, observed {calls}"
                )
            readouts.append(readout)
            log_probabilities.append(logp)
            predictions.append(predicted)

        trajectory = torch.stack(readouts)
        logp = torch.stack(log_probabilities)
        predicted = torch.stack(predictions)
        cumulative = trajectory[1:] - trajectory[:1]
        increments = trajectory[1:] - trajectory[:-1]
        cumulative_gram += torch.einsum("rpd,spd->rs", cumulative, cumulative).double()
        increment_gram += torch.einsum("rpd,spd->rs", increments, increments).double()
        step_hellinger += 0.5 * increments.square().sum(dim=-1).mean(dim=-1).double()
        cumulative_hellinger += 0.5 * cumulative.square().sum(dim=-1).mean(dim=-1).double()
        target_logprob_gain += (logp[1:] - logp[:-1]).mean(dim=-1).double()
        top1_flip_rate += (predicted[1:] != predicted[:-1]).float().mean(dim=-1).double()
        native_nll -= float(logp[-1].mean())
        context_off_nll -= float(logp[0].mean())

        if window_index == 0:
            raw, _, _, _ = readout_state(model, window, positions, None, None, device)
            native_hook_error = float((raw - trajectory[-1]).abs().max())

    count = len(windows)
    summary = trajectory_summary(
        cumulative_gram / count,
        increment_gram / count,
        relative_threshold,
        pinv_rcond,
    )
    summary.update(
        {
            "layer": layer_index,
            "step_hellinger_squared": (step_hellinger / count).cpu().tolist(),
            "cumulative_hellinger_squared": (cumulative_hellinger / count).cpu().tolist(),
            "target_logprob_gain": (target_logprob_gain / count).cpu().tolist(),
            "top1_flip_rate": (top1_flip_rate / count).cpu().tolist(),
            "native_nll": native_nll / count,
            "context_off_nll": context_off_nll / count,
            "native_context_nll_gain": (context_off_nll - native_nll) / count,
            "native_hook_max_abs_error": native_hook_error,
        }
    )
    return summary


def evaluate_model(
    model_path: str,
    windows: torch.Tensor,
    positions: torch.Tensor,
    layers: list[int] | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).to(device)
    model.eval()
    model_type = model.config.model_type
    if model_type not in MODEL_LABELS:
        raise ValueError(f"expected one of {sorted(MODEL_LABELS)}, got {model_type}")
    loops = int(getattr(model.config, "loop_count", 1))
    selected_layers = layers or list(range(len(model.model.layers)))
    if any(index < 0 or index >= len(model.model.layers) for index in selected_layers):
        raise ValueError(f"invalid layer selection {selected_layers}")

    layer_results = []
    for layer_index in selected_layers:
        result = evaluate_layer(
            model,
            windows,
            positions,
            layer_index,
            loops,
            device,
            args.relative_threshold,
            args.pinv_rcond,
        )
        layer_results.append(result)
        print(
            f"[itr] {MODEL_LABELS[model_type]} layer={layer_index} "
            f"ITR={result['itr']:.3f} "
            f"mITR={[round(value, 3) for value in result['marginal_itr']]} "
            f"context-NLL-gain={result['native_context_nll_gain']:+.4f}",
            flush=True,
        )

    cumulative_gram = torch.tensor(
        np.mean([result["cumulative_gram"] for result in layer_results], axis=0)
    )
    increment_gram = torch.tensor(
        np.mean([result["increment_gram"] for result in layer_results], axis=0)
    )
    aggregate = trajectory_summary(
        cumulative_gram,
        increment_gram,
        args.relative_threshold,
        args.pinv_rcond,
    )
    for key in (
        "step_hellinger_squared",
        "cumulative_hellinger_squared",
        "target_logprob_gain",
        "top1_flip_rate",
    ):
        aggregate[key] = np.mean([result[key] for result in layer_results], axis=0).tolist()
    aggregate["native_context_nll_gain"] = float(
        np.mean([result["native_context_nll_gain"] for result in layer_results])
    )
    step_effects = aggregate["step_hellinger_squared"]
    aggregate["later_pass_effect_fraction"] = (
        float(sum(step_effects[1:]) / sum(step_effects)) if sum(step_effects) else 0.0
    )
    aggregate["later_pass_target_logprob_gain"] = float(
        sum(aggregate["target_logprob_gain"][1:])
    )
    aggregate["native_hook_max_abs_error"] = max(
        result["native_hook_max_abs_error"] for result in layer_results
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "model_path": os.path.abspath(model_path),
        "model_type": model_type,
        "label": MODEL_LABELS[model_type],
        "loop_count": loops,
        "layers": selected_layers,
        "aggregate": aggregate,
        "layer_results": layer_results,
    }


def write_results(out_dir: Path, result: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "itr_eval.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    rows = []
    for model in result["models"]:
        for layer in model["layer_results"]:
            for pass_index in range(model["loop_count"]):
                rows.append(
                    {
                        "model": model["label"],
                        "layer": layer["layer"],
                        "pass": pass_index + 1,
                        "itr_at_depth": layer["itr_by_depth"][pass_index],
                        "marginal_itr": layer["marginal_itr"][pass_index],
                        "step_hellinger_squared": layer["step_hellinger_squared"][pass_index],
                        "cumulative_hellinger_squared": layer["cumulative_hellinger_squared"][
                            pass_index
                        ],
                        "target_logprob_gain": layer["target_logprob_gain"][pass_index],
                        "top1_flip_rate": layer["top1_flip_rate"][pass_index],
                    }
                )
    with (out_dir / "itr_events.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "itr_eval.md").open("w", encoding="utf-8") as handle:
        handle.write("# Causal Readout ITR\n\n")
        handle.write(
            "Each prefix restores contextual mixer computation while retaining "
            "token-isolated local computation at later passes.\n\n"
        )
        handle.write(
            "| Model | ITR | mITR | Later effect | Later $\\Delta\\log p$ | "
            "Context NLL gain |\n"
        )
        handle.write("|---|---:|---|---:|---:|---:|\n")
        for model in result["models"]:
            aggregate = model["aggregate"]
            mitr = ", ".join(f"{value:.3f}" for value in aggregate["marginal_itr"])
            handle.write(
                f"| {model['label']} | {aggregate['itr']:.3f} | {mitr} | "
                f"{aggregate['later_pass_effect_fraction']:.1%} | "
                f"{aggregate['later_pass_target_logprob_gain']:+.3f} | "
                f"{aggregate['native_context_nll_gain']:+.4f} |\n"
            )


def plot_results(out_dir: Path, result: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    models = result["models"]
    max_loops = max(model["loop_count"] for model in models)
    metrics = (
        ("marginal_itr", "Marginal ITR", "viridis", 0.0, 1.0),
        (
            "step_hellinger_squared",
            r"Prediction effect ($H^2$)",
            "magma",
            0.0,
            max(
                max(layer["step_hellinger_squared"])
                for model in models
                for layer in model["layer_results"]
            ),
        ),
    )
    gain_limit = max(
        abs(value)
        for model in models
        for layer in model["layer_results"]
        for value in layer["target_logprob_gain"]
    )

    fig, axes = plt.subplots(
        3,
        len(models),
        figsize=(7.0, 6.0),
        constrained_layout=True,
        squeeze=False,
    )
    for column, model in enumerate(models):
        layers = model["layers"]
        for row, (key, label, cmap, lower, upper) in enumerate(metrics):
            matrix = np.full((len(layers), max_loops), np.nan)
            for layer_row, layer in enumerate(model["layer_results"]):
                matrix[layer_row, : model["loop_count"]] = layer[key]
            image = axes[row, column].imshow(
                matrix,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                vmin=lower,
                vmax=upper,
            )
            if column == len(models) - 1:
                fig.colorbar(image, ax=axes[row, :], shrink=0.8, pad=0.02)
            axes[row, column].set_title(model["label"] if row == 0 else "")
            axes[row, column].set_yticks(range(len(layers)), labels=layers)
            axes[row, column].set_xticks(range(max_loops), labels=range(1, max_loops + 1))
            if column == 0:
                axes[row, column].set_ylabel(f"{label}\nLayer")

        gain = np.full((len(layers), max_loops), np.nan)
        for layer_row, layer in enumerate(model["layer_results"]):
            gain[layer_row, : model["loop_count"]] = layer["target_logprob_gain"]
        image = axes[2, column].imshow(
            gain,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-gain_limit, vcenter=0.0, vmax=gain_limit),
        )
        if column == len(models) - 1:
            fig.colorbar(image, ax=axes[2, :], shrink=0.8, pad=0.02)
        axes[2, column].set_yticks(range(len(layers)), labels=layers)
        axes[2, column].set_xticks(range(max_loops), labels=range(1, max_loops + 1))
        axes[2, column].set_xlabel("Contextual mixer pass")
        if column == 0:
            axes[2, column].set_ylabel("True-token $\\Delta\\log p$\nLayer")

    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"itr_heatmaps.{suffix}", dpi=300)
    plt.close(fig)

    colors = {"NoLoop": "#777777", "MixerLoop": "#2574A9", "FullLoop": "#D35400"}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45), constrained_layout=True)
    for model in models:
        aggregate = model["aggregate"]
        passes = np.arange(1, model["loop_count"] + 1)
        for axis, key in zip(
            axes,
            ("itr_by_depth", "step_hellinger_squared", "target_logprob_gain"),
        ):
            axis.plot(
                passes,
                aggregate[key],
                marker="o",
                color=colors[model["label"]],
                label=model["label"],
            )
    axes[0].set_ylabel("Readout ITR")
    axes[0].set_title("Cumulative directions")
    axes[1].set_ylabel(r"Prediction effect ($H^2$)")
    axes[1].set_title("Finite output change")
    axes[2].axhline(0, color="#888888", linewidth=0.8)
    axes[2].set_ylabel(r"True-token $\Delta\log p$")
    axes[2].set_title("Task contribution")
    for axis in axes:
        axis.set_xticks(range(1, max_loops + 1))
        axis.set_xlabel("Contextual pass")
        axis.title.set_fontsize(10)
        axis.xaxis.label.set_fontsize(9)
        axis.yaxis.label.set_fontsize(9)
        axis.tick_params(labelsize=8)
    axes[0].legend(
        frameon=False,
        fontsize=7,
        loc="upper left",
    )
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"itr_summary.{suffix}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_paths", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_windows", type=int, default=8)
    parser.add_argument("--prediction_positions", type=int, default=16)
    parser.add_argument("--relative_threshold", type=float, default=1e-6)
    parser.add_argument("--pinv_rcond", type=float, default=1e-8)
    parser.add_argument("--data_dir", default="data/climbmix-10b")
    parser.add_argument("--dataset_split", choices=["train", "validation"], default="validation")
    parser.add_argument("--data_seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.num_windows < 1 or args.prediction_positions < 1:
        raise ValueError("num_windows and prediction_positions must be positive")
    device = torch.device(args.device)
    windows = load_windows(args)
    positions = prediction_positions(args.seq_len, args.prediction_positions)
    models = [
        evaluate_model(path, windows, positions, args.layers, args, device)
        for path in args.model_paths
    ]
    result = {
        "definition": {
            "context_off": (
                "Apply the same mixer independently to every token, resetting "
                "convolution and recurrent state between tokens."
            ),
            "trajectory": (
                "Enable contextual mixer calls in prefix order and measure finite "
                "changes in sqrt(next-token probability) at the final readout."
            ),
            "step_effect": "Squared Hellinger distance between consecutive prefix policies.",
            "task_effect": "Change in ground-truth next-token log probability.",
        },
        "seq_len": args.seq_len,
        "num_windows": args.num_windows,
        "prediction_positions": positions.tolist(),
        "dataset_split": args.dataset_split,
        "data_seed": args.data_seed,
        "models": models,
    }
    out_dir = Path(args.out_dir)
    write_results(out_dir, result)
    plot_results(out_dir, result)
    print(f"[itr] wrote {out_dir / 'itr_eval.json'}")


if __name__ == "__main__":
    main()
