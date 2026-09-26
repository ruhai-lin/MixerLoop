"""Streaming signed-activation statistics for I004 A5.

The observer consumes actual capture tensors without changing the model. Histogram
edges come from the first fit batch at each (scale, position); every condition and
validation batch then uses those edges. Only moments and window scalars persist.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import torch


POSITIONS = {"attention_post", "ffn_input", "block_post", "gate_silu", "up", "product"}
METRICS = ("rms", "max_abs", "positive_fraction", "negative_fraction", "zero_fraction",
           "top_1pct_energy", "top_10pct_energy")
CONDITION_FIELDS = ("model", "scale", "layer", "pass", "position_name", "split")


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _pair_cosine(unit_sum, unit_self_dot, count):
    # Subtract the measured self dot products, preserving finite-precision evidence.
    return ((unit_sum.square().sum().item() - unit_self_dot) / (count * (count - 1))
            if count > 1 else None)


def _statistics(values, edges):
    """Reduce one [windows, selected tokens, width] batch on its input device."""
    x = values.detach().to(dtype=torch.float64)
    batch, tokens, width = x.shape
    flat = x.reshape(-1, width)
    square = x.square()
    energy = square.sum(-1)
    finite = torch.isfinite(x)
    valid_token = finite.all(-1)
    directional = valid_token & (energy > 0)
    denominator = torch.where(directional, energy, torch.ones_like(energy))
    largest = square.topk(math.ceil(width * 0.10), dim=-1, sorted=True).values
    token_metrics = torch.stack((
        (energy / width).sqrt(), x.abs().amax(-1),
        (x > 0).to(torch.float64).mean(-1),
        (x < 0).to(torch.float64).mean(-1),
        (x == 0).to(torch.float64).mean(-1),
        largest[..., :math.ceil(width * 0.01)].sum(-1) / denominator,
        largest.sum(-1) / denominator,
    ), dim=-1)
    valid_metrics = torch.stack([valid_token] * 5 + [directional] * 2, dim=-1)
    metric_values = torch.where(valid_metrics, token_metrics, 0)
    # One scalar table per window, with a count for each metric's true denominator.
    window_table = torch.stack((
        metric_values.sum(1), metric_values.square().sum(1),
        valid_metrics.sum(1),
        torch.where(valid_metrics, token_metrics, torch.inf).amin(1),
        torch.where(valid_metrics, token_metrics, -torch.inf).amax(1),
    ), dim=-1).cpu()
    units = torch.where(directional[..., None], x / denominator.sqrt()[..., None], 0)
    unit_sums = units.sum(1)
    unit_self_dots = units.square().sum((1, 2))
    window_counts = torch.stack((
        valid_token.sum(1), directional.sum(1),
        (valid_token & (energy == 0)).sum(1), (~finite).sum((1, 2)),
    ), dim=-1).cpu()
    window_directions = torch.stack((unit_sums.square().sum(-1), unit_self_dots), -1).cpu()
    variance, mean = torch.var_mean(flat, dim=0, correction=0)
    bins = edges.to(device=x.device, dtype=x.dtype)
    finite_values = x[finite]
    inside = finite_values[(finite_values >= bins[0]) & (finite_values <= bins[-1])]
    # Interior edges give left-closed bins; the final upper boundary is included.
    histogram = torch.bincount(torch.bucketize(inside, bins[1:-1], right=True),
                               minlength=len(bins) - 1).cpu()
    tails = torch.stack(((finite_values < bins[0]).sum(),
                         (finite_values > bins[-1]).sum())).cpu()
    state = {
        "count": batch * tokens, "width": width,
        "mean": mean.cpu(), "m2": (variance * (batch * tokens)).cpu(),
        "unit_sum": unit_sums.sum(0).cpu(),
        "unit_self_dot": unit_self_dots.sum().item(),
        "nonzero_finite_tokens": int(window_counts[:, 1].sum()),
        "zero_tokens": int(window_counts[:, 2].sum()),
        "nonfinite_elements": int(window_counts[:, 3].sum()),
        "nonfinite_tokens": batch * tokens - int(window_counts[:, 0].sum()),
        "histogram": histogram, "underflow": int(tails[0]), "overflow": int(tails[1]),
        "metric_sum": window_table[:, :, 0].sum(0),
        "metric_square_sum": window_table[:, :, 1].sum(0),
        "metric_count": window_table[:, :, 2].sum(0),
        "metric_min": window_table[:, :, 3].amin(0),
        "metric_max": window_table[:, :, 4].amax(0),
    }
    return state, window_table, window_counts, window_directions


def merge_moments(destination, source):
    """Combine disjoint token batches using their counts and centered moments."""
    old_count, new_count = destination["count"], source["count"]
    total = old_count + new_count
    delta = source["mean"] - destination["mean"]
    destination["m2"] += source["m2"] + delta.square() * (old_count * new_count / total)
    destination["mean"] += delta * (new_count / total)
    destination["count"] = total
    for key in ("unit_sum", "unit_self_dot", "nonzero_finite_tokens", "zero_tokens",
                "nonfinite_elements", "nonfinite_tokens", "histogram", "underflow",
                "overflow", "metric_sum", "metric_square_sum", "metric_count"):
        destination[key] += source[key]
    destination["metric_min"] = torch.minimum(destination["metric_min"], source["metric_min"])
    destination["metric_max"] = torch.maximum(destination["metric_max"], source["metric_max"])


def summarize_moments(state):
    result = {"token_count": state["count"], "width": state["width"],
              "coordinate_mean": state["mean"],
              "coordinate_variance_population": state["m2"] / state["count"],
              "histogram": state["histogram"], "underflow": state["underflow"],
              "overflow": state["overflow"], "nonfinite_elements": state["nonfinite_elements"],
              "nonfinite_tokens": state["nonfinite_tokens"], "zero_tokens": state["zero_tokens"],
              "cosine_nonzero_tokens": state["nonzero_finite_tokens"],
              "mean_pair_cosine": _pair_cosine(state["unit_sum"], state["unit_self_dot"],
                                               state["nonzero_finite_tokens"]),
              "unit_self_dot_roundoff": state["unit_self_dot"] - state["nonzero_finite_tokens"]}
    for index, name in enumerate(METRICS):
        count = int(state["metric_count"][index])
        result[name] = {"count": count, "undefined_count": state["count"] - count,
                        "mean": (state["metric_sum"][index].item() / count if count else None),
                        "sum": state["metric_sum"][index].item(),
                        "sum_squares": state["metric_square_sum"][index].item(),
                        "min": state["metric_min"][index].item(),
                        "max": state["metric_max"][index].item()}
    return _json_value(result)


class DistributionObserver:
    def __init__(self, output_dir, histogram_edges=None, bins=128):
        self.output_dir = Path(output_dir)
        self.bins = bins
        self.edges = dict(histogram_edges or {})
        self.edge_sources = {key: {"source": "explicit"} for key in self.edges}
        self.moments = {}
        self.windows = []
        self.calls = 0
        self.elapsed_seconds = 0.0
        self.input_dtypes = set()
        self.observed_token_vectors = 0
        self.max_selected_tensor_bytes = 0

    def state_dict(self):
        """Batch-checkpoint state; no model parameters or raw activations."""
        return {"bins": self.bins, "edges": self.edges,
                "edge_sources": _json_value(self.edge_sources), "moments": self.moments,
                "windows": _json_value(self.windows), "calls": self.calls,
                "elapsed_seconds": self.elapsed_seconds,
                "input_dtypes": sorted(self.input_dtypes),
                "observed_token_vectors": self.observed_token_vectors,
                "max_selected_tensor_bytes": self.max_selected_tensor_bytes}

    def load_state_dict(self, state):
        self.bins = state["bins"]
        self.edges = state["edges"]
        self.edge_sources = state["edge_sources"]
        self.moments = state["moments"]
        self.windows = state["windows"]
        self.calls = state["calls"]
        self.elapsed_seconds = state["elapsed_seconds"]
        self.input_dtypes = set(state["input_dtypes"])
        self.observed_token_vectors = state["observed_token_vectors"]
        self.max_selected_tensor_bytes = state["max_selected_tensor_bytes"]

    def __call__(self, meta, tensor):
        position = meta["position_name"]
        if position not in POSITIONS:
            return
        started = time.perf_counter()
        selected = tensor[:, meta["selected_positions"], :]
        self.input_dtypes.add(str(tensor.dtype))
        self.observed_token_vectors += selected.shape[0] * selected.shape[1]
        self.max_selected_tensor_bytes = max(self.max_selected_tensor_bytes,
                                             selected.numel() * selected.element_size())
        splits = meta["split"]
        if isinstance(splits, str):
            splits = [splits] * len(selected)
        edge_key = f"{meta['scale']}/{position}"
        if edge_key not in self.edges:
            fit_indices = [i for i, split in enumerate(splits) if split == "fit"]
            exploration = selected[fit_indices].detach().to(torch.float64)
            finite = exploration[torch.isfinite(exploration)]
            low, high = finite.amin().item(), finite.amax().item()
            if low == high:
                # A constant exploration batch still needs distinct histogram edges.
                low, high = low - 0.5, high + 0.5
            self.edges[edge_key] = torch.linspace(low, high, self.bins + 1, dtype=torch.float64)
            self.edge_sources[edge_key] = {
                "rule": "first_fit_batch_same_scale_position",
                "model": meta["model"], "layer": meta["layer"], "pass": meta["pass"],
                "sample_ids": [meta["sample_ids"][i] for i in fit_indices],
                "selected_positions": meta["selected_positions"],
                "split": "fit", "low": low, "high": high,
            }
        for split in dict.fromkeys(splits):
            indices = [i for i, value in enumerate(splits) if value == split]
            group_meta = {key: meta[key] for key in CONDITION_FIELDS if key != "split"}
            group_meta["split"] = split
            key = tuple(group_meta[field] for field in CONDITION_FIELDS)
            state, table, counts, directions = _statistics(selected[indices], self.edges[edge_key])
            if key in self.moments:
                merge_moments(self.moments[key], state)
            else:
                self.moments[key] = state
            for local, original in enumerate(indices):
                row = {**group_meta, "sample_id": meta["sample_ids"][original],
                       "selected_token_count": selected.shape[1],
                       "zero_tokens": int(counts[local, 2]),
                       "nonfinite_elements": int(counts[local, 3])}
                for metric, name in enumerate(METRICS):
                    count = int(table[local, metric, 2])
                    row[name] = table[local, metric, 0].item() / count if count else None
                    row[name + "_count"] = count
                    row[name + "_sum"] = table[local, metric, 0].item()
                nonzero = int(counts[local, 1])
                row["cosine_nonzero_tokens"] = nonzero
                row["mean_pair_cosine"] = (
                    (directions[local, 0] - directions[local, 1]).item() / (nonzero * (nonzero - 1))
                    if nonzero > 1 else None)
                self.windows.append(row)
        self.calls += 1
        self.elapsed_seconds += time.perf_counter() - started

    def write(self, output_dir=None):
        output = Path(output_dir or self.output_dir) / "a5_distribution"
        output.mkdir(parents=True, exist_ok=True)
        summaries = []
        for key, state in self.moments.items():
            summaries.append({**dict(zip(CONDITION_FIELDS, key)), **summarize_moments(state)})
        payload = {"statistics_dtype": "float64", "model_dtype_changed": False,
                   "observed_input_dtypes": sorted(self.input_dtypes),
                   "histogram_interval": "left_closed_right_open_except_closed_final_bin",
                   "histogram_sources": self.edge_sources, "histogram_edges": self.edges,
                   "observer_calls": self.calls, "observer_wall_seconds": self.elapsed_seconds,
                   "observed_token_vectors": self.observed_token_vectors,
                   "max_selected_tensor_bytes": self.max_selected_tensor_bytes,
                   "moment_tensor_bytes": sum(value.numel() * value.element_size()
                                              for state in self.moments.values()
                                              for value in state.values()
                                              if isinstance(value, torch.Tensor)),
                   "window_rows": len(self.windows),
                   "groups": summaries}
        (output / "summary.json").write_text(json.dumps(_json_value(payload), indent=2) + "\n")
        with (output / "windows.jsonl").open("w") as stream:
            for row in self.windows:
                stream.write(json.dumps(_json_value(row)) + "\n")
        torch.save({"moments": self.moments, "edges": self.edges,
                    "edge_sources": self.edge_sources}, output / "moments.pt")
        _write_comparisons(summaries, output / "comparisons.csv")
        return output


def _write_comparisons(summaries, path):
    rows = []
    lookup = {tuple(item[field] for field in CONDITION_FIELDS): item for item in summaries}
    for item in summaries:
        if str(item["model"]).lower() not in {"mixerloop", "mixer", "fullloop"}:
            continue
        baselines = [(item["model"], baseline_pass)
                     for baseline_pass in range(1, item["pass"])]
        baselines.append(("gdn", 1))
        for baseline_model, baseline_pass in baselines:
            values = {**item, "model": baseline_model, "pass": baseline_pass}
            key = tuple(values[field] for field in CONDITION_FIELDS)
            baseline = lookup.get(key)
            if baseline is None:
                continue
            for metric in (*METRICS, "mean_pair_cosine"):
                value = item[metric] if metric == "mean_pair_cosine" else item[metric]["mean"]
                base = baseline[metric] if metric == "mean_pair_cosine" else baseline[metric]["mean"]
                rows.append({**{field: item[field] for field in CONDITION_FIELDS},
                             "baseline_model": baseline_model, "baseline_pass": baseline_pass, "metric": metric,
                             "value": value, "baseline_value": base,
                             "delta": value - base if value is not None and base is not None else None})
    fields = (*CONDITION_FIELDS, "baseline_model", "baseline_pass", "metric", "value", "baseline_value", "delta")
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_observer(output_dir):
    return DistributionObserver(output_dir)


def smoke(output_dir, device):
    """Small numerical diagnostic; this is not a model experiment or acceptance gate."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    x = torch.tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [1., -2., 0., 4., 0., 3., -1., 2., 0., 1., -3., 0., 5.],
                       [-3., 1., 2., 0., 1., -2., 3., -4., 0., 1., -1., 2., 0.]],
                      [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [0., -4., -2., 1., 0., 0., 2., 3., -1., 0., -2., 1., 0.],
                       [7., 0., 0., -1., 2., 0., -3., 1., 0., 1., 0., 0., -2.]]],
                     dtype=torch.float32, device=device)
    edges = torch.tensor([-2., 0., 1., 2.], dtype=torch.float64)
    full, _, _, _ = _statistics(x, edges)
    first, _, _, _ = _statistics(x[:1], edges)
    second, _, _, _ = _statistics(x[1:], edges)
    merge_moments(first, second)
    flat = x.double().reshape(-1, x.shape[-1])
    nonzero = flat.norm(dim=-1) > 0
    units = flat[nonzero] / flat[nonzero].norm(dim=-1, keepdim=True)
    pairwise = units @ units.T
    off_diagonal = ~torch.eye(len(units), dtype=torch.bool, device=device)
    direct_cosine = pairwise[off_diagonal].mean().item()
    summary = summarize_moments(full)
    direct = {name: [] for name in METRICS}
    for row in flat.cpu().tolist():
        squares = sorted((value * value for value in row), reverse=True)
        total = sum(squares)
        direct["rms"].append(math.sqrt(total / len(row)))
        direct["max_abs"].append(max(abs(value) for value in row))
        for name, predicate in (("positive_fraction", lambda value: value > 0),
                                ("negative_fraction", lambda value: value < 0),
                                ("zero_fraction", lambda value: value == 0)):
            direct[name].append(sum(predicate(value) for value in row) / len(row))
        if total:
            direct["top_1pct_energy"].append(sum(squares[:math.ceil(len(row) * .01)]) / total)
            direct["top_10pct_energy"].append(sum(squares[:math.ceil(len(row) * .10)]) / total)
    errors = {"merged_mean_max_abs": (first["mean"] - full["mean"]).abs().max().item(),
              "merged_m2_max_abs": (first["m2"] - full["m2"]).abs().max().item(),
              "direct_variance_max_abs": (full["m2"] / len(flat) - flat.var(0, correction=0).cpu()).abs().max().item(),
              "direct_pair_cosine_abs": abs(summary["mean_pair_cosine"] - direct_cosine),
              "merged_histogram_count_max_abs": (first["histogram"] - full["histogram"]).abs().max().item()}
    errors.update({"direct_" + name + "_mean_abs":
                   abs(summary[name]["mean"] - sum(values) / len(values))
                   for name, values in direct.items()})
    raw = flat.cpu().flatten().tolist()
    edge_list = edges.tolist()
    direct_histogram = [sum(left <= value < right or (i == len(edge_list) - 2 and value == right)
                            for value in raw)
                        for i, (left, right) in enumerate(zip(edge_list, edge_list[1:]))]
    errors["direct_histogram_count_max_abs"] = max(abs(a - b) for a, b in
                                                   zip(summary["histogram"], direct_histogram))
    errors["direct_underflow_count_abs"] = abs(summary["underflow"] - sum(v < edge_list[0] for v in raw))
    errors["direct_overflow_count_abs"] = abs(summary["overflow"] - sum(v > edge_list[-1] for v in raw))
    observer = DistributionObserver(output, {"15m/ffn_input": edges})
    meta = {"model": "gdn", "scale": "15m", "layer": 3, "pass": 1,
            "position_name": "ffn_input", "sample_ids": ["smoke-0", "smoke-1"],
            "selected_positions": [0, 1, 2], "split": ["fit", "validation"]}
    observer(meta, x)
    state_path = output / "smoke_observer_state.pt"
    torch.save(observer.state_dict(), state_path)
    restored = DistributionObserver(output)
    restored.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    next_meta = {**meta, "sample_ids": ["smoke-2", "smoke-3"]}
    observer(next_meta, -x * .5)
    restored(next_meta, -x * .5)
    errors["resume_mean_max_abs"] = max(
        (state["mean"] - restored.moments[key]["mean"]).abs().max().item()
        for key, state in observer.moments.items())
    errors["resume_m2_max_abs"] = max(
        (state["m2"] - restored.moments[key]["m2"]).abs().max().item()
        for key, state in observer.moments.items())
    errors["resume_histogram_count_max_abs"] = max(
        (state["histogram"] - restored.moments[key]["histogram"]).abs().max().item()
        for key, state in observer.moments.items())
    errors["resume_window_count_abs"] = abs(len(observer.windows) - len(restored.windows))
    observer.write()
    evidence = {"purpose": "small_tensor_numerical_diagnostic_not_model_result",
                "device": str(x.device), "gpu": torch.cuda.get_device_name() if x.is_cuda else None,
                "torch_version": torch.__version__, "input": x.cpu().tolist(),
                "histogram_edges": edges.tolist(), "errors": errors,
                "direct_pair_cosine": direct_cosine, "streaming": summary}
    (output / "smoke.json").write_text(json.dumps(_json_value(evidence), indent=2) + "\n")
    print(json.dumps(errors))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--moments", type=Path, help="Reproduce summaries from a completed moments.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.smoke:
        import wandb

        args.output_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(project="physics_for_llm", entity="gyy0592-ucsc", mode="offline",
                         dir=str(args.output_dir), config={"idea": "I004-A5", "kind": "numeric_smoke",
                                                          "device": args.device})
        exit_code = 0
        try:
            evidence = smoke(args.output_dir, args.device)
            run.log(evidence["errors"])
            run.config.update({"gpu": evidence["gpu"], "torch_version": evidence["torch_version"]})
        except BaseException:
            exit_code = 1
            raise
        finally:
            run.finish(exit_code=exit_code)
    elif args.moments:
        saved = torch.load(args.moments, map_location="cpu", weights_only=True)
        observer = DistributionObserver(args.output_dir, saved["edges"])
        observer.moments = saved["moments"]
        observer.edge_sources = saved["edge_sources"]
        observer.write()


if __name__ == "__main__":
    main()
