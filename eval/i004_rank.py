"""I004 A4: complete raw/centered spectra from streamed token vectors.

The capture callback receives ``observer(meta, full_sequence_tensor)``. Metadata
contains model, scale, layer, pass, position_name, sample_ids, selected_positions
and split. This module owns only statistics; the caller owns model execution,
sample selection and observation positions. No captured activation is retained.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import torch


POSITIONS = ("attention_post", "ffn_input", "block_post")
GROUP_FIELDS = ("scale", "model", "layer", "pass", "position_name")


class RankAccumulator:
    """Accumulate full FP64 raw and centered second moments, one batch at a time."""

    def __init__(self, width: int, device: str | torch.device):
        self.count = 0
        self.vector_sum = torch.zeros(width, dtype=torch.float64, device=device)
        self.mean = torch.zeros_like(self.vector_sum)
        self.raw_gram = torch.zeros(width, width, dtype=torch.float64, device=device)
        self.centered_scatter = torch.zeros_like(self.raw_gram)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        rows = values.detach().reshape(-1, values.shape[-1]).to(
            device=self.vector_sum.device, dtype=torch.float64
        )
        batch_count = rows.shape[0]
        batch_mean = rows.mean(dim=0)
        centered = rows - batch_mean
        new_count = self.count + batch_count
        delta = batch_mean - self.mean
        # Merge centered scatter without subtracting two large raw moments.
        self.centered_scatter.addmm_(centered.T, centered)
        self.centered_scatter.add_(
            torch.outer(delta, delta), alpha=self.count * batch_count / new_count
        )
        self.mean.add_(delta, alpha=batch_count / new_count)
        self.raw_gram.addmm_(rows.T, rows)
        self.vector_sum.add_(rows.sum(dim=0))
        self.count = new_count

    def tensor_state(self) -> dict:
        return {
            "count": self.count,
            "vector_sum": self.vector_sum.cpu(),
            "mean": self.mean.cpu(),
            "raw_gram": self.raw_gram.cpu(),
            "centered_scatter": self.centered_scatter.cpu(),
        }

    @torch.no_grad()
    def spectra(self) -> dict:
        return {
            "raw": spectrum_from_scatter(self.raw_gram, self.count),
            "centered": spectrum_from_scatter(self.centered_scatter, self.count, centered=True),
        }


@torch.no_grad()
def spectrum_from_scatter(scatter: torch.Tensor, count: int, centered: bool = False) -> dict:
    """Report signed eigenvalues and finite-precision diagnostics before clipping.

Negative computed eigenvalues are retained in the report; only square-root
inputs are clipped to zero. Near-zero classifications are numerical diagnostics,
never a claim of exact matrix rank. No epsilon is added to spectral weights.
"""
    symmetric = (scatter + scatter.T) * 0.5
    eigenvalues = torch.linalg.eigvalsh(symmetric).flip(0)
    width = scatter.shape[0]
    tolerance = width * torch.finfo(scatter.dtype).eps * eigenvalues.abs().max()
    clipped = eigenvalues.clamp_min(0)
    singular = clipped.sqrt()
    singular_sum = singular.sum()
    square_sum = clipped.sum()
    nonzero = singular > 0
    if singular_sum.item() == 0:
        probabilities = torch.zeros_like(singular)
        effective_rank = None
        energy_dimensions = {"90": None, "95": None}
        energy_fraction = torch.zeros_like(singular)
    else:
        probabilities = singular / singular_sum
        entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
        effective_rank = entropy.exp().item()
        energy_fraction = clipped / square_sum
        cumulative = energy_fraction.cumsum(dim=0)
        energy_dimensions = {
            str(int(fraction * 100)): min(
                int(torch.searchsorted(cumulative, fraction).item()) + 1, width
            )
            for fraction in (0.90, 0.95)
        }
    trace = scatter.diagonal().sum().item()
    return {
        "count": count,
        "width": width,
        "singular_values": singular.cpu().tolist(),
        "normalized_singular_values": probabilities.cpu().tolist(),
        "energy_fraction": energy_fraction.cpu().tolist(),
        "eigenvalues_before_clipping": eigenvalues.cpu().tolist(),
        "entropy_effective_rank": effective_rank,
        "energy_dimensions_90": energy_dimensions["90"],
        "energy_dimensions_95": energy_dimensions["95"],
        "sum_squares_from_scatter": trace,
        "sum_squares_after_eigenvalue_clipping": square_sum.item(),
        "mean_square_norm": trace / count,
        "sample_total_variance": trace / (count - 1) if centered and count > 1 else None,
        "zero_energy": singular_sum.item() == 0,
        "negative_eigenvalue_count": int((eigenvalues < 0).sum().item()),
        "negative_eigenvalue_mass": (-eigenvalues[eigenvalues < 0]).sum().item(),
        "numerical_eigenvalue_tolerance": tolerance.item(),
        "within_numerical_tolerance_count": int((eigenvalues.abs() <= tolerance).sum().item()),
        "negative_beyond_tolerance_count": int((eigenvalues < -tolerance).sum().item()),
        "maximum_scatter_asymmetry": (scatter - scatter.T).abs().max().item(),
        "exact_rank_claim": False,
    }


class RankObserver:
    """Select scored positions and accumulate all windows per model/layer/pass."""

    def __init__(self, output_dir: str | Path, device: str | None = None):
        self.output_dir = Path(output_dir)
        self.device = device
        self.groups: dict[tuple, RankAccumulator] = {}
        self.metadata: dict[tuple, dict] = {}
        self.sample_ids: dict[tuple, list] = {}
        self.started_at = time.perf_counter()
        self.input_dtypes: set[str] = set()

    @torch.no_grad()
    def __call__(self, meta: dict, tensor: torch.Tensor) -> None:
        if meta["position_name"] in POSITIONS:
            indices = torch.as_tensor(meta["selected_positions"], device=tensor.device, dtype=torch.long)
            selected = tensor.index_select(1, indices)
            self.update_selected(meta, selected)

    @torch.no_grad()
    def update_selected(self, meta: dict, tensor: torch.Tensor) -> None:
        """Consume already selected original-space vectors from a completed cache."""
        key = tuple(meta[field] for field in GROUP_FIELDS)
        if key not in self.groups:
            device = self.device or str(tensor.device)
            self.groups[key] = RankAccumulator(tensor.shape[-1], device)
            self.metadata[key] = {field: meta[field] for field in GROUP_FIELDS}
            self.metadata[key]["selected_positions"] = [int(position) for position in meta["selected_positions"]]
            self.sample_ids[key] = []
        self.groups[key].update(tensor)
        self.sample_ids[key].extend(int(sample_id) for sample_id in meta["sample_ids"])
        self.input_dtypes.add(str(tensor.dtype))

    def state_dict(self) -> dict:
        """Snapshot every observed batch as CPU tensors and primitive metadata.

        A1 owns when this snapshot is committed together with cache progress.
        CPU copies are independent of the live accumulator so later updates do
        not change an already returned snapshot. Accumulation stays FP64.
        """
        entries = []
        for key, accumulator in self.groups.items():
            moments = accumulator.tensor_state()
            entries.append({
                "key": list(key),
                "metadata": dict(self.metadata[key]),
                "sample_ids": list(self.sample_ids[key]),
                "device": str(accumulator.vector_sum.device),
                "moments": {
                    name: value.clone() if isinstance(value, torch.Tensor) else value
                    for name, value in moments.items()
                },
            })
        return {
            "interface_version": 1,
            "entries": entries,
            "input_dtypes": sorted(self.input_dtypes),
            "elapsed_seconds": time.perf_counter() - self.started_at,
        }

    def load_state_dict(self, state: dict) -> None:
        """Replace accumulated observations from A1's last committed snapshot.

        An explicitly selected current device controls restoration; otherwise
        each group's saved device is used. No model or cache state is modified.
        """
        self.groups = {}
        self.metadata = {}
        self.sample_ids = {}
        for entry in state["entries"]:
            key = tuple(entry["key"])
            moments = entry["moments"]
            device = self.device or entry["device"]
            accumulator = RankAccumulator(moments["vector_sum"].numel(), device)
            accumulator.count = moments["count"]
            for name in ("vector_sum", "mean", "raw_gram", "centered_scatter"):
                getattr(accumulator, name).copy_(moments[name])
            self.groups[key] = accumulator
            self.metadata[key] = dict(entry["metadata"])
            self.sample_ids[key] = list(entry["sample_ids"])
        self.input_dtypes = set(state["input_dtypes"])
        self.started_at = time.perf_counter() - state["elapsed_seconds"]

    def write(self, output_dir: str | Path | None = None) -> dict:
        directory = Path(output_dir) if output_dir is not None else self.output_dir
        directory.mkdir(parents=True, exist_ok=True)
        records = []
        states = []
        for key, accumulator in self.groups.items():
            spectra = accumulator.spectra()
            metadata = dict(self.metadata[key])
            metadata["sample_ids"] = self.sample_ids[key]
            metadata["window_observations"] = len(self.sample_ids[key])
            metadata["unique_window_count"] = len(set(self.sample_ids[key]))
            metadata["accumulation_device"] = str(accumulator.vector_sum.device)
            states.append({"metadata": metadata, **accumulator.tensor_state()})
            for form, spectrum in spectra.items():
                records.append({**metadata, "form": form, **spectrum})
        comparisons = compare_passes(records)
        report = {
            "study": "I004 A4",
            "accumulation_dtype": "torch.float64",
            "input_dtypes": sorted(self.input_dtypes),
            "aggregation": "all observed windows; token rows; original coordinates",
            "elapsed_seconds_including_capture": time.perf_counter() - self.started_at,
            "records": records,
            "comparisons": comparisons,
        }
        (directory / "rank_spectra.json").write_text(json.dumps(report, indent=2) + "\n")
        torch.save(states, directory / "rank_moments.pt")
        scalar_rows = [
            {field: value for field, value in row.items() if not isinstance(value, list)}
            for row in records
        ]
        if scalar_rows:
            with (directory / "rank_summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(scalar_rows[0]))
                writer.writeheader()
                writer.writerows(scalar_rows)
        return report


def compare_passes(records: list[dict]) -> list[dict]:
    """Within each model/layer/position/form, report later pass minus every earlier pass."""
    by_condition = {
        tuple(row[field] for field in GROUP_FIELDS) + (row["form"],): row
        for row in records
    }
    comparisons = []
    metrics = (
        "entropy_effective_rank", "energy_dimensions_90", "energy_dimensions_95",
        "sum_squares_from_scatter", "mean_square_norm", "sample_total_variance",
    )
    for row in records:
        for baseline_pass in range(1, row["pass"]):
            key = tuple(
                baseline_pass if field == "pass" else row[field] for field in GROUP_FIELDS
            ) + (row["form"],)
            baseline = by_condition.get(key)
            if baseline is not None:
                delta = {
                    metric: row[metric] - baseline[metric]
                    if row[metric] is not None and baseline[metric] is not None else None
                    for metric in metrics
                }
                comparisons.append({
                    **{field: row[field] for field in GROUP_FIELDS}, "form": row["form"],
                    "baseline_pass": baseline_pass, "operation_minus_baseline": delta,
                    "same_sample_ids": row["sample_ids"] == baseline["sample_ids"],
                    "same_selected_positions": row["selected_positions"] == baseline["selected_positions"],
                })
    return comparisons


def make_observer(output_dir: str | Path) -> RankObserver:
    """A1 factory: observer follows the tensor device and preserves FP64 moments."""
    return RankObserver(output_dir)


def smoke(output_dir: Path, device: str, widths: list[int]) -> dict:
    """Post-implementation diagnostic: streamed full spectra versus direct SVD."""
    started = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(0)
    rows = []
    for width in widths:
        matrix = torch.randn(width + 17, width, generator=generator, device=device, dtype=torch.float64)
        matrix += torch.linspace(-1, 1, width, device=device, dtype=torch.float64)
        accumulator = RankAccumulator(width, device)
        for batch in matrix.split(61):
            accumulator.update(batch)
        observed = accumulator.spectra()
        for form, values in (("raw", matrix), ("centered", matrix - matrix.mean(dim=0))):
            direct = torch.linalg.svdvals(values)
            streamed = torch.tensor(observed[form]["singular_values"], device=device, dtype=torch.float64)
            probabilities = direct / direct.sum()
            direct_rank = (-(probabilities * probabilities.log()).sum()).exp().item()
            rows.append({
                "width": width, "form": form, "count": matrix.shape[0],
                "max_absolute_singular_difference": (streamed - direct).abs().max().item(),
                "relative_singular_l2_difference": ((streamed - direct).norm() / direct.norm()).item(),
                "effective_rank_difference": observed[form]["entropy_effective_rank"] - direct_rank,
                "square_sum_difference": observed[form]["sum_squares_from_scatter"] - values.square().sum().item(),
                "negative_beyond_tolerance_count": observed[form]["negative_beyond_tolerance_count"],
            })
    report = {
        "kind": "post_implementation_numeric_smoke", "device": device,
        "torch_version": torch.__version__, "seed": 0, "dtype": "torch.float64",
        "elapsed_seconds": time.perf_counter() - started, "comparisons": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rank_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--widths", nargs="+", type=int, default=[288, 512, 768])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.smoke:
        import wandb

        args.output_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            entity="gyy0592-ucsc", project="physics_for_llm", mode="offline",
            dir=str(args.output_dir.resolve()), job_type="a4-numeric-smoke",
            config={"device": args.device, "widths": args.widths, "seed": 0,
                    "dtype": "torch.float64", "output_dir": str(args.output_dir.resolve())},
        )
        exit_code = 1
        try:
            report = smoke(args.output_dir, args.device, args.widths)
            report["wandb_run_dir"] = run.dir
            report["wandb_run_id"] = run.id
            for row in report["comparisons"]:
                run.log({**row, "smoke_elapsed_seconds": report["elapsed_seconds"]})
            (args.output_dir / "rank_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
            exit_code = 0
        finally:
            run.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
