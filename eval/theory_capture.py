#!/usr/bin/env python3
"""Spec 01 streaming capture utilities for the LT3 theory evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

SELECTION_SEED = 20260722
DEFAULT_SAMPLE_COUNT = 1024
DEFAULT_SEQ_LEN = 512
DEFAULT_METRIC_START = 32
DEFAULT_METRIC_END = 510
VALIDATION_FILE = "shard_06542.bin"

AUTHORIZED_OUTPUT_LIMIT_BYTES = 100_000_000_000
OUTPUT_ROOT_CAP_BYTES = 99_000_000_000
BYTE_BUDGET = {
    "full_hidden_plus_logit_audit": 46_000_000_000,
    "hidden_state_audit": 13_500_000_000,
    "per_sample_sufficient_statistics": 8_000_000_000,
    "exact_online_mitr_gram_sufficient_statistics": 4_000_000_000,
    "results_figures_provenance": 4_000_000_000,
    "selected_probe_checkpoints": 20_000_000_000,
    "temporary_reserve": 3_500_000_000,
}

FULL_AUDIT_SAMPLE_COUNT = 4
HIDDEN_AUDIT_SAMPLE_COUNT = 64


class OutputBudgetError(RuntimeError):
    """Raised before a write that would exceed the configured output-root cap."""


@dataclass(frozen=True)
class RecordSpec:
    checkpoint: str
    scale: str
    condition: str
    layer: int
    pass_index: int
    record_type: str

    def to_row(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "scale": self.scale,
            "condition": self.condition,
            "layer": self.layer,
            "pass": self.pass_index,
            "record_type": self.record_type,
        }


@dataclass(frozen=True)
class AuditPlan:
    full_hidden_plus_logit_sample_ids: list[int]
    hidden_state_sample_ids: list[int]
    hidden_only_sample_ids: list[int]

    def should_persist(self, sample_id: int, kind: str) -> bool:
        if kind == "logits":
            return sample_id in self.full_hidden_plus_logit_sample_ids
        if kind == "hidden":
            return sample_id in self.hidden_state_sample_ids
        raise ValueError(f"Unknown audit tensor kind: {kind!r}")


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def root_size_bytes(path: Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def assert_output_write_allowed(output_root: Path, estimated_write_bytes: int, cap_bytes: int = OUTPUT_ROOT_CAP_BYTES) -> None:
    current = root_size_bytes(output_root)
    if current + int(estimated_write_bytes) > int(cap_bytes):
        raise OutputBudgetError(
            f"Output root {output_root} would exceed cap: "
            f"{current} existing bytes + {estimated_write_bytes} write bytes > {cap_bytes}"
        )


def write_bytes_guarded(path: Path, payload: bytes, *, output_root: Path, cap_bytes: int = OUTPUT_ROOT_CAP_BYTES) -> None:
    output_root = Path(output_root)
    path = Path(path)
    assert_output_write_allowed(output_root, len(payload), cap_bytes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_json_guarded(path: Path, payload: Any, *, output_root: Path, cap_bytes: int = OUTPUT_ROOT_CAP_BYTES) -> None:
    write_bytes_guarded(path, _json_bytes(payload), output_root=output_root, cap_bytes=cap_bytes)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validation_bin_path(data_dir: Path) -> Path:
    return Path(data_dir) / VALIDATION_FILE


def validation_window_count(token_count: int, seq_len: int = DEFAULT_SEQ_LEN) -> int:
    return int(token_count) // int(seq_len) - 1


def _split_counts(sample_count: int) -> tuple[int, int]:
    if sample_count <= 1:
        return sample_count, 0
    fit_count = round(sample_count * 819 / 1024)
    fit_count = max(1, min(sample_count - 1, fit_count))
    return fit_count, sample_count - fit_count


def build_fixed_sample_artifacts(
    data_dir: Path,
    output_dir: Path,
    *,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    seed: int = SELECTION_SEED,
    seq_len: int = DEFAULT_SEQ_LEN,
    metric_start: int = DEFAULT_METRIC_START,
    metric_end: int = DEFAULT_METRIC_END,
    cap_bytes: int = OUTPUT_ROOT_CAP_BYTES,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    bin_path = validation_bin_path(data_dir)
    tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
    num_windows = validation_window_count(len(tokens), seq_len)
    if sample_count > num_windows:
        raise ValueError(f"Requested {sample_count} samples but validation shard has only {num_windows} windows")

    sample_ids = sorted(random.Random(seed).sample(range(num_windows), sample_count))
    records = []
    for sample_id in sample_ids:
        start = sample_id * seq_len
        input_tokens = np.asarray(tokens[start : start + seq_len], dtype=np.uint16)
        records.append(
            {
                "sample_id": int(sample_id),
                "shard": VALIDATION_FILE,
                "window_index": int(sample_id),
                "start_token": int(start),
                "num_tokens": int(seq_len),
                "num_non_padding_tokens": int(np.count_nonzero(input_tokens)),
                "input_sha256": sha256_bytes(input_tokens.tobytes()),
            }
        )

    fit_count, heldout_count = _split_counts(sample_count)
    selection = {
        "seed": int(seed),
        "sample_count": int(sample_count),
        "seq_len": int(seq_len),
        "validation_file": VALIDATION_FILE,
        "num_validation_windows": int(num_windows),
        "metric_positions": {"start": int(metric_start), "end": int(metric_end)},
        "records": records,
    }
    split = {
        "seed": int(seed),
        "sample_count": int(sample_count),
        "fit_count": int(fit_count),
        "heldout_count": int(heldout_count),
        "fit_sample_ids": sample_ids[:fit_count],
        "heldout_sample_ids": sample_ids[fit_count:],
    }

    write_json_guarded(output_dir / "theory_eval_ids.json", selection, output_root=output_dir, cap_bytes=cap_bytes)
    write_json_guarded(output_dir / "probe_split_metadata.json", split, output_root=output_dir, cap_bytes=cap_bytes)
    return {"selection": selection, "split": split}


def build_audit_plan(sample_ids: Iterable[int]) -> AuditPlan:
    ordered = list(sample_ids)
    full = ordered[:FULL_AUDIT_SAMPLE_COUNT]
    hidden = ordered[:HIDDEN_AUDIT_SAMPLE_COUNT]
    full_set = set(full)
    return AuditPlan(
        full_hidden_plus_logit_sample_ids=full,
        hidden_state_sample_ids=hidden,
        hidden_only_sample_ids=[sample_id for sample_id in hidden if sample_id not in full_set],
    )


def audit_tensor_filename(
    *,
    checkpoint: str,
    scale: str,
    condition: str,
    layer: int,
    pass_index: int,
    sample_id: int,
    record_type: str,
    kind: str,
) -> str:
    if kind not in {"hidden", "logits"}:
        raise ValueError(f"Unknown audit tensor kind: {kind!r}")
    return (
        f"checkpoint={checkpoint}__scale={scale}__condition={condition}__layer={layer}"
        f"__pass={pass_index}__sample_id={sample_id}__record_type={record_type}__kind={kind}.pt"
    )


def build_record_specs(
    *,
    condition: str,
    scale: str,
    num_layers: int,
    loop_count: int,
    checkpoint: str | None = None,
) -> list[RecordSpec]:
    checkpoint = checkpoint or f"{condition}-{scale}"
    records: list[RecordSpec] = []
    for layer in range(int(num_layers)):
        if condition == "gdn":
            records.append(RecordSpec(checkpoint, scale, condition, layer, 1, "hA"))
        elif condition == "mixerloop":
            for pass_index in range(1, int(loop_count) + 1):
                records.append(RecordSpec(checkpoint, scale, condition, layer, pass_index, "hA"))
            records.append(RecordSpec(checkpoint, scale, condition, layer, int(loop_count), "hF"))
        elif condition == "fullloop":
            for pass_index in range(1, int(loop_count) + 1):
                records.append(RecordSpec(checkpoint, scale, condition, layer, pass_index, "hA"))
                records.append(RecordSpec(checkpoint, scale, condition, layer, pass_index, "hF"))
        else:
            raise ValueError(f"Unknown condition: {condition!r}")
    return records


def _limited_layers(layers: Iterable[torch.nn.Module], max_layers: int | None) -> list[tuple[int, torch.nn.Module]]:
    pairs = list(enumerate(layers))
    if max_layers is None:
        return pairs
    return pairs[: int(max_layers)]


def _loop_count(model: torch.nn.Module) -> int:
    backbone = model.model
    value = getattr(backbone, "loop_count", None)
    if value is None and hasattr(model, "config"):
        value = getattr(model.config, "loop_count", None)
    return int(value or 1)


def model_state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _mixerloop_records(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    condition: str,
    scale: str,
    max_layers: int | None,
) -> list[tuple[RecordSpec, torch.Tensor]]:
    backbone = model.model
    checkpoint = f"{condition}-{scale}"
    hidden = backbone.embeddings(input_ids)
    records: list[tuple[RecordSpec, torch.Tensor]] = []
    loop_count = _loop_count(model)
    residual_weight = getattr(backbone, "residual_weight", None)

    for layer_index, layer in _limited_layers(backbone.layers, max_layers):
        for loop_index in range(loop_count):
            h_input = hidden
            attn_out, _, _ = layer.mixer(layer.attn_norm(hidden))
            hidden = hidden + attn_out
            if residual_weight is not None:
                hidden = hidden + residual_weight[loop_index].view(1, 1, -1) * h_input
            records.append(
                (
                    RecordSpec(checkpoint, scale, condition, layer_index, loop_index + 1, "hA"),
                    hidden.detach(),
                )
            )
        hidden = hidden + layer.ffn(layer.ffn_norm(hidden))
        records.append(
            (
                RecordSpec(checkpoint, scale, condition, layer_index, loop_count, "hF"),
                hidden.detach(),
            )
        )
    return records


def _fullloop_records(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    condition: str,
    scale: str,
    max_layers: int | None,
) -> list[tuple[RecordSpec, torch.Tensor]]:
    backbone = model.model
    checkpoint = f"{condition}-{scale}"
    hidden = backbone.embeddings(input_ids)
    records: list[tuple[RecordSpec, torch.Tensor]] = []
    loop_count = _loop_count(model)
    residual_weight = getattr(backbone, "residual_weight", None)

    for loop_index in range(loop_count):
        h_input = hidden
        for layer_index, layer in _limited_layers(backbone.layers, max_layers):
            attn_out, _, _ = layer.mixer(layer.attn_norm(hidden))
            hidden = hidden + attn_out
            records.append(
                (
                    RecordSpec(checkpoint, scale, condition, layer_index, loop_index + 1, "hA"),
                    hidden.detach(),
                )
            )
            hidden = hidden + layer.ffn(layer.ffn_norm(hidden))
            records.append(
                (
                    RecordSpec(checkpoint, scale, condition, layer_index, loop_index + 1, "hF"),
                    hidden.detach(),
                )
            )
        if residual_weight is not None:
            hidden = hidden + residual_weight[loop_index].view(1, 1, -1) * h_input
    return records


def _gdn_records(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    condition: str,
    scale: str,
    max_layers: int | None,
) -> list[tuple[RecordSpec, torch.Tensor]]:
    backbone = model.model
    checkpoint = f"{condition}-{scale}"
    hidden = backbone.embeddings(input_ids)
    records: list[tuple[RecordSpec, torch.Tensor]] = []

    for layer_index, layer in _limited_layers(backbone.layers, max_layers):
        residual = hidden
        attn_out, _, _ = layer.attn(
            hidden_states=layer.attn_norm(hidden),
            use_cache=False,
            output_attentions=False,
        )
        h_a = residual + attn_out
        records.append((RecordSpec(checkpoint, scale, condition, layer_index, 1, "hA"), h_a.detach()))

        if getattr(layer.config, "fuse_norm", False):
            normed, residual_after_norm = layer.mlp_norm(attn_out, residual, True)
            hidden = residual_after_norm + layer.mlp(normed)
        else:
            hidden = h_a + layer.mlp(layer.mlp_norm(h_a))
    return records


def collect_record_hidden_states(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    condition: str,
    scale: str,
    max_layers: int | None = None,
) -> list[tuple[RecordSpec, torch.Tensor]]:
    """Compute required record-location hidden states in memory without persisting raw tensors."""

    model.eval()
    with torch.no_grad():
        if condition == "mixerloop":
            return _mixerloop_records(model, input_ids, condition=condition, scale=scale, max_layers=max_layers)
        if condition == "fullloop":
            return _fullloop_records(model, input_ids, condition=condition, scale=scale, max_layers=max_layers)
        if condition == "gdn":
            return _gdn_records(model, input_ids, condition=condition, scale=scale, max_layers=max_layers)
    raise ValueError(f"Unknown condition: {condition!r}")


def logits_from_record_hidden(model: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Apply the model's final normalization and LM head to a record hidden state."""

    model.eval()
    with torch.no_grad():
        return model.lm_head(model.model.norm(hidden))


def next_token_sufficient_stats(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    metric_start: int = DEFAULT_METRIC_START,
    metric_end: int = DEFAULT_METRIC_END,
) -> dict[str, float | int]:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, seq, vocab], got {tuple(logits.shape)}")
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, seq], got {tuple(input_ids.shape)}")
    if metric_start < 0 or metric_end >= input_ids.shape[1] - 1 or metric_end < metric_start:
        raise ValueError("metric positions must be within input positions that have a next-token label")

    position_slice = slice(metric_start, metric_end + 1)
    labels = input_ids[:, metric_start + 1 : metric_end + 2].reshape(-1)
    selected_logits = logits[:, position_slice, :].reshape(labels.numel(), logits.shape[-1]).float()
    ce_sum = F.cross_entropy(selected_logits, labels, reduction="sum").item()
    topk = selected_logits.topk(min(5, selected_logits.shape[-1]), dim=-1).indices
    top1_correct = int((topk[:, 0] == labels).sum().item())
    top5_correct = int((topk == labels[:, None]).any(dim=-1).sum().item())
    return {
        "ce_sum": float(ce_sum),
        "num_positions": int(labels.numel()),
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
    }


def build_stats_row(
    *,
    checkpoint: str,
    scale: str,
    condition: str,
    sample_id: int,
    readout: str,
    stats: dict[str, Any],
    precision: str,
    input_sha256: str,
    layer: int | None,
    pass_index: int | None,
    record_type: str,
) -> dict[str, Any]:
    return {
        "checkpoint": checkpoint,
        "scale": scale,
        "condition": condition,
        "sample_id": int(sample_id),
        "layer": "" if layer is None else int(layer),
        "pass": "" if pass_index is None else int(pass_index),
        "record_type": record_type,
        "readout": readout,
        "precision": precision,
        "input_sha256": input_sha256,
        "ce_sum": float(stats["ce_sum"]),
        "num_positions": int(stats["num_positions"]),
        "top1_correct": int(stats["top1_correct"]),
        "top5_correct": int(stats["top5_correct"]),
    }


def write_stats_csv_guarded(rows: list[dict[str, Any]], path: Path, *, output_root: Path) -> None:
    if not rows:
        payload = b""
    else:
        import io

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        payload = buffer.getvalue().encode("utf-8")
    write_bytes_guarded(path, payload, output_root=output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    ids = subparsers.add_parser("ids", help="write deterministic Spec 01 sample and split artifacts")
    ids.add_argument("--data_dir", type=Path, default=Path("data/climbmix-10b"))
    ids.add_argument("--output_dir", type=Path, default=Path("outputs/theory_eval_local_20260722"))
    ids.add_argument("--sample_count", type=int, default=DEFAULT_SAMPLE_COUNT)
    ids.add_argument("--seed", type=int, default=SELECTION_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "ids":
        artifacts = build_fixed_sample_artifacts(
            args.data_dir,
            args.output_dir,
            sample_count=args.sample_count,
            seed=args.seed,
        )
        print(json.dumps({"records": len(artifacts["selection"]["records"]), "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
