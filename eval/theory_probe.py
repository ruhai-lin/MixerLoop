#!/usr/bin/env python3
"""Spec 02 probe/readout helpers for the LT3 theory evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from eval.theory_capture import (
    DEFAULT_METRIC_END,
    DEFAULT_METRIC_START,
    OUTPUT_ROOT_CAP_BYTES,
    OutputBudgetError,
    build_record_specs,
    build_stats_row,
    logits_from_record_hidden,
    model_state_digest,
    next_token_sufficient_stats,
    root_size_bytes,
    sha256_bytes,
    write_json_guarded,
)

PROBE_DETAIL_COLUMNS = [
    "checkpoint",
    "scale",
    "condition",
    "layer",
    "pass",
    "record_type",
    "readout",
    "heldout_loss",
    "top1",
    "top5",
    "ci_low",
    "ci_high",
]


@dataclass(frozen=True)
class FixedProbeDataset:
    spec01_root: Path
    seed: int
    sample_ids: list[int]
    fit_sample_ids: list[int]
    heldout_sample_ids: list[int]
    records_by_sample_id: dict[int, dict[str, Any]]
    metric_start: int
    metric_end: int
    theory_eval_ids_sha256: str
    probe_split_metadata_sha256: str


@dataclass(frozen=True)
class Spec02RunConfig:
    spec01_root: Path
    output_root: Path
    wandb_enabled: bool = False
    formal: bool = False


@dataclass(frozen=True)
class ProbeTarget:
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
class ProbeTrainingConfig:
    optimizer: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    token_batch_size: int = 2048
    epochs: int = 5
    keep_tail_batch: bool = True
    selection_rule: str = "best_heldout_loss"
    fused_adamw_on_cuda: bool = True
    torch_compile_on_cuda: bool = True
    pin_memory_cpu_batches: bool = True
    non_blocking_cpu_to_gpu_transfer: bool = True
    record_gpu_trace: bool = True


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def load_fixed_probe_dataset(spec01_root: Path) -> FixedProbeDataset:
    spec01_root = Path(spec01_root)
    ids_path = spec01_root / "theory_eval_ids.json"
    split_path = spec01_root / "probe_split_metadata.json"
    selection = _load_json(ids_path)
    split = _load_json(split_path)

    records = list(selection["records"])
    sample_ids = [int(record["sample_id"]) for record in records]
    fit_sample_ids = [int(sample_id) for sample_id in split["fit_sample_ids"]]
    heldout_sample_ids = [int(sample_id) for sample_id in split["heldout_sample_ids"]]
    sample_id_set = set(sample_ids)
    if len(sample_id_set) != len(sample_ids):
        raise ValueError("theory_eval_ids.json contains duplicate sample_id values")
    if set(fit_sample_ids) & set(heldout_sample_ids):
        raise ValueError("probe split fit and heldout ids overlap")
    if set(fit_sample_ids) | set(heldout_sample_ids) != sample_id_set:
        raise ValueError("probe split ids do not exactly match theory_eval_ids records")
    if int(selection["seed"]) != int(split["seed"]):
        raise ValueError("selection and split seeds differ")

    metric_positions = selection["metric_positions"]
    return FixedProbeDataset(
        spec01_root=spec01_root,
        seed=int(selection["seed"]),
        sample_ids=sample_ids,
        fit_sample_ids=fit_sample_ids,
        heldout_sample_ids=heldout_sample_ids,
        records_by_sample_id={int(record["sample_id"]): dict(record) for record in records},
        metric_start=int(metric_positions["start"]),
        metric_end=int(metric_positions["end"]),
        theory_eval_ids_sha256=_sha256_file(ids_path),
        probe_split_metadata_sha256=_sha256_file(split_path),
    )


def parse_released_checkpoint_inventory(outputs_root: Path) -> list[dict[str, Any]]:
    outputs_root = Path(outputs_root)
    rows: list[dict[str, Any]] = []
    for checkpoint_dir in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
        checkpoint = checkpoint_dir.name
        if "-" not in checkpoint:
            continue
        condition, scale = checkpoint.rsplit("-", 1)
        if condition not in {"gdn", "mixerloop", "fullloop"} or scale not in {"15m", "42m", "110m"}:
            continue
        config_path = checkpoint_dir / "config.json"
        weights_path = checkpoint_dir / "model.safetensors"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing checkpoint config: {config_path}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"missing checkpoint weights: {weights_path}")
        config = _load_json(config_path)
        num_layers = config.get("num_hidden_layers", config.get("n_layer", config.get("num_layers")))
        if num_layers is None:
            raise ValueError(f"checkpoint config lacks layer count: {config_path}")
        rows.append(
            {
                "checkpoint": checkpoint,
                "condition": condition,
                "scale": scale,
                "checkpoint_path": str(checkpoint_dir),
                "config_path": str(config_path),
                "weights_path": str(weights_path),
                "config_sha256": _sha256_file(config_path),
                "model_type": str(config.get("model_type", "")),
                "num_layers": int(num_layers),
                "loop_count": int(config.get("loop_count", 1)),
                "hidden_size": int(config.get("hidden_size", 0)),
                "weights_loaded": False,
            }
        )
    expected = {f"{condition}-{scale}" for condition in ("gdn", "mixerloop", "fullloop") for scale in ("15m", "42m", "110m")}
    found = {row["checkpoint"] for row in rows}
    missing = sorted(expected - found)
    if missing:
        raise FileNotFoundError(f"missing released checkpoints: {missing}")
    order = {name: index for index, name in enumerate(sorted(expected, key=lambda name: (name.rsplit('-', 1)[1], name.rsplit('-', 1)[0])))}
    return sorted(rows, key=lambda row: order[row["checkpoint"]])


def enumerate_probe_targets(*, condition: str, scale: str, num_layers: int, loop_count: int) -> list[ProbeTarget]:
    return [
        ProbeTarget(
            checkpoint=record.checkpoint,
            scale=record.scale,
            condition=record.condition,
            layer=record.layer,
            pass_index=record.pass_index,
            record_type=record.record_type,
        )
        for record in build_record_specs(
            condition=condition,
            scale=scale,
            num_layers=num_layers,
            loop_count=loop_count,
        )
        if not (record.condition == "mixerloop" and record.record_type == "hF")
    ]


def build_runtime_target_manifest(checkpoint_inventory: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoint_inventory:
        for target in enumerate_probe_targets(
            condition=str(checkpoint["condition"]),
            scale=str(checkpoint["scale"]),
            num_layers=int(checkpoint["num_layers"]),
            loop_count=int(checkpoint.get("loop_count", 1)),
        ):
            row = target.to_row()
            row.update(
                {
                    "checkpoint_path": str(checkpoint["checkpoint_path"]),
                    "config_sha256": str(checkpoint["config_sha256"]),
                    "readouts": ["linear_probe", "frozen_lm_head"],
                    "runtime_status": "NOT_RUN",
                }
            )
            rows.append(row)
    return rows


def build_metric_token_batches(
    *,
    fit_sample_ids: Sequence[int],
    positions: Iterable[int],
    token_batch_size: int,
) -> list[dict[str, Any]]:
    if token_batch_size <= 0:
        raise ValueError("token_batch_size must be positive")
    tokens = [(int(sample_id), int(position)) for sample_id in fit_sample_ids for position in positions]
    batches: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(tokens), int(token_batch_size))):
        chunk = tokens[start : start + int(token_batch_size)]
        batches.append(
            {
                "batch_index": batch_index,
                "sample_ids": sorted({sample_id for sample_id, _ in chunk}),
                "token_positions": chunk,
                "token_count": len(chunk),
                "is_tail": len(chunk) < int(token_batch_size),
            }
        )
    return batches


def record_frozen_state_transition(
    model: torch.nn.Module,
    *,
    stage: str,
    operation: Callable[[], Any],
) -> dict[str, Any]:
    before = model_state_digest(model)
    operation()
    after = model_state_digest(model)
    return {
        "stage": str(stage),
        "before_digest": before,
        "after_digest": after,
        "frozen": before == after,
    }


def train_linear_probe_step(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    config: ProbeTrainingConfig,
    base_model: torch.nn.Module | None = None,
) -> dict[str, Any]:
    if config.optimizer != "AdamW":
        raise ValueError("Spec 02 probe optimizer must be AdamW")
    if hidden.ndim != 3:
        raise ValueError(f"hidden must have shape [batch, seq, hidden], got {tuple(hidden.shape)}")
    if labels.shape != hidden.shape[:2]:
        raise ValueError("labels must match hidden batch and sequence dimensions")

    base_state_before = model_state_digest(base_model) if base_model is not None else ""
    features = hidden.detach().reshape(-1, hidden.shape[-1]).float()
    flat_labels = labels.reshape(-1).long()
    probe = torch.nn.Linear(hidden.shape[-1], int(num_classes)).to(features.device)
    before = [param.detach().clone() for param in probe.parameters()]
    optimizer_kwargs: dict[str, Any] = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    if config.fused_adamw_on_cuda and features.device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(probe.parameters(), **optimizer_kwargs)
    loss = F.cross_entropy(probe(features), flat_labels)
    loss.backward()
    optimizer.step()
    delta_l1 = 0.0
    for before_param, after_param in zip(before, probe.parameters()):
        delta_l1 += float((after_param.detach() - before_param).abs().sum().item())
    base_state_after = model_state_digest(base_model) if base_model is not None else ""
    return {
        "optimizer": "AdamW",
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "loss": float(loss.detach().item()),
        "fused_adamw_runtime": bool("fused" in optimizer_kwargs),
        "base_state_before": base_state_before,
        "base_state_after": base_state_after,
        "base_state_unchanged": base_state_before == base_state_after,
        "probe_parameter_delta_l1": delta_l1,
    }


def build_streamed_sample_stats_row(
    *,
    target: ProbeTarget,
    sample_id: int,
    readout: str,
    ce_sum: float,
    num_positions: int,
    top1_correct: int,
    top5_correct: int,
    precision: str,
    input_sha256: str,
) -> dict[str, Any]:
    if num_positions <= 0:
        raise ValueError("num_positions must be positive")
    row = target.to_row()
    row.update(
        {
            "sample_id": int(sample_id),
            "readout": str(readout),
            "precision": str(precision),
            "input_sha256": str(input_sha256),
            "ce_sum": float(ce_sum),
            "num_positions": int(num_positions),
            "top1_correct": int(top1_correct),
            "top5_correct": int(top5_correct),
            "loss": float(ce_sum) / int(num_positions),
            "top1": int(top1_correct) / int(num_positions),
            "top5": int(top5_correct) / int(num_positions),
        }
    )
    return row


def frozen_lm_head_stats_row(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    target: ProbeTarget,
    sample_id: int,
    precision: str,
    input_sha256: str,
) -> dict[str, Any]:
    logits = logits_from_record_hidden(model, hidden)
    stats = next_token_sufficient_stats(
        logits,
        input_ids,
        metric_start=DEFAULT_METRIC_START,
        metric_end=DEFAULT_METRIC_END,
    )
    return build_stats_row(
        checkpoint=target.checkpoint,
        scale=target.scale,
        condition=target.condition,
        sample_id=sample_id,
        readout="frozen_lm_head",
        stats=stats,
        precision=precision,
        input_sha256=input_sha256,
        layer=target.layer,
        pass_index=target.pass_index,
        record_type=target.record_type,
    )


def build_probe_detail_row(*, target: ProbeTarget, readout: str, bootstrap: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": target.checkpoint,
        "scale": target.scale,
        "condition": target.condition,
        "layer": target.layer,
        "pass": target.pass_index,
        "record_type": target.record_type,
        "readout": str(readout),
        "heldout_loss": float(bootstrap["heldout_loss"]),
        "top1": float(bootstrap["top1"]),
        "top5": float(bootstrap["top5"]),
        "ci_low": float(bootstrap["ci_low"]),
        "ci_high": float(bootstrap["ci_high"]),
    }


def bootstrap_readout_ci(rows: Sequence[dict[str, Any]], *, n_bootstrap: int = 1000, seed: int = 20260722) -> dict[str, Any]:
    if not rows:
        raise ValueError("bootstrap requires at least one per-sample row")
    sample_ids = [int(row["sample_id"]) for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("bootstrap rows must be unique by sample_id")

    def aggregate(selected: Sequence[dict[str, Any]]) -> tuple[float, float, float]:
        ce = sum(float(row["ce_sum"]) for row in selected)
        positions = sum(int(row["num_positions"]) for row in selected)
        top1 = sum(int(row["top1_correct"]) for row in selected)
        top5 = sum(int(row["top5_correct"]) for row in selected)
        return ce / positions, top1 / positions, top5 / positions

    heldout_loss, top1, top5 = aggregate(rows)
    rng = random.Random(seed)
    losses = []
    rows_list = list(rows)
    for _ in range(int(n_bootstrap)):
        draw = [rows_list[rng.randrange(len(rows_list))] for _ in rows_list]
        losses.append(aggregate(draw)[0])
    ci_low, ci_high = np.quantile(np.asarray(losses, dtype=np.float64), [0.025, 0.975])
    return {
        "heldout_loss": float(heldout_loss),
        "top1": float(top1),
        "top5": float(top5),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_unit": "sample_id",
        "bootstrap_samples": int(n_bootstrap),
        "num_samples": len(rows),
        "sample_ids": sample_ids,
    }


def select_deterministic_audit_sample_ids(
    dataset: FixedProbeDataset,
    *,
    full_raw_count: int = 4,
    hidden_state_count: int = 64,
) -> dict[str, Any]:
    if full_raw_count < 0 or hidden_state_count < 0:
        raise ValueError("audit counts must be non-negative")
    if full_raw_count > hidden_state_count:
        raise ValueError("full_raw_count cannot exceed hidden_state_count")
    sample_ids = list(dataset.sample_ids)
    if hidden_state_count > len(sample_ids):
        raise ValueError("hidden_state_count exceeds fixed sample count")
    return {
        "selection_rule": "first_fixed_sample_ids",
        "full_raw_sample_ids": sample_ids[:full_raw_count],
        "hidden_state_sample_ids": sample_ids[:hidden_state_count],
        "full_raw_count": int(full_raw_count),
        "hidden_state_count": int(hidden_state_count),
    }


def build_spec02_run_manifest(
    *,
    dataset: FixedProbeDataset,
    checkpoint_inventory: Sequence[dict[str, Any]],
    output_root: Path,
    command: Sequence[str],
    wandb_plan: dict[str, Any],
    cap_bytes: int,
    audit_selection: dict[str, Any],
) -> dict[str, Any]:
    return {
        "spec": "02_probe_readout",
        "command": list(command),
        "output_root": str(output_root),
        "output_root_cap_bytes": int(cap_bytes),
        "no_full_raw_archive_dependency": True,
        "spec01": {
            "root": str(dataset.spec01_root),
            "seed": dataset.seed,
            "sample_count": len(dataset.sample_ids),
            "fit_sample_ids": list(dataset.fit_sample_ids),
            "heldout_sample_ids": list(dataset.heldout_sample_ids),
            "metric_positions": {"start": dataset.metric_start, "end": dataset.metric_end},
            "theory_eval_ids_sha256": dataset.theory_eval_ids_sha256,
            "probe_split_metadata_sha256": dataset.probe_split_metadata_sha256,
        },
        "checkpoint_count": len(checkpoint_inventory),
        "checkpoints": list(checkpoint_inventory),
        "git": {
            "commit": "NOT_RECORDED",
            "status_short": "NOT_RECORDED",
            "recording_rule": "runtime command must record exact git commit and status before real execution",
        },
        "wandb": dict(wandb_plan),
        "audit_selection": dict(audit_selection),
        "runtime_proof_placeholders": {
            "checkpoint_loaded": "INSUFFICIENT_INFORMATION",
            "probe_training": "INSUFFICIENT_INFORMATION",
            "frozen_head_eval": "INSUFFICIENT_INFORMATION",
            "runtime_acceleration": "INSUFFICIENT_INFORMATION",
            "tables": "INSUFFICIENT_INFORMATION",
            "figures": "INSUFFICIENT_INFORMATION",
            "wandb_runtime": "INSUFFICIENT_INFORMATION",
        },
        "runtime_status": "NOT_RUN",
    }


def build_spec02_run_checks(
    manifest: dict[str, Any],
    *,
    acceleration_matrix: Sequence[dict[str, Any]],
    runtime_executed: bool,
) -> dict[str, Any]:
    runtime_results = {str(row["row"]): str(row["result"]) for row in acceleration_matrix}
    runtime_placeholders = dict(
        manifest.get(
            "runtime_proof_placeholders",
            {
                "checkpoint_loaded": "INSUFFICIENT_INFORMATION",
                "probe_training": "INSUFFICIENT_INFORMATION",
                "frozen_head_eval": "INSUFFICIENT_INFORMATION",
                "runtime_acceleration": "INSUFFICIENT_INFORMATION",
            },
        )
    )
    runtime_placeholders_complete = bool(runtime_placeholders) and all(
        str(value) == "PASS" for value in runtime_placeholders.values()
    )
    runtime_rows_complete = bool(runtime_results) and all(result == "PASS" for result in runtime_results.values())
    runtime_rows_result = (
        "PASS"
        if runtime_executed and runtime_rows_complete
        else "INSUFFICIENT_INFORMATION"
    )
    runtime_proof_status = (
        "PASS"
        if runtime_executed and runtime_placeholders_complete and runtime_rows_complete
        else "INSUFFICIENT_INFORMATION"
    )
    return {
        "spec": "02_probe_readout",
        "runtime_executed": bool(runtime_executed),
        "command_present": bool(manifest.get("command")),
        "checkpoint_count": int(manifest.get("checkpoint_count", 0)),
        "output_root_cap_bytes": int(manifest.get("output_root_cap_bytes", 0)),
        "no_full_raw_archive_dependency": bool(manifest.get("no_full_raw_archive_dependency", False)),
        "wandb_mode": str(manifest.get("wandb", {}).get("mode", "")),
        "runtime_proof_status": runtime_proof_status,
        "runtime_proof_placeholders": runtime_placeholders,
        "runtime_rows_result": runtime_rows_result,
        "acceleration_rows": runtime_results,
    }


def build_figure_table_inputs(detail_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not detail_rows:
        raise ValueError("detail_rows must contain at least one row")
    missing = [column for column in PROBE_DETAIL_COLUMNS if any(column not in row for row in detail_rows)]
    if missing:
        raise ValueError(f"detail rows missing required columns: {sorted(set(missing))}")
    return {
        "detail_columns": list(PROBE_DETAIL_COLUMNS),
        "row_count": len(detail_rows),
        "record_types": sorted({str(row["record_type"]) for row in detail_rows}),
        "heatmap_index": ["checkpoint", "condition", "scale", "readout", "record_type", "layer", "pass"],
        "heatmap_values": ["heldout_loss", "top1", "top5"],
        "scale_curve_index": ["condition", "scale", "readout", "record_type"],
        "scale_curve_values": ["heldout_loss", "top1", "top5"],
        "runtime_artifact_status": "NOT_CREATED",
    }


def write_spec02_json_guarded(
    path: Path,
    payload: Any,
    *,
    output_root: Path,
    cap_bytes: int = OUTPUT_ROOT_CAP_BYTES,
) -> None:
    write_json_guarded(path, payload, output_root=output_root, cap_bytes=cap_bytes)


def build_wandb_mode_plan(*, formal: bool, wandb_enabled: bool) -> dict[str, Any]:
    if not wandb_enabled:
        return {
            "wandb_enabled": False,
            "mode": "not_initialized",
            "cloud_sync": False,
            "env": {},
        }
    if formal:
        return {
            "wandb_enabled": True,
            "mode": "online",
            "project": "physics_for_llm",
            "entity": "gyy0592-ucsc",
            "cloud_sync": True,
            "env": {"WANDB_PROJECT": "physics_for_llm"},
        }
    return {
        "wandb_enabled": True,
        "mode": "offline",
        "project": "physics_for_llm",
        "entity": "gyy0592-ucsc",
        "cloud_sync": False,
        "env": {"WANDB_MODE": "offline", "WANDB_PROJECT": "physics_for_llm"},
    }


def build_mixerloop_acceleration_matrix(*, device_type: str, wandb_enabled: bool) -> list[dict[str, str]]:
    rows = [
        ("M-009", "BF16/autocast", "probe_autocast", "record dtype trace when runtime is executed"),
        ("M-010", "sequence packing", "make_packed_probe_batch/pack_sequences", "record packing use or semantic non-applicability"),
        ("M-011", "segment-causal masks", "packed segment-causal mask tests", "record no-leakage mask proof if packing is used"),
        ("M-012", "fixed packed row counts", "packed_batch_rows config", "record fixed row counts for compile-stable packed batches"),
        ("M-013", "fastest attention/FlexAttention applicability", "attention_implementation=flex_attention", "record non-applicability or runtime attention path"),
        ("M-014", "torch.compile", "torch.compile", "record compile trace and no fallback"),
        ("M-015", "fused AdamW", "torch.optim.AdamW(fused=True)", "record optimizer construction and update equivalence"),
        ("M-016", "pin-memory CPU batches", "pin_probe_batch_memory", "record requested and actual pinned-memory status"),
        ("M-017", "non-blocking CPU-to-GPU transfer", "move_probe_batch_to_device(non_blocking=True)", "record copy event count, timing, and non-blocking flag"),
        ("M-018", "microbatch/gradient-accumulation accounting", "reference optimizer-step accounting", "record microbatch counts, loss weights, and optimizer steps"),
        ("M-019", "phase timing", "reference synchronized phase timers", "record synchronized phase timings with units and denominators"),
        ("M-020", "GPU utilization/memory trace", "reference GPU sampler", "record GPU utilization and memory trace for GPU runs"),
        ("M-021", "same-sample/same-weight equivalence", "run_vprobe_same_sample_weight_equivalence", "record same labels, loss, gradients, and one-step update tolerance"),
        ("M-022", "durable manifest/check artifacts", "config/logs/records/artifacts/run_manifest/run_checks", "record retained config, logs, records, manifest, and checks"),
        ("M-023", "W&B online-first/offline fallback", "ResilientWandbRun/init_wandb_run", "record no-W&B, non-formal offline, or formal online metadata"),
    ]
    device_note = f"planned device_type={device_type}; wandb_enabled={wandb_enabled}"
    return [
        {
            "row": row,
            "semantic_owner": owner,
            "reference_source_path": reference,
            "chosen_path": f"MixerLoop Spec 02 helper/CLI will {requirement}; {device_note}",
            "required_runtime_evidence": requirement,
            "observed_runtime_evidence": "placeholder: no Spec 02 runtime yet",
            "result": "INSUFFICIENT_INFORMATION",
            "proof_limit": "source/config/test schema only until a gated Spec 02 runtime records this row",
        }
        for row, owner, reference, requirement in rows
    ]


def _git_text(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"NOT_RECORDED: {exc}"


def _load_fixed_input_ids(dataset: FixedProbeDataset, *, sample_id: int, data_dir: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    record = dataset.records_by_sample_id[int(sample_id)]
    shard_path = Path(data_dir) / str(record["shard"])
    if not shard_path.is_file():
        raise FileNotFoundError(f"fixed sample shard not found: {shard_path}")
    tokens = np.memmap(shard_path, dtype=np.uint16, mode="r")
    start = int(record["start_token"])
    end = start + int(record["num_tokens"])
    input_tokens = np.asarray(tokens[start:end], dtype=np.uint16)
    actual_sha = sha256_bytes(input_tokens.tobytes())
    if actual_sha != str(record["input_sha256"]):
        raise ValueError(f"fixed sample sha mismatch for sample_id={sample_id}: {actual_sha} != {record['input_sha256']}")
    return torch.as_tensor(input_tokens.astype(np.int64), dtype=torch.long).unsqueeze(0), dict(record)


def _embedding_layer(backbone: torch.nn.Module) -> torch.nn.Module:
    if hasattr(backbone, "embeddings"):
        return backbone.embeddings
    if hasattr(backbone, "embed_tokens"):
        return backbone.embed_tokens
    raise AttributeError("loaded model backbone lacks embeddings/embed_tokens")


def _call_gdn_mixer(layer: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    norm = getattr(layer, "attn_norm", None) or getattr(layer, "norm", None)
    mixer = getattr(layer, "mixer", None) or getattr(layer, "attn", None)
    if norm is None or mixer is None:
        raise AttributeError("GDN layer lacks required norm/mixer owner for hA capture")
    try:
        output = mixer(norm(hidden), attention_mask=None, use_cache=False, output_attentions=False)
    except TypeError:
        output = mixer(norm(hidden), attention_mask=None)
    attn_out = output[0] if isinstance(output, (tuple, list)) else output
    return hidden + attn_out


def _capture_gdn_h_a_pass1(model: torch.nn.Module, input_ids: torch.Tensor, *, layer_index: int) -> torch.Tensor:
    backbone = model.model
    layers = list(backbone.layers)
    if not 0 <= int(layer_index) < len(layers):
        raise ValueError(f"layer_index {layer_index} outside loaded layer count {len(layers)}")
    hidden = _embedding_layer(backbone)(input_ids)
    for current_layer_index, layer in enumerate(layers[: int(layer_index) + 1]):
        h_a = _call_gdn_mixer(layer, hidden)
        if current_layer_index == int(layer_index):
            return h_a
        if hasattr(layer, "ffn_norm") and hasattr(layer, "ffn"):
            hidden = h_a + layer.ffn(layer.ffn_norm(h_a))
        elif hasattr(layer, "mlp_norm") and hasattr(layer, "mlp"):
            hidden = h_a + layer.mlp(layer.mlp_norm(h_a))
        else:
            raise AttributeError("GDN layer lacks FFN/MLP owner needed to advance to later layers")
    raise RuntimeError("unreachable GDN hA capture state")


def _capture_record_hidden(model: torch.nn.Module, input_ids: torch.Tensor, target: ProbeTarget) -> torch.Tensor:
    if target.condition == "gdn":
        if target.pass_index != 1 or target.record_type != "hA":
            raise ValueError("gdn/No Loop exposes only hA pass 1")
        return _capture_gdn_h_a_pass1(model, input_ids, layer_index=target.layer)
    backbone = model.model
    layers = list(backbone.layers)
    if not 0 <= target.layer < len(layers):
        raise ValueError(f"layer {target.layer} outside loaded layer count {len(layers)}")
    if target.condition == "mixerloop":
        if target.record_type != "hA":
            raise ValueError("Mixer-only Spec 02 runner exposes hA targets only")
        hidden = _embedding_layer(backbone)(input_ids)
        for layer_index, layer in enumerate(layers):
            h = hidden
            for loop_index in range(int(backbone.loop_count)):
                h_input = h
                attn_out, _, _ = layer.mixer(layer.attn_norm(h))
                h = h + attn_out
                if getattr(backbone, "residual_weight", None) is not None:
                    h = h + backbone.residual_weight[loop_index].view(1, 1, -1) * h_input
                if layer_index == target.layer and loop_index + 1 == target.pass_index:
                    return h
            hidden = h + layer.ffn(layer.ffn_norm(h))
        raise RuntimeError("unreachable mixerloop capture state")
    if target.condition == "fullloop":
        hidden = _embedding_layer(backbone)(input_ids)
        for loop_index in range(int(backbone.loop_count)):
            h_input = hidden
            for layer_index, layer in enumerate(layers):
                attn_out, _, _ = layer.mixer(layer.attn_norm(hidden))
                h_a = hidden + attn_out
                h_f = h_a + layer.ffn(layer.ffn_norm(h_a))
                if layer_index == target.layer and loop_index + 1 == target.pass_index:
                    if target.record_type == "hA":
                        return h_a
                    if target.record_type == "hF":
                        return h_f
                    raise ValueError(f"Unknown record_type {target.record_type!r}")
                hidden = h_f
            if getattr(backbone, "residual_weight", None) is not None:
                hidden = hidden + backbone.residual_weight[loop_index] * h_input
        raise RuntimeError("unreachable fullloop capture state")
    raise ValueError(f"Unsupported condition {target.condition!r}")


def _train_and_eval_probe_step(
    train_hidden: torch.Tensor,
    train_input_ids: torch.Tensor,
    heldout_hidden: torch.Tensor,
    heldout_input_ids: torch.Tensor,
    *,
    metric_start: int,
    metric_end: int,
    vocab_size: int,
    device: torch.device,
) -> dict[str, Any]:
    positions = slice(int(metric_start), int(metric_end) + 1)
    train_features = train_hidden[:, positions, :].reshape(-1, train_hidden.shape[-1]).detach().float()
    train_labels = train_input_ids[:, int(metric_start) + 1 : int(metric_end) + 2].reshape(-1).long()
    heldout_features = heldout_hidden[:, positions, :].reshape(-1, heldout_hidden.shape[-1]).detach().float()
    heldout_labels = heldout_input_ids[:, int(metric_start) + 1 : int(metric_end) + 2].reshape(-1).long()
    probe = torch.nn.Linear(train_hidden.shape[-1], int(vocab_size)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    start = time.perf_counter()
    train_logits = probe(train_features)
    train_loss = torch.nn.functional.cross_entropy(train_logits, train_labels)
    train_loss.backward()
    optimizer.step()
    probe.eval()
    with torch.no_grad():
        heldout_logits = probe(heldout_features)
        heldout_loss = torch.nn.functional.cross_entropy(heldout_logits, heldout_labels)
        topk = torch.topk(heldout_logits, k=min(5, heldout_logits.shape[-1]), dim=-1).indices
        top1 = int((topk[:, 0] == heldout_labels).sum().item())
        top5 = int((topk == heldout_labels[:, None]).any(dim=-1).sum().item())
    elapsed = time.perf_counter() - start
    return {
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "train_token_count": int(train_labels.numel()),
        "train_loss": float(train_loss.detach().cpu().item()),
        "heldout_token_count": int(heldout_labels.numel()),
        "heldout_loss": float(heldout_loss.detach().cpu().item()),
        "heldout_top1": top1,
        "heldout_top5": top5,
        "elapsed_seconds": float(elapsed),
    }


def _write_guarded_with_record(path: Path, payload: Any, *, output_root: Path, cap_bytes: int) -> dict[str, Any]:
    before = root_size_bytes(output_root)
    write_json_guarded(path, payload, output_root=output_root, cap_bytes=cap_bytes)
    after = root_size_bytes(output_root)
    return {
        "path": str(path),
        "bytes_before": int(before),
        "bytes_after": int(after),
        "cap_bytes": int(cap_bytes),
    }


def _metric_features_and_labels(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    metric_start: int,
    metric_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = slice(int(metric_start), int(metric_end) + 1)
    features = hidden[:, positions, :].reshape(-1, hidden.shape[-1]).detach().cpu().float()
    labels = input_ids[:, int(metric_start) + 1 : int(metric_end) + 2].reshape(-1).detach().cpu().long()
    return features, labels


def _probe_sample_stats(
    target: ProbeTarget,
    *,
    sample_id: int,
    logits: torch.Tensor,
    labels: torch.Tensor,
    precision: str,
    input_sha256: str,
) -> dict[str, Any]:
    logits = logits.detach()
    labels = labels.detach().to(logits.device).long()
    ce_sum = torch.nn.functional.cross_entropy(logits.float(), labels, reduction="sum")
    topk = torch.topk(logits, k=min(5, logits.shape[-1]), dim=-1).indices
    return build_streamed_sample_stats_row(
        target=target,
        sample_id=sample_id,
        readout="linear_probe",
        ce_sum=float(ce_sum.cpu().item()),
        num_positions=int(labels.numel()),
        top1_correct=int((topk[:, 0] == labels).sum().cpu().item()),
        top5_correct=int((topk == labels[:, None]).any(dim=-1).sum().cpu().item()),
        precision=precision,
        input_sha256=input_sha256,
    )


def _capture_gdn_target_samples(
    model: torch.nn.Module,
    dataset: FixedProbeDataset,
    sample_ids: Sequence[int],
    *,
    data_dir: Path,
    target: ProbeTarget,
    device: torch.device,
) -> dict[str, Any]:
    feature_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        input_ids_cpu, fixed_record = _load_fixed_input_ids(dataset, sample_id=int(sample_id), data_dir=data_dir)
        input_ids = input_ids_cpu.to(device)
        with torch.no_grad():
            hidden = _capture_record_hidden(model, input_ids, target)
            logits = model.lm_head(model.model.norm(hidden))
            stats = next_token_sufficient_stats(
                logits,
                input_ids,
                metric_start=dataset.metric_start,
                metric_end=dataset.metric_end,
            )
        rows.append(
            build_streamed_sample_stats_row(
                target=target,
                sample_id=int(sample_id),
                readout="frozen_lm_head",
                ce_sum=float(stats["ce_sum"]),
                num_positions=int(stats["num_positions"]),
                top1_correct=int(stats["top1_correct"]),
                top5_correct=int(stats["top5_correct"]),
                precision="float32",
                input_sha256=str(fixed_record["input_sha256"]),
            )
        )
        features, labels = _metric_features_and_labels(
            hidden,
            input_ids,
            metric_start=dataset.metric_start,
            metric_end=dataset.metric_end,
        )
        feature_chunks.append(features)
        label_chunks.append(labels)
        records.append(fixed_record)
    return {
        "features": torch.cat(feature_chunks, dim=0),
        "labels": torch.cat(label_chunks, dim=0),
        "rows": rows,
        "records": records,
    }


def _evaluate_probe_rows(
    probe: torch.nn.Module,
    target: ProbeTarget,
    captured: dict[str, Any],
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    position_count = int(captured["rows"][0]["num_positions"])
    for index, frozen_row in enumerate(captured["rows"]):
        start = index * position_count
        end = start + position_count
        features = captured["features"][start:end].to(device)
        labels = captured["labels"][start:end].to(device)
        with torch.no_grad():
            logits = probe(features)
        rows.append(
            _probe_sample_stats(
                target,
                sample_id=int(frozen_row["sample_id"]),
                logits=logits,
                labels=labels,
                precision="float32",
                input_sha256=str(frozen_row["input_sha256"]),
            )
        )
    return rows


def _sample_gpu_utilization_memory() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _move_probe_batch_to_device(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    pin_memory: bool,
    non_blocking: bool,
    transfer_evidence: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    if pin_memory and device.type == "cuda":
        features = features.pin_memory()
        labels = labels.pin_memory()
    transfer_evidence["cpu_to_gpu_pin_memory_requested"] = bool(pin_memory and device.type == "cuda")
    transfer_evidence["cpu_to_gpu_non_blocking_requested"] = bool(non_blocking and device.type == "cuda")
    transfer_evidence["cpu_to_gpu_pinned_memory"] = bool(
        getattr(features, "is_pinned", lambda: False)() and getattr(labels, "is_pinned", lambda: False)()
    )
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        moved_features = features.to(device, non_blocking=non_blocking)
        moved_labels = labels.to(device, non_blocking=non_blocking)
        end_event.record()
        torch.cuda.synchronize(device)
        transfer_evidence["cpu_to_gpu_copy_seconds"] += float(start_event.elapsed_time(end_event) / 1000.0)
        transfer_evidence["cpu_to_gpu_copy_event_count"] += 1
        transfer_evidence["copied_shape"] = list(features.shape)
        transfer_evidence["labels_shape"] = list(labels.shape)
        return moved_features, moved_labels
    return features.to(device), labels.to(device)


def _train_probe_full_split(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    heldout_features: torch.Tensor,
    heldout_labels: torch.Tensor,
    *,
    config: ProbeTrainingConfig,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(20260722)
    probe = torch.nn.Linear(train_features.shape[-1], int(vocab_size)).to(device)
    optimizer_kwargs: dict[str, Any] = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    if config.fused_adamw_on_cuda and device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(probe.parameters(), **optimizer_kwargs)
    compile_runtime = False
    compiled_probe: torch.nn.Module = probe
    if config.torch_compile_on_cuda and device.type == "cuda":
        compiled_probe = torch.compile(probe, mode="reduce-overhead")
        compile_runtime = True
    batch_size = int(config.token_batch_size)
    history = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    transfer_evidence: dict[str, Any] = {
        "cpu_to_gpu_copy_event_count": 0,
        "cpu_to_gpu_copy_seconds": 0.0,
        "cpu_to_gpu_pin_memory_requested": False,
        "cpu_to_gpu_non_blocking_requested": False,
        "cpu_to_gpu_pinned_memory": False,
        "copied_shape": [],
        "labels_shape": [],
    }
    gpu_samples: list[dict[str, Any]] = []
    peak_memory_before = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        peak_memory_before = float(torch.cuda.max_memory_allocated(device) / 1_000_000)
        if config.record_gpu_trace:
            gpu_samples.append(_sample_gpu_utilization_memory())
    start_time = time.perf_counter()
    for epoch in range(int(config.epochs)):
        probe.train()
        ce_sum = 0.0
        token_count = 0
        for start in range(0, train_labels.numel(), batch_size):
            end = min(start + batch_size, train_labels.numel())
            features, labels = _move_probe_batch_to_device(
                train_features[start:end],
                train_labels[start:end],
                device=device,
                pin_memory=config.pin_memory_cpu_batches,
                non_blocking=config.non_blocking_cpu_to_gpu_transfer,
                transfer_evidence=transfer_evidence,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = compiled_probe(features)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            ce_sum += float(loss.detach().cpu().item()) * int(labels.numel())
            token_count += int(labels.numel())
        if device.type == "cuda" and config.record_gpu_trace:
            gpu_samples.append(_sample_gpu_utilization_memory())
        probe.eval()
        heldout_ce = 0.0
        heldout_count = 0
        with torch.no_grad():
            for start in range(0, heldout_labels.numel(), batch_size):
                end = min(start + batch_size, heldout_labels.numel())
                features, labels = _move_probe_batch_to_device(
                    heldout_features[start:end],
                    heldout_labels[start:end],
                    device=device,
                    pin_memory=config.pin_memory_cpu_batches,
                    non_blocking=config.non_blocking_cpu_to_gpu_transfer,
                    transfer_evidence=transfer_evidence,
                )
                logits = compiled_probe(features)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                heldout_ce += float(loss.detach().cpu().item()) * int(labels.numel())
                heldout_count += int(labels.numel())
        train_loss = ce_sum / token_count
        heldout_loss = heldout_ce / heldout_count
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "heldout_loss": heldout_loss})
        if heldout_loss < best_loss:
            best_loss = heldout_loss
            best_state = copy.deepcopy(probe.state_dict())
    peak_memory_after = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_after = float(torch.cuda.max_memory_allocated(device) / 1_000_000)
        if config.record_gpu_trace:
            gpu_samples.append(_sample_gpu_utilization_memory())
    if best_state is not None:
        probe.load_state_dict(best_state)
    acceleration = {
        "torch_compile": {
            "result": "PASS" if compile_runtime else "NOT_APPLICABLE",
            "formal_run_runtime": bool(compile_runtime),
            "mode": "reduce-overhead" if compile_runtime else "",
            "no_eager_fallback": bool(compile_runtime),
        },
        "pin_memory_cpu_batches": {
            "result": "PASS" if transfer_evidence["cpu_to_gpu_pinned_memory"] else "INSUFFICIENT_INFORMATION",
            "formal_run_runtime": device.type == "cuda",
            "cpu_to_gpu_pinned_memory": bool(transfer_evidence["cpu_to_gpu_pinned_memory"]),
            "cpu_to_gpu_pin_memory_requested": bool(transfer_evidence["cpu_to_gpu_pin_memory_requested"]),
        },
        "non_blocking_cpu_to_gpu_transfer": {
            "result": "PASS" if transfer_evidence["cpu_to_gpu_copy_event_count"] > 0 and transfer_evidence["cpu_to_gpu_non_blocking_requested"] else "INSUFFICIENT_INFORMATION",
            "formal_run_runtime": device.type == "cuda",
            "cpu_to_gpu_copy_non_blocking": bool(transfer_evidence["cpu_to_gpu_non_blocking_requested"]),
            "cpu_to_gpu_copy_event_count": int(transfer_evidence["cpu_to_gpu_copy_event_count"]),
            "cpu_to_gpu_copy_seconds": float(transfer_evidence["cpu_to_gpu_copy_seconds"]),
            "copied_shape": transfer_evidence["copied_shape"],
            "labels_shape": transfer_evidence["labels_shape"],
        },
        "gpu_utilization_memory_trace": {
            "result": "PASS" if device.type == "cuda" and gpu_samples else "INSUFFICIENT_INFORMATION",
            "formal_run_runtime": device.type == "cuda",
            "gpu_utilization_samples": gpu_samples,
            "peak_gpu_memory_allocated_mb": peak_memory_after,
            "peak_gpu_memory_before_mb": peak_memory_before,
        },
    }
    return probe, {
        "config": {
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "token_batch_size": config.token_batch_size,
            "epochs": config.epochs,
            "selection_rule": config.selection_rule,
            "fused_adamw_on_cuda": config.fused_adamw_on_cuda,
            "fused_adamw_runtime": bool("fused" in optimizer_kwargs),
            "torch_compile_on_cuda": config.torch_compile_on_cuda,
            "pin_memory_cpu_batches": config.pin_memory_cpu_batches,
            "non_blocking_cpu_to_gpu_transfer": config.non_blocking_cpu_to_gpu_transfer,
            "record_gpu_trace": config.record_gpu_trace,
        },
        "history": history,
        "best_heldout_loss": best_loss,
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "acceleration": acceleration,
    }


def run_gdn15m_layer0_target(args: argparse.Namespace) -> dict[str, Any]:
    import fla.models  # noqa: F401, registers gated_deltanet with Transformers
    import custom_models  # noqa: F401, registers local loop model types
    from transformers import AutoModelForCausalLM

    start_time = time.perf_counter()
    output_root = Path(args.output_root)
    cap_bytes = int(args.cap_bytes)
    if root_size_bytes(output_root) > cap_bytes:
        raise OutputBudgetError(f"output root already exceeds cap: {root_size_bytes(output_root)} > {cap_bytes}")
    dataset = load_fixed_probe_dataset(Path(args.spec01_root))
    fit_ids = dataset.fit_sample_ids[: int(args.max_fit_samples)] if args.max_fit_samples else dataset.fit_sample_ids
    heldout_ids = (
        dataset.heldout_sample_ids[: int(args.max_heldout_samples)]
        if args.max_heldout_samples
        else dataset.heldout_sample_ids
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Spec 02 target run requires CUDA; CPU-only substitution is not allowed")
    os.environ["WANDB_MODE"] = "offline"
    checkpoint_path = Path(args.checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.float32,
    ).to(device)
    model.eval()
    checkpoint_name = str(args.checkpoint_name) if args.checkpoint_name else checkpoint_path.name
    scale = str(args.scale) if args.scale else checkpoint_name.removeprefix("gdn-")
    target = ProbeTarget(checkpoint_name, scale, str(args.condition), int(args.layer), int(args.pass_index), str(args.record_type))
    before_digest = model_state_digest(model)
    capture_start = time.perf_counter()
    fit = _capture_gdn_target_samples(
        model,
        dataset,
        fit_ids,
        data_dir=Path(args.data_dir),
        target=target,
        device=device,
    )
    heldout = _capture_gdn_target_samples(
        model,
        dataset,
        heldout_ids,
        data_dir=Path(args.data_dir),
        target=target,
        device=device,
    )
    capture_elapsed = time.perf_counter() - capture_start
    probe, training = _train_probe_full_split(
        fit["features"],
        fit["labels"],
        heldout["features"],
        heldout["labels"],
        config=ProbeTrainingConfig(),
        vocab_size=int(model.config.vocab_size),
        device=device,
    )
    probe_rows = _evaluate_probe_rows(probe, target, heldout, device=device)
    frozen_bootstrap = bootstrap_readout_ci(heldout["rows"], n_bootstrap=1000, seed=20260722)
    probe_bootstrap = bootstrap_readout_ci(probe_rows, n_bootstrap=1000, seed=20260722)
    detail_rows = [
        build_probe_detail_row(target=target, readout="frozen_lm_head", bootstrap=frozen_bootstrap),
        build_probe_detail_row(target=target, readout="linear_probe", bootstrap=probe_bootstrap),
    ]
    figure_inputs = build_figure_table_inputs(detail_rows)
    after_digest = model_state_digest(model)
    manifest = {
        "spec": "02_probe_readout",
        "run": "gdn15m_layer0_hA_pass1_fixed_split",
        "command": list(sys.argv),
        "working_directory": str(Path.cwd()),
        "checkpoint_path": str(checkpoint_path),
        "target": target.to_row(),
        "fit_sample_count": len(fit_ids),
        "heldout_sample_count": len(heldout_ids),
        "spec01": {
            "root": str(dataset.spec01_root),
            "theory_eval_ids_sha256": dataset.theory_eval_ids_sha256,
            "probe_split_metadata_sha256": dataset.probe_split_metadata_sha256,
            "metric_positions": {"start": dataset.metric_start, "end": dataset.metric_end},
        },
        "device": str(device),
        "precision": "float32",
        "wandb": {"enabled": False, "mode": "offline", "run_dir": ""},
        "frozen_state": {"before_digest": before_digest, "after_digest": after_digest, "frozen": before_digest == after_digest},
        "capture_elapsed_seconds": float(capture_elapsed),
        "training": training,
        "frozen_head_detail": detail_rows[0],
        "probe_detail": detail_rows[1],
        "figure_table_inputs": figure_inputs,
        "output_root_bytes_before_run": root_size_bytes(output_root),
    }
    writes = []
    writes.append(_write_guarded_with_record(output_root / "run_manifest.json", manifest, output_root=output_root, cap_bytes=cap_bytes))
    writes.append(_write_guarded_with_record(output_root / "tables" / "frozen_head_heldout_rows.json", heldout["rows"], output_root=output_root, cap_bytes=cap_bytes))
    writes.append(_write_guarded_with_record(output_root / "tables" / "probe_heldout_rows.json", probe_rows, output_root=output_root, cap_bytes=cap_bytes))
    writes.append(_write_guarded_with_record(output_root / "tables" / "detail_rows.json", detail_rows, output_root=output_root, cap_bytes=cap_bytes))
    writes.append(_write_guarded_with_record(output_root / "tables" / "training_history.json", training, output_root=output_root, cap_bytes=cap_bytes))
    checks = {
        "exit_result": "PASS",
        "checkpoint_loaded": True,
        "frozen_base_unchanged": before_digest == after_digest,
        "fit_sample_count": len(fit_ids),
        "heldout_sample_count": len(heldout_ids),
        "probe_best_heldout_loss": training["best_heldout_loss"],
        "probe_detail": detail_rows[1],
        "frozen_head_detail": detail_rows[0],
        "write_records": writes,
    }
    writes.append(_write_guarded_with_record(output_root / "run_checks.json", checks, output_root=output_root, cap_bytes=cap_bytes))
    manifest["output_root_bytes_after_run"] = root_size_bytes(output_root)
    manifest["write_records"] = writes
    _write_guarded_with_record(output_root / "run_manifest.json", manifest, output_root=output_root, cap_bytes=cap_bytes)
    result = {
        "exit_result": "PASS",
        "output_root": str(output_root),
        "fit_sample_count": len(fit_ids),
        "heldout_sample_count": len(heldout_ids),
        "frozen_head_heldout_loss": detail_rows[0]["heldout_loss"],
        "probe_heldout_loss": detail_rows[1]["heldout_loss"],
        "elapsed_seconds": float(time.perf_counter() - start_time),
    }
    print(json.dumps(result, sort_keys=True))
    return manifest


def run_immediate_real_smoke(args: argparse.Namespace) -> dict[str, Any]:
    import fla.models  # noqa: F401, registers gated_deltanet with Transformers
    import custom_models  # noqa: F401, registers local loop model types
    from transformers import AutoModelForCausalLM

    start_time = time.perf_counter()
    output_root = Path(args.output_root)
    cap_bytes = int(args.cap_bytes)
    pre_root_bytes = root_size_bytes(output_root)
    if pre_root_bytes > cap_bytes:
        raise OutputBudgetError(f"output root already exceeds cap: {pre_root_bytes} > {cap_bytes}")

    dataset = load_fixed_probe_dataset(Path(args.spec01_root))
    sample_id = int(args.sample_id) if args.sample_id is not None else int(dataset.fit_sample_ids[0])
    input_ids_cpu, fixed_record = _load_fixed_input_ids(dataset, sample_id=sample_id, data_dir=Path(args.data_dir))
    heldout_sample_id = (
        int(args.heldout_sample_id) if args.heldout_sample_id is not None else int(dataset.heldout_sample_ids[0])
    )
    heldout_input_ids_cpu, heldout_fixed_record = _load_fixed_input_ids(
        dataset,
        sample_id=heldout_sample_id,
        data_dir=Path(args.data_dir),
    )
    checkpoint_path = Path(args.checkpoint_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if device.type != "cuda":
        raise RuntimeError("Spec 02 immediate smoke requires a real CUDA path; CPU-only substitution is not allowed")

    os.environ["WANDB_MODE"] = "offline"
    wandb_info: dict[str, Any] = {"enabled": False, "mode": "offline", "run_dir": ""}
    wandb_run = None
    if not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project="physics_for_llm",
            entity="gyy0592-ucsc",
            mode="offline",
            dir=str(output_root),
            config={"spec": "02_probe_readout", "checkpoint": str(checkpoint_path), "sample_id": sample_id},
        )
        wandb_info = {
            "enabled": True,
            "mode": "offline",
            "run_id": str(getattr(wandb_run, "id", "")),
            "run_dir": str(getattr(wandb_run, "dir", "")),
            "sync_command": f"wandb sync {Path(getattr(wandb_run, 'dir', '')).parent}",
        }

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.float32,
    ).to(device)
    model.eval()
    input_ids = input_ids_cpu.to(device)
    heldout_input_ids = heldout_input_ids_cpu.to(device)
    before_digest = model_state_digest(model)
    with torch.no_grad():
        hidden = _capture_gdn_h_a_pass1(model, input_ids, layer_index=int(args.layer))
        heldout_hidden = _capture_gdn_h_a_pass1(model, heldout_input_ids, layer_index=int(args.layer))
        logits = model.lm_head(model.model.norm(hidden))
        frozen_stats = next_token_sufficient_stats(
            logits,
            input_ids,
            metric_start=dataset.metric_start,
            metric_end=dataset.metric_end,
        )
        heldout_logits = model.lm_head(model.model.norm(heldout_hidden))
        heldout_frozen_stats = next_token_sufficient_stats(
            heldout_logits,
            heldout_input_ids,
            metric_start=dataset.metric_start,
            metric_end=dataset.metric_end,
        )
    probe_stats = _train_and_eval_probe_step(
        hidden,
        input_ids,
        heldout_hidden,
        heldout_input_ids,
        metric_start=dataset.metric_start,
        metric_end=dataset.metric_end,
        vocab_size=int(model.config.vocab_size),
        device=device,
    )
    after_digest = model_state_digest(model)
    elapsed = time.perf_counter() - start_time

    target = ProbeTarget("gdn-15m", "15m", "gdn", int(args.layer), 1, "hA")
    sample_row = build_streamed_sample_stats_row(
        target=target,
        sample_id=sample_id,
        readout="frozen_lm_head",
        ce_sum=float(frozen_stats["ce_sum"]),
        num_positions=int(frozen_stats["num_positions"]),
        top1_correct=int(frozen_stats["top1_correct"]),
        top5_correct=int(frozen_stats["top5_correct"]),
        precision="float32",
        input_sha256=str(fixed_record["input_sha256"]),
    )
    manifest = {
        "spec": "02_probe_readout",
        "smoke": "immediate",
        "command": list(sys.argv),
        "working_directory": str(Path.cwd()),
        "checkpoint_path": str(checkpoint_path),
        "target": target.to_row(),
        "sample_id": sample_id,
        "heldout_sample_id": heldout_sample_id,
        "fixed_record": fixed_record,
        "heldout_fixed_record": heldout_fixed_record,
        "spec01": {
            "root": str(dataset.spec01_root),
            "theory_eval_ids_sha256": dataset.theory_eval_ids_sha256,
            "probe_split_metadata_sha256": dataset.probe_split_metadata_sha256,
            "metric_positions": {"start": dataset.metric_start, "end": dataset.metric_end},
        },
        "device": str(device),
        "precision": "float32",
        "frozen_state": {
            "before_digest": before_digest,
            "after_digest": after_digest,
            "frozen": before_digest == after_digest,
        },
        "frozen_head": sample_row,
        "heldout_frozen_head": build_streamed_sample_stats_row(
            target=target,
            sample_id=heldout_sample_id,
            readout="frozen_lm_head",
            ce_sum=float(heldout_frozen_stats["ce_sum"]),
            num_positions=int(heldout_frozen_stats["num_positions"]),
            top1_correct=int(heldout_frozen_stats["top1_correct"]),
            top5_correct=int(heldout_frozen_stats["top5_correct"]),
            precision="float32",
            input_sha256=str(heldout_fixed_record["input_sha256"]),
        ),
        "probe_step": probe_stats,
        "wandb": wandb_info,
        "elapsed_seconds": float(elapsed),
        "tokens_per_second": float(
            (probe_stats["train_token_count"] + probe_stats["heldout_token_count"])
            / max(probe_stats["elapsed_seconds"], 1e-12)
        ),
        "output_root_bytes_before_run": int(pre_root_bytes),
    }
    if wandb_run is not None:
        wandb_run.log({"smoke/frozen_head_loss": sample_row["loss"], "smoke/probe_loss": probe_stats["loss"]})
        wandb_run.finish(exit_code=0)

    write_records = []
    write_records.append(_write_guarded_with_record(output_root / "run_manifest.json", manifest, output_root=output_root, cap_bytes=cap_bytes))
    write_records.append(_write_guarded_with_record(output_root / "records" / "frozen_head_sample.json", sample_row, output_root=output_root, cap_bytes=cap_bytes))
    checks = {
        "exit_result": "PASS",
        "checkpoint_loaded": True,
        "frozen_base_unchanged": before_digest == after_digest,
        "frozen_head_num_positions": int(frozen_stats["num_positions"]),
        "heldout_frozen_head_num_positions": int(heldout_frozen_stats["num_positions"]),
        "probe_train_token_count": int(probe_stats["train_token_count"]),
        "probe_heldout_token_count": int(probe_stats["heldout_token_count"]),
        "probe_heldout_loss": float(probe_stats["heldout_loss"]),
        "wandb_mode": wandb_info["mode"],
        "write_records": write_records,
    }
    write_records.append(_write_guarded_with_record(output_root / "run_checks.json", checks, output_root=output_root, cap_bytes=cap_bytes))
    manifest["output_root_bytes_after_run"] = root_size_bytes(output_root)
    manifest["write_records"] = write_records
    _write_guarded_with_record(output_root / "run_manifest.json", manifest, output_root=output_root, cap_bytes=cap_bytes)
    print(json.dumps({"exit_result": "PASS", "output_root": str(output_root), "elapsed_seconds": elapsed}, sort_keys=True))
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke-immediate", help="run the immediate real Spec 02 gdn-15m smoke")
    smoke.add_argument("--checkpoint-path", type=Path, default=Path("outputs/gdn-15m"))
    smoke.add_argument("--spec01-root", type=Path, default=Path("outputs/theory_eval_local_20260722/current_cli_1024"))
    smoke.add_argument("--data-dir", type=Path, default=Path("data/climbmix-10b"))
    smoke.add_argument("--output-root", type=Path, default=Path("outputs/theory_eval_local_20260722/spec02_probe_readout/smoke_immediate"))
    smoke.add_argument("--cap-bytes", type=int, default=99_000_000_000)
    smoke.add_argument("--sample-id", type=int, default=None)
    smoke.add_argument("--heldout-sample-id", type=int, default=None)
    smoke.add_argument("--layer", type=int, default=0)
    smoke.add_argument("--device", default="cuda")
    smoke.add_argument("--no-wandb", action="store_true")
    target = subparsers.add_parser("run-gdn15m-layer0", help="run Spec 02 gdn-15m layer 0 hA pass 1 fixed-split target")
    target.add_argument("--checkpoint-path", type=Path, default=Path("outputs/gdn-15m"))
    target.add_argument("--checkpoint-name", default=None)
    target.add_argument("--scale", default=None)
    target.add_argument("--spec01-root", type=Path, default=Path("outputs/theory_eval_local_20260722/current_cli_1024"))
    target.add_argument("--data-dir", type=Path, default=Path("data/climbmix-10b"))
    target.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/theory_eval_local_20260722/spec02_probe_readout/gdn15m_layer0_hA_pass1"),
    )
    target.add_argument("--cap-bytes", type=int, default=99_000_000_000)
    target.add_argument("--layer", type=int, default=0)
    target.add_argument("--pass-index", type=int, default=1)
    target.add_argument("--record-type", default="hA")
    target.add_argument("--condition", default="gdn")
    target.add_argument("--device", default="cuda")
    target.add_argument("--max-fit-samples", type=int, default=None)
    target.add_argument("--max-heldout-samples", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "smoke-immediate":
        run_immediate_real_smoke(args)
    elif args.command == "run-gdn15m-layer0":
        run_gdn15m_layer0_target(args)


if __name__ == "__main__":
    main()
