#!/usr/bin/env python3
"""Spec 04 single-point skip ablation and streamed statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM

import custom_models  # noqa: F401
from eval.theory_capture import OUTPUT_ROOT_CAP_BYTES, model_state_digest, next_token_sufficient_stats, root_size_bytes, write_json_guarded
from eval.theory_probe import _load_fixed_input_ids, load_fixed_probe_dataset, parse_released_checkpoint_inventory

DETAIL_COLUMNS = [
    "checkpoint",
    "scale",
    "condition",
    "layer",
    "pass",
    "normal_loss",
    "skip_loss",
    "delta_loss",
    "mITR",
    "core_score_if_run",
]
FORMAL_CHECKPOINTS = tuple(
    f"{condition}-{scale}"
    for scale in ("15m", "42m", "110m")
    for condition in ("mixerloop", "fullloop")
)
SAMPLE_COLUMNS = [
    "checkpoint",
    "scale",
    "condition",
    "layer",
    "pass",
    "sample_id",
    "input_sha256",
    "precision",
    "normal_ce_sum",
    "skip_ce_sum",
    "num_positions",
]


def validate_formal_scope(
    *,
    spec01_root: Path,
    outputs_root: Path,
    output_root: Path,
    mitr_paths: Sequence[Path],
    cap_bytes: int = OUTPUT_ROOT_CAP_BYTES,
) -> dict[str, Any]:
    """Validate the six-checkpoint Spec04 contract without loading a model."""
    if output_root.exists():
        raise FileExistsError(f"formal Spec04 output root must be absent: {output_root}")
    wandb_root = output_root.parent / f".{output_root.name}.wandb"
    if wandb_root.exists():
        raise FileExistsError(f"formal Spec04 W&B sibling root must be absent: {wandb_root}")
    dataset = load_fixed_probe_dataset(Path(spec01_root))
    if len(dataset.sample_ids) != 1024:
        raise ValueError(f"formal Spec04 requires 1024 fixed samples, found {len(dataset.sample_ids)}")
    inventory = parse_released_checkpoint_inventory(Path(outputs_root))
    selected = [row for row in inventory if str(row["checkpoint"]) in FORMAL_CHECKPOINTS]
    found = {str(row["checkpoint"]) for row in selected}
    expected = set(FORMAL_CHECKPOINTS)
    if found != expected:
        raise ValueError(f"formal Spec04 checkpoint scope mismatch: missing={sorted(expected - found)} extra={sorted(found - expected)}")
    target_count = sum(int(row["num_layers"]) * int(row["loop_count"]) for row in selected)
    if target_count != 208:
        raise ValueError(f"formal Spec04 requires 208 skip positions, found {target_count}")

    required_mitr_keys = {
        (str(row["condition"]), str(row["scale"]), layer, pass_index)
        for row in selected
        for layer in range(int(row["num_layers"]))
        for pass_index in range(2, int(row["loop_count"]) + 1)
    }
    observed_mitr_keys: set[tuple[str, str, int, int]] = set()
    source_groups: dict[str, set[tuple[str, str]]] = {}
    source_keys: dict[tuple[str, str, int, int], str] = {}
    for path in mitr_paths:
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(f"missing frozen mITR source: {source_path}")
        groups: set[tuple[str, str]] = set()
        with source_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (str(row["condition"]), str(row["scale"]), int(row["layer"]), int(row["pass"]))
                if key in required_mitr_keys:
                    if not _finite(row["mITR"]):
                        raise ValueError(f"non-finite frozen mITR value for {key}")
                    groups.add((key[0], key[1]))
                    previous_source = source_keys.get(key)
                    if previous_source is not None:
                        raise ValueError(f"frozen mITR key appears in multiple/duplicate sources: {key}: {previous_source}, {source_path}")
                    source_keys[key] = str(source_path)
                    observed_mitr_keys.add(key)
        source_groups[str(source_path)] = groups
    expected_groups = {(str(row["condition"]), str(row["scale"])) for row in selected}
    expected_source_group_sets = {
        frozenset(expected_groups - {("fullloop", "110m")}),
        frozenset({("fullloop", "110m")}),
    }
    if {frozenset(groups) for groups in source_groups.values()} != expected_source_group_sets:
        raise ValueError(f"frozen mITR source separation mismatch: {source_groups}")
    missing_mitr = sorted(required_mitr_keys - observed_mitr_keys)
    if missing_mitr:
        raise ValueError(f"frozen Spec03 mITR coverage is incomplete; missing {len(missing_mitr)} keys")
    return {
        "spec": "04_skip_causality_core",
        "scope": "all-six-mixerloop-and-fullloop-checkpoints",
        "checkpoints": [str(row["checkpoint"]) for row in selected],
        "checkpoint_count": len(selected),
        "sample_count": len(dataset.sample_ids),
        "skip_position_count": target_count,
        "expected_skip_sample_rows": target_count * len(dataset.sample_ids),
        "frozen_mitr_key_count": len(observed_mitr_keys),
        "output_root": str(output_root),
        "wandb_root": str(wandb_root),
        "output_root_cap_bytes": int(cap_bytes),
        "wandb_mode": "offline",
        "frozen_mitr_paths": [str(Path(path)) for path in mitr_paths],
        "frozen_mitr_source_groups": {path: sorted(groups) for path, groups in source_groups.items()},
        "frozen_mitr_source_separation_pass": True,
        "sequence_concatenation": False,
        "runtime_started": False,
    }


def merge_core_scores(
    detail_rows: Sequence[dict[str, Any]],
    core_results: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Fill the final detail-table CORE column from selected-run JSON records."""
    scores: dict[tuple[str, int, int], float] = {}
    for result in core_results:
        selection = result.get("selection")
        if not isinstance(selection, dict):
            raise ValueError("CORE result lacks a selection record")
        key = (str(selection["checkpoint"]), int(selection["layer"]), int(selection["pass"]))
        score = result.get("core_metric")
        if not _finite(score):
            raise ValueError(f"CORE result has non-finite metric for {key}")
        value = float(score)
        if key in scores:
            raise ValueError(f"duplicate CORE selection result for {key}")
        scores[key] = value
    merged = []
    applied = 0
    for row in detail_rows:
        output = dict(row)
        key = (str(row["checkpoint"]), int(row["layer"]), int(row["pass"]))
        if key in scores:
            output["core_score_if_run"] = scores[key]
            applied += 1
        merged.append(output)
    return merged, applied


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0 + 1.0
        for ordered_index in order[index:end]:
            ranks[ordered_index] = rank
        index = end
    return ranks


def spearman_rho(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("Spearman inputs must have equal length >= 2")
    x = _rankdata([float(value) for value in x_values])
    y = _rankdata([float(value) for value in y_values])
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_norm = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    y_norm = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    if x_norm == 0.0 or y_norm == 0.0:
        return 0.0
    return numerator / (x_norm * y_norm)


def _valid_mitr_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _finite(row.get("mITR")) and _finite(row.get("delta_loss"))]


def bootstrap_spearman(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260722,
) -> dict[str, Any]:
    valid = _valid_mitr_rows(rows)
    if len(valid) < 2:
        return {
            "rho": None,
            "ci_low": None,
            "ci_high": None,
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_unit": "layer_pass",
            "usable_rows": len(valid),
        }
    x = [float(row["mITR"]) for row in valid]
    y = [float(row["delta_loss"]) for row in valid]
    rho = spearman_rho(x, y)
    rng = random.Random(int(seed))
    boot = []
    for _ in range(int(bootstrap_samples)):
        indices = [rng.randrange(len(valid)) for _ in valid]
        boot.append(spearman_rho([x[index] for index in indices], [y[index] for index in indices]))
    boot.sort()
    low_index = max(0, min(len(boot) - 1, int(0.025 * len(boot))))
    high_index = max(0, min(len(boot) - 1, int(0.975 * len(boot)) - 1))
    return {
        "rho": float(rho),
        "ci_low": float(boot[low_index]),
        "ci_high": float(boot[high_index]),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_unit": "layer_pass",
        "usable_rows": len(valid),
    }


def select_extreme_positions(rows: Sequence[dict[str, Any]], *, count: int = 5) -> dict[str, list[dict[str, Any]]]:
    if count <= 0:
        raise ValueError("count must be positive")
    ordered = sorted(rows, key=lambda row: (float(row["delta_loss"]), int(row["layer"]), int(row["pass"])))
    projection = lambda row: {
        "checkpoint": str(row["checkpoint"]),
        "scale": str(row["scale"]),
        "condition": str(row["condition"]),
        "layer": int(row["layer"]),
        "pass": int(row["pass"]),
        "delta_loss": float(row["delta_loss"]),
        "mITR": row.get("mITR", ""),
    }
    return {
        "bottom": [projection(row) for row in ordered[:count]],
        "top": [projection(row) for row in ordered[-count:][::-1]],
    }


def mixer_state_digest(model: torch.nn.Module) -> dict[str, Any]:
    digest = hashlib.sha256()
    mixer_count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ != "GatedDeltaNet":
            continue
        mixer_count += 1
        digest.update(name.encode("utf-8"))
        digest.update(model_state_digest(module).encode("ascii"))
    return {
        "digest": digest.hexdigest(),
        "mixer_count": mixer_count,
        "cache_protocol": "past_key_values=None; update_layer_cache is a no-op",
    }


def _write_csv_guarded(rows: Sequence[dict[str, Any]], path: Path, *, fieldnames: Sequence[str], output_root: Path, cap_bytes: int) -> dict[str, Any]:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue().encode("utf-8")
    current = root_size_bytes(output_root)
    if current + len(payload) > int(cap_bytes):
        raise RuntimeError(f"output root cap exceeded before write: {current} + {len(payload)} > {cap_bytes}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "pre_write_root_bytes": current,
        "post_write_root_bytes": root_size_bytes(output_root),
        "cap_bytes": int(cap_bytes),
    }


def _append_jsonl_guarded(path: Path, payload: dict[str, Any], *, output_root: Path, cap_bytes: int) -> dict[str, Any]:
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    before = root_size_bytes(output_root)
    if before + len(encoded) > int(cap_bytes):
        raise RuntimeError(f"output root cap exceeded before stage write: {before} + {len(encoded)} > {cap_bytes}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(encoded)
    return {
        "path": str(path),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "pre_write_root_bytes": before,
        "post_write_root_bytes": root_size_bytes(output_root),
        "cap_bytes": int(cap_bytes),
    }


def _write_json_guarded_record(path: Path, payload: Any, *, output_root: Path, cap_bytes: int) -> dict[str, Any]:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    before = root_size_bytes(output_root)
    if before + len(encoded) > int(cap_bytes):
        raise RuntimeError(f"output root cap exceeded before JSON write: {before} + {len(encoded)} > {cap_bytes}")
    write_json_guarded(path, payload, output_root=output_root, cap_bytes=cap_bytes)
    return {
        "path": str(path),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "pre_write_root_bytes": before,
        "post_write_root_bytes": root_size_bytes(output_root),
        "cap_bytes": int(cap_bytes),
    }


def _write_self_audited_json(
    path: Path,
    write_records: Sequence[dict[str, Any]],
    *,
    output_root: Path,
    cap_bytes: int,
    fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before = root_size_bytes(output_root)
    self_record: dict[str, Any] = {
        "path": str(path),
        "bytes": 0,
        "sha256": None,
        "pre_write_root_bytes": before,
        "post_write_root_bytes": before,
        "cap_bytes": int(cap_bytes),
        "self_referential": True,
    }
    for _ in range(10):
        payload = {
            "write_records": list(write_records),
            "output_root_cap_bytes": int(cap_bytes),
            **(fields or {}),
            "self_write_record": self_record,
        }
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        updated = {**self_record, "bytes": len(encoded), "post_write_root_bytes": before + len(encoded)}
        if updated == self_record:
            break
        self_record = updated
    else:
        raise RuntimeError("self-audited JSON record did not converge")
    if before + len(encoded) > int(cap_bytes):
        raise RuntimeError(f"output root cap exceeded before self-audited JSON write: {before} + {len(encoded)} > {cap_bytes}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    actual = root_size_bytes(output_root)
    if actual != self_record["post_write_root_bytes"]:
        raise RuntimeError(f"self-audited JSON post-write byte mismatch: recorded {self_record['post_write_root_bytes']} actual {actual}")
    return self_record


def _init_offline_wandb(output_root: Path) -> tuple[Any, dict[str, Any]]:
    mode = os.environ.get("WANDB_MODE", "offline")
    if mode != "offline":
        raise RuntimeError(f"Spec04 requires WANDB_MODE=offline for this runner, got {mode!r}")
    import wandb

    wandb_root = output_root.parent / f".{output_root.name}.wandb"
    if wandb_root.exists() and any(wandb_root.iterdir()):
        raise FileExistsError(f"Spec04 W&B run root must be new and empty: {wandb_root}")
    wandb_root.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project="physics_for_llm",
        entity="gyy0592-ucsc",
        mode="offline",
        dir=str(wandb_root),
        name="spec04-skip-ablation",
        config={"spec": "04_skip_causality_core", "output_root": str(output_root)},
    )
    return run, {
        "mode": "offline",
        "cloud_sync": False,
        "initialized": True,
        "run_id": str(run.id),
        "run_dir": str(run.dir),
        "run_root": str(wandb_root),
        "project": "physics_for_llm",
        "entity": "gyy0592-ucsc",
    }


def _aggregate_detail_rows(sample_rows: Sequence[dict[str, Any]], mitr_index: dict[tuple[str, str, int, int], float]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, int], dict[str, float]] = {}
    for row in sample_rows:
        key = (str(row["checkpoint"]), str(row["scale"]), str(row["condition"]), int(row["layer"]), int(row["pass"]))
        item = grouped.setdefault(key, {"normal_ce_sum": 0.0, "skip_ce_sum": 0.0, "num_positions": 0.0})
        item["normal_ce_sum"] += float(row["normal_ce_sum"])
        item["skip_ce_sum"] += float(row["skip_ce_sum"])
        item["num_positions"] += float(row["num_positions"])
    detail = []
    for checkpoint, scale, condition, layer, pass_index in sorted(grouped):
        item = grouped[(checkpoint, scale, condition, layer, pass_index)]
        normal_loss = item["normal_ce_sum"] / item["num_positions"]
        skip_loss = item["skip_ce_sum"] / item["num_positions"]
        detail.append(
            {
                "checkpoint": checkpoint,
                "scale": scale,
                "condition": condition,
                "layer": layer,
                "pass": pass_index,
                "normal_loss": normal_loss,
                "skip_loss": skip_loss,
                "delta_loss": skip_loss - normal_loss,
                "mITR": mitr_index.get((condition, scale, layer, pass_index), ""),
                "core_score_if_run": "",
            }
        )
    return detail


def load_mitr_index(paths: Sequence[Path]) -> dict[tuple[str, str, int, int], float]:
    index: dict[tuple[str, str, int, int], float] = {}
    for path in paths:
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(f"missing mITR source: {source_path}")
        with source_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (str(row["condition"]), str(row["scale"]), int(row["layer"]), int(row["pass"]))
                if key in index:
                    raise ValueError(f"duplicate frozen mITR key across sources: {key}")
                index[key] = float(row["mITR"])
    return index


def run_skip_eval(
    *,
    spec01_root: Path,
    data_dir: Path,
    output_root: Path,
    outputs_root: Path = Path("outputs"),
    mitr_paths: Sequence[Path] = (),
    device: str = "cuda",
    cap_bytes: int = OUTPUT_ROOT_CAP_BYTES,
    sample_limit: int | None = None,
    batch_size: int = 4,
    checkpoint_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Spec04 output root must be absent before launch: {output_root}")
    output_root.mkdir(parents=True)
    pre_wandb_root_bytes = root_size_bytes(output_root)
    wandb_run, wandb_metadata = _init_offline_wandb(output_root)
    post_wandb_root_bytes = root_size_bytes(output_root)
    if post_wandb_root_bytes > int(cap_bytes):
        raise RuntimeError(f"output root cap exceeded after W&B initialization: {post_wandb_root_bytes} > {cap_bytes}")
    dataset = load_fixed_probe_dataset(Path(spec01_root))
    sample_ids = dataset.sample_ids[: int(sample_limit)] if sample_limit is not None else list(dataset.sample_ids)
    inventory = parse_released_checkpoint_inventory(Path(outputs_root))
    checkpoints = [row for row in inventory if row["condition"] in {"mixerloop", "fullloop"}]
    if checkpoint_names:
        requested = {str(name) for name in checkpoint_names}
        available = {str(row["checkpoint"]) for row in checkpoints}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"requested checkpoints are unavailable: {missing}")
        checkpoints = [row for row in checkpoints if str(row["checkpoint"]) in requested]
    if not checkpoints:
        raise RuntimeError("no MixerLoop or Full Loop checkpoints found")
    mitr_index = load_mitr_index(list(mitr_paths))
    sample_rows: list[dict[str, Any]] = []
    write_records: list[dict[str, Any]] = []
    checkpoint_summaries: list[dict[str, Any]] = []
    stage_path = output_root / "runtime_stage.jsonl"
    start_time = time.time()
    for checkpoint_spec in checkpoints:
        checkpoint_path = Path(str(checkpoint_spec["checkpoint_path"]))
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True, dtype=torch.float32).to(device)
        model.eval()
        before_digest = model_state_digest(model)
        before_mixer_digest = mixer_state_digest(model)
        targets = [
            (layer, pass_index)
            for layer in range(int(checkpoint_spec["num_layers"]))
            for pass_index in range(1, int(checkpoint_spec["loop_count"]) + 1)
        ]
        checkpoint_start = time.perf_counter()
        for batch_start in range(0, len(sample_ids), int(batch_size)):
            batch_ids = sample_ids[batch_start : batch_start + int(batch_size)]
            inputs = torch.cat(
                [_load_fixed_input_ids(dataset, sample_id=sample_id, data_dir=Path(data_dir))[0] for sample_id in batch_ids],
                dim=0,
            ).to(device)
            with torch.no_grad():
                normal_logits = model(input_ids=inputs).logits
                normal_stats = [
                    next_token_sufficient_stats(normal_logits[index : index + 1], inputs[index : index + 1], metric_start=dataset.metric_start, metric_end=dataset.metric_end)
                    for index in range(len(batch_ids))
                ]
                for layer, pass_index in targets:
                    skip_logits = model(input_ids=inputs, skip_layer=layer, skip_pass=pass_index).logits
                    for index, sample_id in enumerate(batch_ids):
                        skip_stats = next_token_sufficient_stats(skip_logits[index : index + 1], inputs[index : index + 1], metric_start=dataset.metric_start, metric_end=dataset.metric_end)
                        fixed_record = dataset.records_by_sample_id[int(sample_id)]
                        sample_rows.append(
                            {
                                "checkpoint": str(checkpoint_spec["checkpoint"]),
                                "scale": str(checkpoint_spec["scale"]),
                                "condition": str(checkpoint_spec["condition"]),
                                "layer": int(layer),
                                "pass": int(pass_index),
                                "sample_id": int(sample_id),
                                "input_sha256": str(fixed_record["input_sha256"]),
                                "precision": "float32",
                                "normal_ce_sum": float(normal_stats[index]["ce_sum"]),
                                "skip_ce_sum": float(skip_stats["ce_sum"]),
                                "num_positions": int(normal_stats[index]["num_positions"]),
                            }
                        )
            stage = {
                "stage": "batch_complete",
                "checkpoint": str(checkpoint_spec["checkpoint"]),
                "sample_start": int(batch_start),
                "sample_count": len(batch_ids),
                "target_count": len(targets),
                "output_root_bytes": root_size_bytes(output_root),
                "elapsed_seconds": time.time() - start_time,
            }
            write_records.append(_append_jsonl_guarded(stage_path, stage, output_root=output_root, cap_bytes=cap_bytes))
            if root_size_bytes(output_root) > int(cap_bytes):
                raise RuntimeError(f"output root cap exceeded after stage write: {root_size_bytes(output_root)} > {cap_bytes}")
        after_digest = model_state_digest(model)
        after_mixer_digest = mixer_state_digest(model)
        frozen = before_digest == after_digest
        mixer_frozen = before_mixer_digest == after_mixer_digest
        if not frozen or not mixer_frozen:
            raise RuntimeError(f"frozen-state digest mutation detected for {checkpoint_spec['checkpoint']}")
        checkpoint_summaries.append(
            {
                "checkpoint": str(checkpoint_spec["checkpoint"]),
                "condition": str(checkpoint_spec["condition"]),
                "scale": str(checkpoint_spec["scale"]),
                "num_layers": int(checkpoint_spec["num_layers"]),
                "loop_count": int(checkpoint_spec["loop_count"]),
                "sample_count": len(sample_ids),
                "target_count": len(targets),
                "elapsed_seconds": time.perf_counter() - checkpoint_start,
                "frozen_state_pass": True,
                "mixer_state_pass": mixer_frozen,
                "mixer_state_digest_before": before_mixer_digest,
                "mixer_state_digest_after": after_mixer_digest,
                "sequence_concatenation": False,
            }
        )
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    detail_rows = _aggregate_detail_rows(sample_rows, mitr_index)
    by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    for row in detail_rows:
        by_checkpoint.setdefault(str(row["checkpoint"]), []).append(row)
    spearman_rows = []
    selections = {}
    for checkpoint, rows in sorted(by_checkpoint.items()):
        summary = bootstrap_spearman(rows)
        spearman_rows.append({"checkpoint": checkpoint, **summary})
        selections[checkpoint] = select_extreme_positions(rows)
    detail_record = _write_csv_guarded(detail_rows, output_root / "skip_ablation_detail.csv", fieldnames=DETAIL_COLUMNS, output_root=output_root, cap_bytes=cap_bytes)
    write_records.append(detail_record)
    sample_record = _write_csv_guarded(sample_rows, output_root / "skip_ablation_sample_stats.csv", fieldnames=SAMPLE_COLUMNS, output_root=output_root, cap_bytes=cap_bytes)
    write_records.append(sample_record)
    spearman_record = _write_csv_guarded(spearman_rows, output_root / "spearman_bootstrap.csv", fieldnames=["checkpoint", "rho", "ci_low", "ci_high", "bootstrap_samples", "bootstrap_unit", "usable_rows"], output_root=output_root, cap_bytes=cap_bytes)
    write_records.append(spearman_record)
    selection_path = output_root / "core_selection_manifest.json"
    selection_record = _write_json_guarded_record(selection_path, selections, output_root=output_root, cap_bytes=cap_bytes)
    write_records.append(selection_record)
    manifest_path = output_root / "run_manifest.json"
    checks_path = output_root / "run_checks.json"
    audit_path = output_root / "write_audit.json"
    manifest = {
        "spec": "04_skip_causality_core",
        "run_type": "skip_ablation_streamed",
        "runtime_started": True,
        "accepted_output": False,
        "sample_count": len(sample_ids),
        "checkpoint_count": len(checkpoints),
        "checkpoint_selection": [str(row["checkpoint"]) for row in checkpoints],
        "checkpoint_summaries": checkpoint_summaries,
        "target_count": len(detail_rows),
        "bootstrap_samples": 10_000,
        "sequence_concatenation": False,
        "batch_size": int(batch_size),
        "spec01_root": str(spec01_root),
        "output_root": str(output_root),
        "output_root_cap_bytes": int(cap_bytes),
        "output_root_bytes_before_wandb": pre_wandb_root_bytes,
        "output_root_bytes_after_wandb": post_wandb_root_bytes,
        "wandb": wandb_metadata,
        "precision": "float32",
        "frozen_state_pass": all(summary["frozen_state_pass"] for summary in checkpoint_summaries),
        "artifacts": [
            detail_record,
            sample_record,
            spearman_record,
            selection_record,
            {"path": str(stage_path), "bytes": stage_path.stat().st_size},
            {"path": str(manifest_path), "required": True},
            {"path": str(checks_path), "required": True},
            {"path": str(audit_path), "required": True},
        ],
        "write_records": list(write_records),
        "command": list(sys.argv),
        "elapsed_seconds": time.time() - start_time,
    }
    manifest_record = _write_json_guarded_record(manifest_path, manifest, output_root=output_root, cap_bytes=cap_bytes)
    write_records.append(manifest_record)
    audit_record = _write_self_audited_json(
        audit_path,
        write_records,
        output_root=output_root,
        cap_bytes=cap_bytes,
        fields={"planned_final_writes": [str(checks_path)]},
    )
    write_records.append(audit_record)
    checks = {
        "runtime_started": True,
        "accepted_output": False,
        "checkpoint_loaded": len(checkpoint_summaries) == len(checkpoints),
        "frozen_state_pass": manifest["frozen_state_pass"],
        "mixer_state_pass": all(summary["mixer_state_pass"] for summary in checkpoint_summaries),
        "finite_detail_rows": all(_finite(row["normal_loss"]) and _finite(row["skip_loss"]) and _finite(row["delta_loss"]) for row in detail_rows),
        "required_artifacts_written": all(
            Path(item["path"]).is_file() for item in manifest["artifacts"] if Path(item["path"]) != checks_path
        ),
        "output_root_cap_bytes": int(cap_bytes),
        "output_root_bytes": root_size_bytes(output_root),
        "wandb_mode": wandb_metadata["mode"],
        "sequence_concatenation": False,
        "output_root_bytes_before_checks": root_size_bytes(output_root),
        "output_root_bytes_before_wandb": pre_wandb_root_bytes,
        "output_root_bytes_after_wandb": post_wandb_root_bytes,
        "write_records": list(write_records),
    }
    checks_record = _write_self_audited_json(
        checks_path,
        write_records,
        output_root=output_root,
        cap_bytes=cap_bytes,
        fields=checks,
    )
    write_records.append(checks_record)
    if not all(Path(item["path"]).is_file() for item in manifest["artifacts"]):
        raise RuntimeError("required Spec04 artifacts were not all written")
    if root_size_bytes(output_root) > int(cap_bytes):
        raise RuntimeError(f"output root cap exceeded after checks: {root_size_bytes(output_root)} > {cap_bytes}")
    wandb_run.finish()
    return {"manifest": manifest, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec01-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--mitr-csv", type=Path, action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cap-bytes", type=int, default=OUTPUT_ROOT_CAP_BYTES)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint", dest="checkpoint_names", action="append", default=[])
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--formal-preflight", action="store_true")
    args = parser.parse_args()
    if args.formal and args.formal_preflight:
        parser.error("--formal and --formal-preflight are mutually exclusive")
    if args.formal or args.formal_preflight:
        if args.sample_limit is not None:
            parser.error("formal Spec04 paths require all 1,024 fixed samples; omit --sample-limit")
        if args.formal and set(args.checkpoint_names) != set(FORMAL_CHECKPOINTS):
            parser.error("--formal requires all six formal checkpoint names")
        formal_plan = validate_formal_scope(
            spec01_root=args.spec01_root,
            outputs_root=args.outputs_root,
            output_root=args.output_root,
            mitr_paths=args.mitr_csv,
            cap_bytes=args.cap_bytes,
        )
        if args.formal_preflight:
            print(json.dumps(formal_plan, sort_keys=True))
            return
    result = run_skip_eval(
        spec01_root=args.spec01_root,
        data_dir=args.data_dir,
        output_root=args.output_root,
        outputs_root=args.outputs_root,
        mitr_paths=args.mitr_csv,
        device=args.device,
        cap_bytes=args.cap_bytes,
        sample_limit=args.sample_limit,
        batch_size=args.batch_size,
        checkpoint_names=args.checkpoint_names,
    )
    print(json.dumps(result["checks"], sort_keys=True))


if __name__ == "__main__":
    main()
