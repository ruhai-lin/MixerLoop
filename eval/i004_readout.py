"""A1 original-head readout and durable shared features; run from MixerLoop root."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import torch

from eval.i004_capture import capture_batch, head_metrics, layer_roles, load_model
from eval.i004_data import load_windows


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def save_shard(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".writing")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_observer(specification, output_dir):
    module_name, factory_name = specification.rsplit(":", 1)
    if module_name.endswith(".py"):
        source = Path(module_name)
        name = f"i004_observer_{source.stem}"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    return getattr(module, factory_name)(output_dir)


def publish_progress(output, manifest, per_condition, observers):
    """Publish one consistent batch; retain only the current and previous slot."""
    batch_number = len(manifest["resource_batches"])
    relative = Path("progress") / f"slot_{batch_number % 2}.pt"
    snapshot = {"per_condition": per_condition,
                "observers": [observer.state_dict() for observer, _ in observers]}
    save_shard(output / relative, snapshot)
    manifest["progress_path"] = str(relative)
    write_json(output / "manifest.json", manifest)


def metric_rows(metrics, ids, splits, condition_id):
    return [
        {"condition_id": condition_id, "window_id": int(ids[index]), "split": splits[index],
         "nll_sum": float(metrics["nll_sum"][index]),
         "correct_sum": int(metrics["correct_sum"][index]),
         "p_correct_sum": float(metrics["p_correct_sum"][index]), "count": metrics["count"]}
        for index in range(len(ids))
    ]


def aggregate(rows):
    output = {}
    for split in ("fit", "validation"):
        selected = [row for row in rows if row["split"] == split]
        count = sum(row["count"] for row in selected)
        if count:
            output[split] = {
                "ce": sum(row["nll_sum"] for row in selected) / count,
                "top1": sum(row["correct_sum"] for row in selected) / count,
                "p_correct": sum(row["p_correct_sum"] for row in selected) / count,
                "count": count, "num_windows": len(selected),
            }
    return output


def paired_differences(conditions, per_condition):
    rows = []
    for condition_id, condition in conditions.items():
        baseline_ids = [f"{condition['model']}-{condition['scale']}:native_full"]
        if condition["model"] in ("mixerloop", "fullloop"):
            baseline_ids.extend(
                f"{condition['model']}-{condition['scale']}:layer{condition['layer']}:loop{loop}"
                for loop in range(1, condition["loop"]))
        for baseline_id in baseline_ids:
            baseline = {row["window_id"]: row for row in per_condition[baseline_id]}
            for row in per_condition[condition_id]:
                ref = baseline[row["window_id"]]
                rows.append({
                    "condition_id": condition_id, "baseline_id": baseline_id,
                    "window_id": row["window_id"], "split": row["split"],
                    "delta_ce": (row["nll_sum"] - ref["nll_sum"]) / row["count"],
                    "delta_top1": (row["correct_sum"] - ref["correct_sum"]) / row["count"],
                    "delta_p_correct": (row["p_correct_sum"] - ref["p_correct_sum"]) / row["count"],
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", nargs="+", default=["15m", "42m", "110m"])
    parser.add_argument("--models", nargs="+", default=["gdn", "mixerloop"])
    parser.add_argument("--batch-windows", type=int, default=4)
    parser.add_argument("--head-token-chunk", type=int, default=512)
    parser.add_argument("--smoke", action="store_true", help="Two existing fit and two existing validation windows; full context/depth/loops.")
    parser.add_argument("--observer", action="append", default=[], help="Importable module:factory; factory(output_dir) returns observer(meta,tensor).")
    parser.add_argument("--observer-output", action="append", type=Path, default=[], help="Corresponding observer's independent worktree output directory.")
    parser.add_argument("--pause-after-batches", type=int, help="Publish this many additional batches and return, for short interruption/resume smoke.")
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = load_windows(args.selection, args.data_root)
    indices = list(range(len(dataset["sample_ids"])))
    if args.smoke:
        indices = ([i for i in indices if dataset["split"][i] == "fit"][:2]
                   + [i for i in indices if dataset["split"][i] == "validation"][:2])
    manifest_path = args.output / "manifest.json"
    previous_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    observer_specs = previous_manifest["observer_specs"] if previous_manifest is not None else args.observer
    observer_outputs = (previous_manifest.get("observer_outputs", args.observer_output)
                        if previous_manifest is not None else args.observer_output)
    observers = []
    for observer_index, specification in enumerate(observer_specs):
        module_name = specification.rsplit(":", 1)[0]
        output_dir = (Path(observer_outputs[observer_index]) if observer_index < len(observer_outputs)
                      else args.output / "observers" / Path(module_name).stem)
        output_dir.mkdir(parents=True, exist_ok=True)
        observers.append((load_observer(specification, output_dir), output_dir))

    def observe(meta, tensor):
        for observer, _ in observers:
            observer(meta, tensor)

    manifest = {
        "format": "i004_head_features_v1", "precision": "float32",
        "feature_location": "source_model_final_norm_after_target_block_ffn_residual",
        "features_shape": ["windows", 479, "hidden_width"],
        "selection": str(args.selection.resolve()), "data_root": str(args.data_root.resolve()),
        "data_provenance": dataset["provenance"], "smoke": args.smoke,
        "positions": dataset["positions"].tolist(), "conditions": {}, "original_heads": {},
        "native_full_metrics": {}, "metrics_path": "metrics.jsonl", "resource_batches": [],
        "seed": args.seed, "gpu": torch.cuda.get_device_name(), "code_commit": os.environ.get("I004_CODE_COMMIT", "recorded_by_parent_git_history"),
        "num_windows": len(indices), "observer_specs": observer_specs,
        "observer_outputs": [str(path) for _, path in observers],
        "completed_windows": {}, "finished": False,
    }
    per_condition = {}
    if previous_manifest is not None:
        manifest = previous_manifest
        if "progress_path" in manifest:
            snapshot = torch.load(args.output / manifest["progress_path"], map_location="cpu", weights_only=True)
            per_condition = snapshot["per_condition"]
            for (observer, _), state in zip(observers, snapshot["observers"]):
                observer.load_state_dict(state)
        else:
            # The first completed A1 smoke predates resumable observer snapshots.
            # Its published feature shards and scalar rows remain reusable.
            for line in (args.output / manifest["metrics_path"]).read_text().splitlines():
                row = json.loads(line)
                per_condition.setdefault(row["condition_id"], []).append(row)
            manifest["completed_windows"] = {}
            for condition in manifest["conditions"].values():
                checkpoint = f"{condition['model']}-{condition['scale']}"
                manifest["completed_windows"][checkpoint] = sum(shard["num_windows"] for shard in condition["shards"])
        print(json.dumps({"resuming_completed_windows": manifest["completed_windows"]}), flush=True)
    import wandb
    tracking = wandb.init(project="i004-local", mode="offline", dir=str(args.output),
                          name="a1-smoke" if args.smoke else "a1-capture", config=vars(args))
    started = time.perf_counter()
    batches_this_invocation = 0
    with (args.output / "metrics.jsonl").open("w") as metrics_file:
        for rows in per_condition.values():
            for row in rows:
                metrics_file.write(json.dumps(row) + "\n")
        for scale in args.scales:
            for kind in args.models:
                checkpoint = f"{kind}-{scale}"
                completed = manifest["completed_windows"].get(checkpoint, 0)
                if completed == len(indices):
                    continue
                model = load_model(args.weights_root / checkpoint)
                roles = layer_roles(model)
                original_head_path = Path("heads") / f"{checkpoint}.pt"
                if checkpoint not in manifest["original_heads"]:
                    save_shard(args.output / original_head_path, {
                        "weight": model.lm_head.weight.detach().cpu().clone(), "bias": None,
                        "model": kind, "scale": scale, "source": str((args.weights_root / checkpoint).resolve()),
                    })
                manifest["original_heads"][checkpoint] = str(original_head_path)
                baseline_id = f"{checkpoint}:native_full"
                per_condition.setdefault(baseline_id, [])
                for offset in range(completed, len(indices), args.batch_windows):
                    chosen = indices[offset:offset + args.batch_windows]
                    input_ids = dataset["input_ids"][chosen].pin_memory().to("cuda", non_blocking=True)
                    labels = dataset["labels"][chosen]
                    ids = dataset["sample_ids"][chosen]
                    splits = [dataset["split"][i] for i in chosen]
                    positions = dataset["positions"].to("cuda")
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    batch_started = time.perf_counter()
                    metadata = {"scale": scale, "sample_ids": ids, "split": splits,
                                "selected_positions": dataset["positions"]}
                    records = capture_batch(model, input_ids, observer=observe if observers else None,
                                            metadata=metadata,
                                            observer_layers=[layer for layer, role in roles.items() if role != "transfer"],
                                            capture_full_loop=kind == "fullloop")
                    torch.cuda.synchronize()
                    capture_seconds = time.perf_counter() - batch_started
                    readout_started = time.perf_counter()
                    with torch.inference_mode():
                        native = model.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state
                        native_metrics = head_metrics(native[:, positions], model.lm_head.weight, labels, args.head_token_chunk)
                    native_rows = metric_rows(native_metrics, ids, splits, baseline_id)
                    per_condition[baseline_id].extend(native_rows)
                    for row in native_rows:
                        metrics_file.write(json.dumps(row) + "\n")
                    final = records[(len(model.model.layers), max(loop for layer, loop in records if layer == len(model.model.layers)))]["head_input"]
                    native_difference = (final - native).abs()
                    equivalence = {"max_abs": float(native_difference.max()), "mean_abs": float(native_difference.mean())}
                    write_seconds = 0.0
                    for (layer, loop), record in records.items():
                        condition_id = f"{checkpoint}:layer{layer}:loop{loop}"
                        condition = manifest["conditions"].setdefault(condition_id, {
                            "model": kind, "scale": scale, "layer": layer, "loop": loop,
                            "role": roles[layer], "shards": [],
                        })
                        selected = record["head_input"][:, positions].contiguous()
                        metrics = head_metrics(selected, model.lm_head.weight, labels, args.head_token_chunk)
                        rows = metric_rows(metrics, ids, splits, condition_id)
                        per_condition.setdefault(condition_id, []).extend(rows)
                        for row in rows:
                            metrics_file.write(json.dumps(row) + "\n")
                        write_started = time.perf_counter()
                        relative = Path("features") / checkpoint / f"layer{layer}_loop{loop}" / f"batch_{offset:06d}.pt"
                        save_shard(args.output / relative, {"features": selected.cpu(), "labels": labels,
                                   "window_ids": ids, "positions": dataset["positions"], "split": splits})
                        write_seconds += time.perf_counter() - write_started
                        condition["shards"].append({"path": str(relative), "num_windows": len(chosen)})
                        condition["original_metrics"] = aggregate(per_condition[condition_id])
                    torch.cuda.synchronize()
                    batch_seconds = time.perf_counter() - batch_started
                    resource = {"checkpoint": checkpoint, "offset": offset, "windows": len(chosen),
                                "capture_seconds": capture_seconds, "readout_and_write_seconds": time.perf_counter() - readout_started,
                                "write_seconds": write_seconds, "batch_seconds": batch_seconds,
                                "windows_per_second": len(chosen) / batch_seconds,
                                "peak_memory_allocated": torch.cuda.max_memory_allocated(),
                                "native_final_normed_difference": equivalence}
                    manifest["resource_batches"].append(resource)
                    manifest["native_full_metrics"][baseline_id] = aggregate(per_condition[baseline_id])
                    manifest["completed_windows"][checkpoint] = offset + len(chosen)
                    metrics_file.flush()
                    publish_progress(args.output, manifest, per_condition, observers)
                    tracking.log({"batch_seconds": batch_seconds, "capture_seconds": capture_seconds,
                                  "peak_memory_bytes": resource["peak_memory_allocated"],
                                  "native_max_abs": equivalence["max_abs"]})
                    print(json.dumps(resource), flush=True)
                    del records, native, final, native_difference, input_ids, selected
                    batches_this_invocation += 1
                    if args.pause_after_batches is not None and batches_this_invocation >= args.pause_after_batches:
                        manifest["elapsed_seconds"] = manifest.get("elapsed_seconds", 0.0) + time.perf_counter() - started
                        write_json(args.output / "manifest.json", manifest)
                        tracking.finish()
                        print(json.dumps({"paused_after_committed_batch": len(manifest["resource_batches"]),
                                          "resume_command": "Repeat the same command without --pause-after-batches."}), flush=True)
                        return
                del model
                torch.cuda.empty_cache()
    with (args.output / "paired_differences.jsonl").open("w") as destination:
        for row in paired_differences(manifest["conditions"], per_condition):
            destination.write(json.dumps(row) + "\n")
    for observer, output_dir in observers:
        observer.write(output_dir)
    manifest["elapsed_seconds"] = manifest.get("elapsed_seconds", 0.0) + time.perf_counter() - started
    manifest["bytes_written"] = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    manifest["finished"] = True
    write_json(args.output / "manifest.json", manifest)
    tracking.finish()
    print(json.dumps({"output": str(args.output), "conditions": len(manifest["conditions"]),
                      "elapsed_seconds": manifest["elapsed_seconds"], "bytes_written": manifest["bytes_written"]}), flush=True)


if __name__ == "__main__":
    main()
