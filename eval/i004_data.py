"""Load the existing I004 ClimbMix windows without drawing new samples.

The manifest and its sibling ``probe_split_metadata.json`` own sample selection
and the fit/validation split. ``data_root`` is the directory containing the
manifest's binary shards. Each label follows its corresponding input position.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def load_windows(manifest_path, data_root, limit=None):
    """Return CPU token tensors and metadata for the saved selection order.

    ``input_ids`` has one full sequence per row; ``labels`` contains only the
    next-token targets at ``positions``. ``limit`` selects a prefix for smoke
    runs and never changes the saved sample IDs or split assignments.
    """
    manifest_path = Path(manifest_path).resolve()
    data_root = Path(data_root).resolve()
    split_path = manifest_path.with_name("probe_split_metadata.json")
    manifest = json.loads(manifest_path.read_text())
    split_metadata = json.loads(split_path.read_text())
    selected = manifest["records"][:limit]
    fit_ids = set(split_metadata["fit_sample_ids"])
    positions = torch.arange(
        manifest["metric_positions"]["start"],
        manifest["metric_positions"]["end"] + 1,
        dtype=torch.long,
    )
    input_ids = torch.empty(
        (len(selected), manifest["seq_len"]), dtype=torch.long
    )
    shards = {}
    records = []
    for row, saved in enumerate(selected):
        shard = saved.get("shard", manifest["validation_file"])
        if shard not in shards:
            shards[shard] = np.memmap(data_root / shard, mode="r", dtype=np.uint16)
        start = saved["start_token"]
        stop = start + saved["num_tokens"]
        input_ids[row] = torch.from_numpy(
            np.array(shards[shard][start:stop], dtype=np.int64)
        )
        records.append(
            {
                name: saved[name]
                for name in (
                    "sample_id", "shard", "window_index", "start_token",
                    "num_tokens", "num_non_padding_tokens",
                )
                if name in saved
            }
        )
    splits = [
        "fit" if record["sample_id"] in fit_ids else "validation"
        for record in records
    ]
    return {
        "input_ids": input_ids,
        "labels": input_ids[:, positions + 1],
        "sample_ids": torch.tensor(
            [record["sample_id"] for record in records], dtype=torch.long
        ),
        "split": splits,
        "records": records,
        "positions": positions,
        "provenance": {
            "manifest_path": str(manifest_path),
            "split_path": str(split_path),
            "data_root": str(data_root),
            "selection_seed": manifest["seed"],
            "source_sample_count": manifest["sample_count"],
            "selected_sample_count": len(records),
            "selected_split_counts": dict(Counter(splits)),
            "source_dtype": "uint16",
            "seq_len": manifest["seq_len"],
            "metric_positions": manifest["metric_positions"],
            "source_shards": [str(data_root / shard) for shard in shards],
        },
    }
