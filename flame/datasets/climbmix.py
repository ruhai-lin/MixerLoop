#!/usr/bin/env python3
"""Download and pretokenize the fixed 170-shard ClimbMix-10B corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import requests
import sentencepiece as spm
import torch

REPOSITORY_ID = "karpathy/climbmix-400b-shuffle"
REPOSITORY = f"https://huggingface.co/datasets/{REPOSITORY_ID}/resolve/main"
TRAIN_SHARDS = tuple(range(170))
VALIDATION_SHARD = 6542


class ClimbMixDataset(torch.utils.data.IterableDataset):
    """Stateful reader matching the pretokenized LT2.c/LT3.c data stream."""

    TRAIN_FILES = tuple(f"shard_{index:05d}.bin" for index in TRAIN_SHARDS)
    VALIDATION_FILE = f"shard_{VALIDATION_SHARD:05d}.bin"

    def __init__(self, data_dir: str, split: str = "train", seed: int = 42):
        super().__init__()
        if not data_dir:
            raise ValueError("ClimbMix requires --training.data_dir")
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.split = split or "train"
        self.seed = int(seed)
        if self.split not in {"train", "validation", "val"}:
            raise ValueError(f"ClimbMix split must be train or validation, got {self.split!r}")

        expected = self.TRAIN_FILES if self.split == "train" else (self.VALIDATION_FILE,)
        missing = [name for name in expected if not (self.data_dir / name).is_file()]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"ClimbMix is incomplete in {self.data_dir}: missing {len(missing)} shard(s), e.g. {preview}. "
                "Run `python -m flame.datasets.climbmix` first."
            )
        self.shards = [self.data_dir / name for name in expected]
        self.seq_len: int | None = None
        self.rank = 0
        self.world_size = 1
        self._state: dict[str, Any] | None = None

    @property
    def num_shards(self) -> int:
        return len(self.shards)

    def configure(self, seq_len: int, rank: int, world_size: int, num_workers: int) -> None:
        if num_workers != 0:
            raise ValueError("Exact ClimbMix checkpoint/resume requires --training.num_workers 0")
        self.seq_len = int(seq_len)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def _new_state(self) -> dict[str, Any]:
        rng = random.Random(self.seed + 1_000_003 * self.rank)
        shard_order = list(range(len(self.shards)))
        rng.shuffle(shard_order)
        return {
            "epoch": 0,
            "rng_state": rng.getstate(),
            "shard_order": shard_order,
            "shard_position": 0,
            "window_order": None,
            "window_position": 0,
        }

    def __iter__(self):
        if self.seq_len is None:
            raise RuntimeError("ClimbMix dataset must be configured before iteration")
        if self._state is None:
            self._state = self._new_state()

        rng = random.Random()
        rng.setstate(self._state["rng_state"])
        while True:
            if self._state["shard_position"] >= len(self._state["shard_order"]):
                self._state["epoch"] += 1
                self._state["shard_position"] = 0
                rng.shuffle(self._state["shard_order"])
                self._state["rng_state"] = rng.getstate()

            shard_index = self._state["shard_order"][self._state["shard_position"]]
            tokens = np.memmap(self.shards[shard_index], dtype=np.uint16, mode="r")
            if self._state["window_order"] is None:
                # Mirror LT3.c: floor(N / L) - 1 windows per shard.
                num_windows = len(tokens) // self.seq_len - 1
                if num_windows <= 0:
                    raise ValueError(f"Shard is too short for seq_len={self.seq_len}: {self.shards[shard_index]}")
                order = list(range(num_windows))
                rng.shuffle(order)
                self._state["window_order"] = order
                self._state["window_position"] = 0
                self._state["rng_state"] = rng.getstate()

            order = self._state["window_order"]
            while self._state["window_position"] < len(order):
                window_index = order[self._state["window_position"]]
                self._state["window_position"] += 1
                start = window_index * self.seq_len
                chunk = torch.from_numpy(tokens[start : start + self.seq_len + 1].astype(np.int64))
                yield {"input_ids": chunk[:-1], "labels": chunk[1:]}

            self._state["shard_position"] += 1
            self._state["window_order"] = None
            self._state["window_position"] = 0

    def state_dict(self) -> dict[str, Any]:
        return {"state": deepcopy(self._state)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._state = deepcopy(state_dict.get("state"))


def filename(index: int, suffix: str) -> str:
    return f"shard_{index:05d}.{suffix}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(args: tuple[int, Path]) -> Path:
    index, output_dir = args
    target = output_dir / filename(index, "parquet")
    if target.is_file():
        return target
    temporary = target.with_suffix(".parquet.part")
    url = f"{REPOSITORY}/{target.name}"
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    os.replace(temporary, target)
    return target


def tokenize_one(args: tuple[Path, Path, Path]) -> tuple[Path, int]:
    parquet_path, output_path, tokenizer_path = args
    if output_path.is_file():
        return output_path, output_path.stat().st_size // np.dtype(np.uint16).itemsize

    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    parquet = pq.ParquetFile(parquet_path)
    temporary = output_path.with_suffix(".bin.part")
    token_count = 0
    with temporary.open("wb") as handle:
        for row_group in range(parquet.num_row_groups):
            pieces: list[int] = []
            texts = parquet.read_row_group(row_group, columns=["text"]).column("text").to_pylist()
            for text in texts:
                pieces.append(tokenizer.bos_id())
                pieces.extend(tokenizer.encode(text.strip()))
            tokens = np.asarray(pieces, dtype=np.uint16)
            tokens.tofile(handle)
            token_count += int(tokens.size)
    os.replace(temporary, output_path)
    return output_path, token_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data/climbmix-10b"))
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "assets/tokenizer/tokenizer.model",
    )
    parser.add_argument("--download_workers", type=int, default=8)
    parser.add_argument("--tokenize_workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--skip_tokenize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    indices = (*TRAIN_SHARDS, VALIDATION_SHARD)

    if not args.skip_download:
        with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
            for path in executor.map(download_one, ((index, data_dir) for index in indices)):
                print(f"[download] {path.name}")

    parquet_paths = [data_dir / filename(index, "parquet") for index in indices]
    missing = [path for path in parquet_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} parquet shards; first missing: {missing[0]}")

    token_counts: dict[str, int] = {}
    if not args.skip_tokenize:
        jobs = [
            (path, path.with_suffix(".bin"), tokenizer_path)
            for path in parquet_paths
        ]
        with ProcessPoolExecutor(max_workers=args.tokenize_workers) as executor:
            for path, count in executor.map(tokenize_one, jobs):
                token_counts[path.name] = count
                print(f"[tokenize] {path.name}: {count:,} tokens")

    bin_paths = [path.with_suffix(".bin") for path in parquet_paths]
    missing = [path for path in bin_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} token shards; first missing: {missing[0]}")

    records = []
    for index, parquet_path, bin_path in zip(indices, parquet_paths, bin_paths):
        count = token_counts.get(bin_path.name, bin_path.stat().st_size // 2)
        records.append(
            {
                "index": index,
                "split": "validation" if index == VALIDATION_SHARD else "train",
                "parquet": parquet_path.name,
                "parquet_sha256": sha256(parquet_path),
                "tokens": bin_path.name,
                "tokens_sha256": sha256(bin_path),
                "token_count": count,
            }
        )
    manifest = {
        "dataset": REPOSITORY_ID,
        "train_shards": 170,
        "validation_shard": VALIDATION_SHARD,
        "tokenizer_sha256": sha256(tokenizer_path),
        "train_tokens": sum(record["token_count"] for record in records if record["split"] == "train"),
        "validation_tokens": sum(
            record["token_count"] for record in records if record["split"] == "validation"
        ),
        "shards": records,
    }
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {manifest_path}")


if __name__ == "__main__":
    main()
