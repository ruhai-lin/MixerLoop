#!/usr/bin/env python3
"""Build the anonymous AAAI code-and-data supplement."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = Path(__file__).resolve().parent / "MixerLoop-Code-and-Data.zip"
ARCHIVE_ROOT = Path("MixerLoop")
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".sh", ".toml", ".txt"}
SOURCE_DIRS = ("assets", "configs", "custom_models", "eval", "flame", "tests")
TOP_LEVEL_FILES = ("LICENSE", "flame-installation.md", "pyproject.toml", "pytest.ini", "train.sh")
ITR_RUNS = (
    "itr-readout-15m",
    "itr-readout-110m",
    "itr-readout-15m-seed2028",
    "itr-readout-15m-context256",
    "itr-readout-random-control",
)
MODELS = tuple(
    f"{architecture}-{scale}"
    for architecture in ("gdn", "mixerloop", "fullloop")
    for scale in ("15m", "110m")
)


def sanitize(text: str) -> str:
    replacements = {
        "Ruhai Lin": "Anonymous Authors",
        "ruhai-lin/MixerLoop": "anonymous/MixerLoop",
        "github.com/ruhai-lin/MixerLoop": "anonymous.invalid/MixerLoop",
        "ruhai-lin/LT3.c": "anonymous/legacy-mixerloop",
        "ruhai-lin/LT2.c": "anonymous/legacy-fullloop",
        "ruhai-lin/gdn.c": "anonymous/legacy-gdn",
        "/home/ruhai/Projects/MixerLoop/": "",
        "LT2.c/LT3.c": "the legacy implementations",
        "LT3.c": "the legacy mixer-loop implementation",
        "/tmp/mixerloop-random-15m-20260725": "outputs/random-mixerloop-15m",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(
        r"/home/[^/\"'\\s]+/(?:Projects|Documents)/[^/\"'\\s]*/MixerLoop/",
        "",
        text,
    )


def strip_project_urls(text: str) -> str:
    """Remove author-controlled web pointers forbidden during AAAI review."""
    return re.sub(
        r"\n\[project\.urls\]\n.*?(?=\n\[[^\n]+\]\n|\Z)",
        "\n",
        text,
        flags=re.DOTALL,
    )


def archive_name(relative: Path) -> str:
    return str(ARCHIVE_ROOT / relative)


def add_bytes(
    archive: zipfile.ZipFile,
    relative: Path,
    payload: bytes,
    *,
    executable: bool = False,
) -> None:
    info = zipfile.ZipInfo(archive_name(relative), date_time=(2026, 7, 25, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = ((0o755 if executable else 0o644) & 0xFFFF) << 16
    archive.writestr(info, payload)


def add_file(archive: zipfile.ZipFile, source: Path, relative: Path) -> None:
    payload = source.read_bytes()
    if source.name == "provenance.json":
        provenance = json.loads(payload)
        for key in ("legacy_checkpoint", "legacy_sha256", "source_repo", "source_revision"):
            provenance.pop(key, None)
        training = provenance.get("training_config", {})
        for key in ("out_dir", "wandb_project", "wandb_run_name"):
            training.pop(key, None)
        payload = (json.dumps(provenance, indent=2) + "\n").encode("utf-8")
    elif source.suffix.lower() in TEXT_SUFFIXES:
        text = sanitize(payload.decode("utf-8"))
        if relative == Path("pyproject.toml"):
            text = strip_project_urls(text)
        payload = text.encode("utf-8")
    add_bytes(
        archive,
        relative,
        payload,
        executable=source.suffix == ".sh" or source.name in {"itr_eval.py", "throughput_eval.py"},
    )


def source_files() -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []
    for name in TOP_LEVEL_FILES:
        files.append((ROOT / name, Path(name)))
    for directory in SOURCE_DIRS:
        for source in sorted((ROOT / directory).rglob("*")):
            if source.is_file() and "__pycache__" not in source.parts and source.suffix != ".pyc":
                files.append((source, source.relative_to(ROOT)))
    files.append((ROOT / "paper/make_figures.py", Path("paper/make_figures.py")))
    for pattern in ("*.pdf", "*.png"):
        for source in sorted((ROOT / "paper/Figures").glob(pattern)):
            files.append((source, source.relative_to(ROOT)))

    for model in MODELS:
        base = ROOT / "outputs" / model
        files.append((base / "eval/core_eval.csv", Path("outputs") / model / "eval/core_eval.csv"))
        files.append((base / "provenance.json", Path("outputs") / model / "provenance.json"))
    for run in ITR_RUNS:
        base = ROOT / "outputs" / run
        for name in ("itr_eval.json", "itr_eval.md", "itr_events.csv"):
            files.append((base / name, Path("outputs") / run / name))
    for source in sorted((ROOT / "outputs").glob("throughput-*.json")):
        files.append((source, source.relative_to(ROOT)))
    return files


def main() -> None:
    readme = (ROOT / "paper/code-supplement-readme.md").read_text(encoding="utf-8")
    files = source_files()
    missing = [str(source) for source, _ in files if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"missing supplement input: {missing[0]}")

    with zipfile.ZipFile(DESTINATION, "w") as archive:
        add_bytes(archive, Path("README.md"), sanitize(readme).encode("utf-8"))
        for source, relative in files:
            add_file(archive, source, relative)

    print(f"Wrote {DESTINATION} ({DESTINATION.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
