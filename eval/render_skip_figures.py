"""Render Spec04 delta-loss heatmaps and labeled mITR/delta-loss scatters."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


REQUIRED_COLUMNS = {
    "checkpoint",
    "scale",
    "condition",
    "layer",
    "pass",
    "delta_loss",
    "mITR",
}
OUTPUT_ROOT_CAP_BYTES = 99_000_000_000
FORMAL_CHECKPOINTS = tuple(
    f"{condition}-{scale}"
    for scale in ("15m", "42m", "110m")
    for condition in ("mixerloop", "fullloop")
)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_skip_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"skip table missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("skip table has no rows")
    for row in rows:
        if not _finite(row["delta_loss"]):
            raise ValueError("skip table contains a non-finite delta_loss")
    return rows


def _color(value: float, low: float, high: float) -> tuple[int, int, int]:
    if high <= low:
        fraction = 0.5
    else:
        fraction = (value - low) / (high - low)
    fraction = max(0.0, min(1.0, fraction))
    return int(255 * fraction), int(80 + 130 * (1.0 - abs(2 * fraction - 1.0))), int(255 * (1.0 - fraction))


def _heatmap(rows: list[dict[str, str]], title: str) -> Image.Image:
    layers = sorted({int(row["layer"]) for row in rows})
    passes = sorted({int(row["pass"]) for row in rows})
    values = {(int(row["layer"]), int(row["pass"])): float(row["delta_loss"]) for row in rows}
    if any((layer, pass_id) not in values for layer in layers for pass_id in passes):
        raise ValueError("delta-loss heatmap requires every layer/pass cell")
    low, high = min(values.values()), max(values.values())
    cell_w, cell_h = 120, 36
    left, top, right, bottom = 100, 70, 40, 70
    image = Image.new("RGB", (left + cell_w * len(passes) + right, top + cell_h * len(layers) + bottom), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((left, 20), title, fill="black", font=font)
    draw.text((10, top - 20), "layer", fill="black", font=font)
    for column, pass_id in enumerate(passes):
        draw.text((left + column * cell_w + 40, top - 20), f"pass {pass_id}", fill="black", font=font)
    for row_index, layer in enumerate(layers):
        y = top + row_index * cell_h
        draw.text((35, y + 12), str(layer), fill="black", font=font)
        for column, pass_id in enumerate(passes):
            x = left + column * cell_w
            value = values[(layer, pass_id)]
            draw.rectangle((x, y, x + cell_w - 2, y + cell_h - 2), fill=_color(value, low, high), outline="black")
            draw.text((x + 8, y + 12), f"{value:.4f}", fill="black", font=font)
    draw.text((left, top + cell_h * len(layers) + 20), f"delta_loss ({low:.4f} to {high:.4f})", fill="black", font=font)
    return image


def _scatter(rows: Iterable[dict[str, str]], title: str) -> Image.Image:
    points = [
        (float(row["mITR"]), float(row["delta_loss"]), int(row["layer"]), int(row["pass"]))
        for row in rows
        if _finite(row.get("mITR"))
    ]
    if not points:
        raise ValueError("mITR/delta-loss scatter requires at least one finite mITR row")
    width, height = 1400, max(760, 120 + 18 * len(points))
    left, top, right, bottom = 90, 70, 360, 100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((left, 20), title, fill="black", font=font)
    plot_right = width - right
    draw.line((left, height - bottom, plot_right, height - bottom), fill="black", width=2)
    draw.line((left, height - bottom, left, top), fill="black", width=2)
    x_low, x_high = min(point[0] for point in points), max(point[0] for point in points)
    y_low, y_high = min(point[1] for point in points), max(point[1] for point in points)
    x_span = max(x_high - x_low, 1e-12)
    y_span = max(y_high - y_low, 1e-12)
    for x_value, y_value, layer, pass_id in points:
        x = left + (x_value - x_low) / x_span * (plot_right - left)
        y = height - bottom - (y_value - y_low) / y_span * (height - top - bottom)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="red", outline="black")
    label_x = plot_right + 24
    draw.text((label_x, top - 22), "point labels", fill="black", font=font)
    for index, (x_value, y_value, layer, pass_id) in enumerate(sorted(points, key=lambda point: (point[1], point[0], point[2], point[3]))):
        x = left + (x_value - x_low) / x_span * (plot_right - left)
        y = height - bottom - (y_value - y_low) / y_span * (height - top - bottom)
        label_y = top + 18 * index
        draw.line((x, y, label_x - 4, label_y + 5), fill="gray", width=1)
        draw.text((label_x, label_y), f"L{layer}P{pass_id}", fill="black", font=font)
    draw.text((plot_right // 2 - 20, height - 35), "mITR", fill="black", font=font)
    draw.text((8, top - 10), "delta_loss", fill="black", font=font)
    return image


def _file_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def render_skip_figures(
    skip_csv: Path,
    output_dir: Path,
    *,
    output_root_cap_bytes: int = OUTPUT_ROOT_CAP_BYTES,
    expected_checkpoints: tuple[str, ...] | None = None,
) -> dict[str, object]:
    rows = load_skip_rows(skip_csv)
    output_root = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(str(row["checkpoint"]), []).append(row)
    if expected_checkpoints is not None:
        expected = set(expected_checkpoints)
        found = set(groups)
        if found != expected:
            raise ValueError(f"formal figure checkpoint scope mismatch: missing={sorted(expected - found)} extra={sorted(found - expected)}")
    generated: list[dict[str, object]] = []
    for checkpoint, group in sorted(groups.items()):
        safe_name = checkpoint.replace("/", "_")
        heatmap_path = output_dir / f"delta_loss_heatmap_{safe_name}.png"
        scatter_path = output_dir / f"mitr_delta_loss_scatter_{safe_name}.png"
        _heatmap(group, f"delta_loss: {checkpoint}").save(heatmap_path, format="PNG")
        _scatter(group, f"mITR vs delta_loss: {checkpoint}").save(scatter_path, format="PNG")
        generated.extend(
            [
                {"path": str(heatmap_path), "checkpoint": checkpoint, "kind": "delta_loss_heatmap", "labeled_axes": True},
                {"path": str(scatter_path), "checkpoint": checkpoint, "kind": "mitr_delta_loss_scatter", "labeled_points": True},
            ]
        )
    if expected_checkpoints is not None:
        if len(generated) != 12:
            raise ValueError(f"formal figure contract requires 12 images, found {len(generated)}")
        if sum(item["kind"] == "delta_loss_heatmap" for item in generated) != 6:
            raise ValueError("formal figure contract requires six delta-loss heatmaps")
        if sum(item["kind"] == "mitr_delta_loss_scatter" for item in generated) != 6:
            raise ValueError("formal figure contract requires six mITR/delta-loss scatters")
    manifest = {
        "source_csv": str(skip_csv),
        "source_sha256": hashlib.sha256(skip_csv.read_bytes()).hexdigest(),
        "renderer": "Pillow",
        "formal_six_checkpoints": expected_checkpoints is not None,
        "generated": generated,
        "output_root": str(output_root),
        "output_root_bytes": 0,
        "output_root_cap_bytes": int(output_root_cap_bytes),
    }
    manifest_path = output_dir / "figure_manifest.json"
    for _ in range(16):
        before = _file_bytes(output_root)
        if before > int(output_root_cap_bytes):
            raise RuntimeError(f"figure output exceeds cap: {before} > {output_root_cap_bytes}")
        manifest["output_root_bytes"] = before
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        after = _file_bytes(output_root)
        if after == before:
            break
    else:
        raise RuntimeError("figure manifest byte count did not stabilize")
    final_bytes = _file_bytes(output_root)
    manifest["output_root_bytes"] = final_bytes
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final_bytes = _file_bytes(output_root)
    if final_bytes > int(output_root_cap_bytes):
        raise RuntimeError(f"figure output exceeds cap: {final_bytes} > {output_root_cap_bytes}")
    manifest["output_root_bytes"] = final_bytes
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formal-six-checkpoints", action="store_true")
    args = parser.parse_args()
    print(json.dumps(render_skip_figures(
        args.skip_csv,
        args.output_dir,
        expected_checkpoints=FORMAL_CHECKPOINTS if args.formal_six_checkpoints else None,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
