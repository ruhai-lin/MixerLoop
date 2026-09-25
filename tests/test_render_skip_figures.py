import csv
import json
from pathlib import Path

from eval.render_skip_figures import FORMAL_CHECKPOINTS, render_skip_figures


def test_render_skip_figures_writes_labeled_heatmap_and_scatter(tmp_path: Path):
    source = tmp_path / "skip_ablation_detail.csv"
    rows = []
    for layer in (0, 1):
        for pass_id in (1, 2):
            rows.append(
                {
                    "checkpoint": "mixerloop-15m",
                    "scale": "15m",
                    "condition": "mixerloop",
                    "layer": layer,
                    "pass": pass_id,
                    "normal_loss": "1.0",
                    "skip_loss": str(1.1 + layer + pass_id / 10),
                    "delta_loss": str(layer + pass_id / 10),
                    "mITR": str(0.2 * pass_id + 0.1 * layer),
                    "core_score_if_run": "",
                }
            )
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output_dir = tmp_path / "figures"
    manifest = render_skip_figures(source, output_dir)

    names = {Path(item["path"]).name for item in manifest["generated"]}
    assert "delta_loss_heatmap_mixerloop-15m.png" in names
    assert "mitr_delta_loss_scatter_mixerloop-15m.png" in names
    assert all((output_dir / name).exists() for name in names)
    assert manifest["generated"]
    assert manifest["output_root_bytes"] == sum(
        path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()
    )
    saved = json.loads((output_dir / "figure_manifest.json").read_text())
    assert saved["generated"] == manifest["generated"]


def test_render_skip_figures_enforces_formal_six_checkpoint_contract(tmp_path: Path):
    source = tmp_path / "formal_detail.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["checkpoint", "scale", "condition", "layer", "pass", "delta_loss", "mITR"],
        )
        writer.writeheader()
        for checkpoint in FORMAL_CHECKPOINTS:
            condition, scale = checkpoint.rsplit("-", 1)
            writer.writerow(
                {
                    "checkpoint": checkpoint,
                    "scale": scale,
                    "condition": condition,
                    "layer": 0,
                    "pass": 1,
                    "delta_loss": "0.1",
                    "mITR": "0.2",
                }
            )

    manifest = render_skip_figures(
        source,
        tmp_path / "formal_figures",
        expected_checkpoints=FORMAL_CHECKPOINTS,
    )

    assert manifest["formal_six_checkpoints"] is True
    assert len(manifest["generated"]) == 12
    assert sum(item["kind"] == "delta_loss_heatmap" for item in manifest["generated"]) == 6
    assert sum(item["kind"] == "mitr_delta_loss_scatter" for item in manifest["generated"]) == 6
