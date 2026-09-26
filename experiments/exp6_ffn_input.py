"""Combined rank and FFN-input figure: python3 experiments/exp6_ffn_input.py."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
import numpy as np

from exp5_representation_rank import draw_rank, rank_changes, rank_colorbar


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
ROLES = ("mid", "end")
METRICS = (
    ("top_10pct_energy", "Top-10%\nenergy", "Δenergy (pp)", 100.0, 2),
    ("mean_pair_cosine", "Mean pair\ncosine", "Δmean cosine", 1.0, 3),
)


def read_csv(name):
    with (ROOT / "data" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def input_changes(checkpoints, rows):
    lookup = {(r["pair"], r["role"], r["metric"]): r for r in rows}
    expected = {(p["pair"], role, metric) for p in checkpoints for role in ROLES for metric, *_ in METRICS}
    assert set(lookup) == expected and len(rows) == len(expected)
    matrices = []
    for metric, _, _, factor, _ in METRICS:
        values = []
        for checkpoint in checkpoints:
            record = []
            for role in ROLES:
                row = lookup[checkpoint["pair"], role, metric]
                assert (row["model"], row["pass"], row["position_name"], row["split"],
                        row["baseline_model"], row["baseline_pass"]) == (
                    "mixerloop", "4", "ffn_input", "validation", "mixerloop", "1")
                delta = float(row["delta"])
                np.testing.assert_allclose(delta, float(checkpoint[f"{role}_ffn_input_{metric}_delta"]),
                                           rtol=0, atol=1e-12)
                record.append(factor * delta)
            values.append(record)
        matrix = np.array(values)
        assert np.isfinite(matrix).all()
        matrices.append(matrix)
    return matrices


def plot_inputs(checkpoints, rank_matrix, matrices):
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.6), sharey=True,
                             gridspec_kw={"width_ratios": [6, 2, 2]})
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.32, top=0.78, wspace=0.20)
    mesh = draw_rank(axes[0], checkpoints, rank_matrix)
    rank_box = axes[0].get_position()
    rank_colorbar(fig, mesh, [rank_box.x0 + 0.025, 0.12, rank_box.width - 0.05, 0.035])
    fig.text((rank_box.x0 + rank_box.x1) / 2, 0.945, "Representation Rank",
             ha="center", va="top", fontsize=13, fontweight="semibold", color="#3AA278")
    for ax, label in ((axes[0], "(a)"), (axes[1], "(b)")):
        fig.text(ax.get_position().x0 - 0.015, 0.945, label,
                 ha="right", va="top", fontsize=9, fontweight="semibold")
    negative = 0.65 * np.array(to_rgb("#6667AB")) + 0.35
    positive = 0.65 * np.array(to_rgb("#D99532")) + 0.35
    cmap = LinearSegmentedColormap.from_list("signed_change", [negative, "#F7F7F4", positive])
    for ax, matrix, (_, title, unit, factor, decimals) in zip(axes[1:], matrices, METRICS):
        # Each statistic keeps its own units and a symmetric zero-centered scale.
        precision = 1 if factor == 100 else 100
        limit = float(np.ceil(np.max(np.abs(matrix)) * precision) / precision)
        mesh = ax.pcolormesh(matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                             edgecolors="white", linewidth=0.6, rasterized=False)
        ax.set(xticks=[0.5, 1.5], xticklabels=["Mid", "End"],
               xlim=(0, 2), ylim=(len(checkpoints), 0))
        ax.tick_params(length=0, pad=5, labelleft=False)
        position = ax.get_position()
        fig.text((position.x0 + position.x1) / 2, 0.945, title,
                 ha="center", va="top", fontsize=13, fontweight="semibold", color="#3AA278")
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i, j in np.ndindex(matrix.shape):
            ax.text(j + 0.5, i + 0.5, f"{matrix[i, j]:+.{decimals}f}",
                    ha="center", va="center", fontsize=8)
        cax = fig.add_axes([position.x0 + 0.015, 0.12, position.width - 0.03, 0.035])
        colorbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[-limit, 0, limit])
        colorbar.set_label(unit, fontsize=9, labelpad=4)
        colorbar.outline.set_visible(False)
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor("face")  # Avoid seams between vector colorbar cells.
        colorbar.ax.tick_params(length=2, labelsize=9)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = OUTPUT / f"exp5_6_rank_ffn_input.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 10, "axes.labelpad": 7,
        "text.color": "#303330", "axes.labelcolor": "#303330",
        "axes.edgecolor": "#777A73", "xtick.color": "#303330", "ytick.color": "#303330",
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "savefig.bbox": None,
    })
    checkpoints = read_csv("mechanism_checkpoints.csv")
    rank_matrix = rank_changes(checkpoints, read_csv("exp5_representation_rank.csv"))
    matrices = input_changes(checkpoints, read_csv("exp6_ffn_input.csv"))
    for (_, _, unit, _, _), matrix in zip(METRICS, matrices):
        print(f"{unit}: {matrix.min():+.4f} to {matrix.max():+.4f}")
    plot_inputs(checkpoints, rank_matrix, matrices)


if __name__ == "__main__":
    main()
