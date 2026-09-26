"""Representation-rank changes: python3 experiments/exp5_representation_rank.py."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
ROLES = ("mid", "end")
POSITIONS = ("attention_post", "ffn_input", "block_post")


def read_csv(name):
    with (ROOT / "data" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def rank_changes(checkpoints, rows):
    lookup = {(r["pair"], r["role"], r["position"]): r for r in rows}
    expected = {(p["pair"], role, position) for p in checkpoints for role in ROLES for position in POSITIONS}
    assert set(lookup) == expected and len(rows) == len(expected)
    values = []
    for checkpoint in checkpoints:
        scale = checkpoint["pair"].split("/")[0]
        with (ROOT.parent / "configs" / f"mixerloop_{scale}.json").open() as stream:
            width = json.load(stream)["hidden_size"]
        assert width == int(checkpoint["hidden_width"])
        record = []
        for role in ROLES:
            for position in POSITIONS:
                row = lookup[checkpoint["pair"], role, position]
                assert (row["form"], row["split"], row["operation"], row["baseline"]) == (
                    "centered", "pooled_fit_and_validation", "mixerloop_loop4", "mixerloop_loop1")
                delta = float(row["delta_entropy_effective_rank"])
                np.testing.assert_allclose(delta, float(checkpoint[f"{role}_{position}_centered_rank_delta"]),
                                           rtol=0, atol=1e-12)
                record.append(100.0 * delta / width)
        values.append(record)
    matrix = np.array(values)
    assert np.isfinite(matrix).all()
    return matrix


def draw_rank(ax, checkpoints, matrix):
    # Light endpoints keep signed changes readable beneath dark cell labels.
    negative = 0.65 * np.array(to_rgb("#6667AB")) + 0.35
    positive = 0.65 * np.array(to_rgb("#D99532")) + 0.35
    cmap = LinearSegmentedColormap.from_list("signed_change", [negative, "#F7F7F4", positive])
    limit = float(np.ceil(np.max(np.abs(matrix))))
    mesh = ax.pcolormesh(matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                         edgecolors="white", linewidth=0.6, rasterized=False)
    labels = [f"{p['pair'].split('/')[0].upper()} · s{p['pair'].rsplit('-s', 1)[1]}" for p in checkpoints]
    ax.set(xticks=np.arange(6) + 0.5,
           xticklabels=["Mixer\nout", "FFN\nin", "Block\nout"] * 2,
           yticks=np.arange(len(labels)) + 0.5, yticklabels=labels,
           xlim=(0, 6), ylim=(len(labels), 0))
    ax.tick_params(length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(3, color="white", linewidth=2.5)
    for x, title in ((0.25, "Mid"), (0.75, "End")):
        ax.text(x, 1.05, title, transform=ax.transAxes, ha="center", fontsize=10)
    for i, j in np.ndindex(matrix.shape):
        ax.text(j + 0.5, i + 0.5, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    return mesh


def rank_colorbar(fig, mesh, bounds):
    limit = mesh.norm.vmax
    cax = fig.add_axes(bounds)
    colorbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[-limit, 0, limit])
    colorbar.set_label("Δeffective rank / hidden width (%)", fontsize=9, labelpad=4)
    colorbar.outline.set_visible(False)
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor("face")  # Avoid seams between vector colorbar cells.
    colorbar.ax.tick_params(length=2, labelsize=9)


def plot_rank(checkpoints, matrix):
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    fig.subplots_adjust(left=0.25, right=0.975, bottom=0.32, top=0.80)
    mesh = draw_rank(ax, checkpoints, matrix)
    rank_colorbar(fig, mesh, [0.22, 0.12, 0.56, 0.035])
    fig.suptitle("Representation Rank", x=0.5, y=0.96,
                 fontsize=13, fontweight="semibold", color="#3AA278")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = OUTPUT / f"exp5_representation_rank.{extension}"
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
    matrix = rank_changes(checkpoints, read_csv("exp5_representation_rank.csv"))
    print(f"Rank change / width (%): {matrix.min():+.3f} to {matrix.max():+.3f}")
    plot_rank(checkpoints, matrix)


if __name__ == "__main__":
    main()
