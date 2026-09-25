"""Render the paper's mechanism diagnostics from compact, versioned inputs.

Run from the repository root with ``python experiments/exp4_mechanism_diagnostics.py``.
The figures are descriptive diagnostics of four ClimbMix checkpoints; they do
not estimate a causal mediation effect.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "exp4_mechanism"
DEFAULT_OUTPUT = ROOT / "outputs"
KEEP = (
    "100m/climbmix-10B-s1337",
    "13m/climbmix-10B-s1337",
    "13m/climbmix-10B-s42",
    "13m/climbmix-10B-s2026",
)
WIDTHS = {"13m": 256.0, "100m": 768.0}
GREEN = "#3AA278"
CORAL = "#D75C5D"
PERIWINKLE = "#6667AB"
CHARCOAL = "#303330"
EDGE = "#777A73"
GRID = "#E8E5DE"
PALE = "#F7F7F4"


def read_csv(data_dir: Path, name: str) -> list[dict[str, str]]:
    with (data_dir / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.labelpad": 7,
            "axes.titlesize": 10,
            "text.color": CHARCOAL,
            "axes.labelcolor": CHARCOAL,
            "axes.edgecolor": EDGE,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = output_dir / f"{name}.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)


def render_quality(rows: list[dict[str, str]], output_dir: Path) -> None:
    metrics = (
        ("ce", "Cross entropy (lower is better)", 1.0),
        ("top1", "Top-1 accuracy (%)", 100.0),
        ("p_correct", "Correct-token probability (%)", 100.0),
    )
    colors = {"1337": PERIWINKLE, "2026": CORAL, "42": GREEN}
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.0), sharex="col")
    for column, scale in enumerate(("13m", "100m")):
        selected = sorted({r["pair_key"] for r in rows if r["pair_key"].startswith(scale + "/")})
        for row_index, (metric, ylabel, factor) in enumerate(metrics):
            ax = axes[row_index, column]
            for pair in selected:
                same = sorted(
                    (r for r in rows if r["pair_key"] == pair),
                    key=lambda r: int(r["loops"]),
                )
                seed = pair.rsplit("-s", 1)[1]
                ax.plot(
                    [int(r["loops"]) for r in same],
                    [factor * float(r[metric]) for r in same],
                    marker="o",
                    color=colors[seed],
                    label=f"seed {seed}",
                )
            ax.set_ylabel(ylabel)
            ax.set_xticks([1, 2, 3, 4])
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            if row_index == 0:
                ax.set_title("13M ClimbMix" if scale == "13m" else "100M ClimbMix", pad=5)
            if row_index == 2:
                ax.set_xlabel("Native mixer calls per physical block")
        axes[0, column].legend(frameon=False, fontsize=8, handlelength=2.2, loc="best")
    fig.suptitle("Fixed-checkpoint MixerLoop quality", fontsize=13, fontweight="semibold", color=GREEN)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.08, top=0.91, wspace=0.28, hspace=0.30)
    save_figure(fig, output_dir, "exp4_quality_by_loops")


def render_paired_gain(rows: list[dict[str, str]], output_dir: Path) -> None:
    metrics = (
        ("ce", "Cross entropy (lower is better)", 1.0),
        ("top1", "Top-1 accuracy (%)", 100.0),
        ("p_correct", "Correct-token probability (%)", 100.0),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.6), sharey=True)
    labels = [f"{pair.split('/', 1)[0].upper()} | seed {pair.rsplit('-s', 1)[1]}" for pair in KEEP]
    for ax, (metric, title, factor) in zip(axes, metrics):
        values = {r["pair_key"]: r for r in rows if r["metric"] == metric}
        means = np.array([factor * float(values[p]["mean_delta"]) for p in KEEP])
        lows = np.array([factor * float(values[p]["ci95_low"]) for p in KEEP])
        highs = np.array([factor * float(values[p]["ci95_high"]) for p in KEEP])
        ax.errorbar(
            means,
            np.arange(len(KEEP)),
            xerr=[means - lows, highs - means],
            fmt="o",
            color=GREEN,
            markersize=4.5,
            capsize=2.5,
            linewidth=1.2,
        )
        ax.axvline(0, color=EDGE, linewidth=0.8)
        ax.set_title(title, fontsize=10, fontweight="semibold")
        ax.set_xlabel("Loop 4 − loop 1" + (" (pp)" if metric != "ce" else ""))
        ax.grid(axis="x", color=GRID, linewidth=0.6)
    axes[0].set_yticks(np.arange(len(KEEP)), labels, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Paired loop-count differences", fontsize=13, fontweight="semibold", color=GREEN)
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.20, top=0.82, wspace=0.38)
    save_figure(fig, output_dir, "exp4_paired_loop_gain")


def render_diagnostics(
    performance: list[dict[str, str]],
    rank_rows: list[dict[str, str]],
    ffn_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    performance_by_pair = {row["pair"]: row for row in performance}
    ordered = [performance_by_pair[pair] for pair in KEEP]
    labels = [f"{pair.split('/', 1)[0].upper()} | seed {pair.rsplit('-s', 1)[1]}" for pair in KEEP]
    rank_lookup = {
        (row["pair"], row["role"], row["position"]): float(row["delta_entropy_effective_rank"])
        for row in rank_rows
        if row["pair"] in KEEP
        and row["form"] == "centered"
        and row["split"] == "pooled_fit_and_validation"
        and row["operation"] == "mixerloop_loop4"
        and row["baseline"] == "mixerloop_loop1"
    }
    ffn_lookup = {
        (row["pair"], row["role"], row["metric"]): float(row["delta"])
        for row in ffn_rows
        if row["pair"] in KEEP
        and row["model"] == "mixerloop"
        and row["pass"] == "4"
        and row["position_name"] == "ffn_input"
        and row["split"] == "validation"
        and row["baseline_model"] == "mixerloop"
        and row["baseline_pass"] == "1"
    }
    rank_positions = ("attention_post", "ffn_input", "block_post")
    rank_matrix = np.array(
        [
            [
                100.0 * rank_lookup[(row["pair"], role, position)] / float(row["hidden_width"])
                for role in ("mid", "end")
                for position in rank_positions
            ]
            for row in ordered
        ]
    )
    ffn_metrics = ("top_10pct_energy", "mean_pair_cosine")
    ffn_matrix = np.array(
        [
            [100.0 * ffn_lookup[(row["pair"], role, metric)] for role in ("mid", "end") for metric in ffn_metrics]
            for row in ordered
        ]
    )
    core = np.array([float(row["core_mixer_minus_gdn_points"]) for row in ordered])
    cmap = LinearSegmentedColormap.from_list("mechanism_signed", [CORAL, PALE, GREEN], N=256)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.0, 5.5),
        gridspec_kw={"width_ratios": [0.9, 1.65, 1.2]},
    )
    core_ax, rank_ax, ffn_ax = axes
    core_colors = [GREEN if value > 0 else CORAL for value in core]
    y = np.arange(len(labels))
    core_ax.barh(y, core, color=core_colors, edgecolor=EDGE, linewidth=0.35)
    core_ax.axvline(0.0, color=EDGE, linewidth=0.8)
    core_ax.set_yticks(y, labels)
    core_ax.invert_yaxis()
    core_ax.set_xlabel("ΔCORE (MixerLoop − GDN, points)")
    core_ax.set_title("Paired CORE difference\nthree wins, one loss", fontsize=10, pad=5)
    core_ax.grid(axis="x", color=GRID, linewidth=0.6)
    span = max(float(np.max(np.abs(core))), 1.0)
    core_ax.set_xlim(-0.25 * span, 1.17 * span)
    for index, value in enumerate(core):
        core_ax.text(value + 0.035, index, f"{value:+.3f}", ha="left", va="center", fontsize=7.5)
    core_ax.legend(
        handles=[
            Patch(facecolor=GREEN, label="CORE win (3)"),
            Patch(facecolor=CORAL, label="CORE loss (1)"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=7,
        handlelength=1.0,
    )
    panels = [
        (
            rank_ax,
            rank_matrix,
            ["mid\nattention out", "mid\nFFN in", "mid\nblock out", "end\nattention out", "end\nFFN in", "end\nblock out"],
            "Centered effective-rank change\n(normalized by hidden width, %)",
        ),
        (
            ffn_ax,
            ffn_matrix,
            ["mid\nenergy", "mid\ncosine ×100", "end\nenergy", "end\ncosine ×100"],
            "FFN-input distribution change\n(top-10% energy / mean cosine ×100)",
        ),
    ]
    for ax, matrix, ticks, title in panels:
        limit = max(float(np.max(np.abs(matrix))), 1.0)
        image = ax.pcolormesh(
            np.arange(matrix.shape[1] + 1),
            np.arange(matrix.shape[0] + 1),
            matrix,
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            shading="flat",
        )
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_xticks(np.arange(len(ticks)) + 0.5, ticks, fontsize=8)
        ax.set_yticks(np.arange(len(labels)) + 0.5, ["" for _ in labels], fontsize=8)
        ax.set_xlim(0, matrix.shape[1])
        ax.set_ylim(matrix.shape[0], 0)
        ax.tick_params(length=0, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j + 0.5, i + 0.5, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=7, color=CHARCOAL)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
        colorbar.set_label("change", fontsize=8)
        colorbar.ax.tick_params(labelsize=7, length=2)
    fig.suptitle("Representation refinement and conditional FFN readout diagnostics", fontsize=13, fontweight="semibold", color=GREEN)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.85, bottom=0.20, wspace=0.35)
    save_figure(fig, output_dir, "exp4_layer_position_diagnostics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    configure_style()
    render_quality(read_csv(args.data_dir, "quality.csv"), args.output_dir)
    render_paired_gain(read_csv(args.data_dir, "comparisons.csv"), args.output_dir)
    render_diagnostics(
        read_csv(args.data_dir, "performance_link.csv"),
        read_csv(args.data_dir, "a4_rank_differences.csv"),
        read_csv(args.data_dir, "a5_ffn_input_differences.csv"),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
