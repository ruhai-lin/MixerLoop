"""Fixed-checkpoint loop-count ablation: python3 experiments/exp4_loop_quality.py."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
GREEN = "#3AA278"
SCALES = {
    "13m": (GREEN, "o", "-", "13M (mean + seed range)"),
    "100m": ("#6667AB", "s", "--", "100M (seed 1337)"),
}
METRICS = (
    ("ce", "Cross entropy", "Loss", 1.0),
    ("top1", "Top-1 accuracy", "Accuracy (%)", 100.0),
    ("p_correct", "Token probability", "Probability (%)", 100.0),
)


def read_csv(name):
    with (ROOT / "data" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(quality, comparisons):
    observations = {(r["pair_key"], int(r["loops"])): r for r in quality}
    assert len(observations) == len(quality)
    pairs = {r["pair_key"] for r in quality}
    assert set(observations) == {(p, t) for p in pairs for t in range(1, 5)}
    expected = {(p, metric) for p in pairs for metric, *_ in METRICS}
    assert {(r["pair_key"], r["metric"]) for r in comparisons} == expected
    assert len(comparisons) == len(expected)
    summary = []
    for row in comparisons:
        pair, metric = row["pair_key"], row["metric"]
        first = float(observations[pair, 1][metric])
        last = float(observations[pair, 4][metric])
        mean, low, high = (float(row[k]) for k in ("mean_delta", "ci95_low", "ci95_high"))
        np.testing.assert_allclose(last - first, mean, rtol=0, atol=1e-12)
        assert low <= mean <= high
        # Express every summary as improvement, keeping CI endpoints ordered.
        factor = -1.0 if metric == "ce" else 100.0
        low, high = sorted((factor * low, factor * high))
        summary.append({
            "checkpoint": pair,
            "metric": {"ce": "ce_reduction", "top1": "top1_gain", "p_correct": "correct_token_probability_gain"}[metric],
            "unit": "loss" if metric == "ce" else "percentage_points",
            "improvement": factor * mean,
            "ci95_low": low,
            "ci95_high": high,
        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "exp4_loop_quality_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(f"{row['checkpoint']:30} {row['metric']:31} "
              f"{row['improvement']:.3f} [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]")


def plot_quality(rows):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharex=True)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.185, top=0.69, wspace=0.48)
    loops = np.arange(1, 5)
    for ax, (metric, title, ylabel, factor) in zip(axes, METRICS):
        for scale, (color, marker, linestyle, label) in SCALES.items():
            selected = [r for r in rows if r["pair_key"].startswith(scale + "/")]
            pairs = sorted({r["pair_key"] for r in selected})
            lookup = {(r["pair_key"], int(r["loops"])): factor * float(r[metric]) for r in selected}
            values = np.array([[lookup[pair, t] for t in loops] for pair in pairs])
            if len(pairs) > 1:
                # The envelope is the observed seed range at each loop count.
                ax.fill_between(loops, values.min(axis=0), values.max(axis=0),
                                color=color, alpha=0.16, linewidth=0)
            ax.plot(loops, values.mean(axis=0), color=color, marker=marker,
                    linestyle=linestyle, label=label)
        values = np.array([factor * float(r[metric]) for r in rows])
        padding = 0.08 * np.ptp(values)
        ax.set(ylim=(values.min() - padding, values.max() + padding),
               xlim=(0.8, 4.2), xticks=loops, ylabel=ylabel, xlabel="Mixer loops")
        ax.set_box_aspect(1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="y", color="#E8E5DE", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=10, pad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.885),
               ncol=2, frameon=False, fontsize=9, handlelength=2.2)
    for extension in ("pdf", "png"):
        path = OUTPUT / f"exp4_loop_quality.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 10, "axes.labelpad": 7,
        "text.color": "#303330", "axes.labelcolor": "#303330",
        "axes.edgecolor": "#777A73", "xtick.color": "#303330", "ytick.color": "#303330",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "lines.linewidth": 1.6, "lines.markersize": 4.5, "pdf.fonttype": 42,
        "savefig.bbox": None,
    })
    rows = read_csv("exp4_loop_quality.csv")
    summarize(rows, read_csv("exp4_loop_differences.csv"))
    plot_quality(rows)


if __name__ == "__main__":
    main()
