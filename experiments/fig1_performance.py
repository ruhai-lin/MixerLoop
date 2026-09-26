"""Draw the performance panel; combine it with the architecture in LaTeX.

python3 experiments/fig1_performance.py
Requires matplotlib.
"""

from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
COLORS = {"MixerLoop": "#3AA278", "GDN 400M": "#E79596",
          "GDN 1.3B": "#D75C5D", "FullLoop": "#6667AB"}
TASKS = ("ARC-E", "ARC-C", "HellaSwag", "PIQA", "WinoGrande", "BoolQ")
# Values supplied by the author on 2026-09-25. GDN provenance is pending.
# Score order follows TASKS.
RESULTS = (
    {"model": "GDN", "scale": "400M", "tokens": "15B",
     "scores": (59.80, 28.58, 40.46, 65.94, 51.46, 60.03), "average": 51.05},
    {"model": "FullLoop", "scale": "450M", "tokens": "100B",
     "scores": (66.43, 33.89, 52.62, 70.27, 61.48, 54.13), "average": 56.47},
    {"model": "MixerLoop", "scale": "450M", "tokens": "100B",
     "scores": (66.37, 34.04, 50.03, 71.06, 54.54, 57.55), "average": 55.60},
    {"model": "GDN", "scale": "1.3B", "tokens": "100B",
     "scores": (71.21, 38.39, 55.76, 72.25, 57.45, 60.24), "average": 59.22},
)


def plot_performance():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": "#303330",
        "axes.labelcolor": "#303330",
        "axes.edgecolor": "#777A73",
        "xtick.color": "#303330",
        "ytick.color": "#303330",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.28, top=0.98)
    width = 0.18
    score_labels = []
    for index, row in enumerate(RESULTS):
        assert len(row["scores"]) == len(TASKS)
        score = mean(row["scores"])
        assert abs(score - row["average"]) < 0.00501
        label = f"{row['model']} {row['scale']}"
        color = COLORS[label if row["model"] == "GDN" else row["model"]]
        positions = [task + (index - 1.5) * width for task in range(len(TASKS))]
        ax.bar(positions, row["scores"], width=width, color=color,
               linewidth=0, label=label, zorder=3)
        for x, value in zip(positions, row["scores"]):
            score_labels.append((x, value))
        print(f"{row['model']:10s} {row['scale']:>4s} / {row['tokens']:>4s}: {score:.4f}")

    ax.set_xticks(range(len(TASKS)),
                  ("ARC-E", "ARC-C", "Hella.", "PIQA", "WinoG.", "BoolQ"), fontsize=7.5)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.set(xlim=(-0.6, 5.6), ylim=(0, 88), ylabel="Score (%)")
    ax.set_yticks((0, 20, 40, 60, 80))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
              frameon=False, fontsize=8, handlelength=1.1,
              handletextpad=0.5, columnspacing=1.0)
    ax.yaxis.label.set_size(10)
    ax.grid(axis="y", color="#E8E5DE", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for x, value in score_labels:
        ax.text(x, value + 1.2, f"{value:.2f}", ha="center", va="bottom",
                rotation=90, fontsize=5.5, color="#303330", zorder=5)

    output = ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = output / f"f1_performance.{extension}"
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    plot_performance()
