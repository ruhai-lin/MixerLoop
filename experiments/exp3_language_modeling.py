"""Training-loss figure: python3 experiments/exp3_language_modeling.py."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"

SERIES = (
    ("GDN", "#D75C5D", "-"),
    ("MixerLoop", "#3AA278", "--"),
    ("FullLoop", "#6667AB", "-."),
)
SMOOTHING_WINDOW = 17


def load_rows():
    with (ROOT / "data/exp3_loss.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def centered_mean(values, window):
    radius = window // 2
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - radius)
        stop = min(len(values), index + radius + 1)
        smoothed.append(sum(values[start:stop]) / (stop - start))
    return smoothed


def plot_loss(rows):
    x = [float(row["tokens_B_from_draft"]) for row in rows]
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    fig.subplots_adjust(left=0.18, right=0.94, bottom=0.20, top=0.85)
    fig.suptitle("Language Modeling", x=0.5, y=0.96,
                 fontsize=16, fontweight="semibold", color="#3AA278")

    for name, color, linestyle in SERIES:
        y = [float(row[name]) for row in rows]
        ax.plot(
            x,
            centered_mean(y, SMOOTHING_WINDOW),
            color=color,
            linestyle=linestyle,
            linewidth=1.1,
            label=name,
        )

    ax.set(
        xlim=(1, 10),
        ylim=(3.08, 3.66),
        xlabel="Processed Tokens (Billion)",
        ylabel="Logged training loss",
    )
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.grid(color="#E8E5DE", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=12,
              handlelength=2.2, borderaxespad=0.3)

    # Center the full panel, including the space occupied by axis labels.
    fig.canvas.draw()
    bounds = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    position = ax.get_position()
    ax.set_position([position.x0 + 0.5 - (bounds.x0 + bounds.x1) / 2,
                     position.y0, position.width, position.height])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = OUTPUT / f"exp3_language_modeling.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.labelpad": 7,
        "text.color": "#303330",
        "axes.labelcolor": "#303330",
        "axes.edgecolor": "#777A73",
        "xtick.color": "#303330",
        "ytick.color": "#303330",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
    })
    plot_loss(load_rows())
