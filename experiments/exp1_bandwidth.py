"""Measured throughput figure: python3 experiments/exp1_bandwidth.py.

Reads data/exp1_bandwidth.csv, the retained measurement summary.
Writes PDF and PNG figures under outputs/. Requires matplotlib.
Architecture analysis lives in exp2_memory_wall.py.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
GREEN = "#3AA278"
CORAL = "#D75C5D"
GRAY = "#E8E5DE"


def save(fig, name):
    # Center the full panel, including the space occupied by axis labels.
    fig.canvas.draw()
    ax = fig.axes[0]
    bounds = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    position = ax.get_position()
    ax.set_position([position.x0 + 0.5 - (bounds.x0 + bounds.x1) / 2,
                     position.y0, position.width, position.height])
    output = ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)


def throughput(rows):
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    fig.subplots_adjust(left=0.18, right=0.94, bottom=0.20, top=0.85)
    fig.suptitle("Bandwidth Scaling", x=0.5, y=0.96,
                 fontsize=13, fontweight="semibold", color=GREEN)
    for model, color, marker in [("GDN", CORAL, "o"), ("MixerLoop", GREEN, "s")]:
        data = [r for r in rows if r["model"] == model]
        x = [float(r["port_ceiling_GB_s"]) for r in data]
        y = [float(r["host_tok_s"]) for r in data]
        errors = [[v - float(r["host_min_tok_s"]) for r, v in zip(data, y)],
                  [float(r["host_max_tok_s"]) - v for r, v in zip(data, y)]]
        ax.errorbar(x, y, yerr=errors, color=color, marker=marker, markersize=4.5,
                    linewidth=1.6, capsize=2, label=f"{model}  T={'1' if model == 'GDN' else '4'}")
        for xx, yy in zip(x, y):
            offset = 9 if model == "GDN" else -15
            ax.annotate(f"{yy:.1f}", (xx, yy), xytext=(0, offset),
                        textcoords="offset points", ha="center", fontsize=8)
    ax.set(xlim=(1.65, 10.5), ylim=(90, 560), ylabel="Throughput (tokens/s)",
           xlabel="Bandwidth (GB/s)")
    ax.set_xticks(x, [f"{v:.1f}\n{hp} HP" for v, hp in zip(x, (1, 2, 4))])
    ax.grid(axis="y", color=GRAY, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=9,
              handlelength=2.2, borderaxespad=0.3)
    save(fig, "exp1_bandwidth")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 10, "axes.labelpad": 7,
                         "text.color": "#303330", "axes.labelcolor": "#303330",
                         "axes.edgecolor": "#777A73", "xtick.color": "#303330",
                         "ytick.color": "#303330",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.8, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    with (ROOT / "data/exp1_bandwidth.csv").open(newline="") as handle:
        throughput(list(csv.DictReader(handle)))
