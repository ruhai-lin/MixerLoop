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
GREEN = "#38543E"
BLUE = "#36576B"
GRAY = "#E0E4E7"


def save(fig, name):
    output = ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight",
                    facecolor="white")
        print(path)
    plt.close(fig)


def throughput(rows):
    fig, ax = plt.subplots(figsize=(6.7, 4.3))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.19, top=0.88)
    ax.set_title("More bandwidth exposes the cost of extra Mixer passes", loc="left")
    for model, color, marker in [("GDN", BLUE, "o"), ("MixerLoop", GREEN, "s")]:
        data = [r for r in rows if r["model"] == model]
        x = [float(r["lm_stream_GB_s"]) for r in data]
        y = [float(r["host_tok_s"]) for r in data]
        errors = [[v - float(r["host_min_tok_s"]) for r, v in zip(data, y)],
                  [float(r["host_max_tok_s"]) - v for r, v in zip(data, y)]]
        ax.errorbar(x, y, yerr=errors, color=color, marker=marker, markersize=7,
                    linewidth=1.8, capsize=3, label=f"{model}  T={'1' if model == 'GDN' else '4'}")
        for xx, yy in zip(x, y):
            offset = 12 if model == "GDN" else -20
            ax.annotate(f"{yy:.1f}", (xx, yy), xytext=(0, offset),
                        textcoords="offset points", ha="center", fontsize=9)
    ax.set(xlim=(1.85, 10), ylim=(100, 540), ylabel="Throughput (tokens/s)",
           xlabel="Measured parameter-stream bandwidth (GB/s)")
    ax.set_xticks(x, [f"{v:.2f}\n{hp} HP" for v, hp in zip(x, (1, 2, 4))])
    ax.grid(axis="y", color=GRAY, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.12, 0.015, "KV260 · 150 MHz · same compute datapath · median and observed range of 7 runs",
             fontsize=8.5, color="#555555")
    save(fig, "exp1_bandwidth")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.titlepad": 15,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.8, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    with (ROOT / "data/exp1_bandwidth.csv").open(newline="") as handle:
        throughput(list(csv.DictReader(handle)))
