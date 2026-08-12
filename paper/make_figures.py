#!/usr/bin/env python3
"""Build the vector figures used by the AAAI manuscript."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURES = Path(__file__).resolve().parent / "Figures"


def save_outlined_pdf(figure, destination: Path) -> None:
    """Save figure text as vector outlines, avoiding CID fonts in submission PDFs."""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as raw:
        figure.savefig(raw.name)
        subprocess.run(
            [
                "gs",
                "-q",
                "-dBATCH",
                "-dNOPAUSE",
                "-dSAFER",
                "-dCompatibilityLevel=1.5",
                "-dNoOutputFonts",
                "-sDEVICE=pdfwrite",
                f"-sOutputFile={destination}",
                raw.name,
            ],
            check=True,
        )


def rounded_box(
    axis,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    face: str,
    edge: str,
    fontsize: float = 9.0,
    weight: str = "normal",
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.5,
        facecolor=face,
        edgecolor=edge,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
    )
    return patch


def arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#4b4b4b",
    connection: str = "arc3",
    width: float = 1.4,
):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=width,
            color=color,
            connectionstyle=connection,
        )
    )


def architecture_figure() -> None:
    blue, blue_fill = "#0072B2", "#DCEFFD"
    green, green_fill = "#00865A", "#DFF4ED"
    orange = "#C95000"
    grey, grey_fill = "#737983", "#F6F7F8"

    figure, axis = plt.subplots(figsize=(7.1, 2.46))
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 4.15)
    axis.axis("off")

    axis.text(
        2.75,
        3.88,
        "FullLoop: full-stack recurrence",
        ha="center",
        fontsize=11,
        weight="bold",
    )
    rounded_box(axis, 0.12, 2.65, 0.62, 0.64, r"$h^0$", face=grey_fill, edge=grey)
    rounded_box(
        axis,
        1.10,
        2.55,
        1.26,
        0.84,
        "$B_1$\nGDN + FFN",
        face=blue_fill,
        edge=blue,
        weight="bold",
    )
    axis.text(2.70, 2.96, r"$\cdots$", fontsize=13, ha="center", va="center")
    rounded_box(
        axis,
        3.05,
        2.55,
        1.26,
        0.84,
        "$B_L$\nGDN + FFN",
        face=green_fill,
        edge=green,
        weight="bold",
    )
    rounded_box(axis, 4.72, 2.65, 0.62, 0.64, r"$h^T$", face=grey_fill, edge=grey)
    arrow(axis, (0.74, 2.97), (1.10, 2.97))
    arrow(axis, (2.36, 2.97), (2.55, 2.97))
    arrow(axis, (2.87, 2.97), (3.05, 2.97))
    arrow(axis, (4.31, 2.97), (4.72, 2.97))
    arrow(
        axis,
        (4.20, 2.47),
        (1.20, 2.47),
        color=orange,
        connection="arc3,rad=-0.34",
        width=1.9,
    )
    axis.text(
        2.70,
        2.32,
        r"repeat $(B_L\circ\cdots\circ B_1)^T$",
        color=orange,
        ha="center",
        fontsize=9,
    )

    axis.text(
        9.05,
        3.88,
        "MixerLoop: mixer recurrence",
        ha="center",
        fontsize=11,
        weight="bold",
    )
    rounded_box(axis, 6.38, 2.65, 0.62, 0.64, r"$h_i$", face=grey_fill, edge=grey)
    rounded_box(
        axis,
        7.35,
        2.55,
        1.36,
        0.84,
        "GDN\nmixer",
        face=blue_fill,
        edge=blue,
        weight="bold",
    )
    rounded_box(
        axis,
        9.45,
        2.55,
        1.36,
        0.84,
        "dense\nFFN",
        face=green_fill,
        edge=green,
        weight="bold",
    )
    rounded_box(axis, 11.18, 2.65, 0.68, 0.64, r"$h_{i+1}$", face=grey_fill, edge=grey)
    arrow(axis, (7.00, 2.97), (7.35, 2.97))
    arrow(axis, (8.71, 2.97), (9.45, 2.97))
    arrow(axis, (10.81, 2.97), (11.18, 2.97))
    arrow(
        axis,
        (8.57, 2.47),
        (7.49, 2.47),
        color=blue,
        connection="arc3,rad=-0.42",
        width=1.9,
    )
    axis.text(
        8.03,
        2.08,
        r"mixer $T\times$",
        color=blue,
        ha="center",
        fontsize=9,
        weight="bold",
    )
    axis.text(
        10.13,
        2.08,
        r"FFN $1\times$",
        color=green,
        ha="center",
        fontsize=9,
        weight="bold",
    )

    axis.plot([0.1, 11.9], [1.82, 1.82], color="#D0D3D8", linewidth=1.0)
    labels = [
        ("1  COMPOSE", "repeat one\nstate update"),
        ("2  EXPOSE", "new contextual\ninfluence"),
        ("3  OBSERVE", "prediction\nchanges"),
        ("4  STOP", "marginal effect\nsaturates"),
        ("5  ALLOCATE", "observable effect\nper unit cost"),
    ]
    widths = [1.8, 2.18, 1.9, 1.9, 2.15]
    x = 0.12
    patches = []
    for (title, body), width in zip(labels, widths):
        patches.append(
            rounded_box(
                axis,
                x,
                0.20,
                width,
                1.28,
                body,
                face=grey_fill,
                edge=grey,
            )
        )
        axis.text(x + 0.12, 1.30, title, fontsize=9, weight="bold", va="center")
        x += width + 0.28
    for left, right in zip(patches, patches[1:]):
        arrow(
            axis,
            (left.get_x() + left.get_width() + 0.04, 0.84),
            (right.get_x() - 0.04, 0.84),
            width=1.1,
        )

    save_outlined_pdf(figure, FIGURES / "architecture.pdf")
    plt.close(figure)


def readout_itr_figure() -> None:
    paths = (
        ROOT / "outputs" / "itr-readout-15m" / "itr_eval.json",
        ROOT / "outputs" / "itr-readout-110m" / "itr_eval.json",
    )
    results = [json.loads(path.read_text()) for path in paths]
    colors = {"NoLoop": "#666666", "MixerLoop": "#0072B2", "FullLoop": "#C95000"}
    markers = {"NoLoop": "s", "MixerLoop": "o", "FullLoop": "^"}

    figure, axes = plt.subplots(2, 3, figsize=(7.1, 4.15), constrained_layout=True)
    keys = ("itr_by_depth", "step_hellinger_squared", "target_logprob_gain")
    titles = (
        "Cumulative directions",
        "Finite prediction change",
        "True-token contribution",
    )
    ylabels = (
        "Readout ITR",
        r"Squared Hellinger $H^2$",
        r"True-token $\Delta\log p$",
    )
    panel = 0
    for row, (result, scale) in enumerate(zip(results, ("15M", "110M"))):
        for column, (key, title, ylabel) in enumerate(zip(keys, titles, ylabels)):
            axis = axes[row, column]
            for model in result["models"]:
                values = model["aggregate"][key]
                passes = np.arange(1, len(values) + 1)
                axis.plot(
                    passes,
                    values,
                    color=colors[model["label"]],
                    marker=markers[model["label"]],
                    linewidth=1.7,
                    markersize=4.5,
                    label=model["label"],
                )
            if key == "target_logprob_gain":
                axis.axhline(0, color="#888888", linewidth=0.8)
            axis.set_xticks((1, 2, 3, 4))
            axis.set_xlabel("Contextual mixer pass", fontsize=9)
            axis.set_ylabel(ylabel, fontsize=9)
            axis.tick_params(labelsize=9)
            if row == 0:
                axis.set_title(title, fontsize=10, weight="bold")
            axis.text(
                0.98 if column == 1 else 0.02,
                0.96,
                f"({chr(ord('a') + panel)}) {scale}",
                transform=axis.transAxes,
                ha="right" if column == 1 else "left",
                va="top",
                fontsize=9,
                weight="bold",
            )
            panel += 1
    axes[0, 2].legend(frameon=False, fontsize=9, loc="upper right")
    save_outlined_pdf(figure, FIGURES / "readout_itr.pdf")
    plt.close(figure)


def layerwise_mitr_figure() -> None:
    paths = (
        ROOT / "outputs" / "itr-readout-15m" / "itr_eval.json",
        ROOT / "outputs" / "itr-readout-110m" / "itr_eval.json",
    )
    results = [json.loads(path.read_text()) for path in paths]
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 3.25), constrained_layout=True)
    last_image = None
    for row, (result, scale) in enumerate(zip(results, ("15M", "110M"))):
        models = {model["label"]: model for model in result["models"]}
        for column, label in enumerate(("MixerLoop", "FullLoop")):
            axis = axes[row, column]
            values = np.asarray(
                [
                    layer["marginal_itr"][1:]
                    for layer in models[label]["layer_results"]
                ]
            )
            last_image = axis.imshow(
                values,
                origin="lower",
                aspect="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            axis.set_xticks((0, 1, 2), labels=(2, 3, 4))
            layer_count = values.shape[0]
            tick_step = 1 if layer_count <= 6 else 2
            layer_ticks = np.arange(0, layer_count, tick_step)
            axis.set_yticks(layer_ticks, labels=layer_ticks)
            axis.set_xlabel("Contextual mixer pass", fontsize=9)
            axis.set_ylabel("Physical layer", fontsize=9)
            axis.tick_params(labelsize=9)
            if row == 0:
                axis.set_title(label, fontsize=10, weight="bold")
            axis.text(
                0.02,
                0.96,
                f"({chr(ord('a') + 2 * row + column)}) {scale}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                weight="bold",
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
            )
    colorbar = figure.colorbar(last_image, ax=axes, pad=0.02, shrink=0.92)
    colorbar.set_label("Marginal ITR", fontsize=9)
    colorbar.ax.tick_params(labelsize=9)
    save_outlined_pdf(figure, FIGURES / "layerwise_mitr.pdf")
    plt.close(figure)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    architecture_figure()
    readout_itr_figure()
    layerwise_mitr_figure()


if __name__ == "__main__":
    main()
