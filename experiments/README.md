# Paper experiments

Reproducible analyses and figures for the paper. Contributors and agents should follow this guide when adding experiments. Experiments 1 and 3 are the layout references for new figures.

## Experiments

| Experiment | Question | Script | Analysis |
|---|---|---|---|
| 1: Bandwidth | When does extra recurrent compute become visible in decode throughput? | [exp1_bandwidth.py](exp1_bandwidth.py) | [exp1_bandwidth.md](exp1_bandwidth.md) |
| 2: Memory wall | How do active FFN traffic and mixer compute differ across architectures? | [exp2_memory_wall.py](exp2_memory_wall.py) | [exp2_memory_wall.md](exp2_memory_wall.md) |
| 3: Language modeling | How do the training-loss trajectories compare across model variants? | [exp3_language_modeling.py](exp3_language_modeling.py) | [exp3_language_modeling.md](exp3_language_modeling.md) |

Use Python with NumPy and Matplotlib. Run from the repository root:

```bash
python3 experiments/exp1_bandwidth.py
python3 experiments/exp2_memory_wall.py
python3 experiments/exp3_language_modeling.py
```

The scripts run offline and resolve paths relative to `Path(__file__).resolve().parent`.

## Files and reproducibility

```text
experiments/
├── README.md                   # Figure and data conventions
├── exp1_bandwidth.py           # One script per experiment
├── exp1_bandwidth.md           # Question, method, results, reproduction
├── exp2_memory_wall.py
├── exp2_memory_wall.md
├── exp3_language_modeling.py
├── exp3_language_modeling.md
├── data/                       # Version-controlled inputs
│   ├── exp1_bandwidth.csv
│   └── exp3_loss.csv
└── outputs/                    # Generated files; ignored by Git
    ├── exp1_bandwidth.pdf
    ├── exp1_bandwidth.png
    └── ...
```

Add each experiment as `expN_short_name.py` and `expN_short_name.md`, then add it to the table above. Keep this directory's navigation in this README.

Commit the script, analysis, and compact source data needed to reproduce the result. Put generated PDFs, PNGs, derived CSV tables, and local previews in `outputs/`. The root `.gitignore` ignores `/experiments/outputs/`; `experiments/data/` remains version-controlled. Large instrument logs and hardware build products stay local.

The analysis document records the question, data provenance, units, formulas, transformations, results, and reproduction command. Record smoothing windows, excluded samples, aggregation rules, and the meaning of uncertainty bars. Identify measurements, analytical estimates, and illustrative trajectories. Preserve the numerical coordinates when changing presentation.

Experiment 1's CSV contains measurement summaries. Experiment 2 reads the canonical MixerLoop configs and keeps public-model configurations with pinned sources in its registry. Experiment 3 documents its training-loss smoothing in its analysis.

## Eight-color palette

Use these exact HEX values for plotting. The first three colors have Pantone TCX references; the remaining five are coordinating screen colors selected for this palette.

| Order | Color | HEX | Pantone TCX reference | Reserved model |
|---|---|---|---|---|
| 1 | Ming green | `#3AA278` | 16-5930 Ming Green | MixerLoop |
| 2 | Coral red | `#D75C5D` | 17-1644 Spiced Coral | GDN |
| 3 | Periwinkle | `#6667AB` | 17-3938 Very Peri | FullLoop |
| 4 | Lake blue | `#4395C4` | — | — |
| 5 | Amber | `#D99532` | — | — |
| 6 | Berry | `#B65C91` | — | — |
| 7 | Yellow green | `#8C9F45` | — | — |
| 8 | Slate gray | `#78838F` | — | — |

The palette takes inspiration from the lighter midtones in the [Pantone Autumn/Winter 2027/28 outlook](https://www.pantone.com/na/en-us/products/trend-books/pantoneview-colour-planner-autumn-winter-2027-28). It is the project's own selection.

```python
PALETTE = (
    "#3AA278", "#D75C5D", "#6667AB", "#4395C4",
    "#D99532", "#B65C91", "#8C9F45", "#78838F",
)
MODEL_COLORS = {"MixerLoop": PALETTE[0], "GDN": PALETTE[1], "FullLoop": PALETTE[2]}
```

### Match color to the data

- **Unordered categories:** assign colors in palette order to a stable category list. Keep each category's color across related figures and subplots, even when a category is absent. Use the reserved mappings whenever MixerLoop, GDN, or FullLoop appears; assign other categories the remaining colors.
- **Ordered values:** use a single-hue sequence for loop counts, model sizes, or bandwidth levels. Increasing values use darker shades. For green, interpolate from `#E3F2EB` to `#3AA278`. Use the light end for filled regions; keep line colors dark enough to read on white.
- **Signed differences:** use a diverging scale centered on zero, for example coral `#D75C5D` through off-white `#F7F7F4` to green `#3AA278`. State which direction is positive and use symmetric limits when comparing signed magnitudes.
- **Emphasis:** draw the primary result at full opacity. Use light fills or reduced opacity for uncertainty and supporting regions. Keep labels opaque.
- **Dense comparisons:** pair colors with line styles or markers. Eight colors are available, but small figures usually read best with two to four main series. Use separate panels when curves overlap heavily.

Color alone does not distinguish every pair under color-vision deficiencies. In particular, check red/green, purple/blue, and amber/yellow-green combinations in the final figure. Use a color-vision simulator and grayscale preview for dense comparisons; retain identifiable line styles or markers.

## Layout and typography

| Element | Setting |
|---|---|
| Canvas | 3.6 × 3.6 inches, white |
| PDF | Vector, embedded TrueType fonts, fixed square page |
| PNG | 300 dpi, 1080 × 1080 pixels |
| Font | DejaVu Sans |
| Title | 13 pt, semibold, green `#3AA278`, centered on the canvas |
| Axis labels | 10 pt; label padding 7 pt |
| Ticks and legend | 9 pt |
| Point annotations | 8 pt |
| Text and ticks | Charcoal `#303330` |
| Axis edges | `#777A73`, 0.8 pt; top and right spines hidden |
| Grid | `#E8E5DE`, 0.55–0.7 pt, behind the data |
| Curves | 1.1 pt for dense traces; 1.6 pt for sparse series |
| Markers | 4.5 pt; error-bar caps 2 pt |
| Legend | Frameless, inside unused plot space; handle length 2.2 |
| Starting margins | left 0.18, right 0.94, bottom 0.20, top 0.85 |

Use a short title and axis labels with explicit units. Show enough ticks to read the scale without crowding. Keep methodological details in the experiment document or paper caption. Larger multi-panel figures can use more space while preserving these font sizes at their final publication size.

After plotting, inspect the exported figure for excessive empty space inside the axes and around the panel. Adjust limits, legend placement, and margins so the data fills the available area comfortably, with enough room for labels and uncertainty bars. Preserve meaningful baselines and comparisons when tightening limits; bar charts normally start at zero.

### Matplotlib starting point

Copy this setup into the experiment's script and add its data plotting code at the indicated location. Set semantic colors explicitly for model comparisons.

```python
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
PALETTE = (
    "#3AA278", "#D75C5D", "#6667AB", "#4395C4",
    "#D99532", "#B65C91", "#8C9F45", "#78838F",
)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.labelpad": 7,
    "axes.prop_cycle": plt.cycler(color=PALETTE),
    "text.color": "#303330",
    "axes.labelcolor": "#303330",
    "axes.edgecolor": "#777A73",
    "xtick.color": "#303330",
    "ytick.color": "#303330",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "pdf.fonttype": 42,
    "savefig.bbox": None,
})

fig, ax = plt.subplots(figsize=(3.6, 3.6))
fig.subplots_adjust(left=0.18, right=0.94, bottom=0.20, top=0.85)
fig.suptitle("Short Title", x=0.5, y=0.96, fontsize=13,
             fontweight="semibold", color=PALETTE[0])
ax.set_axisbelow(True)
ax.grid(axis="y", color="#E8E5DE", linewidth=0.6)

# Plot data, set limits/ticks/labels, and place any legend here.
# ax.legend(frameon=False, fontsize=9, handlelength=2.2,
#           borderaxespad=0.3, loc="upper right")

# Center the complete panel, including axis labels and tick labels.
fig.canvas.draw()
bounds = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
position = ax.get_position()
shift = 0.5 - (bounds.x0 + bounds.x1) / 2
ax.set_position([position.x0 + shift, position.y0,
                 position.width, position.height])

OUTPUT.mkdir(parents=True, exist_ok=True)
for extension in ("pdf", "png"):
    fig.savefig(OUTPUT / f"expN_short_name.{extension}", dpi=300, facecolor="white")
plt.close(fig)
```

Keep the fixed canvas when exporting; `bbox_inches="tight"` changes the page dimensions. Adjust layout inside the canvas to remove excessive whitespace.

Experiment 2 retains its existing architecture-map styling. Its layout-specific colors are recorded in `exp2_memory_wall.py`; use the palette above for new figures.

## Review before sharing

1. Run the script from the repository root and verify its numerical checks. Confirm data provenance, units, smoothing, and uncertainty definitions in the analysis document.
2. Render the PDF and inspect it alongside the PNG at publication size. Check label clipping, overlapping text, legend placement, and line visibility.
3. Check that titles and full panels are centered, related figures have matching dimensions, and neither the outer margins nor the data region contains excessive unused space.
4. Verify palette assignments across figures. For crowded comparisons, inspect color-vision and grayscale previews and distinguish series with line styles or markers.
5. Confirm the square PDF is 259.2 × 259.2 points and its PNG is 1080 × 1080 pixels. Regenerate outputs from the script after any correction.
6. Review Git status: include scripts, analyses, and input data; keep generated outputs local.
