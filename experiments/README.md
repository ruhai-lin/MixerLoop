# Paper experiments

This directory contains independent paper analyses and figures. It is not part of training, export, or deployment.

| Experiment | Question | Script | Analysis |
|---|---|---|---|
| 1: Bandwidth | When does extra recurrent compute become visible in decode throughput? | [exp1_bandwidth.py](exp1_bandwidth.py) | [exp1_bandwidth.md](exp1_bandwidth.md) |
| 2: Memory wall | How do active FFN traffic and mixer compute differ across architectures? | [exp2_memory_wall.py](exp2_memory_wall.py) | [exp2_memory_wall.md](exp2_memory_wall.md) |

## Run

Use a Python environment with NumPy and Matplotlib. From the repository root:

```bash
python3 experiments/exp1_bandwidth.py
python3 experiments/exp2_memory_wall.py
```

Both scripts run offline and resolve paths relative to their own location.
Experiment 2 reads the canonical MixerLoop configs in the repository; public-model configurations and pinned sources remain in its registry.
No shared plotting framework or additional configuration loader is required.

## Files and data

Each experiment has one Python entry point and one Markdown analysis.
Keep the question, accounting rules, sources, interpretation, limitations, and reproduction commands in each analysis.
Commit scripts, analysis, and the compact input data required to reproduce figures.
The existing experiment 1 CSV contains measurement summaries; it must not be described as raw per-run samples.
Large instrument logs and hardware build products remain local.

Generated vector PDFs, 300-dpi PNG previews, and derived CSV tables go into `outputs/`, which is ignored by Git.
Experiment 1 produces `exp1_bandwidth.pdf/png`; experiment 2 produces `exp2_memory_wall.pdf/png`, `model_architecture_metrics.csv`, and `hardware_balance_metrics.csv`.
Do not hand-edit generated figures or tables.

## Figure style

Use a white background, DejaVu Sans, restrained line widths, readable labels at final publication size, and no decorative panels.
Use a square 3.3-inch canvas for a compact mechanism figure; wider measurement figures may use the available paper width.
Export vector PDF with embedded TrueType fonts and PNG at 300 dpi.

| Role | Color |
|---|---|
| MixerLoop | Forest green `#435E48` |
| Memory Wall | Mustard `#C7A33A` (label `#B28D24`) |
| MoE | Slate blue `#7896A3`, fill `#EDF2F2` |
| Dense | Terracotta `#B87563`, fill `#F4EBE7` |
| Auxiliary trajectories | Sage gray `#BCC5BA` |
| Axes and text | Charcoal `#303330` |
| Legend | White background, `#B9B7B0` border, `#555753` text |

These are the current architecture-figure colors. The existing throughput figure retains its validated styling in this organizational migration.
Distinguish measurements, analytical calculations, and schematic guides explicitly in the experiment text.
Do not imply that an illustrative sweep is a measured result. State the meaning of error bars and hardware balance lines.
Preserve model coordinates when adjusting presentation.

## Verification

Run both scripts after changes. Experiment 2 includes accounting and registry assertions.
For path-only refactors, compare numerical CSV outputs and rendered figures against the prior artifacts.
Inspect PDF renders for clipped labels and overlapping text before using them in the paper.
