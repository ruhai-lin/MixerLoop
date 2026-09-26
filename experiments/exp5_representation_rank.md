# Experiment 5: representation rank

Four mixer calls change effective rank differently across capture positions. All four checkpoints increase rank at the mid FFN input and at both block outputs. At the end capture, mixer-output and FFN-input rank decrease. The result is a position-dependent redistribution of representation rank.

## Data and method

`data/exp5_representation_rank.csv` contains 24 rows from `analysis/updated_release_mechanisms/a1_a4_a5/` in the contributor's document workspace: four checkpoints, two capture roles (`mid`, `end`), and three positions (`attention_post`, `ffn_input`, `block_post`). The source protocol is centered effective rank, pooled fit and validation, comparing MixerLoop loop 4 against loop 1. The retained rank records each list 1,024 windows and 490,496 tokens.

For every cell the figure displays:

```text
100 × (effective_rank_T4 - effective_rank_T1) / hidden_width
```

Hidden width is read from the canonical MixerLoop config: 256 for 13M and 768 for 100M. The script verifies these widths and all rank deltas against `data/mechanism_checkpoints.csv`. Source position names appear as Mixer out, FFN in, and Block out in the figure.

The source tables preserve the centering and evaluation protocol, but omit the exact rank estimator and the layer indices represented by `mid` and `end`. These definitions remain to be supplied from the source analysis. This script reproduces the provided summaries.

## Results

The displayed changes span -4.05% to +3.28% of hidden width. Mid FFN-input changes are +0.40% to +1.05%; end mixer-output changes are -4.05% to -2.09%; end block-output changes are +0.93% to +3.28%.

The shared metadata also preserves the corresponding trained MixerLoop-versus-GDN CORE comparison:

| Checkpoint pair | CORE difference (points) |
|---|---|
| 100M, s1337 | +2.2971 |
| 13M, s1337 | +0.7841 |
| 13M, s2026 | -0.1965 |
| 13M, s42 | +0.3777 |

CORE compares model variants; the heatmap compares loop counts within MixerLoop. Their relationship is descriptive: the 13M seed-2026 CORE decrease remains part of the retained evidence.

## Reproduction

```bash
python3 experiments/exp5_representation_rank.py
```

Run from the repository root with NumPy and Matplotlib. The script prints the normalized range and exports `experiments/outputs/exp5_representation_rank.pdf` and `.png` on a 5.4 × 3.6 inch canvas. The PDF heatmap and colorbar are vector; the PNG preview is 300 dpi. Purple represents negative changes and amber positive changes on one symmetric scale centered at zero.

Experiment 6 imports this script's rank calculation and drawing functions to place the same heatmap in panel (a) of the combined figure. Both panels share the checkpoint rows and use equal cell widths and heights.
