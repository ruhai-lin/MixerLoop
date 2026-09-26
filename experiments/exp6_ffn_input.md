# Experiment 6: FFN input statistics

At the end capture, four mixer passes increase both top-10% energy concentration and mean pairwise cosine for all four checkpoints. Mid-capture changes are smaller and have mixed signs. The figure describes how the FFN input changes before the block's single FFN computation.

## Data and method

`data/exp6_ffn_input.csv` contains 16 rows from `analysis/updated_release_mechanisms/a1_a4_a5/` in the contributor's document workspace: four checkpoints, two capture roles (`mid`, `end`), and two metrics. Records use validation FFN-input activations and compare MixerLoop pass 4 against pass 1.

Each panel displays the supplied `delta` after checking it against `data/mechanism_checkpoints.csv`:

- **Top-10% energy:** `100 × delta`, expressed as change in energy share in percentage points (pp).
- **Mean pair cosine:** `delta` in the original dimensionless cosine units.

Each metric has its own symmetric color scale centered on zero, with purple for negative changes and amber for positive changes. Numbers printed in every cell retain the sign and unit interpretation independently of color.

The compact source records omit the exact `mid/end` layer indices, the energy-ranking convention, and the pair-sampling procedure for cosine. These details need to be recovered from the source analysis before interpreting the metrics beyond their reported changes.

## Results

End-capture energy concentration increases by 4.92 to 12.70 pp. Mean pair cosine increases by 0.066 to 0.218. Mid-capture energy changes range from -0.25 to +1.26 pp; cosine changes range from -0.0046 to +0.0199.

These statistics complement the position-specific effective-rank changes in experiment 5. Greater concentration and cosine similarity describe activation geometry; their connection to prediction quality is examined alongside experiment 4 and the retained CORE comparisons.

## Reproduction

```bash
python3 experiments/exp6_ffn_input.py
```

Run from the repository root with NumPy and Matplotlib. The script prints both metric ranges and exports `experiments/outputs/exp5_6_rank_ffn_input.pdf` and `.png` on an 8.4 × 3.6 inch canvas.

Panel (a) uses experiment 5's rank calculation and drawing functions. Panel (b) contains the two FFN-input metrics. All ten columns have equal cell width, with four aligned rows and checkpoint labels only at the far left. Each metric retains its own colorbar and units. The rank title and both FFN metric titles use the same green 13 pt style. The PDF cells and colorbars are vector; the PNG preview is 300 dpi.
