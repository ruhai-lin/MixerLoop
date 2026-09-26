# Experiment 4: loop quality

Increasing inference mixer calls from one to four improves cross entropy, top-1 accuracy, and correct-token probability for all four retained ClimbMix checkpoints. This experiment measures how prediction quality changes within each fixed trained MixerLoop checkpoint.

## Data and method

The inputs are compact tables from `analysis/updated_release_mechanisms/a7/` in the contributor's document workspace:

- `data/exp4_loop_quality.csv`: four checkpoints at loop counts 1, 2, 3, and 4; each row summarizes 205 validation windows and 98,195 tokens.
- `data/exp4_loop_differences.csv`: the 12 paired T=4 minus T=1 metric differences and their stored 95% confidence limits.

The checkpoints are 13M seeds 1337, 42, and 2026, and 100M seed 1337, trained on ClimbMix for 10B tokens. Each inference loop executes the shared mixer; the layer's FFN executes once. T=1 here uses the same trained MixerLoop weights as T=4.

The figure has three square panels, one per metric, with both model sizes in each panel. The 13M line is the equal-weight mean of seeds 1337, 42, and 2026 at each loop count; its light envelope spans their observed minimum and maximum. The 100M line shows seed 1337. Colors, markers, and line styles distinguish model size. Accuracy and correct-token probability are converted to percentages. Lines connect the recorded loop counts without smoothing.

Increasing mixer depth consistently improves prediction quality across all three evaluated 13M seeds. The 100M checkpoint exhibits the same trend. The seed-range envelope displays the variability around this shared trend; paired confidence intervals remain in the table below.

## Results

The table reports improvement from T=1 to T=4. Positive values mean lower cross entropy or higher accuracy/probability. Brackets contain the supplied 95% confidence limits; probability changes use percentage points (pp).

| Checkpoint | CE reduction | Top-1 gain (pp) | Correct-token probability gain (pp) |
|---|---|---|---|
| 13M, s1337 | 2.383 [2.338, 2.425] | 26.098 [25.342, 26.870] | 20.400 [19.613, 21.210] |
| 13M, s42 | 2.278 [2.227, 2.328] | 24.212 [23.454, 24.968] | 19.279 [18.485, 20.040] |
| 13M, s2026 | 1.995 [1.954, 2.037] | 22.781 [22.090, 23.531] | 19.770 [19.012, 20.520] |
| 100M, s1337 | 2.458 [2.417, 2.497] | 28.725 [28.006, 29.476] | 23.135 [22.300, 23.931] |

The plotting script verifies every mean difference against the curve endpoints. It reverses the CE sign and orders the transformed confidence limits when producing the improvement table. The source tables contain confidence limits but omit the estimator and resampling method; those details need to accompany any paper claim based on interval coverage.

## Reproduction

From the repository root, with NumPy and Matplotlib installed:

```bash
python3 experiments/exp4_loop_quality.py
```

Outputs in `experiments/outputs/`:

- `exp4_loop_quality.pdf`: vector, 7.2 × 3.2 inches; one row of three square plotting areas.
- `exp4_loop_quality.png`: 300 dpi preview.
- `exp4_loop_quality_summary.csv`: full-precision improvements and confidence limits, also printed to the console.

The paired-difference summary belongs to this experiment and is presented as a table alongside the loop-count curves.
