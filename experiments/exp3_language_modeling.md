# Experiment 3: Language modeling

This figure compares logged training loss for the 13M GDN, MixerLoop, and FullLoop runs in `data/exp3_loss.csv`.

The plot uses the recorded `tokens_B_from_draft` values directly and focuses the x-axis on 1 to 10 billion processed tokens. A centered 17-sample mean keeps per-update noise legible at the square figure size while preserving the recorded trend.

## Reproduce

Run from the repository root:

```bash
python3 experiments/exp3_language_modeling.py
```

The script writes `outputs/exp3_language_modeling.pdf` and `outputs/exp3_language_modeling.png`. Both use a square 3.6-inch canvas matching experiment 1. The PDF is vector; the 300-dpi PNG is 1080 × 1080 pixels.
