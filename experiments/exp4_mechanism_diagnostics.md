# Experiment 4: mechanism diagnostics

This experiment packages the final, paper-facing diagnostics for the
mechanism hypothesis: repeated MixerLoop mixer updates can refine the hidden
representation before the block's single FFN readout. The implementation
semantics are in `custom_models/mixerloop/layers.py:15-65`: the GDN mixer is
called `loop_count` times, the residual-weight update is applied at each loop,
and one `GatedMLP` is applied after the loop.

The figures are descriptive evidence, not a causal mediation estimate. They
show four fixed-checkpoint ClimbMix pairs and retain the observed three-win,
one-loss CORE boundary.

## Reproduction

From the repository root, with NumPy and Matplotlib installed:

```bash
python experiments/exp4_mechanism_diagnostics.py
```

The script writes vector PDF and 300-dpi PNG files to `experiments/outputs/`:

- `exp4_quality_by_loops.pdf/.png`: cross entropy, Top-1 accuracy, and
  correct-token probability across native mixer calls 1–4.
- `exp4_paired_loop_gain.pdf/.png`: loop-4-minus-loop-1 means with the stored
  95% confidence intervals.
- `exp4_layer_position_diagnostics.pdf/.png`: paired CORE differences,
  effective-rank changes, and FFN-input energy/cosine changes.

The input directory is `experiments/data/exp4_mechanism/`. It contains compact
derived tables rather than raw training logs or checkpoints. The script derives
the rank panel as centered effective-rank change divided by the canonical
hidden width (256 for 13m and 768 for 100m), and scales the FFN-input deltas by
100 for the heatmap labels.

## Scope and provenance

The retained pairs are exactly:

| Pair | Evidence role |
|---|---|
| `100m/climbmix-10B-s1337` | 100M fixed-checkpoint pair |
| `13m/climbmix-10B-s1337` | 13M seed 1337 pair |
| `13m/climbmix-10B-s42` | 13M seed 42 pair |
| `13m/climbmix-10B-s2026` | 13M seed 2026 pair |

The compact tables were filtered from the verified paper analyses
`analysis/updated_release_mechanisms/a7/` and
`analysis/updated_release_mechanisms/a1_a4_a5/` in the document workspace:

- `quality.csv`: 16 quality rows (four pairs × four loop counts), each based
  on 205 validation windows.
- `comparisons.csv`: 12 loop-4-minus-loop-1 quality summaries (three metrics
  × four pairs), including the stored confidence limits.
- `performance_link.csv`: four CORE links and the selected rank/FFN summary
  fields; `hidden_width` is taken from the canonical 13m/100m configurations.
- `a4_rank_differences.csv`: 24 centered rank rows (two capture roles × three
  positions × four pairs), filtered to the pooled fit/validation,
  MixerLoop-loop4 versus loop1 protocol.
- `a5_ffn_input_differences.csv`: 16 FFN-input rows (two roles × two metrics
  × four pairs), filtered to validation, MixerLoop pass 4 versus pass 1.

The source analyses contain additional FineWeb and exploratory rows; they are
not copied here because they are outside the paper's retained Figure 4 scope.
No checkpoints, raw logs, tokens, credentials, or generated outputs are
versioned by this experiment.

## Interpretation boundary

The quality curves establish the measured loop-count trajectories for these
fixed checkpoints. The diagnostic panel reports changes in representation
shape and FFN-input statistics alongside the paired CORE result. These
observations support the stated refinement/readout hypothesis as a bounded
mechanistic interpretation; they do not prove that representation changes
caused the CORE difference, and the retained CORE loss is an explicit
counterexample to a universal performance claim.
