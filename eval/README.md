# Mechanism-experiment runtime sources

This directory contains the execution and analysis primitives used by the
paper's mechanism studies. It does not contain checkpoints, validation shards,
tokenizers, W&B logs, generated tables, or accepted experimental outputs.

## Sources

| File | Purpose | Output boundary |
|---|---|---|
| `theory_capture.py` | Spec 01 deterministic sample/split capture from a fixed validation shard. | Writes sample/split JSON only to a caller-provided output root. |
| `theory_probe.py` | Spec 02 hidden-state capture, frozen LM-head readout, linear probes, and bootstrap summaries for GDN, MixerLoop, and FullLoop paths. | CUDA-oriented run manifests, probe rows, checks, and analysis inputs; no checkpoints are bundled. |
| `skip_eval.py` | Spec 04 skip-causality runtime/preflight over the fixed six-checkpoint inventory. | Writes runtime stages, skip ablations, mITR/CORE selection manifests, and audit JSON. |
| `render_skip_figures.py` | Offline renderer for skip delta-loss heatmaps and mITR/delta-loss scatter plots. | Writes PNGs and `figure_manifest.json` to a new caller-provided output directory. |

The four `tests/test_*.py` files exercise deterministic capture, probe record
schemas, skip metrics/preflight, and skip figure naming/counts. They use
temporary directories for generated test data.

## Paper experiment data generation

These sources generate experiment measurements; they are separate from plot
scripts and the derived CSVs under `experiments/data/`.

| Paper evidence | Generator | Generated measurements |
|---|---|---|
| Figure 5 effective-rank statistics (A4) | `i004_readout.py`, `i004_capture.py`, `i004_data.py`, `i004_rank.py` | Streams token vectors into raw/centered spectra and entropy effective rank; writes `rank_spectra.json` and `rank_moments.pt`. |
| Figure 5 FFN-input statistics (A5) | `i004_readout.py`, `i004_capture.py`, `i004_data.py`, `i004_distribution.py` | Streams activation-distribution, energy-concentration, and cosine summaries; writes `summary.json`, `windows.jsonl`, `moments.pt`, and `comparisons.csv`. |
| Figure 6 block-local loop-count sweep (E023) | `i006_local_target_sweep.py`, `i004_capture.py`, `i004_data.py` | Scores validation windows while changing one middle or final block to one through four Mixer calls and keeping other blocks at four; writes per-condition rows and `manifest.json`. |

Example A4/A5 capture (run from the repository root; replace each `/data/...`
path with the corresponding local checkpoint, selection, and shard path):

```bash
python -m eval.i004_readout \
  --weights-root /data/checkpoints \
  --selection /data/selection/manifest.json \
  --data-root /data/selection/shards \
  --output /data/outputs/i004 \
  --observer eval.i004_rank:make_observer \
  --observer-output /data/outputs/i004/a4_rank \
  --observer eval.i004_distribution:create_observer \
  --observer-output /data/outputs/i004/a5_distribution
```

Example E023 sweep using the checkpoint pairs in the existing quality table
(all checkpoint and token data remain caller-supplied):

```bash
python eval/i006_local_target_sweep.py \
  --repo . \
  --weights-root /data/checkpoints \
  --selection /data/selection/manifest.json \
  --data-root /data/selection/shards \
  --quality experiments/data/exp4_mechanism/quality.csv \
  --output /data/outputs/e023
```

These commands generate measurements, not paper figures. Selection metadata,
binary token shards, tokenizer/model code, and checkpoint weights are not
included; provide the validated inputs used by the corresponding study. These
scripts do not train models.

## Commands and dependencies

Run from the repository root after installing the project dependencies:

```bash
python eval/theory_capture.py ids --help
python eval/theory_probe.py --help
python eval/theory_probe.py smoke-immediate --help
python eval/skip_eval.py --formal-preflight --help
python eval/render_skip_figures.py --help
python -m eval.i004_readout --help
python eval/i006_local_target_sweep.py --help
PYTHONPATH=. pytest -q tests/test_theory_capture.py tests/test_theory_probe.py tests/test_skip_eval.py tests/test_render_skip_figures.py
```

The capture/probe/skip/A4/A5/E023 paths require Python, PyTorch, NumPy, Transformers,
flash-linear-attention (`fla`), and the local `custom_models` package. Probe
and model-based capture/sweep runs are CUDA-oriented and require caller-supplied
checkpoints, the fixed validation shard, tokenizer, and output roots. The
renderer also requires Matplotlib and Pillow. W&B is used offline by the skip
and A4/A5 diagnostic paths.

`trust_remote_code=True` is used by model loading in the probe/skip paths;
only use checkpoints whose local code and provenance have been audited. The
CLIs accept local paths and record them in manifests, so do not put credentials
in command-line arguments.

## Evidence and limits

The sources above provide measurement-generation paths for the listed A4/A5
and E023 evidence, as well as the Spec 01/02 capture/probe and Spec 04 skip
studies. A script run is not itself an accepted paper result: interpret its
output only with the corresponding study record, protocol, and validated inputs.

`run_manifest.json` and `run_checks.json` from `skip_eval.py` explicitly set
`accepted_output=false`. They are execution/audit artifacts, not accepted paper
results. Do not commit them, checkpoints, weights, W&B directories, large raw
logs, or generated figures to this repository.

The runtime code preserves several explicit boundaries: the full split can
consume substantial RAM, output-byte caps do not cap RAM, `trust_remote_code`
executes checkpoint-provided code, and the skip renderer checks the six
checkpoint names/12-file count but does not prove every layer/pass cell is
present. The README and study records must retain these limits.
