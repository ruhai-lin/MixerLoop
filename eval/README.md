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

## Commands and dependencies

Run from the repository root after installing the project dependencies:

```bash
python eval/theory_capture.py ids --help
python eval/theory_probe.py --help
python eval/theory_probe.py smoke-immediate --help
python eval/skip_eval.py --formal-preflight --help
python eval/render_skip_figures.py --help
PYTHONPATH=. pytest -q tests/test_theory_capture.py tests/test_theory_probe.py tests/test_skip_eval.py tests/test_render_skip_figures.py
```

The capture/probe/skip paths require Python, PyTorch, NumPy, Transformers,
flash-linear-attention (`fla`), and the local `custom_models` package. Probe
and formal skip runs are CUDA-oriented and require caller-supplied checkpoints,
the fixed validation shard, tokenizer, and output roots. The renderer also
requires Matplotlib and Pillow. W&B is forced offline by the skip runtime.

`trust_remote_code=True` is used by model loading in the probe/skip paths;
only use checkpoints whose local code and provenance have been audited. The
CLIs accept local paths and record them in manifests, so do not put credentials
in command-line arguments.

## Evidence and limits

These modules provide the mechanics needed to reproduce the paper's Spec 01/02
capture/probe and Spec 04 skip studies. File names alone do not establish a
mapping to a paper experiment ID; pair a run with the corresponding study
record and protocol before interpreting it as E009/E010/E014 evidence.

`run_manifest.json` and `run_checks.json` from `skip_eval.py` explicitly set
`accepted_output=false`. They are execution/audit artifacts, not accepted paper
results. Do not commit them, checkpoints, weights, W&B directories, large raw
logs, or generated figures to this repository.

The runtime code preserves several explicit boundaries: the full split can
consume substantial RAM, output-byte caps do not cap RAM, `trust_remote_code`
executes checkpoint-provided code, and the skip renderer checks the six
checkpoint names/12-file count but does not prove every layer/pass cell is
present. The README and study records must retain these limits.
