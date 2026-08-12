# MixerLoop: Anonymous Code and Data Supplement

This archive contains the implementation, fixed configurations, tokenizer,
evaluation programs, figure generator, and raw result tables used by the
submission “Allocating Recurrent Compute in Looped Language Models.”
Checkpoint tensors and the ClimbMix corpus are omitted because of archive size;
both are reproducible from the included training and data-preparation entry
points.

## Environment and tests

Python 3.11 is recommended. Install a PyTorch build compatible with the local
CUDA driver, then install the package and run the regression suite:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio
pip install -e '.[dev,eval]'
pytest -q
```

The submitted runs used two 24 GB RTX 3090 Ti GPUs, Ubuntu 22.04, Python 3.10,
PyTorch 2.13.0/CUDA 13.0, Transformers 4.57.3, and Flash Linear Attention
0.5.1. Additional CUDA/NCCL installation notes are in
`flame-installation.md`.

## Data and training

Download and tokenize the fixed 170-shard ClimbMix subset plus its held-out
shard:

```bash
python -m flame.datasets.climbmix --data_dir data/climbmix-10b
```

The paper recipe fixes a global batch of 512 sequences, context length 1,024,
100,000 updates, and seed 1,337. The same launcher trains all reported
architectures:

```bash
NGPU=2 bash train.sh gdn 15m
NGPU=2 bash train.sh mixerloop 15m
NGPU=2 bash train.sh fullloop 15m

NGPU=2 MICRO_BATCH=1 bash train.sh gdn 110m
NGPU=2 MICRO_BATCH=1 bash train.sh mixerloop 110m
NGPU=2 MICRO_BATCH=1 bash train.sh fullloop 110m
```

Each run exports a Transformers checkpoint under `outputs/{architecture}-{size}`.

## Evaluation

Run the complete CORE bundle:

```bash
python eval/core_eval.py \
  --model_path outputs/mixerloop-15m \
  --out_dir outputs/mixerloop-15m/eval
```

Run the finite causal readout-ITR intervention:

```bash
python eval/itr_eval.py \
  --model_paths outputs/gdn-15m outputs/mixerloop-15m outputs/fullloop-15m \
  --data_dir data/climbmix-10b \
  --seq_len 128 --num_windows 16 --prediction_positions 16 \
  --data_seed 2027 \
  --out_dir outputs/itr-readout-15m
```

Run the synchronized BF16 prefill protocol used in the paper:

```bash
python eval/throughput_eval.py \
  --model_paths outputs/mixerloop-15m outputs/fullloop-15m \
  --seq_len 1024 --batch_size 32 --warmup 6 --iterations 20 \
  --out_file outputs/throughput-15m-batch32.json
```

`outputs/` in this archive contains the submitted CORE CSV files, ITR
JSON/CSV files (including the random and robustness controls), throughput
timings, and sanitized checkpoint provenance. Rebuild the two quantitative
manuscript figures and a code-rendered architecture schematic with:

```bash
python paper/make_figures.py
```

The final Figure 1 schematic is included as `paper/Figures/main.png`.
