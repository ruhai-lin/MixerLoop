<div align="center">

# MixerLoop

**Allocate recurrent compute to the token mixer, not automatically to the entire block.**

</div>

MixerLoop is the reference implementation for studying where recurrent depth
should be placed in Gated Delta Networks (GDNs). The repository exposes four
matched architectures, a fixed ClimbMix training recipe, deterministic CORE
evaluation, and Iterative Transport Rank (ITR) analysis.

## Architectures

Let each physical layer contain a GDN mixer $A_i$ and an FFN $F_i$, and let the
loop count be $T=4$.

| Name | Recurrent computation | Role |
|---|---|---|
| `gdn` | $F_i \circ A_i$ | no-loop control |
| `mixerloop` | $F_i \circ A_i^T$ | proposed architecture |
| `ffnloop` | $F_i^T \circ A_i$ | allocation control |
| `fullloop` | $(B_L \circ \cdots \circ B_1)^T$ | LT2-compatible full-stack loop |

`FullLoop` always means repetition of the complete physical layer stack. The
repository intentionally contains no local interleaved-loop variant.

## Installation

Python 3.11 is recommended. Install a PyTorch build that matches the local CUDA
driver before installing this package.

```bash
git clone https://github.com/ruhai-lin/MixerLoop.git
cd MixerLoop
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel

# Select the appropriate wheel index for your CUDA installation.
pip install torch torchvision torchaudio
pip install -e .
```

CUDA/NCCL and extension-build details are documented in
[`flame-installation.md`](flame-installation.md).

The software contract pins `transformers==4.57.3`, `datasets==4.5.0`,
`flash-linear-attention==0.5.1`, `torchdata==0.11.0`, and TorchTitan commit
`0b44d4c`. The release validation used PyTorch `2.13.0+cu130`; PyTorch itself
is left as a lower-bounded dependency because its wheel must match the host CUDA
installation.

## Prepare ClimbMix-10B

The paper corpus is fixed to train shards `00000`–`00169` from
`karpathy/climbmix-400b-shuffle`; shard `06542` is reserved for validation. The
included tokenizer is the exact SentencePiece model used to create the released
checkpoints.

```bash
python -m flame.datasets.climbmix --data_dir data/climbmix-10b
```

The resulting 170 training shards contain 10,690,433,521 tokens. The built-in
backend writes `manifest.json` with per-shard SHA-256 hashes and token counts.
Training reads the uint16 token files directly, so there is no
online-tokenization drift.

## Reproduce training

The locked recipe uses sequence length 1024, a global batch of 512 sequences,
AdamW with $(\beta_1,\beta_2)=(0.9,0.95)$, learning rate $5\times10^{-4}$,
1,000 warmup updates, and cosine decay to zero. The released legacy models ran
for 100,000 updates: 52,428,800,000 processed tokens over the fixed 10B-token
corpus.

The launcher derives gradient accumulation from GPU count and per-GPU
micro-batch while holding the global batch fixed:

```bash
# Primary missing comparison on two 24 GB GPUs
NGPU=2 DATA_DIR=data/climbmix-10b \
  bash train.sh ffnloop 15m

NGPU=2 DATA_DIR=data/climbmix-10b MICRO_BATCH=1 \
  bash train.sh ffnloop 110m
```

The same entry point reproduces every architecture:

```bash
NGPU=2 DATA_DIR=data/climbmix-10b \
  bash train.sh mixerloop 15m
```

Set `STEPS`, `SEED`, `OUTPUT`, or `MICRO_BATCH` through environment variables.
Changing `STEPS` creates a shorter diagnostic run and is not the paper recipe.
Set `WANDB=1` to enable Weights & Biases logging. Final DCP checkpoints are
automatically exported to Transformers `model.safetensors` format.

## Released checkpoints

Released checkpoints are hosted at
[`ruhai-lin/MixerLoop`](https://huggingface.co/ruhai-lin/MixerLoop) as
Transformers folders `{gdn,mixerloop,fullloop}-{15m,42m,110m}`. Downloaded
weights must use the same layout under local `outputs/` (for example
`outputs/mixerloop-15m`), matching the paths expected by training and
evaluation. New models trained by this repository are saved directly in
Transformers format; the one-time legacy conversion utilities are intentionally
not part of the release codebase.

Clone the full release into `outputs/`:

```bash
hf download ruhai-lin/MixerLoop --local-dir outputs
```

Or pull a single checkpoint (same path convention):

```bash
hf download ruhai-lin/MixerLoop \
  --include "mixerloop-15m/*" \
  --local-dir outputs
```

## Evaluation

CORE evaluation downloads Karpathy's fixed evaluation bundle on first use:

```bash
python eval/core_eval.py \
  --model_path outputs/mixerloop-15m \
  --out_dir outputs/mixerloop-15m/eval/core
```

Run the optional lm-eval suite after `pip install -e '.[eval]'`:

```bash
python eval/harness_eval.py \
  --model_path outputs/mixerloop-15m \
  --out_dir outputs/mixerloop-15m/eval/harness
```

The causal readout ITR monitor compares NoLoop, MixerLoop, and FullLoop on the
same held-out windows. It disables cross-token computation while retaining the
same mixer's token-local computation, then restores contextual mixer passes in
prefix order:

```bash
python eval/itr_eval.py \
  --model_paths \
    outputs/gdn-15m \
    outputs/mixerloop-15m \
    outputs/fullloop-15m \
  --data_dir data/climbmix-10b \
  --out_dir outputs/itr-readout-15m
```

The evaluator reports finite probability-space ITR/marginal ITR, squared
Hellinger effect, signed ground-truth log-probability gain, and layer-by-pass
heatmaps. It does not use input gradients or Jacobian sensitivity as a proxy for
task value.

The paper's synchronized BF16 prefill protocol is also executable directly:

```bash
python eval/throughput_eval.py \
  --model_paths outputs/mixerloop-15m outputs/fullloop-15m \
  --batch_size 32 \
  --out_file outputs/throughput-15m-batch32.json
```

Published results are:

| Architecture | 15M CORE | 42M CORE | 110M CORE |
|---|---:|---:|---:|
| GDN | 0.0501 | 0.0918 | 0.1416 |
| MixerLoop | **0.0652** | **0.1122** | **0.1556** |
| FFNLoop | -- | -- | -- |
| FullLoop | 0.0552 | 0.1072 | **0.1752** |

## Repository layout

```text
configs/                 15M and 110M configs for four architectures
custom_models/           MixerLoop, FFNLoop, and FullLoop implementations
flame/                   compact Flame/TorchTitan training engine
flame/datasets/          fixed ClimbMix download, tokenization, and reader
eval/                    CORE, lm-eval, and ITR entry points
tests/                   model, data-resume, and metric regression tests
```

## Reproducibility notes

- `ClimbMix-10B` names the unique corpus; the released models processed 52.43B
  training tokens over that corpus.
- Training examples include an explicit next-token label, matching the original
  LT2.c/LT3.c objective at all 1,024 positions.
- Checkpoint/resume includes the exact shuffled shard and window cursor.
- Weight decay is applied to matrix parameters but not norms, biases, `A_log`,
  or `dt_bias`, matching the original optimizer grouping.

## Acknowledgments

The training runtime builds on [Flame](https://github.com/fla-org/flame),
[TorchTitan](https://github.com/pytorch/torchtitan), and
[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention).
