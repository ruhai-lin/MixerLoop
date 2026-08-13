<div align="center">

# MixerLoop

**Recurrent compute allocation for Gated Delta Networks, from pretraining to FPGA deployment.**

</div>

MixerLoop repeats the Gated DeltaNet mixer while executing the FFN once. The
repeated passes share weights, so recurrent compute can increase without adding
model parameters.

```text
for physical layer i:
    for loop slot in range(T):
        h = h + GDN_i(RMSNorm_i(h))
    h = h + FFN_i(FFNNorm_i(h))
```

There are no loop-specific parameters or learned residuals. T=1 is exactly the
native GDN block. During autoregressive decoding, every loop slot owns an
independent recurrent state.

## Deployment profile

This repository fixes one software/hardware anchor before exploring ablations:

| Field | Value |
|---|---:|
| dataset | TinyStories |
| tokenizer / vocabulary | Llama 2 / 32,000 |
| context length | 256 |
| hidden size | 256 |
| physical layers | 8 |
| heads / head dimension | 8 / 32 |
| FFN intermediate size | 768 |
| short convolution | 4 |
| loop count | 1–4; release training uses T=4 |
| deployment arithmetic | W8A8, group size 32 |
| checkpoint format | GDNe v2 |

The HF config records the training loop count. The Q8 format does not: T=1 and
T=4 use the same tensor order and weight layout.

## Environment

Run inside the WSL Linux filesystem, for example
`/home/<user>/Projects/MixerLoop`, rather than `/mnt/c`. The Python environment
lives at the repository root.

```bash
git clone https://github.com/ruhai-lin/MixerLoop.git
cd MixerLoop
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel ninja packaging
# Install the PyTorch wheel appropriate for this machine first.
python -m pip install torch
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Training and export

`train.sh` pins the TinyStories revision and uses the included Llama 2
tokenizer. A successful run saves the distributed checkpoint, HF model,
tokenizer, and GDNe v2 Q8 weight automatically.

Local 1,000-step T=4 anchor:

```bash
LOOP_COUNT=4 STEPS=1000 SEQ_LEN=256 NGPU=1 \
  MICRO_BATCH=1 GLOBAL_BATCH=8 CHECKPOINT_INTERVAL=1000 \
  bash train.sh
```

The default output is `outputs/tinystories15m_t4`:

```text
outputs/tinystories15m_t4/
├── checkpoint/step-1000/
├── config.json
├── model.safetensors
├── tinystories15m_t4_q8.bin
├── tokenizer.model
├── tokenizer.bin
└── source/
```

The full run uses the same command and configuration with `STEPS=100000`.
Microbatch size may change for available memory, but comparisons should keep
the effective global batch, seed, data revision, optimizer, and token budget
fixed.

The release recipe matches the existing 15M training budget: global batch 512,
context 256 (131,072 tokens per optimizer step), AdamW at `5e-4`, 1,000 warmup
steps, cosine decay to zero, and 100,000 optimizer steps. On the 2×3090 Ti
machine it runs as:

```bash
LOOP_COUNT=4 STEPS=100000 SEQ_LEN=256 NGPU=2 \
  MICRO_BATCH=128 GLOBAL_BATCH=512 WARMUP_STEPS=1000 \
  bash train.sh
```

## Hardware baseline

`hardware/` is the verified T=1 single-kernel GDN accelerator imported from
`gdn.hls`. It is intentionally retained as the clean hardware milestone; T=4
scheduling and memory/computation overlap are the next hardware phase.

The canonical comparison weights are stored as:

```text
hardware/model/tinystories15m_t1_q8.bin
hardware/model/tinystories15m_t4_q8.bin  # added after the full T=4 run
```

See `hardware/README.md` for the accelerator and toolchain details.

## Repository layout

```text
assets/tokenizer/        Llama 2 tokenizer
configs/                 fixed MixerLoop deployment profile
custom_models/mixerloop/ Transformers model implementation
flame/                   FLAME/TorchTitan training and final export
eval/                    language-model evaluation entry points
hardware/                golden GDN accelerator and deployment weights
tests/                   model and export contracts
```

Earlier paper experiments and implementation milestones live outside the
release tree under the locally ignored `references/` directory.
