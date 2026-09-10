# MixerLoop

MixerLoop repeats the Gated DeltaNet mixer T times and executes the FFN once.
Mixer weights are shared across loops. A zero-initialized
`residual_weight[T, dim]` is shared across all physical layers:

```text
for physical layer i:
    for t in range(T):
        h_input = h
        h = h + GDN_i(RMSNorm_i(h))
        h = h + residual_weight[t] * h_input
    h = h + FFN_i(FFNNorm_i(h))
```

## Canonical configurations

Each size has `configs/gdn_<size>.json` (FLA's native GDN) and
`configs/mixerloop_<size>.json` (MixerLoop T=4). Both use tied embeddings,
short convolution size 4, and `expand_v=2`.

| Size | Hidden | Layers | Heads | Key / value head dim | FFN | Vocabulary | Context |
|---|---:|---:|---:|---:|---:|---:|---:|
| 13m | 256 | 5 | 6 | 32 / 64 | 704 | 32,000 | 1,024 |
| 100m | 768 | 9 | 9 | 64 / 128 | 2,112 | 32,000 | 1,024 |
| 600m | 1,024 | 21 | 8 | 96 / 192 | 2,816 | 128,256 | 4,096 |
| 1p3b | 2,048 | 22 | 8 | 192 / 384 | 5,632 | 128,256 | 4,096 |

13m/100m use the included Llama 2 tokenizer. 600m/1p3b retain the
[LT2](https://github.com/chili-lab/LT2) Llama 3 tokenizer and 4096-token context
recipe; supply that tokenizer explicitly.

Names identify geometry presets. Instantiated GDN parameter counts are
12,895,356 / 100,444,194 / 445,771,024 / 1,578,945,120 respectively.
MixerLoop adds exactly `4 * hidden_size` shared residual parameters.

## Environment

Run inside the WSL Linux filesystem, e.g. `~/Projects/MixerLoop`, rather than
`/mnt/c`. Use one environment at the repository root.

```bash
git clone https://github.com/ruhai-lin/MixerLoop.git
cd MixerLoop
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel ninja packaging
# Select a PyTorch wheel appropriate for the GPU.
python -m pip install torch
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Training

Use FLAME's native CLI to select the configuration, tokenizer and dataset.
Example: 13m, FineWeb-Edu, global batch 128, 10,000,007,168 input tokens:

```bash
torchrun --nproc_per_node=1 -m flame.train \
  --model.config configs/mixerloop_13m.json \
  --model.tokenizer_path assets/tokenizer \
  --job.dump_folder outputs/fineweb13m_t4 \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name sample-10BT \
  --training.dataset_split train --training.streaming \
  --training.seq_len 1024 --training.context_len 1024 \
  --training.batch_size 8 --training.gradient_accumulation_steps 16 \
  --training.steps 76294 --training.seed 1337 \
  --training.data_parallel_replicate_degree 1 \
  --training.data_parallel_shard_degree 1 \
  --optimizer.name AdamW --optimizer.implementation fused \
  --optimizer.lr 5e-4 --optimizer.beta1 0.9 --optimizer.beta2 0.95 \
  --optimizer.weight_decay 0.1 \
  --lr_scheduler.warmup_steps 1000 --lr_scheduler.decay_type cosine \
  --checkpoint.enable_checkpoint --checkpoint.interval 2000
```

For matched GDN, select `configs/gdn_13m.json` and a separate output directory.
Keep global batch, seed, data order, schedule and token budget matched.
Larger recipes require their matching tokenizer/context.

AdamW excludes parameters with fewer than two dimensions, `A_log`, and
`dt_bias` from weight decay. Matrix parameters, including `residual_weight`,
decay. Checkpoint save/load and rolling retention use the original TorchTitan
`CheckpointManager`, without the later milestone extension.

`train.sh` remains a TinyStories convenience recipe using the 13m geometry.
Use the native CLI for other datasets and formal experiments.

## CORE evaluation

```bash
python eval/core_eval.py \
  --model_path outputs/fineweb13m_t4 \
  --tokenizer_path assets/tokenizer \
  --out_dir outputs/fineweb13m_t4/core_eval
```

The evaluator expects an HF checkpoint and adapts
[nanochat CORE](https://github.com/karpathy/nanochat/blob/master/nanochat/core_eval.py)
to HF models and batched inference. Task definitions and examples come from the
[eval_bundle used by nanochat](https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip).
Random baselines come from the official DCLM v2 metadata pinned to
[`361714b`](https://github.com/mlfoundations/dclm/commit/361714bdd60bb9b7f4b2d8354cebbf0dec0c329e).
CORE v2 is the mean of `(accuracy - baseline) / (1 - baseline)`.

The official bundle is cached under `core_bundle/nanochat/`, separately from
legacy patched bundles. Official DCLM metadata is downloaded separately beside
the bundle; no private baseline table is maintained and bundle metadata is
never overwritten. Results record `Core_v2`, `eval_version`, the source URLs,
and SHA-256 hashes of the task configuration and scoring metadata.

## Hardware and local artifacts

`hardware/` preserves the earlier KV260 milestone; it does not yet implement
the residual MixerLoop architecture or these canonical geometries.
See `hardware/README.md` for the milestone implementation and toolchain.
Training does not automatically produce compatible Q8 hardware weights.

Checkpoints, weights and logs stay local and are excluded from Git.
Historical outputs are kept under `outputs/legacy/` on each machine.
Active isolated runs retain their existing source and output paths.

## Layout

```text
assets/tokenizer/        Llama 2 tokenizer
configs/                 matched canonical configurations
custom_models/mixerloop/ Transformers model implementation
flame/                   training and checkpoint conversion
eval/                    CORE and lm-eval entry points
hardware/                earlier accelerator milestone
tests/                   model, data, optimizer and export contracts
outputs/legacy/          local historical artifacts (ignored)
```
