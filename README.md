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
| 1p6b | 2,048 | 22 | 8 | 192 / 384 | 5,632 | 128,256 | 4,096 |

13m/100m use the included Llama 2 tokenizer. 600m/1p6b retain the
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

Use `train.sh` to pass FLAME's native CLI arguments without a separate recipe wrapper.
The script runs training, then converts the final DCP to standard HF files.
Example: 13m, FineWeb-Edu, global batch 128, 10,000,007,168 input tokens:

```bash
WANDB_PROJECT=mixerloop WANDB_NAME=fineweb13m_mixerloop_10b_s1337 NGPU=1 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --model.config configs/mixerloop_13m.json \
  --model.tokenizer_path assets/tokenizer \
  --job.dump_folder outputs/mixerloop-13m/fineweb-edu-10B-s1337 \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name sample-10BT \
  --training.dataset_split train --training.streaming \
  --training.data_dir '' --training.num_workers 0 \
  --training.seq_len 1024 --training.context_len 1024 \
  --training.batch_size 8 --training.gradient_accumulation_steps 16 \
  --training.steps 76294 --training.seed 1337 --training.max_norm 1.0 \
  --training.data_parallel_replicate_degree 1 \
  --training.data_parallel_shard_degree 1 \
  --training.tensor_parallel_degree 1 --training.disable_loss_parallel \
  --training.mixed_precision_param bfloat16 --training.mixed_precision_reduce float32 \
  --activation_checkpoint.mode none \
  --optimizer.name AdamW --optimizer.implementation fused \
  --optimizer.lr 5e-4 --optimizer.eps 1e-8 --optimizer.beta1 0.9 --optimizer.beta2 0.95 \
  --optimizer.weight_decay 0.1 \
  --lr_scheduler.warmup_steps 1000 --lr_scheduler.decay_type cosine --lr_scheduler.lr_min 0.0 \
  --checkpoint.enable_checkpoint --checkpoint.interval 2000 \
  --checkpoint.keep_latest_k 2 --checkpoint.load_step -1 \
  --metrics.log_freq 20 --metrics.enable_wandb
```

For matched GDN, select `configs/gdn_13m.json` and a separate output directory.
Keep global batch, seed, data order, schedule and token budget matched.
Larger recipes require their matching tokenizer/context.

AdamW excludes parameters with fewer than two dimensions, `A_log`, and
`dt_bias` from weight decay. Matrix parameters, including `residual_weight`,
decay. Checkpoint save/load and rolling retention use the original TorchTitan
`CheckpointManager`, without the later milestone extension.

Specify dataset revision only for the selected dataset. Local parquet uses
`--training.dataset parquet --training.data_files 'path/*.parquet'`;
pretokenized ClimbMix uses `--training.dataset climbmix --training.data_dir path`.
Set microbatch and gradient accumulation explicitly, keeping the intended global batch.
For parquet, also pass `--training.data_dir ''` to clear the ClimbMix directory default.
Before a formal launch, record the full command, resolved model config, dataset,
seed, learning rate, batch accounting, steps, processed tokens, GPU count, git SHA
and environment. A W&B run name does not set any training parameters.
When resuming W&B logging, provide the existing `WANDB_RUN_ID` and
`WANDB_RESUME=must` explicitly; the launcher preserves these environment values.

To export an already completed run without training again:

```bash
python -m flame.utils.convert_dcp_to_hf \
  --path outputs/mixerloop-13m/fineweb-edu-10B-s1337 --step 76294 \
  --config outputs/mixerloop-13m/fineweb-edu-10B-s1337/config.json \
  --tokenizer assets/tokenizer
```

Use the run's resolved config, not a subsequently edited geometry preset.
This exports HF only; no Q8 or hardware tokenizer is produced.
DCP and logs remain local in the run directory. Upload only HF model/tokenizer
files and `eval/core_eval.csv`, never the whole training directory.

## CORE evaluation

```bash
python eval/core_eval.py \
  --model_path outputs/mixerloop-13m/fineweb-edu-10B-s1337 \
  --out_dir outputs/mixerloop-13m/fineweb-edu-10B-s1337/eval \
  --max_per_task -1 --no_head_stats
```

The evaluator expects an HF checkpoint and adapts
[nanochat CORE](https://github.com/karpathy/nanochat/blob/master/nanochat/core_eval.py)
to HF models and batched inference. Task definitions and examples come from the
[eval_bundle used by nanochat](https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip).
Baselines are read from that bundle, with the three
[DCLM CORE v2 corrections](https://github.com/mlfoundations/dclm/pull/115)
applied in memory: CommonsenseQA 40.3%, LSAT AR 25%, and language identification 25%.
CORE v2 is the mean of `(accuracy - baseline) / (1 - baseline)`.

The bundle is cached under `core_bundle/` and is never modified by the evaluator.
Results record `Core_v2` and `eval_version`; no separate metadata download is needed.

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
