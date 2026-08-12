#!/usr/bin/env bash

set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Follow the installation steps in README.md first." >&2
  exit 2
fi

export PATH="$ROOT/.venv/bin:$PATH"

LOOP_COUNT=${LOOP_COUNT:-4}
NGPU=${NGPU:-1}
NNODE=${NNODE:-1}
STEPS=${STEPS:-1000}
SEQ_LEN=${SEQ_LEN:-256}
MICRO_BATCH=${MICRO_BATCH:-1}
GLOBAL_BATCH=${GLOBAL_BATCH:-8}
SEED=${SEED:-1337}
DATASET=${DATASET:-roneneldan/TinyStories}
DATASET_SPLIT=${DATASET_SPLIT:-train}
DATASET_REVISION=${DATASET_REVISION:-f54c09fd23315a6f9c86f9dc80f725de7d8f9c64}
TOKENIZER=${TOKENIZER:-assets/tokenizer}
OUTPUT=${OUTPUT:-outputs/tinystories-15m-t${LOOP_COUNT}}
NUM_WORKERS=${NUM_WORKERS:-0}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-2000}

if ((LOOP_COUNT < 1 || LOOP_COUNT > 4)); then
  echo "LOOP_COUNT must be between 1 and 4" >&2
  exit 2
fi
denominator=$((NGPU * MICRO_BATCH))
if ((GLOBAL_BATCH % denominator != 0)); then
  echo "NGPU * MICRO_BATCH must divide GLOBAL_BATCH" >&2
  exit 2
fi
GRAD_ACCUM=${GRAD_ACCUM:-$((GLOBAL_BATCH / denominator))}
if ((NGPU * MICRO_BATCH * GRAD_ACCUM != GLOBAL_BATCH)); then
  echo "NGPU * MICRO_BATCH * GRAD_ACCUM must equal GLOBAL_BATCH" >&2
  exit 2
fi

train_args=(
  --job.config_file flame/models/fla.toml
  --job.dump_folder "$OUTPUT"
  --model.config configs/mixerloop_15m.json
  --model.tokenizer_path "$TOKENIZER"
  --model.loop_count "$LOOP_COUNT"
  --optimizer.name AdamW
  --optimizer.implementation fused
  --optimizer.eps 1e-8
  --optimizer.beta1 0.9
  --optimizer.beta2 0.95
  --optimizer.weight_decay 0.1
  --optimizer.lr "${LEARNING_RATE:-5e-4}"
  --lr_scheduler.warmup_steps "${WARMUP_STEPS:-100}"
  --lr_scheduler.decay_type cosine
  --lr_scheduler.lr_min 0.0
  --training.batch_size "$MICRO_BATCH"
  --training.seq_len "$SEQ_LEN"
  --training.context_len "$SEQ_LEN"
  --training.gradient_accumulation_steps "$GRAD_ACCUM"
  --training.steps "$STEPS"
  --training.max_norm 1.0
  --training.dataset "$DATASET"
  --training.dataset_split "$DATASET_SPLIT"
  --training.dataset_revision "$DATASET_REVISION"
  --training.data_dir ""
  --training.streaming
  --training.num_workers "$NUM_WORKERS"
  --training.seed "$SEED"
  --training.data_parallel_replicate_degree "$NGPU"
  --training.data_parallel_shard_degree 1
  --training.tensor_parallel_degree 1
  --training.disable_loss_parallel
  --checkpoint.enable_checkpoint
  --checkpoint.interval "$CHECKPOINT_INTERVAL"
  --checkpoint.keep_latest_k 2
  --checkpoint.load_step -1
  --metrics.log_freq "${LOG_FREQ:-20}"
)
if [[ "${WANDB:-0}" == "1" ]]; then
  train_args+=(--metrics.enable_wandb)
fi

mkdir -p "$OUTPUT/source"
cp -a assets configs custom_models flame pyproject.toml train.sh "$OUTPUT/source/"

run_name="mixerloop-$(basename "$OUTPUT")"
export WANDB_PROJECT=${WANDB_PROJECT:-mixerloop}
export WANDB_NAME=${WANDB_NAME:-$run_name}
export WANDB_RUN_ID=${WANDB_RUN_ID:-${run_name}-$(date +%Y%m%d%H%M)}
export WANDB_RESUME=allow

echo "Training MixerLoop T=$LOOP_COUNT for $STEPS steps in $OUTPUT"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun \
  --nnodes="$NNODE" \
  --nproc_per_node="$NGPU" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR:-localhost}:${MASTER_PORT:-0}" \
  --local-ranks-filter="${LOG_RANK:-0}" \
  --role=rank \
  --tee=3 \
  --log-dir="$OUTPUT/logs" \
  -m flame.train "${train_args[@]}"

resolved_config="$OUTPUT/config.json"
if [[ ! -f "$resolved_config" ]]; then
  echo "Training did not write resolved config: $resolved_config" >&2
  exit 1
fi

echo "Exporting HF and hardware artifacts"
python -m flame.utils.convert_dcp_to_hf \
  --path "$OUTPUT" \
  --step "$STEPS" \
  --config "$resolved_config" \
  --tokenizer "$TOKENIZER"

echo "Complete: $OUTPUT"
