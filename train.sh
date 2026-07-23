#!/usr/bin/env bash

set -euo pipefail

argument_value() {
  local wanted="$1"
  shift
  while (($#)); do
    if [[ "$1" == "$wanted" && $# -ge 2 ]]; then
      printf '%s' "$2"
      return 0
    fi
    shift
  done
  return 1
}

# `train.sh ARCH SIZE` is the locked ClimbMix paper recipe. Any other argument
# list is forwarded as ordinary Flame CLI overrides.
if [[ $# -eq 2 && "$1" =~ ^(gdn|mixerloop|ffnloop|fullloop)$ && "$2" =~ ^(15m|110m)$ ]]; then
  arch="$1"
  size="$2"
  NNODE=${NNODE:-1}
  NGPU=${NGPU:-2}
  SEED=${SEED:-1337}
  STEPS=${STEPS:-100000}
  DATA_DIR=${DATA_DIR:-data/climbmix-10b}
  TOKENIZER=${TOKENIZER:-assets/tokenizer}
  OUTPUT=${OUTPUT:-outputs/${arch}-${size}}

  if [[ "$size" == "15m" ]]; then
    MICRO_BATCH=${MICRO_BATCH:-8}
  else
    MICRO_BATCH=${MICRO_BATCH:-1}
  fi

  denominator=$((NGPU * MICRO_BATCH))
  if ((512 % denominator != 0)); then
    echo "NGPU * MICRO_BATCH must divide the global batch of 512 sequences" >&2
    exit 2
  fi
  GRAD_ACCUM=${GRAD_ACCUM:-$((512 / denominator))}
  if ((NGPU * MICRO_BATCH * GRAD_ACCUM != 512)); then
    echo "The paper recipe requires NGPU * MICRO_BATCH * GRAD_ACCUM = 512" >&2
    exit 2
  fi

  train_args=(
    --job.config_file flame/models/fla.toml
    --job.dump_folder "$OUTPUT"
    --model.config "configs/${arch}_${size}.json"
    --model.tokenizer_path "$TOKENIZER"
    --optimizer.name AdamW
    --optimizer.implementation fused
    --optimizer.eps 1e-8
    --optimizer.beta1 0.9
    --optimizer.beta2 0.95
    --optimizer.weight_decay 0.1
    --optimizer.lr 5e-4
    --lr_scheduler.warmup_steps 1000
    --lr_scheduler.decay_type cosine
    --lr_scheduler.lr_min 0.0
    --training.batch_size "$MICRO_BATCH"
    --training.seq_len 1024
    --training.context_len 1024
    --training.gradient_accumulation_steps "$GRAD_ACCUM"
    --training.steps "$STEPS"
    --training.max_norm 1.0
    --training.dataset karpathy/climbmix-400b-shuffle
    --training.dataset_split train
    --training.data_dir "$DATA_DIR"
    --training.num_workers 0
    --training.seed "$SEED"
    --training.data_parallel_replicate_degree "$NGPU"
    --training.data_parallel_shard_degree 1
    --training.tensor_parallel_degree 1
    --training.disable_loss_parallel
    --checkpoint.interval 2000
    --checkpoint.keep_latest_k 2
    --checkpoint.load_step -1
    --metrics.log_freq 20
  )
  if [[ "${WANDB:-0}" == "1" ]]; then
    train_args+=(--metrics.enable_wandb)
  fi
else
  NNODE=${NNODE:-1}
  NGPU=${NGPU:-8}
  train_args=("$@")
fi

LOG_RANK=${LOG_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-0}

path=$(argument_value --job.dump_folder "${train_args[@]}") || {
  echo "Missing --job.dump_folder" >&2
  exit 2
}
steps=$(argument_value --training.steps "${train_args[@]}") || {
  echo "Missing --training.steps" >&2
  exit 2
}
config=$(argument_value --model.config "${train_args[@]}") || {
  echo "Missing --model.config" >&2
  exit 2
}
tokenizer=$(argument_value --model.tokenizer_path "${train_args[@]}") || {
  echo "Missing --model.tokenizer_path" >&2
  exit 2
}

model_type=$(
  python -c \
    "import fla.models, custom_models, sys; from transformers import AutoConfig; print(AutoConfig.from_pretrained(sys.argv[1]).model_type)" \
    "$config"
)

mkdir -p "$path/source"
cp -a assets configs custom_models eval flame pyproject.toml train.sh "$path/source/"

run_name="${model_type}-$(basename "$path")"
run_id="${WANDB_RUN_ID:-${run_name}-$(date +%Y%m%d%H%M)}"
export WANDB_PROJECT=${WANDB_PROJECT:-mixerloop}
export WANDB_NAME=${WANDB_NAME:-$run_name}
export WANDB_RUN_ID=$run_id
export WANDB_RESUME=allow

echo "Launching ${model_type} training in ${path}"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun \
  --nnodes="$NNODE" \
  --nproc_per_node="$NGPU" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --local-ranks-filter="$LOG_RANK" \
  --role=rank \
  --tee=3 \
  --log-dir="$path/logs" \
  -m flame.train "${train_args[@]}"

echo "Converting final DCP checkpoint to Hugging Face format"
python -m flame.utils.convert_dcp_to_hf \
  --path "$path" \
  --step "$steps" \
  --config "$config" \
  --tokenizer "$tokenizer"

echo "Training and export complete: $path"
