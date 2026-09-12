#!/usr/bin/env bash
set -euo pipefail

# Use the root environment; training settings remain native FLAME arguments.
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

argument_value() {
  local key="$1"
  shift
  while (($#)); do
    if [[ "$1" == "$key" && $# -ge 2 ]]; then
      printf '%s' "$2"
      return
    elif [[ "$1" == "$key="* ]]; then
      printf '%s' "${1#*=}"
      return
    fi
    shift
  done
  echo "Missing $key" >&2
  return 2
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec python -m flame.train --help
fi

path=$(argument_value --job.dump_folder "$@")
steps=$(argument_value --training.steps "$@")
tokenizer=$(argument_value --model.tokenizer_path "$@")
export WANDB_PROJECT=mixerloop
# Supply WANDB_NAME explicitly in formal experiment commands.
export WANDB_NAME=${WANDB_NAME:-$(basename "$path")}

torchrun \
  --nnodes="${NNODE:-1}" \
  --nproc_per_node="${NGPU:-8}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR:-localhost}:${MASTER_PORT:-0}" \
  --local-ranks-filter="${LOG_RANK:-0}" \
  --role=rank --tee=3 --log-dir="$path/logs" \
  -m flame.train "$@"

echo "Training completed; exporting final DCP to HF"
python -m flame.utils.convert_dcp_to_hf \
  --path "$path" --step "$steps" \
  --config "$path/config.json" --tokenizer "$tokenizer"
echo "HF export completed: $path"
