#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd -- "$SCRIPT_DIR/.." && pwd)
MODEL_DIR=${1:-$PROJECT/model}
VITIS_ROOT=${XILINX_VITIS:-/opt/xilinx/2025.2/Vitis}
PLATFORM_ROOT=${PLATFORM_ROOT:-$VITIS_ROOT/base_platforms/xilinx_kv260_base_202520_1}
XCLBIN="$PROJECT/outputs/link/binary_container_1.xclbin"
BUNDLE="$PROJECT/outputs/bundle"
TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL_DIR/tokenizer.bin}

if [[ ! -f "$XCLBIN" ]]; then
  echo "missing xclbin: $XCLBIN (run scripts/build_link.sh first)" >&2
  exit 1
fi
if [[ ! -f "$PROJECT/outputs/host/gdn_host" ]]; then
  echo "missing host: $PROJECT/outputs/host/gdn_host (run scripts/build_host.sh first)" >&2
  exit 1
fi
if [[ ! -f "$TOKENIZER_PATH" ]]; then
  echo "missing tokenizer.bin: $TOKENIZER_PATH" >&2
  exit 1
fi

mkdir -p "$BUNDLE/model"

cp -f "$PROJECT/outputs/host/gdn_host" "$BUNDLE/gdn_host"
cp -f "$XCLBIN" "$BUNDLE/binary_container_1.bin"
cp -f "$TOKENIZER_PATH" "$BUNDLE/model/tokenizer.bin"

copy_weight() {
  local src=$1
  local dst=$2
  if [[ -f "$src" ]]; then
    cp -f "$src" "$dst"
  fi
}

copy_weight "$MODEL_DIR/tinystories15m_t1_q8.bin" "$BUNDLE/model/tinystories15m_t1_q8.bin"
copy_weight "$MODEL_DIR/tinystories15m_t4_q8.bin" "$BUNDLE/model/tinystories15m_t4_q8.bin"
copy_weight "$MODEL_DIR/model.q8.bin" "$BUNDLE/model/model.q8.bin"

if [[ ! -f "$BUNDLE/model/model.q8.bin" && -f "$BUNDLE/model/tinystories15m_t1_q8.bin" ]]; then
  cp -f "$BUNDLE/model/tinystories15m_t1_q8.bin" "$BUNDLE/model/model.q8.bin"
fi
if [[ ! -f "$BUNDLE/model/tinystories15m_t1_q8.bin" && ! -f "$BUNDLE/model/model.q8.bin" ]]; then
  echo "missing T=1 weights in $MODEL_DIR" >&2
  exit 1
fi

if [[ -f "$PLATFORM_ROOT/sw/boot/pl.dtbo" ]]; then
  cp -f "$PLATFORM_ROOT/sw/boot/pl.dtbo" "$BUNDLE/pl.dtbo"
else
  echo "missing platform device-tree overlay: $PLATFORM_ROOT/sw/boot/pl.dtbo" >&2
  exit 1
fi

cat > "$BUNDLE/shell.json" <<'JSON'
{
  "shell_type": "XRT_FLAT",
  "num_slots": "1"
}
JSON

chmod +x "$BUNDLE/gdn_host"
echo "bundle ready: $BUNDLE"
ls -l "$BUNDLE" "$BUNDLE/model"
