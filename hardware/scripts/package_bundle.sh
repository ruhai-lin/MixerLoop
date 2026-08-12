#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd -- "$SCRIPT_DIR/.." && pwd)
MODEL_DIR=${1:?Usage: $0 MODEL_DIR}
VITIS_ROOT=${XILINX_VITIS:-/opt/xilinx/2025.2/Vitis}
PLATFORM_ROOT=${PLATFORM_ROOT:-$VITIS_ROOT/base_platforms/xilinx_kv260_base_202520_1}
XCLBIN="$PROJECT/outputs/link/binary_container_1.xclbin"
BUNDLE="$PROJECT/outputs/bundle"
WEIGHT_PATH=${WEIGHT_PATH:-$MODEL_DIR/model.q8.bin}
TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL_DIR/tokenizer.bin}

if [[ ! -f "$XCLBIN" ]]; then
  echo "missing xclbin: $XCLBIN (run scripts/build_link.sh first)" >&2
  exit 1
fi
if [[ ! -f "$WEIGHT_PATH" || ! -f "$TOKENIZER_PATH" ]]; then
  echo "missing model.q8.bin or tokenizer.bin in $MODEL_DIR" >&2
  exit 1
fi

mkdir -p "$BUNDLE/model"

cp -f "$PROJECT/outputs/host/gdn_host" "$BUNDLE/gdn_host"
cp -f "$XCLBIN" "$BUNDLE/binary_container_1.bin"
cp -f "$WEIGHT_PATH" "$BUNDLE/model/model.q8.bin"
cp -f "$TOKENIZER_PATH" "$BUNDLE/model/tokenizer.bin"

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
