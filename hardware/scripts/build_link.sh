#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd -- "$SCRIPT_DIR/.." && pwd)
VITIS_ROOT=${XILINX_VITIS:-/opt/xilinx/2025.2/Vitis}
PLATFORM_ROOT=${PLATFORM_ROOT:-$VITIS_ROOT/base_platforms/xilinx_kv260_base_202520_1}
PLATFORM=${PLATFORM:-$PLATFORM_ROOT/xilinx_kv260_base_202520_1.xpfm}
FREQ=${FREQ:-150000000}

OUT="$PROJECT/outputs/link"
mkdir -p "$OUT" "$PROJECT/outputs/logs/link"

if ! command -v v++ >/dev/null 2>&1; then
  echo "v++ not found; source the Vitis settings64.sh first" >&2
  exit 1
fi
if [[ ! -f "$PLATFORM" ]]; then
  echo "missing KV260 platform: $PLATFORM" >&2
  exit 1
fi

v++ -l -t hw \
  --platform "$PLATFORM" \
  "$PROJECT/outputs/hls/decode/work/decode.xo" \
  -o "$OUT/binary_container_1.xclbin" \
  --clock.default_freqhz "$FREQ" \
  --connectivity.sp decode_1.packed_params:HP0 \
  --connectivity.sp decode_1.side:HP0 \
  --connectivity.sp decode_1.next_token:HP0 \
  --save-temps \
  --temp_dir "$OUT/_x" \
  --log_dir "$PROJECT/outputs/logs/link" \
  --report_dir "$OUT/reports" \
  2>&1 | tee "$PROJECT/outputs/logs/vpp_link.log"
