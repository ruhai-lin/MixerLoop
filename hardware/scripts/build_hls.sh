#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd -- "$SCRIPT_DIR/.." && pwd)
OUT="$PROJECT/outputs/hls/decode"
LOG="$PROJECT/outputs/logs"

mkdir -p "$OUT" "$LOG"

if ! command -v v++ >/dev/null 2>&1; then
  echo "v++ not found; source the Vitis settings64.sh first" >&2
  exit 1
fi

cat > "$OUT/hls_config.cfg" <<CFG
part=xck26-sfvc784-2LV-c

[hls]
flow_target=vitis
package.output.format=xo
package.output.syn=1
syn.top=decode
syn.file=$PROJECT/src/decode.cpp
syn.cflags=-I$PROJECT/src -DBUILD_DECODE_KERNEL
syn.interface.m_axi_max_widen_bitwidth=128
clock=150MHz
CFG

v++ -c --mode hls \
  --config "$OUT/hls_config.cfg" \
  --work_dir "$OUT/work" \
  2>&1 | tee "$LOG/decode_hls.log"
