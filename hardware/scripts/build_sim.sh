#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd -- "$SCRIPT_DIR/.." && pwd)
VITIS_INCLUDE=${VITIS_INCLUDE:-${XILINX_VITIS:-/opt/xilinx/2025.2/Vitis}/include}
OUT="$PROJECT/outputs/sim"
CXX=${CXX:-g++}

if [[ ! -f "$VITIS_INCLUDE/ap_int.h" ]]; then
  echo "missing Vitis HLS headers: $VITIS_INCLUDE/ap_int.h" >&2
  exit 1
fi

mkdir -p "$OUT"

COMMON_FLAGS=(-Wall -Wextra -Wno-unknown-pragmas -O2 -std=c++20
              -I"$PROJECT/src" -I"$VITIS_INCLUDE")

"$CXX" "${COMMON_FLAGS[@]}" -DBUILD_DECODE_KERNEL \
  -c "$PROJECT/src/decode.cpp" -o "$OUT/decode_kernel.o"
"$CXX" "${COMMON_FLAGS[@]}" -DUSE_CPU_ONLY \
  -c "$PROJECT/src/decode.cpp" -o "$OUT/decode_cpu.o"
"$CXX" "${COMMON_FLAGS[@]}" \
  -c "$PROJECT/src/weight.cpp" -o "$OUT/weight.o"
"$CXX" "${COMMON_FLAGS[@]}" -DUSE_CPU_ONLY \
  -c "$PROJECT/tools/kernel_sim.cpp" -o "$OUT/kernel_sim.o"

"$CXX" "$OUT/decode_kernel.o" "$OUT/decode_cpu.o" "$OUT/weight.o" \
  "$OUT/kernel_sim.o" -lm -o "$OUT/kernel_sim"

echo "built: $OUT/kernel_sim"

if [[ $# -gt 0 ]]; then
  "$OUT/kernel_sim" "$1" "${2:-8}"
fi
