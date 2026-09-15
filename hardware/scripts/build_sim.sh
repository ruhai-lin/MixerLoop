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

COMMON_FLAGS=(-Wall -Wextra -Wno-unknown-pragmas -O2 -std=c++20 -pthread
              -I"$PROJECT/src" -I"$VITIS_INCLUDE")
# Match the FPGA's exp/log/sqrt model. System libm can move an activation
# across an INT8 rounding boundary and change an otherwise correct token.
MATH_LIB="$VITIS_INCLUDE/../../lnx64/lib/csim"
FPO_LIB="$VITIS_INCLUDE/../lnx64/tools/fpo_v7_1"
SIM_LIBS=(-L"$MATH_LIB" -L"$FPO_LIB" -Wl,--disable-new-dtags,-rpath,"$MATH_LIB:$FPO_LIB"
          -Wl,--no-as-needed -lhlsm-GCC46 -lIp_floating_point_v7_1_bitacc_cmodel
          -Wl,--as-needed -lm -pthread)

"$CXX" "${COMMON_FLAGS[@]}" -DBUILD_DECODE_KERNEL \
  -c "$PROJECT/src/decode.cpp" -o "$OUT/decode_kernel.o"
"$CXX" "${COMMON_FLAGS[@]}" -DUSE_CPU_ONLY \
  -c "$PROJECT/src/decode.cpp" -o "$OUT/decode_cpu.o"
"$CXX" "${COMMON_FLAGS[@]}" \
  -c "$PROJECT/src/weight.cpp" -o "$OUT/weight.o"
"$CXX" "${COMMON_FLAGS[@]}" -DUSE_CPU_ONLY \
  -c "$PROJECT/tools/kernel_sim.cpp" -o "$OUT/kernel_sim.o"

"$CXX" "$OUT/decode_kernel.o" "$OUT/decode_cpu.o" "$OUT/weight.o" \
  "$OUT/kernel_sim.o" "${SIM_LIBS[@]}" -o "$OUT/kernel_sim"

echo "built: $OUT/kernel_sim"

if [[ $# -gt 0 ]]; then
  if [[ ${COSIM:-0} == 1 ]]; then
    export GDN_SIM_WEIGHT=$(realpath "$1")
    export GDN_SIM_STEPS=${2:-1}
    if [[ $# -gt 2 ]]; then export GDN_SIM_LOOPS=$3; fi
    if [[ ${4:-} == gates ]]; then export GDN_SIM_GATES=1; fi
    if [[ ${4:-} == generate ]]; then export GDN_SIM_GENERATE=1; fi
    vitis-run --mode hls --cosim \
      --config "$PROJECT/outputs/hls/decode/hls_config.cfg" \
      --work_dir "$PROJECT/outputs/hls/decode/work" \
      --hls.tb.file "$PROJECT/tools/kernel_sim.cpp" \
      --hls.tb.cflags "-I$PROJECT/src -DUSE_CPU_ONLY" \
      --hls.cosim.ldflags "$OUT/decode_cpu.o $OUT/weight.o ${SIM_LIBS[*]}" \
      --hls.cosim.trace_level port
    # Argmax alone can miss a corrupted weight. Enforce the HLS memory
    # dependence promises too, including the ring's empty/wrap boundary.
    if grep -q 'Critical WARNING: Due to pragma' \
        "$PROJECT/outputs/hls/decode/work/hls/sim/verilog/"*.log; then
      echo "RTL memory dependence violation; do not release this kernel" >&2
      exit 1
    fi
  else
    "$OUT/kernel_sim" "$@"
  fi
fi
