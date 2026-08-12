#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd -- "$SCRIPT_DIR/.." && pwd)
COMMON=${COMMON:?set COMMON to the extracted Xilinx common-image directory}
ENV_SETUP=${ENV_SETUP:-$COMMON/environment-setup-cortexa72-cortexa53-xilinx-linux}
SYSROOT=${SYSROOT:-$COMMON/sysroots/cortexa72-cortexa53-xilinx-linux}

if [[ ! -f "$ENV_SETUP" ]]; then
  echo "missing sysroot environment setup: $ENV_SETUP" >&2
  exit 1
fi

unset LD_LIBRARY_PATH
# shellcheck disable=SC1090
set +u
source "$ENV_SETUP"
set -u

mkdir -p "$PROJECT/outputs/host" "$PROJECT/outputs/logs"

# -mcmodel=large with PIC/PIE disabled: the packed parameter buffer pushes the
# host well past the small code model's reach.
aarch64-xilinx-linux-g++ -Wall -Wextra -std=c++2a -O2 \
  -mcmodel=large -fno-PIC -fno-PIE -no-pie -g --sysroot="$SYSROOT" \
  -I"$PROJECT/src" \
  -I"$SYSROOT/usr/include/xrt" \
  "$PROJECT/src/main.cpp" \
  "$PROJECT/src/decode.cpp" \
  "$PROJECT/src/weight.cpp" \
  "$PROJECT/src/vocab.cpp" \
  -L"$SYSROOT/usr/lib" \
  -lxilinxopencl -lxrt_coreutil -lpthread -lrt -ldl \
  -o "$PROJECT/outputs/host/gdn_host" \
  2>&1 | tee "$PROJECT/outputs/logs/host_build.log"

echo "built: $PROJECT/outputs/host/gdn_host"
