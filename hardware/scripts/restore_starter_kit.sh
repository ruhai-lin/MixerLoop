#!/usr/bin/env bash
set -euo pipefail

# Put the board back on the default starter-kit app after benchmarking. The
# accelerator bitstream keeps the fan spinning loudly, so always run this once
# measurements are collected.

KV260=${KV260:-ubuntu@192.168.137.123}
ssh "$KV260" "sudo xmutil unloadapp >/dev/null 2>&1; \
  sudo xmutil loadapp k26-starter-kits; \
  sudo xmutil listapps"

echo "restored k26-starter-kits on $KV260"
