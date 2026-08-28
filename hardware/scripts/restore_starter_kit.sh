#!/usr/bin/env bash
set -euo pipefail

# Put the board back on the default starter-kit app after benchmarking. The
# accelerator bitstream keeps the fan spinning loudly, so always run this once
# measurements are collected.

KV260=${KV260:-ubuntu@192.168.137.123}
SUDO_PASSWORD=${SUDO_PASSWORD:-}
if [[ -n "$SUDO_PASSWORD" ]]; then
  printf '%s\n' "$SUDO_PASSWORD" |
    ssh "$KV260" "sudo -S -p '' bash -lc 'xmutil unloadapp >/dev/null 2>&1 || true; xmutil loadapp k26-starter-kits; xmutil listapps'"
else
  ssh "$KV260" "sudo xmutil unloadapp >/dev/null 2>&1 || true; \
    sudo xmutil loadapp k26-starter-kits; \
    sudo xmutil listapps"
fi

echo "restored k26-starter-kits on $KV260"
