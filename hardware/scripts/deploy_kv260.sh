#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd -- "$SCRIPT_DIR/.." && pwd)
BUNDLE="$PROJECT/outputs/bundle"
KV260=${KV260:-ubuntu@192.168.137.123}
REMOTE=${REMOTE:-/home/ubuntu/Projects/gdn_bundle}
APP=${APP:-gdn}

ssh "$KV260" "rm -rf $REMOTE && mkdir -p $REMOTE/model"
scp "$BUNDLE/gdn_host" "$BUNDLE/binary_container_1.bin" \
    "$BUNDLE/pl.dtbo" "$BUNDLE/shell.json" "$KV260:$REMOTE/"
scp "$BUNDLE/model/model.q8.bin" "$BUNDLE/model/tokenizer.bin" \
    "$KV260:$REMOTE/model/"

# The board account must provide passwordless sudo (or already be root).
ssh "$KV260" "sudo mkdir -p /lib/firmware/xilinx/$APP && \
  sudo cp $REMOTE/binary_container_1.bin $REMOTE/pl.dtbo $REMOTE/shell.json \
     /lib/firmware/xilinx/$APP/ && \
  sudo xmutil unloadapp >/dev/null 2>&1; \
  sudo xmutil loadapp $APP"

echo "deployed to $KV260:$REMOTE (app=$APP)"

# After board testing, switch back to the quiet starter app so the fan settles.
# Call with RESTORE_STARTER=1 (default) after a smoke run, or RESTORE_STARTER=0
# to leave the gdn bitstream loaded.
RESTORE_STARTER=${RESTORE_STARTER:-1}
if [[ "$RESTORE_STARTER" == "1" ]]; then
  ssh "$KV260" "sudo xmutil unloadapp >/dev/null 2>&1; \
    sudo xmutil loadapp k26-starter-kits"
  echo "restored k26-starter-kits on $KV260 (fan quiet)"
fi
