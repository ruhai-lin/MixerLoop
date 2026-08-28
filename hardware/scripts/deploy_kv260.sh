#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd -- "$SCRIPT_DIR/.." && pwd)
BUNDLE="$PROJECT/outputs/bundle"
KV260=${KV260:-ubuntu@192.168.137.123}
REMOTE=${REMOTE:-/home/ubuntu/Projects/gdn_bundle}
APP=${APP:-gdn}

case "$REMOTE" in
  /home/*/Projects/*_bundle) ;;
  *)
    echo "refusing to replace unexpected remote path: $REMOTE" >&2
    exit 1
    ;;
esac
if [[ ! "$APP" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "invalid app name: $APP" >&2
  exit 1
fi
printf -v remote_q '%q' "$REMOTE"
printf -v remote_model_q '%q' "$REMOTE/model"

ssh "$KV260" "rm -rf -- $remote_q && mkdir -p -- $remote_model_q"
scp "$BUNDLE/gdn_host" "$BUNDLE/binary_container_1.bin" \
    "$BUNDLE/pl.dtbo" "$BUNDLE/shell.json" "$KV260:$remote_q/"
scp "$BUNDLE"/model/* "$KV260:$remote_model_q/"

# The board account may require a sudo password. Set SUDO_PASSWORD rather than
# storing it in the repository. Empty means passwordless sudo.
SUDO_PASSWORD=${SUDO_PASSWORD:-}
remote_sudo() {
  local command_q
  printf -v command_q '%q' "$*"
  if [[ -n "$SUDO_PASSWORD" ]]; then
    printf '%s\n' "$SUDO_PASSWORD" |
      ssh "$KV260" "sudo -S -p '' bash -lc $command_q"
  else
    ssh "$KV260" "sudo bash -lc $command_q"
  fi
}

remote_sudo "mkdir -p /lib/firmware/xilinx/$APP && \
  cp $REMOTE/binary_container_1.bin $REMOTE/pl.dtbo $REMOTE/shell.json \
     /lib/firmware/xilinx/$APP/ && \
  xmutil unloadapp >/dev/null 2>&1 || true; \
  xmutil loadapp $APP"

echo "deployed to $KV260:$REMOTE (app=$APP)"

# After board testing, switch back to the quiet starter app so the fan settles.
# Call with RESTORE_STARTER=1 (default) after a smoke run, or RESTORE_STARTER=0
# to leave the gdn bitstream loaded.
RESTORE_STARTER=${RESTORE_STARTER:-1}
if [[ "$RESTORE_STARTER" == "1" ]]; then
  remote_sudo "xmutil unloadapp >/dev/null 2>&1 || true; xmutil loadapp k26-starter-kits"
  echo "restored k26-starter-kits on $KV260 (fan quiet)"
fi
