#!/usr/bin/env bash
# From the workstation: put a commit's files (and the Unitree SDK wrapper at that commit's
# submodule pin) on the G1's Jetson, which gets no git checkout, and optionally an export:
#   bash deploy/onboard/g1/push.sh [REV, default main] [EXPORT_DIR]
# The export lands in deploy/ckpt/<its folder name>. Files the commit no longer has are
# deleted on the robot, except the env, the binding's build, the exports and work/.
# ROBOT (default unitree_g1) is the ssh host, DEST (default morphmimic) the
# folder under the robot's home.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REV="${1:-main}"
ROBOT="${ROBOT:-unitree_g1}"
DEST="${DEST:-morphmimic}"
WRAPPER=third_party/unitree_sdk2_wrapper

TREE="$(mktemp -d)"
trap 'rm -rf "$TREE"' EXIT
git -C "$REPO" archive "$REV" | tar -x -C "$TREE"
git -C "$REPO/$WRAPPER" archive "$(git -C "$REPO" rev-parse "$REV:$WRAPPER")" \
  | tar -x -C "$TREE/$WRAPPER"
git -C "$REPO" rev-parse --short "$REV" > "$TREE/DEPLOYED_COMMIT"
rsync -a --delete --filter 'P /.venv' --filter 'P /work' --filter 'P /deploy/ckpt/**' \
  --filter "P /$WRAPPER/python_binding/build" "$TREE/" "$ROBOT:$DEST/"
echo "[push] $(cat "$TREE/DEPLOYED_COMMIT") -> $ROBOT:$DEST"
if [ -n "${2:-}" ]; then
  # The repo ships no exports, so a first push has no deploy/ckpt on the robot yet.
  ssh "$ROBOT" mkdir -p "$DEST/deploy/ckpt"
  rsync -a "${2%/}/" "$ROBOT:$DEST/deploy/ckpt/$(basename "${2%/}")/"
  echo "[push] export -> $ROBOT:$DEST/deploy/ckpt/$(basename "${2%/}")"
fi
