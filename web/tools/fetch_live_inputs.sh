#!/usr/bin/env bash
# Copy the live demo's inputs from the export host into web/.cache/live/: the policy export
# (policy.onnx and <robot>.json) and the ten clips' reference .npz files for the three robots.
# Usage: web/tools/fetch_live_inputs.sh
# Required environment (no defaults):
#   SM_LIVE_HOST       ssh destination of the export host (user@host)
#   SM_LIVE_POLICY     remote deploy/export_onnx.py folder (policy.onnx, <robot>.json, SHA256SUMS)
#   SM_LIVE_CLIP_DIRS  two remote clip directories, space-separated, each holding
#                      <robot>/<clip>.npz: the base clips, and the one that supplies the
#                      robot-dance clip
# Remote paths are relative to the remote home directory, or absolute.
set -euo pipefail
usage() { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
for v in SM_LIVE_HOST SM_LIVE_POLICY SM_LIVE_CLIP_DIRS; do
  [ -n "${!v:-}" ] || { echo "error: $v is not set" >&2; usage; }
done
read -r CLIPS_BASE CLIPS_DANCE CLIPS_EXTRA <<<"$SM_LIVE_CLIP_DIRS" || true
if [ -z "${CLIPS_DANCE:-}" ] || [ -n "${CLIPS_EXTRA:-}" ]; then
  echo "error: SM_LIVE_CLIP_DIRS needs exactly two directories" >&2; usage
fi
HOST=$SM_LIVE_HOST
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT="$ROOT/web/.cache/live"
DANCE=victory_dance_robot_dance_180_R_003__A306
BASE_CLIPS="walking_random_direction_R_001__A432 dance_basic_padeburee_270_R_loop_001__A324 dance_basic_turn_v1_360_R_loop_001__A325 dance_hiphop_shuffle_sequence_v_001_R_003__A306 dance_hiphop_hip_hop_i_R_001__A313 dance_latino_jive_sequence_R_002__A313 dance_macarena_003__A467 dance_western_ridding_alone_R_003__A306 dancing_routine_V002_002__A365"

mkdir -p "$OUT/policy"
rsync -a --include='policy.onnx' --include='g1.json' --include='x2.json' --include='t2.json' \
  --include='SHA256SUMS' --exclude='*' "$HOST:$SM_LIVE_POLICY/" "$OUT/policy/"
(cd "$OUT/policy" && sha256sum -c --quiet SHA256SUMS)

for r in g1 x2 t2; do
  mkdir -p "$OUT/clips/$r"
  inc=()
  for c in $BASE_CLIPS; do inc+=(--include="$c.npz"); done
  rsync -a "${inc[@]}" --exclude='*' "$HOST:$CLIPS_BASE/$r/" "$OUT/clips/$r/"
  rsync -a --include="$DANCE.npz" --exclude='*' "$HOST:$CLIPS_DANCE/$r/" "$OUT/clips/$r/"
done
n=$(find "$OUT/clips" -name '*.npz' | wc -l)
[ "$n" -eq 30 ] || { echo "error: expected 30 clip files, found $n" >&2; exit 1; }
echo "fetched the policy export and $n clips into $OUT"
