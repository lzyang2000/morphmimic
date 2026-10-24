#!/usr/bin/env bash
# Freeze a unified_v2 checkpoint on the recording host, export it, and record the 10 demo clips.
# Usage: web/tools/record_remote.sh <iteration|auto> [export|assets|record|fetch|all]
# Run from the web-demo worktree on the workstation. Required environment (no defaults):
#   SM_REC_HOST     ssh destination of the recording host (user@host)
#   SM_REC_REMOTE   remote work directory for the code copy, exports, recordings and logs
#   SM_CKPT_DIR     remote unified_v2 run directory holding <robot>/ckpt_<it>.pt
#   SM_ASSETS_DIRS  two remote export_assets directories, space-separated: the base assets, and
#                   the one that supplies the robot-dance clip
# Remote paths are relative to the remote home directory, or absolute.
set -euo pipefail
usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
for v in SM_REC_HOST SM_REC_REMOTE SM_CKPT_DIR SM_ASSETS_DIRS; do
  [ -n "${!v:-}" ] || { echo "error: $v is not set" >&2; usage; }
done
[ $# -ge 1 ] || usage
IT=$1; STAGE=${2:-all}
HOST=$SM_REC_HOST; REMOTE=$SM_REC_REMOTE; CKPT=$SM_CKPT_DIR
read -r ASSETS_BASE ASSETS_DANCE ASSETS_EXTRA <<<"$SM_ASSETS_DIRS" || true
if [ -z "${ASSETS_DANCE:-}" ] || [ -n "${ASSETS_EXTRA:-}" ]; then
  echo "error: SM_ASSETS_DIRS needs exactly two directories" >&2; usage
fi
q() { printf %q "$1"; }
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CLIPS="walking_random_direction_R_001__A432 dance_basic_padeburee_270_R_loop_001__A324 dance_basic_turn_v1_360_R_loop_001__A325 dance_hiphop_shuffle_sequence_v_001_R_003__A306 dance_hiphop_hip_hop_i_R_001__A313 dance_latino_jive_sequence_R_002__A313 dance_macarena_003__A467 dance_western_ridding_alone_R_003__A306 dancing_routine_V002_002__A365 victory_dance_robot_dance_180_R_003__A306"

if [ "$IT" = auto ]; then   # newest iteration shared by all three runs and older than 60 s
  IT=$(ssh "$HOST" "CKPT=$(q "$CKPT") bash -s" <<'R'
cd; for r in g1 x2 t2; do for f in "$CKPT"/$r/ckpt_*.pt; do
  [ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt 60 ] && basename "$f" .pt | sed 's/ckpt_//'; done | sort -u; done | sort -n | uniq -c | awk '$1==3{print $2}' | tail -1
R
)
  echo "frozen iteration $IT"
fi

if [ "$STAGE" = export ] || [ "$STAGE" = all ]; then
  ssh "$HOST" "cd && mkdir -p $(q "$REMOTE")/morphmimic-web"
  rsync -a --delete --exclude .venv --exclude .git --exclude node_modules --exclude .cache "$ROOT/" "$HOST:$REMOTE/morphmimic-web/"
  ssh "$HOST" "REMOTE=$(q "$REMOTE") CKPT=$(q "$CKPT") IT=$(q "$IT") bash -s" <<'R'
set -e; export PATH=$HOME/.local/bin:$PATH
cd; CKPT=$(cd "$CKPT" && pwd); cd "$REMOTE"; W=$(pwd); cd "$W/morphmimic-web"
uv sync --group export
CUDA_VISIBLE_DEVICES= uv run --group export python -m deploy.export_onnx "$CKPT" "$W/sm_web_policy_$IT" --iteration "$IT"
R
fi

if [ "$STAGE" = assets ] || [ "$STAGE" = all ]; then
  ssh "$HOST" "REMOTE=$(q "$REMOTE") BASE=$(q "$ASSETS_BASE") DANCE=$(q "$ASSETS_DANCE") bash -s" <<'R'
set -e; cd; BASE=$(cd "$BASE" && pwd); DANCE=$(cd "$DANCE" && pwd); cd "$REMOTE"
rm -rf sm_web_assets; cp -r "$BASE" sm_web_assets
for r in g1 x2 t2; do cp "$DANCE/clips/$r/victory_dance_robot_dance_180_R_003__A306.npz" sm_web_assets/clips/$r/; done
R
fi

if [ "$STAGE" = record ] || [ "$STAGE" = all ]; then
  # one-clip invocations, skipping combos whose json exists, so a crash can be resumed by rerunning
  ssh "$HOST" "REMOTE=$(q "$REMOTE") IT=$(q "$IT") CLIPS=$(q "$CLIPS") bash -s" <<'R'
set -e; cd; cd "$REMOTE"; W=$(pwd)
cat > "$W/sm_web_record_$IT.sh" <<S
cd "$W/morphmimic-web"; export PATH=\$HOME/.local/bin:\$PATH
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=
for c in $CLIPS; do for r in g1 x2 t2; do for i in motion keypoint; do
  [ -f "$W/sm_web_rec_$IT/\$c/\${r}_\$i.json" ] && continue
  uv run python web/tools/record_clips.py --deploy "$W/sm_web_policy_$IT" --assets "$W/sm_web_assets" --clips \$c --robots \$r --interfaces \$i --out "$W/sm_web_rec_$IT" || echo "CRASH \$c \$r \$i"
done; done; done
echo ALLDONE
S
setsid nohup bash "$W/sm_web_record_$IT.sh" > "$W/sm_web_record_$IT.log" 2>&1 < /dev/null &
R
  echo "recording started; poll $REMOTE/sm_web_record_$IT.log on $HOST for ALLDONE"
fi

if [ "$STAGE" = fetch ]; then
  C="$ROOT/web/.cache"; mkdir -p "$C/recordings" "$C/assets"
  rsync -a "$HOST:$REMOTE/sm_web_rec_$IT/" "$C/recordings/"
  rsync -a --include='g1.xml' --include='x2.xml' --include='t2.xml' --include='meshes/***' --exclude='*' "$HOST:$REMOTE/sm_web_assets/" "$C/assets/"
  echo "unified_v2 $IT" > "$C/policy.txt"
fi
