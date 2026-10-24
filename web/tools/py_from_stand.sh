#!/usr/bin/env bash
# The Python reference for the live demo's headless check: the CPU sim2sim
# (`python -m deploy.run ... --from-stand 2 --cue-s 1`) for every live clip and robot on one
# interface, one run at a time, each summary written to web/.cache/headless_live/py/.
# deploy.run paces the sim at real time, so all 30 runs take about 4 minutes per robot.
# Usage: web/tools/py_from_stand.sh [interface (motion)] [robots (g1 x2 t2)]
# Inputs (git-ignored): web/.cache/live/policy (the export: policy.onnx, <robot>.json),
# web/.cache/assets (<robot>.xml, meshes/) and web/.cache/live/clips/<robot>/<clip>.npz.
set -euo pipefail
IFACE=${1:-motion}
shift || true
ROBOTS=${*:-g1 x2 t2}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CACHE="$ROOT/web/.cache"
ASSETS="$CACHE/headless_live/assets"
OUT="$CACHE/headless_live/py"
mkdir -p "$ASSETS" "$OUT"
# deploy.run wants one folder with <robot>.xml and clips/<robot>/<clip>.npz; the XML's meshdir
# is relative to the folder it is loaded from.
for f in g1.xml x2.xml t2.xml meshes; do ln -sfn "$CACHE/assets/$f" "$ASSETS/$f"; done
ln -sfn "$CACHE/live/clips" "$ASSETS/clips"
CLIPS=$(node -e 'const c=require(process.argv[1]);console.log(c.clips.map((x)=>x.id).join(" "))' \
  "$ROOT/web/public/live/clips.json")
cd "$ROOT"
for robot in $ROBOTS; do
  for clip in $CLIPS; do
    out="$OUT/${robot}_${IFACE}_${clip}.json"
    [ -s "$out" ] && { echo "skip $robot $clip (done)"; continue; }
    echo "== $robot $IFACE $clip"
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= uv run --group export python -m deploy.run \
      --deploy "$CACHE/live/policy" --assets "$ASSETS" --robot "$robot" --clip "$clip" \
      --interface "$IFACE" --from-stand 2 --cue-s 1 --summary "$out.tmp" > "$OUT/${robot}_${IFACE}_${clip}.log" 2>&1
    mv "$out.tmp" "$out"
    grep -o '"joint_err_rms_rad": [0-9.e-]*\|"falls": [0-9]*' "$out" | tr '\n' ' '; echo
  done
done
