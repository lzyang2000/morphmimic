#!/usr/bin/env bash
# Prepare hardware videos for the project page: strip metadata and audio, optionally mirror,
# scale (720 wide for portrait sources, 1280 wide for landscape), encode H.264 yuv420p with
# faststart at CRF 24, and save a poster frame (JPEG, -q:v 3) at 1.5 s of the result.
#
# Usage: web/tools/prep_hardware.sh [--out DIR] [--mirror] INPUT NAME [[--mirror] INPUT NAME ...]
#   --out DIR  output directory (default: web/.cache/hardware next to this script)
#   --mirror   flip the next INPUT horizontally
# Writes DIR/NAME.mp4 and DIR/NAME_poster.jpg for each INPUT. The page's three videos were made with
#   web/tools/prep_hardware.sh <upper-body motion source> g1_hardware_motion \
#     --mirror <walk source> g1_hardware_walk <teleoperation source> g1_hardware_teleop
# deploy_ghpages.mjs then copies them to static/videos/.
set -euo pipefail
usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
OUT="$(cd "$(dirname "$0")/.." && pwd)/.cache/hardware"
if [ "${1:-}" = --out ]; then
  [ $# -ge 2 ] || usage
  OUT=$2; shift 2
fi
[ $# -ge 2 ] || usage
command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null || { echo "error: needs ffmpeg and ffprobe" >&2; exit 1; }
mkdir -p "$OUT"

while [ $# -gt 0 ]; do
  MIRROR=0
  if [ "$1" = --mirror ]; then MIRROR=1; shift; fi
  [ $# -ge 2 ] || usage
  IN=$1; NAME=$2; shift 2
  [ -f "$IN" ] || { echo "error: no such file: $IN" >&2; exit 1; }
  # Displayed size (ffprobe applies no rotation here; the sources carry none).
  IFS=, read -r W H < <(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
                          -of csv=p=0 "$IN")
  if [ "$H" -gt "$W" ]; then SCALE="scale=720:-2"; else SCALE="scale=1280:-2"; fi
  VF=$SCALE
  [ "$MIRROR" = 1 ] && VF="hflip,$SCALE"
  ffmpeg -y -loglevel error -i "$IN" -map_metadata -1 -map 0:v:0 -an -vf "$VF" \
    -c:v libx264 -preset slow -crf 24 -pix_fmt yuv420p -movflags +faststart "$OUT/$NAME.mp4"
  ffmpeg -y -loglevel error -ss 1.5 -i "$OUT/$NAME.mp4" -frames:v 1 -q:v 3 "$OUT/${NAME}_poster.jpg"
  echo "$NAME: ${W}x${H} -> $(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
    -of csv=p=0:s=x "$OUT/$NAME.mp4")$([ "$MIRROR" = 1 ] && echo ', mirrored')"
done
