#!/usr/bin/env bash
# PICO straight to this machine (a robot's own computer, e.g. the G1's Jetson), in Docker:
# builds the image once, then runs the PC service and the streamer; arguments go to the
# streamer:
#   bash deploy/onboard/pico/run.sh --target human [--record /data/session.npz]
# /data in the container is $PICO_DATA here (default ~/pico_data): recordings and the
# service's log. Ctrl-C stops both. Rebuild: docker rmi morphmimic_pico:1.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
IMAGE=morphmimic_pico:1
DATA="${PICO_DATA:-$HOME/pico_data}"

docker image inspect "$IMAGE" > /dev/null 2>&1 || docker build -t "$IMAGE" "$HERE"
mkdir -p "$DATA"
exec docker run --rm -i $([ -t 0 ] && echo -t) --init --network host --name morphmimic_pico \
  -v "$REPO:/repo:ro" -v "$DATA:/data" "$IMAGE" "$@"
