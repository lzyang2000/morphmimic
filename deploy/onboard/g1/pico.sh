#!/usr/bin/env bash
# The G1's PICO side in one command: asks for the headset's IP (192.168.123.<n>, in its
# Wi-Fi settings; Enter keeps the last one), routes it to the Wi-Fi (sudo), then runs the
# PC service and the human streamer (../pico/run.sh); arguments go to the streamer:
#   bash deploy/onboard/g1/pico.sh [--record /data/session.npz]
# The route: the robot's eth0 (its internal network) and wlan0 share 192.168.123.0/24 and
# eth0 wins, so without it the headset's packets arrive on the Wi-Fi and the replies leave
# on eth0. The host routes on wlan0 (earlier headsets) are removed first.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mapfile -t OLD < <(ip -4 route show dev wlan0 | awk '$1 ~ /^192\.168\.123\.[0-9]+$/ {print $1}')
LAST="${OLD[0]:-}"
LAST="${LAST##*.}"
while true; do
  read -rp "headset IP${LAST:+ (Enter: 192.168.123.$LAST)}: 192.168.123." N
  N="${N:-$LAST}"
  [[ "$N" =~ ^[1-9][0-9]{0,2}$ ]] && (( N <= 254 )) && break
  echo "  the last number of the headset's IP, 1-254"
done
HEADSET="192.168.123.$N"
for old in "${OLD[@]}"; do
  sudo ip route del "$old" dev wlan0
done
sudo ip route add "$HEADSET/32" dev wlan0
ip route get "$HEADSET" | head -1
ping -c 1 -W 1 "$HEADSET" > /dev/null && echo "headset $HEADSET answers" \
  || echo "no ping reply from $HEADSET (headset off, another IP, or it ignores pings)"
exec bash "$HERE/../pico/run.sh" --target human "$@"
