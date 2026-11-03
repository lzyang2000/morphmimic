#!/usr/bin/env bash
# In the container: start the PC service (as its runService.sh does, log in /data), wait
# for its SDK port, then run the streamer with this script's arguments.
set -euo pipefail

SVC=/opt/apps/roboticsservice
export LD_LIBRARY_PATH="$SVC:$SVC/lib:$SVC/SDK/arm64"
export QT_PLUGIN_PATH="$SVC/plugins"
(cd "$SVC" && exec ./RoboticsServiceProcess) > /data/service.log 2>&1 &

for _ in $(seq 50); do
  (echo > /dev/tcp/127.0.0.1/60061) 2> /dev/null && break
  sleep 0.2
done
(echo > /dev/tcp/127.0.0.1/60061) 2> /dev/null \
  || { echo "[pico] the PC service did not open port 60061; see service.log"; exit 1; }
echo "[pico] PC service up; headset app: connect to this machine's Wi-Fi IP"

exec /opt/venv/bin/python -m deploy.vr_teleop.streamer "$@"
