# Onboard: the deploy stack on a robot's own computer

The policy node and the robot node run on the robot's computer, and the PICO headset talks
to it directly over Wi-Fi: no workstation in the loop.

```
PICO headset --Wi-Fi--> robot computer: container morphmimic_pico (PC service -> streamer.py --target human)
                          --ZMQ :5556--> policy_node.py (policy.onnx) <--ZMQ--> <robot>_node.py --> motors
```

| Path | Does |
|---|---|
| `pico/` | the PICO PC service and the human streamer in an Ubuntu 22.04 container, any robot |
| `latency.py` | a running policy node's state-to-action time, no robot |
| `g1/` | the G1's Jetson: setup, run, timing, the computer's quirks ([g1/README.md](g1/README.md)) |

## PICO in a container (`pico/`)

The robot computers run an older Ubuntu than the XRoboToolkit PC service supports. Its
headless arm64 build and the PICO SDK's `libPXREARobotSDK.so` need glibc 2.35 (Ubuntu
22.04), and the G1's Jetson has 2.31. So the container is Ubuntu 22.04 with:

- the service's files, unpacked with `dpkg-deb -x`. Its Qt needs ICU 70, glib,
  double-conversion, pcre2-16 and dbus from apt; the GUI, X11, Wayland and SQL-server
  plugins stay unresolved because the service does not use them.
- a Python 3.11 env with the SDK, numpy, pyzmq and msgpack at the repo's versions.

The image is arm64 only, because the headless service exists only for arm64. It needs
the `docker` group, not sudo.

```bash
bash deploy/onboard/pico/run.sh --target human [--record /data/session.npz]
bash deploy/onboard/pico/run.sh --target human --replay /data/<session>.npz   # no headset
```

- The first run builds the image `morphmimic_pico:1` (about a minute); `docker rmi morphmimic_pico:1`
  makes the next run rebuild it. The repo is mounted read-only at `/repo`, so a code
  change needs no rebuild.
- Host network: the container's ports are the robot's (TCP 63901: the headset connects;
  60061, local: the SDK; 5556: the streamer). In the headset app the PC service's address
  is the robot's Wi-Fi address.
- `/data` is `$PICO_DATA` (default `~/pico_data`): recordings and the service's log
  (`service.log`). Ctrl-C stops the streamer and the service.
- The human streamer only: GMR is not in the image, so the motion interface
  (`--target gmr`) still runs its streamer on a PC (the README's "Teleop").
- If the robot's internal network shares the headset's subnet, the robot answers the
  headset on the wrong port: route the headset to the Wi-Fi (the G1: `g1/README.md`).

## Latency (`latency.py`)

Plays the robot side of the protocol at 50 Hz with a fixed standing state and times each
action's return. Run it beside a policy node (and the streamer, for teleop), never during
a hardware run (it binds the action port):

```bash
.venv/bin/python -m deploy.sim2sim.policy_node deploy/ckpt/<name> --robot g1 \
  --interface human --source teleop &
.venv/bin/python -m deploy.onboard.latency deploy/ckpt/<name> --robot g1
```

The robot node waits 10 ms for each step's action (a late one is dropped: the last target
holds), so its p99 should stay well under that.
