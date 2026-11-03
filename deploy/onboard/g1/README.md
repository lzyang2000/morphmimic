# The G1 from its own computer

The G1's onboard Jetson runs the whole teleop loop: the PICO container (`../pico`,
[../README.md](../README.md)), the policy node and `g1_node`.

| File | Where | Does |
|---|---|---|
| `push.sh` | your PC | copies a commit's files (and optionally an export) to the robot, which needs no git checkout |
| `install.sh` | robot | builds the deploy env `.venv` and the Unitree SDK binding |
| `pico.sh` | robot | asks for the headset's IP, routes it to the Wi-Fi (sudo), runs `../pico/run.sh` |
| `check_link.py` | robot | read-only check: motion mode, IMU, every encoder, the remote |

## The computer

- The G1's development computer is a Jetson Orin: Ubuntu 20.04 / JetPack 5.1
  (glibc 2.31), aarch64, 8 cores, user `unitree` at 192.168.123.164 on `eth0` (the
  robot's internal network). Join it to your Wi-Fi (`wlan0`), the same network as the
  headset. That is also how it reaches the internet during setup.
- The deploy env is not `uv sync`, since the project's dependencies are the training
  stack (torch, mjlab). The robot's `.venv` has only what the policy node and `g1_node`
  import: numpy, onnx, onnxruntime, pyzmq, msgpack, scipy and mujoco.
- The policy runs on the CPU (onnxruntime). JetPack 5's TensorRT and NVIDIA's torch
  wheels exist only for the system Python 3.8, and the CPU is fast enough
  ([Timing](#timing)).
- The user needs to be in the `docker` group for the PICO container. Only the headset
  route needs sudo.

## Setup

From your PC, with an ssh host `unitree_g1` for the robot (or set `ROBOT=<host>`) and
the submodule `third_party/unitree_sdk2_wrapper` checked out:

```bash
bash deploy/onboard/g1/push.sh main <export_dir>   # lands in ~/morphmimic, the export in deploy/ckpt/<its name>
```

On the robot, once (and again after a push that changes the SDK wrapper):

```bash
cd ~/morphmimic && bash deploy/onboard/g1/install.sh   # a few minutes, no sudo
```

In the headset app, set the PC service's address to the robot's Wi-Fi address
(`ip -br addr show wlan0`).

## Run

Hang the robot first: `g1_node` releases Unitree's motion mode as it starts, before
START, and the robot goes limp. Run in tmux on the robot, so a dropped ssh does not take
the node down:

```bash
cd ~/morphmimic
.venv/bin/python -m deploy.onboard.g1.check_link          # every encoder sane? tilt?
# Window 1: the headset's route, the PC service and the streamer (the first run builds
# the image, ~1 min).
bash deploy/onboard/g1/pico.sh [--record /data/session.npz]
# Window 2: the policy node and g1_node.
.venv/bin/python -m deploy.run --deploy deploy/ckpt/<name> --robot g1 --hardware g1 \
  --net eth0 --teleop --interface human --human-points full [--record ~/pico_data/ticks.npz]
```

`pico.sh` asks for the last number of the headset's IP (in the headset's Wi-Fi
settings; Enter keeps the last one) and for the sudo password. It needs the route
because when the robot's `eth0` and `wlan0` share a subnet, `eth0` wins and the
headset's replies leave on the wrong port. The script removes earlier host routes on
`wlan0`, adds a route for the headset, shows it and pings the headset (no reply can also
mean the headset ignores pings).

Then:

1. On the remote, press **START** (the robot moves to the stand over 2 s), then **A**
   (the policy runs and holds the stand until you calibrate).
2. On the headset, stand upright and still and press **A+B+X+Y** (IDLE).
3. Stand roughly as the robot stands and press **A+X** (POSE: the robot follows you).
4. **A+X** again returns to IDLE.

The remote's **B** damps, and **SELECT** damps and exits. The headset's A+B+X+Y does not
damp the robot; it goes back to the stand. The main README's
[Teleop](../../../README.md#4-teleop) lists every headset control.
`--human-points five` runs the five-point setting (pelvis, wrists, ankles).

A policy trained on `g1_amber` (a G1 with weak waist motors) runs the same way with
`--robot g1_amber`; the export's `g1_amber.json` brings its gains.

**The streamer can run anywhere.** Everything on the robot, as above, is the default.
The PICO service and the streamer can also run on any machine on the robot's network:
run the PC service and `deploy.vr_teleop.streamer --target human` there, point the
headset app at that machine, skip `pico.sh`, and add `--streamer tcp://<its
address>:5556` to the robot's command. This frees the Jetson's CPU and shortens the
policy's command latency ([Timing](#timing)). A streamer that keeps running between
robot runs also keeps its mode and calibration. Before each run, restart it, or press
A+B+X+Y and recalibrate; otherwise the next run starts in POSE with the old calibration.

`--record` saves every policy tick: the measured joint positions and velocities, the
pelvis IMU, the joint targets sent, whether each step's action arrived in time, and when
the state was read and the command sent (`t_state`, `t_sent`). Keep recordings outside
the deploy folder: `push.sh` deletes files there that are not in git (except `.venv`,
`work/` and `deploy/ckpt/`).

## Timing

A policy step with a human-interface export takes about 4.3 ms on the Jetson's CPU:
2.3 ms for ONNX and 1.9 ms for the goal and features. The next step's reference window
and human features (about 1.2 ms) are built right after each action goes out, so they
are not on the path from state to command.

On our G1, the time from state read to command sent (`t_sent - t_state`) has these
medians:

| Setup | Median |
|---|---|
| Policy and a replayed session on the Jetson | 4.3 to 4.7 ms |
| Policy, PICO service and live streamer on the Jetson | 4.9 ms (p99 7.1) |
| A PC driving the robot | 2.1 ms |

None of these comes near `g1_node`'s 10 ms wait; an action later than that is dropped
and the last target holds.

A policy trained with the delay redrawn every physics step kept stepping after a step at
about 5 ms. A policy trained with the default held delay (see the main README's Train
notes) does not.

To keep the step short, the policy node runs only the part of the graph that computes
the action (onnxruntime would otherwise run the whole graph) on 4 threads. The default
thread pool busy-waits on every core after each call, which also slows `g1_node` and the
SDK. Keep the Jetson otherwise idle while the robot runs.
