# MorphMimic: Morphology-Conditioned World Model and Policy Sharing Across Humanoids

Lizhi Yang, Fengze Xie\*, Yuyang Wang\*, Lu Gan, Yisong Yue, Aaron D. Ames

\* Equal contribution.

[[Website]](https://lzyang2000.github.io/morphmimic/)
[[Paper]](https://lzyang2000.github.io/morphmimic/static/morphmimic.pdf)
[[Video]](https://lzyang2000.github.io/morphmimic/static/videos/morphmimic.mp4)
[[Live demo]](https://lzyang2000.github.io/morphmimic/demo/)

**v0.1:** a human interface (the policy follows a PICO headset's body tracking
directly, with all 24 joints or five points), a [pretrained policy](#pretrained-policy)
for the G1, X2 Ultra and T2, `g1_amber` for a G1 with weak waist motors, a held
actuator delay in training for robustness to latency on hardware, teleop running
entirely on the G1's own computer ([`deploy/onboard/`](deploy/onboard/README.md)), and
updated X2 Ultra and T2 models (vendor gains, Booster's motor model).

> **Disclaimer.** MorphMimic builds on SuperMimic (Liao, Shao, Long, Zakka, Horowitz,
> Abbeel and Sreenath, UC Berkeley; [citation below](#citation)). The base algorithm is
> our own implementation of that paper, written from the paper alone. It is not the
> authors' code, and they have not reviewed or endorsed it or our extensions. Any
> mistakes are ours.

SuperMimic trains a humanoid motion-tracking policy by backpropagating through a
learned, differentiable world model. MorphMimic trains several robots at once: the
Unitree G1, the AgiBot X2 Ultra and the Booster T2, simulated in
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp). The world model, the policy,
or both can be shared between the robots. Every robot is mapped to the G1's 29-joint
layout, and the networks get a morphology vector (gains, limits, limb lengths, masses)
that tells them which body they are driving.

The policy takes two kinds of goal:

- **motion**: a full reference motion of the robot (all joints and bodies).
- **human**: a human's motion, given as the 24 SMPL joints of a uniform skeleton over the
  next 10 frames. It can use all 24 joints (`full`) or only the pelvis, wrists and ankles
  (`five`, what a VR headset with two controllers and two ankle trackers gives you).
  A PICO headset's body tracking goes straight into the policy, with no retargeting.
  The human encoder is distilled from the motion interface and feeds the same decoder,
  so one policy serves both.

**No state estimation.** The policy reads only the IMU (orientation and angular
velocity) and the joint encoders. Its goals do not depend on where the robot is: the
motion goal's path is in the reference's own heading frame, and the human skeleton is in
its own heading frame plus the human chest's orientation relative to the robot's torso.
You do not need odometry, motion capture or a base-velocity estimator. Only the critics,
which are used only in training, see the robot's offset from the reference.

If you do have the robot's pose (from motion capture or a state estimator), the
[anchor mode](#anchor-mode) gives the motion goal relative to the robot itself, so the
policy corrects its drift from the reference path.

## Layout

```
src/morphmimic/
  data/          SEED CSV -> 50 Hz npz, SOMA BVH -> human sidecars, clip filters,
                 subsets, GPU motion library
  env/           mjlab tracking env (multi-clip command, reset sampling, observations)
  model/         state layout, analytic reward heads, world model, policy, critic,
                 human goal features
  robots/        G1, X2 Ultra and T2: registry, translator to the 29-joint layout,
                 morphology vector, vendored X2 and T2 models and their actuators
  train/         replay buffer, collector, imagination, trainer, multi-robot sharing
  eval/          benchmark, drift diagnostic, policy loader
  scripts/       convert_seed, convert_human, check_human, build_subsets, train, eval,
                 drift, play, render
tools/soma_retargeter/   retargets the SEED motions to the X2 and the T2
deploy/
  export_onnx.py         training checkpoint -> policy.onnx + <robot>.json
  export_ckpt.py         training checkpoint -> inference-only checkpoint
  run.py                 one-command deployment: the policy node plus the sim or the G1
  sim2sim/               policy node (onnxruntime) + CPU MuJoCo sim node, over ZMQ
  vr_teleop/             PICO headset reader, human skeleton and GMR targets, streamer
  hardware/              the real G1 on the same protocol as the sim, safety layer
  onboard/               everything on the G1's own computer, PICO in a container
  tools/                 headset-free test sessions for teleop
```

## Setup

You need an NVIDIA GPU for training and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recursive git@github.com:lzyang2000/morphmimic.git && cd morphmimic
uv sync                                   # Python 3.11, mjlab 1.6.0, torch (cu128)
uv sync --group export --group teleop     # adds onnxruntime, pyzmq and GMR for deployment
```

`warp-lang` is pinned below 1.18 because 1.18 needs a CUDA 13 driver. Drivers with
CUDA 12.9 (575 or newer) work.

## Data

Training uses the [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed)
motion dataset. The commands below use `$DATA` for wherever you keep it.

**G1.** Download `g1/csv/<session>/*.csv` and `seed_metadata_v003.csv`. Then convert the
clips to 50 Hz npz and build the subsets (about 5 minutes with 96 workers, about 91 GB
on disk):

```bash
uv run python -m morphmimic.scripts.convert_seed \
  --csv-dir $DATA/seed/g1/csv --metadata $DATA/seed/seed_metadata_v003.csv \
  --out-dir $DATA/seed_g1_npz50 --workers 96
uv run python -m morphmimic.scripts.build_subsets \
  --manifest $DATA/seed_g1_npz50/manifest.csv --out-dir $DATA/seed_g1_npz50/subsets
```

This writes `subsets/{tiny,1h,7h,40h,all}.yaml`. The filters keep 129,398 clips
(262.8 h).

**Human sidecars** (for the human interface). Download SEED's SOMA uniform-skeleton BVH
files (`soma_uniform/bvh/<session>/*.bvh`) and convert them to 50 Hz sidecars of the 24
SMPL joints, one per clip, in `human50/` next to the clips. All three robots use the
same clips, so the G1's sidecars serve the X2 and the T2 too:

```bash
uv run python -m morphmimic.scripts.convert_human --npz-root $DATA/seed_g1_npz50 \
  --bvh-root $DATA/seed/soma_uniform/bvh --workers 96
uv run python -m morphmimic.scripts.check_human \
  --subset-yaml $DATA/seed_g1_npz50/subsets/1h.yaml      # sidecars against their clips
```

**X2 Ultra and T2.** SEED has no retarget for these robots, so we retarget the SOMA BVH
files with NVIDIA's [SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter)
(commit `cbdb2b4`). The scripts in `tools/soma_retargeter/` handle the steps around it:

1. `stage_subset.py` links the BVH files of a G1 subset into one folder.
2. `assemble_robots.py` builds the retargeter's robot folder: its own X2 target with
   an effector fix, and our hand-written T2 target, both with the simulator's soft
   joint limits.
3. Fill the `__IMPORT__`, `__EXPORT__` and `__ROBOTS__` placeholders in `batch_x2.json`
   or `batch_t2.json` with absolute paths, and run the retargeter's
   `app/bvh_to_csv_converter.py --config <that file>`. For the full set, split the work
   with `shard_stage.py` and `launch_shards.sh`, then check the output with
   `validate_csv.py`.
4. Convert the CSVs, and build the subsets from the G1's clip lists so every robot has
   the same clips:

```bash
for r in x2 t2; do
  uv run python -m morphmimic.scripts.convert_seed --robot $r \
    --csv-dir $DATA/seed_${r}_csv --metadata $DATA/seed/seed_metadata_v004.csv \
    --out-dir $DATA/seed_${r}_npz50 --workers 64
  uv run python -m morphmimic.scripts.build_subsets \
    --manifest $DATA/seed_${r}_npz50/manifest.csv --out-dir $DATA/seed_${r}_npz50/subsets \
    --clips-from $DATA/seed_g1_npz50/subsets/all.yaml
done
```

## Train

The defaults follow SuperMimic's Table A.5: 4096 envs per GPU, H=32 imagined steps,
k=8, γ=0.99, λ=0.95, learning rate 2e-4 with cosine decay. Multi-robot runs put one
robot on each GPU, so they need three GPUs on one machine.

```bash
G1=$DATA/seed_g1_npz50/subsets/all.yaml
X2=$DATA/seed_x2_npz50/subsets/all.yaml
T2=$DATA/seed_t2_npz50/subsets/all.yaml
COMMON="--kp-interface --kp-student copy --no-kp-preview-mix \
  --human-root $DATA/seed_g1_npz50/human50 --library-dtype float16 \
  --iterations 60000 --ckpt-every 4000 --log-dir logs"
```

`COMMON` is our current recipe. It trains the human interface alongside the motion
interface. The human encoder learns from the motion encoder (`--kp-student copy`): it
copies the motion interface's action and pulls its latent towards the motion latent, on
its own imagined states. Half of its samples see all 24 joints and half see only the
five (`--kp-masking mixed`, the default), and it always sees the full 10-frame preview
(`--no-kp-preview-mix`).

**Unified: one world model and one policy for all three robots.**

```bash
uv run python -m morphmimic.scripts.train --robots g1 x2 t2 \
  --subset-yamls $G1 $X2 $T2 --share all --gpu-ids "[0,1,2]" $COMMON --run-name unified
```

Our own G1 has weak waist roll and pitch motors, so we train with `--robots g1_amber x2
t2` instead (same subset yamls). See [Robot models](#robot-models). The policy we run on
our G1 comes from that run.

**One world model, a policy per robot.**

```bash
uv run python -m morphmimic.scripts.train --robots g1 x2 t2 \
  --subset-yamls $G1 $X2 $T2 --share wm --gpu-ids "[0,1,2]" $COMMON --run-name shared_wm
```

**One policy, a world model per robot.**

```bash
uv run python -m morphmimic.scripts.train --robots g1 x2 t2 \
  --subset-yamls $G1 $X2 $T2 --share policy --gpu-ids "[0,1,2]" $COMMON --run-name shared_policy
```

**Single robot** (its own world model and policy, one GPU).

```bash
uv run python -m morphmimic.scripts.train --robot g1 --subset-yaml $G1 --gpu 0 \
  $COMMON --run-name single_g1        # or --robot x2 / t2 with $X2 / $T2
```

Notes:

- `--kp-interface` adds the human interface and needs the human sidecars
  (`--human-root`). Leave out the four human flags to train the motion interface only.
- `--kp-drift` and `--kp-jitter` add slow drift and per-joint noise to the human
  encoder's input. Both are off by default because a three-robot policy trained with
  them chattered on our real G1.
- A multi-robot run writes each robot's checkpoints to `logs/<run>/<robot>/`
  (`ckpt_<it>.pt` and `latest.pt`).
- To resume, rerun the same command with `--resume logs/<run>/latest.pt` (single robot)
  or `--resume logs/<run>` (multi-robot).
- For a quick try, use the 1 h subset (`subsets/1h.yaml`) and drop `--library-dtype`.
- Memory at 4096 envs: about 28 GB per GPU on the 1 h subset and 51 to 58 GB on the full
  set. With the full set, each robot's process also needs about 120 to 130 GB of host
  RAM.
- The first iterations compile the updates into CUDA graphs, which takes a few minutes.
  `--compile none` runs eagerly instead.
- By default, training adds an actuator delay of 0 to 20 ms, drawn once per episode and
  then held, because a real robot's latency is steady (`--actuator-delay-max-ms`,
  `--actuator-delay-fixed`). With the earlier default, 0 to 40 ms redrawn every physics
  step (`--no-actuator-delay-fixed`), our three-robot policy kept stepping after a step
  on the real G1 once its command latency rose by a few ms. `--no-actuator-delay` turns
  the delay off.
- The recipe stops at 60,000 iterations. The cosine decay still runs over its default
  250,000 (`--lr-decay-iters`), so the learning rate is still high there. Our later
  checkpoints of the same run (78,000 and on) swayed while standing on the G1, and the
  training metrics did not show it.
- `--wandb` logs to Weights & Biases, using `WANDB_ENTITY` and `WANDB_PROJECT` (default
  `morphmimic`) from the environment.
- `uv run python -m morphmimic.scripts.train --help` lists every option.

### Anchor mode

By default the motion goal ignores where the robot is, so the policy cannot correct
drift from the reference path. `--goal-anchor robot` gives the reference anchor (torso)
position and orientation relative to the robot's own anchor instead. Drift then shows up
in the goal and the policy steers back onto the path. The cost is that the policy needs
the robot's world position and heading at deploy time, from motion capture or a state
estimator.

```bash
uv run python -m morphmimic.scripts.train --robots g1 x2 t2 \
  --subset-yamls $G1 $X2 $T2 --share all --gpu-ids "[0,1,2]" \
  --goal-anchor robot --library-dtype float16 --iterations 40000 --ckpt-every 4000 \
  --log-dir logs --run-name unified_anchor
```

- The anchor mode trains the motion interface only. It refuses `--kp-interface`, because
  the human goals have no position.
- A checkpoint keeps its anchor mode. Resuming or exporting it with the other mode fails
  with a message naming the flag to use.
- Export with `--goal-anchor robot` (step 1 below); the anchor goes into each
  `<robot>.json`. Run it with `--interface motion`.
- The sim node sends the floating-base position and yaw with every state message, so
  the anchor mode works in sim2sim as is. On hardware the pose comes from a `PoseSource`
  (`deploy/hardware/pose_source.py`). The G1 node has none yet, so it sends no pose and
  the policy node stops rather than act without one.

## Evaluate

The benchmark plays every clip of a subset once from its first frame, with a
deterministic policy and no randomisation, and writes a per-clip table (completion,
tracking errors, drift):

```bash
uv run python -m morphmimic.scripts.eval --ckpt logs/unified/g1/ckpt_60000.pt \
  --subset-yaml $G1 --num-envs 1024                             # the motion interface
uv run python -m morphmimic.scripts.eval --ckpt logs/unified/g1/ckpt_60000.pt \
  --subset-yaml $G1 --interface human --human-points full       # or five
```

`scripts.play` and `scripts.render` show a checkpoint on a clip in mjlab; `--help` lists
their options.

## Robot models

The X2 Ultra runs AgiBot's own joint gains with MuJoCo's built-in PD. The T2 runs
Booster's deployment gains with a model of Booster's motors, whose torque falls off with
speed as in Booster's own training. The earlier release of this repository (commit
`aa40fe4`) used a DC-motor model for the X2 and other gains for the T2, so its numbers
will not reproduce exactly with this code.

`g1_amber` is the G1 with the waist roll and pitch damping raised from 1.81 to 3.0, for
a robot with weak waist motors (like ours). Nothing else changes: the body, the
reference motions and the deployment are the G1's. On the robot, run it with `--robot
g1_amber --hardware g1`.

## Deploy

Deployment does not need the training stack. A policy process (numpy, onnx and
onnxruntime only) and a robot process talk over ZMQ. The robot process is either CPU
MuJoCo (sim2sim) or the real G1 (sim2real); the policy process is the same for both.

### Pretrained policy

The policy we run on our G1 is attached to the
[v0.1 release](https://github.com/lzyang2000/morphmimic/releases/tag/v0.1). It comes
from the unified recipe with `--robots g1_amber x2 t2`, at 60,000 iterations, exported
as in step 1. The one `policy.onnx` drives the G1 (as `g1_amber`), the X2 Ultra and the
T2, through either interface.

```bash
curl -LO https://github.com/lzyang2000/morphmimic/releases/download/v0.1/morphmimic_policy.zip
unzip morphmimic_policy.zip
(cd morphmimic_policy && sha256sum -c SHA256SUMS)
W=morphmimic_policy
```

Use `$W` wherever the steps below use an export, and skip step 1. Pass `--robot
g1_amber` for the G1 (`--robot g1_amber --hardware g1` on the robot), or `--robot x2` /
`--robot t2`.

The policy is provided as is, with no warranty and no support. We have tested it on a
real G1 only, where we picked this checkpoint; your G1 may behave differently. On the X2
Ultra and the T2 it has run only in the sim: their sim2real tests are waiting for the
robots. Start any robot hung or held, and keep the remote ready to damp it.

### 1. The policy

Export a trained run to ONNX. A multi-robot run gives one `policy.onnx` for all three
robots and both interfaces, plus one `<robot>.json` per robot (its normalisers,
morphology vector and constants):

```bash
uv run --group export python -m deploy.export_onnx logs/unified out/unified --iteration 60000
```

A single-robot checkpoint (`logs/single_g1/ckpt_60000.pt`) exports the same way. Before
writing anything, the export checks that the ONNX model reproduces the PyTorch actions.

### 2. Simulation assets (once per robot)

This writes the robot's MuJoCo model and the reference clips you want to play. It needs
mjlab and a GPU; everything after it runs on the CPU.

```bash
uv run python -m deploy.sim2sim.export_assets --robot g1 \
  --subset-yaml $DATA/seed_g1_npz50/subsets/1h.yaml \
  --clips walk_ff_loop_315_R_001__A257 --out assets
```

`--clips` takes any clip names in that subset. Use the robot's own subset
(`seed_x2_npz50` for `--robot x2`, and so on).

### 3. Play a motion

```bash
W=out/unified                   # the export from step 1
uv run --group export python -m deploy.run --deploy $W --assets assets \
  --robot g1 --clip walk_ff_loop_315_R_001__A257 --interface motion --viewer native
```

- `--interface motion` tracks the whole reference. `--interface human --human-points
  full` (or `five`) tracks the clip's human motion instead; pass its sidecar with
  `--human-clip $DATA/seed_g1_npz50/human50/<session>/<clip>.npz`.
- `--viewer native` opens MuJoCo's viewer. Without a display, use
  `MUJOCO_GL=egl ... --video out.mp4`.
- `--delay-ms 20` adds actuator delay.
- At the end, the run prints whether the clip completed, the joint and body errors, and
  the drift.

### 4. Teleop

You need a PICO 4 (or 4 Pro) headset with its two controllers, two PICO Motion Trackers
on your ankles, and a Linux PC on the same Wi-Fi.

One-time setup:

1. `bash deploy/install_pico.sh` builds the PICO SDK binding into `.venv`. After that,
   use `uv sync --group teleop --inexact`, because a plain `uv sync` removes the SDK.
2. On the PC, install and start the
   [XRoboToolkit PC service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases)
   (a `.deb`, needs sudo).
3. On the headset, enable developer mode and install the
   [XRoboToolkit app](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases).
   Pair and calibrate the ankle trackers in PICO's Motion Tracker app.
4. In XRoboToolkit, enter the PC's IP address, tick Head and Controller, select Send,
   and set Pico Motion Tracker to Full body.

Then run the streamer and the sim in two terminals:

```bash
# Terminal 1: headset -> the operator on the uniform skeleton at 50 Hz (no retargeting)
uv run --group teleop python -m deploy.vr_teleop.streamer --target human

# Terminal 2: policy and sim, the human interface
uv run --group export --group teleop python -m deploy.run --deploy $W \
  --assets assets --robot x2 --teleop --interface human --human-points full --viewer native
```

`--human-points five` uses only your pelvis, wrists and ankles. For the motion
interface, the streamer retargets you to the robot with GMR instead
(`streamer --robot x2`, then `--interface motion`).

Controls (A and B on the right controller, X and Y on the left):

| Press | Mode | What happens |
|---|---|---|
| A+B+X+Y | OFF | Calibrate and go to IDLE. Stand upright and still. |
| A+X | IDLE | POSE: the robot follows you. Stand roughly as the robot stands first: the robot blends into your pose over 0.5 s. |
| A+X | POSE | Back to IDLE: the robot returns to its standing pose. |
| hold left Menu | POSE | Pause: the robot holds its current pose. Release to continue. |
| A+B+X+Y | IDLE or POSE | Stop: back to OFF. The robot returns to its standing pose and ignores you until you calibrate again. |

The robot never damps from the headset. In OFF and IDLE it holds its standing pose; only
the robot's remote damps it ([The real G1](#5-the-real-g1)). A button combo fires after
it is held for 0.1 s, once per press.

When you enter POSE, your heading is aligned to the robot's. Where you stand in the room
does not matter. If the stream stops for 3 s, the robot goes to IDLE. With GMR
(`--target gmr`), `--human-height 1.7` on the streamer sets your height for the
retargeting (default 1.8 m); the human target measures you at calibration instead.

**Without a headset.** Record a session once with `streamer.py --record session.npz`
and replay it for any robot and either interface (`--replay session.npz`). You can also
make a GMR session from a G1 clip:

```bash
uv run --group teleop python -m deploy.tools.make_gmr_session wave.npz \
  --clip $DATA/seed_g1_npz50/230713/wave_R_001__A430.npz
MUJOCO_GL=egl uv run --group export --group teleop python -m deploy.run \
  --deploy $W --assets assets --robot t2 --teleop --replay wave.npz \
  --max-steps 700 --video t2_wave.mp4
```

The generated session calibrates at 1 s, enters POSE at 4 s, plays the clip, then
returns to IDLE.

### 5. The real G1

In our paper, the shared policy from the 40,000-iteration unified run, exported as in
step 1, dances, walks and follows a teleoperator on a physical G1.

By default, everything runs on the G1's own computer and the headset talks to the robot
directly; see [`deploy/onboard/`](deploy/onboard/README.md). The streamer (the PICO
service and `streamer.py`) can also run on any machine on the robot's network, and the
policy node reads it with `--streamer tcp://<host>:5556`. The steps below drive the
robot from a PC on its network instead.

Once, on the PC connected to the robot:

```bash
bash deploy/install_unitree_sdk.sh   # builds the Unitree SDK binding into .venv (sudo for apt)
```

Then, with the robot powered on and hung or held, add `--hardware g1 --net <iface>` to
the same command. Start with neither `--clip` nor `--teleop`, so the policy tracks the
neutral stand. `--max-steps 1500` ends each policy run after 30 s:

```bash
uv run --group export --group teleop python -m deploy.run --deploy $W \
  --robot g1 --hardware g1 --net <iface> --interface motion --max-steps 1500
```

Add `--assets assets --clip <name>` to play a motion, or `--teleop --interface human
--human-points full` to follow the headset (with the streamer on `--target human`).
`--record ticks.npz` saves every policy tick: the measured joint positions and
velocities, the IMU, and the joint targets sent. The robot's remote controls the node:

| Button | What happens |
|---|---|
| START | From damping, move to the standing pose over 2 s and hold it. |
| A | Start the policy. Without teleop it holds the neutral stand, facing the robot's heading. |
| Y | With a clip, while the policy stands: play the clip, then return to the stand. |
| B | Damp at the current pose (START stands again). |
| SELECT | Damp and exit (Ctrl-C does the same). |

- On hardware a clip always starts from the stand. Each Y turns the clip to the robot's
  heading and blends into its first frame over 2 s (`--from-stand S` changes this).
  After the clip, the robot blends back to the stand in 1 s. A Y while a clip plays is
  ignored. Pick a clip that starts and ends standing still.
- The gains are the training gains from `g1.json`. `--kp-scale` scales only the
  stiffness.
- If no action arrives for 1 s, the robot does not damp: the targets blend to the PD
  stand, and the policy restarts on the stand and takes over again with its first
  action.
- After `--max-steps N` steps, the robot blends back to the PD stand over 2 s and holds
  it. A runs the policy again.
- Keep the policy machine otherwise idle. A busy CPU can stall the policy.
- The policy can run on another machine: run `policy_node.py --bind-host 0.0.0.0
  --robot-host <robot pc>` there, and `g1_node.py --policy-host <policy pc>
  --bind-host 0.0.0.0` on the robot side.

To try a hardware clip in the sim first, pass `--from-stand 2`. The robot starts on the
stand, facing `--start-yaw` degrees, and Y is pressed at each `--cue-s` time. The run
ends 2 s after the last clip:

```bash
uv run --group export --group teleop python -m deploy.run --deploy $W --assets assets \
  --robot g1 --clip <name> --interface motion --from-stand 2 --start-yaw 137 \
  --cue-s 1 --viewer native
```

## Differences from SuperMimic

The base method is SuperMimic's. These are the places where we filled a gap or chose
differently:

- The world model's reward and observation heads are computed analytically, which the
  paper allows. Its termination head is the only learned part. By default it is a
  four-parameter monotone hazard on the height and tilt margins (`--termination`:
  `hazard`, an MLP with `learned`, or the fixed analytic `soft`).
- The world model works in the pelvis frame and also gets the gravity direction and
  body heights as inputs, as in SuperTrack (`--wm-frame heading` for the yaw-only frame
  without them).
- The critic and the world model have LayerNorm after each hidden layer
  (`--no-critic-layer-norm`, `--no-wm-layer-norm`), and the critic's targets are clamped
  at R_max / (1 - γ) = 10, the largest possible return (`--no-critic-target-clip`).
  Without these, our full-set runs diverged.
- The first 500 iterations train only the world model (`--wm-only-iters 0` to skip).
  Imagined rollouts start only where they cannot reach the end of a clip
  (`--no-avoid-clip-end`).
- Episodes end at the last frame of their clip or after 10 s, whichever comes first.
  The reset sampler therefore normalises each bin's return by the best return reachable
  in that time (`--reset-norm actual` normalises by the steps actually run).
- The MLPs run in bfloat16 autocast, about 1.5 times faster than TF32 in our runs
  (`--no-bf16`).
- Goals do not depend on the robot's position, so the policy runs from the IMU and joint
  encoders alone. This is built into the code, not a flag.
- Training randomises the actuator delay (0 to 20 ms, drawn once per episode and held)
  for robustness to latency on the real robot (`--no-actuator-delay`).

`train --help` lists every option.

## License

The code is under the [Apache License 2.0](LICENSE). The vendored robot models under
`src/morphmimic/robots/` and the submodules in `third_party/` keep their own licences.

## Citation

```bibtex
@misc{yang2026morphmimic,
  title        = {{MorphMimic}: Morphology-Conditioned World Model and Policy Sharing Across Humanoids},
  author       = {Yang, Lizhi and Xie, Fengze and Wang, Yuyang and Gan, Lu and Yue, Yisong and Ames, Aaron D.},
  year         = {2026},
  note         = {Technical report},
  howpublished = {\url{https://lzyang2000.github.io/morphmimic/}}
}
```

MorphMimic builds on:

```bibtex
@inproceedings{liao2026supermimic,
  title     = {SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model},
  author    = {Liao and Shao and Long and Zakka and Horowitz and Abbeel and Sreenath},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
  eprint    = {2610.28480},
  archivePrefix = {arXiv}
}
```

We also use [mjlab](https://github.com/mujocolab/mjlab),
[BONES-SEED](https://huggingface.co/datasets/bones-studio/seed),
[SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter),
[GMR](https://github.com/YanjieZe/GMR),
[XRoboToolkit](https://github.com/XR-Robotics), and the X2 Ultra and T2 models vendored
under `src/morphmimic/robots/` with their licences.
