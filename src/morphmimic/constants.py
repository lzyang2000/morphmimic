"""Robot, observation and world-model layout constants."""

from __future__ import annotations

import math
from typing import Literal

# Tracked bodies, same order as mjlab's G1 tracking task.
# Index order matters everywhere.
TRACKED_BODY_NAMES: tuple[str, ...] = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)
ROOT_IDX = 0  # pelvis
ANCHOR_IDX = 7  # torso_link
ANCHOR_BODY_NAME = "torso_link"

# mjlab G1 joint order: left leg 6, right leg 6, waist 3, left arm 7, right arm 7.
_JOINT_STEMS: tuple[str, ...] = (
  "left_hip_pitch",
  "left_hip_roll",
  "left_hip_yaw",
  "left_knee",
  "left_ankle_pitch",
  "left_ankle_roll",
  "right_hip_pitch",
  "right_hip_roll",
  "right_hip_yaw",
  "right_knee",
  "right_ankle_pitch",
  "right_ankle_roll",
  "waist_yaw",
  "waist_roll",
  "waist_pitch",
  "left_shoulder_pitch",
  "left_shoulder_roll",
  "left_shoulder_yaw",
  "left_elbow",
  "left_wrist_roll",
  "left_wrist_pitch",
  "left_wrist_yaw",
  "right_shoulder_pitch",
  "right_shoulder_roll",
  "right_shoulder_yaw",
  "right_elbow",
  "right_wrist_roll",
  "right_wrist_pitch",
  "right_wrist_yaw",
)
JOINT_NAMES: tuple[str, ...] = tuple(f"{j}_joint" for j in _JOINT_STEMS)
# BONES-SEED G1 CSV joint columns (degrees), in the same order as JOINT_NAMES.
SEED_CSV_JOINT_COLUMNS: tuple[str, ...] = tuple(f"{j}_joint_dof" for j in _JOINT_STEMS)

NUM_JOINTS = 29
NUM_BODIES = 14

# World-model state s (world frame):
# [q 29 | qd 29 | pos 14x3 | quat 14x4 | lin 14x3 | ang 14x3].
STATE_DIM = 240
# Canonical features (heading or root frame, `morphmimic.model.state`):
# [q 29 | qd 29 | pos 42 | rot6d 84 | lin 42 | ang 42].
LOCAL_DIM = 268

# Policy observation (single frame):
# [q_rel 29 | qd 29 | imu ang vel 3 | gravity 3 | a_prev 29].
PROPRIO_DIM = 93
ACTION_DIM = 29
HIST_LEN = 10

# Goal: K future reference frames of
# [ref q 29 | ref qd 29 | anchor pos 3 (reference heading frame) | anchor rot6d 6
# (relative to the robot anchor)]. Position-free: no robot position.
GOAL_K = 10
GOAL_FRAME_DIM = 67
GOAL_DIM = 670
# Version of the goal inputs (motion and keypoint), saved in checkpoints as
# "goal_version": 1 = robot-relative positions (checkpoints without the entry),
# 2 = position-free. A policy only works with the version it was trained on.
GOAL_VERSION = 2
# Frame of the motion goal's anchor position (`rewards_torch.goal_features`):
# "reference" (default), the reference's own heading frame (position-free);
# "robot", the current robot anchor frame, as goal version 1 (needs the robot's pose at
# deploy, from a state estimator). Saved in the checkpoint cfg as `goal_anchor`.
GoalAnchor = Literal["reference", "robot"]
GOAL_ANCHORS: tuple[GoalAnchor, ...] = ("reference", "robot")
# Version of the keypoint goal, saved in checkpoints as "kp_goal_version": 1 = the
# five robot bodies, orientations relative to the robot pelvis heading (yaw only;
# checkpoints without the entry), 2 = the five robot bodies, orientations relative to
# the robot pelvis's full orientation (the IMU's), 3 = the human skeleton goal (24 SMPL
# slots), orientations relative to the robot pelvis's full orientation, 4 = the human
# skeleton goal, orientations in the anchor's heading frame at frame 0, as the
# positions, 5 = version 4 plus the robot's heading error to
# the human anchor (yaw only), 6 = version 4 plus the human anchor's orientation
# relative to the robot pelvis, 6D per future frame, 7 = version 6 with that block
# standardised by its own running normaliser, 10 = version 7 with the block measured
# torso to torso: the human Spine3 against the robot anchor body (torso_link), as the
# motion goal's orientation term (8 and 9 are other branches' versions). Only the
# keypoint interface depends on it.
KP_GOAL_VERSION = 10
# Privileged critic input (critics only, never the policy): the reference
# anchor position in the robot anchor frame for each of the K goal frames, divided by
# CRITIC_ANCHOR_POS_SCALE (m, the reward's position kernel std). The reward's global
# anchor-position term depends on this offset, which the position-free goal lacks.
CRITIC_PRIV_DIM = GOAL_K * 3
CRITIC_ANCHOR_POS_SCALE = 0.3

# Human skeleton goal: SMPL's 24 joints in SMPL order, the same as PICO's body joints
# (`deploy.sim2sim.human.PICO_JOINTS`), so slot k is PICO joint k.
HUMAN_JOINTS: tuple[str, ...] = (
  "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee", "Spine2",
  "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot", "Neck",
  "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
  "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
)  # fmt: skip
# The SOMA joint that fills each slot at training time.
HUMAN_SOMA_SOURCE: tuple[str, ...] = (
  "Hips", "LeftLeg", "RightLeg", "Spine1", "LeftShin", "RightShin", "Spine2",
  "LeftFoot", "RightFoot", "Chest", "LeftToeBase", "RightToeBase", "Neck1",
  "LeftShoulder", "RightShoulder", "Head", "LeftArm", "RightArm", "LeftForeArm",
  "RightForeArm", "LeftHand", "RightHand", "LeftHandMiddle1", "RightHandMiddle1",
)  # fmt: skip
NUM_HUMAN = 24
HUMAN_PELVIS = 0
HUMAN_HEAD = 15
HUMAN_SPINE3 = 9
HUMAN_L_ANKLE = 7
HUMAN_R_ANKLE = 8
HUMAN_L_WRIST = 20
HUMAN_R_WRIST = 21
# Slot sets of the settings (`full` = all 24).
HUMAN_FIVE_IDX: tuple[int, ...] = (0, 7, 8, 20, 21)
HUMAN_SIX_IDX: tuple[int, ...] = (0, 9, 7, 8, 20, 21)
HUMAN_THREE_IDX: tuple[int, ...] = (15, 20, 21)
# Position anchors in order of preference: the Pelvis when its slot is on, else the
# Head.
HUMAN_ANCHORS: tuple[int, ...] = (0, 15)
# Features per slot per frame: [pos 3 | rot6d 6].
HUMAN_POINT_DIM = 9
HUMAN_FRAME_DIM = 216  # NUM_HUMAN * HUMAN_POINT_DIM
HUMAN_FEAT_DIM = 2160  # GOAL_K * HUMAN_FRAME_DIM
# Mask bits appended to the features: [slot mask 24 | frame mask GOAL_K].
HUMAN_MASK_DIM = 34
# The human Spine3's orientation relative to the robot anchor body (torso_link), rot6d
# per future frame, after the mask bits (keypoint goal version 10; versions 6-7 measured
# the human anchor against the robot pelvis; a row without the Spine3 slot measures
# its anchor, Pelvis else Head, against the robot torso), standardised by its own
# normaliser; all zero when the heading is masked.
HUMAN_HEADING_DIM = 60  # GOAL_K * 6
HUMAN_GOAL_DIM = 2254  # HUMAN_FEAT_DIM + HUMAN_MASK_DIM + HUMAN_HEADING_DIM
# Raw goal the env carries: per frame and slot the world pose [pos 3 | quat 4].
HUMAN_POSE_DIM = 7
HUMAN_RAW_DIM = 1680  # GOAL_K * NUM_HUMAN * HUMAN_POSE_DIM
# Raw poses then the robot anchor body's (torso_link) quaternion (wxyz, world).
HUMAN_ENV_DIM = 1684  # HUMAN_RAW_DIM + 4
# The SMPL rest frame against the robot pelvis frame on the retargeted references: an
# aligned human anchor is the robot pelvis turned by Rz(HUMAN_YAW_OFFSET) then
# Rx(HUMAN_ROLL_OFFSET) about its own x (SMPL joint frames are y-up). The yaw part is
# 90.0° on 221115/idle_loop_001__A061_M; with both, the relative orientation of the
# Pelvis on the fixture reset poses is within 1.1° of the identity.
HUMAN_YAW_OFFSET = math.pi / 2
HUMAN_ROLL_OFFSET = math.pi / 2
# Target-jitter half-widths per joint and frame: position in m, orientation in rad.
# HEFT's sizes (github.com/Axellwppr/motion_tracking, G1_tracking.yaml); SONIC's
# Table S4 values (5 cm / 1 cm / 0.1 / 0.2 rad) until 2026-10-27.
HUMAN_JITTER = {"pos_xy": 0.005, "pos_z": 0.005, "rot_rp": 0.01, "rot_yaw": 0.01}
# Slow drift of the whole human skeleton (HEFT's student motion randomisation, the
# headset's wandering pelvis and chest): per episode, sinusoids at a log-uniform
# frequency in `freq` Hz with a uniform phase. xy: along a random direction, peak
# speed U[0, xy_speed] m/s; z: a constant U[-z_offset, z_offset] m plus an amplitude
# U[0, z_amp] m; rotation about the pelvis: a random axis, amplitude U[0, rot_amp]
# rad. On top, per frame: a shift of radius U[0, pos_noise] m in a random direction
# and a turn of U[-rot_noise, rot_noise] rad about a random axis.
HUMAN_DRIFT = {
  "xy_speed": 0.2,
  "z_offset": 0.05,
  "z_amp": 0.025,
  "rot_amp": 0.1,
  "freq": (0.25, 1.0),
  "pos_noise": 0.005,
  "rot_noise": 0.02,
}

# Context xi: [foot friction 1 | encoder bias 29 | torso CoM offset 3].
CTX_DIM = 33

DT = 0.02
FPS = 50
# Max per-step reward: (sum of positive reward weights = 5) * DT.
R_MAX = 0.1

# Uniform observation noise half-widths (Table A.1).
OBS_NOISE = {"joint_pos": 0.01, "joint_vel": 0.5, "ang_vel": 0.2, "gravity": 0.05}
