// Policy inputs from what a robot measures: a port of deploy/sim2sim/obs.py and of the input
// side of policy_node.PolicyRunner. Plain ES module, no DOM or Node APIs.
//
// - proprioFrame: one 93-value frame (canonical joints minus the default pose, canonical joint
//   velocities, the IMU gyro and gravity direction in the IMU frame, the last action).
// - goalMotion: the 670-value motion goal (goal version 2): reference joints, the reference
//   anchor path in its own heading frame, the reference anchor orientation relative to the
//   robot's anchor.
// - goalKeypoint: the 450 five-point features, positions in the reference pelvis heading frame,
//   orientations relative to the robot's slot-0 (pelvis) orientation.
//
// Quaternions are wxyz. Reference states are flat Float64Arrays of frames x 240 values:
// [q 29 | qd 29 | pos 14x3 | quat 14x4 | lin 14x3 | ang 14x3]. Everything is computed in double
// precision and the policy tensors are rounded to float32 at the end, as obs.py does.

export const NUM_JOINTS = 29;
export const NUM_BODIES = 14;
export const GOAL_K = 10;
export const ROOT_IDX = 0;
export const ANCHOR_IDX = 7;
export const KP_BODY_IDX = [0, 3, 6, 10, 13]; // pelvis, left/right ankle, left/right wrist
export const NUM_KP = KP_BODY_IDX.length;
export const PROPRIO_DIM = 93;
export const STATE_DIM = 240;
export const POS = 58; // body positions, 14 x 3
export const QUAT = 100; // body orientations, 14 x 4
export const LIN = 156;
export const ANG = 198;
// At control step t the policy sees reference frames t + 1 .. t + K (policy_node.REFERENCE_LEAD).
export const REFERENCE_LEAD = 1;

// Quaternions. Each takes array-likes and writes into `out` (a new Float64Array by default).

export function quatMul(a, b, out = new Float64Array(4)) {
  const aw = a[0], ax = a[1], ay = a[2], az = a[3];
  const bw = b[0], bx = b[1], by = b[2], bz = b[3];
  out[0] = aw * bw - ax * bx - ay * by - az * bz;
  out[1] = aw * bx + ax * bw + ay * bz - az * by;
  out[2] = aw * by - ax * bz + ay * bw + az * bx;
  out[3] = aw * bz + ax * by - ay * bx + az * bw;
  return out;
}

/** Inverse: the conjugate over the squared norm (mjlab's quat_inv). */
export function quatInv(q, out = new Float64Array(4)) {
  const n = Math.max(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3], 1e-9);
  out[0] = q[0] / n;
  out[1] = -q[1] / n;
  out[2] = -q[2] / n;
  out[3] = -q[3] / n;
  return out;
}

/** `v` rotated by the inverse of the unit quaternion `q`. */
export function quatApplyInverse(q, v, out = new Float64Array(3)) {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  out[0] = v[0] - w * tx + (y * tz - z * ty);
  out[1] = v[1] - w * ty + (z * tx - x * tz);
  out[2] = v[2] - w * tz + (x * ty - y * tx);
  return out;
}

/** Yaw (rad) of a wxyz quaternion: the xy angle of its x axis (teleop.heading). */
export function heading(q) {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

/** The yaw part of `q` (mjlab's yaw_quat). */
export function yawQuat(q, out = new Float64Array(4)) {
  const yaw = heading(q);
  out[0] = Math.cos(yaw / 2);
  out[1] = 0;
  out[2] = 0;
  out[3] = Math.sin(yaw / 2);
  return out;
}

export function axisAngleQuat(axis, angle, out = new Float64Array(4)) {
  const half = angle / 2;
  const s = Math.sin(half);
  out[0] = Math.cos(half);
  out[1] = s * axis[0];
  out[2] = s * axis[1];
  out[3] = s * axis[2];
  return out;
}

/** First two rotation-matrix columns, row-major (mjlab's matrix_from_quat: divides by the
 * squared norm), written at out[o .. o + 5]. */
export function rot6d(q, out = new Float64Array(6), o = 0) {
  const r = q[0], i = q[1], j = q[2], k = q[3];
  const s = 2 / (r * r + i * i + j * j + k * k);
  out[o] = 1 - s * (j * j + k * k);
  out[o + 1] = s * (i * j - k * r);
  out[o + 2] = s * (i * j + k * r);
  out[o + 3] = 1 - s * (i * i + k * k);
  out[o + 4] = s * (i * k - j * r);
  out[o + 5] = s * (j * k + i * r);
  return out;
}

const IDENTITY = Float64Array.of(1, 0, 0, 0);

/** A robot's policy-side constants: one `robots` entry of public/live/robots.json. */
export class Robot {
  constructor(spec) {
    this.name = spec.name;
    this.jointNames = spec.jointNames;
    this.numNative = spec.jointNames.length;
    this.defaultJointPos = Float64Array.from(spec.defaultJointPos);
    this.actionScale = Float64Array.from(spec.actionScale);
    this.perm = Int32Array.from(spec.perm); // native index of each canonical slot
    this.sign = Float64Array.from(spec.sign);
    this.offset = Float64Array.from(spec.offset);
    this.kpGoalVersion = spec.kpGoalVersion ?? 1;
    const fk = spec.fk;
    this.fkParent = fk.parent;
    this.fkQuat = fk.quat.map((q) => Float64Array.from(q));
    this.fkJoints = fk.joints.map((js) => js.map((j) => ({ native: j.native, axis: Float64Array.from(j.axis), ref: j.ref })));
    this.fkSlot0 = fk.slot0;
    this.fkAnchor = fk.anchor;
    this.fkImu = fk.imu;
    this.stand = spec.stand ? Float64Array.from(spec.stand.state) : null; // the neutral stand [240]
    this.defaultCanonical = this.toCanonical(this.defaultJointPos);
  }

  /** q_canon = sign * (q[perm] - offset) [29]. */
  toCanonical(q, out = new Float64Array(NUM_JOINTS)) {
    for (let i = 0; i < NUM_JOINTS; i++) out[i] = this.sign[i] * (q[this.perm[i]] - this.offset[i]);
    return out;
  }

  toCanonicalDelta(x, out = new Float64Array(NUM_JOINTS)) {
    for (let i = 0; i < NUM_JOINTS; i++) out[i] = this.sign[i] * x[this.perm[i]];
    return out;
  }

  /** Canonical action [29] -> native joint targets: default + to_native(a * scale); untracked
   * joints held at the default. */
  targets(action, out = new Float64Array(this.numNative)) {
    out.set(this.defaultJointPos);
    for (let i = 0; i < NUM_JOINTS; i++) {
      const n = this.perm[i];
      out[n] = this.defaultJointPos[n] + this.sign[i] * action[i] * this.actionScale[n];
    }
    return out;
  }

  /** Orientation of every kinematic-tree body in the root body's frame. */
  treeQuats(q) {
    const out = [];
    const step = new Float64Array(4);
    for (let b = 0; b < this.fkParent.length; b++) {
      const parent = this.fkParent[b];
      let rot = parent < 0 ? Float64Array.from(IDENTITY) : quatMul(out[parent], this.fkQuat[b]);
      for (const j of this.fkJoints[b]) rot = quatMul(rot, axisAngleQuat(j.axis, q[j.native] - j.ref, step));
      out.push(rot);
    }
    return out;
  }

  fromImu(imuQuat, q, body) {
    const rel = this.treeQuats(q);
    return quatMul(quatMul(imuQuat, quatInv(rel[this.fkImu])), rel[body]);
  }

  /** World orientation of the anchor body (torso), from the IMU and the native joints. */
  anchorQuat(imuQuat, q) {
    return this.fromImu(imuQuat, q, this.fkAnchor);
  }

  /** World orientation of tracked slot 0 (the pelvis-equivalent). */
  slot0Quat(imuQuat, q) {
    return this.fromImu(imuQuat, q, this.fkSlot0);
  }
}

/** One proprioception frame [93]: `imuQuat` (unit) gives the gravity direction, `gyro` is
 * the IMU's angular velocity in its own frame. */
export function proprioFrame(robot, q, dq, imuQuat, gyro, lastAction, out = new Float32Array(PROPRIO_DIM)) {
  const qc = robot.toCanonical(q);
  const dqc = robot.toCanonicalDelta(dq);
  const g = quatApplyInverse(imuQuat, [0, 0, -1]);
  for (let i = 0; i < NUM_JOINTS; i++) {
    out[i] = qc[i] - robot.defaultCanonical[i];
    out[29 + i] = dqc[i];
    out[64 + i] = lastAction[i];
  }
  for (let i = 0; i < 3; i++) {
    out[58 + i] = gyro[i];
    out[61 + i] = g[i];
  }
  return out;
}

/** The last `length` proprioception frames, oldest first; the first frame after a reset fills
 * every slot. `push` returns the internal buffer, overwritten by the next push. */
export class History {
  constructor(length) {
    this.length = length;
    this.frames = new Float32Array(length * PROPRIO_DIM);
    this.empty = true;
  }

  reset() {
    this.empty = true;
  }

  push(frame) {
    if (this.empty) {
      for (let k = 0; k < this.length; k++) this.frames.set(frame, k * PROPRIO_DIM);
      this.empty = false;
    } else {
      this.frames.copyWithin(0, PROPRIO_DIM);
      this.frames.set(frame, (this.length - 1) * PROPRIO_DIM);
    }
    return this.frames;
  }
}

const bodyPos = (ref, k, b) => k * STATE_DIM + POS + 3 * b;
const bodyQuat = (ref, k, b) => ref.subarray(k * STATE_DIM + QUAT + 4 * b, k * STATE_DIM + QUAT + 4 * b + 4);

/** Motion goal [K * 67] from the K reference states `ref` [K * 240]
 * (rewards_torch.goal_features). */
export function goalMotion(ref, robotAnchorQuat) {
  const K = ref.length / STATE_DIM;
  const out = new Float32Array(K * 67);
  const refYaw = yawQuat(bodyQuat(ref, 0, ROOT_IDX));
  const o = bodyPos(ref, 0, ROOT_IDX);
  const ox = ref[o], oy = ref[o + 1];
  const inv = quatInv(robotAnchorQuat);
  const v = new Float64Array(3);
  const p = new Float64Array(3);
  const rel = new Float64Array(4);
  const r6 = new Float64Array(6);
  for (let k = 0; k < K; k++) {
    const f = k * 67;
    const s = k * STATE_DIM;
    for (let i = 0; i < 58; i++) out[f + i] = ref[s + i];
    const a = bodyPos(ref, k, ANCHOR_IDX);
    v[0] = ref[a] - ox;
    v[1] = ref[a + 1] - oy;
    v[2] = ref[a + 2] - 0;
    quatApplyInverse(refYaw, v, p);
    out[f + 58] = p[0];
    out[f + 59] = p[1];
    out[f + 60] = p[2];
    rot6d(quatMul(inv, bodyQuat(ref, k, ANCHOR_IDX), rel), r6);
    for (let i = 0; i < 6; i++) out[f + 61 + i] = r6[i];
  }
  return out;
}

/** Five-point features [K * 45] from `ref` [K * 240] (keypoints.keypoint_features, no
 * jitter): positions in the reference pelvis heading frame of frame 0, orientations relative
 * to the robot pelvis's full orientation (version 2) or its heading only (version 1). */
export function goalKeypoint(ref, robotSlot0Quat, version = 2) {
  if (version !== 1 && version !== 2) throw new Error(`unknown five-point goal version ${version}`);
  const K = ref.length / STATE_DIM;
  const out = new Float32Array(K * NUM_KP * 9);
  const refYaw = yawQuat(bodyQuat(ref, 0, KP_BODY_IDX[0]));
  const o = bodyPos(ref, 0, KP_BODY_IDX[0]);
  const ox = ref[o], oy = ref[o + 1];
  const inv = quatInv(version === 1 ? yawQuat(robotSlot0Quat) : robotSlot0Quat);
  const v = new Float64Array(3);
  const p = new Float64Array(3);
  const rel = new Float64Array(4);
  const r6 = new Float64Array(6);
  for (let k = 0; k < K; k++) {
    for (let n = 0; n < NUM_KP; n++) {
      const b = KP_BODY_IDX[n];
      const f = (k * NUM_KP + n) * 9;
      const a = bodyPos(ref, k, b);
      v[0] = ref[a] - ox;
      v[1] = ref[a + 1] - oy;
      v[2] = ref[a + 2] - 0;
      quatApplyInverse(refYaw, v, p);
      out[f] = p[0];
      out[f + 1] = p[1];
      out[f + 2] = p[2];
      rot6d(quatMul(inv, bodyQuat(ref, k, b), rel), r6);
      for (let i = 0; i < 6; i++) out[f + 3 + i] = r6[i];
    }
  }
  return out;
}

/** Reference states of frames frame .. frame + k - 1 [k * 240], held at the clip's last frame
 * (MotionLibrary.future). */
export function referenceWindow(states, frame, k = GOAL_K) {
  const T = states.length / STATE_DIM;
  const out = new Float64Array(k * STATE_DIM);
  for (let i = 0; i < k; i++) {
    const f = Math.min(frame + i, T - 1);
    out.set(states.subarray(f * STATE_DIM, (f + 1) * STATE_DIM), i * STATE_DIM);
  }
  return out;
}

/**
 * One robot's policy inputs per control step (policy_node.PolicyRunner.step without the ONNX
 * run). `reference` has window(stepId, imuQuat, q) and reset() (reference.js); a `cue` method
 * gets the robot side's cue count. After the policy has run, setAction(action) stores its
 * canonical action [29], the next frame's last action.
 */
export class PolicyInputs {
  constructor(robot, reference, { historyLen = 10, kpHorizon = GOAL_K, kpVersion = robot.kpGoalVersion } = {}) {
    if (!(kpHorizon >= 1 && kpHorizon <= GOAL_K)) throw new Error(`kpHorizon must be in 1..${GOAL_K}`);
    this.robot = robot;
    this.reference = reference;
    this.kpVersion = kpVersion;
    this.history = new History(historyLen);
    this.lastAction = new Float64Array(NUM_JOINTS);
    this.prevStep = -1;
    this.lastWindow = null;
    this.frame = new Float32Array(PROPRIO_DIM);
    this.kpPointMask = new Uint8Array(NUM_KP).fill(1);
    this.kpFrameMask = Uint8Array.from({ length: GOAL_K }, (_, k) => (k < kpHorizon ? 1 : 0));
  }

  reset() {
    this.history.reset();
    this.lastAction.fill(0);
  }

  /**
   * The ONNX inputs for control step `stepId` (steps since the robot's last reset; a lower one
   * than before restarts the history and the reference): obs_history [historyLen * 93],
   * goal_motion [670], goal_keypoint [450] (Float32Array), kp_point_mask [5] and kp_frame_mask
   * [10] (Uint8Array, the ONNX bools). obs_history is reused by the next step.
   */
  step(stepId, imuQuat, gyro, q, dq, cues = 0) {
    if (stepId < this.prevStep) {
      this.reset();
      this.reference.reset();
    }
    this.prevStep = stepId;
    if (typeof this.reference.cue === 'function') this.reference.cue(stepId, cues);
    const n = Math.hypot(imuQuat[0], imuQuat[1], imuQuat[2], imuQuat[3]);
    const quat = Float64Array.of(imuQuat[0] / n, imuQuat[1] / n, imuQuat[2] / n, imuQuat[3] / n);
    proprioFrame(this.robot, q, dq, quat, gyro, this.lastAction, this.frame);
    const obs = this.history.push(this.frame);
    const ref = (this.lastWindow = this.reference.window(stepId, quat, q));
    return {
      obs_history: obs,
      goal_motion: goalMotion(ref, this.robot.anchorQuat(quat, q)),
      goal_keypoint: goalKeypoint(ref, this.robot.slot0Quat(quat, q), this.kpVersion),
      kp_point_mask: this.kpPointMask,
      kp_frame_mask: this.kpFrameMask,
    };
  }

  /** The policy's canonical action [29] for this step. */
  setAction(action) {
    for (let i = 0; i < NUM_JOINTS; i++) this.lastAction[i] = action[i];
  }
}
