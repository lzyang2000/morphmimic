// The human interface's goal and the window operations its references need: a port of
// deploy/sim2sim/human.py (human_features, heading_features, human_heading, turn_human,
// blend_human) and of tools/human_loop.py's placed_human. Plain ES module, no DOM or Node APIs.
//
// A human frame is the 24 SMPL slots' world poses [24 * 7]: per slot its position (m, z up),
// then its orientation (wxyz, SMPL's rest convention). A window is K frames [K * 168], frame 0
// the current one. Everything is computed in double precision; the policy tensors are rounded to
// float32 at the end, as policy_node does.

import { GOAL_K, heading, quatApplyInverse, quatInv, quatMul, rot6d, yawQuat } from './obs.js';

export const NUM_HUMAN = 24;
export const HUMAN_FRAME_DIM = NUM_HUMAN * 7;
export const HUMAN_PELVIS = 0;
export const HUMAN_SPINE3 = 9;
export const HUMAN_HEAD = 15;
// The slots of the five-point setting: pelvis, left/right ankle, left/right wrist
// (human.HUMAN_POINTS["five"]), in the five-point overlay's colour order.
export const HUMAN_FIVE = [0, 7, 8, 20, 21];
// An aligned human joint is the robot body turned by Rz(pi / 2) then Rx(pi / 2), so a human
// orientation times UNDO is in the robot's convention (x forward, z up): human._UNDO.
const HALF = Math.PI / 4;
export const UNDO = quatMul(Float64Array.of(Math.cos(HALF), -Math.sin(HALF), 0, 0),
  Float64Array.of(Math.cos(HALF), 0, 0, -Math.sin(HALF)));

/** Slot mask [24] (1 = seen) of a setting: 'full' or 'five' (human.point_mask). */
export function pointMask(points) {
  if (points === 'full') return new Uint8Array(NUM_HUMAN).fill(1);
  if (points !== 'five') throw new Error(`unknown human point setting ${JSON.stringify(points)}`);
  const mask = new Uint8Array(NUM_HUMAN);
  for (const s of HUMAN_FIVE) mask[s] = 1;
  return mask;
}

const anchorSlot = (mask) => (mask[HUMAN_PELVIS] ? HUMAN_PELVIS : HUMAN_HEAD);
const posAt = (k, s) => k * HUMAN_FRAME_DIM + 7 * s;
const quatAt = (f, k, s) => f.subarray(k * HUMAN_FRAME_DIM + 7 * s + 3, k * HUMAN_FRAME_DIM + 7 * s + 7);

/** `v` rotated by the unit quaternion `q` (obs.quat_apply). */
export function quatApply(q, v, out = new Float64Array(3)) {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  out[0] = v[0] + w * tx + (y * tz - z * ty);
  out[1] = v[1] + w * ty + (z * tx - x * tz);
  out[2] = v[2] + w * tz + (x * ty - y * tx);
  return out;
}

/** The goal features [K * 24 * 9] of a window [K * 168] (human.human_features): per frame and
 * slot the position and the rot6d orientation, in the heading frame of the anchor at frame 0
 * (the pelvis when `mask` has it, else the head), positions with the origin at its xy, z
 * absolute. Unmasked: the ONNX graph masks after normalising. */
export function humanFeatures(window, mask) {
  const K = window.length / HUMAN_FRAME_DIM;
  const out = new Float32Array(K * NUM_HUMAN * 9);
  const a = anchorSlot(mask);
  const yaw = yawQuat(quatAt(window, 0, a));
  const inv = quatInv(yaw);
  const o = posAt(0, a);
  const ox = window[o], oy = window[o + 1];
  const v = new Float64Array(3);
  const p = new Float64Array(3);
  const rel = new Float64Array(4);
  const r6 = new Float64Array(6);
  for (let k = 0; k < K; k++) {
    for (let s = 0; s < NUM_HUMAN; s++) {
      const f = (k * NUM_HUMAN + s) * 9;
      const i = posAt(k, s);
      v[0] = window[i] - ox;
      v[1] = window[i + 1] - oy;
      v[2] = window[i + 2] - 0;
      quatApplyInverse(yaw, v, p);
      out[f] = p[0];
      out[f + 1] = p[1];
      out[f + 2] = p[2];
      rot6d(quatMul(inv, quatAt(window, k, s), rel), r6);
      for (let j = 0; j < 6; j++) out[f + 3 + j] = r6[j];
    }
  }
  return out;
}

/** The heading block [K * 6] (human.heading_features): per frame the human Spine3's
 * orientation (else the anchor's) relative to the robot torso `robotQuat` (wxyz, world),
 * inv(robot) * slot * UNDO, as rot6d. */
export function headingFeatures(robotQuat, window, mask) {
  const K = window.length / HUMAN_FRAME_DIM;
  const out = new Float32Array(K * 6);
  const slot = mask[HUMAN_SPINE3] ? HUMAN_SPINE3 : anchorSlot(mask);
  const inv = quatInv(robotQuat);
  const q = new Float64Array(4);
  const rel = new Float64Array(4);
  const r6 = new Float64Array(6);
  for (let k = 0; k < K; k++) {
    rot6d(quatMul(inv, quatMul(quatAt(window, k, slot), UNDO, q), rel), r6);
    for (let j = 0; j < 6; j++) out[k * 6 + j] = r6[j];
  }
  return out;
}

/** Heading (rad) of a human frame [168]: its pelvis's yaw in the robot's convention
 * (human.human_heading). */
export function humanHeading(frame) {
  return heading(quatMul(quatAt(frame, 0, HUMAN_PELVIS), UNDO));
}

/** Human frames [n * 168] turned by `yaw` about the vertical through `origin` (x, y), which
 * goes to x = y = 0 (human.turn_human). */
export function turnHuman(frames, yaw, origin) {
  const turn = Float64Array.of(Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2));
  const out = new Float64Array(frames.length);
  const v = new Float64Array(3);
  const p = new Float64Array(3);
  const q = new Float64Array(4);
  for (let i = 0; i < frames.length; i += 7) {
    v[0] = frames[i] - origin[0];
    v[1] = frames[i + 1] - origin[1];
    v[2] = frames[i + 2] - 0;
    quatApply(turn, v, p);
    out[i] = p[0];
    out[i + 1] = p[1];
    out[i + 2] = p[2];
    quatMul(turn, frames.subarray(i + 3, i + 7), q);
    out.set(q, i + 3);
  }
  return out;
}

/** Human frames `a` -> `b` (equal length) by `alpha`: positions linear, orientations slerped
 * along the shortest arc (human.blend_human, obs.slerp). */
export function blendHuman(a, b, alpha) {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i += 7) {
    for (let j = 0; j < 3; j++) out[i + j] = (1 - alpha) * a[i + j] + alpha * b[i + j];
    const o = i + 3;
    const dot = a[o] * b[o] + a[o + 1] * b[o + 1] + a[o + 2] * b[o + 2] + a[o + 3] * b[o + 3];
    const sb = dot < 0 ? -1 : 1;
    const theta = Math.acos(Math.min(Math.max(Math.abs(dot), -1), 1));
    const sin = Math.sin(theta);
    const near = sin < 1e-8;
    const wa = near ? 1 - alpha : Math.sin((1 - alpha) * theta) / sin;
    const wb = near ? alpha : Math.sin(alpha * theta) / sin;
    const q0 = wa * a[o] + wb * (sb * b[o]);
    const q1 = wa * a[o + 1] + wb * (sb * b[o + 1]);
    const q2 = wa * a[o + 2] + wb * (sb * b[o + 2]);
    const q3 = wa * a[o + 3] + wb * (sb * b[o + 3]);
    const norm = Math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    out[o] = q0 / norm;
    out[o + 1] = q1 / norm;
    out[o + 2] = q2 / norm;
    out[o + 3] = q3 / norm;
  }
  return out;
}

/** Human frames [n * 168] turned and moved so that frame 0's pelvis faces `yaw` (rad) at `at`
 * (x, y): human_loop.placed_human. */
export function placedHuman(frames, yaw, at) {
  const out = turnHuman(frames, yaw - humanHeading(frames), [frames[0], frames[1]]);
  for (let i = 0; i < out.length; i += 7) {
    out[i] += at[0];
    out[i + 1] += at[1];
  }
  return out;
}

/** Human frames frame .. frame + k - 1 [k * 168], held at the last frame (reference_window). */
export function humanWindow(frames, frame, k = GOAL_K) {
  const T = frames.length / HUMAN_FRAME_DIM;
  const out = new Float64Array(k * HUMAN_FRAME_DIM);
  for (let i = 0; i < k; i++) {
    const f = Math.min(frame + i, T - 1);
    out.set(frames.subarray(f * HUMAN_FRAME_DIM, (f + 1) * HUMAN_FRAME_DIM), i * HUMAN_FRAME_DIM);
  }
  return out;
}

/** A clip's human frames from its clips/<id>.bin (float32 [frames, 168]) and its clips.json
 * `human` entry {offset (bytes), frames}, as float64. */
export function humanFromClip(buffer, { offset, frames }) {
  return Float64Array.from(new Float32Array(buffer, offset, frames * HUMAN_FRAME_DIM));
}

/** The 24 slot positions of each window frame [K * 24 * 3], moved by `origin` (x, y): where the
 * reference's (0, 0) is in the world. */
export function humanPoints(window, origin = [0, 0], out = new Float64Array((window.length / 7) * 3)) {
  for (let i = 0, j = 0; i < window.length; i += 7, j += 3) {
    out[j] = window[i] + origin[0];
    out[j + 1] = window[i + 1] + origin[1];
    out[j + 2] = window[i + 2];
  }
  return out;
}
