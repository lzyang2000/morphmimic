// The reference windows the policy tracks: a port of deploy/sim2sim/policy_node.py's
// StandReference, CuedClipReference and placed, and of teleop.py's turn_states and
// blend_states, plus the live demo's loop (LoopReference). Plain ES module, no DOM or Node APIs.
//
// States are flat Float64Arrays of frames x 240 (see obs.js). A reference answers
// window(stepId, imuQuat, q) with the K states [K * 240] the policy tracks at that control step,
// from the robot's IMU orientation and native joint positions; reset() is called when the
// robot is reset (its step count goes back to 0).

import {
  ANG, GOAL_K, KP_BODY_IDX, LIN, NUM_BODIES, NUM_KP, POS, QUAT, REFERENCE_LEAD, STATE_DIM, heading, quatMul,
  referenceWindow,
} from './obs.js';
import { HUMAN_FRAME_DIM, blendHuman, humanHeading, humanPoints, humanWindow, placedHuman } from './human.js';

export { heading };

/** Reference states [n * 240] turned by `yaw` (rad) about the vertical through `origin`
 * (x, y): body positions, orientations and velocities; joints unchanged. */
export function turnStates(states, yaw, origin) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const turn = Float64Array.of(Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2));
  const out = Float64Array.from(states);
  const q = new Float64Array(4);
  const n = states.length / STATE_DIM;
  for (let f = 0; f < n; f++) {
    const base = f * STATE_DIM;
    for (let b = 0; b < NUM_BODIES; b++) {
      const p = base + POS + 3 * b;
      const x = states[p] - origin[0], y = states[p + 1] - origin[1], z = states[p + 2] - 0;
      out[p] = x * c + y * -s + z * 0;
      out[p + 1] = x * s + y * c + z * 0;
      out[p + 2] = x * 0 + y * 0 + z * 1;
      for (const part of [LIN, ANG]) {
        const v = base + part + 3 * b;
        const vx = states[v], vy = states[v + 1], vz = states[v + 2];
        out[v] = vx * c + vy * -s + vz * 0;
        out[v + 1] = vx * s + vy * c + vz * 0;
        out[v + 2] = vx * 0 + vy * 0 + vz * 1;
      }
      const o = base + QUAT + 4 * b;
      quatMul(turn, states.subarray(o, o + 4), q);
      out.set(q, o);
    }
  }
  return out;
}

/** `a` -> `b` (equal-length states) by `alpha`: linear, the body orientations slerped along the
 * shortest arc and normalised. */
export function blendStates(a, b, alpha) {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (1 - alpha) * a[i] + alpha * b[i];
  const n = a.length / STATE_DIM;
  for (let f = 0; f < n; f++) {
    for (let k = 0; k < NUM_BODIES; k++) {
      const o = f * STATE_DIM + QUAT + 4 * k;
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
  }
  return out;
}

/** Reference states turned and moved so that frame 0's pelvis faces `yaw` (rad) at `at` (x, y). */
export function placed(states, yaw, at) {
  const out = turnStates(states, yaw - heading(states.subarray(QUAT, QUAT + 4)), states.subarray(POS, POS + 2));
  const n = states.length / STATE_DIM;
  for (let f = 0; f < n; f++) {
    for (let b = 0; b < NUM_BODIES; b++) {
      const p = f * STATE_DIM + POS + 3 * b;
      out[p] += at[0];
      out[p + 1] += at[1];
    }
  }
  return out;
}

/** One robot's reference states [frames * 240] as float64, from a clips/<id>.bin buffer
 * (float32 little-endian [frames, columns]) and its clips.json entry {offset (bytes), frames}.
 * The file holds the first `columns` (clips.json `columns`) of each state, at least the ones the
 * goals read (all before LIN); the rest, the body velocities, are filled with zeros. */
export function statesFromClip(buffer, { offset, frames }, columns) {
  if (!Number.isInteger(columns) || columns < LIN || columns > STATE_DIM) {
    throw new Error(`clip columns must be an integer from ${LIN} to ${STATE_DIM}, got ${JSON.stringify(columns)}`);
  }
  const src = new Float32Array(buffer, offset, frames * columns);
  const out = new Float64Array(frames * STATE_DIM);
  for (let f = 0; f < frames; f++) {
    for (let i = 0; i < columns; i++) out[f * STATE_DIM + i] = src[f * columns + i];
  }
  return out;
}

/** Frame 0 of `state` (`dim` values per frame) in every one of `k` window frames. */
function repeat(state, dim = STATE_DIM, k = GOAL_K) {
  const out = new Float64Array(k * dim);
  for (let i = 0; i < k; i++) out.set(state.subarray(0, dim), i * dim);
  return out;
}

/** A clip's human frames go frame for frame with its robot states. */
function checkHuman(states, human) {
  const T = states.length / STATE_DIM;
  if (human.length !== T * HUMAN_FRAME_DIM) {
    throw new Error(`human frames ${human.length / HUMAN_FRAME_DIM}, robot clip ${T}`);
  }
}

/**
 * The five reference points (pelvis, left/right ankle, left/right wrist) of each window frame,
 * [K * 5 * 3] frame-major: the targets the five-point goal is built from, placed and turned as
 * the policy tracks them, moved in the plane by `origin` (x, y).
 */
export function fivePoints(window, origin = [0, 0], out = new Float64Array((window.length / STATE_DIM) * NUM_KP * 3)) {
  const K = window.length / STATE_DIM;
  for (let k = 0; k < K; k++) {
    for (let n = 0; n < NUM_KP; n++) {
      const p = k * STATE_DIM + POS + 3 * KP_BODY_IDX[n];
      const o = (k * NUM_KP + n) * 3;
      out[o] = window[p] + origin[0];
      out[o + 1] = window[p + 1] + origin[1];
      out[o + 2] = window[p + 2];
    }
  }
  return out;
}

/** Five points [K * 5 * 3] moved in the plane so the frame-0 pelvis is at `xy` (as sim_node's
 * LiveOverlay draws a window: the goals are position-free, so it follows the robot). */
export function anchorFivePoints(points, xy, out = new Float64Array(points.length)) {
  const dx = xy[0] - points[0], dy = xy[1] - points[1];
  for (let i = 0; i < points.length; i += 3) {
    out[i] = points[i] + dx;
    out[i + 1] = points[i + 1] + dy;
    out[i + 2] = points[i + 2];
  }
  return out;
}

/** The neutral stand in every window frame, turned to the robot's heading at the first step
 * after each reset (policy_node.StandReference). */
export class StandReference {
  constructor(robot) {
    this.robot = robot;
    this.stand = robot.stand;
    this._window = null;
    this.phase = 'stand';
  }

  window(stepId, imuQuat, q) {
    if (this._window === null) {
      const yaw = heading(this.robot.slot0Quat(imuQuat, q));
      this._window = repeat(turnStates(this.stand, yaw, [0, 0]));
    }
    return this._window;
  }

  reset() {
    this._window = null;
  }
}

/**
 * A clip on cue, from the neutral stand (policy_node.CuedClipReference). Until a cue, and
 * again after the clip, the window is the stand facing the robot. A cue while standing turns
 * the clip so its frame-0 pelvis heading is the robot's, places it where the stand is, and
 * plays leadInS from the stand into frame 0, the clip, then returnS from its last frame back
 * into the stand, placed where the clip ends. A cue during the motion is ignored.
 *
 * `phase` (after window()) is what that window was: 'stand', 'lead', 'clip' or 'return'.
 *
 * With `human` (the clip's human frames [T * 168], frame k with the robot clip's frame k) and
 * `humanStand` (the uniform skeleton's neutral stand [168]) it plays a human track in step: the
 * same stand, lead-in, clip and return on the human frames (human_loop.HumanCuedClipReference),
 * whose window of the same control step is `humanWindow` after window(): the human interface's.
 */
export class CuedClipReference {
  constructor(robot, states, { leadInS = 2.0, returnS = 1.0, fps = 50, human = null, humanStand = null } = {}) {
    this.robot = robot;
    this.states = states;
    this.human = human;
    this.humanStand = humanStand;
    if (human) checkHuman(states, human);
    this.fps = fps;
    this.leadIn = Math.round(leadInS * fps);
    this.ret = Math.round(returnS * fps);
    this.stand = robot.stand;
    this.reset();
  }

  /** Frames one cue plays: the lead-in, the clip and the return. */
  get length() {
    return this.leadIn + this.states.length / STATE_DIM + this.ret;
  }

  reset() {
    this._cues = 0;
    this._pending = false;
    this._idle = null; // the stand window, between motions
    this._played = null;
    this._playedFrames = 0;
    this._start = 0;
    this._humanIdle = null;
    this._humanPlayed = null;
    this.humanWindow = null;
    this.phase = 'stand';
  }

  /** `cues`: cues since the robot's last reset (the robot side's counter). */
  cue(stepId, cues) {
    if (cues <= this._cues) return;
    this._cues = cues;
    if (this._played === null && !this._pending) this._pending = true;
  }

  get playing() {
    return this._played !== null || this._pending;
  }

  window(stepId, imuQuat, q) {
    const human = this.human !== null;
    if (this._idle === null) {
      const yaw = heading(this.robot.slot0Quat(imuQuat, q));
      this._idle = repeat(placed(this.stand, yaw, [0, 0]));
      if (human) this._humanIdle = repeat(placedHuman(this.humanStand, yaw, [0, 0]), HUMAN_FRAME_DIM);
    }
    if (this._pending) {
      this._pending = false;
      const yaw = heading(this.robot.slot0Quat(imuQuat, q));
      this._played = this._motion(yaw);
      this._humanPlayed = human ? this._humanMotion(yaw) : null;
      this._playedFrames = this._played.length / STATE_DIM;
      this._start = stepId;
    }
    if (this._played !== null) {
      const frame = stepId - this._start + REFERENCE_LEAD;
      if (frame < this._playedFrames) {
        const clip = this._playedFrames - this.leadIn - this.ret;
        this.phase = frame < this.leadIn ? 'lead' : frame < this.leadIn + clip ? 'clip' : 'return';
        this.humanWindow = human ? humanWindow(this._humanPlayed, frame) : null;
        return referenceWindow(this._played, frame);
      }
      this._idle = repeat(this._played.subarray((this._playedFrames - 1) * STATE_DIM));
      if (human) this._humanIdle = repeat(this._humanPlayed.subarray((this._playedFrames - 1) * HUMAN_FRAME_DIM), HUMAN_FRAME_DIM);
      this._played = null;
      this._humanPlayed = null;
      this._ended(stepId);
    }
    this.phase = 'stand';
    this.humanWindow = this._humanIdle;
    return this._idle;
  }

  /** The human track of a motion (human_loop.HumanCuedClipReference._motion): as _motion, on the
   * human frames, from the human stand window. */
  _humanMotion(yaw) {
    const D = HUMAN_FRAME_DIM;
    const start = this._humanIdle.subarray(0, D);
    const clip = placedHuman(this.human, yaw, [start[0], start[1]]);
    const T = clip.length / D;
    const end = clip.subarray((T - 1) * D);
    const stand = placedHuman(this.humanStand, humanHeading(end), [end[0], end[1]]);
    const out = new Float64Array((this.leadIn + T + this.ret) * D);
    const first = clip.subarray(0, D);
    for (let i = 0; i < this.leadIn; i++) out.set(blendHuman(start, first, (i + 1) / this.leadIn), i * D);
    out.set(clip, this.leadIn * D);
    for (let i = 0; i < this.ret; i++) out.set(blendHuman(end, stand, (i + 1) / this.ret), (this.leadIn + T + i) * D);
    return out;
  }

  /** The motion ended at `stepId`: the window is the stand again. */
  _ended() {}

  _motion(yaw) {
    const start = this._idle.subarray(0, STATE_DIM);
    const clip = placed(this.states, yaw, start.subarray(POS, POS + 2));
    const T = clip.length / STATE_DIM;
    const end = clip.subarray((T - 1) * STATE_DIM);
    const stand = placed(this.stand, heading(end.subarray(QUAT, QUAT + 4)), end.subarray(POS, POS + 2));
    const out = new Float64Array((this.leadIn + T + this.ret) * STATE_DIM);
    const first = clip.subarray(0, STATE_DIM);
    for (let i = 0; i < this.leadIn; i++) out.set(blendStates(start, first, (i + 1) / this.leadIn), i * STATE_DIM);
    out.set(clip, this.leadIn * STATE_DIM);
    for (let i = 0; i < this.ret; i++) out.set(blendStates(end, stand, (i + 1) / this.ret), (this.leadIn + T + i) * STATE_DIM);
    return out;
  }
}

/**
 * The live demo's loop: CuedClipReference cueing itself. The robot stands, the first cue comes
 * firstCueS after a reset, and each next one holdS after the previous motion is back on the
 * stand. setClip(states) changes the clip from the next cue on; the motion playing goes on.
 * While `cueHeld` is set the robot does not cue: it finishes its motion and waits at the stand
 * (the loop re-homes the robots then; see loop.js).
 * `origin` (x, y) is where the reference's (0, 0) is in the world, for fivePointTargets (the
 * goals themselves are position-free).
 */
export class LoopReference extends CuedClipReference {
  constructor(robot, states, { holdS = 1.0, firstCueS = holdS, origin = [0, 0], ...cued } = {}) {
    super(robot, states, cued);
    this.hold = Math.round(holdS * this.fps);
    this.firstCue = Math.round(firstCueS * this.fps);
    this.origin = origin;
    this.nextCue = this.firstCue;
    this.lastWindow = null;
    this.cueHeld = false; // while true the robot waits at the stand past its cue (kept by reset)
  }

  reset() {
    super.reset();
    this.nextCue = this.firstCue;
    this.lastWindow = null;
  }

  /** The clip (and its human frames, with a human track) from the next cue on: a cue builds its
   * whole motion (lead-in, clip, return) when it starts, so the motion playing goes on with the
   * clip it had. */
  setClip(states, human = null) {
    if (this.human !== null) {
      if (human === null) throw new Error('setClip: this reference has a human track, give the clip\'s human frames');
      checkHuman(states, human);
      this.human = human;
    }
    this.states = states;
  }

  window(stepId, imuQuat, q) {
    if (!this.playing && !this.cueHeld && stepId >= this.nextCue) this._pending = true;
    return (this.lastWindow = super.window(stepId, imuQuat, q));
  }

  _ended(stepId) {
    this.nextCue = stepId + this.hold;
  }

  /** The human track's 24 slot positions of the last window, [K * 24 * 3] world positions
   * (frame 0 the current target, then the preview); null before the first step or without a
   * human track. */
  humanTargets(out) {
    return this.humanWindow === null ? null : humanPoints(this.humanWindow, this.origin, out);
  }

  /** The five-point targets of the last window, [K * 5 * 3] world positions: frame 0 is the
   * current target, frames 1 .. K - 1 the preview. */
  fivePointTargets(out) {
    return this.lastWindow === null ? null : fivePoints(this.lastWindow, this.origin, out);
  }
}
