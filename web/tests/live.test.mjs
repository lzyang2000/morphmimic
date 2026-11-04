import test from 'node:test';
import assert from 'node:assert/strict';
import {
  GOAL_K, History, PROPRIO_DIM, PolicyInputs, Robot, STATE_DIM, axisAngleQuat, goalMotion,
  heading, quatApplyInverse, quatInv, quatMul, referenceWindow, rot6d, yawQuat,
} from '../src/live/obs.js';
import {
  CuedClipReference, LoopReference, StandReference, anchorFivePoints, blendStates, fivePoints, placed,
  statesFromClip, turnStates,
} from '../src/live/reference.js';
import {
  HUMAN_FIVE, HUMAN_FRAME_DIM, UNDO, headingFeatures, humanFeatures, humanHeading, humanWindow, placedHuman,
  pointMask, turnHuman,
} from '../src/live/human.js';

const close = (a, b, tol = 1e-12, msg = '') => {
  assert.equal(a.length, b.length, `${msg} length`);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) <= tol, `${msg}[${i}]: ${a[i]} vs ${b[i]}`);
};
const zq = (yaw) => [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
const xq = (a) => [Math.cos(a / 2), Math.sin(a / 2), 0, 0];

// A reference state: joints j, every body at pelvis + (b * 0.1, 0, 0) turned by `yaw`, orientation zq(yaw).
function state(px, py, yaw, j = 0) {
  const s = new Float64Array(STATE_DIM);
  for (let i = 0; i < 29; i++) s[i] = j + 0.01 * i;
  for (let b = 0; b < 14; b++) {
    s[58 + 3 * b] = px + Math.cos(yaw) * 0.1 * b;
    s[58 + 3 * b + 1] = py + Math.sin(yaw) * 0.1 * b;
    s[58 + 3 * b + 2] = 0.8 - 0.05 * b;
    s.set(zq(yaw), 100 + 4 * b);
    s[156 + 3 * b] = 1; // linear velocity along x
  }
  return s;
}
const clip = (frames, yaw0 = 0) => {
  const out = new Float64Array(frames * STATE_DIM);
  for (let f = 0; f < frames; f++) out.set(state(0.5 + 0.02 * f, -0.3, yaw0 + 0.01 * f, f), f * STATE_DIM);
  return out;
};

// A toy robot: identity joint map, the anchor one waist-yaw joint (native 0) above the root, which carries the IMU.
function toySpec(stand = state(0.1, 0.2, 0.3)) {
  const n = 29;
  return {
    name: 'toy',
    jointNames: Array.from({ length: n }, (_, i) => `j${i}`),
    defaultJointPos: Array.from({ length: n }, (_, i) => 0.1 * i),
    actionScale: Array(n).fill(0.5),
    perm: Array.from({ length: n }, (_, i) => i), sign: Array(n).fill(1), offset: Array(n).fill(0),
    fk: {
      parent: [-1, 0], quat: [[1, 0, 0, 0], [1, 0, 0, 0]],
      joints: [[], [{ native: 0, axis: [0, 0, 1], ref: 0 }]], slot0: 0, anchor: 1, imu: 0,
    },
    stand: { state: Array.from(stand) },
  };
}

test('quaternion helpers', () => {
  const a = zq(0.4);
  close(quatMul(a, [1, 0, 0, 0]), a);
  close(quatMul(a, quatInv(a)), [1, 0, 0, 0]);
  close(quatMul(zq(0.4), zq(0.3)), zq(0.7));
  close(quatApplyInverse(zq(Math.PI / 2), [1, 0, 0]), [0, -1, 0], 1e-12);
  close(yawQuat(quatMul(zq(0.7), xq(0.5))), zq(0.7), 1e-12);
  assert.ok(Math.abs(heading(quatMul(zq(-2.5), xq(0.3))) + 2.5) < 1e-12);
  close(axisAngleQuat([0, 0, 1], 0.6), zq(0.6));
  close(rot6d([1, 0, 0, 0]), [1, 0, 0, 1, 0, 0]);
  close(rot6d(zq(Math.PI / 2)), [0, -1, 1, 0, 0, 0], 1e-12); // columns x -> y, y -> -x, row-major
  close(quatInv([2, 0, 0, 0]), [0.5, 0, 0, 0]); // conjugate over the squared norm
});

test('robot: canonical joints, targets, anchor and slot-0 orientation from the IMU', () => {
  const spec = toySpec();
  spec.perm = spec.perm.slice().reverse();
  spec.sign[0] = -1;
  const robot = new Robot(spec);
  const q = Float64Array.from({ length: 29 }, (_, i) => i);
  const c = robot.toCanonical(q);
  assert.equal(c[0], -28);
  assert.equal(c[1], 27);
  const t = robot.targets(Float64Array.from({ length: 29 }, () => 1));
  assert.ok(Math.abs(t[28] - (2.8 - 0.5)) < 1e-12); // slot 0 is native 28, sign -1
  assert.ok(Math.abs(t[0] - (0 + 0.5)) < 1e-12);
  const toy = new Robot(toySpec());
  const imu = zq(0.2);
  const joints = new Float64Array(29);
  joints[0] = 0.5;
  close(toy.slot0Quat(imu, joints), imu, 1e-12);
  close(toy.anchorQuat(imu, joints), zq(0.7), 1e-12);
});

test('history fills on the first push, then shifts oldest out', () => {
  const h = new History(3);
  const f = (v) => new Float32Array(PROPRIO_DIM).fill(v);
  let o = h.push(f(1));
  assert.equal(o.length, 3 * PROPRIO_DIM);
  assert.deepEqual([o[0], o[PROPRIO_DIM], o[2 * PROPRIO_DIM]], [1, 1, 1]);
  o = h.push(f(2));
  assert.deepEqual([o[0], o[PROPRIO_DIM], o[2 * PROPRIO_DIM]], [1, 1, 2]);
  h.reset();
  o = h.push(f(3));
  assert.deepEqual([o[0], o[2 * PROPRIO_DIM]], [3, 3]);
});

test('reference window holds the last frame', () => {
  const states = clip(4);
  const w = referenceWindow(states, 2, 5);
  assert.equal(w.length, 5 * STATE_DIM);
  assert.equal(w[0], 2);
  for (let k = 1; k < 5; k++) assert.equal(w[k * STATE_DIM], 3);
});

test('turnStates turns about a vertical axis; joints unchanged; a full turn is the identity', () => {
  const s = state(1, 0, 0);
  const t = turnStates(s, Math.PI / 2, [1, 0]);
  close(t.subarray(58, 61), [0, 0, 0.8], 1e-12); // pelvis on the axis: origin goes to 0
  close(t.subarray(61, 64), [0, 0.1, 0.75], 1e-12);
  close(t.subarray(100, 104), zq(Math.PI / 2), 1e-12);
  close(t.subarray(156, 159), [0, 1, 0], 1e-12);
  close(t.subarray(0, 58), s.subarray(0, 58), 0);
  const full = turnStates(s, 2 * Math.PI, [0, 0]);
  close(full.subarray(58, 100), s.subarray(58, 100), 1e-12);
});

test('blendStates: endpoints, slerped and normalised orientations', () => {
  const a = state(0, 0, 0);
  const b = state(1, 0, 1.0);
  close(blendStates(a, b, 0), a, 1e-12);
  close(blendStates(a, b, 1), b, 1e-12);
  const m = blendStates(a, b, 0.5);
  close(m.subarray(100, 104), zq(0.5), 1e-12);
  assert.ok(Math.abs(m[58] - 0.5) < 1e-12);
  // The shortest arc: -q is the same rotation.
  const nb = b.slice();
  for (let i = 100; i < 156; i++) nb[i] = -nb[i];
  close(blendStates(a, nb, 0.5).subarray(100, 104), zq(0.5), 1e-12);
});

test('placed puts frame 0 pelvis at `at`, facing `yaw`', () => {
  const p = placed(clip(3, 1.2), -0.4, [2, 3]);
  close(p.subarray(58, 60), [2, 3], 1e-12);
  assert.ok(Math.abs(heading(p.subarray(100, 104)) + 0.4) < 1e-12);
});

test('goals are position-free: a window moved in the plane gives the same goals', () => {
  const w = referenceWindow(clip(12), 1);
  const moved = turnStates(w, 0, [-3, 4]);
  const anchor = zq(0.3);
  close(goalMotion(moved, anchor), goalMotion(w, anchor), 1e-6);
  assert.equal(goalMotion(w, anchor).length, GOAL_K * 67);
});

test('stand reference: the stand turned to the robot heading at the first step', () => {
  const robot = new Robot(toySpec());
  const ref = new StandReference(robot);
  const w = ref.window(0, zq(1.0), new Float64Array(29));
  assert.equal(w.length, GOAL_K * STATE_DIM);
  assert.ok(Math.abs(heading(w.subarray(100, 104)) - 1.3) < 1e-12); // stand faces 0.3, turned by 1.0
  assert.equal(ref.window(1, zq(2.0), new Float64Array(29)), w);
});

// fps 10: lead-in 3 frames, return 2, hold 2, first cue at step 2, clip 5 frames: one cue plays
// 10 frames from step 2, its window starting at frame step - 2 + 1.
const LOOP = { leadInS: 0.3, returnS: 0.2, holdS: 0.2, firstCueS: 0.2, fps: 10 };
const phasesOf = (ref, robot, n, before = () => {}) => {
  const out = [];
  for (let s = 0; s < n; s++) {
    before(s);
    ref.window(s, zq(0.5), new Float64Array(29));
    out.push(ref.phase[0]);
  }
  return out.join('');
};

test('loop: stand, lead-in, clip, return, hold, then it cues itself again', () => {
  const robot = new Robot(toySpec());
  const ref = new LoopReference(robot, clip(5), LOOP);
  assert.equal(ref.length, 10);
  // steps 0-1 stand; 2-3 lead (frames 1, 2); 4-8 clip; 9-10 return; 11-12 hold; 13 cue again.
  assert.equal(phasesOf(ref, robot, 26), 'ssllcccccrrssllcccccrrssll');
});

test('loop matches CuedClipReference cued by hand at the same steps', () => {
  const robot = new Robot(toySpec());
  const loop = new LoopReference(robot, clip(5), LOOP);
  const cued = new CuedClipReference(robot, clip(5), LOOP);
  for (let s = 0; s < 30; s++) {
    cued.cue(s, (s >= 2) + (s >= 13) + (s >= 24));
    const q = new Float64Array(29);
    close(loop.window(s, zq(0.5 + 0.01 * s), q), cued.window(s, zq(0.5 + 0.01 * s), q), 0, `step ${s}`);
  }
});

test('loop: the stand is placed where the clip ends and the next motion starts there', () => {
  const robot = new Robot(toySpec());
  const ref = new LoopReference(robot, clip(5), LOOP);
  let w;
  for (let s = 0; s < 12; s++) w = ref.window(s, zq(0.5), new Float64Array(29));
  const lastClip = placed(clip(5), 0.5, [0, 0]).subarray(4 * STATE_DIM, 5 * STATE_DIM);
  close(w.subarray(58, 60), lastClip.subarray(58, 60), 1e-12);
  // The stand at heading of the clip's last frame.
  assert.ok(Math.abs(heading(w.subarray(100, 104)) - heading(lastClip.subarray(100, 104))) < 1e-12);
});

test('a clip switch waits for the next cue', () => {
  const robot = new Robot(toySpec());
  const a = clip(5);
  const ref = new LoopReference(robot, a, LOOP);
  const same = new LoopReference(robot, a, LOOP);
  const seq = phasesOf(ref, robot, 26, (s) => {
    if (s === 5) ref.setClip(clip(8));
    if (s < 13) close(ref.window(s, zq(0.5), new Float64Array(29)), same.window(s, zq(0.5), new Float64Array(29)), 0);
  });
  // The second cue (step 13) plays the 8-frame clip.
  assert.equal(seq.slice(13), 'llccccccccrrs');
});

test('reset: back to the stand at the new heading, first cue again', () => {
  const robot = new Robot(toySpec());
  const ref = new LoopReference(robot, clip(5), LOOP);
  for (let s = 0; s < 6; s++) ref.window(s, zq(0.5), new Float64Array(29));
  assert.equal(ref.phase, 'clip');
  ref.reset();
  const w = ref.window(0, zq(-1.0), new Float64Array(29));
  assert.equal(ref.phase, 'stand');
  assert.ok(Math.abs(heading(w.subarray(100, 104)) + 1.0) < 1e-12);
  ref.window(1, zq(-1.0), new Float64Array(29));
  ref.window(2, zq(-1.0), new Float64Array(29));
  assert.equal(ref.phase, 'lead');
});

test('five points of a window, moved to an origin or anchored at a pelvis', () => {
  const w = referenceWindow(clip(12), 0);
  const p = fivePoints(w, [10, 20]);
  assert.equal(p.length, GOAL_K * 5 * 3);
  close(p.subarray(0, 3), [w[58] + 10, w[59] + 20, w[60]]);
  close(p.subarray(3, 6), [w[58 + 9] + 10, w[59 + 9] + 20, w[60 + 9]]); // body 3, left ankle
  const k1 = 15 + 3 * 4; // frame 1, right wrist (body 13)
  close(p.subarray(k1, k1 + 3), [w[240 + 58 + 39] + 10, w[240 + 59 + 39] + 20, w[240 + 60 + 39]]);
  const a = anchorFivePoints(p, [1, 2]);
  close(a.subarray(0, 3), [1, 2, w[60]], 1e-12);
});

test('loop reference: five-point targets of the current window and the preview, at its origin', () => {
  const robot = new Robot(toySpec());
  const ref = new LoopReference(robot, clip(5), { ...LOOP, origin: [0, -1.5] });
  const w = ref.window(0, zq(0.5), new Float64Array(29));
  close(ref.fivePointTargets(), fivePoints(w, [0, -1.5]));
});

test('clip states from the packed buffer, float32 -> float64', () => {
  const f = new Float32Array(3 * STATE_DIM).map((_, i) => i * 0.5);
  const s = statesFromClip(f.buffer, { offset: STATE_DIM * 4, frames: 2 }, STATE_DIM);
  assert.ok(s instanceof Float64Array);
  assert.equal(s.length, 2 * STATE_DIM);
  assert.equal(s[0], STATE_DIM * 0.5);
});

test('policy inputs: history and reference restart when the step goes back', () => {
  const robot = new Robot(toySpec());
  const inputs = new PolicyInputs(robot, new LoopReference(robot, clip(5), LOOP), { historyLen: 2 });
  const meas = (v) => [[1, 0, 0, 0], [0, 0, v], new Float64Array(29).fill(v), new Float64Array(29)];
  let x = inputs.step(0, ...meas(1));
  assert.equal(x.obs_history.length, 2 * PROPRIO_DIM);
  assert.deepEqual(Object.keys(x), ['obs_history', 'goal_motion']);
  inputs.setAction(new Float32Array(29).fill(0.25));
  x = inputs.step(1, ...meas(2));
  assert.equal(x.obs_history[PROPRIO_DIM + 92], 0.25); // the last action, newest frame
  assert.equal(x.obs_history[60], 1); // gyro z of the older frame
  x = inputs.step(0, ...meas(3));
  assert.equal(x.obs_history[60], 3);
  assert.equal(x.obs_history[92], 0); // last action cleared
  // The IMU quaternion is normalised first.
  const y = inputs.step(1, [0, 2, 0, 0], [0, 0, 3], new Float64Array(29).fill(3), new Float64Array(29));
  close(y.obs_history.subarray(PROPRIO_DIM + 61, PROPRIO_DIM + 64), [0, 0, 1], 1e-7); // upside down
});

// A human frame: slot s at pelvis + (0.05 s along the heading, 0, -0.02 s), every slot facing
// `yaw` in the robot's convention (its orientation times UNDO is zq(yaw)).
function humanFrame(px, py, yaw) {
  const f = new Float64Array(HUMAN_FRAME_DIM);
  const q = quatMul(zq(yaw), quatInv(UNDO));
  for (let s = 0; s < 24; s++) {
    f.set([px + Math.cos(yaw) * 0.05 * s, py + Math.sin(yaw) * 0.05 * s, 0.9 - 0.02 * s], 7 * s);
    f.set(q, 7 * s + 3);
  }
  return f;
}
const humanClip = (frames) => {
  const out = new Float64Array(frames * HUMAN_FRAME_DIM);
  for (let f = 0; f < frames; f++) out.set(humanFrame(1 + 0.02 * f, 2, 0.8 + 0.01 * f), f * HUMAN_FRAME_DIM);
  return out;
};

test('human: placed frames face the yaw at `at`, the goal is position-free, the five-point mask', () => {
  const p = placedHuman(humanClip(4), -0.4, [2, 3]);
  close(p.subarray(0, 2), [2, 3], 1e-12);
  assert.ok(Math.abs(humanHeading(p) + 0.4) < 1e-12);
  const w = humanWindow(humanClip(12), 1);
  const moved = turnHuman(w, 0, [-3, 4]);
  for (const m of ['full', 'five']) close(humanFeatures(moved, pointMask(m)), humanFeatures(w, pointMask(m)), 1e-6, m);
  assert.equal(humanFeatures(w, pointMask('full')).length, GOAL_K * 24 * 9);
  assert.deepEqual([...pointMask('five')].flatMap((v, s) => (v ? [s] : [])), HUMAN_FIVE);
  // Spine3 against a torso facing the same way: the identity.
  close(headingFeatures(zq(0.8), humanWindow(humanClip(1), 0), pointMask('full')).subarray(0, 6), [1, 0, 0, 1, 0, 0], 1e-12);
});

test('loop reference: the human track plays in step with the robot clip, from the human stand', () => {
  const robot = new Robot(toySpec());
  const ref = new LoopReference(robot, clip(5), { ...LOOP, human: humanClip(5), humanStand: humanFrame(0, 0, 0) });
  const inputs = new PolicyInputs(robot, ref, { historyLen: 2 });
  const meas = [zq(1.0), [0, 0, 0], new Float64Array(29), new Float64Array(29)];
  const phases = [];
  for (let s = 0; s < 14; s++) {
    const x = inputs.step(s, ...meas, 0, s % 2 ? 'five' : 'full');
    assert.deepEqual(Object.keys(x), ['obs_history', 'goal_human', 'human_point_mask', 'human_frame_mask', 'human_heading', 'human_heading_mask']);
    if (s === 0) assert.ok(Math.abs(humanHeading(ref.humanWindow) - heading(zq(1.0))) < 1e-12, 'the stand faces the robot');
    phases.push(ref.phase);
  }
  assert.deepEqual(phases.slice(2, 12), ['lead', 'lead', 'clip', 'clip', 'clip', 'clip', 'clip', 'return', 'return', 'stand']);
  assert.throws(() => ref.setClip(clip(6), humanClip(5)), /human frames 5, robot clip 6/);
  assert.throws(() => new PolicyInputs(robot, new StandReference(robot)).step(0, ...meas, 0, 'full'), /human track/);
});
