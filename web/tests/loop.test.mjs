// The live loop (src/live/loop.js) on a fake sim and policy: the fixed-step clock, clip switches
// at the next cue, a fall resetting only the robot that fell, and the re-home when the robots
// drift apart.
import test from 'node:test';
import assert from 'node:assert/strict';
import { STATE_DIM } from '../src/live/obs.js';
import { LiveLoop, stepLoop } from '../src/live/loop.js';
import { Driver } from '../src/live/driver.js';

const zq = (yaw) => [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];

// A reference state: joints j + 0.01 i, bodies along x from (px, py), facing yaw.
function state(px, py, yaw, j = 0) {
  const s = new Float64Array(STATE_DIM);
  for (let i = 0; i < 29; i++) s[i] = j + 0.01 * i;
  for (let b = 0; b < 14; b++) {
    s[58 + 3 * b] = px + Math.cos(yaw) * 0.1 * b;
    s[58 + 3 * b + 1] = py + Math.sin(yaw) * 0.1 * b;
    s[58 + 3 * b + 2] = 0.8 - 0.05 * b;
    s.set(zq(yaw), 100 + 4 * b);
  }
  return s;
}
// A clip whose joints are base + frame (so a window says which clip and frame it came from).
const clip = (frames, base) => {
  const out = new Float64Array(frames * STATE_DIM);
  for (let f = 0; f < frames; f++) out.set(state(0.02 * f, 0, 0, base + f), f * STATE_DIM);
  return out;
};

// A toy robot (as in live.test.mjs): identity joint map, slot 0 = root = IMU body.
const toySpec = (name) => ({
  name, kpGoalVersion: 2,
  jointNames: Array.from({ length: 29 }, (_, i) => `j${i}`),
  defaultJointPos: Array(29).fill(0),
  actionScale: Array(29).fill(0.5),
  perm: Array.from({ length: 29 }, (_, i) => i), sign: Array(29).fill(1), offset: Array(29).fill(0),
  fk: { parent: [-1, 0], quat: [[1, 0, 0, 0], [1, 0, 0, 0]], joints: [[], [{ native: 0, axis: [0, 0, 1], ref: 0 }]], slot0: 0, anchor: 1, imu: 0 },
  stand: { state: Array.from(state(0, 0, 0, -1)) },
});

// The sim's interface, scripted: robot r's home stand is (0, 1.5 r) and `pos[r]` where it is
// now (a test moves it); a reset puts it back home. `falls` = {step: [robots]} makes those
// robots report a fall after that control step.
class FakeSim {
  constructor(n, falls = {}) {
    this.robots = Array.from({ length: n }, (_, r) => ({ name: `toy${r}` }));
    this.numRobots = n;
    this.falls = falls;
    this.steps = 0;
    this.resets = [];
    this.fullResets = 0;
    this.targets = Array.from({ length: n }, () => null);
    this.pos = Array.from({ length: n }, (_, r) => [0, 1.5 * r]);
    this.time = 0;
  }
  slot0(r) { return { pos: Float64Array.of(this.pos[r][0], this.pos[r][1], 0.8), quat: Float64Array.of(1, 0, 0, 0) }; }
  measure(r, out = {}) {
    out.quat = Float64Array.of(1, 0, 0, 0);
    out.gyro = new Float64Array(3);
    out.q = new Float64Array(29);
    out.q[0] = 0.01 * this.steps; // the measurements change from step to step
    out.dq = new Float64Array(29);
    return out;
  }
  setTargets(r, t) { this.targets[r] = Float64Array.from(t); }
  step() { this.steps++; this.time += 0.02; }
  fallen(r) { return (this.falls[this.steps] ?? []).includes(r); }
  resetRobot(r) { this.resets.push([this.steps, r]); this.pos[r] = [0, 1.5 * r]; }
  resetAll() { this.fullResets++; this.time = 0; this.pos = this.pos.map((_, r) => [0, 1.5 * r]); }
}

// The policy's interface: zeros, `delayMs` late; it fails the test if two calls overlap.
class FakePolicy {
  constructor(delayMs = 0) { this.delayMs = delayMs; this.calls = []; this.running = 0; }
  async act(r, inputs, iface) {
    assert.equal(this.running, 0, 'act calls overlap');
    this.running++;
    this.calls.push({ r, iface, goal: Float32Array.from(inputs.goal_motion) });
    if (this.delayMs) await new Promise((res) => setTimeout(res, this.delayMs));
    else await null;
    this.running--;
    return new Float32Array(29);
  }
}

// 5 frames per lead-in, return and hold; first cue after 5 steps.
const LOOP = { leadInS: 0.1, returnS: 0.1, holdS: 0.1 };
const make = (n, clips, { falls, delayMs, ...opts } = {}) => {
  const sim = new FakeSim(n, falls);
  const policy = new FakePolicy(delayMs);
  const specs = Array.from({ length: n }, (_, r) => toySpec(`toy${r}`));
  return { sim, policy, loop: new LiveLoop(sim, policy, specs, Array(n).fill(clips), { loop: LOOP, ...opts }) };
};

test('stepLoop caps catch-up steps', async () => {
  const clock = { t: null };
  const opt = { dtMs: 20, maxSteps: 4 };
  assert.equal(stepLoop(clock, 1000, opt), 0, 'the first call starts the clock');
  assert.equal(stepLoop(clock, 1019, opt), 0);
  assert.equal(stepLoop(clock, 1050, opt), 2);
  assert.equal(clock.t, 1040, 'the remainder stays due');
  assert.equal(stepLoop(clock, 1050 + 1000, opt), 4, 'a long stall runs at most maxSteps');
  assert.equal(clock.t, 2050, 'and the rest of the backlog is dropped');
  assert.equal(stepLoop(clock, 2069, opt), 0);
  assert.equal(stepLoop(clock, 2070, opt), 1);

  // The loop: a tick after a stall runs maxCatchUp control steps, not the backlog; pause and
  // resume do not replay the paused time.
  const { sim, loop } = make(2, clip(20, 10), { maxCatchUp: 3 });
  assert.equal(await loop.tick(0), 0);
  assert.equal(await loop.tick(500), 3);
  assert.equal(sim.steps, 3);
  assert.equal(await loop.tick(519), 0);
  assert.equal(await loop.tick(520), 1);
  loop.pause();
  assert.equal(await loop.tick(10000), 0, 'paused: nothing runs');
  loop.resume();
  assert.equal(await loop.tick(20000), 0, 'resume restarts the clock');
  assert.equal(await loop.tick(20020), 1);
  assert.equal(sim.steps, 5);

  // A tick that comes while one is running does nothing (the steps are not overlapped).
  const slow = make(3, clip(20, 10), { maxCatchUp: 2, delayMs: 2 });
  await slow.loop.tick(0);
  const a = slow.loop.tick(40);
  assert.equal(await slow.loop.tick(1000), 0);
  assert.equal(await a, 2);
  assert.equal(slow.policy.calls.length, 6);
});

test('switch waits for the stand', async () => {
  const A = clip(12, 100), B = clip(12, 500);
  const { loop } = make(1, A, { clipId: 'A' });
  const R = loop.robots[0];
  const frame0 = () => R.inputs.lastWindow[0]; // joint 0 of the window's frame 0
  const phases = [];
  const run = async (n) => { for (let i = 0; i < n; i++) { await loop.controlStep(); phases.push([R.reference.phase, frame0()]); } };
  await run(5 + 5 + 3); // the stand, the lead-in, 3 frames into clip A
  assert.equal(R.reference.phase, 'clip');
  assert.equal(loop.status()[0].clip, 'A');
  loop.setClip([B], 'B');
  assert.equal(loop.status()[0].nextClip, 'B');
  const at = phases.length;
  await run(60);
  const after = phases.slice(at);
  // A plays on to its end, returns to the stand and holds it; only the next cue plays B.
  const cue = after.findIndex(([p]) => p === 'lead');
  assert.ok(cue > 0, 'cued again');
  const beforeCue = after.slice(0, cue);
  assert.ok(beforeCue.filter(([p]) => p === 'clip').every(([, j]) => j >= 100 && j < 200), 'the clip phase before the cue is A');
  assert.ok(beforeCue.some(([p, j]) => p === 'clip' && j === 111), 'A plays to its last frame');
  assert.ok(beforeCue.some(([p]) => p === 'return'), 'A returns to the stand');
  const standRun = beforeCue.filter(([p]) => p === 'stand').length;
  assert.equal(standRun, 5, 'the stand is held for holdS before the cue');
  assert.ok(beforeCue.findLastIndex(([p]) => p !== 'stand') < cue - 5, 'the hold is the last thing before the cue');
  const clipB = after.slice(cue).filter(([p]) => p === 'clip');
  assert.ok(clipB.length > 0 && clipB.every(([, j]) => j >= 500 && j < 600), 'the next cue plays B');
  assert.equal(loop.status()[0].clip, 'B');
  assert.equal(R.motions, 2, 'A, then B, played to the stand (73 steps)');
});

test('fall resets one robot', async () => {
  const { sim, loop } = make(3, clip(30, 10), { falls: { 8: [1] } });
  for (let i = 0; i < 7; i++) await loop.controlStep();
  assert.deepEqual(loop.robots.map((R) => R.stepId), [7, 7, 7]);
  assert.deepEqual(loop.robots.map((R) => R.reference.phase), ['lead', 'lead', 'lead'], 'cued at step 5');
  await loop.controlStep(); // robot 1 falls in this one
  assert.deepEqual(sim.resets, [[8, 1]], 'only robot 1 is reset, after the physics step');
  assert.equal(sim.fullResets, 0);
  assert.deepEqual(loop.robots.map((R) => R.falls), [0, 1, 0]);
  assert.deepEqual(loop.robots.map((R) => R.stepId), [8, 0, 8], 'robot 1 restarts at step 0');
  const R1 = loop.robots[1];
  assert.equal(R1.reference.nextCue, R1.reference.firstCue, 'its loop starts over: first cue re-armed');
  assert.deepEqual(R1.reference.origin, [0, 1.5], 'its targets are placed at its spot again');
  await loop.controlStep();
  assert.deepEqual(loop.robots.map((R) => R.reference.phase), ['lead', 'stand', 'lead'], 'the others play on');
  assert.deepEqual(loop.robots.map((R) => R.stepId), [9, 1, 9]);
  // Robot 1's policy inputs started over: its history holds the post-reset frame in every slot;
  // robot 0's still holds the steps before.
  const slots = (R) => {
    const f = R.inputs.history.frames, n = f.length / 10;
    return Array.from({ length: 10 }, (_, k) => Array.from(f.subarray(k * n, k * n + n)).join());
  };
  assert.equal(new Set(slots(R1)).size, 1);
  assert.ok(new Set(slots(loop.robots[0])).size > 1);
  assert.equal(R1.inputs.prevStep, 0);
});

test('after a fall the robot reports the clip its next cue plays', async () => {
  const { loop } = make(2, clip(30, 10), { falls: { 8: [1] }, clipId: 'A' });
  for (let i = 0; i < 7; i++) await loop.controlStep();
  loop.setClip(Array(2).fill(clip(30, 500)), 'B');
  assert.deepEqual(loop.status().map((s) => [s.clip, s.nextClip]), [['A', 'B'], ['A', 'B']], 'A plays on');
  await loop.controlStep(); // robot 1 falls: its motion of A is gone, it stands until its cue plays B
  assert.deepEqual(loop.status().map((s) => [s.phase, s.clip]), [['lead', 'A'], ['stand', 'B']]);
  loop.restart();
  assert.deepEqual(loop.status().map((s) => s.clip), ['B', 'B'], 'a restart drops the motion playing too');
});

// Steps a loop one control step at a time, logging per step: the phases, the full resets so far,
// `rehoming`, and the step ids.
async function stepLog(loop, sim, n, log = []) {
  for (let i = 0; i < n; i++) {
    await loop.controlStep();
    log.push({
      step: sim.steps,
      phases: loop.robots.map((R) => R.reference.phase),
      resets: sim.fullResets,
      rehoming: loop.rehoming,
      ids: loop.robots.map((R) => R.stepId),
    });
  }
  return log;
}

// LOOP with a 12-frame clip: first cue after 5 steps, each motion 5 + 12 + 5 = 22 frames, then a
// hold of 5 steps before the next cue: cues at steps 5, 32, 59, ... of a robot's step count.
const REHOME = { spread: 2, fadeS: 0.06 }; // a 3-step fade

test('a spread past the threshold re-homes all robots once, at the next cue', async () => {
  const { sim, loop } = make(3, clip(12, 10), { rehome: REHOME });
  const twin = make(3, clip(12, 10), { rehome: null }); // the same loop that never re-homes
  const both = async (n) => { await stepLog(twin.loop, twin.sim, n); return stepLog(loop, sim, n); };

  // A spread under the threshold does nothing; the robots move together does not count.
  sim.pos = [[3, 0], [3, 1.5], [4.5, 3]]; // all 3 m forward, robot 2 another 1.5 m
  let log = await both(40);
  assert.equal(sim.fullResets, 0);
  assert.equal(loop.rehomeArmed, false);
  assert.ok(Math.abs(loop.spread() - 1.5) < 1e-12);

  // Mid-motion (step 40 is in the clip of the second motion) robot 0 moves 2.5 m away.
  assert.deepEqual(log.at(-1).phases, ['clip', 'clip', 'clip']);
  sim.pos[0] = [3, -2.5];
  log = await both(1);
  assert.equal(loop.rehomeArmed, true, 'armed by the step after the move');
  assert.equal(sim.fullResets, 0, 'but nothing happens mid-motion');

  // The twin's next cue: the step at which its robots enter the lead-in again.
  const twinLog = await stepLog(twin.loop, twin.sim, 60);
  const cueAt = twinLog.find((e, i) => i > 0 && e.phases[0] === 'lead' && twinLog[i - 1].phases[0] === 'stand').step;
  log = log.concat(await stepLog(loop, sim, 60));
  const cut = log.findIndex((e) => e.resets === 1);
  assert.ok(cut > 0, 're-homed');
  // The cut comes after the step before the cue: the step the twin cues in starts the stand
  // again, every robot at step 0 of its loop.
  assert.equal(log[cut].step, cueAt - 1, `re-home after step ${cueAt - 1}, the cue was at ${cueAt}`);
  assert.deepEqual(log[cut].ids, [0, 0, 0]);
  assert.ok(log.slice(0, cut).every((e) => e.resets === 0), 'none before the cue');
  assert.ok(log.slice(1, cut).every((e) => e.phases.every((p) => p !== 'lead' || e === log[0])), 'no new motion while armed');
  assert.deepEqual(log[cut - 1].phases, ['stand', 'stand', 'stand'], 'every robot waited at the stand');
  // The fade: `rehoming` is set for the fade's 3 steps before the cut.
  assert.deepEqual(log.slice(cut - 4, cut).map((e) => e.rehoming), [false, true, true, true]);
  assert.equal(log[cut].rehoming, false, 'cleared by the cut');
  assert.deepEqual(sim.pos, [[0, 0], [0, 1.5], [0, 3]], 'home stands');
  assert.deepEqual(loop.robots.map((R) => R.reference.origin), [[0, 0], [0, 1.5], [0, 3]]);
  assert.ok(sim.time > 1, 'the sim clock goes on through a re-home');
  assert.equal(loop.rehomes, 1);
  assert.equal(loop.rehomeArmed, false);

  // The loops go on from home together, with no further re-home.
  log = await stepLog(loop, sim, 120);
  assert.equal(sim.fullResets, 1, 'exactly one re-home');
  assert.equal(loop.rehomes, 1);
  assert.ok(log.some((e) => e.phases.every((p) => p === 'clip')), 'the clip plays again');
  assert.ok(loop.robots.every((R) => R.motions >= 4));
});

test('a re-home waits for every robot to be due at its cue', async () => {
  // Robot 1 falls at step 20: its loop starts over (cue at step 25, its motion ends at 47),
  // 20 steps behind robot 0 (motion 5 to 27, next cue at 32). The spread passes the threshold
  // at step 31, while robot 1 is mid-motion.
  const { sim, loop } = make(2, clip(12, 10), { rehome: REHOME, falls: { 20: [1] } });
  const pre = await stepLog(loop, sim, 30);
  assert.deepEqual(pre.at(-1).phases, ['stand', 'clip']);
  sim.pos[1] = [2.5, 1.5];
  const log = await stepLog(loop, sim, 80);
  const cut = log.findIndex((e) => e.resets === 1);
  assert.ok(cut > 0);
  // Robot 0 finished its motion first and held at the stand past its own cue, until robot 1
  // was due too.
  const r0Waits = log.slice(0, cut).filter((e) => e.phases[0] === 'stand').length;
  assert.ok(r0Waits > 10, `robot 0 waited ${r0Waits} steps at the stand`);
  assert.ok(log.slice(1, cut).every((e) => !(e.phases[0] === 'lead')), 'robot 0 started no motion');
  assert.equal(log[cut - 1].phases[1], 'stand');
  assert.ok(log.slice(0, cut).some((e) => e.phases[1] === 'return'), 'robot 1 finished its motion first');
  assert.equal(sim.fullResets, 1);
  assert.deepEqual(log[cut].ids, [0, 0]);
  // A restart while armed disarms it.
  const again = make(2, clip(12, 10), { rehome: REHOME });
  await stepLog(again.loop, again.sim, 10);
  again.sim.pos[1] = [2.5, 1.5];
  await stepLog(again.loop, again.sim, 1);
  assert.equal(again.loop.rehomeArmed, true);
  again.loop.restart();
  assert.equal(again.loop.rehomeArmed, false);
  assert.ok(again.loop.robots.every((R) => !R.reference.cueHeld));
  const after = await stepLog(again.loop, again.sim, 40);
  assert.equal(again.sim.fullResets, 1, 'only the restart');
  assert.ok(after.some((e) => e.phases[0] === 'lead'), 'and the loops cue again');
});

test('a fall that pushes the cut back clears rehoming until the cut is near again', async () => {
  // Two robots, cues at steps 5, 32, ... Armed at step 11 (both mid-motion, ending at 27), the
  // cut is due at their next cue (step 32): rehoming from step 29. Robot 1 falls at step 30, so
  // its cue is 5 steps off again: rehoming clears, then comes back for the last 3 steps.
  const { sim, loop } = make(2, clip(12, 10), { rehome: REHOME, falls: { 30: [1] } });
  await stepLog(loop, sim, 10);
  sim.pos[1] = [2.5, 1.5];
  const log = await stepLog(loop, sim, 40);
  const cut = log.findIndex((e) => e.resets === 1);
  assert.ok(cut > 0, 're-homed');
  const fall = log.findIndex((e) => e.step === 30);
  assert.equal(log[fall - 1].rehoming, true, 'the fade had started');
  assert.equal(log[fall].ids[1], 0, 'robot 1 fell and was reset');
  assert.equal(log[fall].rehoming, false, 'the fall put the cut off: the stage is uncovered');
  assert.deepEqual(log.slice(cut - 4, cut).map((e) => e.rehoming), [false, true, true, true], 'the fade again before the cut');
  assert.equal(log[cut].rehoming, false);
  assert.equal(sim.fullResets, 1);
});

test('tick calls onStep after each control step it runs; with the driver, one frame per step', async () => {
  const { sim, loop } = make(2, clip(20, 10), { maxCatchUp: 4 });
  const seen = [];
  assert.equal(await loop.tick(0, () => seen.push(sim.steps)), 0);
  assert.equal(await loop.tick(70, () => seen.push(sim.steps)), 3);
  assert.deepEqual(seen, [1, 2, 3], 'after each step, in order');

  // The worker's driver on the same loop: a tick that runs N steps posts N frames.
  const run = make(2, clip(20, 10), { maxCatchUp: 4 });
  let now = 0;
  const pending = [];
  const frames = [];
  const d = new Driver({
    post: (m) => assert.fail(`unexpected message ${JSON.stringify(m)}`),
    postFrame: () => frames.push(run.loop.steps),
    loadClip: async () => null,
    now: () => now,
    timers: { set: (fn) => { pending.push(fn); return pending.length; }, clear: () => {} },
  });
  d.claimInit();
  d.start(run.loop, [null], 0);
  await pending.shift()(); // starts the clock
  assert.deepEqual(frames, []);
  now = 80; // 4 steps due
  await pending.shift()();
  assert.deepEqual(frames, [1, 2, 3, 4], 'four steps, four frames');
  now = 1000; // a stall: at most maxCatchUp steps, a frame each
  await pending.shift()();
  assert.deepEqual(frames, [1, 2, 3, 4, 5, 6, 7, 8]);
});
