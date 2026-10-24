// The worker's command side (src/live/driver.js) on a fake loop and fake timers: commands never
// run inside a control step (even across pause, a queued command and resume), messages are
// checked, a run error stops the loop until restart, and clips are fetched when first selected.
import test from 'node:test';
import assert from 'node:assert/strict';
import { Driver } from '../src/live/driver.js';

// A loop whose tick() stays in flight until the test releases it.
class FakeLoop {
  constructor() {
    this.paused = false;
    this.busy = false;
    this.dtMs = 20;
    this.sim = { time: 0 };
    this.log = [];
    this.pending = [];
    this.failNext = false;
  }
  tick(now, onStep) {
    if (this.paused || this.busy) return Promise.resolve(0);
    this.busy = true;
    this.log.push('tick');
    return new Promise((res, rej) => this.pending.push({ res, rej, onStep })).finally(() => { this.busy = false; });
  }
  /** The tick in flight runs n control steps (onStep after each) and resolves. */
  release(n = 1) {
    const p = this.pending.shift();
    for (let i = 0; i < n; i++) p.onStep?.();
    p.res(n);
  }
  fail(e) { this.pending.shift().rej(e); }
  nextDue() { return this.paused ? null : 0; }
  pause() { this.paused = true; }
  resume() { this.paused = false; }
  restart() { assert.ok(!this.busy, 'restart ran during a control step'); this.log.push('restart'); }
  setClip(states, id) { assert.ok(!this.busy, 'setClip ran during a control step'); this.log.push(`clip ${id} ${states}`); }
  setInterface(i) { assert.ok(!this.busy, 'setInterface ran during a control step'); this.log.push(`iface ${i}`); }
}

class FakeTimers {
  constructor() { this.next = 1; this.live = new Map(); }
  set(fn, ms) { const id = this.next++; this.live.set(id, fn); return id; }
  clear(id) { this.live.delete(id); }
  /** Fires every pending timer once (as the event loop would). */
  fire() { const fns = [...this.live.values()]; this.live.clear(); return Promise.all(fns.map((f) => f())); }
}

const flush = () => new Promise((r) => setImmediate(r));

function make({ clips = ['A', null, 'C'], loadClip } = {}) {
  const posted = [];
  const timers = new FakeTimers();
  const loop = new FakeLoop();
  let frames = 0;
  const d = new Driver({
    post: (m) => posted.push(m),
    postFrame: () => { frames++; },
    loadClip: loadClip ?? (async (i) => `S${i}`),
    now: () => 0,
    timers,
  });
  return { d, loop, timers, posted, start: () => d.start(loop, [...clips], 0), frames: () => frames };
}

test('commands before ready are refused, and a second init too', () => {
  const { d, posted, start } = make();
  d.handle({ type: 'restart' });
  assert.match(posted.at(-1).message, /not ready/);
  assert.equal(posted.at(-1).stage, 'restart');
  assert.equal(d.claimInit(), true);
  d.handle({ type: 'setClip', clip: 1 });
  assert.match(posted.at(-1).message, /not ready/, 'still loading');
  assert.equal(d.claimInit(), false);
  assert.match(posted.at(-1).message, /already initialised/);
  start();
  assert.equal(d.claimInit(), false, 'refused after ready as well');
});

test('setClip and setInterface check their arguments', () => {
  const { d, posted, start, loop } = make();
  d.claimInit();
  start();
  for (const clip of [3, -1, 1.5, '1', null, undefined]) {
    posted.length = 0;
    d.handle({ type: 'setClip', clip });
    assert.equal(posted.length, 1, `clip ${clip}`);
    assert.match(posted[0].message, /no clip/);
  }
  posted.length = 0;
  d.handle({ type: 'setInterface', iface: 'five' });
  assert.match(posted[0].message, /no interface/);
  d.handle({ type: 'bogus' });
  assert.match(posted.at(-1).message, /unknown message/);
  d.handle(null);
  assert.match(posted.at(-1).message, /unknown message/);
  assert.deepEqual(loop.log, [], 'nothing reached the loop');
});

test('pause, a queued restart and resume never run the restart inside a control step', async () => {
  const { d, loop, timers, start } = make();
  d.claimInit();
  start();
  const first = timers.fire(); // pump 1: its control step is in flight
  await flush();
  assert.deepEqual(loop.log, ['tick']);
  d.handle({ type: 'pause' });
  d.handle({ type: 'restart' }); // queued: a step is in flight
  d.handle({ type: 'setClip', clip: 2 });
  d.handle({ type: 'resume' }); // sets a timer
  assert.equal(timers.live.size, 1);
  await timers.fire(); // pump 2 must not run the queued commands while pump 1's step runs
  await flush();
  assert.deepEqual(loop.log, ['tick'], 'nothing ran mid-step');
  loop.release();
  await first;
  assert.deepEqual(loop.log, ['tick', 'restart', 'clip 2 C'], 'the queued commands ran once, after the step');
  assert.equal(timers.live.size, 1, 'and the loop goes on');
  timers.fire(); // the next pump ticks again (left in flight)
  await flush();
  assert.equal(loop.log.at(-1), 'tick');
});

test('a run error stops the loop until restart; pause and resume do not clear it', async () => {
  const { d, loop, timers, posted, start } = make();
  d.claimInit();
  start();
  const p = timers.fire();
  await flush();
  loop.fail(new Error('boom'));
  await p;
  assert.equal(posted.at(-1).stage, 'run');
  assert.match(posted.at(-1).message, /boom/);
  assert.match(posted.at(-1).message, /restart to run again/);
  assert.doesNotMatch(posted.at(-1).message, /resume/);
  assert.equal(timers.live.size, 0, 'stopped');
  d.handle({ type: 'resume' }); // not paused: nothing to resume
  assert.equal(timers.live.size, 0, 'resume does not run it again');
  d.handle({ type: 'pause' });
  d.handle({ type: 'resume' }); // the page hidden and shown again, say
  assert.equal(timers.live.size, 0, 'pause and resume do not run it again either');
  assert.equal(d.halted, true);
  assert.deepEqual(loop.log, ['tick']);
  d.handle({ type: 'restart' });
  assert.equal(loop.log.at(-1), 'restart');
  assert.equal(d.halted, false);
  assert.equal(timers.live.size, 1, 'restart schedules again');
  const q = timers.fire();
  await flush();
  assert.equal(loop.log.at(-1), 'tick', 'and the loop runs');
  loop.release();
  await q;
});

test('a clip not loaded yet is fetched once, and only the last selection is applied', async () => {
  const calls = [];
  const gates = {};
  const loadClip = (i) => { calls.push(i); return new Promise((res, rej) => { gates[i] = { res, rej }; }); };
  const { d, loop, posted, start } = make({ clips: ['A', null, null, 'D'], loadClip });
  d.claimInit();
  start();
  d.handle({ type: 'setClip', clip: 1 });
  d.handle({ type: 'setClip', clip: 1 });
  await flush();
  assert.deepEqual(calls, [1], 'fetched once');
  d.handle({ type: 'setClip', clip: 2 });
  await flush();
  gates[1].res('S1');
  await flush();
  assert.deepEqual(loop.log, [], 'clip 1 arrived after clip 2 was selected: not applied');
  gates[2].res('S2');
  await flush();
  assert.deepEqual(loop.log, ['clip 2 S2']);
  d.handle({ type: 'setClip', clip: 1 }); // loaded now: no fetch
  assert.deepEqual(calls, [1, 2]);
  assert.equal(loop.log.at(-1), 'clip 1 S1');
  d.handle({ type: 'setClip', clip: 2 });
  assert.equal(loop.log.at(-1), 'clip 2 S2');
  // A failed fetch is reported and the clip playing stays.
  const { d: d2, loop: l2, posted: p2, start: s2 } = make({ loadClip: async () => { throw new Error('HTTP 404'); } });
  d2.claimInit();
  s2();
  d2.handle({ type: 'setClip', clip: 1 });
  await flush();
  await flush();
  assert.deepEqual(l2.log, []);
  assert.equal(p2.at(-1).stage, 'clip');
  assert.equal(p2.at(-1).clip, 1);
  assert.match(p2.at(-1).message, /404/);
  assert.equal(posted.filter((m) => m.type === 'error').length, 0);
});

test('perturb and perturbEnd reach the sim at once, checked', () => {
  const { d, loop, posted, start } = make();
  const calls = [];
  loop.sim.setPerturb = (...a) => calls.push(['set', ...a]);
  loop.sim.endPerturb = () => calls.push(['end']);
  d.handle({ type: 'perturb', robot: 0, body: 1, local: [0, 0, 0], target: [0, 1, 1] });
  assert.match(posted.at(-1).message, /not ready/);
  d.claimInit();
  start();
  posted.length = 0;
  d.handle({ type: 'perturb', robot: 2, body: 70, local: [0.1, 0, 0], target: [1, 2, 3] });
  d.handle({ type: 'perturbEnd' });
  assert.deepEqual(calls, [['set', 2, 70, [0.1, 0, 0], [1, 2, 3]], ['end']]);
  assert.deepEqual(posted, []);
  for (const bad of [
    { robot: 1.5, body: 1, local: [0, 0, 0], target: [0, 0, 0] },
    { robot: 0, body: '1', local: [0, 0, 0], target: [0, 0, 0] },
    { robot: 0, body: 1, local: [0, 0], target: [0, 0, 0] },
    { robot: 0, body: 1, local: [0, 0, 0], target: [0, NaN, 0] },
    { robot: 0, body: 1, local: [0, 0, 0] },
  ]) {
    posted.length = 0;
    d.handle({ type: 'perturb', ...bad });
    assert.equal(posted.length, 1, JSON.stringify(bad));
    assert.equal(posted[0].stage, 'perturb');
  }
  assert.equal(calls.length, 2, 'nothing bad reached the sim');
  assert.deepEqual(loop.log, [], 'the loop is not touched');
});

test('a perturb mid-step is applied without waiting (physics runs only after the policy)', async () => {
  const { d, loop, timers, start } = make();
  const calls = [];
  loop.sim.setPerturb = (...a) => calls.push(a);
  d.claimInit();
  start();
  timers.fire(); // a control step in flight
  await flush();
  d.handle({ type: 'perturb', robot: 0, body: 1, local: [0, 0, 0], target: [0, 1, 1] });
  assert.equal(calls.length, 1);
  loop.release();
});

test('a tick that runs several control steps posts a frame after each', async () => {
  const { d, loop, timers, start, frames } = make();
  d.claimInit();
  start();
  const p = timers.fire();
  await flush();
  loop.release(3);
  await p;
  assert.equal(frames(), 3, 'three steps, three frames');
  const q = timers.fire();
  await flush();
  loop.release(0);
  await q;
  assert.equal(frames(), 3, 'a tick that ran nothing posts nothing');
});

test('while it runs below real time a tick catches up at most 2 steps, and 4 again once it keeps up', async () => {
  let now = 0;
  const posted = [];
  const timers = new FakeTimers();
  const loop = new FakeLoop();
  loop.maxCatchUp = 4;
  const d = new Driver({ post: (m) => posted.push(m), postFrame: () => {}, loadClip: async () => null, now: () => now, timers });
  d.claimInit();
  d.start(loop, ['A'], 0);
  // One pump per 2 s window of wall time, advancing the sim by `sim` seconds.
  const window = async (sim) => {
    now += 2000;
    loop.sim.time += sim;
    const p = timers.fire();
    await flush();
    loop.release(1);
    await p;
  };
  await window(0); // starts the rate window
  await window(1.0); // rtf 0.5
  assert.equal(loop.maxCatchUp, 4, 'one slow window is not enough');
  await window(1.0);
  assert.equal(d.rate.rtf, 0.5);
  assert.equal(loop.maxCatchUp, 2, 'two slow windows in a row');
  await window(1.0);
  assert.equal(loop.maxCatchUp, 2);
  await window(2.0); // real time again
  assert.equal(loop.maxCatchUp, 4);
  assert.deepEqual(posted, []);
});
