// Drag-to-push (src/live/sim.js): the spring force on the grabbed point is capped, applied only to
// the grabbed robot, and cleared when the drag ends and on every reset. The MuJoCo tests run on
// the real scene (public/live/, made by tools/gen_live_assets.py) and are skipped without it; the
// last one also needs public/live/policy.onnx and runs the shared policy through a push.
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { LiveSim, pushForce } from '../src/live/sim.js';

const CAP = 100; // N: the hard cap the page never shows
const LIVE = join(resolve(dirname(fileURLToPath(import.meta.url)), '..'), 'public/live');
const HAVE_SCENE = existsSync(join(LIVE, 'scene.xml')) && existsSync(join(LIVE, 'scene.json'));
const HAVE_POLICY = HAVE_SCENE && existsSync(join(LIVE, 'policy.onnx')) && existsSync(join(LIVE, 'clips.json'));
const norm = (a, o = 0) => Math.hypot(a[o], a[o + 1], a[o + 2]);

test('pushForce: never over the cap, whatever the distance or speed', () => {
  const out = new Float64Array(3);
  const p = [0.3, -1.2, 0.9];
  let saturated = 0;
  for (const d of [0, 1e-6, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 3, 10, 1e3, 1e9]) {
    for (const dir of [[1, 0, 0], [0, -1, 0], [0, 0, 1], [0.6, 0.48, -0.64]]) {
      for (const v of [[0, 0, 0], [2, -1, 0.5], [-40, 30, 10]]) {
        const target = p.map((x, i) => x + d * dir[i]);
        pushForce(p, v, target, out);
        const f = norm(out);
        assert.ok(Number.isFinite(f), `finite at d=${d}`);
        assert.ok(f <= CAP + 1e-9, `|F| = ${f} at d=${d}, v=${v}`);
        if (f > CAP - 1e-6) saturated++;
      }
    }
  }
  assert.ok(saturated > 0, 'far targets reach the cap');
  // A short pull on a point at rest is the plain spring: along the pull, growing with it.
  pushForce([0, 0, 0], [0, 0, 0], [0.01, 0, 0], out);
  assert.ok(out[0] > 0 && out[1] === 0 && out[2] === 0);
  const f1 = out[0];
  pushForce([0, 0, 0], [0, 0, 0], [0.02, 0, 0], out);
  assert.ok(Math.abs(out[0] - 2 * f1) < 1e-9, 'linear below the cap');
  // The damper resists the point's own velocity.
  pushForce([0, 0, 0], [1, 0, 0], [0, 0, 0], out);
  assert.ok(out[0] < 0);
  pushForce([0, 0, 0], [0, 0, 0], [0, 0, 0], out);
  assert.equal(norm(out), 0, 'no pull, no force');
});

let mujocoP = null;
async function makeSim() {
  if (!mujocoP) mujocoP = import('@mujoco/mujoco').then((m) => m.default());
  const mujoco = await mujocoP;
  const scene = JSON.parse(readFileSync(join(LIVE, 'scene.json'), 'utf8'));
  return new LiveSim(mujoco, readFileSync(join(LIVE, 'scene.xml'), 'utf8'), scene);
}

/** Robot `r`'s bodies (ids) and, per robot, the largest |xfrc_applied| entry over its bodies. */
const bodiesOf = (sim, r) => sim.bodyNames.flatMap((n, b) => (n.startsWith(`${sim.robots[r].name}/`) ? [b] : []));
function xfrcMax(sim, r) {
  const x = sim.data.xfrc_applied;
  let m = 0;
  for (const b of bodiesOf(sim, r)) for (let k = 0; k < 6; k++) m = Math.max(m, Math.abs(x[6 * b + k]));
  return m;
}
const allZero = (sim) => sim.data.xfrc_applied.every((v) => v === 0);

test('a far drag on the G1 is capped, applied at the grab point, and only on the G1', { skip: !HAVE_SCENE && 'no public/live' }, async () => {
  const sim = await makeSim();
  try {
    const g1 = 0;
    const body = sim.robots[g1].imuBodyId; // the pelvis
    const local = [0.05, 0.1, 0.02];
    const far = [0, 1e4, 1]; // 10 km sideways
    sim.setPerturb(g1, body, local, far);
    for (let i = 0; i < 25; i++) {
      sim.step();
      const x = sim.data.xfrc_applied;
      const F = [x[6 * body], x[6 * body + 1], x[6 * body + 2]];
      const f = norm(F);
      assert.ok(f <= CAP + 1e-9, `step ${i}: |F| = ${f}`);
      assert.ok(f > CAP - 1e-6, `step ${i}: a far pull is at the cap (${f})`);
      assert.ok(F[1] > 0.99 * f, 'toward the target');
      for (let r = 1; r < sim.numRobots; r++) assert.equal(xfrcMax(sim, r), 0, `robot ${r} untouched`);
      for (const b of bodiesOf(sim, g1)) {
        if (b !== body) for (let k = 0; k < 6; k++) assert.equal(x[6 * b + k], 0, `body ${b}`);
      }
    }
    // The torque is (point - com) x F, as mj_applyFT puts a force at a point (one physics step
    // older than the current pose, so loosely).
    const d = sim.data;
    const x = d.xfrc_applied;
    const F = [x[6 * body], x[6 * body + 1], x[6 * body + 2]];
    const T = [x[6 * body + 3], x[6 * body + 4], x[6 * body + 5]];
    const R = d.xmat.slice(9 * body, 9 * body + 9);
    const p = [0, 1, 2].map((i) => d.xpos[3 * body + i] + R[3 * i] * local[0] + R[3 * i + 1] * local[1] + R[3 * i + 2] * local[2]);
    const a = [0, 1, 2].map((i) => p[i] - d.xipos[3 * body + i]);
    const want = [a[1] * F[2] - a[2] * F[1], a[2] * F[0] - a[0] * F[2], a[0] * F[1] - a[1] * F[0]];
    for (let k = 0; k < 3; k++) assert.ok(Math.abs(T[k] - want[k]) < 0.5, `torque ${k}: ${T[k]} vs ${want[k]}`);
    assert.ok(norm(T) > 1, 'an off-centre grab twists the body');

    sim.endPerturb();
    assert.ok(allZero(sim), 'cleared at once when the drag ends');
    sim.step();
    assert.ok(allZero(sim), 'and stays cleared');
    assert.equal(sim.perturbing, false);
  } finally {
    sim.dispose();
  }
});

test('restart and a fall reset clear the push; a reset drag stays off until it ends', { skip: !HAVE_SCENE && 'no public/live' }, async () => {
  const sim = await makeSim();
  try {
    const t2 = 2;
    const body = sim.robots[t2].imuBodyId;
    sim.setPerturb(t2, body, [0, 0, 0], [5, 1.5, 1]);
    sim.step();
    assert.ok(xfrcMax(sim, t2) > 0, 'pushing');
    assert.equal(xfrcMax(sim, 0), 0);
    assert.equal(xfrcMax(sim, 1), 0);
    sim.resetAll(); // restart and re-home
    assert.ok(allZero(sim), 'restart clears it');
    assert.equal(sim.perturbing, false);
    sim.setPerturb(t2, body, [0, 0, 0], [5, 1.5, 1]); // the same drag goes on: ignored
    sim.step();
    assert.ok(allZero(sim), 'a drag that a reset cut stays off');
    sim.endPerturb();
    sim.setPerturb(t2, body, [0, 0, 0], [5, 1.5, 1]); // a new drag
    sim.step();
    assert.ok(xfrcMax(sim, t2) > 0, 'a new drag pushes again');

    sim.resetRobot(0); // another robot falls: the drag on the T2 goes on
    sim.step();
    assert.ok(xfrcMax(sim, t2) > 0, 'a fall of another robot does not cut it');
    sim.resetRobot(t2); // the dragged robot falls
    assert.ok(allZero(sim), 'its fall reset clears it');
    sim.setPerturb(t2, body, [0, 0, 0], [5, 1.5, 1]);
    sim.step();
    assert.ok(allZero(sim), 'until the drag ends');
    sim.endPerturb();
  } finally {
    sim.dispose();
  }
});

test('setPerturb checks its arguments', { skip: !HAVE_SCENE && 'no public/live' }, async () => {
  const sim = await makeSim();
  try {
    const x2Body = sim.robots[1].imuBodyId;
    assert.throws(() => sim.setPerturb(0, x2Body, [0, 0, 0], [0, 0, 1]), /not a body of/);
    assert.throws(() => sim.setPerturb(0, 0, [0, 0, 0], [0, 0, 1]), /not a body of/); // the world
    assert.throws(() => sim.setPerturb(3, 1, [0, 0, 0], [0, 0, 1]), /no robot/);
    assert.throws(() => sim.setPerturb(0, 1, [0, NaN, 0], [0, 0, 1]), /local/);
    assert.throws(() => sim.setPerturb(0, 1, [0, 0, 0], [0, 0]), /target/);
    assert.throws(() => sim.setPerturb(0, 1, [0, 0, 0], [0, Infinity, 1]), /target/);
    sim.step();
    assert.ok(allZero(sim));
  } finally {
    sim.dispose();
  }
});

test('headless: the G1 pushed sideways at the cap for 0.3 s while it stands', { skip: !HAVE_POLICY && 'no public/live policy', timeout: 300000 }, async () => {
  const [{ loadPolicy }, { LiveLoop }, { statesFromClip }] = await Promise.all([
    import('../src/live/policy.js'), import('../src/live/loop.js'), import('../src/live/reference.js'),
  ]);
  const robots = JSON.parse(readFileSync(join(LIVE, 'robots.json'), 'utf8'));
  const clipIndex = JSON.parse(readFileSync(join(LIVE, 'clips.json'), 'utf8'));
  const sim = await makeSim();
  try {
    const bytes = readFileSync(join(LIVE, 'policy.onnx'));
    const ort = await import('onnxruntime-web/wasm');
    const policy = await loadPolicy(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength), robots.robots, { ort });
    const c = clipIndex.clips[0];
    const buf = readFileSync(join(LIVE, 'clips', `${c.id}.bin`));
    const states = robots.robots.map((r) => statesFromClip(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), c.robots[r.name], clipIndex.columns));
    const loop = new LiveLoop(sim, policy, robots.robots, states, {
      iface: 'motion', clipId: 0, loop: robots.loop, fps: robots.fps, historyLen: robots.historyLen,
    });
    const g1 = 0;
    const body = sim.robots[g1].imuBodyId;
    const y0 = sim.slot0(g1).pos[1];
    const others0 = [1, 2].map((r) => sim.slot0(r).pos);
    for (let i = 0; i < 10; i++) await loop.controlStep(); // 0.2 s of standing
    const pushSteps = Math.round(0.3 * robots.fps);
    let fell = false;
    let peak = 0;
    for (let i = 0; i < pushSteps; i++) {
      const p = sim.slot0(g1).pos;
      sim.setPerturb(g1, body, [0, 0, 0], [p[0], p[1] + 100, p[2]]); // the cursor far to the side
      const out = await loop.controlStep();
      if (out.includes(g1)) fell = true;
      peak = Math.max(peak, norm(sim.data.xfrc_applied, 6 * body));
      assert.ok(peak <= CAP + 1e-9, `|F| = ${peak}`);
    }
    const moved = sim.slot0(g1).pos[1] - y0;
    sim.endPerturb();
    assert.ok(allZero(sim), 'cleared');
    for (let i = 0; i < 100; i++) {
      const out = await loop.controlStep();
      if (out.includes(g1)) fell = true;
    }
    assert.ok(peak > CAP - 1e-6, `the push reached the cap (${peak})`);
    const st = loop.status();
    for (const s of st) assert.ok(Number.isFinite(s.jointErr), `${s.name} runs on`);
    for (const r of [1, 2]) {
      assert.equal(st[r].falls, 0, `${st[r].name} did not fall`);
      const p = sim.slot0(r).pos;
      assert.ok(Math.hypot(p[0] - others0[r - 1][0], p[1] - others0[r - 1][1]) < 0.1, `${st[r].name} was not pushed`);
    }
    assert.equal(st[g1].falls > 0, fell, 'a fall is counted and reset by the loop');
    console.log(`G1 pushed for ${pushSteps} steps: moved ${moved.toFixed(3)} m sideways during the push, `
      + `${st[g1].falls} falls, joint error after ${st[g1].jointErr.toFixed(3)} rad`);
    if (!fell) assert.ok(moved > 0.002, `the push moved the G1 toward +y (${moved})`);
  } finally {
    sim.dispose();
  }
});
