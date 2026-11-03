// The packed reference clips: clips/<id>.bin holds only the state columns the goals read (the
// first 156 of 240); statesFromClip (src/live/reference.js) fills the body velocities with zeros.
import test from 'node:test';
import assert from 'node:assert/strict';
import { GOAL_K, LIN, STATE_DIM, goalMotion } from '../src/live/obs.js';
import { statesFromClip } from '../src/live/reference.js';

// Deterministic pseudo-random numbers.
function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (1664525 * s + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;
}
const unitQuat = (r) => {
  const q = [r(), r(), r(), r()];
  const n = Math.hypot(...q);
  return q.map((x) => x / n);
};
function states(frames, seed) {
  const r = rng(seed);
  const out = new Float64Array(frames * STATE_DIM);
  for (let f = 0; f < frames; f++) {
    for (let i = 0; i < STATE_DIM; i++) out[f * STATE_DIM + i] = r();
    for (let b = 0; b < 14; b++) out.set(unitQuat(r), f * STATE_DIM + 100 + 4 * b);
  }
  return out;
}

// Packs states as gen_live_assets.pack_clips does: float32, the first `columns` of each state.
function pack(blocks, columns) {
  const total = blocks.reduce((n, s) => n + (s.length / STATE_DIM) * columns, 0);
  const buf = new Float32Array(total);
  const entries = [];
  let at = 0;
  for (const s of blocks) {
    const frames = s.length / STATE_DIM;
    entries.push({ offset: at * 4, frames });
    for (let f = 0; f < frames; f++) {
      for (let i = 0; i < columns; i++) buf[at++] = s[f * STATE_DIM + i];
    }
  }
  return { buffer: buf.buffer, entries };
}

test('statesFromClip reads [frames, 156] and zero-fills the velocities back to 240 columns', () => {
  const a = states(7, 1);
  const b = states(5, 2);
  const { buffer, entries } = pack([a, b], LIN);
  assert.equal(buffer.byteLength, 4 * 12 * 156);
  for (const [s, e] of [[a, entries[0]], [b, entries[1]]]) {
    const got = statesFromClip(buffer, e, LIN);
    assert.equal(got.length, e.frames * STATE_DIM);
    for (let f = 0; f < e.frames; f++) {
      for (let i = 0; i < STATE_DIM; i++) {
        const want = i < LIN ? Math.fround(s[f * STATE_DIM + i]) : 0;
        assert.equal(got[f * STATE_DIM + i], want, `frame ${f} column ${i}`);
      }
    }
  }
  // A full 240-column file still reads as it is.
  const full = pack([a], STATE_DIM);
  assert.deepEqual(statesFromClip(full.buffer, full.entries[0], STATE_DIM), Float64Array.from(a, Math.fround));
  for (const bad of [undefined, 100, 155, 241, 156.5]) {
    assert.throws(() => statesFromClip(buffer, entries[0], bad), /columns/, String(bad));
  }
});

test('the goals do not read the dropped columns: same goals with and without the velocities', () => {
  const s = states(GOAL_K, 3);
  const { buffer, entries } = pack([s], LIN);
  const zeroed = statesFromClip(buffer, entries[0], LIN);
  const kept = Float64Array.from(s, Math.fround);
  const anchor = unitQuat(rng(4));
  assert.deepEqual(goalMotion(zeroed, anchor), goalMotion(kept, anchor));
});
