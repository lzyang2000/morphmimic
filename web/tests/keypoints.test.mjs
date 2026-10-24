import test from 'node:test';
import assert from 'node:assert/strict';
import {
  KP_COLORS, KP_POINTS, KP_TRAIL, KpTrail, TRAIL_ALPHA, anchorTargets, trailAlpha,
} from '../src/keypoints.js';

test('five colours (sim_node order: pelvis, ankles, wrists), 50-frame trail', () => {
  assert.equal(KP_POINTS, 5);
  assert.equal(KP_TRAIL, 50);
  assert.deepEqual(KP_COLORS[0], [1.0, 0.85, 0.1]);
  assert.deepEqual(KP_COLORS[4], [0.2, 0.4, 1.0]);
});

test('anchorTargets puts the frame-0 pelvis at the robot pelvis xy and keeps heights and shape', () => {
  // 2 frames x 5 points: point p of frame k at (10 k + p, 100 + p, 0.1 p)
  const t = new Float64Array(2 * 5 * 3);
  for (let k = 0; k < 2; k++) for (let p = 0; p < 5; p++) t.set([10 * k + p, 100 + p, 0.1 * p], (k * 5 + p) * 3);
  const out = new Float64Array(t.length);
  assert.equal(anchorTargets(t, [1, 2], out), out);
  assert.deepEqual([...out.subarray(0, 3)], [1, 2, 0]);
  const o = (1 * 5 + 3) * 3; // frame 1, left wrist
  assert.deepEqual([...out.subarray(o, o + 3)].map((x) => +x.toFixed(9)), [1 + 13, 2 + 3, 0.3]);
  const nan = new Float64Array(30).fill(NaN);
  assert.equal(anchorTargets(nan, [0, 0], out), null, 'no targets before the first step');
});

test('KpTrail keeps the last len entries, oldest first, and clears', () => {
  const tr = new KpTrail(3);
  const entry = (v) => Float64Array.from({ length: 15 }, (_, i) => v + i / 100);
  for (const v of [1, 2]) tr.push(entry(v));
  assert.equal(tr.length, 2);
  assert.equal(tr.buf[tr.at(0, 0)], 1);
  assert.equal(tr.buf[tr.at(1, 4) + 2], 2 + 14 / 100);
  for (const v of [3, 4, 5]) tr.push(entry(v));
  assert.equal(tr.length, 3);
  assert.deepEqual([0, 1, 2].map((i) => tr.buf[tr.at(i, 0)]), [3, 4, 5]);
  tr.clear();
  assert.equal(tr.length, 0);
  tr.push(entry(9));
  assert.equal(tr.buf[tr.at(0, 0)], 9);
});

test('trailAlpha runs from 0.3 at the oldest segment to 0.9 at the newest', () => {
  assert.equal(trailAlpha(0, 49), TRAIL_ALPHA[0]);
  assert.ok(Math.abs(trailAlpha(48, 49) - TRAIL_ALPHA[1]) < 1e-12);
  assert.ok(trailAlpha(10, 49) < trailAlpha(40, 49));
  assert.equal(trailAlpha(0, 1), TRAIL_ALPHA[0]);
});
