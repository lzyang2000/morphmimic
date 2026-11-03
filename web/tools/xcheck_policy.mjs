// Cross-check of src/live/policy.js (onnxruntime-web, WASM, one thread, SIMD) on the page's
// pruned public/live/policy.onnx against Python onnxruntime on the full released export.
// tools/xcheck_policy.py writes 64 random input sets per robot with Python's outputs; this runs
// the same inputs through Policy.act (the motion interface, the page's only one) and exits 1 if any action differs by more than 1e-4 (or is not a Float32Array
// of 29). Then it times `act` and says whether 3 robots at 50 Hz fit the 20 ms control step.
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { loadPolicy, ACTION_DIM } from '../src/live/policy.js';

const TOL = 1e-4;
const BUDGET_MS = 20;
const web = new URL('..', import.meta.url);
const json = (p) => JSON.parse(readFileSync(new URL(p, web), 'utf8'));

const live = json('public/live/robots.json');
const ref = json('.cache/xcheck_policy/reference.json');
const bytes = readFileSync(new URL('public/live/policy.onnx', web));
const ort = await import('onnxruntime-web/wasm');
const policy = await loadPolicy(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength), live.robots, { ort });

const asInputs = (r, i) => ({
  obs_history: Float32Array.from(r.obs_history[i]),
  goal_motion: Float32Array.from(r.goal_motion[i]),
});

let failed = false;
console.log(`onnxruntime-web ${ort.env.versions?.web ?? ''}, threads ${ort.env.wasm.numThreads}, simd ${ort.env.wasm.simd}`);
for (let r = 0; r < ref.robots.length; r++) {
  const want = ref.robots[r];
  if (want.name !== live.robots[r].name) throw new Error(`robot order: ${want.name} vs ${live.robots[r].name}`);
  const diff = { motion: 0 };
  for (let i = 0; i < ref.cases; i++) {
    const inputs = asInputs(want, i);
    for (const iface of ['motion']) {
      const got = await policy.act(r, inputs, iface);
      if (!(got instanceof Float32Array) || got.length !== ACTION_DIM) { console.log(`FAIL ${want.name} ${iface}: result is not Float32Array(29)`); failed = true; continue; }
      const w = want[`action_${iface}`][i];
      for (let j = 0; j < ACTION_DIM; j++) diff[iface] = Math.max(diff[iface], Math.abs(got[j] - w[j]));
    }
  }
  const ok = diff.motion <= TOL;
  failed ||= !ok;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${want.name}: max |diff| motion ${diff.motion.toExponential(2)} (${ref.cases} cases, tol ${TOL})`);
}

// Benchmark: act over cycling robots and cases, one interface at a time, then a 3-robot step.
const stats = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return { mean: xs.reduce((a, b) => a + b, 0) / xs.length, p95: s[Math.floor(0.95 * (s.length - 1))], max: s[s.length - 1] };
};
const N = 300;
const fmt = (s) => `mean ${s.mean.toFixed(2)} ms, p95 ${s.p95.toFixed(2)} ms, max ${s.max.toFixed(2)} ms`;
for (const iface of ['motion']) {
  const t = [];
  for (let n = 0; n < N; n++) {
    const r = n % ref.robots.length;
    const inputs = asInputs(ref.robots[r], n % ref.cases);
    const t0 = performance.now();
    await policy.act(r, inputs, iface);
    t.push(performance.now() - t0);
  }
  console.log(`act (${iface}): ${fmt(stats(t))}`);
}
const step = [];
const all = ref.robots.map((w) => asInputs(w, 0));
for (let n = 0; n < 150; n++) {
  const t0 = performance.now();
  for (let r = 0; r < all.length; r++) await policy.act(r, all[r], 'motion');
  step.push(performance.now() - t0);
}
const s3 = stats(step);
console.log(`3 robots per control step: ${fmt(s3)} (budget ${BUDGET_MS} ms for inference, physics and drawing not included)`);
console.log(`3 x 50 Hz ${s3.p95 < BUDGET_MS / 2 ? 'fits comfortably' : s3.p95 < BUDGET_MS ? 'fits, but with little room left for physics' : 'DOES NOT FIT'}`);
process.exit(failed ? 1 : 0);
