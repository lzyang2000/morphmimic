// policy.js's start-up order: onnxruntime-web is started (its WebAssembly compiled) on a tiny
// model before the policy download, so a browser that cannot run it fails before the download.
import test from 'node:test';
import assert from 'node:assert/strict';
import { PROBE_MODEL, loadPolicy, startOrt } from '../src/live/policy.js';

// An ORT stand-in that logs its calls; `fail` makes session creation reject.
function fakeOrt(log, { fail = false } = {}) {
  return {
    env: { wasm: {} },
    Tensor: class { constructor(type, data, dims) { Object.assign(this, { type, data, dims }); } },
    InferenceSession: {
      async create(bytes) {
        log.push(bytes === PROBE_MODEL ? 'create probe' : `create ${bytes.length} bytes`);
        if (fail) throw new Error('no WebAssembly backend');
        return {
          release: async () => log.push('release'),
          run: async (feeds, [name]) => ({ [name]: { data: new Float32Array(29) } }),
        };
      },
    },
  };
}

const fetchFn = (log) => async (url) => {
  log.push(`fetch ${url}`);
  return new Response(new Uint8Array(1000));
};

test('ORT starts on the probe model before the policy is downloaded', async () => {
  const log = [];
  const robots = [{ name: 'toy', inputs: Object.fromEntries(['obsMean', 'obsVar', 'goalMean', 'goalVar', 'humanMean', 'humanVar', 'headingMean', 'headingVar', 'morph'].map((k) => [k, [0]])) }];
  await loadPolicy('policy.onnx', robots, { ort: fakeOrt(log), fetchFn: fetchFn(log) });
  assert.deepEqual(log.slice(0, 4), ['create probe', 'release', 'fetch policy.onnx', 'create 1000 bytes']);
});

test('a backend that cannot start fails before the download', async () => {
  const log = [];
  await assert.rejects(loadPolicy('policy.onnx', [], { ort: fakeOrt(log, { fail: true }), fetchFn: fetchFn(log) }), /no WebAssembly backend/);
  assert.deepEqual(log, ['create probe'], 'nothing fetched');
});

test('the probe model runs on the real onnxruntime-web', async () => {
  const ort = await import('onnxruntime-web/wasm');
  ort.env.wasm.numThreads = 1;
  await startOrt(ort);
  const s = await ort.InferenceSession.create(PROBE_MODEL);
  const out = await s.run({ x: new ort.Tensor('float32', Float32Array.of(2.5), [1]) });
  assert.deepEqual(Array.from(out.y.data), [2.5]);
  await s.release();
});
