// The shared policy in the browser: one onnxruntime-web session (WASM, single thread, SIMD)
// serving every robot. Each robot has its own set of input tensors, built once (normalisers
// and morph from robots.json) and refilled in place every control step, so a step allocates
// nothing but the 29-float result.
//
// The model is the released policy cut down to its motion output (tools/gen_live_assets.py), so
// `act` runs the motion interface only and returns a fresh Float32Array(29): the canonical
// action, which PolicyInputs.setAction takes back unchanged.
//
// Single thread means the page needs no cross-origin isolation (SharedArrayBuffer) headers.
// The ORT module and the WASM location are injectable: Node loads them from node_modules, the
// page passes the URL Vite emits for the .wasm (src/live/ort_browser.js).

export const ACTION_DIM = 29;
const OUTPUTS = { motion: 'action_motion' };
// Per-step inputs, in the order PolicyInputs.step returns them, with their element counts and
// ONNX types.
const STEP_INPUTS = [
  ['obs_history', 930, 'float32'],
  ['goal_motion', 670, 'float32'],
];
// robots.json inputs key -> ONNX input name.
const STATIC_INPUTS = {
  obsMean: 'obs_mean', obsVar: 'obs_var', goalMean: 'goal_mean',
  goalVar: 'goal_var', morph: 'morph',
};

/** Reads `url` (or a Response) into an ArrayBuffer, calling onProgress(loaded, total) as it comes. */
export async function download(url, onProgress, fetchFn = fetch) {
  const res = await fetchFn(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body) {
    const buf = await res.arrayBuffer();
    onProgress?.(buf.byteLength, buf.byteLength);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let at = 0;
  for (const c of chunks) { out.set(c, at); at += c.length; }
  return out.buffer;
}

export class Policy {
  constructor(ort, session, robots) {
    this.ort = ort;
    this.session = session;
    this.slots = robots.map((spec, i) => this._slot(spec, i));
  }

  _slot(spec, i) {
    const { Tensor } = this.ort;
    const feeds = {};
    const step = {};
    for (const [name, n, type] of STEP_INPUTS) {
      const data = type === 'bool' ? new Uint8Array(n) : new Float32Array(n);
      feeds[name] = new Tensor(type, data, [1, n]);
      step[name] = data;
    }
    for (const [key, name] of Object.entries(STATIC_INPUTS)) {
      const v = spec.inputs?.[key];
      if (!v) throw new Error(`robot ${i} (${spec.name}): inputs.${key} missing`);
      feeds[name] = new Tensor('float32', Float32Array.from(v), [v.length]);
    }
    return { feeds, step };
  }

  /**
   * One policy step for robot `robotIdx` (index into the robots passed to loadPolicy).
   * `inputs` is PolicyInputs.step's result; `iface` is 'motion'.
   * Resolves to the canonical action, Float32Array[29].
   */
  async act(robotIdx, inputs, iface) {
    const name = OUTPUTS[iface];
    if (!name) throw new Error(`interface must be motion, got ${iface}`);
    const slot = this.slots[robotIdx];
    if (!slot) throw new Error(`no robot ${robotIdx}`);
    for (const [key, n] of STEP_INPUTS) {
      const src = inputs[key];
      if (!src || src.length !== n) throw new Error(`${key}: expected ${n} values, got ${src?.length}`);
      slot.step[key].set(src);
    }
    const out = await this.session.run(slot.feeds, [name]);
    const t = out[name];
    const action = Float32Array.from(t.data);
    t.dispose?.();
    return action;
  }

  async release() { await this.session.release(); }
}

// A 71-byte ONNX model (Identity, float[1] -> float[1], opset 13).
export const PROBE_MODEL = Uint8Array.of(
  8, 7, 18, 0, 58, 59, 10, 16, 10, 1, 120, 18, 1, 121, 34, 8, 73, 100, 101, 110, 116, 105, 116, 121,
  18, 5, 112, 114, 111, 98, 101, 90, 15, 10, 1, 120, 18, 10, 10, 8, 8, 1, 18, 4, 10, 2, 8, 1, 98, 15,
  10, 1, 121, 18, 10, 10, 8, 8, 1, 18, 4, 10, 2, 8, 1, 66, 4, 10, 0, 16, 13,
);

/** Starts onnxruntime-web (fetches and compiles its WebAssembly, sets up its environment) on
 * PROBE_MODEL, so a browser that cannot run it fails here, before the policy download. */
export async function startOrt(ort) {
  const session = await ort.InferenceSession.create(PROBE_MODEL, { executionProviders: ['wasm'] });
  await session.release();
}

/**
 * Loads the policy. `source` is a URL, an ArrayBuffer or a Uint8Array; `robots` is robots.json's
 * `robots` array (its order is the robotIdx of `act`). Options: onProgress(loaded, total) for the
 * download, `ort` (the onnxruntime-web module, default: import 'onnxruntime-web/wasm'),
 * `wasmPaths` (ORT's env.wasm.wasmPaths: a prefix or {mjs, wasm} URLs). ORT is started
 * (startOrt) before the download.
 */
export async function loadPolicy(source, robots, { onProgress, ort, wasmPaths, fetchFn } = {}) {
  ort ??= await import('onnxruntime-web/wasm');
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  ort.env.wasm.proxy = false;
  if (wasmPaths) ort.env.wasm.wasmPaths = wasmPaths;
  await startOrt(ort);
  const bytes = typeof source === 'string' ? await download(source, onProgress, fetchFn) : source;
  const session = await ort.InferenceSession.create(
    bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes),
    { executionProviders: ['wasm'], graphOptimizationLevel: 'all' },
  );
  const policy = new Policy(ort, session, robots);
  // First run pays for lazy initialisation; do it now, on zeros (both branches).
  for (let r = 0; r < robots.length; r++) {
    for (const iface of Object.keys(OUTPUTS)) {
      const zeros = Object.fromEntries(STEP_INPUTS.map(([k, n, t]) => [k, t === 'bool' ? new Uint8Array(n).fill(1) : new Float32Array(n)]));
      await policy.act(r, zeros, iface);
    }
  }
  return policy;
}
