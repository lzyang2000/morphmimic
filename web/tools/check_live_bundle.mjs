// Headless browser check of the live worker in a real build: bundles a minimal page that starts
// src/live/worker.js (into .cache/livecheck/dist, base /morphmimic/demo/ unless DEMO_BASE), serves
// it with `vite preview`, and drives it in headless Chrome (DevTools protocol, no extra packages)
// with every host but 127.0.0.1 unresolvable, so nothing can come from a CDN. It checks:
//   - the worker boots (MuJoCo WASM, policy, clips) and posts frames;
//   - both .wasm files (MuJoCo, onnxruntime-web) are fetched from the build's assets/ under the
//     base, and every request (page and worker) stays on the preview server;
//   - setClip, setInterface, pause/resume and restart are answered;
// and prints the worker's real-time factor. Exits 1 on any failure.
//
//   node tools/check_live_bundle.mjs [--seconds 20] [--port 4318]
//
// Needs web/public/live (gen_live_assets.py). CHROME overrides the browser (default
// /usr/bin/google-chrome).

import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const opt = (name, def) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : def;
};
const SECONDS = Number(opt('seconds', 20));
const PORT = Number(opt('port', 4318));
const CHROME = process.env.CHROME || '/usr/bin/google-chrome';
const BASE = (process.env.DEMO_BASE || '/morphmimic/demo/').replace(/\/?$/, '/');
const DIR = join(WEB, '.cache/livecheck');

// ------------------------------------------------------------------ build ----
mkdirSync(DIR, { recursive: true });
writeFileSync(join(DIR, 'index.html'), `<!doctype html>
<html><head><meta charset="utf-8"><title>live worker check</title></head>
<body><script type="module" src="./main.js"></script></body></html>
`);
// The page: starts the worker, records what it posts, and exercises every command.
writeFileSync(join(DIR, 'main.js'), `
const out = { progress: {}, frames: 0, errors: [], ready: null, last: null, log: [] };
window.__live = out;
const w = new Worker(new URL('../../src/live/worker.js', import.meta.url), { type: 'module' });
w.onerror = (e) => out.errors.push('worker: ' + (e.message || e.type));
let firstFrame = null;
w.onmessage = ({ data: m }) => {
  if (m.type === 'progress') out.progress[m.stage] = [m.loaded, m.total];
  else if (m.type === 'ready') { out.ready = { robots: m.robots, bodies: m.bodies.length, clips: m.clips.length, goalK: m.goalK }; out.tReady = performance.now(); }
  else if (m.type === 'error') out.errors.push(m.stage + ': ' + m.message);
  else if (m.type === 'frame') {
    out.frames++;
    firstFrame ??= m;
    out.last = { time: m.time, steps: m.steps, rtf: m.rtf, poses: m.poses.length, finitePoses: m.poses.filter(Number.isFinite).length, status: m.status, pelvis: Array.from(m.poses.subarray(7, 10)) };
  }
};
w.postMessage({ type: 'init', base: new URL('live/', document.baseURI).href, clip: 0, iface: 'motion' });
out.send = (msg) => { out.log.push([performance.now(), msg.type, out.last?.time ?? null]); w.postMessage(msg); };
`);
writeFileSync(join(DIR, 'vite.config.mjs'), `
import { mergeConfig } from 'vite';
import base from '../../vite.config.js';
export default mergeConfig(base, {
  root: ${JSON.stringify(DIR)},
  publicDir: ${JSON.stringify(join(WEB, 'public'))},
  build: { outDir: ${JSON.stringify(join(DIR, 'dist'))}, emptyOutDir: true },
  preview: { port: ${PORT}, strictPort: true, host: '127.0.0.1' },
});
`);
const vitebin = join(WEB, 'node_modules/.bin/vite');
const cfg = ['--config', join(DIR, 'vite.config.mjs')];
const b = spawnSync(vitebin, ['build', ...cfg, '--logLevel', 'warn'], { cwd: WEB, encoding: 'utf8' });
if (b.status !== 0) { console.error(b.stdout, b.stderr); process.exit(1); }
const assets = spawnSync('ls', ['-la', join(DIR, 'dist/assets')], { encoding: 'utf8' }).stdout;
console.log('built:\n' + assets.split('\n').filter((l) => /\.(wasm|js)$/.test(l)).map((l) => '  ' + l.split(/\s+/).slice(4).join(' ')).join('\n'));

// ------------------------------------------------------------------ serve + chrome ----
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const procs = [];
let profile = null;
function cleanup() {
  for (const p of procs) if (p.exitCode === null) p.kill('SIGKILL');
  if (profile) rmSync(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
  profile = null;
}
process.on('exit', cleanup);
process.on('SIGINT', () => process.exit(130));
async function waitFor(fn, ms, what) {
  const end = Date.now() + ms;
  for (;;) {
    try { const v = await fn(); if (v) return v; } catch { /* not yet */ }
    if (Date.now() > end) throw new Error(`timed out waiting for ${what}`);
    await sleep(100);
  }
}
const vite = spawn(vitebin, ['preview', ...cfg], { cwd: WEB, stdio: ['ignore', 'pipe', 'pipe'] });
procs.push(vite);
let viteLog = '';
vite.stdout.on('data', (d) => { viteLog += d; });
vite.stderr.on('data', (d) => { viteLog += d; });
const origin = `http://127.0.0.1:${PORT}`;
const pageUrl = `${origin}${BASE}`;
await waitFor(async () => (await fetch(pageUrl)).ok, 20000, `vite preview at ${pageUrl}\n${viteLog}`);

profile = mkdtempSync(join(WEB, '.cache/chrome-'));
const chrome = spawn(CHROME, [
  '--headless=new', '--no-first-run', '--no-default-browser-check', '--disable-extensions',
  '--mute-audio', `--user-data-dir=${profile}`, '--remote-debugging-port=0',
  // Any host but the preview server fails to resolve: a CDN fetch would fail the boot.
  '--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1', 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });
procs.push(chrome);
chrome.stderr.on('data', () => {});
const [devPort, devPath] = await waitFor(() => {
  const lines = readFileSync(join(profile, 'DevToolsActivePort'), 'utf8').trim().split('\n');
  return lines.length >= 2 ? lines : null;
}, 20000, 'Chrome DevTools port');
const ws = new WebSocket(`ws://127.0.0.1:${devPort}${devPath}`);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
let nextId = 1;
const pending = new Map();
const listeners = [];
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) {
    const { res, rej } = pending.get(msg.id);
    pending.delete(msg.id);
    if (msg.error) rej(new Error(`${msg.error.message} ${msg.error.data ?? ''}`));
    else res(msg.result);
  } else if (msg.method) for (const l of listeners) l(msg);
};
const send = (method, params = {}, sessionId) => new Promise((res, rej) => {
  const id = nextId++;
  pending.set(id, { res, rej });
  ws.send(JSON.stringify({ id, method, params, sessionId }));
});

const requests = []; // [session kind, url, mime]
const failed = [];
const errors = [];
const kinds = new Map();
listeners.push((m) => {
  const kind = kinds.get(m.sessionId);
  if (!kind) return;
  if (m.method === 'Network.responseReceived') requests.push([kind, m.params.response.url, m.params.response.mimeType, m.params.response.status]);
  else if (m.method === 'Network.loadingFailed') failed.push([kind, m.params.errorText, m.params.blockedReason ?? '']);
  else if (m.method === 'Runtime.exceptionThrown') errors.push(`${kind} exception: ${m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text}`);
  else if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') errors.push(`${kind} console.error: ${m.params.args.map((a) => a.value ?? a.description).join(' ')}`);
  else if (m.method === 'Target.attachedToTarget') {
    const s = m.params.sessionId;
    kinds.set(s, m.params.targetInfo.type);
    send('Network.enable', {}, s).catch(() => {});
    send('Runtime.enable', {}, s).catch(() => {});
    send('Runtime.runIfWaitingForDebugger', {}, s).catch(() => {});
  }
});
const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
kinds.set(sessionId, 'page');
const S = (method, params) => send(method, params, sessionId);
await S('Runtime.enable');
await S('Network.enable');
await S('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: true, flatten: true });
async function evaluate(expression) {
  const r = await S('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
  return r.result.value;
}
const state = () => evaluate('JSON.parse(JSON.stringify({...window.__live, send: undefined}))');

let ok = true;
const fail = (why) => { ok = false; console.log(`FAIL ${why}`); };
try {
  const t0 = Date.now();
  await S('Page.navigate', { url: pageUrl });
  await waitFor(async () => {
    const s = await state();
    if (s.errors.length) throw Object.assign(new Error(s.errors.join('; ')), { fatal: true });
    return s.ready && s.frames > 0;
  }, 180000, 'the worker to be ready').catch(async (e) => { console.log(JSON.stringify(await state().catch(() => ({})))); throw e; });
  const boot = await state();
  console.log(`ready in ${((Date.now() - t0) / 1000).toFixed(1)} s: ${JSON.stringify(boot.ready)}`);
  console.log(`progress: ${JSON.stringify(boot.progress)}`);
  // Run, then exercise the commands.
  const run = async (s) => { await sleep(s * 1000); return state(); };
  const a = await run(SECONDS * 0.4);
  await evaluate("window.__live.send({ type: 'setClip', clip: 9 })");
  const b2 = await run(SECONDS * 0.3);
  await evaluate("window.__live.send({ type: 'pause' })");
  await sleep(300);
  const p0 = await state();
  const p1 = await run(1.5);
  await evaluate("window.__live.send({ type: 'resume' })");
  const r1 = await run(SECONDS * 0.2);
  await evaluate("window.__live.send({ type: 'restart' })");
  const z = await run(1);
  const fmt = (s) => `t=${s.last.time.toFixed(2)} s steps=${s.last.steps} rtf=${s.last.rtf.toFixed(2)} frames=${s.frames} `
    + s.last.status.map((r) => `${r.name}:${r.phase}/clip${r.clip}->${r.nextClip} falls ${r.falls}`).join(' ');
  for (const [name, s] of [['after run', a], ['after setClip', b2], ['paused', p1], ['resumed', r1], ['restarted', z]]) console.log(`${name}: ${fmt(s)}`);
  if (a.last.time < SECONDS * 0.4 * 0.3) fail(`sim ran only ${a.last.time} s in ${SECONDS * 0.4} s`);
  if (a.last.poses !== boot.ready.bodies * 7) fail('pose frame size');
  if (a.last.finitePoses !== a.last.poses) fail('poses not all finite after the first step');
  if (!b2.last.status.every((r) => r.nextClip === 9)) fail('setClip not applied');
  if (p1.last.steps !== p0.last.steps) fail(`paused but stepping (${p0.last.steps} -> ${p1.last.steps})`);
  if (!(r1.last.steps > p1.last.steps)) fail('resume did not resume');
  const resumedRate = (r1.last.time - p1.last.time) / (SECONDS * 0.2);
  if (resumedRate > 1.05) fail(`resume replayed paused time (${resumedRate.toFixed(2)}x real time)`);
  if (!(z.last.time < r1.last.time && z.last.status.every((r) => r.motions === 0))) fail('restart did not reset the loops');
  const final = await state();
  if (final.errors.length) fail(`worker errors: ${final.errors.join('; ')}`);
  console.log(`sim time per wall second over the run: ${a.last.rtf.toFixed(2)} (first window), ${r1.last.rtf.toFixed(2)} (after resume)`);
} catch (e) {
  fail(e.message);
}

// Requests: everything from the preview server, both .wasm from the base's assets/.
const off = requests.filter(([, u]) => !u.startsWith(origin) && !u.startsWith('data:') && !u.startsWith('blob:'));
const wasm = requests.filter(([, u]) => u.endsWith('.wasm'));
console.log(`requests: ${requests.length} (${[...new Set(requests.map(([k]) => k))].join(', ')})`);
for (const [k, u, mime, st] of wasm) console.log(`  wasm (${k}): ${u.replace(origin, '')} ${mime} ${st}`);
if (off.length) fail(`requests off the server: ${off.map(([, u]) => u).join(', ')}`);
if (failed.length) fail(`failed loads: ${JSON.stringify(failed)}`);
for (const name of ['mujoco', 'ort-wasm-simd-threaded']) {
  const hit = wasm.find(([, u]) => u.replace(origin, '').startsWith(`${BASE}assets/${name}`));
  if (!hit) fail(`${name}*.wasm not loaded from ${BASE}assets/`);
  else if (hit[0] !== 'worker') fail(`${name}.wasm loaded by the ${hit[0]}, not the worker`);
}
if (errors.length) fail(`console: ${errors.join('; ')}`);
console.log(ok ? 'OK: live worker bundle' : 'FAILED');
cleanup();
process.exit(ok ? 0 : 1);
