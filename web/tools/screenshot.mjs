// Headless check of the built live demo: serves dist/ with `vite preview`, loads the page in
// headless Chrome (Chrome DevTools Protocol, no extra packages), lets the simulation run until
// --at seconds of sim time, saves a PNG and prints the robots' status and console errors.
// Exits 1 if the page logged an error, showed an error, or did not boot.
//
//   node tools/screenshot.mjs out.png [--width 1280] [--height 720] [--port 4317] [--dev]
//       --dev serves the sources with the Vite dev server instead of dist/ with vite preview
//   node tools/screenshot.mjs out.png --view N[,IFACE] [--at 6]
//       clip N (0-based sidebar index) on IFACE (motion or keypoint, default motion), opened
//       with ?clip=N&iface=IFACE, captured once the sim has run --at seconds (default 6: the
//       first cue is at 1 s and the lead-in takes 2 s, so 3 s into the clip)
//   node tools/screenshot.mjs out.png --view N --query rehome=0.5 --rehomes 1
//       --query adds to the page's query string; --rehomes K waits (after --at) until the robots
//       have been re-homed K times, then captures 0.6 s of sim time later, after the fade. With a
//       debug build (VITE_DEMO_DEBUG=1 npm run build), ?rehome=<m> lowers the re-home threshold.
//
// Run `npm run build` first. CHROME overrides the browser binary (default /usr/bin/google-chrome).

import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const opt = (name, def) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : def;
};
const out = argv[0] && !argv[0].startsWith('--') ? resolve(argv[0]) : null;
if (!out) {
  console.error('usage: node tools/screenshot.mjs out.png [--width W] [--height H] [--view N[,IFACE]] [--at S] [--query Q] [--rehomes K] [--dev]');
  process.exit(2);
}
const WIDTH = Number(opt('width', 1280));
const HEIGHT = Number(opt('height', 720));
const AT = Number(opt('at', 6));
const CHROME = process.env.CHROME || '/usr/bin/google-chrome';
const PORT = Number(opt('port', 4317));
const DEV = argv.includes('--dev');
const VIEW = opt('view', null);
const QUERY = opt('query', '');
const REHOMES = Number(opt('rehomes', 0));
const BASE = (process.env.DEMO_BASE || '/morphmimic/demo/').replace(/\/?$/, '/');

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
    try {
      const v = await fn();
      if (v) return v;
    } catch (e) {
      if (e?.fatal) throw e; // not yet, unless it can never succeed
    }
    if (Date.now() > end) throw new Error(`timed out waiting for ${what}`);
    await sleep(100);
  }
}

// ------------------------------------------------------------------ server ----
if (!DEV && !existsSync(join(WEB, 'dist/index.html'))) throw new Error('dist/ is missing; run npm run build');
const vite = spawn(join(WEB, 'node_modules/.bin/vite'),
  [...(DEV ? [] : ['preview']), '--port', String(PORT), '--strictPort', '--host', '127.0.0.1'],
  { cwd: WEB, stdio: ['ignore', 'pipe', 'pipe'] });
procs.push(vite);
let viteLog = '';
vite.stdout.on('data', (d) => { viteLog += d; });
vite.stderr.on('data', (d) => { viteLog += d; });
const pageUrl = `http://127.0.0.1:${PORT}${BASE}`;
await waitFor(async () => (await fetch(pageUrl)).ok, 20000, `vite preview at ${pageUrl}\n${viteLog}`);

// ------------------------------------------------------------------ chrome ----
mkdirSync(join(WEB, '.cache'), { recursive: true });
profile = mkdtempSync(join(WEB, '.cache/chrome-'));
const chrome = spawn(CHROME, [
  '--headless=new', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  '--no-first-run', '--no-default-browser-check', '--disable-extensions', '--hide-scrollbars',
  '--mute-audio', `--user-data-dir=${profile}`, '--remote-debugging-port=0',
  `--window-size=${WIDTH},${HEIGHT}`, 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });
procs.push(chrome);
chrome.stderr.on('data', () => {});
const portFile = join(profile, 'DevToolsActivePort');
const [devPort, devPath] = await waitFor(() => {
  const lines = readFileSync(portFile, 'utf8').trim().split('\n');
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
  } else if (msg.method) {
    for (const l of listeners) l(msg);
  }
};
const send = (method, params = {}, sessionId) => new Promise((res, rej) => {
  const id = nextId++;
  pending.set(id, { res, rej });
  ws.send(JSON.stringify({ id, method, params, sessionId }));
});

const errors = [];
const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
const S = (method, params) => send(method, params, sessionId);
listeners.push((m) => {
  if (m.sessionId !== sessionId) return;
  if (m.method === 'Runtime.consoleAPICalled' && (m.params.type === 'error' || m.params.type === 'assert')) {
    errors.push(`console.${m.params.type}: ${m.params.args.map((a) => a.value ?? a.description ?? '').join(' ')}`);
  } else if (m.method === 'Runtime.exceptionThrown') {
    const d = m.params.exceptionDetails;
    errors.push(`exception: ${d.exception?.description ?? d.text}`);
  } else if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') {
    errors.push(`log: ${m.params.entry.text} ${m.params.entry.url ?? ''}`);
  } else if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'warning') {
    console.log(`  console.warn: ${m.params.args.map((a) => a.value ?? a.description ?? '').join(' ')}`);
  }
});
await S('Runtime.enable');
await S('Log.enable');
await S('Page.enable');
await S('Emulation.setDeviceMetricsOverride', {
  width: WIDTH, height: HEIGHT, deviceScaleFactor: 1, mobile: WIDTH < 600,
});

async function evaluate(expression, timeoutMs = 60000) {
  const r = await Promise.race([
    S('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }),
    sleep(timeoutMs).then(() => { throw new Error(`evaluate timed out: ${expression.slice(0, 80)}`); }),
  ]);
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
  return r.result.value;
}

async function capture(path) {
  const { data } = await S('Page.captureScreenshot', { format: 'png' });
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, Buffer.from(data, 'base64'));
  return data;
}

let ok = true;
try {
  const [n = '0', iface = 'motion'] = (VIEW ?? '0,motion').split(',');
  const t0 = Date.now();
  await S('Page.navigate', { url: `${pageUrl}?clip=${Number(n)}&iface=${iface}${QUERY ? `&${QUERY}` : ''}` });
  await waitFor(() => evaluate('Boolean(window.__demo)', 5000), 60000, 'the page script');
  const st = () => evaluate('window.__demo.state()', 10000);
  await waitFor(async () => {
    const s = await st();
    if (s.failed) throw Object.assign(new Error(`the page shows an error: ${s.bootText}`), { fatal: true });
    return s.ready && s.steps > 0;
  }, 240000, 'the demo to boot').catch(async (e) => {
    const s = await st().catch(() => null);
    throw new Error(`${e.message}${s ? ` (${s.bootText || 'no boot text'})` : ''}`);
  });
  console.log(`booted in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  const ready = await st();
  if (ready.iface !== iface) errors.push(`asked for ${iface}, showing ${ready.iface}`);
  await waitFor(async () => {
    const s = await st();
    if (s.failed) throw Object.assign(new Error(`the page shows an error: ${s.bootText}`), { fatal: true });
    return s.time >= AT;
  }, Math.max(60000, AT * 20000), `${AT} s of sim time`);
  if (REHOMES) {
    let sawFade = false;
    const at = await waitFor(async () => {
      const s = await st();
      sawFade ||= s.fading;
      return s.rehomes >= REHOMES ? s.time : null;
    }, 300000, `${REHOMES} re-home(s)`);
    console.log(`re-home ${REHOMES} at sim t=${at.toFixed(2)} s; the stage was seen faded: ${sawFade}`);
    await waitFor(async () => (await st()).time >= at + 0.6, 60000, 'the fade');
  }
  await evaluate('window.__demo.drawn()');
  await capture(out);
  const s = await st();
  console.log(`clip ${n} (${(await evaluate('window.__demo.clips'))[Number(n)]}) on ${s.iface}: ` +
    `sim t=${s.time.toFixed(2)} s, steps ${s.steps}, rtf ${s.rtf.toFixed(2)}, drawn ${s.fps.toFixed(1)} fps, ` +
    `re-homes ${s.rehomes}`);
  for (const r of s.status) {
    console.log(`  ${r.name}: ${r.phase}, clip ${r.clip} -> ${r.nextClip}, falls ${r.falls}, joint err ${r.jointErr.toFixed(3)}`);
  }
  console.log(`wrote ${out}`);
} catch (e) {
  ok = false;
  console.error(`FAIL: ${e.message}`);
}

if (errors.length) {
  ok = false;
  console.error(`console errors (${errors.length}):`);
  for (const e of errors) console.error(`  ${e}`);
} else {
  console.log('console errors: none');
}
// Close Chrome through CDP and wait for it, so its helpers stop writing to the profile before
// cleanup() removes it.
const chromeExit = chrome.exitCode === null ? new Promise((r) => chrome.once('exit', r)) : null;
await Promise.race([send('Browser.close').catch(() => {}), sleep(3000)]);
await Promise.race([chromeExit, sleep(5000)]);
ws.close();
for (const p of procs) p.kill('SIGKILL');
await Promise.all(procs.map((p) => (p.exitCode === null && p.signalCode === null
  ? new Promise((r) => p.once('exit', r)) : null)));
cleanup();
process.exit(ok ? 0 : 1);
