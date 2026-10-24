// Headless check of the project page in the gh-pages worktree: serves it with
// `python3 -m http.server`, loads / in headless Chrome (Chrome DevTools Protocol, no extra
// packages) and saves a full-page PNG; clicks Launch, waits for the embedded demo to boot, selects
// a clip and saves the demo frame; then loads /demo/ on its own, selects the clip and saves that.
// Prints console errors and exits 1 if there were any or the demo did not boot.
//
//   node tools/shot_site.mjs OUTDIR [--width 1440] [--height 900] [--pages DIR] [--port 8000]
//                            [--clip N] [--path /morphmimic/]
//
// Writes OUTDIR/page-<W>.png, OUTDIR/page-<W>-demo.png (the page, scrolled to the launched demo)
// and OUTDIR/demo-<W>.png. The demo must be built with DEMO_BASE=/demo/ for this local server.
// CHROME overrides the browser binary (default /usr/bin/google-chrome).

import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defaultPages } from './pages_worktree.mjs';

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const opt = (name, def) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : def;
};
const OUT = argv[0] && !argv[0].startsWith('--') ? resolve(argv[0]) : null;
if (!OUT) {
  console.error('usage: node tools/shot_site.mjs OUTDIR [--width W] [--height H] [--pages DIR] [--port P] [--clip N]');
  process.exit(2);
}
const WIDTH = Number(opt('width', 1440));
const HEIGHT = Number(opt('height', WIDTH < 600 ? 844 : 900));
const PAGES = resolve(opt('pages', defaultPages(WEB)));
const PORT = Number(opt('port', 8000));
const CLIP = Number(opt('clip', 3)); // 0-based sidebar index; 3 = Street: shuffle
const CHROME = process.env.CHROME || '/usr/bin/google-chrome';
const MOBILE = WIDTH < 600;

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
    } catch { /* not yet */ }
    if (Date.now() > end) throw new Error(`timed out waiting for ${what}`);
    await sleep(100);
  }
}

// ------------------------------------------------------------------ server ----
const server = spawn('python3', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1', '-d', PAGES],
  { stdio: ['ignore', 'ignore', 'pipe'] });
procs.push(server);
const http404 = [];
server.stderr.on('data', (d) => {
  for (const line of String(d).split('\n')) if (/" 404 /.test(line)) http404.push(line.trim());
});
// --path serves the site below a prefix: with --pages pointing at a directory that holds a
// morphmimic link to the worktree, --path /morphmimic/ checks the published demo build.
const root = `http://127.0.0.1:${PORT}${opt('path', '/')}`;
await waitFor(async () => (await fetch(root)).ok, 20000, `http.server at ${root}`);

// ------------------------------------------------------------------ chrome ----
mkdirSync(join(WEB, '.cache'), { recursive: true });
profile = mkdtempSync(join(WEB, '.cache/chrome-'));
const chrome = spawn(CHROME, [
  '--headless=new', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  '--no-first-run', '--no-default-browser-check', '--disable-extensions', '--hide-scrollbars',
  '--mute-audio', '--autoplay-policy=no-user-gesture-required', `--user-data-dir=${profile}`,
  '--remote-debugging-port=0', `--window-size=${WIDTH},${HEIGHT}`, 'about:blank',
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
  }
});
await S('Runtime.enable');
await S('Log.enable');
await S('Page.enable');
await S('Emulation.setDeviceMetricsOverride', {
  width: WIDTH, height: HEIGHT, deviceScaleFactor: 1, mobile: MOBILE,
});
if (MOBILE) await S('Emulation.setTouchEmulationEnabled', { enabled: true });

async function evaluate(expression, timeoutMs = 60000) {
  const r = await Promise.race([
    S('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }),
    sleep(timeoutMs).then(() => { throw new Error(`evaluate timed out: ${expression.slice(0, 80)}`); }),
  ]);
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
  return r.result.value;
}

async function capture(path, fullPage) {
  let data;
  if (!fullPage) {
    ({ data } = await S('Page.captureScreenshot', { format: 'png' }));
  } else {
    // Full page: capture the viewport at each scroll position and stitch the pieces in a canvas.
    // (captureBeyondViewport repeats composited layers such as the hero video on long pages.)
    const total = await evaluate('document.documentElement.scrollHeight');
    // Fixed elements (the scroll-to-top button) would repeat in every piece.
    await evaluate(`(() => { const st = document.createElement('style'); st.id = '__shot';
      st.textContent = '.scroll-to-top { display: none !important; }';
      document.head.append(st); return true; })()`);
    const pieces = [];
    for (let y = 0; ; y += HEIGHT) {
      // behavior 'instant' overrides the page's smooth scrolling, so scrollY is final when read.
      const at = await evaluate(`(async () => { scrollTo({ top: ${y}, behavior: 'instant' });
        await new Promise((r) => setTimeout(r, 450)); return Math.round(scrollY); })()`);
      const shot = await S('Page.captureScreenshot', { format: 'png' });
      pieces.push({ y: at, data: shot.data });
      if (at + HEIGHT >= total) break;
    }
    await evaluate(`(() => { document.getElementById('__shot').remove();
      scrollTo({ top: 0, behavior: 'instant' }); return true; })()`);
    data = await evaluate(`(async () => {
      const pieces = ${JSON.stringify(pieces)};
      const imgs = await Promise.all(pieces.map((p) => new Promise((res, rej) => {
        const im = new Image(); im.onload = () => res(im); im.onerror = rej;
        im.src = 'data:image/png;base64,' + p.data;
      })));
      const c = document.createElement('canvas');
      c.width = ${WIDTH}; c.height = ${total};
      const g = c.getContext('2d');
      imgs.forEach((im, k) => g.drawImage(im, 0, pieces[k].y));
      return c.toDataURL('image/png').split(',')[1];
    })()`, 120000);
  }
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, Buffer.from(data, 'base64'));
  console.log(`wrote ${path}`);
}

// Click clip CLIP in whichever selector the demo shows (sidebar list, or the narrow-screen select).
const pickClip = (win) => `(async () => {
  const w = ${win};
  const d = w.document;
  const b = d.querySelectorAll('#clipList button')[${CLIP}];
  let how;
  if (b && b.offsetParent !== null) { b.click(); how = 'sidebar'; }
  else { const s = d.getElementById('clipSelect'); s.value = s.options[${CLIP}].value;
         s.dispatchEvent(new Event('change')); how = 'select'; }
  await w.__demo.loaded();
  return how + ': ' + d.querySelector('#clipSelect').selectedOptions[0].textContent;
})()`;

let ok = true;
try {
  // 1. The page.
  await S('Page.navigate', { url: root });
  await waitFor(() => evaluate('document.readyState === "complete"', 5000), 30000, 'the page to load');
  // Scroll through once so lazy images and videos load as they would for a reader.
  await evaluate(`(async () => {
    for (let y = 0; y < document.documentElement.scrollHeight; y += innerHeight / 2) {
      scrollTo({ top: y, behavior: 'instant' }); await new Promise((r) => setTimeout(r, 250));
    }
    scrollTo({ top: 0, behavior: 'instant' });
  })()`);
  await sleep(2500); // fonts, FontAwesome icons, images, the banner's first frames
  const overflow = await evaluate('document.documentElement.scrollWidth - innerWidth');
  if (overflow > 0) errors.push(`the page scrolls sideways: ${overflow} px wider than the viewport`);
  else console.log('no horizontal page scroll');
  await capture(join(OUT, `page-${WIDTH}.png`), true);

  // 2. Launch the embedded demo and pick a clip.
  await evaluate(`document.getElementById('demoFrame').scrollIntoView({ block: 'center' }); true`);
  await evaluate(`document.getElementById('demoPoster').click(); true`);
  const t0 = Date.now();
  const frameWin = `document.querySelector('#demoFrame iframe').contentWindow`;
  await waitFor(() => evaluate(`Boolean(${frameWin}.__demo)`, 5000), 120000, 'the embedded demo to boot');
  console.log(`embedded demo booted in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  console.log(`embedded: ${await evaluate(pickClip(frameWin))}`);
  await sleep(3000);
  const box = await evaluate(`(() => { const r = document.getElementById('demoFrame').getBoundingClientRect();
    return [Math.round(r.width), Math.round(r.height)]; })()`);
  console.log(`demo frame ${box[0]} x ${box[1]} px`);
  await capture(join(OUT, `page-${WIDTH}-demo.png`), false);

  // 3. The demo on its own at /demo/.
  await S('Page.navigate', { url: `${root}demo/` });
  await waitFor(() => evaluate('Boolean(window.__demo)', 5000), 120000, 'the demo at /demo/ to boot');
  console.log(`/demo/: ${await evaluate(pickClip('window'))}`);
  await sleep(3000);
  await capture(join(OUT, `demo-${WIDTH}.png`), false);
} catch (e) {
  ok = false;
  console.error(`FAIL: ${e.message}`);
}

for (const l of http404) errors.push(`HTTP 404: ${l}`);
if (errors.length) {
  ok = false;
  console.error(`console errors (${errors.length}):`);
  for (const e of errors) console.error(`  ${e}`);
} else {
  console.log('console errors: none; HTTP 404s: none');
}
const chromeExit = chrome.exitCode === null ? new Promise((r) => chrome.once('exit', r)) : null;
await Promise.race([send('Browser.close').catch(() => {}), sleep(3000)]);
await Promise.race([chromeExit, sleep(5000)]);
ws.close();
for (const p of procs) p.kill('SIGKILL');
await Promise.all(procs.map((p) => (p.exitCode === null && p.signalCode === null
  ? new Promise((r) => p.once('exit', r)) : null)));
cleanup();
process.exit(ok ? 0 : 1);
