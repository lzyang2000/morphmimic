// tools/check_site.mjs on small trees: the live demo's policy and clips pass in its live/ folder
// (a built demo's root, or demo/ on the site) and nowhere else; leak strings, reference fields
// and checkpoints still fail.
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { LEAKS } from '../tools/leaks.mjs';

const TOOL = join(dirname(fileURLToPath(import.meta.url)), '../tools/check_site.mjs');
const POLICY = Buffer.alloc(64, 7);

function tree(files) {
  const root = mkdtempSync(join(tmpdir(), 'check-site-'));
  for (const [rel, body] of Object.entries(files)) {
    mkdirSync(join(root, dirname(rel)), { recursive: true });
    writeFileSync(join(root, rel), body);
  }
  return root;
}

function check(files) {
  const root = tree(files);
  try {
    const r = spawnSync(process.execPath, [TOOL, root], { encoding: 'utf8' });
    return { ok: r.status === 0, out: r.stdout + r.stderr };
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

// A built demo's live/ folder, under `at`.
const live = (at, extra = {}) => ({
  [`${at}live/policy.onnx`]: POLICY,
  [`${at}live/robots.json`]: JSON.stringify({ run: 'r', policy: { file: 'policy.onnx', bytes: POLICY.length, sha256: 'x' } }),
  [`${at}live/clips.json`]: JSON.stringify({ clips: [{ id: 'walk_1', label: 'Walk' }] }),
  [`${at}live/scene.json`]: '{}',
  [`${at}live/scene.xml`]: '<mujoco/>',
  [`${at}live/clips/walk_1.bin`]: Buffer.alloc(32, 1),
  ...extra,
});
const SITE = '<a class="button"><span>Code</span></a>';
const DEMO = '<title>demo</title>';

test('a built demo and the site with the demo under demo/ pass', () => {
  assert.ok(check({ 'index.html': DEMO, ...live('') }).ok, 'web/dist');
  const r = check({ 'index.html': SITE, 'demo/index.html': DEMO, ...live('demo/') });
  assert.ok(r.ok, r.out);
});

test('policy and clip files outside the live demo fail', () => {
  assert.match(check({ 'index.html': SITE, 'static/policy.onnx': POLICY }).out, /policy file outside/);
  assert.match(check({ 'index.html': SITE, 'static/clips/walk_1.bin': 'x' }).out, /under a clips\/ directory/);
  assert.match(check({ 'index.html': SITE, ...live('demo/', { 'demo/live/clips/other.bin': 'x' }) }).out,
    /not one of the live demo's files/);
  assert.match(check({ 'index.html': SITE, ...live('demo/', { 'demo/live/policy.onnx': Buffer.alloc(65) }) }).out,
    /robots.json records 64/);
  assert.match(check({ 'index.html': SITE, ...live('demo/', { 'demo/live/ckpt.pt': 'x' }) }).out, /\.pt files may not ship/);
  assert.match(check({ 'index.html': SITE, 'data/a.npz': 'x' }).out, /\.npz files may not ship/);
});

test('leak strings and reference fields fail, in the live demo too', () => {
  assert.match(check({ 'index.html': SITE, ...live('demo/', { 'demo/live/scene.json': `{"p": "${LEAKS[0]}x"}` }) }).out, /contains/);
  assert.match(check({ 'index.html': DEMO, ...live('', { 'live/robots.json': '{"ghost_qpos": 1}' }) }).out, /ghost_qpos/);
});

test('every static/ file the top-level page references must be in the tree', () => {
  const page = `${SITE}<video poster="static/videos/hero_poster.jpg" data-src="./static/videos/film.mp4?v=2"></video>`
    + `<link href="static/images/favicon.ico?v=3"><div style="background: url('static/videos/hero_poster.jpg')"></div>`;
  const files = {
    'static/videos/hero_poster.jpg': 'j', 'static/videos/film.mp4': 'm', 'static/images/favicon.ico': 'i',
  };
  const ok = check({ 'index.html': page, ...files });
  assert.ok(ok.ok, ok.out);
  const missing = { ...files };
  delete missing['static/videos/film.mp4'];
  assert.match(check({ 'index.html': page, ...missing }).out, /references static\/videos\/film\.mp4, which is not in the tree/);
  // Only the top-level page: a sub-page's references are not checked.
  assert.ok(check({ 'index.html': SITE, 'demo/index.html': '<img src="static/x.png">', ...live('demo/') }).ok);
});
