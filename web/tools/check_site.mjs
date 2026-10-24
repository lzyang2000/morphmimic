// Pre-publish checks for the gh-pages tree (or any built tree, such as web/dist).
//
//   node tools/check_site.mjs <dir>
//
// Exits 1 if any file under <dir> (the worktree's .git entry excepted):
//   - contains, in its bytes or its path, a string that would leak the internal setup
//     (machine paths, host names, the internal repo's name);
//   - contains a reference-motion field name, or is an .npz file or a checkpoint (.pt, .pth,
//     .ckpt);
//   - lies under a .cache/ or clips/ directory, or is an .onnx file, except what the live demo
//     ships (see LIVE below);
// or if a static/ file the top-level index.html references (src, href, poster, data-src,
// url(...); tools/site_refs.mjs) is missing from the tree.
// An empty directory passes.
//
// LIVE: the live demo (at the root of web/dist, under demo/ on the site) runs the policy in the
// browser, so its live/ folder ships the policy and the reference clips the policy tracks:
// live/policy.onnx (its size must match the bytes robots.json records), live/clips/<id>.bin for
// the clips live/clips.json lists, and the live JSON and scene files. Nowhere else.

import { readdirSync, readFileSync, statSync, existsSync } from 'node:fs';
import { join, relative, resolve, sep } from 'node:path';

import { LEAKS } from './leaks.mjs';
import { staticRefs } from './site_refs.mjs';

// Field names of the reference clips. The recordings hold only simulated qpos, so these must not
// appear anywhere in the shipped files.
const REFERENCE_FIELDS = ['ghost_qpos', 'init_qpos', 'init_qvel'];
const FORBIDDEN_DIRS = ['.cache', 'clips'];
const FORBIDDEN_EXT = /\.(npz|pt|pth|ckpt)$/i;
// Where the live demo's folder may be: the root of a built demo, or demo/ on the site.
const LIVE_DIRS = ['live', 'demo/live'];
const LIVE_FILES = new Set(['policy.onnx', 'robots.json', 'clips.json', 'scene.json', 'scene.xml']);

const root = process.argv[2] ? resolve(process.argv[2]) : null;
if (!root || !existsSync(root) || !statSync(root).isDirectory()) {
  console.error('usage: node tools/check_site.mjs <dir>');
  process.exit(2);
}

const readJson = (p) => {
  try {
    return JSON.parse(readFileSync(p, 'utf8'));
  } catch {
    return null;
  }
};

// Per live folder (relative path with /): the files it may ship, with the size each must have
// (null: any size).
const liveAllowed = new Map();
for (const dir of LIVE_DIRS) {
  const abs = join(root, ...dir.split('/'));
  if (!existsSync(join(abs, 'clips.json'))) continue;
  const clips = readJson(join(abs, 'clips.json'));
  const robots = readJson(join(abs, 'robots.json'));
  for (const f of LIVE_FILES) liveAllowed.set(`${dir}/${f}`, null);
  if (robots?.policy?.file === 'policy.onnx') liveAllowed.set(`${dir}/policy.onnx`, robots.policy.bytes);
  for (const c of clips?.clips ?? []) {
    if (typeof c.id === 'string' && /^[\w.-]+$/.test(c.id)) liveAllowed.set(`${dir}/clips/${c.id}.bin`, null);
  }
}
const liveDir = (rel) => LIVE_DIRS.find((d) => rel.startsWith(`${d}/`));

function* walk(dir) {
  for (const ent of readdirSync(dir, { withFileTypes: true })) {
    if (dir === root && ent.name === '.git') continue; // the worktree link, never committed
    const p = join(dir, ent.name);
    if (ent.isDirectory()) yield* walk(p);
    else yield p;
  }
}

const problems = [];
let nFiles = 0;
let nBytes = 0;
const htmlFiles = [];

for (const file of walk(root)) {
  const rel = relative(root, file);
  nFiles += 1;
  const parts = rel.split(sep);
  const relPosix = parts.join('/');
  const live = liveAllowed.has(relPosix);
  if (live && liveAllowed.get(relPosix) !== null && statSync(file).size !== liveAllowed.get(relPosix)) {
    problems.push(`${rel}: ${statSync(file).size} bytes, but robots.json records ${liveAllowed.get(relPosix)}`);
  }
  if (!live && liveDir(relPosix) && liveAllowed.size) {
    problems.push(`${rel}: not one of the live demo's files (policy.onnx, the clips clips.json lists, the live JSON)`);
  }
  for (const d of FORBIDDEN_DIRS) {
    if (!live && parts.slice(0, -1).includes(d)) problems.push(`${rel}: lies under a ${d}/ directory`);
  }
  if (FORBIDDEN_EXT.test(rel)) problems.push(`${rel}: ${rel.slice(rel.lastIndexOf('.'))} files may not ship`);
  if (!live && /\.onnx$/i.test(rel)) problems.push(`${rel}: a policy file outside the live demo's live/policy.onnx`);
  for (const s of LEAKS) if (rel.includes(s)) problems.push(`${rel}: path contains "${s}"`);

  const buf = readFileSync(file);
  nBytes += buf.length;
  for (const s of [...LEAKS, ...REFERENCE_FIELDS]) {
    const at = buf.indexOf(s);
    if (at >= 0) {
      const ctx = buf.subarray(Math.max(0, at - 30), at + s.length + 30).toString('latin1')
        .replace(/[^\x20-\x7e]/g, '.');
      problems.push(`${rel}: contains "${s}" at byte ${at}: ...${ctx}...`);
    }
  }
  if (/\.html?$/i.test(rel)) htmlFiles.push([rel, buf.toString('utf8')]);
}

for (const [rel, html] of htmlFiles) {
  if (rel === 'index.html') {
    for (const ref of staticRefs(html)) {
      const abs = join(root, ...ref.split('/'));
      if (!existsSync(abs) || !statSync(abs).isFile()) problems.push(`${rel}: references ${ref}, which is not in the tree`);
    }
  }
}

console.log(`checked ${nFiles} files (${(nBytes / 1e6).toFixed(1)} MB) under ${root}`);
if (problems.length) {
  console.error(`FAIL: ${problems.length} problem(s)`);
  for (const p of problems) console.error(`  ${p}`);
  process.exit(1);
}
const shipped = [...liveAllowed.keys()].filter((k) => k.endsWith('.onnx') || k.endsWith('.bin')).length;
console.log(`OK: no leak strings, no reference data or checkpoints outside the live demo (${shipped} policy and clip files), `
  + 'no .cache/ files, every static/ file index.html references is there');
