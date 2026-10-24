// Copy the built demo and the page videos into the gh-pages worktree, check the tree, and commit
// there. It never pushes: publishing is a separate, manual step once the authors approve.
//
//   node tools/deploy_ghpages.mjs [--pages DIR] [--no-commit] [-m MESSAGE]
//
//   web/dist/                  -> <pages>/demo/          (replaced wholesale)
//   web/.cache/videos/*.mp4|jpg -> <pages>/static/videos/ (simulation clips, banner)
//   web/.cache/hardware/*.mp4|jpg -> <pages>/static/videos/ (hardware clips)
// Files in static/videos/ that are in neither source are deleted, except a file the page
// (<pages>/index.html) references (src, href, poster, data-src, url(...); tools/site_refs.mjs):
// nothing under static/ that the page uses is ever removed. The film (morphmimic.mp4 and its
// poster) is made outside this repo and lives only there and in the gh-pages tree; place a new
// version in web/.cache/videos to deploy it.
//
// <pages> defaults to the worktree that has gh-pages checked out (tools/pages_worktree.mjs). The page itself (index.html,
// static/css, static/js, static/images) lives in the gh-pages worktree and is edited there.
// Then runs tools/check_site.mjs on <pages> and stops if it fails.
// Commits only a demo built with the default base (/morphmimic/demo/); a preview build
// (DEMO_BASE=/demo/) can be deployed with --no-commit for a local check.

import { spawnSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defaultPages } from './pages_worktree.mjs';
import { staleFiles, staticRefs } from './site_refs.mjs';

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const PUBLISH_BASE = '/morphmimic/demo/';
const argv = process.argv.slice(2);

if (argv.some((a) => a === '--push' || a.startsWith('--push='))) {
  console.error('refusing --push: this tool only commits locally. Push by hand once the authors approve.');
  process.exit(2);
}
const opt = (name, def) => {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : def;
};
const PAGES = resolve(opt('--pages', defaultPages(WEB)));
const COMMIT = !argv.includes('--no-commit');
const MESSAGE = opt('-m', 'Deploy the demo and videos');

function git(...args) {
  const r = spawnSync('git', ['-C', PAGES, ...args], { encoding: 'utf8' });
  if (r.status !== 0) throw new Error(`git ${args.join(' ')} failed:\n${r.stderr}`);
  return r.stdout.trim();
}

// ------------------------------------------------------------------ checks ----
const DIST = join(WEB, 'dist');
if (!existsSync(join(DIST, 'index.html'))) throw new Error('web/dist is missing; run npm run build');
if (!existsSync(join(PAGES, '.git'))) throw new Error(`${PAGES} is not a git worktree`);
const branch = git('symbolic-ref', '--short', 'HEAD'); // works before the first commit too
if (branch !== 'gh-pages') throw new Error(`${PAGES} is on branch ${branch}, expected gh-pages`);

// The base the demo was built with, read from its entry script URL.
const distHtml = readFileSync(join(DIST, 'index.html'), 'utf8');
const m = /<script[^>]+src="([^"]*)assets\/[^"]+\.js"/.exec(distHtml);
const base = m ? m[1] : null;
console.log(`demo build base: ${base ?? '(not found)'}`);
if (COMMIT && base !== PUBLISH_BASE) {
  console.error(`refusing to commit a demo built with base ${base}; rebuild with the default ` +
    `(${PUBLISH_BASE}) or pass --no-commit for a local preview`);
  process.exit(1);
}

// ------------------------------------------------------------------- copy -----
// Both targets end up holding exactly their sources, so no stale build file or video ships.
const demoDir = join(PAGES, 'demo');
rmSync(demoDir, { recursive: true, force: true });
cpSync(DIST, demoDir, { recursive: true });
console.log(`copied web/dist -> demo/ (old demo/ removed first)`);

const videoDir = join(PAGES, 'static/videos');
const sources = new Map(); // file name -> source path
for (const sub of ['videos', 'hardware']) {
  const src = join(WEB, '.cache', sub);
  if (!existsSync(src)) {
    console.warn(`  no ${src}; skipped`);
    continue;
  }
  for (const f of readdirSync(src)) {
    if (!/\.(mp4|jpg)$/i.test(f)) continue;
    if (sources.has(f)) throw new Error(`${f} is in both .cache/videos and .cache/hardware`);
    sources.set(f, join(src, f));
  }
}
mkdirSync(videoDir, { recursive: true });
const pageHtml = join(PAGES, 'index.html');
const refs = existsSync(pageHtml) ? staticRefs(readFileSync(pageHtml, 'utf8')) : new Set();
const { remove, kept } = staleFiles(readdirSync(videoDir), sources, refs, 'static/videos/');
for (const f of kept) console.log(`  kept static/videos/${f}: in no source, but index.html references it`);
for (const f of remove) {
  rmSync(join(videoDir, f), { recursive: true, force: true });
  console.log(`  removed stale static/videos/${f}`);
}
for (const [f, src] of sources) cpSync(src, join(videoDir, f));
console.log(`copied ${sources.size} video and poster files -> static/videos/`);

// ------------------------------------------------------------- site check -----
const check = spawnSync(process.execPath, [join(WEB, 'tools/check_site.mjs'), PAGES], { stdio: 'inherit' });
if (check.status !== 0) {
  console.error('check_site failed; nothing committed');
  process.exit(1);
}

// ----------------------------------------------------------------- commit -----
if (!COMMIT) {
  console.log('--no-commit: left the changes uncommitted');
  process.exit(0);
}
git('add', '-A');
const staged = spawnSync('git', ['-C', PAGES, 'diff', '--cached', '--quiet']);
if (staged.status === 0) {
  console.log('nothing to commit');
} else {
  git('commit', '-q', '-m', MESSAGE);
  console.log(`committed on gh-pages: ${git('log', '-1', '--format=%h %s')}`);
}
console.log('not pushed');
