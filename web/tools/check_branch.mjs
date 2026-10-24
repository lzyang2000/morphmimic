// Pre-push check for this branch: no leak string and no other repository's name (both lists in
// tools/leaks.mjs; repository names matched ignoring case) in any tracked file under web/ (its
// path or its contents) or in any commit message since the base branch.
//
//   node tools/check_branch.mjs [--base REV]
//
// Reads the working-tree copy of every file `git ls-files web` lists, and the messages of
// `git log --format=%B <base>..HEAD`. The base defaults to the merge base of main and HEAD or, when
// they share no history, to the root commit of HEAD (the commit this branch's work starts from).
// Exits 1 on any hit, 0 when clean.

import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { REPO_NAMES, findLeaks } from './leaks.mjs';

// All hits of both lists in buf.
const scan = (buf) => [...findLeaks(buf), ...findLeaks(buf, REPO_NAMES, { ignoreCase: true })];

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const argv = process.argv.slice(2);
const bi = argv.indexOf('--base');

function tryGit(...args) {
  const r = spawnSync('git', ['-C', ROOT, ...args], { encoding: 'utf8', maxBuffer: 1 << 28 });
  return r.status === 0 ? r.stdout : null;
}
function git(...args) {
  const out = tryGit(...args);
  if (out === null) {
    console.error(`git ${args.join(' ')} failed`);
    process.exit(2);
  }
  return out;
}

function defaultBase() {
  const mb = tryGit('merge-base', 'main', 'HEAD');
  if (mb) return { rev: mb.trim(), how: 'merge base with main' };
  const roots = git('rev-list', '--max-parents=0', 'HEAD').split('\n').filter(Boolean);
  return { rev: roots[roots.length - 1], how: 'root commit of HEAD' }; // the oldest root
}
const { rev: BASE, how: BASE_HOW } = bi >= 0
  ? { rev: git('rev-parse', '--verify', `${argv[bi + 1]}^{commit}`).trim(), how: `--base ${argv[bi + 1]}` }
  : defaultBase();

const problems = [];

const files = git('ls-files', '-z', 'web').split('\0').filter(Boolean);
for (const rel of files) {
  for (const h of scan(rel)) problems.push(`${rel}: path contains "${h.s}"`);
  const p = join(ROOT, rel);
  if (!existsSync(p)) continue; // deleted in the working tree, not yet committed
  for (const h of scan(readFileSync(p))) {
    problems.push(`${rel}: contains "${h.s}" at byte ${h.at}: ...${h.ctx}...`);
  }
}

const commits = git('rev-list', `${BASE}..HEAD`).split('\n').filter(Boolean);
for (const c of commits) {
  const msg = git('log', '-1', '--format=%B', c);
  for (const h of scan(msg)) {
    problems.push(`commit ${c.slice(0, 7)} message: contains "${h.s}": ...${h.ctx}...`);
  }
}

console.log(`checked ${files.length} tracked files under web/ and ${commits.length} commit messages ` +
  `(${BASE.slice(0, 7)}..HEAD; base: ${BASE_HOW})`);
if (problems.length) {
  console.error(`FAIL: ${problems.length} problem(s)`);
  for (const p of problems) console.error(`  ${p}`);
  process.exit(1);
}
console.log('OK: no leak strings or other repository names in web/ paths, file contents or commit messages');
