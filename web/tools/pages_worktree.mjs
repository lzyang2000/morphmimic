// Default location of the gh-pages worktree: the worktree of this repository that has gh-pages
// checked out, else ../morphmimic-pages next to this worktree. --pages DIR overrides it.

import { spawnSync } from 'node:child_process';
import { join } from 'node:path';

export function defaultPages(web) {
  const r = spawnSync('git', ['-C', web, 'worktree', 'list', '--porcelain'], { encoding: 'utf8' });
  if (r.status === 0) {
    let path = null;
    for (const line of r.stdout.split('\n')) {
      if (line.startsWith('worktree ')) path = line.slice('worktree '.length);
      else if (line === 'branch refs/heads/gh-pages' && path) return path;
    }
  }
  return join(web, '../../morphmimic-pages');
}
