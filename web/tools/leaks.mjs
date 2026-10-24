// The leak list shared by check_site.mjs (shipped trees) and check_branch.mjs (this branch):
// machine paths, the host name and the internal repo names that must never be published; and the
// names of other repositories, which check_branch.mjs also rejects.
//
// The strings are stored base64-encoded so that no tracked file holds them literally; otherwise
// the branch check would flag its own list.

const ENCODED = [
  'L2hvbWUv', 'eWFuZ2w=', 'dGFsb3M=', 'Y2FsdGVjaC5lZHU6', 'c3VwZXJtaW1pY19leF9pbnRlcm5hbA==',
  'c3VwZXJfbWltaWNfcGx1cw==', 'Y2xhdWRlLTEwMDA=', 'L3Jlc25pY2s=', 'bHVuYXJsYWI=',
];

export const LEAKS = ENCODED.map((s) => Buffer.from(s, 'base64').toString('utf8'));

// Names of other repositories, which this branch's files must not mention (check_branch.mjs only;
// matched ignoring case).
const REPO_ENCODED = [
  'cGVyY2VwdGl2ZV9jYmZfcmw=', 'UEFDLU1BTg==', 'aHVtYW5vaWRfZG9kZ2ViYWxs', 'QVJDTA==',
  'c3VwZXJtaW1pY19yZXBsaWNhdGU=', 'c3VwZXJtaW1pY192aWJlZA==',
];
export const REPO_NAMES = REPO_ENCODED.map((s) => Buffer.from(s, 'base64').toString('utf8'));

// Every string of list found in buf (a Buffer or string), with a little printable context.
// ignoreCase compares ASCII case-insensitively.
export function findLeaks(buf, list = LEAKS, { ignoreCase = false } = {}) {
  let b = Buffer.isBuffer(buf) ? buf : Buffer.from(String(buf));
  const orig = b;
  if (ignoreCase) b = Buffer.from(b.toString('latin1').toLowerCase(), 'latin1');
  const hits = [];
  for (const s of list) {
    const at = b.indexOf(ignoreCase ? s.toLowerCase() : s);
    if (at < 0) continue;
    const ctx = orig.subarray(Math.max(0, at - 30), at + s.length + 30).toString('latin1')
      .replace(/[^\x20-\x7e]/g, '.');
    hits.push({ s, at, ctx });
  }
  return hits;
}
