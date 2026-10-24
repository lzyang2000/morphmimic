import { test } from 'node:test';
import assert from 'node:assert/strict';

import { LEAKS, REPO_NAMES, findLeaks } from '../tools/leaks.mjs';

test('the leak list decodes to nine non-empty strings', () => {
  assert.equal(LEAKS.length, 9);
  for (const s of LEAKS) assert.ok(s.length >= 4 && /^[\x21-\x7e]+$/.test(s), s);
});

test('findLeaks reports each hit with its offset and nothing on clean input', () => {
  assert.deepEqual(findLeaks('a clean page about humanoids'), []);
  const text = `x${LEAKS[1]}y`;
  const hits = findLeaks(Buffer.from(text));
  assert.equal(hits.length, 1);
  assert.equal(hits[0].s, LEAKS[1]);
  assert.equal(hits[0].at, 1);
});

test('other repository names decode and match ignoring case', () => {
  assert.equal(REPO_NAMES.length, 6);
  const hits = findLeaks(`see ${REPO_NAMES[3].toLowerCase()} here`, REPO_NAMES, { ignoreCase: true });
  assert.equal(hits.length, 1);
  assert.equal(hits[0].s, REPO_NAMES[3]);
  assert.deepEqual(findLeaks('vertex-clustering decimation', REPO_NAMES, { ignoreCase: true }), []);
});
