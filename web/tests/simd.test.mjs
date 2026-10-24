// The WebAssembly SIMD probe (src/live/simd.js): the page checks it before any download.
import test from 'node:test';
import assert from 'node:assert/strict';
import { NO_SIMD_MESSAGE, SIMD_PROBE, wasmSimdSupported } from '../src/live/simd.js';

test('the probe module validates where SIMD is supported (Node), and only as a SIMD module', () => {
  assert.equal(WebAssembly.validate(SIMD_PROBE), true);
  assert.equal(wasmSimdSupported(), true, 'the default validate is WebAssembly.validate');
  // Its validity rests on SIMD: the function returns a v128 (type 0x7b) and its body uses the
  // SIMD prefix 0xfd (i8x16.splat, i8x16.popcnt). Node cannot turn SIMD off, so the browser
  // without it is covered by the injected validate below.
  assert.equal(SIMD_PROBE[14], 0x7b, 'result type v128');
  assert.deepEqual(Array.from(SIMD_PROBE.slice(-6)), [0, 0xfd, 15, 0xfd, 98, 11]);
  assert.equal(WebAssembly.validate(SIMD_PROBE.slice(0, -1)), false, 'a truncated module is rejected');
});

test('wasmSimdSupported is false when validate says no, throws, or is missing', () => {
  const seen = [];
  assert.equal(wasmSimdSupported((b) => { seen.push(b); return false; }), false);
  assert.equal(seen[0], SIMD_PROBE, 'it validates the probe module');
  assert.equal(wasmSimdSupported(() => { throw new Error('no'); }), false);
  assert.equal(wasmSimdSupported(null), false);
  assert.equal(wasmSimdSupported(() => 'yes'), false, 'only a true result counts');
  assert.equal(wasmSimdSupported(() => true), true);
  assert.match(NO_SIMD_MESSAGE, /WebAssembly SIMD/);
  assert.match(NO_SIMD_MESSAGE, /Safari 16\.4\+/);
});
