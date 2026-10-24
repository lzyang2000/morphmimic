// Feature check for WebAssembly SIMD, which the demo's WebAssembly (onnxruntime-web's SIMD build,
// MuJoCo) needs. The page runs it before it downloads anything, so a browser without SIMD gets a
// clear message instead of a failure after the download. Pure: `validate` is injectable
// (tests/simd.test.mjs).

// The smallest module that uses a SIMD instruction: (func (result v128) i32.const 0
// i8x16.splat i8x16.popcnt), as the wasm-feature-detect project checks it. It validates only
// where SIMD is supported.
export const SIMD_PROBE = Uint8Array.of(
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
);

/** True when WebAssembly SIMD is available: `validate` (default WebAssembly.validate) accepts the
 * probe module. False when it does not, throws, or there is no WebAssembly. */
export function wasmSimdSupported(validate = globalThis.WebAssembly?.validate?.bind(globalThis.WebAssembly)) {
  if (typeof validate !== 'function') return false;
  try {
    return validate(SIMD_PROBE) === true;
  } catch {
    return false;
  }
}

export const NO_SIMD_MESSAGE = 'This browser lacks WebAssembly SIMD, which the demo needs; '
  + 'try a current Chrome, Edge, Firefox or Safari 16.4+.';
