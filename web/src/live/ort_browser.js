// Browser-side loader for the policy: points onnxruntime-web at the .wasm that Vite emits into
// the build (hashed, next to the page's own assets), so nothing is fetched from a CDN. Not
// imported in Node, where the module finds its own .wasm under node_modules.
import * as ort from 'onnxruntime-web/wasm';
// A relative path: the package's exports map hides its .wasm files from a bare specifier.
import wasmUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import { loadPolicy } from './policy.js';

/** loadPolicy with the bundled ORT: browserPolicy(url, robots, {onProgress}). */
export const browserPolicy = (source, robots, options = {}) =>
  loadPolicy(source, robots, { ...options, ort, wasmPaths: { wasm: wasmUrl } });
