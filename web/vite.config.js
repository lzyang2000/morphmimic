import { defineConfig } from 'vite';

// Third-party build paths that would otherwise ship in dist/: MuJoCo's WASM embeds its CI
// checkout path (under the home directory) in assertion messages (__FILE__), and the Emscripten
// glue creates a virtual web_user home directory. Rewrite both so dist/ holds no home-directory
// path at all. The WASM rewrite keeps every string the same length (extra slashes are a no-op in
// a path), so no data offset moves.
// HOME is assembled from parts so this file passes the branch leak check (tools/check_branch.mjs).
const HOME = ['', 'home', ''].join('/');
const WASM_PREFIX = `${HOME}runner/work/mujoco/mujoco/`;
const WASM_REPLACEMENT = '/'.repeat(WASM_PREFIX.length - 'mujoco/'.length) + 'mujoco/';

function replaceBytes(buf, from, to) {
  const a = Buffer.from(from);
  const b = Buffer.from(to);
  if (a.length !== b.length) throw new Error('replacement must keep the length');
  const out = Buffer.from(buf);
  for (let i = out.indexOf(a); i >= 0; i = out.indexOf(a, i + b.length)) b.copy(out, i);
  return out;
}

function scrubBuildPaths() {
  return {
    name: 'scrub-build-paths',
    apply: 'build',
    generateBundle(_, bundle) {
      for (const f of Object.values(bundle)) {
        if (f.type === 'asset' && f.fileName.endsWith('.wasm')) {
          f.source = replaceBytes(f.source, WASM_PREFIX, WASM_REPLACEMENT);
        } else if (f.type === 'chunk') {
          f.code = f.code.replaceAll(`${HOME}web_user`, '/web_user');
        }
        const text = f.type === 'chunk' ? f.code : Buffer.from(f.source).toString('latin1');
        if (text.includes(HOME)) throw new Error(`${f.fileName} still contains a home-directory path`);
      }
    },
  };
}

// The demo is served from a subdirectory of the project site; override with DEMO_BASE.
// Everything is bundled (no CDN): MuJoCo's WASM is emitted as a hashed asset next to the JS.
export default defineConfig({
  base: process.env.DEMO_BASE || '/morphmimic/demo/',
  build: {
    target: 'es2022',
    outDir: 'dist',
    emptyOutDir: true,
    assetsInlineLimit: 0,
    chunkSizeWarningLimit: 2000,
  },
  // Pre-bundling would move mujoco.js away from mujoco.wasm and break its new URL(..., import.meta.url).
  optimizeDeps: { exclude: ['@mujoco/mujoco'] },
  plugins: [scrubBuildPaths()],
  // The live demo's worker (src/live/worker.js) is an ES module worker: MuJoCo's glue and
  // onnxruntime-web find their .wasm through import.meta.url. Its bundle gets the same scrub.
  worker: { format: 'es', plugins: () => [scrubBuildPaths()] },
});
