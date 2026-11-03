// The live demo page. A Web Worker (src/live/worker.js) runs MuJoCo and the shared policy for the
// three robots at 50 Hz; this thread only draws the poses it posts (three.js, src/render.js) and
// the controls, and turns a drag on a robot into a push
// (src/drag.js).

import { DragPush } from './drag.js';
import { StageFade } from './fade.js';
import { NO_SIMD_MESSAGE, wasmSimdSupported } from './live/simd.js';
import { Renderer, webglAvailable } from './render.js';
import './style.css';

const BASE = import.meta.env.BASE_URL || '/';
const url = (p) => `${BASE}${p}`.replace(/\/{2,}/g, '/');
const $ = (id) => document.getElementById(id);

const ROBOT_NAMES = { g1: 'Unitree G1', x2: 'AgiBot X2 Ultra', t2: 'Booster T2' };
const MB = (b) => (b / 1e6).toFixed(1);

// ----------------------------------------------------------------- boot screen --
const boot = (msg, pct, detail = '') => {
  $('bootMsg').textContent = msg;
  $('bootBar').style.width = `${Math.max(0, Math.min(100, pct))}%`;
  $('bootDetail').textContent = detail;
};

let failed = false;
function bootError(head, detail, { retry = true } = {}) {
  failed = true;
  if (detail) console.error(detail);
  const box = document.createElement('div');
  const h = document.createElement('div');
  h.className = 'err-head';
  h.textContent = head;
  box.append(h);
  if (detail) {
    const d = document.createElement('div');
    d.className = 'err';
    d.textContent = String(detail?.message ?? detail);
    box.append(d);
  }
  if (retry) {
    const hint = document.createElement('div');
    hint.className = 'err-hint';
    hint.textContent = 'Reload the page to try again.';
    box.append(hint);
  }
  $('boot').classList.remove('hidden');
  $('boot').replaceChildren(box);
}

const STAGE_ERRORS = {
  mujoco: 'MuJoCo (the physics engine, WebAssembly) could not load.',
  scene: 'The robot scene could not be downloaded.',
  policy: 'The policy could not be loaded into onnxruntime-web.',
  clip: 'The first clip could not be downloaded.',
};

let noticeTimer = 0;
function showNotice(msg, ms = 6000) {
  const n = $('notice');
  n.textContent = msg;
  n.hidden = false;
  clearTimeout(noticeTimer);
  if (ms) noticeTimer = setTimeout(() => { n.hidden = true; }, ms);
}
function hideNotice() {
  clearTimeout(noticeTimer);
  $('notice').hidden = true;
}

async function fetchOk(path, kind) {
  const res = await fetch(url(path));
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res[kind]();
}

// --------------------------------------------------------------------- pausing --
// The project page embeds the demo in an iframe and posts { smxDemo: 'pause' | 'resume' } when
// the frame scrolls out of or back into view. The demo also pauses while its tab is hidden.
// Registered before boot, so a pause sent while the demo is still loading is honoured.
const run = { user: true, suspended: false, hidden: document.visibilityState === 'hidden' };
const running = () => run.user && !run.suspended && !run.hidden;
let onRunChange = () => {};
addEventListener('message', (e) => {
  if (e.source !== window.parent || e.source === window) return;
  const cmd = e.data && e.data.smxDemo;
  if (cmd === 'pause') run.suspended = true;
  else if (cmd === 'resume') run.suspended = false;
  else return;
  onRunChange();
});
document.addEventListener('visibilitychange', () => {
  run.hidden = document.visibilityState === 'hidden';
  onRunChange();
});

// Optional deep link: ?clip=<sidebar index>. The page's policy has the motion interface only.
// A debug build (VITE_DEMO_DEBUG=1 npm run build) also takes ?rehome=<m>, the re-home threshold,
// so a check can force a re-home soon; a normal build drops that code.
function fromQuery(nClips) {
  const q = new URLSearchParams(location.search);
  const c = Number(q.get('clip'));
  const out = {
    clip: q.has('clip') && Number.isInteger(c) && c >= 0 && c < nClips ? c : 0,
    iface: 'motion',
  };
  if (import.meta.env.VITE_DEMO_DEBUG === '1' && Number(q.get('rehome')) > 0) out.rehomeSpread = Number(q.get('rehome'));
  return out;
}

async function main() {
  if (!webglAvailable()) {
    bootError('This demo needs WebGL, which this browser does not support or has turned off. ' +
      'Try another browser, or turn on hardware acceleration.', null, { retry: false });
    return;
  }
  if (typeof WebAssembly !== 'object' || typeof Worker !== 'function') {
    bootError('This demo needs WebAssembly and Web Workers, which this browser does not provide.', null, { retry: false });
    return;
  }
  // Before any download: the WebAssembly (onnxruntime-web's SIMD build, MuJoCo) needs SIMD.
  if (!wasmSimdSupported()) {
    bootError(NO_SIMD_MESSAGE, null, { retry: false });
    return;
  }

  boot('Loading the scene…', 2);
  const [clipIndex, sceneMeta, meshIndex, meshBuf, visuals] = await Promise.all([
    fetchOk('live/clips.json', 'json'),
    fetchOk('model/scene.json', 'json'),
    fetchOk('model/meshes.json', 'json'),
    fetchOk('model/meshes.bin', 'arrayBuffer'),
    fetchOk('model/visuals.json', 'json'),
  ]).catch((e) => {
    bootError('The robot models could not be downloaded.', e);
    throw e;
  });
  const clips = clipIndex.clips;
  const start = fromQuery(clips.length);

  const renderer = new Renderer($('view'), $('labels'), sceneMeta);
  const nMeshes = renderer.addMeshes(meshIndex, meshBuf, visuals);
  if (nMeshes !== visuals.length) console.warn(`attached ${nMeshes} of ${visuals.length} visual meshes`);
  const nRobots = sceneMeta.robots.length;
  sceneMeta.robots.forEach((r, ri) => renderer.setLabel(ri, ROBOT_NAMES[r.name] ?? r.label));
  // Hidden until the first posted frame poses them.
  sceneMeta.robots.forEach((_, ri) => renderer.setRobotVisible(ri, false));

  const state = {
    ready: false,
    selected: start.clip, // the visitor's clip (the next cue plays it)
    iface: start.iface,
    loading: null, // {clip, loaded, total} while a selected clip downloads
    loadedClips: new Set(), // clips the worker holds
    frame: null, // the last posted frame
    lastFalls: new Array(nRobots).fill(0),
    fellAt: new Array(nRobots).fill(-Infinity),
    halted: false, // a run error stopped the sim until Restart
  };

  // The re-home cut (loop.js puts the robots back on their stands when they drift apart): fade
  // the stage out while the worker says one is coming, cut, fade back in (src/fade.js).
  const fade = new StageFade($('fade'));

  // --------------------------------------------------------------- sidebar --
  const list = $('clipList');
  const select = $('clipSelect');
  const items = [];
  clips.forEach((c, i) => {
    const li = document.createElement('li');
    const b = document.createElement('button');
    b.type = 'button';
    const name = document.createElement('span');
    name.textContent = c.label;
    const dur = document.createElement('span');
    dur.className = 'dur';
    const frames = Object.values(c.robots)[0].frames;
    dur.textContent = `${Math.round(frames / clipIndex.fps)} s`;
    b.append(name, dur);
    b.addEventListener('click', () => selectClip(i));
    li.append(b);
    list.append(li);
    items.push(b);
    select.add(new Option(c.label, String(i)));
  });
  select.addEventListener('change', () => selectClip(Number(select.value)));

  function showSelection() {
    items.forEach((b, i) => b.setAttribute('aria-current', String(i === state.selected)));
    select.value = String(state.selected);
  }
  showSelection();

  // --------------------------------------------------------------- worker --
  let worker;
  try {
    worker = new Worker(new URL('./live/worker.js', import.meta.url), { type: 'module' });
  } catch (e) {
    bootError('The simulation worker could not start.', e);
    return;
  }
  const send = (msg) => worker.postMessage(msg);

  // Drag a robot to push it. The hint shows until the first drag.
  const hint = $('pushHint');
  const drag = new DragPush(renderer, $('stage'), {
    send: (msg) => { if (state.ready) send(msg); },
    simBody: (b) => bodyIds[b],
    onStart: () => {
      if (!hint.classList.contains('gone')) hint.classList.add('gone');
    },
  });

  function selectClip(i) {
    if (!Number.isInteger(i) || i < 0 || i >= clips.length) return;
    state.selected = i;
    showSelection();
    if (!state.ready) return;
    if (!state.loadedClips.has(i)) state.loading = { clip: i, loaded: 0, total: 0 };
    send({ type: 'setClip', clip: i });
    updateStatus();
  }

  // -------------------------------------------------------------- controls --
  const btnPlay = $('btnPlay');
  const btnRestart = $('btnRestart');
  function showPlaying() {
    btnPlay.classList.toggle('paused', !run.user);
    btnPlay.setAttribute('aria-label', run.user ? 'Pause' : 'Play');
  }
  let sentRunning = true; // the worker starts running at ready
  onRunChange = () => {
    showPlaying();
    if (!state.ready) return;
    const on = running();
    if (on === sentRunning) return;
    sentRunning = on;
    if (!on) fade.clear(); // paused mid-fade: the stage stays visible (the next frame fades again)
    send({ type: on ? 'resume' : 'pause' });
    updateStatus();
  };
  const togglePlay = () => {
    run.user = !run.user;
    onRunChange();
  };
  btnPlay.addEventListener('click', togglePlay);
  addEventListener('keydown', (e) => {
    if (e.code !== 'Space' || e.target.closest?.('input, select, textarea, button, a')) return;
    e.preventDefault();
    togglePlay();
  });
  btnRestart.addEventListener('click', () => {
    if (!state.ready) return;
    drag.cancel();
    send({ type: 'restart' });
    renderer.resetView();
    fade.clear();
    state.halted = false;
    hideNotice();
    updateStatus();
  });

  // #status is a polite live region: its text is announced when it changes. The speed figure,
  // updated every 2 s, sits in an aria-hidden part; a screen reader hears only that the demo
  // runs below real time, once, when that starts (or stops).
  const setText = (el, text) => { if (el.textContent !== text) el.textContent = text; };
  function updateStatus() {
    if (!state.ready) return;
    const label = (i) => clips[i]?.label ?? '';
    let text;
    if (state.loading && state.loading.clip === state.selected) {
      const { loaded, total } = state.loading;
      text = `Loading ${label(state.selected)}… ${total ? `${MB(loaded)} / ${MB(total)} MB` : `${MB(loaded)} MB`}`;
    } else {
      const st = state.frame?.status ?? [];
      const now = st.length ? st[0].clip : state.selected;
      const queued = st.some((s) => s.clip !== state.selected);
      text = queued ? `${label(now)}, then ${label(state.selected)} at the next cue` : label(now);
    }
    if (state.halted) text = `Stopped · ${text}`;
    else if (!running()) text = `Paused · ${text}`;
    const slow = !state.halted && running() && state.frame && state.frame.rtf < 0.85;
    setText($('statusText'), text);
    setText($('statusSpeed'), slow ? ` · ${state.frame.rtf.toFixed(1)}x speed` : '');
    setText($('statusSlow'), slow ? ', running below real time' : '');
  }

  // ----------------------------------------------------------------- frames --
  let bodyIds = null; // renderer body -> worker body id

  function onReady(m) {
    const index = new Map(m.bodies.map((n, i) => [n, i]));
    bodyIds = Int32Array.from(sceneMeta.bodies, (n) => {
      if (!index.has(n)) throw new Error(`the simulation has no body ${n}`);
      return index.get(n);
    });
    if (m.robots.length !== nRobots || m.robots.some((r, i) => r.name !== sceneMeta.robots[i].name)) {
      throw new Error('the simulation and the drawn scene list different robots');
    }
    state.ready = true;
    drag.enabled = true;
    btnPlay.disabled = false;
    btnRestart.disabled = false;
    // What the visitor did while it loaded.
    if (state.selected !== start.clip) selectClip(state.selected);
    onRunChange();
  }

  function onFrame(m) {
    state.frame = m;
    // A frame from a step still in flight when the visitor paused does not cover the stage again.
    if (fade.frame({ rehoming: m.rehoming && running(), rehomes: m.rehomes })) {
      // The robots are back on their home stands: the camera jumps there.
      drag.cancel(); // every robot is back on its stand: the drag ends (the worker ended the push)
      renderer.recentre();
    }
    renderer.setPoses(m.poses, bodyIds);
    const now = performance.now();
    for (let r = 0; r < nRobots; r++) {
      renderer.setRobotVisible(r, true);
      const s = m.status[r];
      if (s.falls > state.lastFalls[r]) {
        state.fellAt[r] = now;
        drag.cancel(r); // it is back on its stand: the drag on it ends (the worker ended the push)
      }
      state.lastFalls[r] = s.falls;
      const fell = now - state.fellAt[r] < 2500;
      renderer.setLabel(r, ROBOT_NAMES[s.name] ?? s.name, fell ? 'fell, back to the stand' : '');
    }
    updateStatus();
    if (!firstFrame) {
      firstFrame = true;
      renderer.recentre();
      $('boot').classList.add('hidden');
      hint.hidden = false;
    }
  }

  // ------------------------------------------------------------ boot progress --
  const progress = { mujoco: false, scene: false };
  function onProgress(m) {
    if (state.ready) {
      if (m.stage === 'clip') {
        state.loading = { clip: m.clip, loaded: m.loaded, total: m.total };
        updateStatus();
      }
      return;
    }
    if (m.stage === 'mujoco' || m.stage === 'scene') {
      if (m.total && m.loaded >= m.total) progress[m.stage] = true;
      const done = Number(progress.mujoco) + Number(progress.scene);
      boot(progress.mujoco ? 'Loading the robots…' : 'Loading MuJoCo…', 4 + 8 * done);
    } else if (m.stage === 'policy') {
      // The worker sends the size robots.json records when the server sends no content-length.
      const total = m.total ? Math.max(m.total, m.loaded) : 0;
      if (total && m.loaded >= total) boot('Starting the policy…', 90, `${MB(total)} MB`);
      else if (total) boot('Downloading the policy…', 20 + 70 * (m.loaded / total), `${MB(m.loaded)} / ${MB(total)} MB`);
      else boot('Downloading the policy…', 20, `${MB(m.loaded)} MB`);
    } else if (m.stage === 'clip') {
      const f = m.total ? m.loaded / m.total : 0;
      boot(`Loading ${clips[m.clip]?.label ?? 'the clip'}…`, 90 + 10 * f,
        m.total ? `${MB(m.loaded)} / ${MB(m.total)} MB` : `${MB(m.loaded)} MB`);
    }
  }

  let firstFrame = false;
  worker.onmessage = ({ data: m }) => {
    if (failed) return;
    try {
      switch (m.type) {
        case 'progress': onProgress(m); break;
        case 'ready': onReady(m); break;
        case 'frame': if (state.ready) onFrame(m); break;
        case 'clip':
          state.loadedClips.add(m.clip);
          if (state.loading?.clip === m.clip) state.loading = null;
          updateStatus();
          break;
        case 'error': onWorkerError(m); break;
        default: break;
      }
    } catch (e) {
      bootError('The demo stopped.', e);
      worker.terminate();
    }
  };
  function onWorkerError(m) {
    console.error(`worker (${m.stage}): ${m.message}`);
    if (!state.ready) {
      bootError(STAGE_ERRORS[m.stage] ?? 'The demo could not start.', m.message);
      worker.terminate();
      return;
    }
    if (m.stage === 'clip') {
      if (state.loading?.clip === m.clip) state.loading = null;
      const playing = state.frame?.status?.[0]?.nextClip;
      showNotice(`Could not load "${clips[m.clip]?.label}". ` +
        (Number.isInteger(playing) ? `Still playing "${clips[playing].label}".` : ''));
      if (Number.isInteger(playing) && state.selected === m.clip) {
        state.selected = playing;
        showSelection();
      }
    } else if (m.stage === 'run') {
      // The worker runs nothing more until Restart (pause and resume do not clear it).
      state.halted = true;
      showNotice('The simulation stopped with an error. Press Restart to run it again.', 0);
    } else {
      showNotice(`The demo did not accept that (${m.message}).`);
    }
    updateStatus();
  }
  worker.onerror = (e) => {
    e.preventDefault?.();
    const detail = e.message || 'the worker script failed to load';
    if (!state.ready) {
      bootError('The simulation worker could not start.', detail);
      worker.terminate();
    } else {
      showNotice('The simulation stopped with an error. Reload the page to start it again.', 0);
      console.error(detail);
    }
  };

  boot('Loading MuJoCo…', 4);
  showPlaying();
  send({
    type: 'init', base: new URL('live/', document.baseURI).href, clip: start.clip, iface: start.iface,
    ...(start.rehomeSpread ? { rehomeSpread: start.rehomeSpread } : {}),
  });

  // ------------------------------------------------------------------ draw --
  let frameWaiters = [];
  const fps = { n: 0, t0: performance.now(), value: 0 };
  function draw() {
    requestAnimationFrame(draw);
    if (run.suspended || run.hidden || failed) return;
    drag.update();
    renderer.render();
    fps.n++;
    const now = performance.now();
    if (now - fps.t0 >= 1000) {
      fps.value = (1000 * fps.n) / (now - fps.t0);
      fps.n = 0;
      fps.t0 = now;
    }
    const waiters = frameWaiters;
    frameWaiters = [];
    waiters.forEach((f) => f());
  }
  requestAnimationFrame(draw);

  // Hooks for tools/screenshot.mjs (headless checks).
  window.__demo = {
    clips: clips.map((c) => c.label),
    state: () => ({
      ready: state.ready,
      failed,
      iface: state.iface,
      selected: state.selected,
      time: state.frame?.time ?? 0,
      steps: state.frame?.steps ?? 0,
      rtf: state.frame?.rtf ?? 0,
      rehomes: fade.rehomes,
      halted: state.halted,
      fading: fade.on,
      fps: fps.value,
      status: state.frame?.status ?? [],
      bootText: $('boot').classList.contains('hidden') ? '' : $('boot').textContent.trim(),
    }),
    select: selectClip,
    // Client (x, y) of a body (sceneMeta name) as last drawn, and whether a drag is on.
    bodyClient: (name) => {
      const b = sceneMeta.bodies.indexOf(name);
      return b < 0 ? null : renderer.toClient(renderer.bodyPoint(b, { x: 0, y: 0, z: 0 }));
    },
    dragging: () => (drag.active ? drag.drag.robot : null),
    setInterface(iface) {
      if (iface !== 'motion') throw new Error(`the page's policy has the motion interface only, not ${iface}`);
    },
    drawn: () => new Promise((r) => frameWaiters.push(r)),
    // Resolves once the demo runs and the selected clip is in memory (tools/shot_site.mjs).
    loaded: () => new Promise((res, rej) => {
      const poll = () => {
        if (failed) rej(new Error($('boot').textContent.trim()));
        else if (state.ready && state.frame && !state.loading) res(true);
        else setTimeout(poll, 50);
      };
      poll();
    }),
  };
}

main().catch((e) => {
  if (!failed) bootError('The demo could not start.', e);
});
