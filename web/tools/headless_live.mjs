// Headless check of the live demo's loop in Node: the real MuJoCo WASM model (public/live/scene.xml),
// the real policy on onnxruntime-web (public/live/policy.onnx), every clip on both interfaces with
// the three robots together as in the page, each robot looping (stand, lead-in, clip, return,
// hold) twice. Runs unthrottled and reports, per run and robot:
//   falls; joint RMS error (rad, the canonical joints against frame 0 of the window the policy
//   tracked, averaged over control steps, as sim_node.run_teleop measures it) over both loops and
//   over the first loop's steps that the Python run covers; end drift (m, the robot's pelvis from
//   its current five-point pelvis target); the wall-clock real-time factor (sim s per wall s).
// With the Python sim2sim summaries of tools/py_from_stand.sh in .cache/headless_live/py/ it puts
// the joint errors side by side. Exits 1 on a fall on the motion interface, or a motion-interface
// joint error more than --tol (0.03 rad) from Python's for the same clip and robot; keypoint falls
// and errors are reported only.
//
//   node tools/headless_live.mjs [--clips 0,3] [--ifaces motion,keypoint] [--loops 2] [--tol 0.03]
//   node tools/headless_live.mjs --compare [--ifaces ...]   (the table again, from the last js.json)
//
// Writes .cache/headless_live/js.json and .cache/headless_live/table.md.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import load_mujoco from '@mujoco/mujoco';
import { loadPolicy } from '../src/live/policy.js';
import { LiveSim } from '../src/live/sim.js';
import { LiveLoop } from '../src/live/loop.js';
import { statesFromClip } from '../src/live/reference.js';

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const LIVE = join(WEB, 'public/live');
const OUT = join(WEB, '.cache/headless_live');
const argv = process.argv.slice(2);
const opt = (name, def) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : def;
};
const IFACES = opt('ifaces', 'motion,keypoint').split(',');
const LOOPS = Number(opt('loops', 2));
const TOL = Number(opt('tol', 0.03));
const json = (p) => JSON.parse(readFileSync(p, 'utf8'));

// --compare: no runs, only the table from the last run's js.json and the Python summaries.
const COMPARE = argv.includes('--compare');
const scene = json(join(LIVE, 'scene.json'));
const robots = json(join(LIVE, 'robots.json'));
const clipIndex = json(join(LIVE, 'clips.json'));
const CLIPS = opt('clips', null)?.split(',').map(Number) ?? clipIndex.clips.map((_, i) => i);
let sim, policy;
if (!COMPARE) {
  const mujoco = await load_mujoco();
  sim = new LiveSim(mujoco, readFileSync(join(LIVE, 'scene.xml'), 'utf8'), scene);
  const bytes = readFileSync(join(LIVE, 'policy.onnx'));
  const ort = await import('onnxruntime-web/wasm');
  policy = await loadPolicy(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength), robots.robots, { ort });
  console.log(`onnxruntime-web ${ort.env.versions?.web ?? ''}, MuJoCo WASM, ${scene.robots.map((r) => r.name).join(' ')}; `
    + `${CLIPS.length} clips x ${IFACES.join('/')} x ${LOOPS} loops`);
}
const fps = robots.fps;
const { leadInS, returnS, holdS } = robots.loop;

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

async function runOne(c, iface) {
  const clip = clipIndex.clips[c];
  const buf = readFileSync(join(LIVE, 'clips', `${clip.id}.bin`));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const states = robots.robots.map((r) => statesFromClip(ab, clip.robots[r.name], clipIndex.columns));
  sim.resetAll();
  const loop = new LiveLoop(sim, policy, robots.robots, states, {
    iface, clipId: c, loop: robots.loop, fps, historyLen: robots.historyLen,
  });
  const T = clip.robots.g1.frames;
  const played = Math.round(leadInS * fps) + T + Math.round(returnS * fps);
  // deploy.run --from-stand 2 --cue-s 1 runs cue + played + 100 steps (2 s after the motion).
  const pySteps = Math.round(1 * fps) + played + 100;
  const nominal = Math.round(holdS * fps) + LOOPS * played + (LOOPS - 1) * Math.round(holdS * fps);
  const cap = nominal + 50 * played; // a robot that keeps falling never finishes its loops
  const n = loop.robots.length;
  const err = Array.from({ length: n }, () => []);
  const start = loop.robots.map((_, r) => sim.slot0(r));
  const at = { py: null };
  const t0 = performance.now();
  let steps = 0;
  while (steps < cap && !loop.robots.every((R) => R.motions >= LOOPS)) {
    await loop.controlStep();
    steps++;
    for (let r = 0; r < n; r++) err[r].push(loop.robots[r].jointErr);
    if (steps === pySteps) at.py = loop.robots.map((_, r) => ({ ...sim.slot0(r), falls: loop.robots[r].falls }));
  }
  const wall = (performance.now() - t0) / 1000;
  const simS = steps / fps;
  const targets = new Float64Array(robots.goalK * 5 * 3);
  return loop.robots.map((R, r) => {
    const end = sim.slot0(r);
    const tgt = loop.fivePointTargets(r, targets);
    // Travel over the Python run's steps, in the start heading frame (sim_node.run_teleop).
    const yaw0 = Math.atan2(2 * (start[r].quat[0] * start[r].quat[3] + start[r].quat[1] * start[r].quat[2]),
      1 - 2 * (start[r].quat[2] ** 2 + start[r].quat[3] ** 2));
    const p1 = at.py?.[r].pos ?? end.pos;
    const dx = p1[0] - start[r].pos[0], dy = p1[1] - start[r].pos[1];
    return {
      clip: clip.id, label: clip.label, robot: R.name, iface, frames: T, steps, loops: R.motions,
      falls: R.falls,
      fallsInPyWindow: at.py?.[r].falls ?? R.falls,
      jointErr: mean(err[r]),
      jointErrPyWindow: mean(err[r].slice(0, pySteps)),
      jointErrMax: Math.max(...err[r]),
      endDrift: tgt ? Math.hypot(end.pos[0] - tgt[0], end.pos[1] - tgt[1]) : null,
      travelAlong: dx * Math.cos(yaw0) + dy * Math.sin(yaw0),
      travelAcross: -dx * Math.sin(yaw0) + dy * Math.cos(yaw0),
      simS, wallS: wall, rtf: simS / wall,
    };
  });
}

mkdirSync(OUT, { recursive: true });
const rows = COMPARE ? json(join(OUT, 'js.json')).filter((r) => IFACES.includes(r.iface)) : [];
const tAll = performance.now();
for (const iface of COMPARE ? [] : IFACES) {
  for (const c of CLIPS) {
    const res = await runOne(c, iface);
    rows.push(...res);
    console.log(`${iface.padEnd(8)} ${clipIndex.clips[c].label.padEnd(24)} ${res[0].steps} steps, rtf ${res[0].rtf.toFixed(2)}: `
      + res.map((x) => `${x.robot} falls ${x.falls} err ${x.jointErr.toFixed(4)} (py window ${x.jointErrPyWindow.toFixed(4)}) drift ${x.endDrift?.toFixed(2)}`).join(' | '));
    writeFileSync(join(OUT, 'js.json'), JSON.stringify(rows, null, 1));
  }
}
const totalWall = COMPARE ? rows.filter((r) => r.robot === 'g1').reduce((a, r) => a + r.wallS, 0) : (performance.now() - tAll) / 1000;

// ------------------------------------------------------------------ compare ----
const pyDir = join(OUT, 'py');
const py = (robot, iface, clip) => {
  const f = join(pyDir, `${robot}_${iface}_${clip}.json`);
  return existsSync(f) ? json(f) : null;
};
let failed = false;
const lines = [];
const f3 = (x) => (x == null ? 'n/a' : x.toFixed(3));
for (const iface of IFACES) {
  lines.push(`\n### ${iface}\n`);
  lines.push('| clip | robot | JS falls (2 loops) | JS joint err, Python window | Python joint err | gap | Python falls | JS joint err, 2 loops | JS end drift (m) | JS travel along/across (m) | Python travel along/across (m) | RTF |');
  lines.push('|---|---|---|---|---|---|---|---|---|---|---|---|');
  for (const x of rows.filter((r) => r.iface === iface)) {
    const p = py(x.robot, iface, x.clip);
    const gap = p ? x.jointErrPyWindow - p.joint_err_rms_rad : null;
    // Pass/fail is on the motion interface; keypoint figures are reported only.
    const bad = iface === 'motion' && (x.falls > 0 || (gap != null && Math.abs(gap) > TOL));
    failed ||= bad;
    lines.push(`| ${x.label} | ${x.robot} | ${x.falls} | ${f3(x.jointErrPyWindow)} | ${f3(p?.joint_err_rms_rad)} | ${gap == null ? 'n/a' : (gap >= 0 ? '+' : '') + gap.toFixed(3)}${bad ? ' **FAIL**' : ''} | ${p ? p.falls : 'n/a'} | ${f3(x.jointErr)} | ${f3(x.endDrift)} | ${f3(x.travelAlong)} / ${f3(x.travelAcross)} | ${p ? `${f3(p.travel_along_m)} / ${f3(p.travel_across_m)}` : 'n/a'} | ${x.rtf.toFixed(2)} |`);
  }
}
const simTotal = rows.filter((r) => r.robot === 'g1').reduce((a, r) => a + r.simS, 0);
const gapsOf = (iface) => rows.filter((x) => x.iface === iface).map((x) => [x, py(x.robot, iface, x.clip)])
  .filter(([, p]) => p).map(([x, p]) => x.jointErrPyWindow - p.joint_err_rms_rad);
const summary = [
  `runs: ${rows.length / 3} (x 3 robots), sim ${simTotal.toFixed(0)} s in ${totalWall.toFixed(0)} s wall: real-time factor ${(simTotal / totalWall).toFixed(2)} (3 robots, unthrottled)`,
  `falls: motion ${rows.filter((r) => r.iface === 'motion').reduce((a, r) => a + r.falls, 0)}, keypoint ${rows.filter((r) => r.iface === 'keypoint').reduce((a, r) => a + r.falls, 0)}`,
  ...IFACES.map((iface) => {
    const gaps = gapsOf(iface);
    return gaps.length ? `${iface}: joint error JS - Python over ${gaps.length} clip/robot pairs: mean ${mean(gaps).toFixed(4)}, `
      + `worst |gap| ${Math.max(...gaps.map(Math.abs)).toFixed(4)} rad${iface === 'motion' ? ` (tol ${TOL})` : ' (reported only)'}`
      : `${iface}: no Python summaries to compare`;
  }),
];
writeFileSync(join(OUT, 'table.md'), `${summary.join('\n')}\n${lines.join('\n')}\n`);
console.log(lines.join('\n'));
console.log(summary.join('\n'));
console.log(failed ? 'FAILED' : 'OK');
process.exit(failed ? 1 : 0);
