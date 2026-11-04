// The live demo's simulation thread: MuJoCo physics and the shared policy for the three robots
// (sim.js, loop.js), off the page's thread. The page only draws what this posts. The command
// side (checks, timer, queued commands, clips fetched on demand) is driver.js.
//
// Page -> worker:
//   {type: 'init', base, clip, iface, rehomeSpread?}  base: absolute URL of the live/ folder
//                                      (ends in '/'); clip: index into clips.json; iface: 'motion' |
//                                      'full' | 'five' (the human interface's slot settings);
//                                      rehomeSpread: the re-home threshold (m, default
//                                      2; loop.js). Accepted once.
//   {type: 'setClip', clip}            from each robot's next cue on (the motion playing ends
//                                      first); a clip not loaded yet is fetched first
//   {type: 'setInterface', iface}      from the next control step on
//   {type: 'pause'} / {type: 'resume'} pause stops physics and inference; resume does not replay
//   {type: 'restart'}                  every robot back to its stand, loops start over
//   {type: 'perturb', robot, body, local: [x, y, z], target: [x, y, z]}
//                                      a drag: pull the point `local` (frame of body id `body`,
//                                      ids as in 'ready', one of robot `robot`'s) toward the world
//                                      point `target`, capped (sim.js); sent again as it moves
//   {type: 'perturbEnd'}               the drag ended (a reset also ends it, sim.js)
//   Commands before 'ready' are answered with a 'not ready' error.
// Worker -> page:
//   {type: 'progress', stage, loaded, total, clip?}  stage: 'mujoco' | 'scene' | 'policy' |
//                                      'clip'; bytes (total 0 when unknown; for the policy,
//                                      robots.json's policy.bytes then); clip: its index
//   {type: 'ready', robots: [{name, label, slot0}], bodies: [names], clips: [{id, label,
//       frames}], fps, goalK, human: {five, parents}, run, iteration}
//       human: the five-point setting's slots and the 24 SMPL slots' parents (robots.json)
//       slot0: the robot's pelvis-equivalent body id
//   {type: 'clip', clip}               a clip is loaded and queued for the next cue
//   {type: 'frame', time, steps, rtf, rehoming, rehomes, poses, targets, status}
//       time: sim seconds; steps: control steps since the start; rehoming: the robots are about
//       to be put back on their home stands (loop.js); rehomes: how many times they were (the
//       page fades out over the cut and back in); rtf: sim time over wall time
//       while running, over the last ~2 s; poses: Float64Array [nbody * 7] (x, y, z, qw, qx, qy, qz
//       per body id, bodies as in 'ready'); targets: Float64Array [robots * goalK * 24 * 3], each
//       robot's human targets (the 24 SMPL slots' world positions of its human window: frame 0
//       the current one, then the preview; NaN before its first step); status: per robot {name, phase, clip, nextClip, stepId, falls, motions,
//       jointErr} (clip and nextClip are clips.json indices)
//   {type: 'error', stage, message, clip?}  stage: 'mujoco' | 'scene' | 'policy' | 'clip' |
//       'run' | 'init' | a command name
// A frame is posted after every control step; its buffers are transferred.

import load_mujoco from '@mujoco/mujoco';
import { browserPolicy } from './ort_browser.js';
import { download } from './policy.js';
import { LiveSim } from './sim.js';
import { LiveLoop } from './loop.js';
import { statesFromClip } from './reference.js';
import { NUM_HUMAN, humanFromClip } from './human.js';
import { Driver } from './driver.js';

let sim = null;
let loop = null;
let meta = null;

const post = (msg, transfer) => self.postMessage(msg, transfer ?? []);

function postFrame() {
  const n = loop.robots.length;
  const per = meta.goalK * NUM_HUMAN * 3;
  const targets = new Float64Array(n * per).fill(NaN);
  for (let r = 0; r < n; r++) loop.humanTargets(r, targets.subarray(r * per, (r + 1) * per));
  const poses = sim.bodyPoses();
  post({
    type: 'frame', time: sim.time, steps: loop.steps, rtf: driver.rate.rtf, rehoming: loop.rehoming,
    rehomes: loop.rehomes, poses, targets, status: loop.status(),
  }, [poses.buffer, targets.buffer]);
}

async function loadClip(i) {
  const c = meta.clipIndex.clips[i];
  const buf = await download(new URL(`clips/${c.id}.bin`, meta.base).href,
    (loaded, total) => post({ type: 'progress', stage: 'clip', clip: i, loaded, total }));
  const states = meta.robots.robots.map((r) => statesFromClip(buf, c.robots[r.name], meta.clipIndex.columns));
  post({ type: 'clip', clip: i });
  return { states, human: humanFromClip(buf, c.human) };
}

const driver = new Driver({ post, postFrame, loadClip });

async function fetchOk(url, how) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res[how]();
}

/** Runs `p` (a promise or a function returning one), labelling its failure with `stage`. */
async function stage(name, p) {
  try {
    return await (typeof p === 'function' ? p() : p);
  } catch (e) {
    throw Object.assign(new Error(String(e?.message ?? e)), { stage: name });
  }
}

async function init({ base, clip = 0, iface = 'motion', rehomeSpread = 2.0 }) {
  if (typeof base !== 'string' || !/\/$/.test(base)) throw new Error('init needs base, the URL of live/ ending in /');
  if (!['motion', 'full', 'five'].includes(iface)) throw new Error(`no interface ${JSON.stringify(iface)}`);
  if (!(rehomeSpread > 0)) throw new Error(`rehomeSpread must be a positive number of metres, got ${JSON.stringify(rehomeSpread)}`);
  post({ type: 'progress', stage: 'mujoco', loaded: 0, total: 0 });
  const mujocoP = stage('mujoco', load_mujoco);
  mujocoP.catch(() => {}); // awaited below; this only stops an unhandled-rejection report
  const [xml, scene, robots, clipIndex] = await stage('scene', Promise.all([
    fetchOk(new URL('scene.xml', base), 'text'),
    fetchOk(new URL('scene.json', base), 'json'),
    fetchOk(new URL('robots.json', base), 'json'),
    fetchOk(new URL('clips.json', base), 'json'),
  ]));
  const nClips = clipIndex.clips.length;
  if (!Number.isInteger(clip) || clip < 0 || clip >= nClips) throw new Error(`no clip ${JSON.stringify(clip)}`);
  post({ type: 'progress', stage: 'scene', loaded: 1, total: 1 });
  const mujoco = await mujocoP;
  sim = await stage('mujoco', () => new LiveSim(mujoco, xml, scene));
  post({ type: 'progress', stage: 'mujoco', loaded: 1, total: 1 });
  meta = { base, clipIndex, robots, goalK: robots.goalK };

  const policy = await stage('policy', () => browserPolicy(new URL('policy.onnx', base).href, robots.robots, {
    // robots.json records the file's size, for a server that sends no content-length.
    onProgress: (loaded, total) => post({ type: 'progress', stage: 'policy', loaded, total: total || robots.policy?.bytes || 0 }),
  }));
  const first = await stage('clip', () => loadClip(clip));

  loop = new LiveLoop(sim, policy, robots.robots, first.states, {
    human: first.human, humanStand: Float64Array.from(robots.human.stand), iface, clipId: clip, loop: robots.loop, fps: robots.fps, historyLen: robots.historyLen,
    rehome: { spread: rehomeSpread, fadeS: 0.15 },
  });
  post({
    type: 'ready',
    robots: scene.robots.map((r, i) => ({ name: r.name, label: r.label, slot0: sim.robots[i].slot0 })),
    bodies: sim.bodyNames,
    clips: clipIndex.clips.map((c) => ({ id: c.id, label: c.label, frames: c.robots[robots.robots[0].name].frames })),
    fps: robots.fps,
    goalK: meta.goalK,
    human: { five: robots.human.five, parents: robots.human.parents },
    run: robots.run,
    iteration: robots.iteration,
  });
  const clips = new Array(nClips).fill(null);
  clips[clip] = first;
  postFrame();
  driver.start(loop, clips, clip);
}

self.onmessage = async ({ data: msg }) => {
  if (msg?.type !== 'init') {
    driver.handle(msg);
    return;
  }
  if (!driver.claimInit()) return;
  try {
    await init(msg);
  } catch (e) {
    post({ type: 'error', stage: e?.stage ?? 'init', message: String(e?.message ?? e) });
  }
};
