// Replays tools/xcheck_live.py's recorded sim2sim ticks through the JS policy side
// (src/live/obs.js, src/live/reference.js) and compares every policy input with the one the
// Python policy node fed its ONNX (obs_history and goal_motion on the motion interface;
// obs_history, goal_human, the two human masks, human_heading and its mask on the human
// interface, full and five points), plus the reference phase and the window's five points (motion)
// or 24 human slot positions (human). Exits 1 if any
// tensor differs by more than 1e-5, a phase differs, or a run is missing. It also checks the
// native joint targets the JS makes from each recorded action (Robot.targets).
//
// The JS side gets only what the robot measured each tick (step id, IMU quaternion and gyro,
// q, dq) and the action the policy returned (the next tick's last action). Its reference is
// the demo's LoopReference, which cues itself: the Python run pressed the same cues by hand.
import { readFileSync, existsSync } from 'node:fs';
import { PolicyInputs, Robot } from '../src/live/obs.js';
import { LoopReference, fivePoints, statesFromClip } from '../src/live/reference.js';
import { humanFromClip, humanPoints } from '../src/live/human.js';

const TOL = 1e-5;
const web = new URL('..', import.meta.url);
const path = (p) => new URL(p, web);
const json = (p) => JSON.parse(readFileSync(path(p), 'utf8'));

const live = json('public/live/robots.json');
const clips = json('public/live/clips.json');
const clipBuffers = new Map();
function clipHuman(id) {
  clipStates(id, 'g1'); // loads the buffer
  return humanFromClip(clipBuffers.get(id), clips.clips.find((c) => c.id === id).human);
}
function clipStates(id, robot) {
  if (!clipBuffers.has(id)) {
    const buf = readFileSync(path(`public/live/clips/${id}.bin`));
    clipBuffers.set(id, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  }
  const entry = clips.clips.find((c) => c.id === id);
  if (!entry) throw new Error(`clip ${id} not in clips.json`);
  return statesFromClip(clipBuffers.get(id), entry.robots[robot], clips.columns);
}

const TENSORS = {
  motion: ['obs_history', 'goal_motion', 'kp', 'targets'],
  human: ['obs_history', 'goal_human', 'human_point_mask', 'human_frame_mask', 'human_heading',
    'human_heading_mask', 'human', 'targets'],
};
// The ONNX inputs are float32: the dump prints them with 9 significant digits, which round
// back to the same float32. `kp` (the window's points) and `targets` (the native joint targets
// from the action) are float64, printed exactly.
const FLOAT32 = new Set(['obs_history', 'goal_motion', 'goal_human', 'human_heading']);
// [max |diff|, elements not bit-identical]
const compare = (got, want, f32) => {
  if (got.length !== want.length) return [Infinity, Math.max(got.length, want.length)];
  let m = 0;
  let n = 0;
  for (let i = 0; i < got.length; i++) {
    const w = f32 ? Math.fround(want[i]) : want[i];
    const d = Math.abs(got[i] - w);
    if (d !== 0) n++;
    if (!(d <= m)) m = Number.isNaN(d) ? Infinity : d;
  }
  return [m, n];
};

let failed = false;
let worst = 0;
const lines = [];
for (const spec of live.robots) {
  for (const iface of ['motion', 'full', 'five']) {
    const tensors = TENSORS[iface === 'motion' ? 'motion' : 'human'];
    const human = iface !== 'motion';
    const file = `.cache/xcheck_live/${spec.name}_${iface}.json`;
    if (!existsSync(path(file))) {
      console.error(`MISSING ${file}: run tools/xcheck_live.py first`);
      failed = true;
      continue;
    }
    const run = json(file);
    const robot = new Robot(spec);
    const ref = new LoopReference(robot, clipStates(run.clipA, spec.name), {
      leadInS: live.loop.leadInS, returnS: live.loop.returnS, holdS: live.loop.holdS,
      firstCueS: run.firstCueS, fps: live.fps,
      ...(human ? { human: clipHuman(run.clipA), humanStand: Float64Array.from(live.human.stand) } : {}),
    });
    const inputs = new PolicyInputs(robot, ref, { historyLen: live.historyLen });
    const max = Object.fromEntries(tensors.map((t) => [t, 0]));
    const differ = Object.fromEntries(tensors.map((t) => [t, 0]));
    let elements = 0;
    const phases = {};
    let phaseErrors = 0;
    for (const row of run.rows) {
      if (row.tick === run.switchTick) ref.setClip(clipStates(run.clipB, spec.name), human ? clipHuman(run.clipB) : null);
      const got = inputs.step(row.step, row.quat, row.gyro, row.q, row.dq, 0, iface);
      if (human) got.human = humanPoints(ref.humanWindow);
      else got.kp = fivePoints(inputs.lastWindow);
      const action = Float32Array.from(row.action); // as onnxruntime returns it
      got.targets = robot.targets(action);
      for (const t of tensors) {
        const [m, n] = compare(got[t], row[t], FLOAT32.has(t));
        max[t] = Math.max(max[t], m);
        differ[t] += n;
        elements += row[t].length;
      }
      if (ref.phase !== row.phase) {
        if (phaseErrors++ < 3) console.error(`${spec.name} ${iface} tick ${row.tick}: phase ${ref.phase}, python ${row.phase}`);
      }
      phases[ref.phase] = (phases[ref.phase] ?? 0) + 1;
      inputs.setAction(action);
    }
    const runWorst = Math.max(...tensors.map((t) => max[t]));
    worst = Math.max(worst, runWorst);
    const ok = runWorst <= TOL && phaseErrors === 0 && run.rows.length === run.ticks;
    if (!ok) failed = true;
    const cols = tensors.map((t) => `${t} ${max[t].toExponential(2)} (${differ[t]} differ)`).join('  ');
    lines.push(`${ok ? 'ok  ' : 'FAIL'} ${spec.name} ${iface.padEnd(8)} ${run.rows.length} ticks, falls ${run.falls}, `
      + `phases ${JSON.stringify(phases)}, phase mismatches ${phaseErrors}, ${elements} values\n       ${cols}`);
  }
}
console.log(lines.join('\n'));
console.log(`worst max |diff| ${worst.toExponential(2)} (tolerance ${TOL})`);
if (failed) {
  console.error('FAIL');
  process.exit(1);
}
