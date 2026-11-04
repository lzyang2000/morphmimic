// The live demo's control loop: every robot in the sim (sim.js) driven by the shared policy
// (policy.js) through its own reference loop (reference.js LoopReference) and policy inputs
// (obs.js PolicyInputs), at 50 Hz on one clock.
//
// One control step (`controlStep`), as deploy/sim2sim's sim_node + policy_node do it for one
// robot: measure every robot; for each robot in turn, its inputs, `await policy.act` (one call at
// a time on the shared session) and the action back into its inputs, exactly as the
// Float32Array act returns; every robot's targets to its actuators; `decimation` physics steps;
// then per robot the joint error against frame 0 of the window it tracked, and the fall check.
// A fallen robot alone goes back to its stand, at its spot, with its step count at 0 (the
// policy's history and the reference restart, as policy_node does on a step regression).
//
// Re-home: robots drift apart over many loops (each places its clip where it stands). When the
// spread, the furthest any two robots have moved relative to each other since their home stands
// (max over pairs of |(p_i - home_i) - (p_j - home_j)|, slot-0 xy), passes `rehome.spread` (m),
// no robot starts another motion: each finishes the one it plays and waits at the stand, and once
// every robot is due to cue, all of them go back to their home stands at once (a full reset, the
// sim clock kept), and their loops start over together. `rehoming` is set `rehome.fadeS` before
// that cut (and only while the cut is that close: a fall that puts it further off clears it), so
// the page can fade out and back in over it.
//
// Real time (`tick`): a fixed-step accumulator against wall time. It runs the control steps that
// are due, at most `maxCatchUp` per tick, with a callback after each; a backlog past that is
// dropped, so a slow machine runs the demo slower than real time instead of spiralling. Pause stops the clock; resume
// starts it again from the resume, without replaying the paused time.
//
// Plain module: the worker (worker.js) wraps it in the page, the headless check
// (tools/headless_live.mjs) runs it in Node.

import { NUM_JOINTS, PolicyInputs, Robot } from './obs.js';
import { LoopReference } from './reference.js';

/**
 * The fixed-step accumulator. `clock` = {t: ms the sim has caught up to, or null when it has
 * not started (or was paused)}. Returns the number of control steps to run now, at most
 * `maxSteps`, and advances clock.t by them; when more were due, the rest is dropped (clock.t =
 * now), so the demo slows down instead of spiralling.
 */
export function stepLoop(clock, now, { dtMs, maxSteps }) {
  if (clock.t === null) {
    clock.t = now;
    return 0;
  }
  const due = Math.floor((now - clock.t) / dtMs);
  if (due <= 0) return 0;
  if (due > maxSteps) {
    clock.t = now;
    clock.dropped = (clock.dropped ?? 0) + (due - maxSteps);
    return maxSteps;
  }
  clock.t += due * dtMs;
  return due;
}

/** RMS over the 29 canonical joints of q (native, through the robot's map) against `ref`. */
function jointRms(robot, q, ref, buf) {
  robot.toCanonical(q, buf);
  let s = 0;
  for (let i = 0; i < NUM_JOINTS; i++) s += (buf[i] - ref[i]) ** 2;
  return Math.sqrt(s / NUM_JOINTS);
}

export class LiveLoop {
  /**
   * sim: a LiveSim (or the same interface); policy: {act(robotIdx, inputs, iface) -> Promise of
   * Float32Array(29)}; specs: robots.json's `robots` (same order as the sim's robots);
   * clips: per robot, the starting clip's states (Float64Array [T * 240]); options: iface
   * ('motion', or 'full' / 'five': the human interface, which needs `human`, the starting clip's
   * human frames [T * 168], and `humanStand`, the uniform skeleton's stand [168]; with them every
   * robot's reference plays the human track too, whatever the interface), clipId (a label for
   * the starting clip, reported by status()),
   * loop ({leadInS, returnS, holdS}), fps, historyLen, maxCatchUp, rehome ({spread (m), fadeS},
   * or null for none). The robots' home stands are where the sim has them now.
   */
  constructor(sim, policy, specs, clips, {
    iface = 'motion', clipId = null, loop = {}, fps = 50, historyLen = 10, maxCatchUp = 4,
    rehome = { spread: 2.0, fadeS: 0.15 }, human = null, humanStand = null,
  } = {}) {
    if (specs.length !== sim.numRobots) throw new Error(`${specs.length} robot specs for ${sim.numRobots} robots`);
    this.sim = sim;
    this.policy = policy;
    this.hasHuman = human !== null;
    this.setInterface(iface);
    this.dtMs = 1000 / fps;
    this.maxCatchUp = maxCatchUp;
    this.clock = { t: null };
    this.paused = false;
    this.busy = false;
    this.steps = 0; // control steps run since the start
    this.rehomeSpread = rehome ? rehome.spread : Infinity;
    this.fadeSteps = rehome ? Math.round(rehome.fadeS * fps) : 0;
    this.rehomeArmed = false; // the spread passed the threshold: no new motions, re-home at the cue
    this.rehoming = false; // the cut comes within fadeSteps
    this.rehomes = 0;
    this.robots = specs.map((spec, r) => {
      if (sim.robots && sim.robots[r].name !== spec.name) throw new Error(`robot ${r}: sim ${sim.robots[r].name}, policy ${spec.name}`);
      const robot = new Robot(spec);
      const reference = new LoopReference(robot, clips[r], { ...loop, fps, human, humanStand });
      return {
        name: spec.name,
        robot,
        reference,
        inputs: new PolicyInputs(robot, reference, { historyLen }),
        stepId: 0,
        falls: 0,
        motions: 0, // motions (lead-in, clip, return) played to the end
        jointErr: NaN, // after the last control step
        meas: {},
        targets: new Float64Array(robot.numNative),
        canon: new Float64Array(NUM_JOINTS),
        lastPhase: 'stand',
        clip: clipId, // the clip of the motion playing (or last played)
        nextClip: clipId, // the clip the next cue plays
      };
    });
    this.home = this.robots.map((_, r) => {
      const { pos } = sim.slot0(r);
      return [pos[0], pos[1]];
    });
    for (let r = 0; r < this.robots.length; r++) this._fresh(r);
  }

  /** How far apart the robots have moved since their home stands (m): the largest distance
   * between two robots' displacements, slot-0 xy. */
  spread() {
    const d = this.home.map(([hx, hy], r) => {
      const { pos } = this.sim.slot0(r);
      return [pos[0] - hx, pos[1] - hy];
    });
    let m = 0;
    for (let i = 0; i < d.length; i++) {
      for (let j = i + 1; j < d.length; j++) m = Math.max(m, Math.hypot(d[i][0] - d[j][0], d[i][1] - d[j][1]));
    }
    return m;
  }

  /** After a control step: arm the re-home once the spread passes the threshold, and do it when
   * every robot waits at the stand, due to cue. */
  _checkRehome() {
    if (!this.rehomeArmed) {
      this.rehoming = false;
      if (!(this.spread() > this.rehomeSpread)) return;
      this.rehomeArmed = true;
      for (const R of this.robots) R.reference.cueHeld = true;
    }
    // Steps until the last robot is due to cue (each waits at the stand from then on).
    let wait = 0;
    for (const R of this.robots) {
      const ref = R.reference;
      wait = Math.max(wait, ref.playing ? Infinity : ref.nextCue - R.stepId);
    }
    // Not sticky: a fall that pushes a robot's cue back uncovers the stage until the cut is near
    // again.
    this.rehoming = wait <= this.fadeSteps;
    if (wait <= 0) this._rehome();
  }

  /** Every robot to its home stand, as restart() does, but the sim clock goes on. */
  _rehome() {
    const t = this.sim.time;
    this.restart();
    this.sim.time = t;
    this.rehomes++;
  }

  /** Robot `r`'s policy side from scratch: history, last action, reference (and its origin:
   * the slot-0 body's world x, y in the stand, where the reference's (0, 0) stands). */
  _fresh(r) {
    const R = this.robots[r];
    R.inputs.reset();
    R.inputs.prevStep = -1;
    R.reference.reset();
    const { pos } = this.sim.slot0(r);
    R.reference.origin = [pos[0], pos[1]];
    R.stepId = 0;
    R.lastPhase = 'stand';
    // The motion that was playing is gone (a fall or a restart): the robot stands, and its next
    // cue plays the clip last selected, so that is the clip it reports.
    R.clip = R.nextClip;
  }

  /** One 50 Hz control step for every robot. Resolves to the robots that fell (and were reset). */
  async controlStep() {
    const { sim, robots } = this;
    for (let r = 0; r < robots.length; r++) sim.measure(r, robots[r].meas);
    for (let r = 0; r < robots.length; r++) {
      const R = robots[r];
      const m = R.meas;
      const inputs = R.inputs.step(R.stepId, m.quat, m.gyro, m.q, m.dq, 0, this.iface);
      const action = await this.policy.act(r, inputs, this.iface);
      R.inputs.setAction(action);
      R.robot.targets(action, R.targets);
    }
    for (let r = 0; r < robots.length; r++) sim.setTargets(r, robots[r].targets);
    sim.step();
    this.steps++;
    const fell = [];
    for (let r = 0; r < robots.length; r++) {
      const R = robots[r];
      sim.measure(r, R.meas);
      R.jointErr = jointRms(R.robot, R.meas.q, R.inputs.lastWindow, R.canon);
      const phase = R.reference.phase;
      if (phase === 'stand' && R.lastPhase !== 'stand') R.motions++;
      if (phase !== 'stand' && R.lastPhase === 'stand') R.clip = R.nextClip; // a cue
      R.lastPhase = phase;
      if (sim.fallen(r)) {
        R.falls++;
        fell.push(r);
        sim.resetRobot(r);
        this._fresh(r);
      } else {
        R.stepId++;
      }
    }
    this._checkRehome();
    return fell;
  }

  /** Runs the control steps due at wall time `now` (ms), at most maxCatchUp, calling
   * `onStep()` after each one (the worker posts a frame per control step, so a tick that runs
   * several still draws every pose); resolves to how many ran. Ticks never overlap: one that
   * comes while a step is running does nothing. */
  async tick(now, onStep = null) {
    if (this.paused || this.busy) return 0;
    this.busy = true;
    try {
      const n = stepLoop(this.clock, now, { dtMs: this.dtMs, maxSteps: this.maxCatchUp });
      let ran = 0;
      for (; ran < n && !this.paused; ran++) {
        await this.controlStep();
        onStep?.();
      }
      return ran;
    } finally {
      this.busy = false;
    }
  }

  /** When the next control step is due (ms of the tick clock), or null when not running. */
  nextDue() {
    return this.paused || this.clock.t === null ? null : this.clock.t + this.dtMs;
  }

  pause() {
    this.paused = true;
    this.clock.t = null;
  }

  resume() {
    this.paused = false;
    this.clock.t = null; // the next tick starts the clock: paused time is not replayed
  }

  /** The clip every robot plays from its next cue on: `clip` {states: per robot [T * 240],
   * human: its human frames [T * 168], needed with a human track}; `id` labels it in status().
   * The motion playing goes on to the stand first. */
  setClip({ states, human = null }, id = null) {
    for (let r = 0; r < this.robots.length; r++) {
      this.robots[r].reference.setClip(states[r], human);
      this.robots[r].nextClip = id;
    }
  }

  /** 'motion', or the human interface with all 24 slots ('full') or five ('five'), from the next
   * control step on. */
  setInterface(iface) {
    if (iface !== 'motion' && iface !== 'full' && iface !== 'five') throw new Error(`interface must be motion, full or five, got ${iface}`);
    if (iface !== 'motion' && !this.hasHuman) throw new Error('the human interface needs the clips\' human frames');
    this.iface = iface;
  }

  /** Every robot back to its stand; each loop starts over (stand, first cue after holdS). */
  restart() {
    this.sim.resetAll();
    for (let r = 0; r < this.robots.length; r++) {
      this._fresh(r);
      this.robots[r].reference.cueHeld = false;
    }
    this.rehomeArmed = false;
    this.rehoming = false;
  }

  /** Robot `r`'s human targets of its last window, [K * 24 * 3] world positions of the 24 SMPL
   * slots (frame 0 the current target, then the preview), or null before its first step. */
  humanTargets(r, out) {
    return this.robots[r].reference.humanTargets(out);
  }

  /** Robot `r`'s five-point targets of its last window, [K * 5 * 3] world positions (frame 0
   * the current target, then the preview), or null before its first step. */
  fivePointTargets(r, out) {
    return this.robots[r].reference.fivePointTargets(out);
  }

  /** Per robot: phase, step, falls, motions played, the clip playing (`clip`) and the one the
   * next cue plays (`nextClip`), and the last joint error (rad). */
  status() {
    return this.robots.map((R) => ({
      name: R.name,
      phase: R.reference.phase,
      clip: R.clip,
      nextClip: R.nextClip,
      stepId: R.stepId,
      falls: R.falls,
      motions: R.motions,
      jointErr: R.jointErr,
    }));
  }
}
