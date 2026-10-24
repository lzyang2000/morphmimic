// The robots' side of the live demo: the three robots in one MuJoCo WASM model
// (public/live/scene.xml), simulated as deploy/sim2sim/sim_node.py simulates one.
//
// - Reset: the robot in its neutral stand (scene.json standQpos, which is sim_node.stand_qpos
//   plus its lateral offset), at rest, and its position actuators holding that pose
//   (ctrl = the stand's joint positions), then mj_forward. A full reset starts from
//   mj_resetData as SimRobot.reset does; a one-robot reset clears that robot's qpos, qvel and
//   warm start only, so the others run on untouched.
// - Control step: the native joint targets go straight to ctrl through actNative (the scenes'
//   built-in position PD; ctrl is not clipped to ctrlRange: the exported scenes have
//   ctrllimited off, and so does sim_node), then `decimation` mj_step calls and an mj_forward.
// - Measurements: the IMU body's world xquat, its angular velocity in its own frame
//   (mj_objectVelocity on XBODY, local), and the native joints' qpos and qvel.
// - Fall: the slot-0 body (pelvis-equivalent) below fall.height or tilted past fall.tiltDeg
//   (sim_node.fallen, deploy/vr_teleop/constants).
// - Push (the page's drag): a spring-damper pulls a point fixed on one body of one robot toward a
//   world target (pushForce), its magnitude capped at PUSH_MAX_N. Before every physics step it is
//   recomputed from the body's current pose and velocity and put in xfrc_applied as mj_applyFT
//   puts a force at a point: the force at the body's centre of mass and the torque
//   (point - com) x F. endPerturb, resetAll and a reset of the pushed robot clear it; a drag a
//   reset cut stays off until the page ends it (endPerturb), so the robot is not pushed again
//   while it stands back up.
//
// Works in Node and in a worker: the caller passes the loaded @mujoco/mujoco module, the XML
// text and the parsed scene.json.

import { quatApplyInverse } from './obs.js';

const UP = Float64Array.of(0, 0, 1);

// The push: stiffness (N/m), damping (N s/m) and the hard cap (N). Firm but not instant: the cap
// is reached 0.4 m from the grab point, and nothing larger ever reaches the robot.
const PUSH_K = 250;
const PUSH_C = 20;
export const PUSH_MAX_N = 100;

/** The push on a point at `p` moving at `v` toward `target` (world): k (target - p) - c v, its
 * magnitude capped at PUSH_MAX_N. Into `out` [3]. */
export function pushForce(p, v, target, out = new Float64Array(3)) {
  let n2 = 0;
  for (let k = 0; k < 3; k++) {
    out[k] = PUSH_K * (target[k] - p[k]) - PUSH_C * v[k];
    n2 += out[k] * out[k];
  }
  const n = Math.sqrt(n2);
  if (!(n <= PUSH_MAX_N)) {
    const s = n > 0 && Number.isFinite(n) ? PUSH_MAX_N / n : 0;
    for (let k = 0; k < 3; k++) out[k] *= s;
  }
  return out;
}

const vec3 = (a, what) => {
  if (!Array.isArray(a) && !ArrayBuffer.isView(a)) throw new Error(`${what} must be [x, y, z]`);
  if (a.length !== 3 || !Array.prototype.every.call(a, Number.isFinite)) throw new Error(`${what} must be three finite numbers`);
  return Float64Array.from(a);
};

export class LiveSim {
  constructor(mujoco, xmlText, scene, { dir = '/live' } = {}) {
    this.mj = mujoco;
    this.scene = scene;
    mujoco.FS.mkdirTree(dir);
    mujoco.FS.writeFile(`${dir}/scene.xml`, xmlText);
    this.model = mujoco.MjModel.mj_loadXML(`${dir}/scene.xml`);
    this.data = new mujoco.MjData(this.model);
    const m = this.model;
    if (Math.abs(m.opt.timestep - scene.timestep) > 1e-12) {
      throw new Error(`scene.xml timestep ${m.opt.timestep}, scene.json ${scene.timestep}`);
    }
    if (m.nq !== scene.nq || m.nv !== scene.nv || m.nu !== scene.nu) {
      throw new Error(`scene.xml sizes (${m.nq}, ${m.nv}, ${m.nu}) differ from scene.json`);
    }
    this.decimation = scene.decimation;
    this.controlDt = scene.controlDt;
    this.robots = scene.robots.map((r) => ({
      ...r,
      jointQposadr: Int32Array.from(r.jointQposadr),
      jointDofadr: Int32Array.from(r.jointDofadr),
      actNative: Int32Array.from(r.actNative),
      standQpos: Float64Array.from(r.standQpos),
      slot0: r.trackedBodyIds[0],
    }));
    this.fallHeight = scene.fall.height;
    this.fallUpZ = Math.cos((scene.fall.tiltDeg * Math.PI) / 180);
    this.nbody = m.nbody;
    this.bodyNames = Array.from({ length: m.nbody }, (_, i) => mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY.value, i) ?? '');
    this._vel = new mujoco.DoubleBuffer(6);
    this._xbody = mujoco.mjtObj.mjOBJ_XBODY.value;
    // Each body's robot (-1 for the world): a push may only grab the robot's own bodies.
    this.robotOfBody = Int32Array.from(this.bodyNames, (n) => this.robots.findIndex((R) => n.startsWith(`${R.name}/`)));
    this.push = null; // {robot, body, local, target} while a drag pushes
    this.pushCut = false; // a reset ended the drag: ignore it until the page ends it
    this._p = new Float64Array(3);
    this._pv = new Float64Array(3);
    this._f = new Float64Array(3);
    this.time = 0;
    this.resetAll();
  }

  get numRobots() { return this.robots.length; }

  _place(r) {
    const R = this.robots[r];
    const d = this.data;
    const qpos = d.qpos, qvel = d.qvel, warm = d.qacc_warmstart;
    qpos.set(R.standQpos, R.qposadr);
    qvel.fill(0, R.qveladr, R.qveladr + R.nv);
    warm.fill(0, R.qveladr, R.qveladr + R.nv);
  }

  /** ctrl of robot `r` = its joints' current positions (SimRobot.reset's set_targets(qpos)). */
  _holdPose(r) {
    const R = this.robots[r];
    const qpos = this.data.qpos, ctrl = this.data.ctrl;
    for (let a = 0; a < R.nu; a++) ctrl[R.ctrladr + a] = qpos[R.jointQposadr[R.actNative[a]]];
  }

  /** Every robot to its stand: mj_resetData, the stands, mj_forward, ctrl holding them. Any
   * push is cleared. */
  resetAll() {
    this._cutPush(null);
    this.mj.mj_resetData(this.model, this.data);
    for (let r = 0; r < this.robots.length; r++) this._place(r);
    this.mj.mj_forward(this.model, this.data);
    for (let r = 0; r < this.robots.length; r++) this._holdPose(r);
    this.time = 0;
  }

  /** Robot `r` alone back to its stand; the others are not touched. A push on it is cleared. */
  resetRobot(r) {
    this._cutPush(r);
    this._place(r);
    this.mj.mj_forward(this.model, this.data);
    this._holdPose(r);
  }

  /** Robot `r`'s measurements as the robot reports them: {quat, gyro, q, dq} (native order).
   * Fresh arrays, or `out`'s. */
  measure(r, out = {}) {
    const R = this.robots[r];
    const d = this.data;
    const J = R.jointQposadr.length;
    const quat = (out.quat ??= new Float64Array(4));
    const gyro = (out.gyro ??= new Float64Array(3));
    const q = (out.q ??= new Float64Array(J));
    const dq = (out.dq ??= new Float64Array(J));
    const xquat = d.xquat, qpos = d.qpos, qvel = d.qvel;
    for (let k = 0; k < 4; k++) quat[k] = xquat[4 * R.imuBodyId + k];
    this.mj.mj_objectVelocity(this.model, d, this._xbody, R.imuBodyId, this._vel, 1);
    const v = this._vel.GetView();
    for (let k = 0; k < 3; k++) gyro[k] = v[k];
    for (let j = 0; j < J; j++) {
      q[j] = qpos[R.jointQposadr[j]];
      dq[j] = qvel[R.jointDofadr[j]];
    }
    return out;
  }

  /** Robot `r`'s native joint targets [J] to its position actuators (not clipped). */
  setTargets(r, targets) {
    const R = this.robots[r];
    const ctrl = this.data.ctrl;
    for (let a = 0; a < R.nu; a++) ctrl[R.ctrladr + a] = targets[R.actNative[a]];
  }

  /** Push robot `r` from now on: a spring from the point `local` (in the frame of MuJoCo body
   * `body`, one of the robot's) toward the world point `target`, capped (pushForce). Replaces the
   * push before. Ignored after a reset cut the drag, until endPerturb. */
  setPerturb(r, body, local, target) {
    if (!Number.isInteger(r) || r < 0 || r >= this.robots.length) throw new Error(`no robot ${JSON.stringify(r)}`);
    if (!Number.isInteger(body) || this.robotOfBody[body] !== r) {
      throw new Error(`${JSON.stringify(body)} is not a body of ${this.robots[r].name}`);
    }
    const l = vec3(local, 'local');
    const t = vec3(target, 'target');
    if (this.pushCut) return;
    if (this.push && this.push.body !== body) this._clearXfrc(this.push.body);
    this.push = { robot: r, body, local: l, target: t };
  }

  /** The drag ended: no more push. */
  endPerturb() {
    if (this.push) this._clearXfrc(this.push.body);
    this.push = null;
    this.pushCut = false;
  }

  get perturbing() {
    return this.push !== null;
  }

  /** A reset of robot `r` (null: every robot) ends a push on it, until the page ends the drag. */
  _cutPush(r) {
    if (!this.push || (r !== null && this.push.robot !== r)) return;
    this._clearXfrc(this.push.body);
    this.push = null;
    this.pushCut = true;
  }

  _clearXfrc(body) {
    this.data.xfrc_applied.fill(0, 6 * body, 6 * body + 6);
  }

  /** The push for the coming physics step, from the body's pose and velocity now. */
  _applyPush() {
    const { body, local, target } = this.push;
    const d = this.data;
    const xpos = d.xpos, R = d.xmat, xipos = d.xipos;
    const p = this._p, pv = this._pv, F = this._f;
    for (let i = 0; i < 3; i++) {
      p[i] = xpos[3 * body + i] + R[9 * body + 3 * i] * local[0] + R[9 * body + 3 * i + 1] * local[1]
        + R[9 * body + 3 * i + 2] * local[2];
    }
    // The body frame's velocity (world orientation, at xpos: [angular, linear]), moved to the point.
    this.mj.mj_objectVelocity(this.model, d, this._xbody, body, this._vel, 0);
    const v = this._vel.GetView();
    const rx = p[0] - xpos[3 * body], ry = p[1] - xpos[3 * body + 1], rz = p[2] - xpos[3 * body + 2];
    pv[0] = v[3] + v[1] * rz - v[2] * ry;
    pv[1] = v[4] + v[2] * rx - v[0] * rz;
    pv[2] = v[5] + v[0] * ry - v[1] * rx;
    pushForce(p, pv, target, F);
    const ax = p[0] - xipos[3 * body], ay = p[1] - xipos[3 * body + 1], az = p[2] - xipos[3 * body + 2];
    const x = d.xfrc_applied;
    const o = 6 * body;
    x[o] = F[0];
    x[o + 1] = F[1];
    x[o + 2] = F[2];
    x[o + 3] = ay * F[2] - az * F[1];
    x[o + 4] = az * F[0] - ax * F[2];
    x[o + 5] = ax * F[1] - ay * F[0];
  }

  /** One control step of physics for every robot: `decimation` mj_step (each with the push, if
   * any, recomputed first), then mj_forward. */
  step() {
    for (let i = 0; i < this.decimation; i++) {
      if (this.push) this._applyPush();
      this.mj.mj_step(this.model, this.data);
    }
    this.mj.mj_forward(this.model, this.data);
    this.time += this.controlDt;
  }

  /** Robot `r`'s slot-0 body (pelvis-equivalent): world position and orientation (copies). */
  slot0(r) {
    const id = this.robots[r].slot0;
    const xpos = this.data.xpos, xquat = this.data.xquat;
    return {
      pos: Float64Array.of(xpos[3 * id], xpos[3 * id + 1], xpos[3 * id + 2]),
      quat: Float64Array.of(xquat[4 * id], xquat[4 * id + 1], xquat[4 * id + 2], xquat[4 * id + 3]),
    };
  }

  /** sim_node.fallen: the slot-0 body below the fall height or tilted past the fall angle. */
  fallen(r) {
    const { pos, quat } = this.slot0(r);
    return pos[2] < this.fallHeight || quatApplyInverse(quat, UP)[2] < this.fallUpZ;
  }

  /** Every body's world pose, [nbody * 7] (x, y, z, qw, qx, qy, qz), body id order
   * (bodyNames). */
  bodyPoses(out = new Float64Array(this.nbody * 7)) {
    const xpos = this.data.xpos, xquat = this.data.xquat;
    for (let b = 0; b < this.nbody; b++) {
      out[7 * b] = xpos[3 * b];
      out[7 * b + 1] = xpos[3 * b + 1];
      out[7 * b + 2] = xpos[3 * b + 2];
      for (let k = 0; k < 4; k++) out[7 * b + 3 + k] = xquat[4 * b + k];
    }
    return out;
  }

  dispose() {
    this._vel.delete();
    this.data.delete();
    this.model.delete();
  }
}
