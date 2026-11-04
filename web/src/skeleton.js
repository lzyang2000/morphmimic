// three.js drawing of the human interface's full setting: per robot, the 24 SMPL slots of the
// current human target (frame 0 of the window the policy tracks) as small spheres joined by
// bones to their SMPL parents, drawn over the robots. The window is anchored as the five-point
// overlay anchors its targets (keypoints.anchorTargets: frame-0 pelvis at the robot's pelvis
// x, y; the goals are position-free). Everything is preallocated; a frame only rewrites
// instance matrices.

import * as THREE from 'three';

const JOINT_RADIUS = 0.022;
const BONE_RADIUS = 0.009;
const COLOR = 0xff6c0c; // the page accent
const OPACITY = 0.85;

export class SkeletonOverlay {
  /** parents: per slot its SMPL parent (-1 for the root), robots.json `human.parents`. */
  constructor(scene, nRobots, parents) {
    this.parents = parents;
    this.nSlots = parents.length;
    this.bones = parents.flatMap((p, s) => (p >= 0 ? [[p, s]] : []));
    const mat = () => new THREE.MeshBasicMaterial({
      color: COLOR, transparent: true, opacity: OPACITY, depthTest: false, depthWrite: false, toneMapped: false,
    });
    this.joints = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 12, 8), mat(), nRobots * this.nSlots);
    // Unit cylinder along +y from y = 0 to y = 1, so a bone is scale (r, length, r) at its parent.
    const cyl = new THREE.CylinderGeometry(1, 1, 1, 8, 1, true);
    cyl.translate(0, 0.5, 0);
    this.segs = new THREE.InstancedMesh(cyl, mat(), nRobots * this.bones.length);
    for (const m of [this.joints, this.segs]) {
      m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      m.frustumCulled = false;
      m.renderOrder = 2;
      m.visible = false;
      scene.add(m);
    }
    this._m = new THREE.Matrix4();
    this._a = new THREE.Vector3();
    this._d = new THREE.Vector3();
    this._q = new THREE.Quaternion();
    this._s = new THREE.Vector3();
    this._y = new THREE.Vector3(0, 1, 0);
    this._zero = new THREE.Matrix4().makeScale(0, 0, 0);
    for (let r = 0; r < nRobots; r++) this.clearRobot(r);
  }

  setVisible(on) {
    this.joints.visible = on;
    this.segs.visible = on;
  }

  clearRobot(r) {
    for (let s = 0; s < this.nSlots; s++) this.joints.setMatrixAt(r * this.nSlots + s, this._zero);
    for (let b = 0; b < this.bones.length; b++) this.segs.setMatrixAt(r * this.bones.length + b, this._zero);
  }

  /** Robot r's skeleton from `win`, its anchored window [K * 24 * 3] (frame 0 is drawn). */
  update(r, win) {
    this._s.setScalar(JOINT_RADIUS);
    for (let s = 0; s < this.nSlots; s++) {
      this._a.set(win[3 * s], win[3 * s + 1], win[3 * s + 2]);
      this.joints.setMatrixAt(r * this.nSlots + s, this._m.compose(this._a, this._q.identity(), this._s));
    }
    this.bones.forEach(([p, c], b) => {
      const i = r * this.bones.length + b;
      this._a.set(win[3 * p], win[3 * p + 1], win[3 * p + 2]);
      this._d.set(win[3 * c] - win[3 * p], win[3 * c + 1] - win[3 * p + 1], win[3 * c + 2] - win[3 * p + 2]);
      const len = this._d.length();
      if (!(len > 1e-6)) {
        this.segs.setMatrixAt(i, this._zero);
        return;
      }
      this._q.setFromUnitVectors(this._y, this._d.multiplyScalar(1 / len));
      this._s.set(BONE_RADIUS, len, BONE_RADIUS);
      this.segs.setMatrixAt(i, this._m.compose(this._a, this._q, this._s));
      this._s.setScalar(JOINT_RADIUS);
    });
  }

  commit() {
    this.joints.instanceMatrix.needsUpdate = true;
    this.segs.instanceMatrix.needsUpdate = true;
  }
}
