// three.js drawing of the five-point overlay (src/keypoints.js holds the data side), as
// sim_node.draw_keypoints draws it: per robot and point, a sphere at the current target, a bright
// tube through the preview (the window's frames 0..K-1) and a fading tube through the trail of
// past targets. Everything is preallocated: one InstancedMesh of spheres and one of tube segments
// (unit cylinders placed between consecutive positions), with a per-instance opacity; unused
// segments get a zero matrix. A frame only rewrites instance matrices.

import * as THREE from 'three';

import { KP_COLORS, KP_POINTS, KP_TRAIL, trailAlpha } from './keypoints.js';

// sim_node.TARGET_RADIUS, PREVIEW_RADIUS, TRAIL_RADIUS; the preview's opacity.
const SPHERE_RADIUS = 0.04;
const PREVIEW_RADIUS = 0.016;
const TRAIL_RADIUS = 0.01;
const PREVIEW_ALPHA = 0.95;

// MeshBasicMaterial with an opacity per instance (attribute kpAlpha).
function instanceAlphaMaterial() {
  const mat = new THREE.MeshBasicMaterial({
    transparent: true, depthWrite: false, depthTest: false, toneMapped: false,
  });
  mat.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float kpAlpha;\nvarying float vKpAlpha;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvKpAlpha = kpAlpha;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vKpAlpha;')
      .replace('vec4 diffuseColor = vec4( diffuse, opacity );',
        'vec4 diffuseColor = vec4( diffuse, opacity * vKpAlpha );');
  };
  return mat;
}

export class KeypointOverlay {
  /** goalK: frames per window (the current target and its preview). */
  constructor(scene, nRobots, goalK) {
    this.nRobots = nRobots;
    this.goalK = goalK;
    this.trailSegs = KP_TRAIL - 1;
    this.segsPer = this.trailSegs + goalK - 1;
    const color = new THREE.Color();

    this.spheres = new THREE.InstancedMesh(
      new THREE.SphereGeometry(1, 16, 12),
      new THREE.MeshBasicMaterial({ depthTest: false, toneMapped: false }),
      nRobots * KP_POINTS,
    );
    // Unit cylinder along +y from y = 0 to y = 1, so a segment is scale (r, length, r) at its start.
    const cyl = new THREE.CylinderGeometry(1, 1, 1, 8, 1, true);
    cyl.translate(0, 0.5, 0);
    this.segs = new THREE.InstancedMesh(cyl, instanceAlphaMaterial(), nRobots * KP_POINTS * this.segsPer);
    this.alpha = new Float32Array(nRobots * KP_POINTS * this.segsPer);
    cyl.setAttribute('kpAlpha', new THREE.InstancedBufferAttribute(this.alpha, 1));
    this.alphaAttr = cyl.getAttribute('kpAlpha');
    this.alphaAttr.setUsage(THREE.DynamicDrawUsage);

    for (let ri = 0; ri < nRobots; ri++) {
      for (let p = 0; p < KP_POINTS; p++) {
        color.setRGB(...KP_COLORS[p], THREE.SRGBColorSpace);
        this.spheres.setColorAt(ri * KP_POINTS + p, color);
        for (let s = 0; s < this.segsPer; s++) this.segs.setColorAt((ri * KP_POINTS + p) * this.segsPer + s, color);
      }
    }
    for (const m of [this.spheres, this.segs]) {
      m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      m.frustumCulled = false; // instances move every frame; the base geometry's bounds mean nothing
      m.castShadow = false;
      m.receiveShadow = false;
      m.renderOrder = 2; // after the robots, so the dots and tubes draw over them
      m.visible = false;
      scene.add(m);
    }

    // Scratch, reused every frame.
    this._m = new THREE.Matrix4();
    this._a = new THREE.Vector3();
    this._b = new THREE.Vector3();
    this._d = new THREE.Vector3();
    this._q = new THREE.Quaternion();
    this._s = new THREE.Vector3();
    this._y = new THREE.Vector3(0, 1, 0);
    this._zero = new THREE.Matrix4().makeScale(0, 0, 0);
    for (let ri = 0; ri < nRobots; ri++) this.clearRobot(ri);
  }

  setVisible(on) {
    this.spheres.visible = on;
    this.segs.visible = on;
  }

  /** Hide one robot's points (no targets yet). */
  clearRobot(ri) {
    for (let p = 0; p < KP_POINTS; p++) {
      this.spheres.setMatrixAt(ri * KP_POINTS + p, this._zero);
      for (let s = 0; s < this.segsPer; s++) this.segs.setMatrixAt((ri * KP_POINTS + p) * this.segsPer + s, this._zero);
    }
  }

  _segment(i, a, ao, b, bo, radius, alpha) {
    this._a.set(a[ao], a[ao + 1], a[ao + 2]);
    this._d.set(b[bo] - a[ao], b[bo + 1] - a[ao + 1], b[bo + 2] - a[ao + 2]);
    const len = this._d.length();
    if (!(len > 1e-6)) {
      this.segs.setMatrixAt(i, this._zero);
      return;
    }
    this._q.setFromUnitVectors(this._y, this._d.multiplyScalar(1 / len));
    this._s.set(radius, len, radius);
    this.segs.setMatrixAt(i, this._m.compose(this._a, this._q, this._s));
    this.alpha[i] = alpha;
  }

  /**
   * Draw robot ri's trace.
   * @param win   Float64Array [goalK * 5 * 3]: the anchored window (frame 0 = current targets)
   * @param trail KpTrail of the anchored frame-0 targets drawn so far
   */
  update(ri, win, trail) {
    const K = this.goalK;
    const nTrail = Math.max(0, trail.length - 1);
    for (let p = 0; p < KP_POINTS; p++) {
      const base = (ri * KP_POINTS + p) * this.segsPer;
      this._b.set(win[3 * p], win[3 * p + 1], win[3 * p + 2]);
      this._s.setScalar(SPHERE_RADIUS);
      this.spheres.setMatrixAt(ri * KP_POINTS + p, this._m.compose(this._b, this._q.identity(), this._s));

      let s = 0;
      for (let i = 0; i < nTrail; i++, s++) {
        this._segment(base + s, trail.buf, trail.at(i, p), trail.buf, trail.at(i + 1, p),
          TRAIL_RADIUS, trailAlpha(i, nTrail));
      }
      for (; s < this.trailSegs; s++) this.segs.setMatrixAt(base + s, this._zero);
      for (let k = 1; k < K; k++, s++) {
        this._segment(base + s, win, ((k - 1) * KP_POINTS + p) * 3, win, (k * KP_POINTS + p) * 3,
          PREVIEW_RADIUS, PREVIEW_ALPHA);
      }
    }
  }

  /** Upload the matrices written since the last call. */
  commit() {
    this.spheres.instanceMatrix.needsUpdate = true;
    this.segs.instanceMatrix.needsUpdate = true;
    this.alphaAttr.needsUpdate = true;
  }
}
