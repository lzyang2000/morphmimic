// three.js view of the three robots.
//
// MuJoCo owns the transforms (the live simulation in src/live/worker.js posts every body's world
// pose); this module only draws them. Each MuJoCo body gets a THREE.Group whose pose is copied
// from the latest posted frame, and the visual meshes hang off those groups at the geom's pose in
// its body frame (visuals.json). The scene is z-up like MuJoCo: camera.up is +z, so no axis
// conversion is needed. The camera follows the group's centre from the +x side and pulls back
// when the robots spread out, until the visitor orbits or zooms.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const BG = 0xeef1f5;
const FLOOR = 0xdfe4ea;
// Group centre the camera looks at, and the half extents it frames (robots at y = -1.5, 0, +1.5).
const TARGET_Z = 0.85;
const FRAME_HALF_WIDTH = 2.1;
const FRAME_HALF_HEIGHT = 1.05;
const FRAME_MARGIN = 0.6; // metres kept around the robot furthest from the centre
const FOLLOW_TAU = 0.35; // s: time constant of the camera's easing toward the group
// Default view direction from the target: from +x (in front of the robots), slightly raised.
const VIEW_DIR = new THREE.Vector3(1, 0, 0.3).normalize();
const LABEL_CLEARANCE = 0.14; // metres above the robot's highest point at boot
const DRAG_COLOR = 0xff6c0c; // the drag line and its ends (the accent)

export function webglAvailable() {
  try {
    const c = document.createElement('canvas');
    return Boolean(c.getContext('webgl2') || c.getContext('webgl'));
  } catch {
    return false;
  }
}

export class Renderer {
  /**
   * @param canvas      the <canvas> to draw into (sized by CSS)
   * @param labelLayer  absolutely positioned element over the canvas for the robot labels
   * @param sceneMeta   scene.json: {robots: [{name, label}], bodies: [names]}
   */
  constructor(canvas, labelLayer, sceneMeta) {
    this.canvas = canvas;
    this.labelLayer = labelLayer;
    this.meta = sceneMeta;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(BG);
    this.scene.fog = new THREE.Fog(BG, 12, 32);

    // Metallic materials need something to reflect or they render near black.
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.5;
    this.scene.environmentRotation.set(Math.PI / 2, 0, 0); // the room is y-up; the world is z-up
    pmrem.dispose();

    this.camera = new THREE.PerspectiveCamera(35, 1, 0.05, 200);
    this.camera.up.set(0, 0, 1); // before OrbitControls, which reads it once
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 1.0;
    this.controls.maxDistance = 20;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.03; // stay above the floor
    this.controls.target.set(0, 0, TARGET_Z);
    this.userMoved = false;
    this.controls.addEventListener('start', () => { this.userMoved = true; });
    // OrbitControls sets touch-action: none inline. On touch screens let one-finger vertical drags
    // scroll the page instead, so a visitor can scroll past the demo (see style.css).
    if (typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches) {
      canvas.style.touchAction = 'pan-y';
    }

    this._addLights();
    this._addFloor();

    // One group per robot (so it can be hidden), holding one group per body.
    this.robotGroups = sceneMeta.robots.map((r) => {
      const g = new THREE.Group();
      g.name = r.name;
      this.scene.add(g);
      return g;
    });
    this.robotOfBody = sceneMeta.bodies.map((name) => {
      const ri = sceneMeta.robots.findIndex((r) => name.startsWith(`${r.name}/`));
      if (ri < 0) throw new Error(`body ${name} belongs to no robot`);
      return ri;
    });
    this.bodyGroups = sceneMeta.bodies.map((name, b) => {
      const g = new THREE.Group();
      g.name = name;
      this.robotGroups[this.robotOfBody[b]].add(g);
      return g;
    });
    // Root body of each robot: the first of its bodies in MuJoCo order.
    this.rootBody = sceneMeta.robots.map((_, ri) => this.robotOfBody.indexOf(ri));

    this.labels = sceneMeta.robots.map(() => {
      const el = document.createElement('div');
      el.className = 'robot-label';
      el.innerHTML = '<span class="name"></span><span class="note" hidden></span>';
      labelLayer.appendChild(el);
      return el;
    });
    this.labelHeight = null; // per robot, metres above the root body; measured on the first frame
    this.follow = new THREE.Vector3(0, 0, TARGET_Z);
    // The group's extent seen from the default view, eased: the half width to frame across the
    // view and how far the nearest robot stands in front of the centre (toward the camera).
    this.halfWidth = FRAME_HALF_WIDTH;
    this.nearDepth = 0;
    this._snap = false;
    this._lastDraw = null;
    this._v = new THREE.Vector3();
    this._delta = new THREE.Vector3();

    this.raycaster = new THREE.Raycaster();
    this._ndc = new THREE.Vector2();
    this._addDragLine();

    this.camera.position.copy(this.controls.target).addScaledVector(VIEW_DIR, 6);
    this._resize();
    new ResizeObserver(() => this._resize()).observe(canvas);
  }

  _addLights() {
    const hemi = new THREE.HemisphereLight(0xffffff, 0xaab4c0, 1.4);
    hemi.position.set(0, 0, 1);
    this.scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    const d = 4.5;
    Object.assign(key.shadow.camera, { left: -d, right: d, top: d, bottom: -d, near: 0.5, far: 30 });
    key.shadow.camera.updateProjectionMatrix();
    key.shadow.bias = -0.0008;
    key.shadow.normalBias = 0.02;
    this.scene.add(key, key.target);
    this.keyLight = key;

    // Cool light from behind so the silhouettes separate from the backdrop.
    const rim = new THREE.DirectionalLight(0xc9d8ff, 0.8);
    rim.position.set(-5, 2, 4);
    this.scene.add(rim);
    this._placeKeyLight(0, 0);
  }

  _placeKeyLight(x, y) {
    this.keyLight.position.set(x + 5, y - 3, 8);
    this.keyLight.target.position.set(x, y, 0);
    this.keyLight.target.updateMatrixWorld();
  }

  _addFloor() {
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(120, 120), // in the xy plane, normal +z
      new THREE.MeshStandardMaterial({ color: FLOOR, roughness: 0.95, metalness: 0 }),
    );
    floor.receiveShadow = true;
    this.scene.add(floor);
    this.floor = floor;
    const grid = new THREE.GridHelper(30, 30, 0xbcc5cf, 0xd0d7df);
    grid.rotation.x = Math.PI / 2; // GridHelper lies in xz; turn it into the floor plane
    grid.position.z = 0.002;
    this.scene.add(grid);
    this.grid = grid;
  }

  // The drag (src/drag.js): a thin line from the grabbed point on the body to the cursor's
  // target, drawn over everything, with a dot at each end.
  _addDragLine() {
    const mat = new THREE.LineBasicMaterial({ color: DRAG_COLOR, depthTest: false, transparent: true, opacity: 0.9 });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    const line = new THREE.Line(geo, mat);
    line.frustumCulled = false;
    line.renderOrder = 10;
    const dotMat = new THREE.MeshBasicMaterial({ color: DRAG_COLOR, depthTest: false, transparent: true });
    const dotGeo = new THREE.SphereGeometry(0.018, 12, 8);
    this.dragEnds = [new THREE.Mesh(dotGeo, dotMat), new THREE.Mesh(dotGeo, dotMat)];
    this.dragGroup = new THREE.Group();
    this.dragGroup.add(line, ...this.dragEnds);
    for (const d of this.dragEnds) d.renderOrder = 11;
    this.dragGroup.visible = false;
    this.scene.add(this.dragGroup);
    this.dragLine = line;
    this.drag = null; // {body, local: Vector3, target: Vector3}
  }

  /** Point the raycaster through the canvas pixel at client (x, y). */
  _ray(x, y) {
    const r = this.canvas.getBoundingClientRect();
    this._ndc.set(((x - r.left) / r.width) * 2 - 1, 1 - ((y - r.top) / r.height) * 2);
    this.raycaster.setFromCamera(this._ndc, this.camera);
    return this.raycaster.ray;
  }

  /** The robot under client (x, y): {robot, body (index into sceneMeta.bodies), point (world),
   * local (in the body's frame)}, or null. */
  pick(x, y) {
    this._ray(x, y);
    this.scene.updateMatrixWorld();
    const visible = this.robotGroups.filter((g) => g.visible);
    const hit = this.raycaster.intersectObjects(visible, true)[0];
    if (!hit) return null;
    const b = this.bodyGroups.indexOf(hit.object.parent);
    if (b < 0) return null;
    const local = this.bodyGroups[b].worldToLocal(hit.point.clone());
    return { robot: this.robotOfBody[b], body: b, point: hit.point.clone(), local };
  }

  /** Where the ray through client (x, y) meets `plane`, into `out`; null when it does not. */
  planePoint(x, y, plane, out) {
    return this._ray(x, y).intersectPlane(plane, out);
  }

  /** The drag to draw: a line from `local` on body `body` to `target` (world), or null for none. */
  setDrag(drag) {
    this.drag = drag;
    this.dragGroup.visible = Boolean(drag);
  }

  /** World position of the point `local` on body `body` (index into sceneMeta.bodies). */
  bodyPoint(body, local, out = new THREE.Vector3()) {
    const g = this.bodyGroups[body];
    g.updateWorldMatrix(true, false);
    return out.copy(local).applyMatrix4(g.matrixWorld);
  }

  _placeDrag() {
    if (!this.drag) return;
    const [a, b] = this.dragEnds;
    this.bodyPoint(this.drag.body, this.drag.local, a.position);
    b.position.copy(this.drag.target);
    const pos = this.dragLine.geometry.attributes.position;
    pos.setXYZ(0, a.position.x, a.position.y, a.position.z);
    pos.setXYZ(1, b.position.x, b.position.y, b.position.z);
    pos.needsUpdate = true;
  }

  /** Client (x, y) of a world point, for the headless checks. */
  toClient(p) {
    const r = this.canvas.getBoundingClientRect();
    this._v.copy(p).project(this.camera);
    return [r.left + ((this._v.x + 1) / 2) * r.width, r.top + ((1 - this._v.y) / 2) * r.height];
  }

  /**
   * Build one mesh per visuals.json entry.
   * @param meshIndex meshes.json entries {name, vertOffset, vertCount, faceOffset, faceCount}
   * @param buffer    meshes.bin: all vertices (float32 xyz), then all faces (uint32 triples)
   * @param visuals   visuals.json entries {body, mesh, pos, quat (wxyz), rgba}
   * @returns number of meshes attached
   */
  addMeshes(meshIndex, buffer, visuals) {
    const totalVerts = meshIndex.reduce((n, m) => Math.max(n, m.vertOffset + m.vertCount), 0);
    const totalFaces = meshIndex.reduce((n, m) => Math.max(n, m.faceOffset + m.faceCount), 0);
    if (buffer.byteLength !== 12 * totalVerts + 12 * totalFaces) {
      throw new Error(`meshes.bin is ${buffer.byteLength} bytes, index expects ${12 * (totalVerts + totalFaces)}`);
    }
    const verts = new Float32Array(buffer, 0, totalVerts * 3);
    const faces = new Uint32Array(buffer, 12 * totalVerts, totalFaces * 3);

    const geometries = new Map();
    for (const m of meshIndex) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(
        verts.subarray(m.vertOffset * 3, (m.vertOffset + m.vertCount) * 3), 3));
      geo.setIndex(new THREE.BufferAttribute(
        faces.subarray(m.faceOffset * 3, (m.faceOffset + m.faceCount) * 3), 1));
      geo.computeVertexNormals();
      geometries.set(m.name, geo);
    }

    const bodyIdx = new Map(this.meta.bodies.map((name, b) => [name, b]));
    const materials = new Map();
    let attached = 0;
    for (const v of visuals) {
      const geo = geometries.get(v.mesh);
      const b = bodyIdx.get(v.body);
      if (!geo || b === undefined) {
        console.warn(`skipping visual ${v.mesh} on ${v.body}: unknown mesh or body`);
        continue;
      }
      const key = v.rgba.join(',');
      if (!materials.has(key)) {
        // Lift the darkest trim a little so it does not crush to flat black on a light backdrop.
        const c = v.rgba.slice(0, 3).map((x) => (x < 0.3 ? 0.3 : x));
        materials.set(key, new THREE.MeshStandardMaterial({
          color: new THREE.Color(c[0], c[1], c[2]).convertSRGBToLinear(),
          roughness: 0.5,
          metalness: 0.3,
          // Vertex clustering folds some triangles over, so winding is not consistent: draw both
          // sides and shade from screen-space face normals, which always face the viewer.
          side: THREE.DoubleSide,
          flatShading: true,
        }));
      }
      const mesh = new THREE.Mesh(geo, materials.get(key));
      mesh.castShadow = true;
      mesh.receiveShadow = false; // self-shadowing on the decimated meshes reads as speckle
      mesh.position.set(v.pos[0], v.pos[1], v.pos[2]);
      mesh.quaternion.set(v.quat[1], v.quat[2], v.quat[3], v.quat[0]); // MuJoCo wxyz -> three xyzw
      this.bodyGroups[b].add(mesh);
      attached++;
    }
    return attached;
  }

  /** Label text per robot; note (e.g. "motion") marks a robot shown with another interface. */
  setLabel(robotIdx, name, note = '') {
    const el = this.labels[robotIdx];
    const key = `${name}\n${note}`;
    if (el.dataset.key === key) return;
    el.dataset.key = key;
    el.querySelector('.name').textContent = name;
    const n = el.querySelector('.note');
    n.textContent = note;
    n.hidden = !note;
  }

  setRobotVisible(robotIdx, visible) {
    this.robotGroups[robotIdx].visible = visible;
    this.labels[robotIdx].hidden = !visible;
  }

  /**
   * Copy every body pose from a posted frame.
   * @param poses  Float64Array [nbody * 7]: x, y, z, qw, qx, qy, qz per body id of the sender
   * @param ids    Int32Array: per body of sceneMeta.bodies, its id in `poses`
   */
  setPoses(poses, ids) {
    for (let b = 0; b < this.bodyGroups.length; b++) {
      const o = 7 * ids[b];
      const g = this.bodyGroups[b];
      g.position.set(poses[o], poses[o + 1], poses[o + 2]);
      g.quaternion.set(poses[o + 4], poses[o + 5], poses[o + 6], poses[o + 3]);
    }
    if (!this.labelHeight) this._measureLabels();
  }

  _measureLabels() {
    const box = new THREE.Box3();
    this.labelHeight = this.robotGroups.map((g, ri) => {
      g.updateMatrixWorld(true);
      box.setFromObject(g);
      const root = this.bodyGroups[this.rootBody[ri]].position.z;
      return (box.isEmpty() ? 1.2 : box.max.z - root) + LABEL_CLEARANCE;
    });
  }

  /** Root (pelvis) position of a robot, from the last update(). */
  rootPos(robotIdx) {
    return this.bodyGroups[this.rootBody[robotIdx]].position;
  }

  /** Ease the camera target (and the shadow light) toward the centre of the visible robots,
   * by wall time, so a slow frame rate does not let the robots walk out of view. */
  _follow() {
    const now = performance.now();
    const dt = this._lastDraw === null ? 0 : Math.min(1, (now - this._lastDraw) / 1000);
    this._lastDraw = now;
    let n = 0;
    let x = 0;
    let y = 0;
    this.robotGroups.forEach((g, ri) => {
      if (!g.visible) return;
      const p = this.rootPos(ri);
      x += p.x;
      y += p.y;
      n++;
    });
    if (!n) return;
    const cx = x / n;
    const cy = y / n;
    // Robots wander as they loop: the one furthest across the view sets the width to frame and
    // the one nearest the camera how much further back the camera goes.
    const hx = VIEW_DIR.x / Math.hypot(VIEW_DIR.x, VIEW_DIR.y);
    const hy = VIEW_DIR.y / Math.hypot(VIEW_DIR.x, VIEW_DIR.y);
    let across = 0;
    let near = 0;
    this.robotGroups.forEach((g, ri) => {
      if (!g.visible) return;
      const p = this.rootPos(ri);
      const dx = p.x - cx;
      const dy = p.y - cy;
      across = Math.max(across, Math.abs(dx * hy - dy * hx));
      near = Math.max(near, dx * hx + dy * hy);
    });
    const k = this._snap ? 1 : 1 - Math.exp(-dt / FOLLOW_TAU);
    this.halfWidth += (Math.max(FRAME_HALF_WIDTH, across + FRAME_MARGIN) - this.halfWidth) * k;
    this.nearDepth += (near - this.nearDepth) * k;
    this._v.set(cx, cy, TARGET_Z);
    this.follow.lerp(this._v, k);
    this._snap = false;
    this._delta.copy(this.follow).sub(this.controls.target);
    this.controls.target.add(this._delta);
    this.camera.position.add(this._delta);
    if (!this.userMoved) {
      this.camera.position.copy(this.controls.target).addScaledVector(VIEW_DIR, this._fitDistance());
    }
    this._placeKeyLight(this.follow.x, this.follow.y);
    // The floor and its grid come along in whole metres, so wandering robots never leave them.
    const gx = Math.round(this.follow.x);
    const gy = Math.round(this.follow.y);
    this.floor.position.set(gx, gy, 0);
    this.grid.position.set(gx, gy, 0.002);
  }

  /** Back to the default view: from +x, framing the group (after a restart, say). */
  resetView() {
    this.userMoved = false;
    this._snap = true;
  }

  /** Jump (rather than ease) the camera target onto the robots at the next render. */
  recentre() {
    this._snap = true;
  }

  _placeLabels() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    this.labels.forEach((el, ri) => {
      if (el.hidden || !this.labelHeight) return;
      const p = this.rootPos(ri);
      this._v.set(p.x, p.y, p.z + this.labelHeight[ri]).project(this.camera);
      const visible = this._v.z < 1 && Math.abs(this._v.x) < 1.2 && Math.abs(this._v.y) < 1.2;
      el.style.visibility = visible ? 'visible' : 'hidden';
      if (!visible) return;
      const sx = ((this._v.x + 1) / 2) * w;
      const sy = ((1 - this._v.y) / 2) * h;
      el.style.transform = `translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px) translate(-50%, -100%)`;
    });
  }

  render() {
    this._follow();
    this.controls.update();
    this._placeDrag();
    this.renderer.render(this.scene, this.camera);
    this._placeLabels();
  }

  /** Distance that frames the three robots for the current aspect ratio. */
  _fitDistance() {
    const tanV = Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2);
    const tanH = tanV * this.camera.aspect;
    return Math.max(FRAME_HALF_HEIGHT / tanV, this.halfWidth / tanH) + this.nearDepth + 0.6;
  }

  _resize() {
    const w = this.canvas.clientWidth || 1;
    const h = this.canvas.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (!this.userMoved) {
      this.camera.position.copy(this.controls.target).addScaledVector(VIEW_DIR, this._fitDistance());
    }
  }
}
