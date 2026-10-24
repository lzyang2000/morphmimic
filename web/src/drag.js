// Drag to push: press on a robot (mouse, pen or one finger) and drag. The grabbed point is pulled
// toward the cursor by a capped spring in the simulation worker (src/live/sim.js), so the robot
// has to catch itself. This side only picks the body, keeps the cursor's target and draws the line.
//
// - Press: a raycast against the drawn robots gives the robot, the MuJoCo body and the grab point
//   in that body's frame. Orbit controls do not see the press and are off until the drag ends.
//   A press anywhere else, or while another pointer orbits, leaves orbit, zoom and touch scrolling
//   as they were.
// - Move: the target is where the cursor's ray meets the plane through the grab point that faces
//   the camera (as the drag started). It is recomputed every frame, so it stays under the cursor
//   while the camera follows the robots.
// - Release, cancel or lost capture ends the drag; so does `cancel()` (a fall, re-home or restart).

import * as THREE from 'three';

const MIN_MOVE = 1e-3; // m: smaller target changes are not sent

export class DragPush {
  /**
   * renderer: src/render.js Renderer. stage: the element holding the canvas.
   * send(msg): to the worker (only called while `enabled`). simBody(b): the worker's body id of
   * the drawn body b. onStart(): a drag began.
   */
  constructor(renderer, stage, { send, simBody, onStart = () => {} }) {
    this.r = renderer;
    this.canvas = renderer.canvas;
    this.send = send;
    this.simBody = simBody;
    this.onStart = onStart;
    this.enabled = false;
    this.drag = null; // {pointerId, robot, body, simBody, local, plane, x, y, target, sent}
    this.others = new Set(); // pointers down that are not the drag (orbit and zoom gestures)
    this._normal = new THREE.Vector3();
    this._t = new THREE.Vector3();
    this._hover = null; // client (x, y) of the mouse to test for the grab cursor

    stage.addEventListener('pointerdown', (e) => this._down(e), { capture: true });
    // Pointer events reach the page before the touch events: a touch that grabbed a robot must not
    // also scroll the page (touch-action is pan-y on touch screens, render.js).
    const noScroll = (e) => { if (this.drag) e.preventDefault(); };
    stage.addEventListener('touchstart', noScroll, { passive: false });
    stage.addEventListener('touchmove', noScroll, { passive: false });
    const c = this.canvas;
    c.addEventListener('pointermove', (e) => this._move(e));
    for (const t of ['pointerup', 'pointercancel']) {
      addEventListener(t, (e) => this.others.delete(e.pointerId), { capture: true });
    }
    for (const t of ['pointerup', 'pointercancel', 'lostpointercapture']) {
      c.addEventListener(t, (e) => {
        if (this.drag && e.pointerId === this.drag.pointerId) this.end();
      });
    }
    c.addEventListener('pointerleave', (e) => {
      if (e.pointerType === 'mouse') this._setCursor('');
      if (this.drag && e.pointerId === this.drag.pointerId && !c.hasPointerCapture?.(e.pointerId)) this.end();
    });
  }

  get active() {
    return this.drag !== null;
  }

  _down(e) {
    if (this.drag) {
      // Another finger while dragging: neither a second drag nor an orbit.
      e.stopPropagation();
      e.preventDefault();
      return;
    }
    const grab = this.enabled && e.target === this.canvas && e.isPrimary && this.others.size === 0
      && (e.pointerType !== 'mouse' || e.button === 0);
    const hit = grab ? this.r.pick(e.clientX, e.clientY) : null;
    if (!hit) {
      if (e.target === this.canvas) this.others.add(e.pointerId);
      return;
    }
    e.stopPropagation(); // the orbit controls never see this press
    e.preventDefault();
    try {
      this.canvas.setPointerCapture(e.pointerId);
    } catch {
      // A synthetic pointer: the drag still ends on its pointerup.
    }
    this.r.camera.getWorldDirection(this._normal);
    this.drag = {
      pointerId: e.pointerId,
      robot: hit.robot,
      body: hit.body,
      simBody: this.simBody(hit.body),
      local: hit.local,
      plane: new THREE.Plane().setFromNormalAndCoplanarPoint(this._normal, hit.point),
      x: e.clientX,
      y: e.clientY,
      target: hit.point.clone(),
      sent: null,
    };
    this.r.controls.enabled = false;
    this.r.setDrag({ body: hit.body, local: hit.local, target: this.drag.target });
    this._setCursor('grabbing');
    this._send();
    this.onStart(hit.robot);
  }

  _move(e) {
    const d = this.drag;
    if (d && e.pointerId === d.pointerId) {
      d.x = e.clientX;
      d.y = e.clientY;
      this.update();
    } else if (!d && e.pointerType === 'mouse' && e.buttons === 0) {
      this._hover = [e.clientX, e.clientY];
    }
  }

  /** Every drawn frame (and on a move): the target under the cursor, to the line and the worker.
   * Also the grab cursor when the mouse is over a robot. */
  update() {
    const d = this.drag;
    if (d) {
      if (this.r.planePoint(d.x, d.y, d.plane, this._t)) d.target.copy(this._t);
      this._send();
    } else if (this._hover) {
      const [x, y] = this._hover;
      this._hover = null;
      this._setCursor(this.enabled && this.r.pick(x, y) ? 'grab' : '');
    }
  }

  _send() {
    const d = this.drag;
    if (d.sent && d.sent.distanceTo(d.target) < MIN_MOVE) return;
    d.sent = d.target.clone();
    this.send({
      type: 'perturb', robot: d.robot, body: d.simBody,
      local: [d.local.x, d.local.y, d.local.z], target: [d.target.x, d.target.y, d.target.z],
    });
  }

  /** The drag ends: the push stops, orbit comes back. */
  end() {
    if (!this.drag) return;
    const { pointerId } = this.drag;
    this.drag = null;
    this.send({ type: 'perturbEnd' });
    this.r.controls.enabled = true;
    this.r.setDrag(null);
    this._setCursor('');
    try {
      if (this.canvas.hasPointerCapture?.(pointerId)) this.canvas.releasePointerCapture(pointerId);
    } catch {
      // Already released.
    }
  }

  /** End the drag if it holds robot `robot` (any robot when undefined): it fell, or every robot
   * was put back on its stand. */
  cancel(robot) {
    if (this.drag && (robot === undefined || this.drag.robot === robot)) this.end();
  }

  _setCursor(c) {
    if (this.canvas.style.cursor !== c) this.canvas.style.cursor = c;
  }
}
