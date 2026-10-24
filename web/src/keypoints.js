// Five-point overlay data, as deploy/sim2sim/sim_node.py's LiveOverlay keeps it: the window the
// policy tracks (its five-point targets: pelvis, ankles, wrists; frame 0 the current target, then
// the preview) drawn with its frame-0 pelvis at the robot's pelvis (x, y) of that step, since the
// goals are position-free; and a trail of the last KP_TRAIL frame-0 targets as they were drawn.
// Pure helpers, no DOM or three.js.

import { anchorFivePoints } from './live/reference.js';

// Colours as in the sim2sim five-point overlay (sim_node.KP_COLORS), in target order.
export const KP_COLORS = [
  [1.0, 0.85, 0.1], // pelvis: yellow
  [1.0, 0.45, 0.1], // left ankle: orange
  [0.1, 0.8, 0.9], // right ankle: cyan
  [0.9, 0.15, 0.25], // left wrist: red
  [0.2, 0.4, 1.0], // right wrist: blue
];
export const KP_POINTS = KP_COLORS.length;
export const KP_TRAIL = 50; // past targets kept (1 s at 50 Hz), sim_node's trail_len
export const TRAIL_ALPHA = [0.3, 0.9]; // oldest to newest trail segment, sim_node.TRAIL_ALPHA

/**
 * One robot's targets [K * 5 * 3] moved in the plane so the frame-0 pelvis sits at the robot's
 * pelvis `xy`, into `out`. Returns null (and leaves `out`) before the robot's first step, when
 * the targets are NaN.
 */
export function anchorTargets(targets, xy, out) {
  if (!Number.isFinite(targets[0])) return null;
  return anchorFivePoints(targets, xy, out);
}

/** The last `len` frame-0 targets of one robot (5 points x xyz each), oldest first. */
export class KpTrail {
  constructor(len = KP_TRAIL) {
    this.len = len;
    this.buf = new Float64Array(len * KP_POINTS * 3);
    this.start = 0;
    this.length = 0;
  }

  clear() {
    this.start = 0;
    this.length = 0;
  }

  /** Appends the five points of `frame0` (the first 15 numbers), dropping the oldest when full. */
  push(frame0) {
    const n = KP_POINTS * 3;
    const slot = (this.start + this.length) % this.len;
    for (let i = 0; i < n; i++) this.buf[slot * n + i] = frame0[i];
    if (this.length < this.len) this.length++;
    else this.start = (this.start + 1) % this.len;
  }

  /** Offset into `buf` of point p of the i-th entry (0 = oldest). */
  at(i, p) {
    return (((this.start + i) % this.len) * KP_POINTS + p) * 3;
  }
}

/** Opacity of trail segment i (0 = oldest) of n segments: sim_node's linspace(0.3, 0.9, n). */
export function trailAlpha(i, n) {
  const [a, b] = TRAIL_ALPHA;
  return n <= 1 ? a : a + ((b - a) * i) / (n - 1);
}
