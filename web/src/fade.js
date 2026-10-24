// The stage fade over a re-home cut (loop.js puts the robots back on their home stands): the
// layer fades out while the worker's frames say a cut is coming (`rehoming`), and back in once
// `rehomes` counts the cut. Any other frame uncovers the stage, so a late `rehoming` frame (one
// posted before a Restart, or before a fall put the cut off) cannot leave it covered. Pausing
// uncovers it too (clear()). No three.js; the element only needs classList (tests/fade.test.mjs).

export class StageFade {
  /** el: the fade layer ('on' covers the stage, 'instant' skips the transition). */
  constructor(el, raf = (fn) => requestAnimationFrame(fn)) {
    this.el = el;
    this.raf = raf;
    this.rehomes = 0; // re-homes seen
  }

  get on() {
    return this.el.classList.contains('on');
  }

  /** A posted frame {rehoming, rehomes}. Returns true when it is the first after a cut (the page
   * then snaps the camera and clears the trails). */
  frame({ rehoming, rehomes }) {
    if (rehomes !== this.rehomes) {
      this.rehomes = rehomes;
      this._fadeIn();
      return true;
    }
    if (rehoming) this.el.classList.add('on');
    else this.el.classList.remove('on');
    return false;
  }

  /** Uncover the stage now (Restart, pause). */
  clear() {
    this.el.classList.remove('on');
  }

  _fadeIn() {
    if (!this.on) {
      // The cut came with no warning frame (a slow tick): cover it at once, then fade in.
      this.el.classList.add('instant', 'on');
      void this.el.offsetWidth;
      this.el.classList.remove('instant');
    }
    this.raf(() => this.el.classList.remove('on'));
  }
}
