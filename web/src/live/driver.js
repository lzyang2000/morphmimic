// The worker's command side, kept free of MuJoCo and onnxruntime so it can be tested in Node
// (tests/driver.test.mjs): message checks, the timer that runs the loop (`pump`), the commands
// that wait for the control step in flight, and the clips fetched when they are first selected.
//
// Rules:
// - One init. Commands before the loop is ready are answered with a 'not ready' error; a second
//   init is refused.
// - setClip, setInterface and restart change the loop, so they never run while a control step
//   is in flight (policy.act is awaited, so messages arrive between a step's halves): they are
//   queued and run when the pump that owns the step finishes. Only one pump runs at a time; a
//   timer that fires while one is running does nothing (the running pump schedules the next).
// - A clip that is not in memory yet is fetched first (loadClip); only the clip selected last is
//   applied, and a failed fetch is reported and leaves the current clip playing.
// - perturb and perturbEnd (the page's drag) go to the sim at once, even mid-step: the sim reads
//   the push only inside its physics step, which runs after the policy calls, synchronously.
// - A frame is posted after every control step (loop.tick's callback), so a tick that runs several
//   steps still posts every pose.
// - An error while running stops the timer and is reported; only restart starts it again (the page
//   says "Press Restart"). Pause and resume still toggle the loop's clock meanwhile, but run nothing.

const COMMANDS = new Set(['setClip', 'setInterface', 'pause', 'resume', 'restart', 'perturb', 'perturbEnd']);
const INTERFACES = new Set(['motion', 'full', 'five']);
export const SLOW_RTF = 0.95; // a rate window under this runs below real time
export const SLOW_CATCH_UP = 2; // the catch-up cap while it does

export class Driver {
  /**
   * post(msg): to the page. postFrame(): build and post a frame of the loop's state.
   * loadClip(index) -> Promise of the clip ({states: per robot, human}, LiveLoop.setClip's).
   * now(): ms clock.
   * timers: {set(fn, ms) -> id, clear(id)}.
   */
  constructor({
    post, postFrame, loadClip, now = () => performance.now(),
    // Wrapped: a browser's setTimeout throws "Illegal invocation" when called as a method.
    timers = { set: (fn, ms) => setTimeout(fn, ms), clear: (id) => clearTimeout(id) },
  }) {
    this.post = post;
    this.postFrame = postFrame;
    this.loadClip = loadClip;
    this.now = now;
    this.timers = timers;
    this.loop = null;
    this.clips = []; // per clip index: its states once loaded, else null
    this.loading = new Map(); // clip index -> Promise of its states
    this.wanted = null; // the clip selected last
    this.initState = 'none'; // 'none' | 'started' | 'ready'
    this.timer = null;
    this.pumping = false;
    this.halted = false; // stopped by a run error until restart
    this.queued = [];
    this.rate = { rtf: 1, t0: null, s0: 0, slow: 0 };
    this.catchUp = null; // the loop's own catch-up cap, read at the first rate window
  }

  /** Called on an init message: false (and a reported error) if one came before. */
  claimInit() {
    if (this.initState !== 'none') {
      this.error('init', 'already initialised: init is accepted once');
      return false;
    }
    this.initState = 'started';
    return true;
  }

  /** The loop is built: `clips` has one entry per clip, the states of those already loaded
   * (else null), and `clip` is the one it starts with. Starts the timer. */
  start(loop, clips, clip) {
    this.loop = loop;
    this.clips = clips;
    this.wanted = clip;
    this.initState = 'ready';
    this.schedule(0);
  }

  error(stage, message, extra = {}) {
    this.post({ type: 'error', stage, message, ...extra });
  }

  /** Every message but init. */
  handle(msg) {
    const type = msg && typeof msg === 'object' ? msg.type : undefined;
    if (!COMMANDS.has(type)) return this.error(String(type), `unknown message ${JSON.stringify(type)}`);
    if (this.initState !== 'ready') return this.error(type, `not ready: ${type} was sent before the demo finished loading`);
    try {
      this[type](msg);
    } catch (e) {
      this.error(type, String(e?.message ?? e));
    }
  }

  setClip({ clip }) {
    if (!Number.isInteger(clip) || clip < 0 || clip >= this.clips.length) {
      throw new Error(`no clip ${JSON.stringify(clip)}: expected an integer from 0 to ${this.clips.length - 1}`);
    }
    this.wanted = clip;
    const apply = (states) => this.whenIdle(() => this.loop.setClip(states, clip));
    if (this.clips[clip]) return apply(this.clips[clip]);
    if (!this.loading.has(clip)) {
      this.loading.set(clip, Promise.resolve().then(() => this.loadClip(clip)).then((states) => {
        this.clips[clip] = states;
        return states;
      }).finally(() => this.loading.delete(clip)));
    }
    this.loading.get(clip).then((states) => {
      if (this.wanted === clip) apply(states); // a later selection wins
    }, (e) => this.error('clip', String(e?.message ?? e), { clip }));
  }

  setInterface({ iface }) {
    if (!INTERFACES.has(iface)) throw new Error(`no interface ${JSON.stringify(iface)}: expected motion, full or five`);
    this.whenIdle(() => this.loop.setInterface(iface));
  }

  pause() {
    this.loop.pause();
    if (this.timer !== null) {
      this.timers.clear(this.timer);
      this.timer = null;
    }
    this.rate.t0 = null;
  }

  resume() {
    if (!this.loop.paused) return;
    this.loop.resume(); // the clock restarts: paused time is not replayed
    this.rate.t0 = null;
    this.schedule(0); // nothing while halted: a run error waits for restart
  }

  restart() {
    this.whenIdle(() => {
      this.loop.restart();
      this.halted = false;
      this.rate.t0 = null;
      this.postFrame();
      this.schedule(0);
    });
  }

  /** A drag pushes robot `robot` at the point `local` of body `body` toward `target` (sim.js). */
  perturb({ robot, body, local, target }) {
    const vec = (a) => Array.isArray(a) && a.length === 3 && a.every(Number.isFinite);
    if (!Number.isInteger(robot) || !Number.isInteger(body)) throw new Error('perturb needs integer robot and body');
    if (!vec(local) || !vec(target)) throw new Error('perturb needs local and target, three finite numbers each');
    this.loop.sim.setPerturb(robot, body, local, target);
  }

  perturbEnd() {
    this.loop.sim.endPerturb();
  }

  get idle() {
    return !this.pumping && !this.loop.busy;
  }

  whenIdle(fn) {
    if (this.idle) fn();
    else this.queued.push(fn);
  }

  runQueued() {
    while (this.queued.length && this.idle) {
      const fn = this.queued.shift();
      try {
        fn();
      } catch (e) {
        this.error('command', String(e?.message ?? e));
      }
    }
  }

  schedule(ms) {
    if (this.timer === null && this.loop && !this.loop.paused && !this.halted) {
      this.timer = this.timers.set(() => this.pump(), ms);
    }
  }

  async pump() {
    this.timer = null;
    if (this.pumping || !this.loop) return; // the pump in flight schedules the next one
    this.pumping = true;
    try {
      if (!this.loop.paused && !this.halted) {
        // A frame after every control step, not one per tick: a slow device that runs several
        // steps in a tick still sends every pose.
        const ran = await this.loop.tick(this.now(), () => this.postFrame());
        if (ran > 0) this.measureRate();
      }
    } catch (e) {
      this.halted = true;
      this.error('run', `${String(e?.message ?? e)} (restart to run again)`);
    } finally {
      this.pumping = false;
    }
    this.runQueued();
    const due = this.loop.nextDue();
    this.schedule(due === null ? this.loop.dtMs : Math.max(0, due - this.now()));
  }

  /** Sim seconds per wall second, over windows of about 2 s of running. While it stays below
   * real time (two windows in a row under SLOW_RTF), a tick catches up at most SLOW_CATCH_UP
   * steps, so a tick (and a command queued behind it) waits less; the loop's own cap again once
   * a window keeps up. */
  measureRate() {
    const now = this.now();
    const t = this.loop.sim.time;
    if (this.rate.t0 === null) {
      this.rate.t0 = now;
      this.rate.s0 = t;
    } else if (now - this.rate.t0 >= 2000) {
      this.rate.rtf = (t - this.rate.s0) / ((now - this.rate.t0) / 1000);
      this.rate.t0 = now;
      this.rate.s0 = t;
      this.rate.slow = this.rate.rtf < SLOW_RTF ? this.rate.slow + 1 : 0;
      this.catchUp ??= this.loop.maxCatchUp;
      this.loop.maxCatchUp = this.rate.slow >= 2 ? Math.min(SLOW_CATCH_UP, this.catchUp) : this.catchUp;
    }
  }
}
