// The stage fade over a re-home cut (src/fade.js) on a fake element: it covers the stage only
// while frames say a cut is coming, fades back in after the cut, and a late `rehoming` frame
// cannot leave the stage covered.
import test from 'node:test';
import assert from 'node:assert/strict';
import { StageFade } from '../src/fade.js';

function fakeLayer() {
  const set = new Set();
  const log = [];
  return {
    log,
    offsetWidth: 0,
    classList: {
      add: (...c) => { for (const x of c) set.add(x); log.push(`+${c.join(',')}`); },
      remove: (...c) => { for (const x of c) set.delete(x); },
      contains: (c) => set.has(c),
    },
  };
}

function make() {
  const el = fakeLayer();
  const rafs = [];
  const fade = new StageFade(el, (fn) => rafs.push(fn));
  const flushRaf = () => { while (rafs.length) rafs.shift()(); };
  return { el, fade, flushRaf };
}

test('fades out while a cut is coming, and back in after it', () => {
  const { fade, flushRaf } = make();
  assert.equal(fade.frame({ rehoming: false, rehomes: 0 }), false);
  assert.equal(fade.on, false);
  assert.equal(fade.frame({ rehoming: true, rehomes: 0 }), false);
  assert.equal(fade.on, true, 'covered before the cut');
  assert.equal(fade.frame({ rehoming: false, rehomes: 1 }), true, 'the cut');
  assert.equal(fade.rehomes, 1);
  assert.equal(fade.on, true, 'still covered until the next animation frame');
  flushRaf();
  assert.equal(fade.on, false, 'faded back in');
});

test('a cut with no warning frame is covered at once, then faded in', () => {
  const { el, fade, flushRaf } = make();
  assert.equal(fade.frame({ rehoming: false, rehomes: 1 }), true);
  assert.ok(el.log.includes('+instant,on'));
  assert.equal(el.classList.contains('instant'), false);
  flushRaf();
  assert.equal(fade.on, false);
});

test('a late rehoming frame (after Restart, or a fall that put the cut off) does not leave it covered', () => {
  const { fade } = make();
  fade.frame({ rehoming: true, rehomes: 0 });
  fade.clear(); // Restart
  assert.equal(fade.on, false);
  fade.frame({ rehoming: true, rehomes: 0 }); // posted before the worker saw the restart
  assert.equal(fade.on, true);
  fade.frame({ rehoming: false, rehomes: 0 }); // the loop after the restart (or the fall)
  assert.equal(fade.on, false, 'uncovered by the next frame that says no cut is coming');
  // Pause mid-fade: the page clears it, and passes rehoming false while paused.
  fade.frame({ rehoming: true, rehomes: 0 });
  fade.clear();
  fade.frame({ rehoming: false, rehomes: 0 });
  assert.equal(fade.on, false);
});
