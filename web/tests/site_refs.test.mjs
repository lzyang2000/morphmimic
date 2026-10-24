// tools/site_refs.mjs: the static/ files a page references, and which files deploy_ghpages may
// remove from static/videos (never one the page references).
import test from 'node:test';
import assert from 'node:assert/strict';
import { staleFiles, staticRefs } from '../tools/site_refs.mjs';

test('staticRefs finds src, href, poster, data-src and url(...) references to static/', () => {
  const html = `
    <link rel="icon" href="static/images/favicon.ico?v=3">
    <video autoplay poster="static/videos/hero_poster.jpg"><source src="static/videos/hero.mp4" type="video/mp4"></video>
    <video data-src='./static/videos/morphmimic.mp4#t=1'></video>
    <a href="static/morphmimic.pdf">PDF</a> <a href="demo/">demo</a> <a href="https://example.org/static/x.png">x</a>
    <style>.hero { background: url('static/videos/hero_poster.jpg') } .b { background:url(static/images/a%20b.png) }</style>
    <img src="static/images/fig2.png">`;
  assert.deepEqual([...staticRefs(html)].sort(), [
    'static/images/a b.png', 'static/images/fig2.png', 'static/images/favicon.ico', 'static/morphmimic.pdf',
    'static/videos/hero.mp4', 'static/videos/hero_poster.jpg', 'static/videos/morphmimic.mp4',
  ].sort());
});

test('staleFiles never removes a referenced file', () => {
  const refs = new Set(['static/videos/morphmimic.mp4', 'static/videos/morphmimic_poster.jpg', 'static/videos/hero.mp4']);
  const existing = ['hero.mp4', 'morphmimic.mp4', 'morphmimic_poster.jpg', 'old_clip.mp4', 'banner.mp4'];
  const sources = new Set(['hero.mp4', 'banner.mp4']);
  assert.deepEqual(staleFiles(existing, sources, refs, 'static/videos/'), {
    remove: ['old_clip.mp4'], kept: ['morphmimic.mp4', 'morphmimic_poster.jpg'],
  });
  assert.deepEqual(staleFiles(existing, sources, new Set(), 'static/videos/').remove,
    ['morphmimic.mp4', 'morphmimic_poster.jpg', 'old_clip.mp4'], 'without the page, the film would go');
});
