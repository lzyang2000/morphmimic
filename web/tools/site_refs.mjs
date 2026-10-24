// The local static/ files a page references, for tools/deploy_ghpages.mjs (it never removes one)
// and tools/check_site.mjs (each must be in the tree).

/**
 * The static/ paths `html` references in src, href, poster and data-src attributes and in CSS
 * url(...), relative to the page (a leading ./ dropped, query and fragment cut off), as a Set.
 */
export function staticRefs(html) {
  const refs = new Set();
  const add = (u) => {
    let p = u.trim().replace(/^\.\//, '').split(/[?#]/)[0];
    try {
      p = decodeURI(p);
    } catch {
      // keep it as written
    }
    if (p.startsWith('static/')) refs.add(p);
  };
  for (const m of html.matchAll(/\b(?:src|href|poster|data-src)\s*=\s*(["'])(.*?)\1/gis)) add(m[2]);
  for (const m of html.matchAll(/url\(\s*(["']?)([^"')]*?)\1\s*\)/gi)) add(m[2]);
  return refs;
}

/**
 * The files of a directory that are stale: in `existing` (names), not in `sources` (names it
 * will hold), and not referenced (`refs`, paths as staticRefs gives them; the directory is
 * `prefix`, e.g. 'static/videos/'). Returns {remove, kept}: kept are the referenced ones left in
 * place although no source has them.
 */
export function staleFiles(existing, sources, refs, prefix) {
  const remove = [];
  const kept = [];
  for (const f of existing) {
    if (sources.has(f)) continue;
    (refs.has(`${prefix}${f}`) ? kept : remove).push(f);
  }
  return { remove, kept };
}
