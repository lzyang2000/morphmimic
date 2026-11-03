# MorphMimic website

Browser demo that runs the tracking policy live: MuJoCo (WebAssembly) physics and the shared
policy (onnxruntime-web) for the three robots in a Web Worker, drawn with three.js. A visitor
can drag a robot to push it: a spring toward the cursor, capped in `src/live/sim.js`
(`PUSH_MAX_N`), and the policy has to catch the robot. Plus the
tools that record the runs for the page videos, render the videos and deploy the project page.

The page runs the released v0.1 policy (`morphmimic_policy.zip`: `policy.onnx`, `g1_amber.json`,
`x2.json`, `t2.json`) on its motion interface only; the page's G1 is `g1_amber`, the released
policy's G1 (`tools/gen_live_assets.py` `POLICY_ROBOT`). On screen the robots stand X2 Ultra, G1,
T2 from left to right (`tools/pack_recordings.py` `LATERAL`).

Python tools run from the repo root with `uv run --group export ...`; npm scripts and `.mjs` tools
run from `web/`. Everything they produce goes to the git-ignored `web/.cache/`, `web/public/` and
`web/dist/`.

## Tools, in the order to run them

**v0.1 recordings** were made locally, without step 1: the release zip unpacked into
`.cache/live/policy/`, an `export_assets.py` folder from the release code holding
`g1_amber.xml`, `x2.xml`, `t2.xml`, `meshes/` and `clips/<robot>/<clip>.npz` (with
`clips/g1_amber` the G1's clips) copied to `.cache/assets/`, then
`uv run --group export python web/tools/record_clips.py --deploy web/.cache/live/policy --assets
<that folder> --clips <the ten clips> --robots g1_amber x2 t2 --interfaces motion --out
web/.cache/recordings` and `policy.txt` set by hand.

1. **Record** (`web/tools/record_remote.sh <iteration|auto> [export|assets|record|fetch|all]`, from
   the repo root): freezes a `unified_v2` checkpoint on the recording host, exports it with
   `deploy/export_onnx.py`, and records the ten clips there with `web/tools/record_clips.py`
   (the unchanged `python -m deploy.run` launcher on each plain clip, not from the stand, one run
   per clip, robot and interface, run one at a time because they share ZMQ ports; `<out>/<clip>/<robot>_<interface>.npy` holds the robot's `qpos` per control step and
   a `.json` summary holds `nq` and `frames`). `all` runs export, assets and record and starts the
   recording in the background; rerun with `fetch` once its log ends in `ALLDONE` to copy the
   recordings and the robot assets to `web/.cache/{recordings,assets}`. It needs four environment
   variables and has no defaults:
   - `SM_REC_HOST`: ssh destination of the recording host (`user@host`).
   - `SM_REC_REMOTE`: remote work directory for the code copy, exports, recordings and logs.
   - `SM_CKPT_DIR`: remote `unified_v2` run directory holding `<robot>/ckpt_<it>.pt`.
   - `SM_ASSETS_DIRS`: two remote `export_assets` directories, space-separated: the base assets
     and the one that supplies the robot-dance clip.

   Remote paths are relative to the remote home directory, or absolute.
2. **Generate** (`npm run gen`): `tools/gen_scene.py` builds the meshes the page draws
   (`public/model/`: `meshes.bin`, `meshes.json`, `visuals.json`, and `scene.json` with the body
   names the posted poses are matched to). The recordings are not shipped: they only feed the
   page videos (step 5, which uses `tools/pack_recordings.py`'s layout helpers).
   **Live demo assets**: `web/tools/fetch_live_inputs.sh` (from the repo root) copies the policy
   export (`policy.onnx`, `<robot>.json`) and the ten clips' reference `.npz` files for the three
   robots into `.cache/live/`. It needs `SM_LIVE_HOST` (`user@host`), `SM_LIVE_POLICY` (the remote
   `export_onnx.py` folder) and `SM_LIVE_CLIP_DIRS` (two remote clip directories holding
   `<robot>/<clip>.npz`, space-separated: the base clips and the one with the robot-dance clip),
   with no defaults. Then `uv run --group export python web/tools/gen_live_assets.py` writes
   `public/live/`: the physics scene (`scene.xml`, `scene.json`: the three robots with their
   actuators and collision geoms, one floor, no robot-to-robot contact; the T2's joints are
   DC motors, whose clipped PD torque `src/live/sim.js` computes before each physics step as
   `sim_node.SimRobot.dc_torque` does), the policy side's inputs
   (`robots.json`), `policy.onnx` (the export cut down to `action_motion` and the inputs it
   reads), and the reference states (`clips.json`, `clips/<clip>.bin`:
   only the first 156 of the 240 state columns, the ones the goals read; the page fills the body
   velocities with zeros when it loads a clip).
3. **Test**: `npm test` (live policy side, loop, worker driver, DC motors, five-point overlay data,
   stage fade, WebAssembly SIMD probe, policy start-up order, packed clips, site checks,
   leak-list and drag-to-push tests; the push tests run the real MuJoCo scene and policy from
   `public/live/` and are skipped without it) and the Python tests,
   `uv run --group export python -m unittest discover -s web/tests_py -t .`. The recorder test
   needs `SM_TEST_ASSETS` set to a directory of exported G1 assets and `SM_TEST_DEPLOY` set to an
   `export_onnx.py` folder with `g1.json` (default `deploy/ckpt/goal_v2/onnx` if present), and is
   skipped without them.
4. **Cross-check**: `npm run xcheck-live` checks the live demo's policy side (`src/live/obs.js`,
   `src/live/reference.js`) against the Python deploy code: `tools/xcheck_live.py` runs the CPU
   sim2sim from the stand for each robot (one full loop of the stand, lead-in, clip,
   return and hold, then the next cue with another clip) and writes each tick's measurements and
   policy inputs to `.cache/xcheck_live/`; `tools/xcheck_live.mjs` rebuilds the inputs in JS from
   the same measurements and fails on a difference over 1e-5. It needs `public/live/` and
   `.cache/live/` (step 2).
   `npm run xcheck-policy` checks `src/live/policy.js` (onnxruntime-web, WASM, one thread, SIMD)
   on the page's pruned `public/live/policy.onnx` against Python onnxruntime on the full export:
   64 random input sets per robot, fail over 1e-4; it then times `act` and says whether 3 robots at 50 Hz fit 20 ms.
   In the page, `src/live/ort_browser.js` points ORT at the `.wasm` Vite emits into `dist/`.
   `npm run headless-live` runs the live loop (`src/live/sim.js`, `src/live/loop.js`) in Node on
   the real MuJoCo WASM model and policy: every clip, the three robots together,
   two loops each; it reports falls, joint error, end drift and the real-time factor into
   `.cache/headless_live/`, and fails on a motion-interface fall or a joint error more than
   0.03 rad from the Python sim2sim's. `tools/py_from_stand.sh` writes those
   Python figures first (`deploy.run --from-stand 2 --cue-s 1` per clip and robot, real time,
   about 4 minutes per robot). `npm run check-live-bundle` builds a page that only runs the live
   worker (`src/live/worker.js`), opens it in headless Chrome with every other host blocked, and
   checks that both `.wasm` files load from the build under the base.
5. **Render the page videos**
   (`MUJOCO_GL=egl uv run --group export python web/tools/render_videos.py --all`): the ten clip
   videos, their posters and the banner, from the same recordings, into `.cache/videos/`. Every clip,
   the walk included, is one shot of the three robots from the same group camera.
   Each clip video is at most 4 MB; the tool warns if one cannot be brought under that.
   `--all` then cuts the page's background video, `hero.mp4` and `hero_poster.jpg`, from the
   ballet-turns clip video: 3.36 to 13.26 s, libx264 CRF 23, yuv420p, faststart, no audio,
   metadata stripped, poster at 2.0 s (`--hero` redoes only that, from the clip video already in
   `.cache/videos/`).
   The film (`morphmimic.mp4`, `morphmimic_poster.jpg`) is made outside this repository: place
   it in `.cache/videos/` too, so a deploy ships the version you mean. A deploy never deletes it
   (step 8).
6. **Prepare the hardware videos** (`web/tools/prep_hardware.sh [--out DIR] [--mirror] INPUT NAME
   ...`): strips metadata and audio, mirrors on request, scales and encodes them into
   `.cache/hardware/` with a poster each. The page's three were made with
   `prep_hardware.sh <upper-body motion source> g1_hardware_motion --mirror <walk source>
   g1_hardware_walk <teleoperation source> g1_hardware_teleop` (the teleoperation source is
   already the right way round, so it is not mirrored).
7. **Build** (`npm run build`): runs `gen`, then builds the demo into `dist/` with the base
   `/morphmimic/demo/` (override with `DEMO_BASE`). `VITE_DEMO_DEBUG=1 npm run build` makes a
   debug build that also takes `?rehome=<m>`, the threshold of the re-home (the robots go back to
   their home stands, behind a short fade, at the first cue after they drift more than 2 m apart;
   `src/live/loop.js`), so a check can force one; never deploy it.
8. **Deploy** (`node tools/deploy_ghpages.mjs [--pages DIR] [--no-commit] [-m MSG]`): replaces
   `demo/` in the gh-pages worktree (the one with `gh-pages` checked out) with `dist/`, makes
   `static/videos/` hold exactly `.cache/videos` and `.cache/hardware` (stale files are removed,
   but never one the gh-pages `index.html` references in a src, href, poster, data-src or
   url(...); `tools/site_refs.mjs`), runs `check_site`, and commits on `gh-pages`. It never pushes and refuses `--push`. It commits
   only a build with the default base.
9. **Check the site** (`node tools/check_site.mjs <dir>`): fails on leak strings (machine paths,
   host and internal repo names; the list is in `tools/leaks.mjs`), reference-motion field names,
   `.npz` and checkpoint files, files under `.cache/`, policy (`.onnx`) and clip files anywhere
   but the live demo's `live/` folder (`live/policy.onnx` at the size `robots.json` records and
   the clips `live/clips.json` lists; at the root of `dist/`, under `demo/` on the site), and any
   `href` on the Code button; the site's top-level page must have that button, and every
   `static/` file it references must be in the tree.
   `deploy_ghpages` runs it; run it by hand on any other tree (`dist/`, the gh-pages worktree).
10. **Check the branch before pushing** (`node tools/check_branch.mjs [--base REV]`): fails if a
    leak string or another repository's name appears in the path or contents of any file
    `git ls-files web` lists, or in any commit message in `<base>..HEAD`. The base defaults to the
    merge base of `main` and `HEAD`, or to the root commit of `HEAD` when they share no history.
    Run it before every push of this branch.

## Local preview

`DEMO_BASE=/demo/ npm run build`, `node tools/deploy_ghpages.mjs --no-commit`, then
`node tools/shot_site.mjs <outdir> --width 1440` (and `--width 390`) for screenshots and console
errors. `node tools/screenshot.mjs out.png [--view N] [--at S]` checks the demo on its own:
it opens clip N (`?clip=N`), lets the simulation run S
seconds and saves the frame. Rebuild with
the default base and deploy again before committing.
