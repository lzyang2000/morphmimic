"""Render the page videos from the recorded simulations (the same qpos the browser demo plays).

  MUJOCO_GL=egl uv run --group export python web/tools/render_videos.py --all
  MUJOCO_GL=egl uv run --group export python web/tools/render_videos.py --clip <id> [--frames N]
  uv run --group export python web/tools/render_videos.py --hero

Reads <cache>/recordings (motion interface) and <cache>/assets (full-mesh MJCF), aligns every
robot into the shared layout (pack_recordings.align), and writes to <cache>/videos/:
  <clip_id>.mp4, <clip_id>_poster.jpg, banner.mp4, banner_poster.jpg, hero.mp4, hero_poster.jpg
H.264 yuv420p, 1280x720, 50 fps, no audio, each clip video at most 4 MB. Robots only, no ghost.
Every clip, the walk included, is one shot of the three robots from the same group camera.
The hero (the page's background video) is cut from the rendered ballet-turns clip video
(`--hero`, also part of `--all`): HERO_START to HERO_END s, re-encoded, metadata stripped.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

if __package__ in (None, ""):  # run as a script
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from web.tools import pack_recordings as pack

WEB = pack.WEB
DEFAULT_CACHE = pack.DEFAULT_CACHE
W, H, FPS = 1280, 720, pack.FPS
INTERFACE = "motion"
WALK = pack.CLIPS[0][0]
MAX_BYTES = 4_000_000
LOOK_SMOOTH = 0.05    # lookat low-pass factor per frame
LOOK_Z = 0.7
AZIMUTH = 180.0       # the camera looks from +x, the side every robot faces at frame 0

# (clip id, banner start s, length s) chosen by banner_window(); see the report.
BANNER_CLIP = "dancing_routine_V002_002__A365"
BANNER_SECONDS = 11.0
FLOOR_EMISSION = 0.5   # tuned so the lit floor reads #ffffff while shadows stay grey

# The hero: the spin of the ballet-turns clip, the idle stand before and after it trimmed (bounds
# from joint-speed stillness on all three robots, +-0.3 s); poster at HERO_POSTER_AT s of the cut.
HERO_CLIP = "dance_basic_turn_v1_360_R_loop_001__A325"
HERO_START, HERO_END = 3.36, 13.26
HERO_POSTER_AT = 2.0
HERO_CRF = 23


def build_model(cache: Path = DEFAULT_CACHE):
  """One model: a single floor and sun, the three full-mesh robots prefixed g1/ x2/ t2/."""
  assets = Path(cache) / "assets"
  combined = mujoco.MjSpec()
  base = mujoco.MjSpec.from_file(str(assets / "g1.xml"))
  combined.visual.global_.offwidth, combined.visual.global_.offheight = W, H
  combined.visual.quality.shadowsize = base.visual.quality.shadowsize
  combined.visual.headlight.ambient = base.visual.headlight.ambient
  combined.visual.headlight.diffuse = base.visual.headlight.diffuse
  combined.visual.headlight.specular = base.visual.headlight.specular
  combined.visual.rgba.haze = [1, 1, 1, 1]  # white haze, not MuJoCo's blue
  sky = combined.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                             builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                             rgb1=[1.0, 1.0, 1.0], rgb2=[0.957, 0.965, 0.973], width=512, height=3072)
  for name in pack.ROBOTS:
    child = mujoco.MjSpec.from_file(str(assets / f"{name}.xml"))
    child.meshdir = str((assets / "meshes" / name).resolve())
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      combined.attach(child, prefix=f"{name}/", frame=combined.worldbody.add_frame())
  # Pure white floor: no checker, no grid line, no reflection. Emission lifts it to white under the
  # scene lighting (diffuse alone renders ~0.85); the sun's shadows still darken it under the feet.
  for tex in combined.textures:
    if tex.name.endswith("groundplane"):
      tex.rgb1, tex.rgb2, tex.markrgb = [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]
  for mat in combined.materials:
    if mat.name.endswith("groundplane"):
      mat.reflectance = 0.0
      mat.rgba = [1, 1, 1, 1]
      mat.emission = FLOOR_EMISSION
  # Keep g1's floor and the shared sun; drop the other floors and the per-robot tracking lights.
  for body in list(combined.bodies):
    if body.name in ("x2/terrain", "t2/terrain"):
      combined.delete(body)
  suns = 0
  for light in list(combined.lights):
    if light.name.endswith("sun"):
      suns += 1
      if suns > 1:
        combined.delete(light)
    elif light.mode == mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM:
      combined.delete(light)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = combined.compile()
  adr = {n: int(model.jnt_qposadr[model.joint(f"{n}/robot/floating_base_joint").id])
         for n in pack.ROBOTS}
  return model, adr


def load_clip(clip_id: str, cache: Path = DEFAULT_CACHE) -> dict[str, np.ndarray]:
  rec = Path(cache) / "recordings" / clip_id
  out = {}
  for name in pack.ROBOTS:
    qpos, _, _ = pack._load(rec, name, INTERFACE)
    if qpos is None:
      raise FileNotFoundError(f"no completed {name} {INTERFACE} recording for {clip_id}")
    out[name] = pack.align(qpos, pack.LATERAL[name])
  return out


def camera_path(recs: dict[str, np.ndarray], n: int, *, distance: float,
                elevation: float) -> np.ndarray:
  """Per-frame (lookat x, y, z, azimuth deg, elevation, distance) of the group camera: it looks
  from +x at a fixed azimuth and follows the robots' mean position, low-passed. n frames, the
  arrays already padded to n."""
  pos = np.mean([r[:, 0:2] for r in recs.values()], axis=0)
  cam = np.zeros((n, 6))
  look = pos[0].copy()
  for i in range(n):
    look = (1 - LOOK_SMOOTH) * look + LOOK_SMOOTH * pos[i]
    cam[i] = (look[0], look[1], LOOK_Z, AZIMUTH, elevation, distance)
  return cam


def pad(q: np.ndarray, n: int) -> np.ndarray:
  return q if len(q) >= n else np.concatenate([q, np.repeat(q[-1:], n - len(q), 0)])


def encode_cmd(out: Path, crf: int) -> list[str]:
  return ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow",
          "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]


def save_poster(frame: np.ndarray, path: Path) -> None:
  from PIL import Image
  Image.fromarray(frame).save(path, quality=88)


MAX_CRF = 40


def shrink(path: Path, max_bytes: int, crf: int) -> int:
  """Re-encode path at rising CRF until it fits; returns the final CRF. Warns if the file is still
  over max_bytes once the CRF passes MAX_CRF."""
  while path.stat().st_size > max_bytes and crf < MAX_CRF:
    crf += 3
    tmp = path.with_suffix(".tmp.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-an", "-c:v", "libx264",
                    "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(tmp)], check=True)
    tmp.replace(path)
  size = path.stat().st_size
  if size > max_bytes:
    warnings.warn(f"{path.name} is {size / 1e6:.2f} MB at CRF {crf}, still over the "
                  f"{max_bytes / 1e6:.1f} MB limit", RuntimeWarning, stacklevel=2)
  return crf


def render(recs: dict[str, np.ndarray], out: Path, poster: Path | None, *, start: int = 0,
           frames: int | None = None, distance: float = 4.6, elevation: float = -10.0,
           cache: Path = DEFAULT_CACHE, crf: int = 24, max_bytes: int | None = MAX_BYTES,
           poster_at: float = 0.4) -> int:
  """Render recs to out, one group camera (camera_path) following the robots' mean position."""
  model, adr = build_model(cache)
  total = max(len(r) for r in recs.values())
  stop = total if frames is None else min(total, start + frames)
  start = min(start, stop - 1)
  n = stop - start
  recs = {k: pad(v, stop)[start:stop] for k, v in recs.items()}
  cam_path = camera_path(recs, n, distance=distance, elevation=elevation)
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, height=H, width=W)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  out.parent.mkdir(parents=True, exist_ok=True)
  enc = subprocess.Popen(encode_cmd(out, crf), stdin=subprocess.PIPE)
  poster_i = int(poster_at * n)
  try:
    for i in range(n):
      for name, q in recs.items():
        data.qpos[adr[name]:adr[name] + q.shape[1]] = q[i]
      mujoco.mj_forward(model, data)
      cam.lookat[:] = cam_path[i, 0:3]
      cam.azimuth, cam.elevation, cam.distance = cam_path[i, 3:6]
      renderer.update_scene(data, camera=cam)
      frame = renderer.render()
      if poster is not None and i == poster_i:
        save_poster(frame, poster)
      enc.stdin.write(np.ascontiguousarray(frame).tobytes())
  finally:
    enc.stdin.close()
    enc.wait()
    renderer.close()
  if enc.returncode:
    raise RuntimeError("ffmpeg failed")
  if max_bytes:
    shrink(out, max_bytes, crf)
  return n


def render_clip(clip_id: str, out: Path, poster: Path | None = None, frames: int | None = None,
                cache: Path = DEFAULT_CACHE) -> int:
  recs = load_clip(clip_id, cache)
  return render(recs, Path(out), poster, frames=frames, distance=4.2, elevation=-14.0, cache=cache)


def banner_window(recs: dict[str, np.ndarray], seconds: float = BANNER_SECONDS) -> int:
  """Start frame of the window that is lively (limbs moving) but stays near one spot."""
  n = min(len(r) for r in recs.values())
  win = int(seconds * FPS)
  speed = np.mean([np.linalg.norm(np.diff(r[:n, 7:], axis=0), axis=1) for r in recs.values()], 0)
  pos = np.mean([r[:n, 0:2] for r in recs.values()], 0)
  best, best_score = 0, -np.inf
  for s in range(0, n - win, FPS // 2):
    drift = np.linalg.norm(pos[s:s + win] - pos[s:s + win].mean(0), axis=1).max()
    score = speed[s:s + win].mean() - 0.02 * drift
    if score > best_score:
      best, best_score = s, score
  return best


def render_banner(out_dir: Path, cache: Path = DEFAULT_CACHE) -> tuple[int, int]:
  recs = load_clip(BANNER_CLIP, cache)
  start = banner_window(recs)
  n = render(recs, out_dir / "banner.mp4", out_dir / "banner_poster.jpg", start=start,
             frames=int(BANNER_SECONDS * FPS), distance=6.0, elevation=-8.0,
             cache=cache, crf=26, max_bytes=None)
  return start, n


def hero_cmds(src: Path, out_dir: Path) -> list[list[str]]:
  """The ffmpeg commands that cut hero.mp4 from the clip video `src` and take hero_poster.jpg from
  it: libx264 CRF 23, yuv420p, faststart, no audio, metadata and chapters stripped."""
  video, poster = out_dir / "hero.mp4", out_dir / "hero_poster.jpg"
  strip = ["-map_metadata", "-1", "-map_chapters", "-1", "-fflags", "+bitexact"]
  return [
      ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{HERO_START:.2f}", "-i", str(src),
       "-t", f"{HERO_END - HERO_START:.2f}", "-an", *strip, "-c:v", "libx264", "-crf", str(HERO_CRF),
       "-pix_fmt", "yuv420p", "-flags:v", "+bitexact", "-movflags", "+faststart", str(video)],
      ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{HERO_POSTER_AT:.2f}", "-i", str(video),
       "-frames:v", "1", "-q:v", "3", *strip, str(poster)],
  ]


def render_hero(out_dir: Path) -> tuple[Path, Path]:
  """hero.mp4 and hero_poster.jpg in out_dir, cut from out_dir/<HERO_CLIP>.mp4 (render it first)."""
  src = Path(out_dir) / f"{HERO_CLIP}.mp4"
  if not src.is_file():
    raise FileNotFoundError(f"{src} is missing: render the clip first (--clip {HERO_CLIP})")
  for cmd in hero_cmds(src, Path(out_dir)):
    subprocess.run(cmd, check=True)
  return Path(out_dir) / "hero.mp4", Path(out_dir) / "hero_poster.jpg"


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
  ap.add_argument("--out", type=Path, default=None)
  ap.add_argument("--clip", help="clip id")
  ap.add_argument("--frames", type=int, default=None)
  ap.add_argument("--all", action="store_true", help="every clip plus the banner")
  ap.add_argument("--banner", action="store_true")
  ap.add_argument("--hero", action="store_true", help="cut the hero from the rendered ballet-turns video")
  args = ap.parse_args()
  out = args.out or args.cache / "videos"
  out.mkdir(parents=True, exist_ok=True)
  ids = [c for c, _ in pack.CLIPS] if args.all else ([args.clip] if args.clip else [])
  for cid in ids:
    n = render_clip(cid, out / f"{cid}.mp4", out / f"{cid}_poster.jpg", args.frames, args.cache)
    print(f"{cid}: {n} frames, {(out / f'{cid}.mp4').stat().st_size / 1e6:.2f} MB", flush=True)
  if args.all or args.banner:
    start, n = render_banner(out, args.cache)
    print(f"banner: {BANNER_CLIP} frames {start}..{start + n} ({start / FPS:.1f} s)")
  if args.all or args.hero:
    video, _ = render_hero(out)
    print(f"hero: {HERO_CLIP} {HERO_START}..{HERO_END} s, {video.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
  main()
