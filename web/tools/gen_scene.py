"""Build the browser's kinematic scene from the exported single-robot MJCF files.

  uv run --project .. --group export python tools/gen_scene.py [--cache DIR] [--out DIR] [--cell F]

Reads  <cache>/assets/{g1,x2,t2}.xml (+ meshes/), writes to <out>/model/:
  scene.xml     one model with the three robots' bodies and joints, prefixed g1/ x2/ t2/,
                no geoms, no floor (forward kinematics only, in MuJoCo WASM)
  scene.json    {robots: [{name, label, qposadr, nq}], bodies: [names],
                kpBodies: [[5 indices into bodies] per robot]}: the five-point bodies
                (pelvis or its equivalent, left ankle, right ankle, left wrist, right
                wrist; canonical tracked slots 0, 3, 6, 10, 13 of the robot's spec)
  visuals.json  [{body, mesh, pos, quat, rgba}], one per rendered geom
  meshes.bin    all decimated vertices (float32 xyz), then all faces (uint32 triples)
  meshes.json   [{name, vertOffset, vertCount, faceOffset, faceCount}]
                offsets count vertices / triangles within their block; face indices are
                local to the mesh. Face block starts at byte 12 * total vertices.

Meshes are reduced by vertex-clustering decimation: a cubic grid sized as a fraction of each
mesh's bounding-box diagonal, each cell collapsed onto the average of its vertices, triangles
kept only if their corners fall in three distinct cells.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import mujoco
import numpy as np

WEB = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = WEB / ".cache"
DEFAULT_OUT = WEB / "public"

ROBOTS = (("g1", "G1"), ("x2", "X2"), ("t2", "T2"))
# Cell edge as a fraction of each mesh's bounding-box diagonal. Tuned so the three robots
# together stay under the 250k-triangle budget (0.03 -> 337k, 0.04 -> 219k, 0.05 is too coarse for
# at least one mesh).
DEFAULT_CELL_FRAC = 0.04
BBOX_TOL_FRAC_OF_DIAG = 0.05
VISUAL_MAX_GROUP = 2
# Canonical tracked-body slots of the five-point interface: pelvis (or its equivalent), left
# ankle, right ankle, left wrist, right wrist.
KP_SLOTS = (0, 3, 6, 10, 13)


def decimate(verts: np.ndarray, faces: np.ndarray, cell_frac: float):
  """Vertex clustering. Returns (float32 [V,3], uint32 [F,3])."""
  verts = np.asarray(verts, dtype=np.float64)
  size = verts.max(0) - verts.min(0)
  diag = float(np.linalg.norm(size))
  if not diag > 0:
    raise ValueError("degenerate mesh: bounding box diagonal is 0")
  cell = diag * cell_frac
  idx = np.floor(verts / cell).astype(np.int64)
  idx -= idx.min(0)
  _, inverse = np.unique(idx, axis=0, return_inverse=True)  # lexicographic cell order
  inverse = inverse.reshape(-1)
  n_cells = int(inverse.max()) + 1
  counts = np.bincount(inverse, minlength=n_cells).astype(np.float64)
  rep = np.stack([np.bincount(inverse, weights=verts[:, a], minlength=n_cells) / counts for a in range(3)], 1)

  tri = inverse[np.asarray(faces)]
  keep = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
  tri = tri[keep]
  # Drop duplicates of the same oriented corner set: rotate the smallest cell to the front.
  rot = np.argmin(tri, axis=1)
  canon = np.take_along_axis(tri, (rot[:, None] + np.arange(3)) % 3, axis=1)
  _, first = np.unique(canon, axis=0, return_index=True)
  tri = tri[np.sort(first)]
  if len(tri) == 0:
    raise ValueError(f"every triangle collapsed at cell_frac={cell_frac}")

  used = np.unique(tri)
  remap = np.full(n_cells, -1, dtype=np.int64)
  remap[used] = np.arange(len(used))
  out_v = rep[used].astype(np.float32)
  out_f = remap[tri].astype(np.uint32)

  out_size = out_v.max(0) - out_v.min(0)
  drift = np.abs(size - out_size) / diag
  if drift.max() > BBOX_TOL_FRAC_OF_DIAG:
    raise ValueError(f"bounding box moved {drift.max() * 100:.1f}% of the diagonal; cell_frac too coarse")
  return out_v, out_f


def freeze_inertials(spec: mujoco.MjSpec, model: mujoco.MjModel) -> None:
  """Give every body whose inertia came from its geoms the explicit inertial `model` compiled
  for it, so deleting or replacing its geoms leaves its inertia as it was."""
  for body in spec.bodies:
    if body.name == "world" or body.explicitinertial:
      continue
    i = model.body(body.name).id
    if model.body_mass[i] > 0:
      body.explicitinertial = 1
      body.mass = float(model.body_mass[i])
      body.ipos = model.body_ipos[i]
      body.iquat = model.body_iquat[i]
      body.inertia = model.body_inertia[i]


def strip_to_kinematics(path: Path) -> mujoco.MjSpec:
  """Load a robot MJCF and keep only bodies and joints (with explicit inertials)."""
  spec = mujoco.MjSpec.from_file(str(path))
  freeze_inertials(spec, spec.compile())  # inertia may have come from geoms that are about to go
  for geom in list(spec.geoms):
    spec.delete(geom)
  for coll in (spec.sensors, spec.actuators, spec.excludes, spec.keys, spec.lights, spec.cameras,
               spec.sites, spec.meshes, spec.materials, spec.textures, spec.tendons,
               spec.equalities, spec.pairs):
    for item in list(coll):
      spec.delete(item)
  for body in list(spec.bodies):
    if body.name == "terrain":
      spec.delete(body)
  return spec


def visual_geoms(model: mujoco.MjModel, prefix: str) -> list[dict]:
  """Rendered geoms of a compiled single-robot model (mesh geoms, groups 0..2, alpha > 0)."""
  out = []
  for g in range(model.ngeom):
    if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] > VISUAL_MAX_GROUP:
      continue
    mat = model.geom_matid[g]
    rgba = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g]
    if rgba[3] <= 0:
      continue
    out.append({
        "body": prefix + model.body(model.geom_bodyid[g]).name,
        "mesh": prefix + model.mesh(model.geom_dataid[g]).name,
        "pos": [float(x) for x in model.geom_pos[g]],
        "quat": [float(x) for x in model.geom_quat[g]],
        "rgba": [float(x) for x in rgba],
        "_mesh_id": int(model.geom_dataid[g]),
    })
  return out


def mesh_arrays(model: mujoco.MjModel, mesh_id: int):
  v0, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
  f0, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
  return model.mesh_vert[v0:v0 + vn], model.mesh_face[f0:f0 + fn]


def kp_body_names(name: str) -> list[str]:
  """The robot's five-point bodies, as named in its MJCF (without the robot/ prefix)."""
  from morphmimic.robots import get_robot  # imports torch: only when the scene is built

  tracked = get_robot(name).tracked_bodies
  return [tracked[i] for i in KP_SLOTS]


def build(cache: Path = DEFAULT_CACHE, out: Path = DEFAULT_OUT, cell_frac: float = DEFAULT_CELL_FRAC) -> dict:
  assets = Path(cache) / "assets"
  model_dir = Path(out) / "model"
  model_dir.mkdir(parents=True, exist_ok=True)

  combined = mujoco.MjSpec()
  visuals: list[dict] = []
  mesh_names: list[str] = []
  mesh_data: list[tuple[np.ndarray, np.ndarray]] = []
  src_tris = 0
  for name, _ in ROBOTS:
    path = assets / f"{name}.xml"
    prefix = f"{name}/"
    single = mujoco.MjModel.from_xml_path(str(path))
    done: set[int] = set()
    for vis in visual_geoms(single, prefix):
      mid = vis.pop("_mesh_id")
      if mid not in done:
        done.add(mid)
        verts, faces = mesh_arrays(single, mid)
        src_tris += len(faces)
        mesh_names.append(vis["mesh"])
        mesh_data.append(decimate(verts, faces, cell_frac))
      visuals.append(vis)
    with warnings.catch_warnings():  # attach warns that child options (timestep...) are ignored
      warnings.simplefilter("ignore")
      combined.attach(strip_to_kinematics(path), prefix=prefix, frame=combined.worldbody.add_frame())

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = combined.compile()
    xml = combined.to_xml()
  (model_dir / "scene.xml").write_text(xml)

  robots = []
  for name, label in ROBOTS:
    first = model.joint(f"{name}/robot/floating_base_joint").id
    adr = int(model.jnt_qposadr[first])
    robots.append({"name": name, "label": label, "qposadr": adr})
  for i, r in enumerate(robots):
    end = robots[i + 1]["qposadr"] if i + 1 < len(robots) else int(model.nq)
    r["nq"] = end - r["qposadr"]
  bodies = [model.body(i).name for i in range(1, model.nbody)]
  kp_bodies = [[bodies.index(f"{name}/robot/{b}") for b in kp_body_names(name)]
               for name, _ in ROBOTS]
  (model_dir / "scene.json").write_text(
      json.dumps({"robots": robots, "bodies": bodies, "kpBodies": kp_bodies}, indent=1))
  (model_dir / "visuals.json").write_text(json.dumps(visuals))

  meta, vert_blocks, face_blocks = [], [], []
  vo = fo = 0
  for name, (v, f) in zip(mesh_names, mesh_data):
    meta.append({"name": name, "vertOffset": vo, "vertCount": len(v), "faceOffset": fo, "faceCount": len(f)})
    vert_blocks.append(v.astype("<f4").tobytes())
    face_blocks.append(f.astype("<u4").tobytes())
    vo += len(v)
    fo += len(f)
  (model_dir / "meshes.bin").write_bytes(b"".join(vert_blocks) + b"".join(face_blocks))
  (model_dir / "meshes.json").write_text(json.dumps(meta))
  summary = {"src_triangles": src_tris, "triangles": fo, "vertices": vo,
             "bin_bytes": 12 * (vo + fo), "robots": robots}
  return summary


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
  ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
  ap.add_argument("--cell", type=float, default=DEFAULT_CELL_FRAC)
  args = ap.parse_args()
  s = build(args.cache, args.out, args.cell)
  print(f"triangles {s['src_triangles']} -> {s['triangles']}, {s['vertices']} vertices, "
        f"meshes.bin {s['bin_bytes'] / 1e6:.2f} MB")
  for r in s["robots"]:
    print(f"  {r['name']}: qposadr {r['qposadr']} nq {r['nq']}")


if __name__ == "__main__":
  main()
