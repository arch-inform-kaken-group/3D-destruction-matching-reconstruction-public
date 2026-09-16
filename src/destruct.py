"""
========================================================================
COLOR CONFIGURATION (Source Code Constants)
========================================================================
Adjust these constants at the top of the file to change default material colors:
  - DEFAULT_GRAY (float): Fallback color for untextured meshes (0.0 to 1.0).
  - CLAY (RGB array): Default oxidized surface clay color. Used for fill and surface.
  - CORE (RGB array): Default reduced/dark clay core color for cross-sections.
  - LUM (RGB array): Luminance weights for shading palette colors based on source texture.

These can be overridden via CLI using --surface-color, --core-color, and --fill-color.

========================================================================
COMMAND LINE INTERFACE (CLI) USAGE
========================================================================
Basic Usage:
  python destruct.py -i pottery/mesh.glb -o output_dir/

1. POINT CLOUD & FILL BASICS
  --num-samples N         Number of surface points to sample (default: 2,000,000).
  --fill-color R G B      RGB color for fracture fill fallback (default: CLAY).

2. STRESS & VULNERABILITY FIELD
  --stress-model MODEL    ['fem_lite', 'impact', 'gravity', 'curvature', 'none'].
                          'fem_lite' uses Jacobi iteration; 'impact' uses directional impact;
                          'gravity' uses center-of-mass leverage; 'curvature' uses mean curvature.
  --grid-res N            Voxel grid resolution for FEM-lite (default: 64).
  --jacobi-iters N        Iterations for FEM-lite solver (default: 150).
  --impact-dir X Y Z      Direction vector for 'impact' stress model.
  --w-vol / --w-curv / --w-thin
                          Weights for volumetric stress, curvature, and inverse thickness.
  --thickness-alpha       Exponent for thickness gain (t_ref/t)^alpha (default: 1.5).
  --thickness-cap         Maximum cap for thickness gain (default: 5.0).
  --noise-sigma           Standard deviation for stochastic stress multiplier (default: 0.5).
  --noise-freqs           Number of frequency bands for band-limited noise (default: 48).
  --noise-length-factor   Wavelength scaling factor for stress noise (default: 0.33).

3. SEED GENERATION & SHARD SIZES
  --seed-radius-sigma     Jitter for NMS exclusion radius (0=uniform, >0=stochastic).
  --seed-score-noise      Jitter for NMS stress scores (default: 0.05).
  --rim-weight            Bias multiplier for seeds near mesh boundaries/rims.
  --rim-length-scale      Decay length for rim proximity bias (fraction of bbox diag).
  --curv-bias-weight      Bias multiplier for high-curvature regions.
  --thin-bias-weight      Bias multiplier for thin regions.
  --size-sigma            Log-normal sigma for shard size variation (0=uniform size).
  --power-weight-scale    Global scale for power-diagram weights (default: 0.15).

4. FRACTURE SURFACE RELIEF & TEXTURE
  --height-amp-scale      Amplitude of FBM relief as fraction of local thickness (default: 0.18).
  --height-H              Hurst exponent for FBM (0.0 to 1.0, higher = smoother).
  --height-octaves        Number of FBM octaves (default: 5).
  --grain-amp-scale       Amplitude of high-frequency Gaussian grain (default: 0.03).
  --height-fade-fraction  Fraction of wall thickness where relief fades to 0 at edges.
  --chip-prob             Base probability for micro-chipping at the fracture rim.
  --chip-width-scale      Width of the micro-chipping erosion zone in pixels.
  --fill-color-model      ['constant', 'depth_core']. 'depth_core' blends surface to core.
  --surface-color R G B   RGB color for the outer oxidized margin (default: CLAY).
  --core-color R G B      RGB color for the inner reduced core (default: CORE).

5. ASSEMBLY VISUALS & FRAGMENTATION
  --assembly-color-mode   ['shaded', 'flat']. 'shaded' modulates palette by source luminance.
  --palette-sat           Saturation for the golden-angle fragment palette (default: 0.75).
  --palette-val           Value/brightness for the fragment palette (default: 0.95).
  -n, --fragments N       Target number of fragments/shards (default: 16).
  --explode-distance      Absolute world-unit explode distance (overrides scale).
  --explode-scale         Explode distance as fraction of bounding box diagonal (default: 0.12).
  --min-faces             Minimum faces required to keep a fragment (default: 20).
  --seed                  Random seed for reproducibility (default: random).

Dependencies: pip install numpy scipy trimesh pillow
"""
from __future__ import annotations

import argparse, json, sys, time, traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import trimesh.visual
import trimesh.visual.color
import trimesh.visual.material
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.spatial import cKDTree

try:
    from PIL import Image
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

DEFAULT_GRAY = 0.80
CLAY = np.array([0.75,
                 0.51,
                 0.31])  # Warmer oxidized surface clay (reduced grey)
CORE = np.array([0.62,
                 0.42,
                 0.28])  # Warmer reduced core (rich terracotta, no grey)
LUM = np.array([0.2126, 0.7152, 0.0722])


@dataclass
class PointCloud:
    points: np.ndarray
    colors: np.ndarray

    def __len__(self):
        return int(len(self.points))


@dataclass
class Fragment:
    fid: str
    seed_index: int
    shell: trimesh.Trimesh
    exploded_shell: trimesh.Trimesh
    centroid: np.ndarray
    explode_offset: np.ndarray
    n_faces: int
    area: float
    n_points: int
    palette_rgb: np.ndarray
    vert_colors: Optional[np.ndarray] = None
    ply: Optional[str] = None


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rng_from_seed(s):
    return np.random.default_rng(None if s is None else int(s))


def rng_child(seed, *tags):
    return np.random.default_rng(
        np.random.SeedSequence([int(seed)] + [int(t) for t in tags]))


def bbox_diagonal(m):
    lo, hi = m.bounds
    return float(np.linalg.norm(hi - lo))


def norm01(x):
    x = np.asarray(x, float)
    m = x.max()
    return x / m if m > 1e-12 else np.zeros_like(x)


def random_unit_vector(rng):
    v = rng.normal(size=3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


# ============================ colors / texture ============================
def normalize_colors(arr, n):
    if arr is None: return None
    a = np.asarray(arr)
    if a.size == 0: return None
    if a.ndim == 1:
        if a.size == 3: a = np.tile(a.reshape(1, 3), (n, 1))
        elif a.size == 4: a = np.tile(a[:3].reshape(1, 3), (n, 1))
        elif a.size == n * 3: a = a.reshape(n, 3)
        elif a.size == n * 4: a = a.reshape(n, 4)[:, :3]
        else: return None
    if a.ndim != 2: return None
    if a.shape[0] != n:
        if a.shape[1] == n: a = a.T
        elif a.shape[0] in (3, 4) and a.shape[1] not in (3, 4): a = a.T
        else: return None
    if a.shape[1] >= 3: a = a[:, :3]
    elif a.shape[1] == 1: a = np.repeat(a, 3, axis=1)
    else: return None
    a = a.astype(np.float64, copy=False)
    try:
        if np.nanmax(a) > 1.5: a = a / 255.0
    except Exception:
        pass
    return np.clip(np.nan_to_num(a,
                                 nan=DEFAULT_GRAY,
                                 posinf=1.0,
                                 neginf=0.0),
                   0.0,
                   1.0)


def capture_source_texture(mesh):
    vis = getattr(mesh, "visual", None)
    if not isinstance(vis, trimesh.visual.TextureVisuals): return None, None
    try:
        uv = np.asarray(vis.uv, float)
        img = getattr(vis.material,
                      "image",
                      None) or getattr(vis.material,
                                       "baseColorTexture",
                                       None)
        if uv.ndim != 2 or uv.shape[0] != len(
                mesh.vertices) or uv.shape[1] != 2:
            return None, None
        if img is None: return None, None
        return uv, img
    except Exception:
        return None, None


def _bake_texture_colors(uv, image, n, flip=False):
    try:
        image = image.convert("RGB")
        arr = np.asarray(image, float) / 255.0
        h, w = arr.shape[:2]
        if h < 2 or w < 2 or len(uv) != n: return None
        u = np.clip(uv[:, 0] % 1.0, 0.0, 1.0) * (w - 1)
        v = uv[:, 1] % 1.0
        if flip: v = 1.0 - v
        v = np.clip(v, 0.0, 1.0) * (h - 1)
        x0 = np.floor(u).astype(np.int64)
        y0 = np.floor(v).astype(np.int64)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        fx = (u - x0)[:, None]
        fy = (v - y0)[:, None]
        return np.clip(
            arr[y0,
                x0] * (1 - fx) * (1 - fy) + arr[y0,
                                                x1] * fx * (1 - fy) + arr[y1,
                                                                          x0] *
            (1 - fx) * fy + arr[y1,
                                x1] * fx * fy,
            0,
            1)
    except Exception:
        return None


def color_stats(c):
    c = np.asarray(c, float)
    if c.size == 0: return {"std_mean": 0.0}
    return {"std_mean": float(c.std(axis=0).mean())}


def extract_vertex_colors(mesh, src_uv=None, src_image=None):
    n = len(mesh.vertices)
    vis = getattr(mesh, "visual", None)
    cands = []
    if vis is not None:
        try:
            c = normalize_colors(vis.to_color().vertex_colors, n)
            if c is not None: cands.append(("to_color", c))
        except Exception:
            pass
        c = normalize_colors(getattr(vis, "vertex_colors", None), n)
        if c is not None: cands.append(("vertex_colors", c))
    if HAVE_PIL and src_uv is not None and src_image is not None:
        c = _bake_texture_colors(src_uv, src_image, n)
        if c is not None: cands.append(("texture_uv", c))
        c = _bake_texture_colors(src_uv, src_image, n, flip=True)
        if c is not None: cands.append(("texture_uv_flip", c))
    cols = source = None
    for name, c in cands:
        if color_stats(c)["std_mean"] >= 1e-4:
            cols, source = c, name
            break
    if cols is None:
        cols = cands[0][1] if cands else np.full((n, 3), DEFAULT_GRAY)
        source = (cands[0][0] + "_constant") if cands else "fallback_gray"
    log(f"    colors: source={source} std_mean={color_stats(cols)['std_mean']:.4f}"
        )
    return cols


def apply_vertex_colors(mesh, rgb01):
    rgb01 = normalize_colors(rgb01, len(mesh.vertices))
    if rgb01 is None: rgb01 = np.full((len(mesh.vertices), 3), DEFAULT_GRAY)
    rgba = trimesh.visual.color.to_rgba((np.clip(rgb01,
                                                 0,
                                                 1) * 255).astype(np.uint8))
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba)
    return mesh


def double_sided_texture_material(image):
    return trimesh.visual.material.PBRMaterial(baseColorTexture=image,
                                               metallicFactor=0.0,
                                               roughnessFactor=0.95,
                                               doubleSided=True)


def ramp(t):
    t = np.clip(t, 0, 1)
    return np.stack([
        np.clip(1.6 * t,
                0,
                1),
        np.clip(1.5 * t - 0.45,
                0,
                1) * 0.7,
        np.clip(1.0 - 1.4 * t,
                0,
                1) * 0.85 + 0.15 * (1 - t)
    ],
                    -1)


def vertex_field(mesh, face_vals):
    acc = np.zeros(len(mesh.vertices))
    cnt = np.zeros(len(mesh.vertices))
    np.add.at(acc, mesh.faces.ravel(), np.repeat(face_vals, 3))
    np.add.at(cnt, mesh.faces.ravel(), 1)
    return acc / np.maximum(cnt, 1)


def stress_textured_mesh(mesh, s_face):
    sv = np.clip(vertex_field(mesh, s_face), 0.0, 1.0)
    m = mesh.copy()
    if HAVE_PIL:
        grad = (np.clip(ramp(np.linspace(0.0,
                                         1.0,
                                         256)),
                        0,
                        1) * 255).astype(np.uint8)
        img = Image.fromarray(grad[None, :, :])
        mat = trimesh.visual.material.PBRMaterial(baseColorTexture=img,
                                                  metallicFactor=0.0,
                                                  roughnessFactor=0.95,
                                                  doubleSided=True)
        uv = np.stack([sv, np.full_like(sv, 0.5)], axis=1)
        m.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    else:
        apply_vertex_colors(m, ramp(sv))
    return m


# ============================ palette / assembly colors ============================
def fragment_palette(n, sat=0.75, val=0.95):
    """Deterministic distinct colors: golden-angle hues in HSV."""
    n = max(1, int(n))
    h = (np.arange(n) * 0.618033988749895) % 1.0
    i = (h * 6).astype(int) % 6
    f = (h * 6) - np.floor(h * 6)
    p = val * (1 - sat)
    q = val * (1 - f * sat)
    t = val * (1 - (1 - f) * sat)
    r = np.select([i == 0,
                   i == 1,
                   i == 2,
                   i == 3,
                   i == 4,
                   i == 5],
                  [val,
                   q,
                   p,
                   p,
                   t,
                   val])
    g = np.select([i == 0,
                   i == 1,
                   i == 2,
                   i == 3,
                   i == 4,
                   i == 5],
                  [t,
                   val,
                   val,
                   q,
                   p,
                   p])
    b = np.select([i == 0,
                   i == 1,
                   i == 2,
                   i == 3,
                   i == 4,
                   i == 5],
                  [p,
                   p,
                   t,
                   val,
                   val,
                   q])
    return np.clip(np.stack([r, g, b], -1), 0, 1)


def shade_palette(pal_rgb, base_cols, mode):
    """Per-fragment palette color; 'shaded' modulates by source luminance."""
    base_cols = np.asarray(base_cols, float)
    if mode == "flat" or base_cols.size == 0:
        return np.tile(np.asarray(pal_rgb, float), (max(len(base_cols), 0), 1))
    lum = np.clip(base_cols @ LUM, 0, 1)
    f = 0.35 + 0.65 * lum
    return np.clip(np.asarray(pal_rgb, float)[None, :] * f[:, None], 0, 1)


# ============================ stress field ============================
def build_grid(mesh, res):
    diag = bbox_diagonal(mesh)
    vox = mesh.voxelized(pitch=diag / max(int(res), 16))
    return np.asarray(vox.matrix, bool), vox.transform, diag


def fem_lite(occ, iters, up_axis):
    S = occ.shape
    u = np.zeros((3, ) + S)
    f = np.zeros((3, ) + S)
    f[up_axis] = -1.0 * occ
    proj = occ.any(axis=tuple(i for i in range(3) if i != up_axis))
    z0 = int(np.where(proj)[0].min()) if proj.any() else 0
    idx = [slice(None)] * 3
    idx[up_axis] = slice(z0, z0 + 3)
    mask = np.zeros(S, bool)
    mask[tuple(idx)] = True
    fixed = occ & mask
    for _ in range(int(iters)):
        for c in range(3):
            uc = np.pad(u[c], 1)
            nb = (uc[:-2, 1:-1, 1:-1] + uc[2:, 1:-1, 1:-1] + uc[1:-1, :-2, 1:-1] +
                  uc[1:-1, 2:, 1:-1] + uc[1:-1, 1:-1, :-2] + uc[1:-1, 1:-1, 2:])
            new = (nb + 0.25 * f[c]) / 6.0
            new[~occ] = 0.0
            new[fixed] = 0.0
            u[c] = 0.5 * u[c] + 0.5 * new
    g = np.gradient(u, axis=(1, 2, 3))
    eps = np.zeros((3, 3) + S)
    for i in range(3):
        for j in range(3):
            eps[i, j] = 0.5 * (g[j][i] + g[i][j])
    return np.sqrt(sum(eps[i, j]**2 for i in range(3) for j in range(3)))


def surface_sample(mesh, occ, T, vals):
    W = np.asarray(mesh.triangles_center)
    W2G = np.linalg.inv(T)
    G = W @ W2G[:3, :3].T + W2G[:3, 3]
    idx = np.stack(np.nonzero(occ), 1).astype(float)
    if len(idx) == 0: return np.zeros(len(G))
    world = idx @ T[:3, :3].T + T[:3, 3]
    k = min(4, len(world))
    d, i = cKDTree(world).query(G, k=k)
    if k == 1:
        d = d[:, None]
        i = i[:, None]
    w = 1.0 / (d + 1e-6)
    w /= w.sum(1, keepdims=True)
    return (vals[occ][i] * w).sum(1)


def wall_thickness_world(mesh, occ, T, max_steps=160, step=0.5):
    """World-unit wall thickness: march inward along -normal through voxels,
    measure the leading occupied run length (in voxels), scale by pitch."""
    W = np.asarray(mesh.triangles_center, float)
    N = np.asarray(mesh.face_normals, float)
    W2G = np.linalg.inv(T)
    G = W @ W2G[:3, :3].T + W2G[:3, 3]
    M = W2G[:3, :3] @ (-N.T)
    ln = np.linalg.norm(M, axis=0, keepdims=True)
    D = (M / np.where(ln > 1e-12, ln, 1.0)).T
    S = occ.shape
    F = len(W)
    thick = np.zeros(F)
    state = np.zeros(F, np.int8)  # 0 seek, 1 counting, 2 done
    for k in range(1, int(max_steps) + 1):
        q = G + D * (k * step)
        gi = np.round(q).astype(np.int64)
        valid = ((gi >= 0).all(1) & (gi[:,
                                        0] < S[0]) & (gi[:,
                                                         1] < S[1]) &
                 (gi[:,
                     2] < S[2]))
        ins = np.zeros(F, bool)
        ix = np.flatnonzero(valid)
        if len(ix):
            ins[ix] = occ[gi[ix, 0], gi[ix, 1], gi[ix, 2]]
        start = (state == 0) & ins
        state[start] = 1
        thick[start] += step
        cont = (state == 1) & ins & ~start
        thick[cont] += step
        stop = (state == 1) & ~ins
        state[stop] = 2
        if not (state == 1).any():
            break
    pitch = float(np.abs(np.linalg.det(T[:3, :3]))**(1.0 / 3.0))
    thick *= pitch
    tv = vertex_field(mesh, thick)  # light smoothing
    return tv[np.asarray(mesh.faces)].mean(1)


def thickness_gain(t, t_ref, alpha, cap, floor_q=0.05):
    t = np.asarray(t, float)
    pos = t[t > 0]
    if len(pos) == 0 or alpha == 0.0:
        return np.ones_like(t)
    t_floor = max(float(np.quantile(pos, floor_q)), 1e-6)
    tt = np.clip(t, t_floor, None)
    return np.clip((t_ref / tt)**alpha, 1.0 / cap, cap)


def curvature_field(mesh, radius):
    try:
        k = trimesh.curvature.discrete_mean_curvature_measure(
            mesh,
            mesh.triangles_center,
            radius)
        return np.abs(np.nan_to_num(k))
    except Exception:
        return np.zeros(len(mesh.faces))


def bandlimited_noise(points, rng, wavelength, n_freq=48, jitter=0.35):
    points = np.asarray(points, float)
    dirs = rng.normal(size=(n_freq, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    waves = wavelength * rng.uniform(1.0 - jitter, 1.0 + jitter, n_freq)
    k = (2.0 * np.pi / waves)[:, None] * dirs
    ph = rng.uniform(0.0, 2.0 * np.pi, n_freq)
    val = np.zeros(len(points), dtype=float)
    for i in range(n_freq):
        val += np.sin(points @ k[i] + ph[i])
    val /= np.sqrt(n_freq)
    val -= val.mean()
    s = val.std()
    return val / s if s > 1e-9 else val


def compute_stress(mesh, args, diag):
    """Returns dict(s_hat, thick, t_ref, curv). thick in world units or None."""
    out = {
        "s_hat": np.zeros(len(mesh.faces)),
        "thick": None,
        "t_ref": None,
        "curv": np.zeros(len(mesh.faces))
    }
    if args.stress_model == "none":
        return out
    occ = T = None
    vol = None
    thick = None
    up = int(np.argmax(mesh.bounds[1] - mesh.bounds[0]))
    try:
        occ, T, _ = build_grid(mesh, args.grid_res)
        thick = wall_thickness_world(mesh, occ, T)
        if not (thick > 0).any():
            thick = None
        vol = norm01(surface_sample(mesh, occ, T, fem_lite(occ, args.jacobi_iters, up))) \
            if args.stress_model == "fem_lite" else None
    except Exception as exc:
        log(f"    ! stress voxel stage failed ({exc}); fallback")
        occ = None
        vol = None
        thick = None
    curv = norm01(curvature_field(mesh, diag / 40.0))
    if vol is None:
        if args.stress_model == "impact":
            dv = np.asarray(args.impact_dir, float)
            dv /= np.linalg.norm(dv) + 1e-12
            ip = mesh.bounds.mean(0) + dv * diag * 0.5
            _, iv, _ = trimesh.proximity.closest_point(mesh, ip[None])
            d = np.linalg.norm(mesh.triangles_center -
                               mesh.triangles_center[iv[0]],
                               axis=1)
            vol = norm01(np.exp(-d / (diag / 6.0)))
        elif args.stress_model == "gravity":
            z = mesh.triangles_center[:, up] - mesh.bounds[0][up]
            c2 = np.delete(mesh.bounds.mean(0), up)
            lever = np.linalg.norm(np.delete(mesh.triangles_center,
                                             up,
                                             1) - c2,
                                   axis=1)
            vol = norm01((z / diag) * (lever / diag))
        else:
            vol = curv
    if thick is not None:
        t_ref = float(np.median(thick[thick > 0]))
        t_floor = max(float(np.quantile(thick[thick > 0], 0.05)), 1e-6)
        thin = norm01(1.0 / np.clip(thick, t_floor, None))
    else:
        t_ref = None
        thin = norm01(surface_sample(mesh, occ, T, distance_transform_edt(occ))) \
            if occ is not None else np.zeros(len(mesh.faces))
        thin = 1.0 - thin
    s_hat = norm01(args.w_vol * vol + args.w_curv * curv + args.w_thin * thin)
    out.update(s_hat=s_hat, thick=thick, t_ref=t_ref, curv=curv)
    return out


def nms_maxima(values,
               centers,
               radius,
               n,
               rng=None,
               radius_sigma=0.0,
               score_noise=0.0):
    values = np.asarray(values, float)
    if rng is not None and (radius_sigma > 0 or score_noise > 0):
        score = values.copy()
        if score_noise > 0:
            score = score + score_noise * rng.standard_normal(len(values))
        radii = radius * np.exp(rng.normal(0.0, radius_sigma, len(values))) if radius_sigma > 0 \
            else np.full(len(values), radius)
        order = np.argsort(-score)
        sel = []
        sel_pts = []
        sel_rad = []
        for i in order:
            if len(sel) >= n: break
            p = centers[i]
            r = radii[i]
            ok = True
            for q, rq in zip(sel_pts, sel_rad):
                if np.linalg.norm(p - q) < 0.5 * (r + rq):
                    ok = False
                    break
            if ok:
                sel.append(int(i))
                sel_pts.append(p)
                sel_rad.append(r)
        return np.asarray(sel, dtype=int)
    order = np.argsort(-values)
    sel = []
    for i in order:
        if len(sel) >= n: break
        if all(np.linalg.norm(centers[i] - centers[j]) > radius for j in sel):
            sel.append(int(i))
    return np.asarray(sel, dtype=int)


# ============================ power diagram helpers ============================
def power_mid(si, sj, wi, wj):
    """Bisector plane midpoint for squared-distance power diagram.
    Larger w_i pushes the interface toward j (cell i grows)."""
    d = sj - si
    dd2 = float(d @ d)
    if dd2 < 1e-24:
        return 0.5 * (si + sj)
    return 0.5 * (si + sj) + ((wi - wj) / (2.0 * dd2)) * d


def weighted_labels(points, seeds, weights, chunk=262144):
    labels = np.empty(len(points), dtype=np.int64)
    for s0 in range(0, len(points), chunk):
        X = points[s0:s0 + chunk]
        D = ((X[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2) - weights[None, :]
        labels[s0:s0 + chunk] = D.argmin(axis=1)
    return labels


def weighted_two_nearest(points, seeds, weights):
    D = ((points[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2) - weights[None, :]
    nn = np.argpartition(D, 2, axis=1)[:, :2]
    return np.sort(nn, axis=1)


# ============================ sampling ============================
def sample_surface_points_colors(mesh, colors, n, rng):
    n = int(max(0, n))
    if n <= 0 or len(mesh.faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    colors = normalize_colors(colors, len(mesh.vertices))
    if colors is None: colors = np.full((len(mesh.vertices), 3), DEFAULT_GRAY)
    areas = np.asarray(mesh.area_faces, float)
    tot = float(areas.sum())
    p = areas / tot if tot > 1e-18 else np.full(len(mesh.faces),
                                                1.0 / len(mesh.faces))
    fi = rng.choice(len(mesh.faces), size=n, p=p)
    r = rng.random((n, 2))
    over = r.sum(1) > 1.0
    r[over] = 1.0 - r[over]
    b = np.empty((n, 3))
    b[:, 1:] = r
    b[:, 0] = 1.0 - r.sum(1)
    tri = mesh.triangles[fi]
    col = colors[mesh.faces[fi]]
    pts = np.einsum("ij,ijk->ik", b, tri)
    cols = np.einsum("ij,ijk->ik", b, col)
    return np.asarray(pts, float), np.clip(cols, 0.0, 1.0), np.asarray(fi, np.int64)


def make_plane_basis(normal):
    normal = np.asarray(normal, float)
    normal /= np.linalg.norm(normal) + 1e-12
    h = np.array([1.0, 0.0, 0.0])
    if abs(float(normal @ h)) > 0.9: h = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, h)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(normal, u)
    v /= np.linalg.norm(v) + 1e-12
    return u, v


# ============================ exact triangle-plane intersection ============================
def plane_segments(mesh, face_idx, nv, mid, eps=1e-9):
    """Exact intersection segments + per-segment triangle normals."""
    if len(face_idx) == 0:
        return np.zeros((0, 2, 3)), np.zeros((0, 3))
    F = np.asarray(mesh.faces)[face_idx]
    V = np.asarray(mesh.vertices, float)
    P = V[F]
    d = (P - mid) @ nv
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    nl = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.where(nl > 1e-300, nl, 1.0)
    op = np.abs(d) <= eps
    ea = np.array([0, 1, 2])
    eb = np.array([1, 2, 0])
    da = d[:, ea]
    db = d[:, eb]
    ce = ((da < -eps) & (db > eps)) | ((da > eps) & (db < -eps))
    denom = da - db
    safe = np.abs(denom) > 1e-300
    t = np.where(safe, da / np.where(safe, denom, 1.0), 0.0)
    ept = P[:, ea] + t[..., None] * (P[:, eb] - P[:, ea])
    cand = np.where(op[..., None], P, ept)
    sel = op | ce
    k = sel.sum(1)
    out, onrm = [], []
    m2 = np.flatnonzero(k == 2)
    if len(m2):
        s = sel[m2]
        order = np.argsort(~s, axis=1, kind="stable")[:, :2]
        c = cand[m2]
        idx3 = order[..., None].repeat(3, axis=2)
        p0 = np.take_along_axis(c, idx3[:, :1, :], axis=1)[:, 0, :]
        p1 = np.take_along_axis(c, idx3[:, 1:2, :], axis=1)[:, 0, :]
        segs = np.stack([p0, p1], axis=1)
        L = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
        good = L > eps
        if good.any():
            out.append(segs[good])
            onrm.append(nrm[m2][good])
    for mi in np.flatnonzero(k > 2):
        tri, td = P[mi], d[mi]
        pts = [tri[kk] for kk in range(3) if abs(td[kk]) <= eps]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            da_, db_ = td[a], td[b]
            if (da_ < -eps and db_ > eps) or (da_ > eps and db_ < -eps):
                tt = da_ / (da_ - db_)
                pts.append(tri[a] + tt * (tri[b] - tri[a]))
        uniq = []
        for p in pts:
            if not any(np.linalg.norm(p - q) <= eps for q in uniq):
                uniq.append(p)
        if len(uniq) < 2: continue
        if len(uniq) > 2:
            best, bd = None, -1.0
            for a in range(len(uniq)):
                for b in range(a + 1, len(uniq)):
                    dd = np.linalg.norm(uniq[a] - uniq[b])
                    if dd > bd: bd, best = dd, (uniq[a], uniq[b])
            p0, p1 = best
        else:
            p0, p1 = uniq[0], uniq[1]
        if np.linalg.norm(p1 - p0) > eps:
            out.append(np.asarray([[p0, p1]], dtype=float))
            onrm.append(nrm[mi][None, :])
    if not out:
        return np.zeros((0, 2, 3)), np.zeros((0, 3))
    return np.concatenate(out, axis=0), np.concatenate(onrm, axis=0)


# ============================ chaining / raster helpers ============================
def _chain_polylines(segs2, snap):
    keys = {}
    node = []
    edges = []

    def nid(p):
        k = (round(float(p[0]) / snap), round(float(p[1]) / snap))
        if k not in keys:
            keys[k] = len(node)
            node.append(np.asarray(p, float))
        return keys[k]

    for a, b in segs2:
        ia, ib = nid(a), nid(b)
        if ia != ib: edges.append((ia, ib))
    adj = defaultdict(list)
    for eid, (a, b) in enumerate(edges):
        adj[a].append((eid, b))
        adj[b].append((eid, a))
    used = np.zeros(len(edges), dtype=bool)
    polys = []

    def choose_next(cur, prev, candidates):
        if not candidates: return None
        if prev is None or len(candidates) == 1: return candidates[0]
        incoming = node[cur] - node[prev]
        ni = np.linalg.norm(incoming)
        if ni < 1e-12: return candidates[0]
        incoming = incoming / ni
        best, best_score = None, -np.inf
        for eid, nb in candidates:
            outgoing = node[nb] - node[cur]
            no = np.linalg.norm(outgoing)
            if no < 1e-12: continue
            score = float(incoming @ (outgoing / no))
            if score > best_score: best_score, best = score, (eid, nb)
        return best

    for start_eid, (a, b) in enumerate(edges):
        if used[start_eid]: continue
        used[start_eid] = True
        seq = [a, b]
        start = a
        prev, cur = a, b
        closed = False
        while True:
            cands = [(e2, nb) for e2, nb in adj[cur] if not used[e2]]
            nxt = choose_next(cur, prev, cands)
            if nxt is None: break
            eid, nb = nxt
            used[eid] = True
            if nb == start:
                closed = True
                break
            seq.append(nb)
            prev, cur = cur, nb
        if not closed:
            prev, cur = b, a
            while True:
                cands = [(e2, nb) for e2, nb in adj[cur] if not used[e2]]
                nxt = choose_next(cur, prev, cands)
                if nxt is None: break
                eid, nb = nxt
                used[eid] = True
                if nb == seq[-1]:
                    closed = True
                    break
                seq.insert(0, nb)
                prev, cur = cur, nb
        if len(seq) >= 3:
            polys.append((np.asarray([node[i] for i in seq]), closed))
    return polys


def _shoelace(P2):
    x, y = P2[:, 0], P2[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def _poly_length(P2):
    return float(np.linalg.norm(np.diff(P2, axis=0), axis=1).sum())


def _point_in_poly(poly, p):
    x, y = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cond = (y > p[1]) != (y2 > p[1])
    xt = x + (p[1] - y) * (x2 - x) / np.where(
        np.abs(y2 - y) < 1e-300,
        1e-300,
        y2 - y)
    return int(np.sum(cond & (xt > p[0]))) % 2 == 1


def _scanline_mask(poly2, gs, xmin, ymin, rows, cols):
    px = (poly2[:, 0] - xmin) / gs
    py = (poly2[:, 1] - ymin) / gs
    x1 = np.roll(px, -1)
    y1 = np.roll(py, -1)
    mask = np.zeros((rows, cols), dtype=bool)
    ry = np.arange(rows) + 0.5
    rows_list = []
    xs_list = []
    for k in range(len(px)):
        ya, yb = py[k], y1[k]
        if ya == yb: continue
        lo, hi = (ya, yb) if ya < yb else (yb, ya)
        sel = (ry >= lo) & (ry < hi)
        if not sel.any(): continue
        t = (ry[sel] - ya) / (yb - ya)
        rows_list.append(np.nonzero(sel)[0])
        xs_list.append(px[k] + t * (x1[k] - px[k]))
    if not rows_list: return mask
    rid = np.concatenate(rows_list)
    xv = np.concatenate(xs_list)
    order = np.lexsort((xv, rid))
    rid = rid[order]
    xv = xv[order]
    starts = np.flatnonzero(np.diff(rid)) + 1
    for r, xs in zip(np.split(rid, starts), np.split(xv, starts)):
        if len(xs) % 2: xs = np.append(xs, xs[-1])
        for a, b in zip(xs[0::2], xs[1::2]):
            i0 = max(0, int(np.ceil(a - 0.5)))
            i1 = min(cols - 1, int(np.floor(b - 0.5)))
            if i1 >= i0: mask[r, i0:i1 + 1] = True
    return mask


def _raster_segments(xy2, gs, xmin, ymin, rows, cols):
    mask = np.zeros((rows, cols), dtype=bool)
    for a, b in xy2:
        dv = b - a
        L = float(np.hypot(dv[0], dv[1]))
        n = max(2, int(np.ceil(L / (gs * 0.5))) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        pts = a[None, :] + t * dv[None, :]
        px = np.clip(((pts[:, 0] - xmin) / gs).astype(int), 0, cols - 1)
        py = np.clip(((pts[:, 1] - ymin) / gs).astype(int), 0, rows - 1)
        mask[py, px] = True
    return mask


def _disk(r):
    r = max(1, int(r))
    d = np.arange(-r, r + 1)
    return (d[:, None]**2 + d[None, :]**2) <= r * r


# ============================ fracture surface fields ============================
def _value_noise_2d(shape, g, rng):
    rows, cols = shape
    g = int(max(2, min(g, min(rows, cols))))
    grid = rng.random((g, g))
    yy = np.linspace(0, g - 1, rows)
    xx = np.linspace(0, g - 1, cols)
    y0 = np.floor(yy).astype(int)
    x0 = np.floor(xx).astype(int)
    y1 = np.minimum(y0 + 1, g - 1)
    x1 = np.minimum(x0 + 1, g - 1)
    fy = (yy - y0)[:, None]
    fx = (xx - x0)[None, :]
    a = grid[np.ix_(y0, x0)]
    b = grid[np.ix_(y0, x1)]
    c = grid[np.ix_(y1, x0)]
    d = grid[np.ix_(y1, x1)]
    return a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx + c * fy * (
        1 - fx) + d * fy * fx


def fbm2(shape, rng, H, octaves, base_g=4):
    val = np.zeros(shape)
    amp = 1.0
    g = base_g
    tot = 0.0
    for _ in range(int(octaves)):
        val += amp * _value_noise_2d(shape, g, rng)
        tot += amp
        amp *= 2.0**(-float(H))
        g *= 2
    val -= val.mean()
    s = val.std()
    return val / s if s > 1e-9 else val


# ============================ cross-section fill ============================
def plane_cross_fill(mesh,
                     face_idx,
                     nv,
                     mid,
                     spacing,
                     fill_color,
                     rng,
                     seeds_all,
                     weights_all,
                     si,
                     sj,
                     diag,
                     tag="",
                     max_fill_points=200_000,
                     tree_seeds=None,
                     height_cfg=None,
                     color_cfg=None,
                     chip_cfg=None):
    """v9 fill + v10: power-diagram ownership, micro-chipping, shared FBM
    relief along the plane normal (faded at the intersection curve), and
    depth-based core/margin coloring. Returns (pts, cols, stats)."""

    def dbg(m):
        if tag: log(f"      {tag}: {m}")

    stats = {
        "n_raw": 0,
        "n_seeds": 0,
        "n_kept": 0,
        "n_open": 0,
        "n_closed": 0,
        "matched_open": 0,
        "matched_closed": 0,
        "fill_px": 0,
        "fill_area": 0.0,
        "interface_length": 0.0,
        "height_rms": 0.0,
        "has_fill": False,
        "reason": ""
    }
    segs, norms = plane_segments(mesh, face_idx, nv, mid)
    stats["n_raw"] = len(segs)
    if len(segs) < 2:
        stats["reason"] = f"segments={len(segs)}"
        dbg(f"NO FILL ({stats['reason']})")
        return None, None, stats
    mids = 0.5 * (segs[:, 0] + segs[:, 1])
    if weights_all is not None:
        nn = weighted_two_nearest(mids, seeds_all, weights_all)
        pair = np.array(sorted((int(si), int(sj))))
        keep = np.all(nn == pair, axis=1)
    else:
        _, nn = tree_seeds.query(mids, k=2)
        keep = np.all(nn == [si, sj], axis=1) | np.all(nn == [sj, si], axis=1)
    segs, norms = segs[keep], norms[keep]
    stats["n_seeds"] = len(segs)
    if stats["n_seeds"] < 2:
        stats["reason"] = f"seeds filter: {stats['n_raw']} -> 0"
        dbg(f"NO FILL ({stats['reason']})")
        return None, None, stats
    graze = np.abs(norms @ nv) > 0.75  # tangent grazes, not wall crossings
    segs = segs[~graze]
    stats["n_kept"] = len(segs)
    stats["interface_length"] = float(np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1).sum()) \
        if len(segs) else 0.0
    if len(segs) < 2:
        stats["reason"] = f"grazing filter: {stats['n_seeds']} -> {len(segs)}"
        dbg(f"NO FILL ({stats['reason']})")
        return None, None, stats
    u, v = make_plane_basis(nv)
    rel = segs.reshape(-1, 3) - mid
    xy = np.stack([rel @ u, rel @ v], axis=1).reshape(-1, 2, 2)
    margin = 2.0 * spacing
    xmin, xmax = xy[..., 0].min() - margin, xy[..., 0].max() + margin
    ymin, ymax = xy[..., 1].min() - margin, xy[..., 1].max() + margin
    gs = spacing
    cols = int((xmax - xmin) / gs) + 1
    rows = int((ymax - ymin) / gs) + 1
    if cols > 4000 or rows > 4000:
        scale = max(cols / 4000, rows / 4000)
        gs *= scale
        cols = int((xmax - xmin) / gs) + 1
        rows = int((ymax - ymin) / gs) + 1
    if cols < 3 or rows < 3:
        stats["reason"] = "grid degenerate"
        dbg(f"NO FILL ({stats['reason']})")
        return None, None, stats
    envelope = _raster_segments(xy, gs, xmin, ymin, rows, cols)
    polys = _chain_polylines(xy, snap=max(1e-9, diag * 1e-7))
    if not polys:
        stats["reason"] = f"chaining: {stats['n_kept']} segs -> 0 polylines"
        dbg(f"NO FILL ({stats['reason']})")
        return None, None, stats
    open_p = [p for p, c in polys if not c and len(p) >= 3]
    closed_p = [p for p, c in polys if c and len(p) >= 3]
    stats["n_open"], stats["n_closed"] = len(open_p), len(closed_p)
    fill = np.zeros((rows, cols), dtype=bool)
    n_mc = 0
    used_c = set()
    for a in range(len(closed_p)):
        for b in range(a + 1, len(closed_p)):
            if a in used_c or b in used_c: continue
            L1, L2 = closed_p[a], closed_p[b]
            if _point_in_poly(L2, L1[0]) and not _point_in_poly(L1, L2[0]):
                outer, inner = L2, L1
            elif _point_in_poly(L1, L2[0]) and not _point_in_poly(L2, L1[0]):
                outer, inner = L1, L2
            else:
                continue
            mo = _scanline_mask(outer, gs, xmin, ymin, rows, cols)
            mi = _scanline_mask(inner, gs, xmin, ymin, rows, cols)
            sep_c = float(np.median(cKDTree(inner).query(outer)[0]))
            m = (mo & ~mi) & binary_dilation(envelope,
                                             _disk(sep_c / (2 * gs) + 2))
            if m.sum() >= 16:
                fill |= m
                used_c.add(a)
                used_c.add(b)
                n_mc += 1
    for a in range(len(closed_p)):
        if a in used_c: continue
        L = closed_p[a]
        m = _scanline_mask(L, gs, xmin, ymin, rows, cols)
        m &= binary_dilation(envelope, _disk(2))
        if m.sum() >= 16:
            fill |= m
            n_mc += 1
            used_c.add(a)
    cand = []
    for a in range(len(open_p)):
        for b in range(a + 1, len(open_p)):
            A, B = open_p[a], open_p[b]
            d_ff = np.linalg.norm(A[0] - B[0]) + np.linalg.norm(A[-1] - B[-1])
            d_fr = np.linalg.norm(A[0] - B[-1]) + np.linalg.norm(A[-1] - B[0])
            dAB = cKDTree(B).query(A)[0]
            dBA = cKDTree(A).query(B)[0]
            proximity = max(float(np.percentile(dAB,
                                                90)),
                            float(np.percentile(dBA,
                                                90)))
            endpoint = min(d_ff, d_fr)
            threshold = max(8.0 * gs, 4.0 * proximity)
            if endpoint <= threshold:
                cand.append((endpoint, proximity, a, b, d_ff <= d_fr))
    cand.sort(key=lambda x: (x[0], x[1]))
    n_mo = 0
    used_o = set()
    for endpoint, proximity, a, b, ff in cand:
        if a in used_o or b in used_o: continue
        used_o.add(a)
        used_o.add(b)
        A = open_p[a]
        B = open_p[b] if ff else open_p[b][::-1]
        poly = np.concatenate([A, B[::-1]], axis=0)
        if _shoelace(poly) < 8.0 * gs * gs: continue
        m = _scanline_mask(poly, gs, xmin, ymin, rows, cols)
        m &= binary_dilation(envelope, _disk(proximity / (2 * gs) + 2))
        if m.sum() >= 16:
            fill |= m
            n_mo += 1
    # U-shape fallback: only genuine U polylines (short closing end, long arms)
    for a in range(len(open_p)):
        if a in used_o: continue
        A = open_p[a]
        if len(A) < 3: continue
        end_dist = float(np.linalg.norm(A[0] - A[-1]))
        plen = _poly_length(A)
        if plen < 1e-9 or end_dist > 0.25 * plen: continue
        if _shoelace(A) < 8.0 * gs * gs: continue
        m = _scanline_mask(A, gs, xmin, ymin, rows, cols)
        m &= binary_dilation(envelope, _disk(end_dist / (2 * gs) + 2))
        if m.sum() >= 16:
            fill |= m
            n_mo += 1
            used_o.add(a)
    stats["matched_open"], stats["matched_closed"] = n_mo, n_mc
    # micro-chipping of fracture rim (v10)
    if chip_cfg is not None and fill.any():
        edge = distance_transform_edt(fill)
        p_chip = chip_cfg["prob"] * np.exp(-edge / max(1e-9,
                                                       chip_cfg["width_px"]))
        fill = fill & ~(rng.random(fill.shape) < p_chip)
    fy, fx = np.where(fill)
    if len(fx) < 16:
        stats["reason"] = (
            f"segs={stats['n_raw']} seeds={stats['n_seeds']} kept={stats['n_kept']} "
            f"open={stats['n_open']} closed={stats['n_closed']} "
            f"matched_open={n_mo} matched_closed={n_mc} fill_px={int(fill.sum())}"
        )
        dbg(f"NO FILL ({stats['reason']})")
        return None, None, stats
    stats["fill_px"] = int(fill.sum())
    stats["fill_area"] = float(fill.sum() * gs * gs)
    dbg(f"segs={stats['n_raw']} seeds={stats['n_seeds']} kept={stats['n_kept']} "
        f"open={stats['n_open']} closed={stats['n_closed']} "
        f"matched_open={n_mo} matched_closed={n_mc} fill_px={stats['fill_px']}"
        )
    jx = rng.uniform(-0.45, 0.45, len(fx))
    jy = rng.uniform(-0.45, 0.45, len(fy))
    pts = (mid + (xmin + (fx + 0.5 + jx) * gs)[:,
                                               None] * u +
           (ymin + (fy + 0.5 + jy) * gs)[:,
                                         None] * v)
    # shared fractal relief, faded to 0 at the intersection curve (v10)
    h = np.zeros(len(fx))
    if height_cfg is not None:
        edge = distance_transform_edt(fill)
        fade = np.clip(edge[fy,
                            fx] / max(1e-9,
                                      height_cfg["fade_px"]),
                       0.0,
                       1.0)**0.7
        base_g = int(
            np.clip(
                min(rows,
                    cols) / max(4.0,
                                height_cfg["wavelength_px"]),
                2,
                64))
        field = fbm2((rows,
                      cols),
                     height_cfg["rng"],
                     height_cfg["H"],
                     height_cfg["octaves"],
                     base_g=base_g)
        grain = height_cfg["rng"].standard_normal(len(fx))
        h = (height_cfg["amp"] * field[fy,
                                       fx] +
             height_cfg["grain"] * grain) * fade
        pts = pts + h[:, None] * nv
        stats["height_rms"] = float(np.sqrt(np.mean(h**2)))
    if len(pts) > max_fill_points:
        sel = rng.choice(len(pts), max_fill_points, replace=False)
        pts, h = pts[sel], h[sel]
        fy, fx = fy[sel], fx[sel]
    # cross-section coloring (v10)
    if color_cfg is not None and color_cfg.get("model") == "depth_core":
        d_surf, _ = color_cfg["tree"].query(pts)
        depth = np.clip(d_surf / (0.5 * max(color_cfg["t_local"],
                                            1e-9)),
                        0.0,
                        1.0)
        w = np.clip((depth - 0.25) / 0.50, 0.0, 1.0)
        cols_arr = (1.0 - w)[:, None] * np.asarray(color_cfg["surface_color"], float)[None, :] \
            + w[:, None] * np.asarray(color_cfg["core_color"], float)[None, :]
        cols_arr = cols_arr * (1.0 - 0.15 * norm01(np.abs(h)))[:, None]
        cols_arr = np.clip(cols_arr, 0, 1)
    else:
        cols_arr = np.tile(np.clip(np.asarray(fill_color,
                                              float),
                                   0,
                                   1),
                           (len(pts),
                            1))
    stats["has_fill"] = True
    return pts, cols_arr, stats


# ============================ per-file processing ============================
def process_file(in_path, out_root, args):
    t0 = time.time()
    stem = in_path.stem
    out = out_root / stem
    fdir = out / "fragments"
    out.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    rng = rng_from_seed(args.seed)
    log(f"  [{stem}] loading")
    mesh = trimesh.load(str(in_path), force="scene")
    if isinstance(mesh, trimesh.Scene):
        geoms = [
            g for g in mesh.geometry.values()
            if isinstance(g, trimesh.Trimesh)
        ]
        if not geoms: raise RuntimeError("no usable mesh geometry")
        mesh = geoms[0].copy() if len(
            geoms) == 1 else trimesh.util.concatenate(geoms)
    mesh.process(validate=True)
    mesh.merge_vertices()
    if len(mesh.faces) == 0: raise RuntimeError("mesh has no faces")
    src_uv, src_image = capture_source_texture(mesh)
    colors = extract_vertex_colors(mesh, src_uv, src_image)
    center = mesh.bounds.mean(0)
    diag = bbox_diagonal(mesh)
    expl = args.explode_distance or args.explode_scale * diag
    log(f"  [{stem}] mesh V={len(mesh.vertices)} F={len(mesh.faces)} "
        f"watertight={mesh.is_watertight} texture={'yes' if src_uv is not None else 'no'}"
        )
    pc_pts, pc_cols, _fi = sample_surface_points_colors(mesh, colors, args.num_samples, rng)
    area = float(mesh.area)
    spacing = float(np.sqrt(area / max(len(pc_pts),
                                       1))) if area > 1e-18 else diag / 1000.0
    log(f"  [{stem}] complete pc: {len(pc_pts):,} pts (spacing~{spacing:.4g})")

    # stress + physical thickness (v10)
    bundle = compute_stress(mesh, args, diag)
    s_hat, thick, t_ref, curv = bundle["s_hat"], bundle["thick"], bundle["t_ref"], bundle["curv"]
    P = mesh.triangles_center
    score = s_hat.copy()
    if thick is not None and t_ref:
        score = score * thickness_gain(thick,
                                       t_ref,
                                       args.thickness_alpha,
                                       args.thickness_cap)
    noise_rng = rng_child(args.seed, 101)
    if args.noise_sigma > 0:
        base_shard = diag / max(args.fragments, 2)**(1.0 / 3.0)
        noise = bandlimited_noise(P,
                                  noise_rng,
                                  base_shard * args.noise_length_factor,
                                  n_freq=args.noise_freqs)
        score = score * np.exp(args.noise_sigma * noise)
    else:
        noise = np.zeros(len(P))
    # feature-aware bias: rim proximity, curvature, thinness (v10)
    e = np.sort(np.asarray(mesh.edges), axis=1)
    uniq_e, cnt_e = np.unique(e, axis=0, return_counts=True)
    rim_pts = np.asarray(mesh.vertices)[uniq_e[cnt_e == 1]].mean(1) if (
        cnt_e == 1).any() else None
    if rim_pts is not None and len(rim_pts):
        d_rim = cKDTree(rim_pts).query(P)[0]
        rim_bias = 1.0 + args.rim_weight * np.exp(
            -d_rim / max(1e-9,
                         args.rim_length_scale * diag))
    else:
        rim_bias = np.ones(len(P))
    curv_bias = 1.0 + args.curv_bias_weight * curv
    if thick is not None:
        t_floor = max(float(np.quantile(thick[thick > 0], 0.05)), 1e-6)
        thin_bias = 1.0 + args.thin_bias_weight * norm01(
            1.0 / np.clip(thick,
                          t_floor,
                          None))
    else:
        thin_bias = np.ones(len(P))
    score = norm01(score * rim_bias * curv_bias * thin_bias)

    # stochastic seeds (v10)
    sidx = nms_maxima(score,
                      P,
                      diag / (2.2 * max(args.fragments,
                                        2)**(1 / 3)),
                      args.fragments,
                      rng=rng,
                      radius_sigma=args.seed_radius_sigma,
                      score_noise=args.seed_score_noise)
    seeds = P[sidx]
    nS = len(seeds)
    tree_seeds = cKDTree(seeds)
    log(f"  [{stem}] seeds: {nS}")

    # power-diagram weights for log-normal shard sizes (v10)
    if args.size_sigma > 0 and nS > 1:
        base_spacing = diag / max(args.fragments, 2)**(1.0 / 3.0)
        factors = rng_child(args.seed, 202).lognormal(0.0, args.size_sigma, nS)
        factors /= factors.mean()
        weights = args.power_weight_scale * base_spacing**2 * factors
        weights = np.clip(weights, 0.0, 0.35 * base_spacing**2)
    else:
        weights = np.zeros(nS)
    pc_labels = weighted_labels(pc_pts, seeds, weights)
    face_labels = weighted_labels(P, seeds, weights)

    F = np.asarray(mesh.faces, dtype=np.int64)
    V = np.asarray(mesh.vertices, dtype=float)
    FV = V[F]
    lo_b, hi_b = mesh.bounds
    corners = np.array([[x,
                         y,
                         z] for x in (lo_b[0], hi_b[0])
                        for y in (lo_b[1], hi_b[1])
                        for z in (lo_b[2], hi_b[2])])
    boundary_faces = {}
    for i in range(nS):
        for j in range(i + 1, nS):
            dvec = seeds[j] - seeds[i]
            dd = float(np.linalg.norm(dvec))
            if dd < 1e-12: continue
            n = dvec / dd
            mid = power_mid(seeds[i], seeds[j], weights[i], weights[j])
            sdc = (corners - mid) @ n
            if sdc.min() > 1e-9 or sdc.max() < -1e-9: continue
            sd = (FV - mid) @ n
            crosses = (sd.min(axis=1) <= 1e-9) & (sd.max(axis=1) >= -1e-9)
            fi = np.flatnonzero(crosses)
            if len(fi): boundary_faces[(i, j)] = fi
    log(f"  [{stem}] geometric candidate pairs: {len(boundary_faces)}")

    fill_color = np.asarray(args.fill_color, float)
    stride = max(1, len(pc_pts) // 200_000)
    tree_surf = cKDTree(pc_pts[::stride])
    fill_dict = {}
    pair_stats = {}
    n_fill_total = 0
    for (i, j), face_idx in sorted(boundary_faces.items()):
        nv = seeds[j] - seeds[i]
        dd = float(np.linalg.norm(nv))
        if dd < 1e-12: continue
        nv /= dd
        mid = power_mid(seeds[i], seeds[j], weights[i], weights[j])
        t_local = float(np.median(thick[face_idx])) if thick is not None else \
            float(t_ref if t_ref else 4.0 * spacing)
        wall_px = max(2.0, t_local / max(spacing, 1e-9))
        height_cfg = dict(rng=rng_child(args.seed,
                                        300 + i,
                                        400 + j),
                          H=args.height_H,
                          octaves=args.height_octaves,
                          amp=args.height_amp_scale * t_local,
                          grain=args.grain_amp_scale * t_local,
                          fade_px=max(2.0,
                                      args.height_fade_fraction * wall_px),
                          wavelength_px=1.5 * wall_px)
        color_cfg = dict(model=args.fill_color_model,
                         tree=tree_surf,
                         t_local=t_local,
                         surface_color=np.asarray(args.surface_color,
                                                  float),
                         core_color=np.asarray(args.core_color,
                                               float))
        chip_cfg = dict(prob=args.chip_prob,
                        width_px=max(1.0,
                                     args.chip_width_scale))
        fp, fc, st = plane_cross_fill(mesh, face_idx, nv, mid, spacing, fill_color,
                                      rng, seeds, weights, i, j, diag,
                                      tag=f"pair {i}-{j}", tree_seeds=tree_seeds,
                                      height_cfg=height_cfg, color_cfg=color_cfg,
                                      chip_cfg=chip_cfg)
        st["t_local"] = t_local
        st["plane_normal"] = nv.tolist()
        st["plane_midpoint"] = mid.tolist()
        pair_stats[(i, j)] = st
        if fp is None: continue
        fill_dict.setdefault(i, []).append((fp, fc))
        fill_dict.setdefault(j, []).append((fp, fc))
        n_fill_total += len(fp)
    log(f"  [{stem}] fill points: {n_fill_total:,} on {len(fill_dict)} fragments"
        )

    # fragments, per-fragment palette colors (assembled) vs original colors (exploded)
    palette = fragment_palette(max(nS,
                                   1),
                               sat=args.palette_sat,
                               val=args.palette_val)
    label_faces = [np.where(face_labels == k)[0] for k in range(nS)]
    faces_all = np.asarray(mesh.faces)
    verts_all = np.asarray(mesh.vertices)
    asm_p, asm_c, exp_p, exp_c = [], [], [], []
    assembled, exploded, fragments = [], [], []
    seed_to_fid = {}

    for i in range(nS):
        idx_f = label_faces[i]
        if len(idx_f) < max(4, int(args.min_faces)): continue
        vids, inv = np.unique(faces_all[idx_f].ravel(), return_inverse=True)
        shell = trimesh.Trimesh(vertices=verts_all[vids],
                                faces=inv.reshape(-1,
                                                  3).astype(np.int64),
                                process=False)
        sub_cols = colors[vids]

        # Base shell uses original realistic colors
        shell = apply_vertex_colors(shell, sub_cols)

        centroid = shell.vertices.mean(axis=0)
        dv = centroid - center
        nn = float(np.linalg.norm(dv))
        off = (dv / nn if nn > 1e-9 else random_unit_vector(rng)) * expl

        # Exploded mesh: original realistic colors
        ex_shell = shell.copy()
        ex_shell.apply_translation(off)

        # Assembled mesh: distinct palette colors
        asm_shell = shell.copy()
        vis_cols = shade_palette(palette[i],
                                 sub_cols,
                                 args.assembly_color_mode)
        asm_shell = apply_vertex_colors(asm_shell, vis_cols)

        m = pc_labels == i
        pts, cols = pc_pts[m], pc_cols[m]
        for fp, fc in fill_dict.get(i, []):
            pts = np.vstack([pts, fp])
            cols = np.vstack([cols, fc])

        # Exploded PC: original realistic colors
        exp_p.append(pts + off)
        exp_c.append(cols)

        # Assembled PC: distinct palette colors
        vis_pc = shade_palette(palette[i], cols, args.assembly_color_mode)
        asm_p.append(pts)
        asm_c.append(vis_pc)

        assembled.append(asm_shell)
        exploded.append(ex_shell)

        fid = f"frag_{len(fragments):03d}"
        seed_to_fid[i] = fid
        fragments.append(
            Fragment(fid=fid,
                     seed_index=i,
                     shell=shell,
                     exploded_shell=ex_shell,
                     centroid=centroid,
                     explode_offset=off,
                     n_faces=len(shell.faces),
                     area=float(shell.area),
                     n_points=int(len(pts)),
                     palette_rgb=palette[i],
                     vert_colors=sub_cols))

    def save_pc(pts, cols, path):
        rgba = np.column_stack([(np.clip(cols,
                                         0,
                                         1) * 255).astype(np.uint8),
                                np.full((len(cols),
                                         1),
                                        255,
                                        dtype=np.uint8)])
        trimesh.PointCloud(np.asarray(pts,
                                      float),
                           colors=rgba).export(str(path))

    def rel(p):
        return str(Path(p).relative_to(out))

    files = {}
    save_pc(pc_pts, pc_cols, out / "complete.ply")
    files["complete_pc"] = rel(out / "complete.ply")
    save_pc(np.vstack(asm_p), np.vstack(asm_c), out / "assembled.ply")
    files["assembled_pc"] = rel(out / "assembled.ply")
    save_pc(np.vstack(exp_p), np.vstack(exp_c), out / "exploded.ply")
    files["exploded_pc"] = rel(out / "exploded.ply")
    for f in fragments:
        m = pc_labels == f.seed_index
        pts, cols = pc_pts[m], pc_cols[m]  # realistic colors for training
        for fp, fc in fill_dict.get(f.seed_index, []):
            pts = np.vstack([pts, fp])
            cols = np.vstack([cols, fc])
        pp = fdir / f"{f.fid}.ply"
        save_pc(pts, cols, pp)
        f.ply = rel(pp)
    trimesh.Scene(assembled).export(str(out / "assembled.glb"))
    files["assembled_mesh"] = rel(out / "assembled.glb")
    trimesh.Scene(exploded).export(str(out / "exploded.glb"))
    files["exploded_mesh"] = rel(out / "exploded.glb")
    stress_textured_mesh(mesh, s_hat).export(str(out / "stress_colored.glb"))
    files["stress_mesh"] = rel(out / "stress_colored.glb")

    # adjacency / join-relationship data for matching+reconstruction models
    nodes = []
    for f in fragments:
        nodes.append({
            "id": f.fid,
            "seed_index": int(f.seed_index),
            "centroid": f.centroid.tolist(),
            "explode_offset": f.explode_offset.tolist(),
            "assembled_transform": {
                "rotation": np.eye(3).tolist(),
                "translation": [0.0,
                                0.0,
                                0.0]
            },
            "exploded_transform": {
                "rotation": np.eye(3).tolist(),
                "translation": f.explode_offset.tolist()
            },
            "n_faces": int(f.n_faces),
            "area": float(f.area),
            "n_points": int(f.n_points),
            "palette_rgb": np.round(f.palette_rgb,
                                    4).tolist(),
            "ply": f.ply
        })
    node_by_seed = {f.seed_index: f for f in fragments}
    edges = []
    neighbors = defaultdict(list)
    for (i, j), st in sorted(pair_stats.items()):
        if st["n_kept"] < 1: continue  # no visible shared facet
        if i not in node_by_seed or j not in node_by_seed: continue
        fi, fj = node_by_seed[i], node_by_seed[j]
        edges.append({
            "a":
            fi.fid,
            "b":
            fj.fid,
            "seed_pair": [int(i),
                          int(j)],
            "plane_normal":
            st["plane_normal"],
            "plane_midpoint":
            st["plane_midpoint"],
            "interface_segments":
            int(st["n_kept"]),
            "interface_length":
            float(st["interface_length"]),
            "fill_points":
            int(st["fill_px"]) if st["has_fill"] else 0,
            "fill_area":
            float(st["fill_area"]) if st["has_fill"] else 0.0,
            "mean_wall_thickness":
            float(st["t_local"]),
            "height_rms":
            float(st["height_rms"]),
            "has_fill":
            bool(st["has_fill"]),
            "rel_rotation":
            np.eye(3).tolist(),
            "rel_translation_exploded":
            (fj.explode_offset - fi.explode_offset).tolist()
        })
        neighbors[fi.fid].append(fj.fid)
        neighbors[fj.fid].append(fi.fid)
    adjacency = {
        "object": stem,
        "n_fragments": len(fragments),
        "n_edges": len(edges),
        "nodes": nodes,
        "edges": edges
    }
    (out / "adjacency.json").write_text(
        json.dumps(adjacency,
                   indent=2,
                   default=str))
    files["adjacency"] = rel(out / "adjacency.json")

    areas = np.array([f.area for f in fragments]) if fragments else np.zeros(0)
    meta = {
        "input":
        str(in_path),
        "pipeline":
        "plane_cut_v10_adjacency_palette_relief",
        "seed":
        int(args.seed),
        "n_fragments":
        len(fragments),
        "point_spacing":
        spacing,
        "fill_points":
        int(n_fill_total),
        "thickness": ({
            "median": float(t_ref),
            "p10": float(np.quantile(thick[thick > 0],
                                     0.10)),
            "p90": float(np.quantile(thick[thick > 0],
                                     0.90))
        } if thick is not None and t_ref else None),
        "fragment_areas": ({
            "mean": float(areas.mean()),
            "std": float(areas.std()),
            "cv": float(areas.std() / max(areas.mean(),
                                          1e-12))
        } if len(areas) else None),
        "args":
        vars(args),
        "files":
        files,
        "fragments": [{
            "id": f.fid,
            "ply": f.ply,
            "n_faces": f.n_faces,
            "area": f.area,
            "n_points": f.n_points,
            "palette_rgb": np.round(f.palette_rgb,
                                    4).tolist(),
            "neighbors": neighbors.get(f.fid,
                                       [])
        } for f in fragments]
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    log(f"  [{stem}] done in {time.time()-t0:.1f}s -> {out} "
        f"(fragments={len(fragments)}, adjacency edges={len(edges)})")
    return {
        "input": str(in_path),
        "output": str(out),
        "n_fragments": len(fragments),
        "adjacency_edges": len(edges),
        "status": "ok"
    }


# ============================ CLI ============================
def build_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", "-i", default="pottery", type=Path)
    p.add_argument("--output",
                   "-o",
                   default=r"D:\storage\jomon_kaen\fragmented_plane_cut",
                   type=Path)
    g = p.add_argument_group("point cloud")
    g.add_argument("--num-samples", type=int, default=2_000_000)
    g.add_argument("--fill-color", type=float, nargs=3, default=CLAY.tolist())
    g = p.add_argument_group("stress")
    g.add_argument(
        "--stress-model",
        choices=["fem_lite",
                 "impact",
                 "gravity",
                 "curvature",
                 "none"],
        default="fem_lite")
    g.add_argument("--grid-res", type=int, default=64)
    g.add_argument("--jacobi-iters", type=int, default=150)
    g.add_argument("--impact-dir",
                   type=float,
                   nargs=3,
                   default=[1.0,
                            0.0,
                            0.35])
    g.add_argument("--w-vol", type=float, default=1.0)
    g.add_argument("--w-curv", type=float, default=0.6)
    g.add_argument("--w-thin", type=float, default=0.8)
    g.add_argument("--thickness-alpha",
                   type=float,
                   default=1.5,
                   help="stress gain (t_ref/t)^alpha")
    g.add_argument("--thickness-cap", type=float, default=5.0)
    g.add_argument("--noise-sigma",
                   type=float,
                   default=0.5,
                   help="stochastic stress multiplier sigma")
    g.add_argument("--noise-freqs", type=int, default=48)
    g.add_argument("--noise-length-factor", type=float, default=0.33)
    g = p.add_argument_group("seeds / sizes")
    g.add_argument("--seed-radius-sigma", type=float, default=0.35)
    g.add_argument("--seed-score-noise", type=float, default=0.05)
    g.add_argument("--rim-weight", type=float, default=0.6)
    g.add_argument("--rim-length-scale", type=float, default=0.08)
    g.add_argument("--curv-bias-weight", type=float, default=0.4)
    g.add_argument("--thin-bias-weight", type=float, default=0.5)
    g.add_argument("--size-sigma",
                   type=float,
                   default=0.6,
                   help="log-normal size sigma (0=uniform)")
    g.add_argument("--power-weight-scale", type=float, default=0.15)
    g = p.add_argument_group("fracture surface")
    g.add_argument("--height-amp-scale", type=float, default=0.18)
    g.add_argument("--height-H", type=float, default=0.78)
    g.add_argument("--height-octaves", type=int, default=5)
    g.add_argument("--grain-amp-scale", type=float, default=0.03)
    g.add_argument("--height-fade-fraction", type=float, default=0.35)
    g.add_argument("--chip-prob", type=float, default=0.15)
    g.add_argument("--chip-width-scale", type=float, default=1.5)
    g.add_argument("--fill-color-model",
                   choices=["constant",
                            "depth_core"],
                   default="depth_core")
    g.add_argument("--surface-color",
                   type=float,
                   nargs=3,
                   default=CLAY.tolist())
    g.add_argument("--core-color", type=float, nargs=3, default=CORE.tolist())
    g = p.add_argument_group("assembly visuals")
    g.add_argument("--assembly-color-mode",
                   choices=["shaded",
                            "flat"],
                   default="shaded")
    g.add_argument("--palette-sat", type=float, default=0.75)
    g.add_argument("--palette-val", type=float, default=0.95)
    g = p.add_argument_group("fragmentation")
    g.add_argument("--fragments", "-n", type=int, default=20)
    g.add_argument("--explode-distance", type=float, default=None)
    g.add_argument("--explode-scale", type=float, default=0.12)
    g.add_argument("--min-faces", type=int, default=50)
    g.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.input = Path(args.input).expanduser()
    args.output = Path(args.output).expanduser()
    if args.seed is None:  # reproducible by default
        args.seed = int(np.random.default_rng().integers(1, 2**31 - 1))
    log(f"run seed = {args.seed}")
    files = ([args.input] if args.input.is_file() else sorted(
        p for p in args.input.rglob("*")
        if p.suffix.lower() in {".glb", ".gltf", ".obj", ".ply", ".stl"}))
    if not files:
        log("no inputs found")
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    sums = []
    for k, f in enumerate(files, 1):
        log(f"=== ({k}/{len(files)}) {f.name} ===")
        try:
            sums.append(process_file(f, args.output, args))
        except Exception as exc:
            log(f"  ! FAILED {f.name}: {exc}")
            traceback.print_exc()
            sums.append({
                "input": str(f),
                "n_fragments": 0,
                "status": f"error: {exc}"
            })
    (args.output / "run_summary.json").write_text(
        json.dumps(sums,
                   indent=2,
                   default=str))
    return


if __name__ == "__main__":
    sys.exit(main())
