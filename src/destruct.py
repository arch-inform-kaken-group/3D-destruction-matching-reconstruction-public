"""
physics_based_destruction.py — simplified physics-based fragmentation pipeline for pottery.

Single-file rewrite of the previous destruct.py / viz.py pipeline.

PHYSICS CORE (simplified adaptation of Sellán et al. 2022, "Fracture Modes for
Realtime Destruction", arXiv:2111.05249):
  * The shape is voxelized; voxels form a 6-neighbour graph (our "exploded mesh"
    analogue: graph edges = admissible fracture faults S_i).
  * Fracture modes = lowest-energy piecewise-rigid displacement fields, i.e. the
    sparsified eigenproblem  min  ω·Σ_e η_e‖Du_e‖₂  s.t.  UᵀMU = I  (Eq. 7/15 of the
    paper). The conic solve (Mosek) is replaced by a much simpler Iteratively
    Reweighted Least Squares (IRLS) loop around a constrained sparse eigenproblem
    (ℓ₂,₁ sparsity ≈ reweighted graph-TV); η_e encodes material weakness (thin walls
    break first, cf. anisotropic η in Eq. 26).
  * Prefracture pattern = intersecting the first k modes (thresholding the
    discontinuity field, union of fault sets → connected components).
  * Impact-dependent fracture (§3.3): an impact gaussian is projected onto the
    precomputed modes, w* = Σ U_i U_iᵀ M w, and edges with |w*_a − w*_b| > σ are cut.

ADDED ARCHAEOLOGICAL REALISM (the "main additions"):
  (1) FRACTURE-SURFACE RELIEF: open break surfaces (voxel interface quads) are filled
      with points displaced by a SWAPPABLE noise field (terrain generation).
      Registry: fbm | perlin | worley | ridge | strata | white | hybrid.
      Swap via --relief-noise, or programmatically via register_noise(name, fn).
  (2) WEATHERING: spatially-coherent HSV modification (hue drift, desaturation,
      bleaching, dark staining, mineral deposits) simulating burial time.
  (3) GAPS & CHIPS: each fragment loses an INDEPENDENT random rim band (per-fragment
      noise field), so adjacent unearthed fragments no longer have matching contours
      (Fig. 3 of the reference image); plus stochastic micro-chip bites at rims.

OUTPUTS (per input mesh, under --output/<stem>/):
  complete.ply          pristine sampled point cloud (ground truth surface)
  assembled.ply/.glb    reassembled sherds, palette colours, weathered+gaps
  exploded.ply/.glb     exploded sherds, realistic weathered colours
  fragments/frag_*.ply  individual unearthed sherds (weathered, gaps, chips, fills)
  fracture_affinity.ply surface coloured by the modes' discontinuity field
  adjacency.json        join-relationship graph (nodes/edges, gap statistics)
  metadata.json         full provenance

Usage:
  python physics_based_destruction.py -i pottery/mesh.glb -o out/ -n 16
  python physics_based_destruction.py -i pottery/ --relief-noise worley --weather 0.8
  python physics_based_destruction.py -i m.glb --fracture impact --impact-pos 1 0 2

Dependencies: pip install numpy scipy trimesh pillow
"""
from __future__ import annotations

import argparse, json, sys, time, traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import trimesh
import trimesh.visual
import trimesh.visual.color
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, eigsh

try:
    from PIL import Image
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

# ------------------------------------------------------------------ constants
DEFAULT_GRAY = 0.80
CLAY = np.array([0.75, 0.51, 0.31])          # oxidized surface clay
CORE = np.array([0.62, 0.42, 0.28])          # reduced core (cross-sections)
LUM = np.array([0.2126, 0.7152, 0.0722])
DEPOSIT_COL = np.array([0.92, 0.90, 0.82])   # calcite-like mineral deposit


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def rng_from_seed(s): return np.random.default_rng(None if s is None else int(s))
def rng_child(seed, *tags):
    return np.random.default_rng(np.random.SeedSequence([int(seed)] + [int(t) for t in tags]))
def bbox_diagonal(m):
    lo, hi = m.bounds
    return float(np.linalg.norm(hi - lo))
def norm01(x):
    x = np.asarray(x, float)
    m = x.max()
    return x / m if m > 1e-12 else np.zeros_like(x)


# ============================================================ swappable noise
# Every noise is a callable  f(q:(N,3) coords already divided by wavelength,
#                            rng, H, octaves) -> (N,) zero-mean unit-std field.
def _periodic_table(rng, P, shape):
    return rng.random((P, P, P, *shape))


def _value3(q, tbl):
    P = tbl.shape[0]
    f = np.floor(q); c = q - f
    i = f.astype(np.int64) % P
    c = c * c * (3 - 2 * c)
    acc = np.zeros(len(q))
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (c[:, 0] if dx else 1 - c[:, 0]) * \
                    (c[:, 1] if dy else 1 - c[:, 1]) * \
                    (c[:, 2] if dz else 1 - c[:, 2])
                acc += w * tbl[(i[:, 0] + dx) % P, (i[:, 1] + dy) % P, (i[:, 2] + dz) % P]
    return acc


def noise_fbm(q, rng, H=0.75, octaves=5):
    val, amp, f = np.zeros(len(q)), 1.0, 1.0
    for o in range(int(octaves)):
        val += amp * _value3(q * f + 13.7 * o, _periodic_table(rng, 64, ()))
        amp *= 2.0 ** (-float(H)); f *= 2.0
    return val


def noise_perlin(q, rng, H=0.75, octaves=5):
    P = 64
    g = rng.standard_normal((P, P, P, 3))
    g /= np.linalg.norm(g, axis=3, keepdims=True) + 1e-12
    val, amp, f = np.zeros(len(q)), 1.0, 1.0
    for _ in range(int(octaves)):
        qq = q * f
        i = np.floor(qq).astype(np.int64); c = qq - i
        u = c * c * c * (c * (c * 6 - 15) + 10)
        acc = np.zeros(len(qq))
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    gi = g[(i[:, 0] + dx) % P, (i[:, 1] + dy) % P, (i[:, 2] + dz) % P]
                    off = np.stack([c[:, 0] - dx, c[:, 1] - dy, c[:, 2] - dz], 1)
                    d = (gi * off).sum(1)
                    w = (u[:, 0] if dx else 1 - u[:, 0]) * \
                        (u[:, 1] if dy else 1 - u[:, 1]) * \
                        (u[:, 2] if dz else 1 - u[:, 2])
                    acc += w * d
        val += amp * acc
        amp *= 2.0 ** (-float(H)); f *= 2.0
    return val


def noise_worley(q, rng, H=0.75, octaves=3, k=192):
    lo, hi = q.min(0), q.max(0)
    feat = rng.uniform(lo - 1, hi + 1, (int(k), 3))
    d = cKDTree(feat).query(q, k=2)[0]
    return -(d[:, 1] - d[:, 0])          # cellular walls


def noise_ridge(q, rng, H=0.75, octaves=5):
    val, amp, f, w = np.zeros(len(q)), 1.0, 1.0, 1.0
    for _ in range(int(octaves)):
        n = 1.0 - np.abs(noise_perlin(q * f, rng, H, 1))
        n = n * n
        val += w * n
        w = np.clip(w * n, 0.0, 1.0)     # multifractal weight
        amp *= 2.0 ** (-float(H)); f *= 2.0
    return val


def noise_strata(q, rng, H=0.75, octaves=4):
    warp = noise_fbm(q, rng, H, octaves)
    return np.sin(2.0 * np.pi * (q[:, 2] + 0.6 * warp))   # sedimentary bands


def noise_white(q, rng, H=0.75, octaves=1):
    return rng.standard_normal(len(q))


def noise_hybrid(q, rng, H=0.75, octaves=5):
    a = norm01(noise_fbm(q, rng, H, octaves))
    b = norm01(noise_worley(q, rng, H, 3))
    return 0.65 * a + 0.35 * b


NOISE_REGISTRY: Dict[str, Callable] = {
    "fbm": noise_fbm, "perlin": noise_perlin, "worley": noise_worley,
    "ridge": noise_ridge, "strata": noise_strata, "white": noise_white,
    "hybrid": noise_hybrid,
}


def register_noise(name: str, fn: Callable):
    """Plug in a custom fracture-surface terrain generator (see module docstring)."""
    NOISE_REGISTRY[name] = fn


def relief_field(kind, pts, wavelength, rng, H, octaves):
    fn = NOISE_REGISTRY[kind]
    v = fn(np.asarray(pts, float) / max(wavelength, 1e-9), rng, H, octaves)
    v = np.nan_to_num(v)
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else v


# ============================================================ colour helpers
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
    if a.ndim != 2 or a.shape[0] != n:
        if a.shape[1] == n: a = a.T
        else: return None
    a = a[:, :3] if a.shape[1] >= 3 else np.repeat(a, 3, axis=1)
    a = a.astype(np.float64, copy=False)
    try:
        if np.nanmax(a) > 1.5: a = a / 255.0
    except Exception:
        pass
    return np.clip(np.nan_to_num(a, nan=DEFAULT_GRAY), 0.0, 1.0)


def _bake_texture_colors(uv, image, n):
    if not HAVE_PIL: return None
    try:
        arr = np.asarray(image.convert("RGB"), float) / 255.0
        h, w = arr.shape[:2]
        if h < 2 or w < 2 or len(uv) != n: return None
        u = np.clip(uv[:, 0] % 1.0, 0, 1) * (w - 1)
        v = np.clip(uv[:, 1] % 1.0, 0, 1) * (h - 1)
        x0, y0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
        fx, fy = (u - x0)[:, None], (v - y0)[:, None]
        return np.clip(arr[y0, x0] * (1 - fx) * (1 - fy) + arr[y0, x1] * fx * (1 - fy) +
                       arr[y1, x0] * (1 - fx) * fy + arr[y1, x1] * fx * fy, 0, 1)
    except Exception:
        return None


def extract_vertex_colors(mesh):
    n = len(mesh.vertices)
    vis = getattr(mesh, "visual", None)
    cands = []
    if vis is not None:
        try:
            c = normalize_colors(vis.to_color().vertex_colors, n)
            if c is not None and c.std(axis=0).mean() >= 1e-4: cands.append(c)
        except Exception:
            pass
        if isinstance(vis, trimesh.visual.TextureVisuals):
            uv = np.asarray(getattr(vis, "uv", None), float)
            img = getattr(vis.material, "image", None)
            if uv is not None and uv.ndim == 2 and uv.shape[0] == n and img is not None:
                c = _bake_texture_colors(uv, img, n)
                if c is not None and c.std(axis=0).mean() >= 1e-4: cands.append(c)
    return cands[0] if cands else np.full((n, 3), DEFAULT_GRAY)


def apply_vertex_colors(mesh, rgb01):
    rgb01 = normalize_colors(rgb01, len(mesh.vertices))
    rgba = trimesh.visual.color.to_rgba((np.clip(rgb01, 0, 1) * 255).astype(np.uint8))
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba)
    return mesh


def rgb_to_hsv(c):
    c = np.clip(c, 0, 1)
    mx, mn = c.max(1), c.min(1)
    df = mx - mn
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    h = np.where(mx == r, ((g - b) / np.maximum(df, 1e-12)) % 6,
                 np.where(mx == g, (b - r) / np.maximum(df, 1e-12) + 2,
                          (r - g) / np.maximum(df, 1e-12) + 4)) / 6.0
    h = np.where(df < 1e-9, 0.0, h) % 1.0
    s = np.where(mx > 0, df / np.maximum(mx, 1e-12), 0.0)
    return np.stack([h, s, mx], 1)


def hsv_to_rgb(hsv):
    h, s, v = hsv[:, 0] % 1.0, hsv[:, 1], hsv[:, 2]
    i = (h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])
    return np.clip(np.stack([r, g, b], 1), 0, 1)


@dataclass
class WeatherCfg:
    strength: float = 0.6        # master 0..1
    hue: float = 0.03            # spatially-varying hue drift (turns)
    desat: float = 0.45          # saturation loss
    stain: float = 0.35          # dark humic/manganese staining
    bleach: float = 0.25         # sun/alkali bleaching toward pale
    deposit_cover: float = 0.12  # fraction covered by mineral deposits
    deposit_amp: float = 0.55


def weather_colors(cols, pts, rng, w: WeatherCfg, wavelength, factor=1.0):
    """Burial weathering: coherent HSV modification + deposits (addition #2)."""
    k = w.strength * factor
    if k <= 1e-6: return cols
    n_h = relief_field("fbm", pts, wavelength * 2.0, rng, 0.85, 3)
    n_s = relief_field("fbm", pts + 31.7, wavelength * 1.3, rng, 0.85, 3)
    n_v = relief_field("fbm", pts + 77.3, wavelength * 0.9, rng, 0.85, 4)
    hsv = rgb_to_hsv(cols)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    h = (h + k * w.hue * n_h) % 1.0
    s = np.clip(s * (1 - k * w.desat * (0.5 + 0.5 * norm01(n_s))), 0, 1)
    v = np.clip(v * (1 - k * w.stain * np.clip(n_v, 0, None) * 0.9)
                + k * w.bleach * np.clip(n_s, 0, None) * (0.92 - v) * 0.5, 0, 1)
    out = hsv_to_rgb(np.stack([h, s, v], 1))
    if w.deposit_cover > 0:
        mask = norm01(n_v) > (1.0 - w.deposit_cover)
        out = np.where(mask[:, None],
                       out * (1 - k * w.deposit_amp) + DEPOSIT_COL * (k * w.deposit_amp),
                       out)
    return np.clip(out, 0, 1)


# ============================================================ palette
def fragment_palette(n, sat=0.75, val=0.95):
    n = max(1, int(n))
    h = (np.arange(n) * 0.618033988749895) % 1.0
    i = (h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = val * (1 - sat), val * (1 - f * sat), val * (1 - (1 - f) * sat)
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [val, q, p, p, t, val])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, val, val, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, val, val, q])
    return np.clip(np.stack([r, g, b], 1), 0, 1)


def shade_palette(pal_rgb, base_cols, mode):
    base_cols = np.asarray(base_cols, float)
    if mode == "flat" or base_cols.size == 0:
        return np.tile(np.asarray(pal_rgb, float), (max(len(base_cols), 0), 1))
    f = 0.35 + 0.65 * np.clip(base_cols @ LUM, 0, 1)
    return np.clip(np.asarray(pal_rgb, float)[None, :] * f[:, None], 0, 1)


# ============================================================ sampling
def sample_surface_points_colors(mesh, colors, n, rng):
    n = int(max(0, n))
    if n == 0 or len(mesh.faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.int64)
    areas = np.asarray(mesh.area_faces, float)
    tot = areas.sum()
    p = areas / tot if tot > 1e-18 else np.full(len(mesh.faces), 1 / len(mesh.faces))
    fi = rng.choice(len(mesh.faces), size=n, p=p)
    r = rng.random((n, 2))
    over = r.sum(1) > 1.0
    r[over] = 1.0 - r[over]
    b = np.empty((n, 3)); b[:, 1:] = r; b[:, 0] = 1.0 - r.sum(1)
    tri = mesh.triangles[fi]
    col = colors[mesh.faces[fi]]
    return (np.einsum("ij,ijk->ik", b, tri),
            np.clip(np.einsum("ij,ijk->ik", b, col), 0, 1),
            np.asarray(fi, np.int64))


# ============================================== voxel graph + fracture modes
def build_voxel_graph(mesh, res):
    diag = bbox_diagonal(mesh)
    vox = mesh.voxelized(pitch=diag / max(int(res), 16))
    occ = np.asarray(vox.matrix, bool)
    T = vox.transform
    idx = np.argwhere(occ)
    lab = np.full(occ.shape, -1, np.int64)
    lab[tuple(idx.T)] = np.arange(len(idx))
    ea, eb, eax = [], [], []
    for ax in range(3):
        nb = idx.copy(); nb[:, ax] += 1
        ok = (nb[:, ax] < occ.shape[ax])
        a = lab[tuple(idx[ok].T)]; b = lab[tuple(nb[ok].T)]
        m = (a >= 0) & (b >= 0)
        ea.append(a[m]); eb.append(b[m]); eax.append(np.full(m.sum(), ax))
    E = np.stack([np.concatenate(ea), np.concatenate(eb)], 1) if ea else np.zeros((0, 2), int)
    pitch = float(np.abs(np.linalg.det(T[:3, :3])) ** (1.0 / 3.0))
    centers = idx @ T[:3, :3].T + T[:3, 3]
    strength = distance_transform_edt(occ)[tuple(idx.T)] * pitch   # local half-wall
    return centers, E, np.concatenate(eax), pitch, strength


def fracture_modes(centers, E, eta, k, irls_iters, rng, tol=1e-4):
    """Simplified ICCM (paper Alg. 1): IRLS-reweighted constrained spectral solves.
    Returns (modes U_i, per-mode edge discontinuities J_i)."""
    M = len(centers)
    if len(E) == 0 or M < 8:
        return [], []
    Mv = np.ones(M)
    a, b = E[:, 0], E[:, 1]
    def lap(w):
        off = np.concatenate([-w, -w])
        ri = np.concatenate([a, b]); ci = np.concatenate([b, a])
        L = coo_matrix((off, (ri, ci)), shape=(M, M)).tocsr()
        d = np.zeros(M); np.add.at(d, a, w); np.add.at(d, b, w)
        return L + diags(d)
    C = [np.full(M, 1.0 / np.sqrt(M))]
    Us, Js = [], []
    for _i in range(int(k)):
        w = eta.copy()
        u, jumps = None, None
        for _it in range(max(1, int(irls_iters))):
            L = lap(w)
            Cm = np.column_stack(C)
            B = L @ Cm                      # (M,c)
            D = Cm.T * Mv[None, :]          # (c,M)
            MC = Cm * Mv[:, None]
            E2 = Cm.T @ B
            alpha = 10.0 * float(np.asarray(L.diagonal()).max())
            def mv(v, L=L, B=B, D=D, MC=MC, E2=E2, Cm=Cm, alpha=alpha):
                Dv = D @ v
                return (L @ v - B @ Dv - MC @ (B.T @ v)
                        + MC @ (E2 @ Dv) + alpha * MC @ (Cm.T @ (Mv * v)))
            op = LinearOperator((M, M), matvec=mv, dtype=float)
            try:
                _, vecs = eigsh(op, k=1, which="SA", M=diags(Mv), tol=tol,
                                maxiter=300, v0=rng.standard_normal(M))
                y = vecs[:, 0]
            except Exception:
                y = rng.standard_normal(M)
            u = y - Cm @ (Cm.T @ (Mv * y))          # M-orthogonalise vs previous modes
            nrm = float(np.sqrt(u @ (Mv * u)))
            if nrm < 1e-10:
                u = None
                break
            u /= nrm
            jumps = np.abs(u[a] - u[b])
            w = eta / (jumps + 1e-3)                # ℓ2,1 sparsity reweighting
        if u is None:
            continue
        Us.append(u); Js.append(jumps); C.append(u)
    return Us, Js


def _components(M, Ekeep):
    if len(Ekeep) == 0:
        return np.arange(M), M
    A = coo_matrix((np.ones(len(Ekeep)), (Ekeep[:, 0], Ekeep[:, 1])), shape=(M, M))
    n, comp = connected_components(A, directed=False)
    return comp, n


def _merge(comp, ncomp, E, sizes_min, target):
    parent = np.arange(ncomp)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    sizes = np.bincount(comp, minlength=ncomp)
    cross = E[comp[E[:, 0]] != comp[E[:, 1]]]
    if len(cross):
        pl = np.stack([comp[cross[:, 0]], comp[cross[:, 1]]], 1)
        pl = np.sort(pl, 1)
        uniq, cnt = np.unique(pl, axis=0, return_counts=True)
        adj = defaultdict(list)
        for (x, y), c in zip(uniq, cnt):
            adj[x].append((c, y)); adj[y].append((c, x))
    else:
        adj = defaultdict(list)
    for c in np.argsort(sizes):
        if sizes[c] >= sizes_min:
            break
        rc = find(c)
        if not adj[rc]:
            continue
        _, nb = max(adj[rc])
        parent[rc] = find(nb)
    roots = np.array([find(c) for c in range(ncomp)])
    comp = roots[comp]
    _, comp = np.unique(comp, return_inverse=True)
    ncomp = int(comp.max()) + 1
    sizes = np.bincount(comp, minlength=ncomp)
    while ncomp > target:
        c = int(np.argmin(sizes))
        cross = E[comp[E[:, 0]] != comp[E[:, 1]]]
        if len(cross) == 0:
            break
        pl = np.sort(np.stack([comp[cross[:, 0]], comp[cross[:, 1]]], 1), 1)
        uniq, cnt = np.unique(pl, axis=0, return_counts=True)
        best, bc = None, -1
        for (x, y), cc in zip(uniq, cnt):
            if x == c or y == c:
                if cc > bc: bc, best = cc, (y if x == c else x)
        if best is None:
            break
        comp = np.where(comp == best, c, comp)
        _, comp = np.unique(comp, return_inverse=True)
        ncomp = int(comp.max()) + 1
        sizes = np.bincount(comp, minlength=ncomp)
    return comp, ncomp


def prefracture(centers, E, Js, target, min_vox, rng):
    """Intersect first-k mode fault sets (paper §3): threshold + components."""
    M = len(centers)
    d = np.max([norm01(j) for j in Js], axis=0) if Js else np.zeros(len(E))
    lo, hi, best = 0.30, 0.999, None
    for _ in range(22):                       # bisect threshold → ~target fragments
        q = 0.5 * (lo + hi)
        thr = np.quantile(d, q) if len(d) else 1.0
        comp, n = _components(M, E[d <= thr])
        err = abs(n - target)
        if best is None or err < best[0]:
            best = (err, q, comp, n)
        if n > target: lo = q
        else: hi = q
    _, q, comp, _ = best
    comp, n = _merge(comp, int(comp.max()) + 1, E, max(min_vox, 4), target)
    sc = np.zeros(M)
    if len(d):
        np.maximum.at(sc, E[:, 0], d); np.maximum.at(sc, E[:, 1], d)
    return comp, d, sc


def impact_fracture(centers, E, Us, pos, radius, sigma):
    """Paper §3.3: project impact onto modes, cut where ‖Δw*‖ > σ."""
    M = len(centers)
    w = np.exp(-(np.linalg.norm(centers - np.asarray(pos, float), axis=1) /
                 max(radius, 1e-9)) ** 2)
    Um = np.stack(Us).T if Us else np.zeros((M, 0))
    ws = Um @ (Um.T @ w) if Um.shape[1] else np.zeros(M)
    jw = np.abs(ws[E[:, 0]] - ws[E[:, 1]]) if len(E) else np.zeros(0)
    keep = jw <= sigma * max(jw.max(), 1e-12) if len(jw) else np.zeros(0, bool)
    comp, n = _components(M, E[keep])
    sc = norm01(jw)
    d = np.zeros(len(E)); np.maximum.at(d, E[:, 0], sc); np.maximum.at(d, E[:, 1], sc)
    return comp, d, sc


# ============================================================ interface fill
def interface_fill(comp, E, eax, centers, pitch, strength, spacing, tree_surf,
                   noise_kind, hcfg, ccfg, rng):
    """Fill open break surfaces (addition #1): sample voxel interface quads,
    displace with swappable relief noise, colour by depth-core model."""
    a, b = E[:, 0], E[:, 1]
    cut = comp[a] != comp[b]
    pairs = {}
    if cut.any():
        pl = np.sort(np.stack([comp[a[cut]], comp[b[cut]]], 1), 1)
        eidx = np.flatnonzero(cut)
        for (ci, cj), grp in zip(*[np.unique(pl, axis=0, return_inverse=True)[:2][::-1][::-1],
                                   ] if False else [(u, np.flatnonzero((pl == u).all(1)))
                                                    for u in np.unique(pl, axis=0)]):
            ee = eidx[grp]
            mids = 0.5 * (centers[a[ee]] + centers[b[ee]])
            ax = eax[ee]
            sgn = np.sign(centers[b[ee], ax] - centers[a[ee], ax])
            t_loc = 2.0 * float(np.median(strength[a[ee]])) if len(ee) else 4 * spacing
            pairs[(int(ci), int(cj))] = dict(mids=mids, ax=ax, sgn=sgn, t_local=t_loc)
    fills = {}
    for (ci, cj), pd in pairs.items():
        Q = len(pd["mids"])
        if Q < 1:
            continue
        s = int(np.clip((pitch / max(spacing, 1e-9)) ** 2, 1, 64))
        u_ax, v_ax = (pd["ax"] + 1) % 3, (pd["ax"] + 2) % 3
        off = rng.uniform(-0.5, 0.5, (Q, s, 2)) * pitch
        pts = np.repeat(pd["mids"][:, None, :], s, axis=1).copy()
        pts[..., u_ax] += off[..., 0]
        pts[..., v_ax] += off[..., 1]
        nrm = np.zeros((Q, s, 3)); nrm[..., pd["ax"]] = pd["sgn"][:, None]
        pts = pts.reshape(-1, 3); nrm = nrm.reshape(-1, 3)
        if len(pts) > 300_000:
            sel = rng.choice(len(pts), 300_000, replace=False)
            pts, nrm = pts[sel], nrm[sel]
        d_surf = tree_surf.query(pts)[0]
        wall_px = max(2.0, pd["t_local"] / max(spacing, 1e-9))
        fade = np.clip(d_surf / max(1e-9, hcfg["fade_fraction"] * pd["t_local"]), 0, 1) ** 0.7
        field = relief_field(noise_kind, pts, hcfg["wavelength_scale"] * pd["t_local"],
                             rng, hcfg["H"], hcfg["octaves"])
        h = (hcfg["amp_scale"] * pd["t_local"] * field +
             hcfg["grain_scale"] * pd["t_local"] * rng.standard_normal(len(pts))) * fade
        pts = pts + h[:, None] * nrm
        if ccfg["model"] == "depth_core":
            depth = np.clip(d_surf / (0.5 * max(pd["t_local"], 1e-9)), 0, 1)
            wgt = np.clip((depth - 0.25) / 0.50, 0, 1)
            cols = ((1 - wgt)[:, None] * ccfg["surface_color"][None, :] +
                    wgt[:, None] * ccfg["core_color"][None, :])
            cols = cols * (1 - 0.15 * norm01(np.abs(h)))[:, None]
        else:
            cols = np.tile(ccfg["fill_color"], (len(pts), 1))
        fills[(ci, cj)] = dict(pts=pts, cols=np.clip(cols, 0, 1),
                               d_surf=d_surf, n_quads=Q,
                               area=Q * pitch * pitch, t_local=pd["t_local"],
                               mids=pd["mids"])
    return fills


# ============================================================ gaps and chips
def apply_gaps_chips(frags, fills, gap_cfg, chip_cfg, t_ref, diag, seed):
    """Addition #3: per-fragment independent rim erosion + chip bites, so adjacent
    sherds' contours no longer match (unearthed-material loss)."""
    wl = diag / 10.0
    gap_base = gap_cfg["scale"] * max(t_ref, 1e-6)
    r_chip = chip_cfg["radius_scale"] * max(t_ref, 1e-6)
    stats = {}
    for f in frags:
        i = f["label"]
        rngf = rng_child(seed, 700 + i)
        u_f = rngf.uniform(0.25, 1.0)
        my_fills = [(k, v) for k, v in fills.items() if i in k]
        if not my_fills or gap_base <= 0:
            f["gap_mean"] = 0.0
            continue
        rim_pts = np.vstack([v["mids"] for _, v in my_fills])
        if len(rim_pts) > 50_000:
            rim_pts = rim_pts[rngf.choice(len(rim_pts), 50_000, replace=False)]
        tree_rim = cKDTree(rim_pts)

        def erosion(pts):
            n = relief_field("fbm", pts, wl, rng_child(seed, 900 + i), 0.8, 3)
            return np.clip(gap_base * (u_f + gap_cfg["noise"] * n), 0.0, None)

        b_shell = tree_rim.query(f["pts"])[0]
        e_shell = erosion(f["pts"])
        keep = b_shell > e_shell
        f["pts"], f["cols"] = f["pts"][keep], f["cols"][keep]
        gsum, gn = 0.0, 0
        for k, v in my_fills:
            e_f = erosion(v["pts"])
            vk = v["d_surf"] > e_f
            v.setdefault("keep", {})[i] = vk
            gsum += float((e_f[vk]).mean()) if vk.any() else 0.0
            gn += 1
        # chips: stochastic bites at the fracture rim
        if chip_cfg["count"] > 0:
            zone = np.vstack([v["pts"][v["d_surf"] < 2.5 * r_chip]
                              for _, v in my_fills
                              if (v["d_surf"] < 2.5 * r_chip).any()]) if my_fills else np.zeros((0, 3))
            if len(zone) >= 8:
                cc = zone[rngf.choice(len(zone), min(chip_cfg["count"], len(zone)),
                                      replace=False)]
                tc = cKDTree(cc)
                ds = tc.query(f["pts"])[0]
                f["pts"], f["cols"] = f["pts"][ds > r_chip], f["cols"][ds > r_chip]
                for k, v in my_fills:
                    dk = tc.query(v["pts"])[0] > r_chip
                    v["keep"][i] = v["keep"][i] & dk
        f["gap_mean"] = gsum / max(gn, 1)
        stats[i] = f["gap_mean"]
    return stats


# ============================================================ per-file driver
def process_file(in_path, out_root, args):
    t0 = time.time()
    stem = in_path.stem
    out = out_root / stem
    fdir = out / "fragments"
    out.mkdir(parents=True, exist_ok=True); fdir.mkdir(parents=True, exist_ok=True)
    rng = rng_from_seed(args.seed)
    log(f"  [{stem}] loading")
    mesh = trimesh.load(str(in_path), force="scene")
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms: raise RuntimeError("no usable mesh geometry")
        mesh = geoms[0].copy() if len(geoms) == 1 else trimesh.util.concatenate(geoms)
    mesh.process(validate=True); mesh.merge_vertices()
    if len(mesh.faces) == 0: raise RuntimeError("mesh has no faces")
    colors = extract_vertex_colors(mesh)
    center = mesh.bounds.mean(0)
    diag = bbox_diagonal(mesh)
    expl = args.explode_distance or args.explode_scale * diag

    pc_pts, pc_cols, pc_fi = sample_surface_points_colors(mesh, colors, args.num_samples, rng)
    spacing = float(np.sqrt(mesh.area / max(len(pc_pts), 1))) if mesh.area > 1e-18 else diag / 1e3
    log(f"  [{stem}] pc={len(pc_pts):,} spacing~{spacing:.4g}")

    # --- stage 2/3: voxel graph, weakness, fracture modes (paper Alg. 1)
    centers, E, eax, pitch, strength = build_voxel_graph(mesh, args.grid_res)
    eta = (1e-3 + np.minimum(strength[E[:, 0]], strength[E[:, 1]])) ** args.weak_gamma
    k = max(2, min(args.modes, max(2, len(centers) // 200)))
    Us, Js = fracture_modes(centers, E, eta, k, args.irls_iters, rng_child(args.seed, 11))
    log(f"  [{stem}] voxels={len(centers)} edges={len(E)} modes={len(Us)}")
    if args.fracture == "impact":
        comp, d, vox_score = impact_fracture(centers, E, Us, args.impact_pos,
                                             args.impact_radius or diag / 6.0,
                                             args.impact_sigma)
    else:
        comp, d, vox_score = prefracture(centers, E, Js, args.fragments,
                                         args.min_vox, rng)
    tree_vox = cKDTree(centers)
    pc_lab = comp[tree_vox.query(pc_pts)[1]]
    face_lab = comp[tree_vox.query(np.asarray(mesh.triangles_center))[1]]
    ncomp = int(comp.max()) + 1
    log(f"  [{stem}] fragments (components) = {ncomp}")

    # --- stage 5: interface fills with swappable relief noise
    stride = max(1, len(pc_pts) // 200_000)
    tree_surf = cKDTree(pc_pts[::stride])
    hcfg = dict(H=args.relief_H, octaves=args.relief_octaves,
                amp_scale=args.relief_amp_scale, grain_scale=args.grain_amp_scale,
                fade_fraction=args.fade_fraction,
                wavelength_scale=args.relief_wavelength_scale)
    ccfg = dict(model=args.fill_color_model, surface_color=np.asarray(args.surface_color, float),
                core_color=np.asarray(args.core_color, float),
                fill_color=np.asarray(args.fill_color, float))
    fills = interface_fill(comp, E, eax, centers, pitch, strength, spacing, tree_surf,
                           args.relief_noise, hcfg, ccfg, rng_child(args.seed, 300))
    log(f"  [{stem}] interfaces filled: {len(fills)}")

    # --- fragment records (shell point subsets)
    frags = []
    for i in range(ncomp):
        m = pc_lab == i
        fm = face_lab == i
        if fm.sum() < max(4, args.min_faces) or m.sum() < 64:
            continue
        frags.append(dict(label=i, pts=pc_pts[m].copy(), cols=pc_cols[m].copy(),
                          faces=np.flatnonzero(fm), gap_mean=0.0))
    t_ref = float(2.0 * np.median(strength[strength > 0])) if (strength > 0).any() else 4 * spacing

    # --- stage 7: gaps & chips (independent per fragment)
    apply_gaps_chips(frags, fills, dict(scale=args.gap_scale, noise=args.gap_noise),
                     dict(count=args.chip_count, radius_scale=args.chip_radius_scale),
                     t_ref, diag, args.seed)

    # --- stage 6: weathering (shell fully, fracture faces slightly fresher)
    wrng = rng_child(args.seed, 500)
    wcfg = WeatherCfg(strength=args.weather, hue=args.weather_hue, desat=args.weather_desat,
                      stain=args.weather_stain, bleach=args.weather_bleach,
                      deposit_cover=args.weather_deposit_cover,
                      deposit_amp=args.weather_deposit_amp)
    for f in frags:
        f["cols"] = weather_colors(f["cols"], f["pts"], wrng, wcfg, diag / 6.0)
    for v in fills.values():
        v["cols"] = weather_colors(v["cols"], v["pts"], wrng, wcfg, diag / 6.0,
                                   factor=args.weather_fill_factor)

    # --- stage 8: exports
    palette = fragment_palette(max(len(frags), 1), args.palette_sat, args.palette_val)
    def save_pc(pts, cols, path):
        rgba = np.column_stack([(np.clip(cols, 0, 1) * 255).astype(np.uint8),
                                np.full((len(cols), 1), 255, np.uint8)])
        trimesh.PointCloud(np.asarray(pts, float), colors=rgba).export(str(path))
    def rel(p): return str(Path(p).relative_to(out))

    save_pc(pc_pts, pc_cols, out / "complete.ply")
    files = {"complete_pc": rel(out / "complete.ply")}
    asm_p, asm_c, exp_p, exp_c = [], [], [], []
    assembled, exploded = [], []
    nodes, edges_json = [], []
    for n_i, f in enumerate(frags):
        i = f["label"]
        vids, inv = np.unique(np.asarray(mesh.faces)[f["faces"]].ravel(), return_inverse=True)
        shell = trimesh.Trimesh(vertices=np.asarray(mesh.vertices)[vids],
                                faces=inv.reshape(-1, 3).astype(np.int64), process=False)
        shell = apply_vertex_colors(shell, f["cols"][::max(1, len(f["cols"]) // max(len(vids), 1))][:len(vids)]
                                    if len(vids) else f["cols"])
        centroid = f["pts"].mean(0)
        dv = centroid - center
        nn = float(np.linalg.norm(dv))
        off = (dv / nn if nn > 1e-9 else np.array([0, 0, 1.0])) * expl
        fpts, fcols = [f["pts"]], [f["cols"]]
        for (ci, cj), v in fills.items():
            if i not in (ci, cj): continue
            vk = v["keep"].get(i, np.ones(len(v["pts"]), bool))
            fpts.append(v["pts"][vk]); fcols.append(v["cols"][vk])
        fpts, fcols = np.vstack(fpts), np.vstack(fcols)
        pp = fdir / f"frag_{n_i:03d}.ply"
        save_pc(fpts, fcols, pp)
        exp_p.append(fpts + off); exp_c.append(fcols)
        asm_p.append(fpts)
        asm_c.append(shade_palette(palette[n_i], fcols, args.assembly_color_mode))
        ex = shell.copy(); ex.apply_translation(off)
        assembled.append(apply_vertex_colors(shell.copy(),
                                             shade_palette(palette[n_i], f["cols"], args.assembly_color_mode)))
        exploded.append(apply_vertex_colors(ex, f["cols"]))
        nodes.append(dict(id=f"frag_{n_i:03d}", label=int(i), centroid=centroid.tolist(),
                          explode_offset=off.tolist(), n_points=int(len(fpts)),
                          palette_rgb=np.round(palette[n_i], 4).tolist(),
                          gap_mean=float(f["gap_mean"]), ply=rel(pp)))
    save_pc(np.vstack(asm_p), np.vstack(asm_c), out / "assembled.ply")
    save_pc(np.vstack(exp_p), np.vstack(exp_c), out / "exploded.ply")
    trimesh.Scene(assembled).export(str(out / "assembled.glb"))
    trimesh.Scene(exploded).export(str(out / "exploded.glb"))
    files.update(assembled_pc=rel(out / "assembled.ply"), exploded_pc=rel(out / "exploded.ply"),
                 assembled_mesh=rel(out / "assembled.glb"), exploded_mesh=rel(out / "exploded.glb"))
    # fracture-affinity visualisation (modes' discontinuity field on the surface)
    save_pc(pc_pts, np.clip(np.stack([norm01(vox_score[pc_lab]),
                                      0.25 * np.ones(len(pc_pts)),
                                      1 - norm01(vox_score[pc_lab])], 1), 0, 1),
            out / "fracture_affinity.ply")
    files["affinity"] = rel(out / "fracture_affinity.ply")
    # adjacency graph
    id_of = {f["label"]: f"frag_{n:03d}" for n, f in enumerate(frags)}
    for (ci, cj), v in fills.items():
        if ci not in id_of or cj not in id_of: continue
        gi, gj = v["keep"].get(ci), v["keep"].get(cj)
        edges_json.append(dict(a=id_of[ci], b=id_of[cj], interface_quads=int(v["n_quads"]),
                               interface_area=float(v["area"]),
                               mean_wall_thickness=float(v["t_local"]),
                               fill_points=int(len(v["pts"])),
                               kept_a=int(gi.sum()) if gi is not None else 0,
                               kept_b=int(gj.sum()) if gj is not None else 0,
                               gap_mean=float(0.5 * (next(f["gap_mean"] for f in frags if f["label"] == ci) +
                                                     next(f["gap_mean"] for f in frags if f["label"] == cj)))))
    adj = dict(object=stem, n_fragments=len(frags), n_edges=len(edges_json),
               nodes=nodes, edges=edges_json)
    (out / "adjacency.json").write_text(json.dumps(adj, indent=2, default=str))
    files["adjacency"] = rel(out / "adjacency.json")
    meta = dict(input=str(in_path), pipeline="fracture_modes_v1_weathered_gaps",
                seed=int(args.seed), n_fragments=len(frags), n_modes=len(Us),
                relief_noise=args.relief_noise, weather=args.weather,
                gap_scale=args.gap_scale, point_spacing=spacing, voxel_pitch=pitch,
                args=vars(args), files=files)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    log(f"  [{stem}] done in {time.time()-t0:.1f}s -> {out} "
        f"(frags={len(frags)}, edges={len(edges_json)})")
    return dict(input=str(in_path), output=str(out), n_fragments=len(frags),
                adjacency_edges=len(edges_json), status="ok")


# ====================================================================== CLI
def build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", "-i", default="pottery", type=Path)
    p.add_argument("--output", "-o", default="fragmented_physics", type=Path)
    g = p.add_argument_group("sampling / fragments")
    g.add_argument("--num-samples", type=int, default=2_000_000)
    g.add_argument("--fragments", "-n", type=int, default=16)
    g.add_argument("--min-faces", type=int, default=50)
    g.add_argument("--explode-scale", type=float, default=0.12)
    g.add_argument("--explode-distance", type=float, default=None)
    g.add_argument("--seed", type=int, default=42)
    g = p.add_argument_group("fracture modes (Sellán et al. 2022)")
    g.add_argument("--fracture", choices=["modes", "impact"], default="modes")
    g.add_argument("--grid-res", type=int, default=48)
    g.add_argument("--modes", type=int, default=6, help="k fracture modes")
    g.add_argument("--irls-iters", type=int, default=4, help="sparsification reweights")
    g.add_argument("--weak-gamma", type=float, default=1.0, help="thin-wall weakness exponent (η)")
    g.add_argument("--min-vox", type=int, default=24, help="min voxels per fragment")
    g.add_argument("--impact-pos", type=float, nargs=3, default=[1.0, 0.0, 0.35])
    g.add_argument("--impact-radius", type=float, default=None)
    g.add_argument("--impact-sigma", type=float, default=0.15, help="glue tolerance σ")
    g = p.add_argument_group("fracture-surface relief (swappable noise)")
    g.add_argument("--relief-noise", choices=list(NOISE_REGISTRY), default="fbm")
    g.add_argument("--relief-amp-scale", type=float, default=0.18)
    g.add_argument("--relief-H", type=float, default=0.78)
    g.add_argument("--relief-octaves", type=int, default=5)
    g.add_argument("--relief-wavelength-scale", type=float, default=1.5)
    g.add_argument("--grain-amp-scale", type=float, default=0.03)
    g.add_argument("--fade-fraction", type=float, default=0.35)
    g.add_argument("--fill-color-model", choices=["constant", "depth_core"], default="depth_core")
    g.add_argument("--surface-color", type=float, nargs=3, default=CLAY.tolist())
    g.add_argument("--core-color", type=float, nargs=3, default=CORE.tolist())
    g.add_argument("--fill-color", type=float, nargs=3, default=CLAY.tolist())
    g = p.add_argument_group("weathering (addition 2)")
    g.add_argument("--weather", type=float, default=0.6)
    g.add_argument("--weather-hue", type=float, default=0.03)
    g.add_argument("--weather-desat", type=float, default=0.45)
    g.add_argument("--weather-stain", type=float, default=0.35)
    g.add_argument("--weather-bleach", type=float, default=0.25)
    g.add_argument("--weather-deposit-cover", type=float, default=0.12)
    g.add_argument("--weather-deposit-amp", type=float, default=0.55)
    g.add_argument("--weather-fill-factor", type=float, default=0.6,
                   help="fracture faces weather less than outer surface")
    g = p.add_argument_group("gaps & chips (addition 3)")
    g.add_argument("--gap-scale", type=float, default=0.30, help="rim loss as fraction of wall")
    g.add_argument("--gap-noise", type=float, default=0.6, help="spatial unevenness of rim loss")
    g.add_argument("--chip-count", type=int, default=6, help="chip bites per sherd")
    g.add_argument("--chip-radius-scale", type=float, default=0.8, help="chip radius × wall")
    g = p.add_argument_group("assembly visuals")
    g.add_argument("--assembly-color-mode", choices=["shaded", "flat"], default="shaded")
    g.add_argument("--palette-sat", type=float, default=0.75)
    g.add_argument("--palette-val", type=float, default=0.95)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.input = Path(args.input).expanduser(); args.output = Path(args.output).expanduser()
    if args.seed is None:
        args.seed = int(np.random.default_rng().integers(1, 2**31 - 1))
    log(f"run seed = {args.seed}")
    files = ([args.input] if args.input.is_file() else sorted(
        q for q in args.input.rglob("*")
        if q.suffix.lower() in {".glb", ".gltf", ".obj", ".ply", ".stl"}))
    if not files:
        log("no inputs found"); return 1
    args.output.mkdir(parents=True, exist_ok=True)
    sums = []
    for k, f in enumerate(files, 1):
        log(f"=== ({k}/{len(files)}) {f.name} ===")
        try:
            sums.append(process_file(f, args.output, args))
        except Exception as exc:
            log(f"  ! FAILED {f.name}: {exc}"); traceback.print_exc()
            sums.append(dict(input=str(f), n_fragments=0, status=f"error: {exc}"))
    (args.output / "run_summary.json").write_text(json.dumps(sums, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())