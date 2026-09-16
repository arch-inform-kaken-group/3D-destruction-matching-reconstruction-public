"""
Usage:
    python viz.py --input pottery/AS0001(1).glb --output pipeline_viz/
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import trimesh
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt, binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import destruct as pipe
except ImportError:
    sys.exit(1)

MCM = matplotlib.colormaps
MAX_SCATTER = 150_000
VIEW = dict(elev=22, azim=-90)  # default fallback
GRAY_CTX = np.array([0.55, 0.57, 0.60])
ZOOM = 1.25


#  clean plotting helpers
def _sub(pts, cols=None, n=MAX_SCATTER):
    if len(pts) <= n:
        return (pts, cols)
    idx = np.random.default_rng(0).choice(len(pts), n, replace=False)
    return (pts[idx], None if cols is None else cols[idx])


def view_from_dir(d):
    """(elev, azim) in degrees for a camera placed along unit direction d."""
    d = np.asarray(d, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    elev = np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0)))
    azim = np.degrees(np.arctan2(d[1], d[0]))
    return float(elev), float(azim)


def clean3d(ax,
            lo=None,
            hi=None,
            title="",
            view_dir=None,
            view_angles=None,
            zoom=1.0):
    """Orthographic, equal aspect, no axes/grid with adjustable zoom."""
    ax.set_axis_off()
    ax.grid(False)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass

    if view_angles is not None:
        elev, azim = view_angles[0], view_angles[1]
        if len(view_angles) >= 3 and hasattr(ax, 'roll'):
            ax.view_init(elev=elev, azim=azim, roll=view_angles[2])
        else:
            ax.view_init(elev=elev, azim=azim)
    elif view_dir is not None:
        elev, azim = view_from_dir(view_dir)
        ax.view_init(elev=elev, azim=azim)
    else:
        ax.view_init(**VIEW)

    if lo is not None and hi is not None:
        c = (lo + hi) * 0.5
        # Divide by zoom so higher values shrink the bounding box range (zooming in)
        m = (float(((hi - lo) * 0.5).max()) * 1.02) / float(zoom)
        ax.set_xlim(c[0] - m, c[0] + m)
        ax.set_ylim(c[1] - m, c[1] + m)
        ax.set_zlim(c[2] - m, c[2] + m)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold")


def pc(ax, pts, cols, s=0.4, n=MAX_SCATTER):
    pts, cols = _sub(pts, cols, n)
    ax.scatter(pts[:,
                   0],
               pts[:,
                   1],
               pts[:,
                   2],
               c=cols,
               s=s,
               depthshade=False,
               rasterized=True)


def gray_rgba(n, a=0.12):
    return np.tile([*GRAY_CTX, a], (n, 1))


def clean2d(ax, keep_spines=False):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if not keep_spines:
        for sp in ax.spines.values():
            sp.set_visible(False)


def save(fig, path):
    fig.savefig(str(path), dpi=330, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {path}")


def cmap(name, n=None):
    c = MCM[name]
    return c.resampled(n) if n else c


#  interactive camera capture
def get_user_view_mesh(mesh, title="Set Camera Angle"):
    print(f"\n[!] ACTION REQUIRED: {title}")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    v = mesh.vertices
    f = mesh.faces
    if len(f) > 15000:
        step = len(f) // 15000
        f = f[::step]

    ax.plot_trisurf(v[:,
                      0],
                    v[:,
                      1],
                    v[:,
                      2],
                    triangles=f,
                    color='lightgray',
                    alpha=0.9,
                    edgecolor='none')
    clean3d(ax,
            v.min(0),
            v.max(0),
            f"{title}\n(Rotate with mouse, then close the window to confirm)",
            zoom=ZOOM)

    view_state = {
        'elev': ax.elev,
        'azim': ax.azim,
        'roll': getattr(ax,
                        'roll',
                        0)
    }

    def on_draw(event):
        view_state['elev'] = ax.elev
        view_state['azim'] = ax.azim
        view_state['roll'] = getattr(ax, 'roll', 0)

    fig.canvas.mpl_connect('draw_event', on_draw)
    plt.show(block=True)
    print(
        f"    -> Captured global view: elev={view_state['elev']:.1f}°, azim={view_state['azim']:.1f}°"
    )
    return view_state['elev'], view_state['azim'], view_state['roll']


def get_user_view_plane(pc_pts, segs, mid, u, v, seeds, pair, title):
    print(f"\n[!] ACTION REQUIRED: {title}")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    psub, _ = _sub(pc_pts, None, 15000)
    ax.scatter(psub[:,
                    0],
               psub[:,
                    1],
               psub[:,
                    2],
               s=0.5,
               c='gray',
               alpha=0.3,
               depthshade=False)

    if len(segs):
        rel = segs.reshape(-1, 3) - mid
        xy = np.stack([rel @ u, rel @ v], -1).reshape(-1, 2, 2)
        umin, umax = xy[..., 0].min(), xy[..., 0].max()
        vmin, vmax = xy[..., 1].min(), xy[..., 1].max()
        pad = 0.12 * max(umax - umin, vmax - vmin, 1e-6)
        umin -= pad
        umax += pad
        vmin -= pad
        vmax += pad
        c00 = mid + umin * u + vmin * v
        c10 = mid + umax * u + vmin * v
        c11 = mid + umax * u + vmax * v
        c01 = mid + umin * u + vmax * v
        ax.add_collection3d(
            Poly3DCollection(
                [np.array([c00,
                           c10,
                           c11]),
                 np.array([c00,
                           c11,
                           c01])],
                alpha=0.30,
                facecolor="#4da3ff",
                edgecolor="none"))
        for a_, b_ in [(c00, c10), (c10, c11), (c11, c01), (c01, c00)]:
            ax.plot(*zip(a_, b_), color="#08306b", lw=1.4)
        for sgm in segs:
            ax.plot(sgm[:, 0], sgm[:, 1], sgm[:, 2], color="red", lw=2.4)

    i, j = pair
    ax.scatter(*seeds[i], c="lime", s=60, edgecolors="k", depthshade=False)
    ax.scatter(*seeds[j], c="orange", s=60, edgecolors="k", depthshade=False)

    clean3d(ax,
            pc_pts.min(0),
            pc_pts.max(0),
            f"{title}\n(Rotate with mouse, then close the window to confirm)",
            zoom=ZOOM)

    view_state = {
        'elev': ax.elev,
        'azim': ax.azim,
        'roll': getattr(ax,
                        'roll',
                        0)
    }

    def on_draw(event):
        view_state['elev'] = ax.elev
        view_state['azim'] = ax.azim
        view_state['roll'] = getattr(ax, 'roll', 0)

    fig.canvas.mpl_connect('draw_event', on_draw)
    plt.show(block=True)
    print(
        f"    -> Captured plane view: elev={view_state['elev']:.1f}°, azim={view_state['azim']:.1f}°"
    )
    return view_state['elev'], view_state['azim'], view_state['roll']


#  stage figures (all point-cloud based)
def viz_01_input(pc_pts, pc_cols, outdir, view_angles):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    pc(ax, pc_pts, pc_cols)
    clean3d(
        ax,
        pc_pts.min(0),
        pc_pts.max(0),
        f"Stage 1: Original Artifact\nPoint Cloud Representation ({len(pc_pts):,} pts)",
        view_angles=view_angles,
        zoom=ZOOM)
    save(fig, outdir / "01_input_mesh.png")


def viz_02_density(pc_pts, pc_cols, outdir, view_angles):
    pts, _ = _sub(pc_pts, None, 60_000)
    d = cKDTree(pts).query(pts, k=6)[0][:, -1]
    cols = cmap("viridis")(pipe.norm01(d))[:, :3]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    pc(ax, pts, cols, n=10**9)
    clean3d(ax,
            pts.min(0),
            pts.max(0),
            "Stage 2: Surface Sampling Densityn\nand Spatial Uniformity",
            view_angles=view_angles,
            zoom=ZOOM)
    save(fig, outdir / "02_point_cloud.png")
    rgba = np.column_stack([(np.clip(pc_cols,
                                     0,
                                     1) * 255).astype(np.uint8),
                            np.full((len(pc_cols),
                                     1),
                                    255,
                                    dtype=np.uint8)])
    trimesh.PointCloud(pc_pts,
                       rgba).export(
                           str(outdir / "samples" / "stage02_pointcloud.ply"))


def viz_03_stress(pc_pts, score_f, outdir, view_angles):
    cols = cmap("hot")(np.clip(score_f, 0, 1))[:, :3]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    pc(ax, pc_pts, cols)
    clean3d(
        ax,
        pc_pts.min(0),
        pc_pts.max(0),
        "Stage 3: Computed Structural Stress Map\n(Thickness, Feature Bias, and Multi-frequency Noise)",
        view_angles=view_angles,
        zoom=ZOOM)
    save(fig, outdir / "03_stress_field.png")


def viz_04_thickness(pc_pts, thick_f, outdir, view_angles):
    if thick_f is None:
        return
    cols = cmap("viridis")(np.clip(thick_f / max(thick_f.max(), 1e-12), 0, 1))[:, :3]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    pc(ax, pc_pts, cols)
    clean3d(ax,
            pc_pts.min(0),
            pc_pts.max(0),
            "Stage 3b: Volumetric Wall Thickness Estimation",
            view_angles=view_angles,
            zoom=ZOOM)
    save(fig, outdir / "04_wall_thickness.png")


def viz_05_seeds(pc_pts, score_f, seeds, seed_normals, outdir, view_angles):
    cols = cmap("hot")(np.clip(score_f, 0, 1))[:, :3]
    psub, csub = _sub(pc_pts, cols, 100_000)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    pc(ax, psub, csub * 0.85, s=0.35, n=10**9)  # stress underlay

    # Backface Culling for Seeds
    # Calculate camera direction vector from the captured view angles
    elev, azim = view_angles[0], view_angles[1]
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    cam_dir = np.array([
        np.cos(elev_rad) * np.cos(azim_rad),
        np.cos(elev_rad) * np.sin(azim_rad),
        np.sin(elev_rad)
    ])

    # A seed is visible if its surface normal points towards the camera (dot product > 0)
    visibility = seed_normals @ cam_dir
    visible_seeds = seeds[visibility
                          > 0.1]  # 0.1 threshold avoids edge-on artifacts

    if len(visible_seeds) > 0:
        # Plot visible seeds with high contrast and zorder to ensure they pop over the point cloud
        ax.scatter(visible_seeds[:,
                                 0],
                   visible_seeds[:,
                                 1],
                   visible_seeds[:,
                                 2],
                   c='lime',
                   s=95,
                   edgecolors='black',
                   linewidths=0.8,
                   depthshade=False,
                   zorder=10,
                   label='Visible Seeds')

    clean3d(
        ax,
        pc_pts.min(0),
        pc_pts.max(0),
        f"Stage 4: Fracture Seed Generation ({len(seeds)} Seeds)\nOverlaying Structural Stress Map",
        view_angles=view_angles)
    save(fig, outdir / "05_seeds.png")


def viz_06_labels(pc_pts, pc_labels, nS, outdir, view_angles):
    cols = cmap("tab20", max(nS, 1))(pc_labels % 20)[:, :3]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    pc(ax, pc_pts, cols)
    clean3d(
        ax,
        pc_pts.min(0),
        pc_pts.max(0),
        f"Stage 5: Volumetric Fragment Partitioning ({nS} Regions)\nUsing Log-normal Power Diagrams",
        view_angles=view_angles,
        zoom=ZOOM)
    save(fig, outdir / "06_voronoi_labels.png")


# plane + segments (one pair, tight plane quad)
def viz_07_plane(pc_pts,
                 pc_cols,
                 pc_labels,
                 segs,
                 nv,
                 mid,
                 u,
                 v,
                 seeds,
                 pair,
                 outdir,
                 view_angles):
    i, j = pair
    fig = plt.figure(figsize=(13, 6.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")

    # Highlight the two adjacent fragments at the cut surface
    # Base dimmed context
    rgba = np.full((len(pc_pts), 4), [0.75, 0.75, 0.75, 0.12])

    # Fragment i: Cyan
    rgba[pc_labels == i] = [0.15, 0.75, 0.95, 0.20]

    # Fragment j: Orange
    rgba[pc_labels == j] = [1.00, 0.45, 0.05, 0.20]

    psub, csub = _sub(pc_pts, rgba, 40_000)
    pc(ax1, psub, csub, s=0.4, n=10**9)

    if len(segs):
        rel = segs.reshape(-1, 3) - mid
        xy = np.stack([rel @ u, rel @ v], -1).reshape(-1, 2, 2)
        umin, umax = xy[..., 0].min(), xy[..., 0].max()
        vmin, vmax = xy[..., 1].min(), xy[..., 1].max()
        pad = 0.12 * max(umax - umin, vmax - vmin, 1e-6)
        umin -= pad
        umax += pad
        vmin -= pad
        vmax += pad
        c00 = mid + umin * u + vmin * v
        c10 = mid + umax * u + vmin * v
        c11 = mid + umax * u + vmax * v
        c01 = mid + umin * u + vmax * v
        ax1.add_collection3d(
            Poly3DCollection(
                [np.array([c00,
                           c10,
                           c11]),
                 np.array([c00,
                           c11,
                           c01])],
                alpha=0.15,
                facecolor="#4da3ff",
                edgecolor="#08306b",
                lw=1.0))
        for sgm in segs:
            ax1.plot(sgm[:, 0], sgm[:, 1], sgm[:, 2], color="red", lw=2.4)

    # Highlight seeds with matching colors
    ax1.scatter(*seeds[i],
                c="#00e676",
                s=120,
                edgecolors="black",
                linewidths=1.2,
                depthshade=False,
                zorder=10)
    ax1.scatter(*seeds[j],
                c="#ff9100",
                s=120,
                edgecolors="black",
                linewidths=1.2,
                depthshade=False,
                zorder=10)

    clean3d(ax1,
            pc_pts.min(0),
            pc_pts.max(0),
            f"Stage 7: Cut Plane Intersection",
            view_angles=view_angles)

    ax2 = fig.add_subplot(1, 2, 2)
    if len(segs):
        rel = segs.reshape(-1, 3) - mid
        xy = np.stack([rel @ u, rel @ v], -1).reshape(-1, 2, 2)
        for sgm in xy:
            ax2.plot(sgm[:, 0], sgm[:, 1], "r-", lw=1.6)
        ax2.set_aspect("equal")
        clean2d(ax2)
        ax2.set_title(f"Stage 7: 2D Projected Boundary Segments",
                      fontsize=16,
                      fontweight="bold")
    fig.tight_layout()
    save(fig, outdir / "07_plane_intersection.png")
    if len(segs):
        sp = segs.reshape(-1, 3)
        rgba_seg = np.full((len(sp), 4), [255, 0, 0, 255], dtype=np.uint8)
        trimesh.PointCloud(sp,
                           rgba_seg).export(
                               str(outdir / "samples" /
                                   "stage07_plane_segments.ply"))


# raster fill (same pair)
def viz_08_fill(segs, nv, mid, u, v, spacing, diag, outdir):
    if len(segs) < 2:
        return None
    rel = segs.reshape(-1, 3) - mid
    xy = np.stack([rel @ u, rel @ v], -1).reshape(-1, 2, 2)
    margin = 2.0 * spacing
    xmin = xy[..., 0].min() - margin
    xmax = xy[..., 0].max() + margin
    ymin = xy[..., 1].min() - margin
    ymax = xy[..., 1].max() + margin
    gs = spacing
    cols = int((xmax - xmin) / gs) + 1
    rows = int((ymax - ymin) / gs) + 1
    if cols > 2000 or rows > 2000:
        sc = max(cols / 2000, rows / 2000)
        gs *= sc
        cols = int((xmax - xmin) / gs) + 1
        rows = int((ymax - ymin) / gs) + 1
    if cols < 3 or rows < 3:
        return None

    envelope = pipe._raster_segments(xy, gs, xmin, ymin, rows, cols)
    polys = pipe._chain_polylines(xy, snap=max(1e-9, diag * 1e-7))
    open_p = [p for p, c in polys if not c and len(p) >= 3]
    closed_p = [p for p, c in polys if c and len(p) >= 3]

    fill = np.zeros((rows, cols), dtype=bool)
    used_c = set()
    for a in range(len(closed_p)):
        for b in range(a + 1, len(closed_p)):
            if a in used_c or b in used_c:
                continue
            L1, L2 = closed_p[a], closed_p[b]
            if pipe._point_in_poly(
                    L2,
                    L1[0]) and not pipe._point_in_poly(L1,
                                                       L2[0]):
                outer, inner = L2, L1
            elif pipe._point_in_poly(
                    L1,
                    L2[0]) and not pipe._point_in_poly(L2,
                                                       L1[0]):
                outer, inner = L1, L2
            else:
                continue
            mo = pipe._scanline_mask(outer, gs, xmin, ymin, rows, cols)
            mi = pipe._scanline_mask(inner, gs, xmin, ymin, rows, cols)
            sep_c = float(np.median(cKDTree(inner).query(outer)[0]))
            m = (mo & ~mi) & binary_dilation(envelope,
                                             pipe._disk(sep_c / (2 * gs) + 2))
            if m.sum() >= 16:
                fill |= m
                used_c.add(a)
                used_c.add(b)
    for a in range(len(closed_p)):
        if a in used_c:
            continue
        m = pipe._scanline_mask(closed_p[a], gs, xmin, ymin, rows, cols)
        m &= binary_dilation(envelope, pipe._disk(2))
        if m.sum() >= 16:
            fill |= m
            used_c.add(a)
    cand = []
    for a in range(len(open_p)):
        for b in range(a + 1, len(open_p)):
            A, B = open_p[a], open_p[b]
            d_ff = np.linalg.norm(A[0] - B[0]) + np.linalg.norm(A[-1] - B[-1])
            d_fr = np.linalg.norm(A[0] - B[-1]) + np.linalg.norm(A[-1] - B[0])
            dAB = cKDTree(B).query(A)[0]
            dBA = cKDTree(A).query(B)[0]
            prox = max(float(np.percentile(dAB,
                                           90)),
                       float(np.percentile(dBA,
                                           90)))
            ep = min(d_ff, d_fr)
            if ep <= max(8 * gs, 4 * prox):
                cand.append((ep, prox, a, b, d_ff <= d_fr))
    cand.sort(key=lambda x: (x[0], x[1]))
    used_o = set()
    for ep, prox, a, b, ff in cand:
        if a in used_o or b in used_o:
            continue
        used_o.add(a)
        used_o.add(b)
        A = open_p[a]
        B = open_p[b] if ff else open_p[b][::-1]
        poly = np.concatenate([A, B[::-1]])
        if pipe._shoelace(poly) < 8 * gs * gs:
            continue
        m = pipe._scanline_mask(poly, gs, xmin, ymin, rows, cols)
        m &= binary_dilation(envelope, pipe._disk(prox / (2 * gs) + 2))
        if m.sum() >= 16:
            fill |= m
    for a in range(len(open_p)):
        if a in used_o:
            continue
        A = open_p[a]
        if len(A) < 3:
            continue
        ed = float(np.linalg.norm(A[0] - A[-1]))
        pl = pipe._poly_length(A)
        if pl < 1e-9 or ed > 0.25 * pl or pipe._shoelace(A) < 8 * gs * gs:
            continue
        m = pipe._scanline_mask(A, gs, xmin, ymin, rows, cols)
        m &= binary_dilation(envelope, pipe._disk(ed / (2 * gs) + 2))
        if m.sum() >= 16:
            fill |= m
            used_o.add(a)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(envelope,
                   cmap="gray_r",
                   origin="lower",
                   interpolation="nearest")
    axes[0].set_title("Stage 8A: Bounding Envelope of Kept Segments",
                      fontsize=16,
                      fontweight="bold")

    overlay = np.zeros((*fill.shape, 4))
    overlay[envelope] = [0.6, 0.6, 0.6, 0.4]
    overlay[fill] = [0.2, 0.6, 1.0, 0.85]
    axes[1].imshow(overlay, origin="lower", interpolation="nearest")
    axes[1].set_title(
        # f"Stage 8B: Rasterized Solid Fill Mask ({int(fill.sum())} px)",
        f"Stage 8B: Rasterized Solid Fill Mask",
        fontsize=16,
        fontweight="bold")

    for p, c in polys:
        px = (p[:, 0] - xmin) / gs
        py = (p[:, 1] - ymin) / gs
        axes[2].plot(px, py, "r-" if c else "b-", lw=1.0, alpha=0.8)
    axes[2].set_title(
        f"Stage 8C: Chained Vector Polylines\n(Closed={len(closed_p)}, Open={len(open_p)})",
        fontsize=16,
        fontweight="bold")
    for ax in axes:
        clean2d(ax)
    fig.tight_layout()
    save(fig, outdir / "08_raster_fill.png")
    return fill, (mid, u, v, nv, gs, xmin, ymin, rows, cols)


def get_user_view_fragment(csub, psub, title="Set Fragment 3D View"):
    print(f"\n[!] ACTION REQUIRED: {title}")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Render the interactive preview
    ax.scatter(csub[:,
                    0],
               csub[:,
                    1],
               csub[:,
                    2],
               s=0.3,
               c='gray',
               alpha=0.3,
               depthshade=False)
    ax.scatter(psub[:,
                    0],
               psub[:,
                    1],
               psub[:,
                    2],
               s=1.0,
               c='saddlebrown',
               alpha=0.8,
               depthshade=False)

    clean3d(ax,
            csub.min(0),
            csub.max(0),
            f"{title}\n(Rotate with mouse, then close window to confirm)",
            zoom=ZOOM)

    view_state = {'elev': 22, 'azim': -90, 'roll': 0}

    def on_draw(event):
        view_state['elev'] = ax.elev
        view_state['azim'] = ax.azim
        view_state['roll'] = getattr(ax, 'roll', 0)

    fig.canvas.mpl_connect('draw_event', on_draw)
    plt.show(block=True)
    print(
        f"    -> Captured fragment view: elev={view_state['elev']:.1f}°, azim={view_state['azim']:.1f}°"
    )
    return view_state['elev'], view_state['azim'], view_state['roll']


# fracture terrain (5-Subplot Layout with Interactive Fragment View)
def viz_09_terrain(fill_result,
                   t_local,
                   spacing,
                   outdir,
                   args,
                   rng_seed,
                   pc_pts,
                   pc_labels,
                   pair):
    if fill_result is None or fill_result[0] is None or fill_result[1] is None:
        print("    (skipping terrain viz - no fill for this pair)")
        return
    fill, (mid, u, v, nv, gs, xmin, ymin, rows, cols) = fill_result
    if not fill.any():
        return

    i, j = pair
    # FBM setup
    wall_px = max(2.0, t_local / max(spacing, 1e-9))
    h_rng = np.random.default_rng(np.random.SeedSequence([rng_seed, 900, 901]))
    amp = args.height_amp_scale * t_local
    grain = args.grain_amp_scale * t_local
    fade_px = max(2.0, args.height_fade_fraction * wall_px)
    wl_px = 1.5 * wall_px

    # Compute full field (for mask mapping and profiles)
    edge = distance_transform_edt(fill)
    fade = np.clip(edge / max(1e-9, fade_px), 0, 1)**0.7
    base_g = int(np.clip(min(rows, cols) / max(4.0, wl_px), 2, 64))
    field = pipe.fbm2((rows,
                       cols),
                      h_rng,
                      args.height_H,
                      args.height_octaves,
                      base_g=base_g)
    grain_field = h_rng.standard_normal((rows, cols))
    h_full = (amp * field + grain * grain_field) * fade

    fy, fx = np.where(fill)
    u_coords = xmin + (fx + 0.5) * gs
    v_coords = ymin + (fy + 0.5) * gs

    # 3D points
    pts = mid + u_coords[:, None] * u + v_coords[:, None] * v
    h_vals = h_full[fy, fx]
    pts_disp = pts + h_vals[:, None] * nv
    hrange = max(float(np.ptp(h_vals)), 1e-12)

    # SAVE NOISE LANDSCAPES (2D) & SURFACE POINTS (3D)
    samples_dir = outdir / "samples"

    # 1. Save Base FBM Noise Landscape
    plt.imsave(samples_dir / "stage09_noise_base_fbm.png",
               field,
               cmap="RdBu_r",
               origin="lower")

    # 2. Save Final Displaced Heightfield (with grain and edge fade)
    plt.imsave(samples_dir / "stage09_noise_final_height.png",
               h_full,
               cmap="terrain",
               origin="lower")

    # 3. Save 3D Surface Points as a colored PLY point cloud
    norm_h = (h_vals - h_vals.min()) / hrange
    pts_cols = cmap("terrain")(norm_h)[:, :3]
    rgba_disp = np.column_stack([(np.clip(pts_cols,
                                          0,
                                          1) * 255).astype(np.uint8),
                                 np.full(len(pts_cols),
                                         255,
                                         dtype=np.uint8)])
    trimesh.PointCloud(pts_disp,
                       rgba_disp).export(
                           str(samples_dir / "stage09_surface_points.ply"))
    print(
        f"    saved noise landscapes and stage 09 surface points to {samples_dir}/"
    )

    # ISO-FRAGMENT MASK: Select ONLY Fragment 'i' to expose cut
    ctx_mask = (pc_labels == i)
    ctx = pc_pts[ctx_mask]
    csub, _ = _sub(ctx, None, 35_000)
    psub, hsub = _sub(pts_disp, h_vals, 60_000)

    # Trigger interactive prompt specifically for this 3D fragment view
    fragment_view = get_user_view_fragment(csub,
                                           psub,
                                           title=f"Set 3D View for Fragment")

    # Setup 5-subplot figure layout (2 rows, 3 columns, bottom row spanning)
    fig = plt.figure(figsize=(18, 11))
    gsf = gridspec.GridSpec(2, 3, figure=fig, hspace=0.15, wspace=0.15)

    # 1. Base FBM Heightfield
    ax_a = fig.add_subplot(gsf[0, 0])
    ax_a.imshow(field, cmap="RdBu_r", origin="lower", interpolation="bilinear")
    ax_a.set_title("Base FBM Heightfield", fontsize=10, fontweight="bold")
    clean2d(ax_a)

    # 2. Edge Proximity Fade Mask
    ax_b = fig.add_subplot(gsf[0, 1])
    ax_b.imshow(fade, cmap="magma", origin="lower", interpolation="nearest")
    ax_b.set_title("Edge Proximity Fade Mask", fontsize=10, fontweight="bold")
    clean2d(ax_b)

    # 3. 2D View (Cut Plane Projection)
    ax_c = fig.add_subplot(gsf[0, 2])
    rel_ctx = csub - mid
    ctx_u = rel_ctx @ u
    ctx_v = rel_ctx @ v
    ax_c.scatter(ctx_u,
                 ctx_v,
                 c='lightgray',
                 s=0.3,
                 alpha=0.4,
                 rasterized=True)
    sc = ax_c.scatter(u_coords,
                      v_coords,
                      c=h_vals,
                      cmap="terrain_r",
                      s=2.5,
                      alpha=0.9,
                      rasterized=True)
    ax_c.set_aspect('equal')
    clean2d(ax_c)
    ax_c.set_title(f"2D Projection: Relief on Fragment",
                   fontsize=10,
                   fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax_c, shrink=0.7, pad=0.02)
    cbar.set_label("Displacement", fontsize=8)

    # 4. 3D View (Isolated Fragment + Relief) using your custom angle
    ax_d = fig.add_subplot(gsf[1, 0], projection="3d")
    pc(ax_d, csub, gray_rgba(len(csub), 0.15), s=0.3, n=10**9)
    pc(ax_d, psub, cmap("terrain")((hsub - h_vals.min()) / hrange)[:, :3], s=1.2, n=10**9)
    allp = np.vstack([csub, psub])
    clean3d(ax_d,
            allp.min(0),
            allp.max(0),
            f"3D View: Procedural Relief on Fragment",
            view_angles=fragment_view)

    # 5. Cross-section Profile (Spanning 2 columns)
    ax_e = fig.add_subplot(gsf[1, 1:])
    x_world = xmin + (np.arange(cols) + 0.5) * gs
    prof = h_full[rows // 2, :]
    ax_e.fill_between(x_world, 0, prof, alpha=0.35, color="saddlebrown")
    ax_e.plot(x_world, prof, color="saddlebrown", lw=1.2)
    ax_e.axhline(0, color="gray", lw=0.5, ls="--")
    ax_e.set_title(f"Cross-section Relief Profile",
                   fontsize=10,
                   fontweight="bold")
    clean2d(ax_e, keep_spines=True)

    fig.suptitle("Stage 10: Procedural Fracture Relief Synthesis via FBM",
                 fontsize=15,
                 fontweight="bold",
                 y=0.96)
    save(fig, outdir / "09_fracture_terrain.png")


# adjacency + finals
def viz_10_adjacency(fragments, pair_stats, node_by_seed, outdir):
    if len(fragments) < 2:
        return
    palette = pipe.fragment_palette(len(fragments))
    centroids = np.array([f.centroid for f in fragments])
    order = np.argsort(-centroids.var(axis=0))[:2]
    pos = centroids[:, order]
    id_to_idx = {f.fid: k for k, f in enumerate(fragments)}
    fig, ax = plt.subplots(figsize=(8, 8))
    seen = set()
    for (i, j), st in pair_stats.items():
        if st.get("n_kept",
                  0) < 1 or i not in node_by_seed or j not in node_by_seed:
            continue
        a = id_to_idx[node_by_seed[i].fid]
        b = id_to_idx[node_by_seed[j].fid]
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        ax.plot(pos[[a,
                     b],
                    0],
                pos[[a,
                     b],
                    1],
                color="gray",
                lw=0.9,
                alpha=0.55,
                zorder=1)
    for k in range(len(fragments)):
        ax.scatter(pos[k,
                       0],
                   pos[k,
                       1],
                   c=[palette[k]],
                   s=200,
                   edgecolors="black",
                   linewidths=0.7,
                   zorder=3)
        ax.annotate(fragments[k].fid,
                    (pos[k,
                         0],
                     pos[k,
                         1]),
                    fontsize=6,
                    ha="center",
                    va="bottom",
                    xytext=(0,
                            7),
                    textcoords="offset points")
    ax.set_aspect("equal")
    clean2d(ax)
    ax.set_title(
        f"Stage 12: Topological Adjacency Network\n({len(fragments)} Nodes, {len(seen)} Edges)",
        fontsize=11,
        fontweight="bold")
    save(fig, outdir / "10_adjacency_graph.png")


def viz_11_final(asm_p, asm_c, exp_p, exp_c, mesh, outdir, view_angles):
    lo, hi = mesh.bounds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5),
                                   subplot_kw={"projection": "3d"})
    pc(ax1, np.vstack(asm_p), np.vstack(asm_c))
    clean3d(
        ax1,
        lo,
        hi,
        "Stage 11: Final Virtual Assembly\n(Distinct Palette Colors & Generated Fills)",
        view_angles=view_angles,
        zoom=ZOOM)

    pad = np.linalg.norm(hi - lo) * 0.18
    pc(ax2, np.vstack(exp_p), np.vstack(exp_c))
    clean3d(
        ax2,
        lo - pad,
        hi + pad,
        "Stage 11: Exploded Fragmentation View\n(Original Textures & Geometries)",
        view_angles=view_angles,
        zoom=ZOOM)

    save(fig, outdir / "11_final_views.png")


#  driver
def pair_plane_data(mesh, seeds, weights, pair, face_idx):
    i, j = pair
    nv = seeds[j] - seeds[i]
    dd = float(np.linalg.norm(nv))
    if dd < 1e-12:
        return None
    nv /= dd
    mid = pipe.power_mid(seeds[i], seeds[j], weights[i], weights[j])
    segs, norms = pipe.plane_segments(mesh, face_idx, nv, mid)
    if len(segs):
        m = 0.5 * (segs[:, 0] + segs[:, 1])
        nn = pipe.weighted_two_nearest(m, seeds, weights)
        keep = np.all(nn == np.array(sorted((int(i), int(j)))), axis=1)
        segs, norms = segs[keep], norms[keep]
        segs = segs[~(np.abs(norms @ nv) > 0.75)]
    u, v = pipe.make_plane_basis(nv)
    return nv, mid, u, v, segs


def main(argv=None):
    p = argparse.ArgumentParser(
        description=
        "Pipeline stage visualizer (v10, interactive point-cloud renders)")
    p.add_argument("--input",
                   "-i",
                   default=r"pottery/AS0001(1).glb",
                   type=Path)
    p.add_argument("--output", "-o", default="pipeline_viz", type=Path)
    p.add_argument("--fragments", "-n", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-samples", type=int, default=1_000_000)
    cli = p.parse_args(argv)

    args = pipe.build_parser().parse_args([
        "--input",
        str(cli.input),
        "--output",
        str(Path(cli.output) / "_tmp"),
        "--fragments",
        str(cli.fragments),
        "--seed",
        str(cli.seed),
        "--num-samples",
        str(cli.num_samples)
    ])

    outdir = Path(cli.output).expanduser()
    (outdir / "samples").mkdir(parents=True, exist_ok=True)
    rng = pipe.rng_from_seed(args.seed)

    t0 = time.time()
    print(f"[viz] loading {cli.input}")
    mesh = trimesh.load(str(cli.input), force="scene")
    if isinstance(mesh, trimesh.Scene):
        geoms = [
            g for g in mesh.geometry.values()
            if isinstance(g, trimesh.Trimesh)
        ]
        mesh = geoms[0].copy() if len(
            geoms) == 1 else trimesh.util.concatenate(geoms)
    mesh.process(validate=True)
    mesh.merge_vertices()
    if len(mesh.faces) == 0:
        print("ERROR: no faces")
        return 1
    src_uv, src_image = pipe.capture_source_texture(mesh)
    colors = pipe.extract_vertex_colors(mesh, src_uv, src_image)
    diag = pipe.bbox_diagonal(mesh)

    # 1. INTERACTIVE CAMERA CAPTURE
    global_view = get_user_view_mesh(mesh, "Set Global Camera Angle")

    print("[viz] Stage 1-2: point cloud")
    pc_pts, pc_cols, pc_fidx = pipe.sample_surface_points_colors(
        mesh, colors, args.num_samples, rng)
    area = float(mesh.area)
    spacing = float(np.sqrt(area / max(len(pc_pts),
                                       1))) if area > 1e-18 else diag / 1000
    viz_01_input(pc_pts, pc_cols, outdir, global_view)
    viz_02_density(pc_pts, pc_cols, outdir, global_view)

    print("[viz] Stage 3: stress + thickness")
    bundle = pipe.compute_stress(mesh, args, diag)
    s_hat, thick, t_ref, curv = bundle["s_hat"], bundle["thick"], bundle["t_ref"], bundle["curv"]
    P = mesh.triangles_center
    score = s_hat.copy()
    if thick is not None and t_ref:
        score *= pipe.thickness_gain(thick,
                                     t_ref,
                                     args.thickness_alpha,
                                     args.thickness_cap)
    if args.noise_sigma > 0:
        base_shard = diag / max(args.fragments, 2)**(1 / 3)
        noise = pipe.bandlimited_noise(P,
                                       pipe.rng_child(args.seed,
                                                      101),
                                       base_shard * args.noise_length_factor,
                                       n_freq=args.noise_freqs)
        score *= np.exp(args.noise_sigma * noise)
    e = np.sort(np.asarray(mesh.edges), axis=1)
    uniq_e, cnt_e = np.unique(e, axis=0, return_counts=True)
    rim_pts = np.asarray(mesh.vertices)[uniq_e[cnt_e == 1]].mean(1) if (
        cnt_e == 1).any() else None
    rim_bias = (1 + args.rim_weight * np.exp(-cKDTree(rim_pts).query(P)[0] /
                max(1e-9, args.rim_length_scale * diag))) if rim_pts is not None and len(rim_pts) \
        else np.ones(len(P))
    curv_bias = 1 + args.curv_bias_weight * curv
    if thick is not None:
        t_floor = max(float(np.quantile(thick[thick > 0], 0.05)), 1e-6)
        thin_bias = 1 + args.thin_bias_weight * pipe.norm01(
            1 / np.clip(thick,
                        t_floor,
                        None))
    else:
        thin_bias = np.ones(len(P))
    score = pipe.norm01(score * rim_bias * curv_bias * thin_bias)
    score_pts = score[pc_fidx]
    viz_03_stress(pc_pts, score_pts, outdir, global_view)
    viz_04_thickness(pc_pts,
                     thick[pc_fidx] if thick is not None else None,
                     outdir,
                     global_view)

    print("[viz] Stage 4-5: seeds + labels")
    sidx = pipe.nms_maxima(score,
                           P,
                           diag / (2.2 * max(args.fragments,
                                             2)**(1 / 3)),
                           args.fragments,
                           rng=rng,
                           radius_sigma=args.seed_radius_sigma,
                           score_noise=args.seed_score_noise)
    seeds = P[sidx]
    nS = len(seeds)
    viz_05_seeds(pc_pts,
                 score_pts,
                 seeds,
                 np.asarray(mesh.face_normals)[sidx],
                 outdir,
                 global_view)
    if args.size_sigma > 0 and nS > 1:
        base_sp = diag / max(args.fragments, 2)**(1 / 3)
        factors = pipe.rng_child(args.seed,
                                 202).lognormal(0,
                                                args.size_sigma,
                                                nS)
        factors /= factors.mean()
        weights = np.clip(args.power_weight_scale * base_sp**2 * factors,
                          0,
                          0.35 * base_sp**2)
    else:
        weights = np.zeros(nS)
    pc_labels = pipe.weighted_labels(pc_pts, seeds, weights)
    face_labels = pipe.weighted_labels(P, seeds, weights)
    viz_06_labels(pc_pts, pc_labels, nS, outdir, global_view)

    print("[viz] Stage 6-7: candidate pairs")
    F_all = np.asarray(mesh.faces, dtype=np.int64)
    V_all = np.asarray(mesh.vertices, dtype=float)
    FV = V_all[F_all]
    lo_b, hi_b = mesh.bounds
    corners = np.array([[x,
                         y,
                         z] for x in (lo_b[0], hi_b[0])
                        for y in (lo_b[1], hi_b[1])
                        for z in (lo_b[2], hi_b[2])])
    boundary_faces = {}
    for i in range(nS):
        for j in range(i + 1, nS):
            dv = seeds[j] - seeds[i]
            dd = float(np.linalg.norm(dv))
            if dd < 1e-12:
                continue
            n = dv / dd
            mid = pipe.power_mid(seeds[i], seeds[j], weights[i], weights[j])
            sdc = (corners - mid) @ n
            if sdc.min() > 1e-9 or sdc.max() < -1e-9:
                continue
            sd = (FV - mid) @ n
            fi = np.flatnonzero((sd.min(1) <= 1e-9) & (sd.max(1) >= -1e-9))
            if len(fi):
                boundary_faces[(i, j)] = fi
    print(f"    candidate pairs: {len(boundary_faces)}")

    best_pair, best_data = None, None
    for pair, fidx in boundary_faces.items():
        data = pair_plane_data(mesh, seeds, weights, pair, fidx)
        if data is None:
            continue
        if best_data is None or len(data[4]) > len(best_data[4]):
            best_pair, best_data = pair, data

    if best_data is not None and len(best_data[4]) >= 2:
        nv, mid, u, v, segs = best_data
        fidx = boundary_faces[best_pair]
        print(f"    viz pair {best_pair} with {len(segs)} kept segments")

        # 2. INTERACTIVE PLANE-CUT CAMERA CAPTURE
        plane_view = get_user_view_plane(pc_pts,
                                         segs,
                                         mid,
                                         u,
                                         v,
                                         seeds,
                                         best_pair,
                                         "Set Plane Cut Camera Angle")

        viz_07_plane(pc_pts,
                     pc_cols,
                     pc_labels,
                     segs,
                     nv,
                     mid,
                     u,
                     v,
                     seeds,
                     best_pair,
                     outdir,
                     view_angles=plane_view)
        fill_result = viz_08_fill(segs, nv, mid, u, v, spacing, diag, outdir)
        t_local = float(np.median(thick[fidx])) \
            if thick is not None else float(t_ref if t_ref else 4 * spacing)

        # 2-panel 2D/3D Terrain Generation
        viz_09_terrain(fill_result,
                       t_local,
                       spacing,
                       outdir,
                       args,
                       args.seed,
                       pc_pts,
                       pc_labels,
                       best_pair)
    else:
        print(
            "    ! no pair produced kept segments; skipping plane/terrain figures"
        )

    print("[viz] full fill pass + assembly")
    fill_dict, pair_stats, n_fill_total = {}, {}, 0
    stride = max(1, len(pc_pts) // 200_000)
    tree_surf = cKDTree(pc_pts[::stride])
    tree_seeds = cKDTree(seeds)
    for (i, j), fidx in sorted(boundary_faces.items()):
        nv = seeds[j] - seeds[i]
        dd = float(np.linalg.norm(nv))
        if dd < 1e-12:
            continue
        nv /= dd
        mid = pipe.power_mid(seeds[i], seeds[j], weights[i], weights[j])
        t_loc = float(np.median(thick[fidx])) if thick is not None else \
            float(t_ref if t_ref else 4 * spacing)
        wall_px = max(2.0, t_loc / max(spacing, 1e-9))
        hcfg = dict(rng=pipe.rng_child(args.seed,
                                       300 + i,
                                       400 + j),
                    H=args.height_H,
                    octaves=args.height_octaves,
                    amp=args.height_amp_scale * t_loc,
                    grain=args.grain_amp_scale * t_loc,
                    fade_px=max(2.0,
                                args.height_fade_fraction * wall_px),
                    wavelength_px=1.5 * wall_px)
        ccfg = dict(model=args.fill_color_model,
                    tree=tree_surf,
                    t_local=t_loc,
                    surface_color=np.asarray(args.surface_color,
                                             float),
                    core_color=np.asarray(args.core_color,
                                          float))
        chcfg = dict(prob=args.chip_prob,
                     width_px=max(1.0,
                                  args.chip_width_scale))
        fp, fc, st = pipe.plane_cross_fill(mesh, fidx, nv, mid, spacing,
                                           np.asarray(args.fill_color, float), rng,
                                           seeds, weights, i, j, diag, tag="",
                                           tree_seeds=tree_seeds, height_cfg=hcfg,
                                           color_cfg=ccfg, chip_cfg=chcfg)
        st["t_local"] = t_loc
        st["plane_normal"] = nv.tolist()
        st["plane_midpoint"] = mid.tolist()
        pair_stats[(i, j)] = st
        if fp is None:
            continue
        fill_dict.setdefault(i, []).append((fp, fc))
        fill_dict.setdefault(j, []).append((fp, fc))
        n_fill_total += len(fp)

    palette = pipe.fragment_palette(max(nS,
                                        1),
                                    sat=args.palette_sat,
                                    val=args.palette_val)
    center = mesh.bounds.mean(0)
    expl = args.explode_distance or args.explode_scale * diag
    asm_p, asm_c, exp_p, exp_c = [], [], [], []
    fragments, node_by_seed = [], {}
    for i in range(nS):
        idx_f = np.where(face_labels == i)[0]
        if len(idx_f) < max(4, int(args.min_faces)):
            continue
        vids, inv = np.unique(F_all[idx_f].ravel(), return_inverse=True)
        centroid = V_all[vids].mean(0)
        dv = centroid - center
        nn = float(np.linalg.norm(dv))
        off = (dv / nn if nn > 1e-9 else pipe.random_unit_vector(rng)) * expl
        m = pc_labels == i
        pts, cols = pc_pts[m], pc_cols[m]
        for fp, fc in fill_dict.get(i, []):
            pts = np.vstack([pts, fp])
            cols = np.vstack([cols, fc])
        asm_p.append(pts)
        asm_c.append(
            pipe.shade_palette(palette[i],
                               cols,
                               args.assembly_color_mode))
        exp_p.append(pts + off)
        exp_c.append(cols)
        fid = f"frag_{len(fragments):03d}"
        node_by_seed[i] = type("N",
                               (),
                               {
                                   "fid": fid,
                                   "centroid": centroid,
                                   "explode_offset": off
                               })()
        fragments.append(
            type(
                "Fr",
                (),
                {
                    "fid": fid,
                    "seed_index": i,
                    "centroid": centroid,
                    "n_faces": len(idx_f),
                    "area": float(mesh.area_faces[idx_f].sum())
                })())

    viz_10_adjacency(fragments, pair_stats, node_by_seed, outdir)
    if asm_p:
        viz_11_final(asm_p, asm_c, exp_p, exp_c, mesh, outdir, global_view)
        rgba_a = np.column_stack([(np.clip(np.vstack(asm_c),
                                           0,
                                           1) * 255).astype(np.uint8),
                                  np.full((sum(len(a) for a in asm_p),
                                           1),
                                          255,
                                          dtype=np.uint8)])
        trimesh.PointCloud(
            np.vstack(asm_p),
            rgba_a).export(str(outdir / "samples" / "stage11_assembled.ply"))
        rgba_e = np.column_stack([(np.clip(np.vstack(exp_c),
                                           0,
                                           1) * 255).astype(np.uint8),
                                  np.full((sum(len(a) for a in exp_p),
                                           1),
                                          255,
                                          dtype=np.uint8)])
        trimesh.PointCloud(
            np.vstack(exp_p),
            rgba_e).export(str(outdir / "samples" / "stage11_exploded.ply"))

    print(f"\n[viz] done in {time.time()-t0:.1f}s -> {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
