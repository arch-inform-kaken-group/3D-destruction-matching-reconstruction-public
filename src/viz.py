"""
viz_physics.py — pipeline stage visualizer for physics_based_destruction.py

Mirrors process_file() stage-for-stage (same functions, same order, same seeds),
so every figure corresponds exactly to the exported PLY/GLB/JSON outputs.

Figures produced (under --output/):
  01_input_mesh.png        original artifact point cloud (source colors)
  02_sampling_density.png  kNN spacing uniformity
  03_voxel_graph.png       voxelized solid (wall strength) + admissible fault graph (η)
  04_fracture_modes.png    first k fracture modes U_i on voxels (diverging)
  05_impact_projection.png (only --fracture impact) gaussian impact w and projection w*
  06_fracture_affinity.png surface coloured by modes' discontinuity field
  07_fragment_labels.png   prefracture connected components (tab20)
  08_noise_swatches.png    ALL registered relief noises (the swappable terrain set)
  09_interface_fills.png   open break surfaces filled with selected noise (fresh core colours)
  10_weathering.png        shell before vs after HSV weathering (+ saturation histograms)
  11_gaps_and_chips.png    rim material removed per fragment (red) vs retained (grey)
  12_adjacency_graph.png   shard join graph (PCA layout) + per-sherd mean gap bar chart
  13_final_assembly.png    assembled (palette) vs exploded (weathered) views
  14_fragment_strip.png    individual unearthed sherds
  samples/*.ply            point-cloud exports of key stages

Usage:
  python viz_physics.py --input pottery/AS0001(1).glb --output pipeline_viz/
  python viz_physics.py -i m.glb -o viz/ --relief-noise worley --no-interactive
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import physics_based_destruction as pipe

MCM = matplotlib.colormaps
MAX_SCATTER = 120_000
VIEW = dict(elev=22, azim=-90)
ZOOM = 1.25


# ------------------------------------------------------------------ helpers
def _sub(pts, cols, n):
    pts = np.asarray(pts)
    if len(pts) <= n:
        return pts, (None if cols is None else np.asarray(cols))
    idx = np.random.default_rng(0).choice(len(pts), n, replace=False)
    return pts[idx], (None if cols is None else np.asarray(cols)[idx])


def pc(ax, pts, cols, s=0.6, n=MAX_SCATTER):
    p, c = _sub(pts, cols, n)
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=np.clip(c, 0, 1), s=s,
               edgecolors="none", depthshade=False)


def clean3d(ax, lo, hi, title, view_angles=None, zoom=ZOOM):
    ax.set_axis_off()
    if view_angles is not None:
        elev, azim = view_angles[0], view_angles[1]
        if len(view_angles) >= 3 and hasattr(ax, "roll"):
            ax.view_init(elev=elev, azim=azim, roll=view_angles[2])
        else:
            ax.view_init(elev=elev, azim=azim)
    else:
        ax.view_init(**VIEW)
    c = (lo + hi) * 0.5
    m = (float(((hi - lo) * 0.5).max()) * 1.02) / max(zoom, 1e-9)
    ax.set_xlim(c[0] - m, c[0] + m); ax.set_ylim(c[1] - m, c[1] + m)
    ax.set_zlim(c[2] - m, c[2] + m)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title, fontsize=9)


def save(fig, path):
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def cmap(name, n=None):
    c = MCM[name]
    return c.resampled(n) if n else c


def get_user_view(pts, title, interactive):
    """Rotate the window, close it to lock the camera for all later panels."""
    if not interactive:
        return None
    print(f"\n[!] ACTION REQUIRED: {title}")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    p, _ = _sub(pts, None, 15_000)
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.5, c="gray", alpha=0.35, depthshade=False)
    clean3d(ax, pts.min(0), pts.max(0),
            f"{title}\n(Rotate with mouse, then close the window to confirm)")
    state = {"elev": VIEW["elev"], "azim": VIEW["azim"], "roll": 0}

    def on_draw(_e):
        state["elev"], state["azim"] = ax.elev, ax.azim
        state["roll"] = getattr(ax, "roll", 0)

    fig.canvas.mpl_connect("draw_event", on_draw)
    plt.show(block=True)
    print(f"  -> captured elev={state['elev']:.1f} azim={state['azim']:.1f}")
    return (state["elev"], state["azim"], state["roll"])


# ------------------------------------------------------------------ panels
def viz_01(pc_pts, pc_cols, outdir, va):
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(projection="3d")
    pc(ax, pc_pts, pc_cols)
    clean3d(ax, pc_pts.min(0), pc_pts.max(0),
            f"Stage 1: Original Artifact\nPoint Cloud ({len(pc_pts):,} pts)", va)
    save(fig, outdir / "01_input_mesh.png")


def viz_02(pc_pts, outdir, va):
    pts, _ = _sub(pc_pts, None, 40_000)
    d = cKDTree(pts).query(pts, k=6)[0][:, -1]
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(projection="3d")
    pc(ax, pts, cmap("viridis")(pipe.norm01(d))[:, :3], n=10**9)
    clean3d(ax, pts.min(0), pts.max(0),
            "Stage 2: Sampling Density\n(6-NN distance, uniformity check)", va)
    save(fig, outdir / "02_sampling_density.png")


def viz_03(centers, E, eta, strength, lo, hi, outdir, va):
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    p, s = _sub(centers, strength, MAX_SCATTER)
    pc(ax, p, cmap("viridis")(pipe.norm01(s))[:, :3], s=1.2, n=10**9)
    clean3d(ax, lo, hi, f"Stage 3a: Voxel Solid ({len(centers):,} vox)\ncolour = wall strength", va)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    idx = np.random.default_rng(1).choice(len(E), min(30_000, len(E)), replace=False)
    segs = np.stack([centers[E[idx, 0]], centers[E[idx, 1]]], 1)
    lc = Line3DCollection(segs, colors=cmap("hot")(pipe.norm01(eta[idx]))[:, :3],
                          linewidths=0.35)
    ax.add_collection3d(lc)
    q, _ = _sub(centers, None, 20_000)
    ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=0.3, c="0.85", depthshade=False)
    clean3d(ax, lo, hi, f"Stage 3b: Admissible Fault Graph ({len(E):,} edges)\n"
                        f"colour = weakness η (thin walls hot)", va)
    save(fig, outdir / "03_voxel_graph.png")


def viz_04(centers, Us, Js, lo, hi, outdir, va):
    km = min(4, len(Us))
    if km == 0:
        return
    cols_, rows_ = (2, 2) if km >= 3 else (2, 1)
    fig = plt.figure(figsize=(6.5 * cols_, 6 * rows_))
    for i in range(km):
        ax = fig.add_subplot(rows_, cols_, i + 1, projection="3d")
        p, u = _sub(centers, Us[i], MAX_SCATTER)
        pc(ax, p, cmap("coolwarm")(pipe.norm01(u))[:, :3], s=1.0, n=10**9)
        clean3d(ax, lo, hi, f"Stage 4: Fracture Mode U_{i+1}\n(piecewise-rigid, sparse jumps)", va)
    save(fig, outdir / "04_fracture_modes.png")


def viz_05(centers, w, ws, lo, hi, outdir, va):
    fig = plt.figure(figsize=(13, 6))
    for j, (fld, ttl) in enumerate([(w, "impact field w (gaussian)"),
                                    (ws, "projected impact w* = Σ U_i U_iᵀ M w")]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        p, f = _sub(centers, fld, MAX_SCATTER)
        pc(ax, p, cmap("magma")(pipe.norm01(f))[:, :3], s=1.0, n=10**9)
        clean3d(ax, lo, hi, f"Stage 4b: {ttl}", va)
    save(fig, outdir / "05_impact_projection.png")


def viz_06(pc_pts, score, outdir, va):
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(projection="3d")
    pc(ax, pc_pts, cmap("hot")(np.clip(score, 0, 1))[:, :3])
    clean3d(ax, pc_pts.min(0), pc_pts.max(0),
            "Stage 5: Fracture Affinity\n(max mode discontinuity per surface point)", va)
    save(fig, outdir / "06_fracture_affinity.png")


def viz_07(pc_pts, labels, ncomp, outdir, va):
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(projection="3d")
    pc(ax, pc_pts, cmap("tab20", max(ncomp, 1))(labels % 20)[:, :3])
    clean3d(ax, pc_pts.min(0), pc_pts.max(0),
            f"Stage 6: Prefracture Components ({ncomp})\nmode fault-set intersection", va)
    save(fig, outdir / "07_fragment_labels.png")


def viz_08(selected, outdir):
    names = list(pipe.NOISE_REGISTRY)
    nc = int(np.ceil(np.sqrt(len(names))))
    nr = int(np.ceil(len(names) / nc))
    gx, gy = np.meshgrid(np.linspace(0, 8, 200), np.linspace(0, 8, 200))
    q = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
    fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.2 * nr))
    axes = np.atleast_1d(axes).ravel()
    for axi, nm in zip(axes, names):
        v = pipe.relief_field(nm, q, 1.0, np.random.default_rng(7), 0.78, 5)
        axi.imshow(v.reshape(gx.shape), cmap="terrain", origin="lower")
        axi.set_title(nm + ("  ← selected" if nm == selected else ""), fontsize=9,
                      color=("crimson" if nm == selected else "k"))
        axi.set_xticks([]); axi.set_yticks([])
    for axi in axes[len(names):]:
        axi.set_axis_off()
    fig.suptitle("Stage 7a: Swappable fracture-surface relief noises (terrain registry)",
                 fontsize=10)
    save(fig, outdir / "08_noise_swatches.png")


def viz_09(fills, lo, hi, outdir, va, kind):
    pts = np.vstack([v["pts"] for v in fills.values()]) if fills else np.zeros((0, 3))
    cols = np.vstack([v["cols"] for v in fills.values()]) if fills else np.zeros((0, 3))
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(projection="3d")
    pc(ax, pts, cols, s=0.8)
    clean3d(ax, lo, hi, f"Stage 7b: Open Break Surfaces Filled\nrelief noise = '{kind}', "
                       f"depth-core colouring ({len(pts):,} pts)", va)
    save(fig, outdir / "09_interface_fills.png")


def viz_10(pc_pts, raw_cols, wet_cols, outdir, va):
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(1, 3, 1, projection="3d"); pc(ax, pc_pts, raw_cols)
    clean3d(ax, pc_pts.min(0), pc_pts.max(0), "Stage 8a: freshly fractured surface", va)
    ax = fig.add_subplot(1, 3, 2, projection="3d"); pc(ax, pc_pts, wet_cols)
    clean3d(ax, pc_pts.min(0), pc_pts.max(0), "Stage 8b: after burial weathering\n"
            "(hue drift, desaturation, staining, deposits)", va)
    ax = fig.add_subplot(1, 3, 3)
    for c, lb, col in [(raw_cols, "before", "0.4"), (wet_cols, "after", "sienna")]:
        s = pipe.rgb_to_hsv(np.clip(c, 0, 1))[:, 1]
        ax.hist(s, bins=40, range=(0, 1), histtype="step", lw=1.6, label=lb, color=col)
    ax.legend(fontsize=8); ax.set_xlabel("saturation"); ax.set_yticks([])
    pipe_clean2d(ax)
    ax.set_title("Stage 8c: saturation loss", fontsize=9)
    save(fig, outdir / "10_weathering.png")


def pipe_clean2d(ax):
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def viz_11(kept, removed, lo, hi, outdir, va):
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(projection="3d")
    k, _ = _sub(kept, None, 70_000)
    ax.scatter(k[:, 0], k[:, 1], k[:, 2], s=0.4, c="0.75", depthshade=False)
    if len(removed):
        r, _ = _sub(removed, None, 40_000)
        ax.scatter(r[:, 0], r[:, 1], r[:, 2], s=1.6, c="crimson", depthshade=False)
    clean3d(ax, lo, hi, f"Stage 9: Gaps & Chips\nred = rim material lost per sherd "
                       f"({len(removed):,} pts) → contours no longer match", va)
    save(fig, outdir / "11_gaps_and_chips.png")


def viz_12(nodes, edges, palette, gaps, outdir):
    if not nodes:
        return
    C = np.array([n["centroid"] for n in nodes])
    Cc = C - C.mean(0)
    _, _, Vt = np.linalg.svd(Cc, full_matrices=False)
    P2 = Cc @ Vt[:2].T
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1)
    id2xy = {n["id"]: xy for n, xy in zip(nodes, P2)}
    areas = np.array([e["interface_area"] for e in edges]) if edges else np.zeros(0)
    amax = areas.max() if len(areas) else 1.0
    for e in edges:
        a, b = id2xy[e["a"]], id2xy[e["b"]]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="0.5",
                lw=0.6 + 2.4 * e["interface_area"] / max(amax, 1e-9), zorder=1)
    for n, xy in zip(nodes, P2):
        ax.scatter(*xy, s=18 + 260 * n["n_points"] / max(max(m["n_points"] for m in nodes), 1),
                   color=palette[n["label"] % len(palette)], edgecolors="k", lw=0.6, zorder=2)
        ax.annotate(n["id"], xy, fontsize=6, xytext=(3, 3), textcoords="offset points")
    pipe_clean2d(ax); ax.set_title("Stage 10: shard adjacency graph\n(edge width = interface area)",
                                   fontsize=9)
    ax = fig.add_subplot(1, 2, 2)
    order = np.argsort(gaps)[::-1]
    ax.bar(range(len(gaps)), np.asarray(gaps)[order], color="sienna")
    ax.set_xlabel("sherd (sorted)"); ax.set_ylabel("mean rim loss")
    pipe_clean2d(ax); ax.set_title("Stage 10b: per-sherd gap magnitude", fontsize=9)
    save(fig, outdir / "12_adjacency_graph.png")


def viz_13(asm_p, asm_c, exp_p, exp_c, lo, hi, outdir, va):
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    pc(ax, np.vstack(asm_p), np.vstack(asm_c), s=0.5)
    clean3d(ax, lo, hi, "Stage 11a: reassembled sherds\n(palette IDs, shaded)", va)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    pc(ax, np.vstack(exp_p), np.vstack(exp_c), s=0.5)
    clean3d(ax, lo, hi, "Stage 11b: exploded unearthed sherds\n(weathered colours + fills)", va)
    save(fig, outdir / "13_final_assembly.png")


def viz_14(frags, outdir, va):
    show = sorted(frags, key=lambda f: -len(f["pts"]))[:8]
    if not show:
        return
    nc = min(4, len(show)); nr = int(np.ceil(len(show) / nc))
    fig = plt.figure(figsize=(3.6 * nc, 3.4 * nr))
    for i, f in enumerate(show):
        ax = fig.add_subplot(nr, nc, i + 1, projection="3d")
        pc(ax, f["pts"], f["cols"], s=0.5)
        clean3d(ax, f["pts"].min(0), f["pts"].max(0),
                f"sherd {f['label']}  ({len(f['pts']):,} pts,\ngap {f['gap_mean']:.3g})",
                va, zoom=1.05)
    save(fig, outdir / "14_fragment_strip.png")


# ------------------------------------------------------------------ driver
def main(argv=None):
    p = argparse.ArgumentParser(description="Physics-based destruction pipeline visualizer")
    p.add_argument("--input", "-i", default=r"pottery/AS0001(1).glb", type=Path)
    p.add_argument("--output", "-o", default="pipeline_viz", type=Path)
    p.add_argument("--num-samples", type=int, default=800_000)
    p.add_argument("--fragments", "-n", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid-res", type=int, default=48)
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--fracture", choices=["modes", "impact"], default="modes")
    p.add_argument("--relief-noise", choices=list(pipe.NOISE_REGISTRY), default="fbm")
    p.add_argument("--weather", type=float, default=0.6)
    p.add_argument("--gap-scale", type=float, default=0.30)
    p.add_argument("--no-interactive", action="store_true")
    cli = p.parse_args(argv)

    interactive = (not cli.no_interactive) and \
                  not matplotlib.get_backend().lower().endswith("agg")
    if not interactive:
        matplotlib.use("Agg")

    args = pipe.build_parser().parse_args([
        "--input", str(cli.input), "--output", str(cli.output / "_tmp"),
        "--num-samples", str(cli.num_samples), "--fragments", str(cli.fragments),
        "--seed", str(cli.seed), "--grid-res", str(cli.grid_res),
        "--modes", str(cli.modes), "--fracture", cli.fracture,
        "--relief-noise", cli.relief_noise, "--weather", str(cli.weather),
        "--gap-scale", str(cli.gap_scale)])

    outdir = Path(cli.output).expanduser()
    (outdir / "samples").mkdir(parents=True, exist_ok=True)
    rng = pipe.rng_from_seed(args.seed)
    t0 = time.time()

    print(f"[viz] loading {cli.input}")
    mesh = trimesh.load(str(cli.input), force="scene")
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = geoms[0].copy() if len(geoms) == 1 else trimesh.util.concatenate(geoms)
    mesh.process(validate=True); mesh.merge_vertices()
    colors = pipe.extract_vertex_colors(mesh)
    diag = pipe.bbox_diagonal(mesh)
    lo, hi = mesh.bounds
    center = mesh.bounds.mean(0)

    va = get_user_view(np.asarray(mesh.vertices), "Set Global Camera Angle", interactive)

    print("[viz] stage 1-2: sampling")
    pc_pts, pc_cols, pc_fi = pipe.sample_surface_points_colors(mesh, colors,
                                                               args.num_samples, rng)
    spacing = float(np.sqrt(mesh.area / max(len(pc_pts), 1)))
    viz_01(pc_pts, pc_cols, outdir, va)
    viz_02(pc_pts, outdir, va)
    pipe.save_pc if hasattr(pipe, "save_pc") else None

    print("[viz] stage 3: voxel graph")
    centers, E, eax, pitch, strength = pipe.build_voxel_graph(mesh, args.grid_res)
    eta = (1e-3 + np.minimum(strength[E[:, 0]], strength[E[:, 1]])) ** args.weak_gamma
    viz_03(centers, E, eta, strength, lo, hi, outdir, va)

    print("[viz] stage 4: fracture modes")
    k = max(2, min(args.modes, max(2, len(centers) // 200)))
    Us, Js = pipe.fracture_modes(centers, E, eta, k, args.irls_iters,
                                 pipe.rng_child(args.seed, 11))
    viz_04(centers, Us, Js, lo, hi, outdir, va)
    if cli.fracture == "impact" and Us:
        w = np.exp(-(np.linalg.norm(centers - np.asarray(args.impact_pos), axis=1) /
                     max(args.impact_radius or diag / 6.0, 1e-9)) ** 2)
        Um = np.stack(Us).T
        viz_05(centers, w, Um @ (Um.T @ w), lo, hi, outdir, va)

    print("[viz] stage 5-6: prefracture")
    if cli.fracture == "impact":
        comp, d, vox_score = pipe.impact_fracture(centers, E, Us, args.impact_pos,
                                                  args.impact_radius or diag / 6.0,
                                                  args.impact_sigma)
    else:
        comp, d, vox_score = pipe.prefracture(centers, E, Js, args.fragments,
                                              args.min_vox, rng)
    ncomp = int(comp.max()) + 1
    tree_vox = cKDTree(centers)
    pc_vox = tree_vox.query(pc_pts)[1]
    pc_lab = comp[pc_vox]
    face_lab = comp[tree_vox.query(np.asarray(mesh.triangles_center))[1]]
    viz_06(pc_pts, vox_score[pc_vox], outdir, va)
    viz_07(pc_pts, pc_lab, ncomp, outdir, va)

    print("[viz] stage 7: interface fills")
    stride = max(1, len(pc_pts) // 200_000)
    tree_surf = cKDTree(pc_pts[::stride])
    hcfg = dict(H=args.relief_H, octaves=args.relief_octaves,
                amp_scale=args.relief_amp_scale, grain_scale=args.grain_amp_scale,
                fade_fraction=args.fade_fraction,
                wavelength_scale=args.relief_wavelength_scale)
    ccfg = dict(model=args.fill_color_model,
                surface_color=np.asarray(args.surface_color, float),
                core_color=np.asarray(args.core_color, float),
                fill_color=np.asarray(args.fill_color, float))
    fills = pipe.interface_fill(comp, E, eax, centers, pitch, strength, spacing,
                                tree_surf, args.relief_noise, hcfg, ccfg,
                                pipe.rng_child(args.seed, 300))
    viz_08(args.relief_noise, outdir)
    viz_09(fills, lo, hi, outdir, va, args.relief_noise)

    print("[viz] stage 8-9: weathering + gaps")
    frags = []
    for i in range(ncomp):
        m = pc_lab == i
        fm = face_lab == i
        if fm.sum() < max(4, args.min_faces) or m.sum() < 64:
            continue
        frags.append(dict(label=i, pts=pc_pts[m].copy(), cols=pc_cols[m].copy(),
                          faces=np.flatnonzero(fm), gap_mean=0.0))
    t_ref = float(2.0 * np.median(strength[strength > 0])) if (strength > 0).any() \
        else 4 * spacing
    shell_pts = np.vstack([f["pts"] for f in frags])
    raw_cols = np.vstack([f["cols"] for f in frags])
    wrng = pipe.rng_child(args.seed, 500)
    wcfg = pipe.WeatherCfg(strength=args.weather, hue=args.weather_hue,
                           desat=args.weather_desat, stain=args.weather_stain,
                           bleach=args.weather_bleach,
                           deposit_cover=args.weather_deposit_cover,
                           deposit_amp=args.weather_deposit_amp)
    wet_cols = pipe.weather_colors(raw_cols, shell_pts, wrng, wcfg, diag / 6.0)
    viz_10(shell_pts, raw_cols, wet_cols, outdir, va)
    off = 0
    for f in frags:
        n = len(f["pts"]); f["cols"] = wet_cols[off:off + n]; off += n
    for v in fills.values():
        v["cols"] = pipe.weather_colors(v["cols"], v["pts"], wrng, wcfg, diag / 6.0,
                                        factor=args.weather_fill_factor)
    pre = [(f["pts"].copy()) for f in frags]
    pipe.apply_gaps_chips(frags, fills, dict(scale=args.gap_scale, noise=args.gap_noise),
                          dict(count=args.chip_count,
                               radius_scale=args.chip_radius_scale),
                          t_ref, diag, args.seed)
    removed = []
    for before, f in zip(pre, frags):
        if len(f["pts"]) == 0:
            removed.append(before); continue
        d = cKDTree(f["pts"]).query(before)[0]
        removed.append(before[d > 1e-7])
    removed = np.vstack([r for r in removed if len(r)]) if any(len(r) for r in removed) \
        else np.zeros((0, 3))
    viz_11(np.vstack([f["pts"] for f in frags]), removed, lo, hi, outdir, va)

    print("[viz] stage 10-11: graph + finals")
    palette = pipe.fragment_palette(max(len(frags), 1), args.palette_sat, args.palette_val)
    expl = args.explode_distance or args.explode_scale * diag
    nodes, edges, asm_p, asm_c, exp_p, exp_c = [], [], [], [], [], []
    id_of = {f["label"]: f"frag_{n:03d}" for n, f in enumerate(frags)}
    for n_i, f in enumerate(frags):
        centroid = f["pts"].mean(0)
        dv = centroid - center
        nn = float(np.linalg.norm(dv))
        offv = (dv / nn if nn > 1e-9 else np.array([0, 0, 1.0])) * expl
        fpts, fcols = [f["pts"]], [f["cols"]]
        for (ci, cj), v in fills.items():
            if f["label"] not in (ci, cj):
                continue
            vk = v["keep"].get(f["label"], np.ones(len(v["pts"]), bool))
            fpts.append(v["pts"][vk]); fcols.append(v["cols"][vk])
            if ci in id_of and cj in id_of and ci == f["label"]:
                edges.append(dict(a=id_of[ci], b=id_of[cj],
                                  interface_area=float(v["area"])))
        fpts, fcols = np.vstack(fpts), np.vstack(fcols)
        asm_p.append(fpts)
        asm_c.append(pipe.shade_palette(palette[n_i], fcols, args.assembly_color_mode))
        exp_p.append(fpts + offv); exp_c.append(fcols)
        nodes.append(dict(id=f"frag_{n_i:03d}", label=int(f["label"]),
                          centroid=centroid.tolist(), n_points=int(len(fpts))))
    viz_12(nodes, edges, palette, [f["gap_mean"] for f in frags], outdir)
    viz_13(asm_p, asm_c, exp_p, exp_c, lo, hi, outdir, va)
    viz_14(frags, outdir, va)

    def export(pts, cols, name):
        rgba = np.column_stack([(np.clip(cols, 0, 1) * 255).astype(np.uint8),
                                np.full((len(cols), 1), 255, np.uint8)])
        trimesh.PointCloud(np.asarray(pts, float), colors=rgba).export(
            str(outdir / "samples" / name))
    export(pc_pts, pc_cols, "stage01_pointcloud.ply")
    export(pc_pts, cmap("tab20", max(ncomp, 1))(pc_lab % 20)[:, :3], "stage06_labels.ply")
    export(np.vstack(asm_p), np.vstack(asm_c), "stage11_assembled.ply")
    export(np.vstack(exp_p), np.vstack(exp_c), "stage11_exploded.ply")

    (outdir / "viz_summary.json").write_text(json.dumps(
        dict(input=str(cli.input), seed=args.seed, n_fragments=len(frags),
             n_modes=len(Us), relief_noise=args.relief_noise, weather=args.weather,
             gap_scale=args.gap_scale, interactive=bool(interactive),
             seconds=round(time.time() - t0, 1)), indent=2))
    print(f"\n[viz] done in {time.time()-t0:.1f}s -> {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())