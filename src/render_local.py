"""
Local-PC / GPU animation of GARF reassembly for Jomon pottery.
- Upright orientation fix (source meshes Y-up -> renderer Z-up).
- GPU point-splatting renderer (torch CUDA); matplotlib CPU fallback.
- NEW: builds a 3x3 labeled collage GIF from the first 9 videos.
- NEW: holds the final assembled state for a configurable pause duration.

Deps: numpy scipy trimesh imageio[ffmpeg] pillow  (+ torch for GPU path)
Usage:
  python src/render_local.py --jomon_data ./fragmented_v1 --results ./GARF/logs/GARF-Jomon/jomon_infer/version_0/json_results --out ./animations --device cuda
  # rebuild collage later from existing MP4s (no re-render):
  python src/render_local.py --collage_only --out ./animations
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from PIL import Image, ImageDraw
import imageio

try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

UP_FIX = {
    "z": np.eye(3),
    "y": R.from_euler("x", 90, degrees=True).as_matrix(),
    "x": R.from_euler("y", -90, degrees=True).as_matrix(),
}


# SE(3) helpers (GARF scalar-first quats [w,x,y,z])
def quat_wxyz_to_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def se3_from_vec(v):
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_matrix(v[3:])
    T[:3, 3] = v[:3]
    return T


def slerp(R0, R1, t):
    return Slerp([0, 1], R.concatenate([R.from_matrix(R0), R.from_matrix(R1)]))(t).as_matrix()


def conjugate(T, Q):
    out = np.eye(4)
    out[:3, :3] = Q @ T[:3, :3] @ Q.T
    out[:3, 3] = Q @ T[:3, 3]
    return out


def load_fragment_pcd(ply_path, max_points):
    pc = trimesh.load(str(ply_path))
    pts = np.asarray(pc.vertices, dtype=np.float64)
    try:
        cols = np.asarray(pc.colors, dtype=np.float64)[:, :3] / 255.0
    except Exception:
        cols = None
    if max_points and len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts = pts[sel]
        if cols is not None:
            cols = cols[sel]
    return pts, (cols if cols is not None else np.full((len(pts), 3), 0.62))


def look_at(eye, center, up=np.array([0.0, 0.0, 1.0])):
    f = center - eye
    f /= np.linalg.norm(f) + 1e-12
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-8:
        r = np.cross(f, np.array([0.0, 1.0, 0.0]))
    r /= np.linalg.norm(r) + 1e-12
    u = np.cross(r, f)
    return np.stack([r, u, f])


# GPU point-splatting renderer
class GPURenderer:
    def __init__(self, W, H, device, radius=1, bg=1.0, fov=40.0):
        self.W, self.H, self.dev, self.bg = W, H, device, bg
        self.focal = 0.5 * H / np.tan(np.deg2rad(fov) / 2)
        ys, xs = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        self.off = torch.tensor(np.stack([xs.ravel(), ys.ravel()], -1),
                                device=device, dtype=torch.long)

    @torch.no_grad()
    def render(self, pts, cols, eye, center):
        pts_t = torch.as_tensor(pts, dtype=torch.float32, device=self.dev)
        cols_t = torch.as_tensor(cols, dtype=torch.float32, device=self.dev)
        Rv = torch.as_tensor(look_at(np.asarray(eye, np.float64),
                                     np.asarray(center, np.float64)),
                             dtype=torch.float32, device=self.dev)
        eye_t = torch.as_tensor(np.asarray(eye, np.float32), device=self.dev)
        cam = (pts_t - eye_t) @ Rv.T
        z = cam[:, 2]
        x = cam[:, 0] / z * self.focal + self.W * 0.5
        y = -cam[:, 1] / z * self.focal + self.H * 0.5
        px, py = x.round().long(), y.round().long()
        ok = (z > 1e-4) & (px >= 0) & (px < self.W) & (py >= 0) & (py < self.H)
        px, py, z, cols_t = px[ok], py[ok], z[ok], cols_t[ok]
        N = px.numel()
        HW = self.W * self.H
        if N == 0:
            return np.full((self.H, self.W, 3), self.bg, np.float32)
        K = self.off.shape[0]
        pix = (py.unsqueeze(1) * self.W + px.unsqueeze(1)) + \
              (self.off[:, 1] * self.W + self.off[:, 0]).unsqueeze(0)
        pix = pix.ravel()
        dep = z.unsqueeze(1).expand(N, K).ravel()
        col = cols_t.unsqueeze(1).expand(N, K, 3).reshape(-1, 3)
        inb = (pix >= 0) & (pix < HW)
        pix, dep, col = pix[inb], dep[inb], col[inb]
        depth_buf = torch.full((HW,), float("inf"), device=self.dev)
        depth_buf.scatter_reduce_(0, pix, dep, reduce="amin", include_self=True)
        vis = dep <= depth_buf[pix] + 1e-4
        color_buf = torch.full((HW, 3), self.bg, device=self.dev)
        color_buf[pix[vis]] = col[vis]
        return color_buf.reshape(self.H, self.W, 3).clamp(0, 1).cpu().numpy()


# collage GIF
def video_to_cells(path, cell, n_out):
    """Read an MP4; return n_out uniformly sampled frames resized to (cell, cell)."""
    try:
        reader = imageio.get_reader(str(path), plugin="FFMPEG")
    except Exception:
        reader = imageio.get_reader(str(path))
    frames = []
    for fr in reader:
        frames.append(np.asarray(Image.fromarray(fr).resize((cell, cell), Image.LANCZOS)))
    reader.close()
    if not frames:
        return [np.full((cell, cell, 3), 255, np.uint8)] * n_out
    idx = np.linspace(0, len(frames) - 1, n_out).astype(int)
    return [frames[i] for i in idx]


def build_collage_gif(mp4_paths, out_gif, cell=320, n_frames=40, fps=12,
                      nrows=3, ncols=3, gap=6, bg=255, labels=True):
    """3x3 (nrows x ncols) animated collage GIF from up to 9 videos."""
    slots = list(mp4_paths)[: nrows * ncols]
    seqs = [video_to_cells(p, cell, n_frames) for p in slots]
    names = [Path(p).stem for p in slots]
    while len(seqs) < nrows * ncols:                     # pad missing slots white
        seqs.append([np.full((cell, cell, 3), bg, np.uint8)] * n_frames)
        names.append("")

    W = ncols * cell + (ncols + 1) * gap
    H = nrows * cell + (nrows + 1) * gap
    frames = []
    for t in range(n_frames):
        grid = np.full((H, W, 3), bg, np.uint8)
        for k, seq in enumerate(seqs):
            r, c = divmod(k, ncols)
            y = gap + r * (cell + gap)
            x = gap + c * (cell + gap)
            grid[y:y + cell, x:x + cell] = seq[t]
        if labels:
            img = Image.fromarray(grid)
            draw = ImageDraw.Draw(img)
            for k, nm in enumerate(names):
                if not nm:
                    continue
                r, c = divmod(k, ncols)
                y = gap + r * (cell + gap)
                x = gap + c * (cell + gap)
                draw.text((x + 5, y + 5), nm, fill=(255, 255, 255))   # halo
                draw.text((x + 4, y + 4), nm, fill=(0, 0, 0))         # label
            grid = np.asarray(img)
        frames.append(grid)
    imageio.mimsave(str(out_gif), frames, duration=int(1000 / fps), loop=0)
    print(f"saved collage {out_gif} ({len(slots)} videos, {W}x{H})")


# per-object animation
def render_object(obj_dir, result, out_path, Q, renderer,
                  n_frames=60, fps=30, dpi=300, max_points=30000, pause_seconds=2.0):
    mesh_scale = float(result["mesh_scale"])
    gt = np.asarray(result["gt_trans_rots"], dtype=np.float64)
    pred = np.asarray(result["pred_trans_rots"][-1], dtype=np.float64)
    num_parts = int(result["num_parts"])
    nodes = json.loads((obj_dir / "adjacency.json").read_text())["nodes"]

    part_pts, part_cols, T0, T1 = [], [], [], []
    for i, node in enumerate(nodes[:num_parts]):
        pts, cols = load_fragment_pcd(obj_dir / node["ply"], max_points)
        pts = (pts / mesh_scale) @ Q.T
        inv_gt = np.linalg.inv(se3_from_vec(gt[i]))
        part_pts.append(pts)
        part_cols.append(cols)
        T0.append(conjugate(inv_gt, Q))
        T1.append(conjugate(se3_from_vec(pred[i]) @ inv_gt, Q))

    extremes = []
    for Ts in (T0, T1):
        for i in range(num_parts):
            extremes.append((Ts[i][:3, :3] @ part_pts[i].T).T + Ts[i][:3, 3])
    allp = np.concatenate(extremes)
    center = allp.mean(0)
    half = float(np.abs(allp - center).max()) * 1.15
    dist = half * 2.8

    frames = []
    for f in range(n_frames):
        t = f / max(1, n_frames - 1)
        te = t * t * (3 - 2 * t)
        elev, azim = np.deg2rad(20), np.deg2rad(45 + 60 * te)
        eye = center + dist * np.array([np.cos(elev) * np.cos(azim),
                                        np.cos(elev) * np.sin(azim),
                                        np.sin(elev)])
        if renderer is not None:
            P, C = [], []
            for i in range(num_parts):
                Rt = slerp(T0[i][:3, :3], T1[i][:3, :3], te)
                tt = (1 - te) * T0[i][:3, 3] + te * T1[i][:3, 3]
                P.append((Rt @ part_pts[i].T).T + tt)
                C.append(part_cols[i])
            frames.append((renderer.render(np.concatenate(P), np.concatenate(C),
                                           eye, center) * 255).astype(np.uint8))
        else:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(6, 6), dpi=dpi)
            ax = fig.add_subplot(111, projection="3d")
            for i in range(num_parts):
                Rt = slerp(T0[i][:3, :3], T1[i][:3, :3], te)
                tt = (1 - te) * T0[i][:3, 3] + te * T1[i][:3, 3]
                p = (Rt @ part_pts[i].T).T + tt
                ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=part_cols[i], s=1, depthshade=False)
            ax.set_xlim(center[0] - half, center[0] + half)
            ax.set_ylim(center[1] - half, center[1] + half)
            ax.set_zlim(center[2] - half, center[2] + half)
            ax.set_axis_off()
            ax.view_init(elev=20, azim=45 + 60 * te)
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
            plt.close(fig)

    # Hold the final assembled state for `pause_seconds`
    pause_frames = int(pause_seconds * fps)
    if frames and pause_frames > 0:
        last_frame = frames[-1]
        for _ in range(pause_frames):
            frames.append(last_frame)

    try:
        imageio.mimsave(str(out_path), frames, fps=fps, plugin="FFMPEG")
    except Exception:
        imageio.mimsave(str(out_path), frames, fps=fps)
    print(f"saved {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jomon_data", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", default="./animations")
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--pause-seconds", type=float, default=2.0,
                    help="Duration in seconds to hold the final assembled frame")
    ap.add_argument("--source-up", choices=["y", "z", "x"], default="y")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resolution", type=int, default=800)
    ap.add_argument("--point-radius", type=int, default=1)
    ap.add_argument("--max-points", type=int, default=30000)
    # collage options
    ap.add_argument("--collage_only", action="store_true",
                    help="skip rendering; build collage from existing MP4s in --out")
    ap.add_argument("--no_collage", action="store_true")
    ap.add_argument("--collage_names", nargs="*", default=None,
                    help="object stems to place in the collage (in order)")
    ap.add_argument("--cell", type=int, default=320)
    ap.add_argument("--collage_frames", type=int, default=40)
    ap.add_argument("--collage_fps", type=int, default=12)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rendered = []

    if not args.collage_only:
        assert args.jomon_data and args.results, "--jomon_data and --results required unless --collage_only"
        Q = UP_FIX[args.source_up]
        dev = args.device
        if dev == "auto":
            dev = "cuda" if (HAVE_TORCH and torch.cuda.is_available()) else "cpu"
        renderer = None
        if dev == "cuda":
            renderer = GPURenderer(args.resolution, args.resolution,
                                   torch.device("cuda"), radius=args.point_radius)
            print("[INFO] GPU renderer (torch CUDA)")
        else:
            print("[INFO] CPU matplotlib fallback")

        jomon, results = Path(args.jomon_data), Path(args.results)
        for jf in sorted(results.glob("*.json")):
            result = json.loads(jf.read_text())
            name = result["name"]
            if args.objects and name not in args.objects:
                continue
            obj_dir = jomon / name
            if not (obj_dir / "adjacency.json").exists():
                print(f"skip {name}: no adjacency.json")
                continue
            rendered.append(render_object(obj_dir, result, out / f"{name}.mp4",
                                          Q, renderer, n_frames=args.frames,
                                          fps=args.fps, max_points=args.max_points,
                                          pause_seconds=args.pause_seconds))

    # collage
    if args.no_collage:
        return
    candidates = rendered if rendered else sorted(out.glob("*.mp4"))
    if args.collage_names:
        picked = [p for n in args.collage_names for p in candidates if p.stem == n]
        if picked:
            candidates = picked
    if not candidates:
        print("[WARN] no MP4s available for collage")
        return
    build_collage_gif(candidates[:9], out / "collage_3x3.gif",
                      cell=args.cell, n_frames=args.collage_frames, fps=args.collage_fps)


if __name__ == "__main__":
    main()