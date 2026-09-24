"""
Local-PC animation of GARF reassembly for Jomon pottery.
Reads destruct.py output + GARF inference json_results, renders MP4.

Deps (local only): numpy scipy matplotlib imageio[ffmpeg] trimesh
Usage:
  python src/render_local.py \
      --jomon_data ./jomon_data \
      --results ./GARF/logs/GARF-Jomon/<run>/json_results \
      --out ./animations
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio


# ---------------- SE(3) helpers (GARF uses scalar-first quaternions [w,x,y,z])
def quat_wxyz_to_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def se3_from_vec(v):
    """[tx,ty,tz, qw,qx,qy,qz] -> 4x4 matrix."""
    v = np.asarray(v, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_matrix(v[3:])
    T[:3, 3] = v[:3]
    return T


def slerp(R0, R1, t):
    return R.slerp(t, R.from_matrix(R0), R.from_matrix(R1)).as_matrix()


def load_fragment_pcd(ply_path):
    pc = trimesh.load(str(ply_path))
    pts = np.asarray(pc.vertices, dtype=np.float64)
    cols = None
    try:
        cols = np.asarray(pc.colors, dtype=np.float64)[:, :3] / 255.0
    except Exception:
        pass
    return pts, cols


def render_object(obj_dir, result, out_path, n_frames=60, fps=30, dpi=100):
    name = result["name"]
    mesh_scale = float(result["mesh_scale"])
    gt = np.asarray(result["gt_trans_rots"], dtype=np.float64)        # (P, 7)
    pred_steps = result["pred_trans_rots"]                            # list of steps
    pred = np.asarray(pred_steps[-1], dtype=np.float64)               # final step (P, 7)
    num_parts = int(result["num_parts"])

    adj = json.loads((obj_dir / "adjacency.json").read_text())
    nodes = adj["nodes"]

    # Assemble per-part geometry in scaled world frame + per-part transforms
    part_pts, part_cols, T_starts, T_ends = [], [], [], []
    for i, node in enumerate(nodes[:num_parts]):
        pts, cols = load_fragment_pcd(obj_dir / node["ply"])
        pts = pts / mesh_scale                                        # scaled world frame

        gt_se3 = se3_from_vec(gt[i])
        pred_se3 = se3_from_vec(pred[i])
        inv_gt = np.linalg.inv(gt_se3)
        T_start = inv_gt                       # scattered (model input)
        T_end = pred_se3 @ inv_gt              # predicted assembled
        part_pts.append(pts)
        part_cols.append(cols if cols is not None else np.full((len(pts), 3), 0.6))
        T_starts.append(T_start)
        T_ends.append(T_end)

    # Fixed axis limits from assembled bounds
    all_pts = np.concatenate(part_pts)
    center = all_pts.mean(0)
    half = float(np.abs(all_pts - center).max()) * 1.15

    frames = []
    for f in range(n_frames):
        t = f / max(1, n_frames - 1)
        # ease in-out
        te = t * t * (3 - 2 * t)

        fig = plt.figure(figsize=(6, 6), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        for i in range(num_parts):
            R0, t0 = T_starts[i][:3, :3], T_starts[i][:3, 3]
            R1, t1 = T_ends[i][:3, :3], T_ends[i][:3, 3]
            Rt = slerp(R0, R1, te)
            tt = (1 - te) * t0 + te * t1
            pts_t = (Rt @ part_pts[i].T).T + tt
            ax.scatter(pts_t[:, 0], pts_t[:, 1], pts_t[:, 2],
                       c=part_cols[i], s=1, depthshade=False)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=45 + 60 * te)   # slow orbit

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(img)
        plt.close(fig)

    imageio.mimsave(str(out_path), frames, fps=fps)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jomon_data", required=True)
    ap.add_argument("--results", required=True, help="GARF json_results dir")
    ap.add_argument("--out", default="./animations")
    ap.add_argument("--objects", nargs="*", default=None, help="optional name filter")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    jomon = Path(args.jomon_data)
    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for jf in sorted(results.glob("*.json")):
        result = json.loads(jf.read_text())
        name = result["name"]
        if args.objects and name not in args.objects:
            continue
        obj_dir = jomon / name
        if not (obj_dir / "adjacency.json").exists():
            print(f"skip {name}: no adjacency.json")
            continue
        render_object(obj_dir, result, out / f"{name}.mp4",
                      n_frames=args.frames, fps=args.fps)


if __name__ == "__main__":
    main()