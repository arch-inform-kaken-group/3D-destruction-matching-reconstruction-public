# """
# JomonDataset: reads destruct.py output (fragments/*.ply + adjacency.json)
# and emits the exact dict format produced by BreakingBadWeighted.transform(),
# so DenoiserFlowMatching + FracSeg work with zero changes.

# Point-cloud count: destruct.py may store many points per fragment; we
# subsample to num_points_to_sample (5000) total, weighted by fragment point
# count (area proxy), matching GARF's weighted sampling (Appendix B.1).
# """
# import json
# from pathlib import Path, PurePath
# from typing import List

# import numpy as np
# import trimesh
# from scipy.spatial import cKDTree
# from torch.utils.data import Dataset, default_collate

# from ..transform import recenter_pc, rotate_pc, shuffle_pc


# class JomonDataset(Dataset):
#     def __init__(
#         self,
#         split: str = "train",                 # "train" | "test"
#         data_root: str = "data",
#         held_out: int = 10,
#         min_parts: int = 2,
#         max_parts: int = 20,
#         num_points_to_sample: int = 5000,
#         min_points_per_part: int = 20,
#         random_anchor: bool = False,
#         **kwargs,
#     ):
#         super().__init__()
#         assert split in ("train", "test")
#         self.split = split
#         self.data_root = Path(data_root)
#         self.held_out = held_out
#         self.min_parts = min_parts
#         self.max_parts = max_parts
#         self.num_points_to_sample = num_points_to_sample
#         self.min_points_per_part = min_points_per_part
#         self.random_anchor = random_anchor

#         assert self.max_parts * self.min_points_per_part <= self.num_points_to_sample

#         # Discover objects that have adjacency.json
#         all_objs = sorted(
#             d.name for d in self.data_root.iterdir()
#             if d.is_dir() and (d / "adjacency.json").exists()
#         )
#         # Filter by fragment count
#         valid = []
#         for name in all_objs:
#             try:
#                 adj = json.loads((self.data_root / name / "adjacency.json").read_text())
#                 n = adj["n_fragments"]
#                 if self.min_parts <= n <= self.max_parts:
#                     valid.append(name)
#             except Exception:
#                 continue

#         # Deterministic hold-out: last `held_out` (sorted) objects are test
#         if split == "train":
#             self.data_list = valid[: len(valid) - held_out] if held_out > 0 else valid
#         else:
#             self.data_list = valid[len(valid) - held_out:] if held_out > 0 else []

#     def __len__(self):
#         return len(self.data_list)

#     # ------------------------------------------------------------------ utils
#     @staticmethod
#     def _estimate_normals(points: np.ndarray, k: int = 20) -> np.ndarray:
#         """PCA normals from k-NN, oriented outward from the fragment centroid."""
#         n = len(points)
#         if n == 0:
#             return np.zeros((0, 3))
#         k = min(k, n)
#         _, idx = cKDTree(points).query(points, k=k)
#         nbrs = points[idx]                                   # (n, k, 3)
#         centered = nbrs - nbrs.mean(axis=1, keepdims=True)
#         cov = np.einsum("nki,nkj->nij", centered, centered)
#         _, eigvecs = np.linalg.eigh(cov)
#         normals = eigvecs[:, :, 0]                           # smallest eigenvalue
#         # orient outward
#         centroid = points.mean(axis=0)
#         outward = points - centroid
#         flip = np.sum(normals * outward, axis=1) < 0
#         normals[flip] *= -1
#         normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
#         return normals

#     def _compute_ppp(self, counts: np.ndarray, num_parts: int) -> List[int]:
#         """Points-per-part proportional to fragment size, exact total."""
#         base = self.min_points_per_part
#         remaining = self.num_points_to_sample - base * num_parts
#         total = counts.sum()
#         ppp = [base + int(remaining * c / total) for c in counts]
#         # fix rounding so sum == num_points_to_sample exactly
#         diff = self.num_points_to_sample - sum(ppp)
#         order = np.argsort(ppp)[::-1]
#         i = 0
#         while diff != 0:
#             step = 1 if diff > 0 else -1
#             ppp[order[i % num_parts]] += step
#             diff -= step
#             i += 1
#         return ppp

#     @staticmethod
#     def _subsample(n: int, target: int, rng: np.random.Generator) -> np.ndarray:
#         if n == 0:
#             return np.zeros(0, dtype=int)
#         if target <= n:
#             return rng.choice(n, target, replace=False)
#         return rng.choice(n, target, replace=True)

#     def _pad(self, arr: np.ndarray) -> np.ndarray:
#         d = np.array(arr)
#         pad_shape = (self.max_parts,) + tuple(d.shape[1:])
#         out = np.zeros(pad_shape, dtype=np.float32)
#         out[: d.shape[0]] = d
#         return out

#     # ------------------------------------------------------------- transform
#     def _transform(self, pc_gt_list, n_gt_list, num_parts, index, name,
#                    mesh_scale, edges, nodes, rng):
#         """Mirrors BreakingBadWeighted.transform()."""
#         points_per_part = np.array([len(pc) for pc in pc_gt_list])
#         offset = np.concatenate([[0], np.cumsum(points_per_part)])
#         pointclouds_gt = np.concatenate(pc_gt_list)                 # (N, 3)
#         pointclouds_normals_gt = np.concatenate(n_gt_list)          # (N, 3)
#         N_total = pointclouds_gt.shape[0]

#         pointclouds, pointclouds_normals, quaternions, translations = [], [], [], []
#         scale = []
#         for part_idx in range(num_parts):
#             s, e = offset[part_idx], offset[part_idx + 1]
#             pointcloud, translation = recenter_pc(pointclouds_gt[s:e])
#             pointcloud, pointcloud_normals, quaternion = rotate_pc(
#                 pointcloud, pointclouds_normals_gt[s:e]
#             )
#             pointcloud, pointcloud_normals, _ = shuffle_pc(pointcloud, pointcloud_normals)
#             current_scale = np.max(np.abs(pointcloud))
#             if current_scale < 1e-8:
#                 current_scale = 1.0
#             scale.append(current_scale)
#             pointcloud /= current_scale
#             pointclouds.append(pointcloud)
#             pointclouds_normals.append(pointcloud_normals)
#             quaternions.append(quaternion)
#             translations.append(translation)

#         pointclouds = np.concatenate(pointclouds).astype(np.float32)          # (N, 3)
#         pointclouds_normals = np.concatenate(pointclouds_normals).astype(np.float32)
#         quaternions = np.stack(quaternions).astype(np.float32)                # (P, 4)
#         translations = np.stack(translations).astype(np.float32)              # (P, 3)
#         scale = np.array(scale).astype(np.float32)

#         # Pad to max_parts
#         points_per_part = self._pad(points_per_part).astype(np.int64)
#         quaternions = self._pad(quaternions)
#         translations = self._pad(translations)
#         scale = self._pad(scale)

#         # Reference part (largest fragment, or random anchor)
#         ref_part = np.zeros(self.max_parts, dtype=np.float32)
#         ref_idx = int(np.argmax(points_per_part[:num_parts]))
#         if self.random_anchor:
#             can = points_per_part[:num_parts] > self.num_points_to_sample * 0.05
#             if can.any():
#                 ref_idx = int(rng.choice(np.where(can)[0], 1)[0])
#         ref_part[ref_idx] = 1
#         ref_part = ref_part.astype(bool)

#         # Connectivity graph from adjacency edges
#         graph = np.zeros((self.max_parts, self.max_parts), dtype=bool)
#         id2idx = {node["id"]: i for i, node in enumerate(nodes)}
#         for edge in edges:
#             a, b = edge.get("a"), edge.get("b")
#             if a in id2idx and b in id2idx:
#                 i, j = id2idx[a], id2idx[b]
#                 graph[i, j] = graph[j, i] = True

#         # fracture_surface_gt is unused by the FM loss; provide zeros of correct length
#         fracture_surface_gt = np.zeros(N_total, dtype=np.int8)

#         return {
#             "index": index,
#             "name": name,
#             "num_parts": num_parts,
#             "pointclouds": pointclouds,
#             "pointclouds_gt": pointclouds_gt.astype(np.float32),
#             "pointclouds_normals": pointclouds_normals.astype(np.float32),
#             "pointclouds_normals_gt": pointclouds_normals_gt.astype(np.float32),
#             "fracture_surface_gt": fracture_surface_gt,
#             "quaternions": quaternions,
#             "translations": translations,
#             "points_per_part": points_per_part.astype(np.int64),
#             "graph": graph,
#             "scale": scale[:, np.newaxis],
#             "ref_part": ref_part,
#             "removal": 0,
#             "redundancy": 0,
#             "removal_pieces": "",
#             "redundant_pieces": "",
#             "pieces": ",".join(node["id"] for node in nodes),
#             "mesh_scale": mesh_scale,
#             # NOTE: no "meshes" key -> test_step skips glb assembly export;
#             # animation is handled by src/render_local.py instead.
#         }

#     # -------------------------------------------------------------- getitem
#     def __getitem__(self, index):
#         name = self.data_list[index]
#         obj_dir = self.data_root / name
#         adj = json.loads((obj_dir / "adjacency.json").read_text())
#         nodes = adj["nodes"]
#         edges = adj.get("edges", [])
#         num_parts = len(nodes)
#         rng = np.random.default_rng(seed=index)

#         # Load assembled point clouds
#         frag_points = []
#         for node in nodes:
#             ply_rel_path = PurePath(node["ply"].replace("\\", "/"))
#             ply_path = obj_dir / ply_rel_path
#             pc = trimesh.load(ply_path)
#             frag_points.append(np.asarray(pc.vertices, dtype=np.float64))

#         # Global scale to unit extent (like BreakingBadBase)
#         all_pts = np.concatenate(frag_points)
#         extents = all_pts.max(0) - all_pts.min(0)
#         mesh_scale = float(max(1.0, extents.max()))
#         frag_points = [p / mesh_scale for p in frag_points]

#         # Estimate normals per fragment
#         frag_normals = [self._estimate_normals(p) for p in frag_points]

#         # Subsample to exact total, weighted by fragment point count
#         counts = np.array([len(p) for p in frag_points], dtype=np.float64)
#         ppp = self._compute_ppp(counts, num_parts)
#         pc_gt_list, n_gt_list = [], []
#         for i in range(num_parts):
#             sel = self._subsample(len(frag_points[i]), ppp[i], rng)
#             pc_gt_list.append(frag_points[i][sel])
#             n_gt_list.append(frag_normals[i][sel])

#         return self._transform(
#             pc_gt_list, n_gt_list, num_parts, index, name,
#             mesh_scale, edges, nodes, rng
#         )

#     # -------------------------------------------------------------- collate
#     @staticmethod
#     def collate_fn(batch):
#         collated = {}
#         for key in batch[0].keys():
#             if key == "meshes":
#                 collated[key] = [item[key] for item in batch]
#             else:
#                 collated[key] = default_collate([item[key] for item in batch])
#         return collated








"""
JomonDataset v4 — on-the-fly fragmentation, GARF-paper sampling, memory-hardened.

Sampling (GARF Appendix B.1 / breaking_bad/weighted.py):
  * M = num_points_to_sample (5000) points per object
  * area-weighted point allocation across fragments
  * Poisson-disk sampling (trimesh.sample_surface_even) with uniform
    area-weighted padding when the even-sampler falls short
  * exact face normals for shell points (no PCA approximation)
  * optional fracture-fill points (PCA normals) via include_fill / fill_ratio

Memory hardening (WSL2 / long-run fixes):
  * bounded per-worker mesh LRU cache (_MESH_CACHE_MAX) + OOM retry on load
  * gc.collect() + libc malloc_trim(0) every _TRIM_EVERY generated items
  * datamodule side: spawn context + persistent workers (no CoW RSS bloat)
"""
import ctypes
import gc
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import logging
import trimesh

from scipy.spatial import cKDTree
from torch.utils.data import Dataset, default_collate

from ..transform import recenter_pc, rotate_pc, shuffle_pc

# ---- import destruct from src/ ----
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import destruct

destruct.QUIET = True   # silence logs inside DataLoader workers
trimesh.util.log.setLevel(logging.ERROR)

# ---- memory helpers ----
_LIBC = None


def _return_memory_to_os():
    """Return freed numpy/glibc arenas to the OS (kills RSS ratcheting)."""
    global _LIBC
    if _LIBC is None:
        try:
            _LIBC = ctypes.CDLL("libc.so.6")
        except OSError:
            _LIBC = False
    gc.collect()
    if _LIBC:
        _LIBC.malloc_trim(0)


_MESH_CACHE = {}        # per-worker process cache: path -> processed trimesh
_MESH_CACHE_MAX = 2     # bounded; light (decimated) meshes keep this cheap
_TRIM_EVERY = 25        # return memory to OS every N generated items per worker


class JomonDataset(Dataset):
    def __init__(
        self,
        split: str = "train",                  # "train" | "test"
        pottery_dir: str = "pottery",          # dir of intact meshes
        held_out: int = 10,
        min_parts: int = 2,
        max_parts: int = 20,
        num_points_to_sample: int = 5000,
        min_points_per_part: int = 20,
        random_anchor: bool = False,
        # on-the-fly destruct settings
        num_fragments: int = 8,
        destruct_num_samples: int = 20_000,    # internal labeling resolution
        destruct_stress_model: str = "fem_lite",  # "curvature" = ~3x faster
        destruct_grid_res: int = 40,
        destruct_jacobi_iters: int = 60,
        skip_fill: bool = False,
        min_fragments: int = 3,                # hard lower bound on fragments
        # GARF sampling settings
        mesh_sample_strategy: str = "poisson",   # "fps" | "poisson" | "wpd"
        include_fill: bool = True,
        fill_ratio: float = 0.15,
        fps_oversample: int = 12,   # candidate-pool multiplier for FPS
        test_base_seed: int = 12345,
        wpd_detail_ratio: float = 0.30,
        wpd_curv_thresh: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        assert split in ("train", "test")
        self.split = split
        self.pottery_dir = Path(pottery_dir)
        self.held_out = held_out
        self.min_parts = min_parts
        self.max_parts = max_parts
        self.num_points_to_sample = num_points_to_sample
        self.min_points_per_part = min_points_per_part
        self.random_anchor = random_anchor
        self.min_fragments = min_fragments
        self.mesh_sample_strategy = mesh_sample_strategy
        self.wpd_detail_ratio = wpd_detail_ratio
        self.wpd_curv_thresh = wpd_curv_thresh
        self.include_fill = include_fill
        self.fill_ratio = fill_ratio
        self.fps_oversample = fps_oversample
        self.test_base_seed = test_base_seed
        self._n_items = 0

        assert self.max_parts * self.min_points_per_part <= self.num_points_to_sample

        self.destruct_args = destruct.fast_destruct_args(
            num_fragments=num_fragments,
            num_samples=destruct_num_samples,
            stress_model=destruct_stress_model,
            grid_res=destruct_grid_res,
            jacobi_iters=destruct_jacobi_iters,
            skip_fill=skip_fill,
            min_fragments=min_fragments,
        )

        # discover meshes; deterministic hold-out: last `held_out` are test
        exts = {".glb", ".gltf", ".obj", ".ply", ".stl"}
        all_meshes = sorted(
            p for p in self.pottery_dir.rglob("*") if p.suffix.lower() in exts
        )
        if split == "train":
            self.mesh_list = all_meshes[: len(all_meshes) - held_out] if held_out > 0 else all_meshes
        else:
            self.mesh_list = all_meshes[len(all_meshes) - held_out:] if held_out > 0 else []

    def __len__(self):
        return len(self.mesh_list)

    @property
    def data_list(self):
        """Unified interface with the PLY variant: object names (stems)."""
        return [p.stem for p in self.mesh_list]

    # ------------------------------------------------------------- seeding
    def _seed_for(self, index):
        if self.split == "train":
            return int(np.random.default_rng().integers(1, 2**31 - 1))  # fresh fracture
        return self.test_base_seed + index                              # reproducible

    # ------------------------------------------------- mesh load (cached, OOM-safe)
    @staticmethod
    def _load_mesh(path):
        key = str(path)
        hit = _MESH_CACHE.get(key)
        if hit is not None:
            return hit.copy()
        mesh = None
        for attempt in range(2):
            try:
                loaded = trimesh.load(key, force="scene")
                if isinstance(loaded, trimesh.Scene):
                    geoms = [g for g in loaded.geometry.values()
                             if isinstance(g, trimesh.Trimesh)]
                    if not geoms:
                        raise RuntimeError("no usable mesh geometry")
                    mesh = geoms[0].copy() if len(geoms) == 1 \
                        else trimesh.util.concatenate(geoms)
                else:
                    mesh = loaded
                mesh.process(validate=True)
                mesh.merge_vertices()
                break
            except (OSError, MemoryError):
                gc.collect()
                if attempt == 1:
                    raise
        if len(mesh.faces) == 0:
            raise RuntimeError("mesh has no faces")
        while len(_MESH_CACHE) >= _MESH_CACHE_MAX:
            _MESH_CACHE.pop(next(iter(_MESH_CACHE)))
        _MESH_CACHE[key] = mesh
        return mesh.copy()

    # ------------------------------------------------- GARF B.1 point budgets
    def _ppp_area(self, areas: np.ndarray, total: int) -> List[int]:
        """Area-weighted allocation, exact sum (mirrors breaking_bad/weighted.py)."""
        n = len(areas)
        base = self.min_points_per_part
        remaining = total - base * n
        tot_area = float(areas.sum())
        if tot_area < 1e-12:
            ppp = [total // n] * n
        else:
            ppp = [base + int(remaining * a / tot_area) for a in areas]
        ppp[int(np.argmax(ppp))] += total - sum(ppp)
        ppp[int(np.argmin(ppp))] += total - sum(ppp)
        return [max(0, int(v)) for v in ppp]

    @staticmethod
    def _allocate(weights: np.ndarray, total: int) -> np.ndarray:
        """Pure proportional allocation (no base), exact sum — used for fill."""
        weights = np.asarray(weights, float)
        n = len(weights)
        if total <= 0 or n == 0:
            return np.zeros(n, dtype=int)
        if weights.sum() <= 0:
            counts = np.full(n, total // n, dtype=int)
            counts[: total - counts.sum()] += 1
            return counts
        raw = weights / weights.sum() * total
        counts = np.floor(raw).astype(int)
        remainder = total - counts.sum()
        order = np.argsort(-(raw - counts))
        for k in range(int(remainder)):
            counts[order[k % n]] += 1
        return counts

    # ------------------------------------------------- GARF B.1 surface sampling
    def _sample_shell(self, mesh: trimesh.Trimesh, count: int, seed: int):
        """Poisson-disk (sample_surface_even) + uniform padding; exact face normals."""
        if count <= 0 or len(mesh.faces) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        if self.mesh_sample_strategy == "poisson":
            pcd, idx = trimesh.sample.sample_surface_even(
                mesh, count=int(count), seed=seed)
            if len(pcd) < count:                      # pad shortfall uniformly
                pad_pcd, pad_idx = trimesh.sample.sample_surface(
                    mesh, count=int(count) - len(pcd), seed=seed + 1)
                pcd = np.concatenate([pcd, pad_pcd], axis=0)
                idx = np.concatenate([idx, pad_idx], axis=0)
        else:
            pcd, idx = trimesh.sample.sample_surface(
                mesh, count=int(count), seed=seed)
        return pcd.astype(np.float64), mesh.face_normals[idx].astype(np.float64)

    @staticmethod
    def _quick_normals(points: np.ndarray, k: int = 8) -> np.ndarray:
        """PCA normals for small point sets (fracture fill only)."""
        n = len(points)
        if n == 0:
            return np.zeros((0, 3))
        k = min(k, n)
        _, idx = cKDTree(points).query(points, k=k)
        nbrs = points[idx]
        centered = nbrs - nbrs.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centered, centered)
        _, eigvecs = np.linalg.eigh(cov)
        normals = eigvecs[:, :, 0]
        outward = points - points.mean(axis=0)
        flip = np.sum(normals * outward, axis=1) < 0
        normals[flip] *= -1
        return normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    def _sample_fragment(self, shell, fill_list, shell_count, fill_count, rng):
        seed = int(rng.integers(0, 2**31 - 1))
        if shell_count > 0 and len(shell.faces) > 0:
            if self.mesh_sample_strategy == "wpd":
                pts, nms = self._sample_shell_wpd(shell, shell_count, seed)
            elif self.mesh_sample_strategy == "poisson":
                pts, nms = self._even_sample(shell, shell_count, seed)
            else:  # "fps"
                pts, nms = self._sample_shell_fps(shell, shell_count, seed, rng)
        elif shell_count > 0 and len(shell.vertices) > 0:
            sel = rng.choice(len(shell.vertices), int(shell_count),
                             replace=(shell_count > len(shell.vertices)))
            pts = shell.vertices[sel].astype(np.float64)
            nms = shell.vertex_normals[sel].astype(np.float64)
        else:
            pts, nms = np.zeros((0, 3)), np.zeros((0, 3))
        if fill_count > 0 and fill_list:                        # fracture fill
            all_fp = np.vstack([fp for fp, _ in fill_list])
            if len(all_fp) > 0:
                sel = rng.choice(len(all_fp), int(fill_count),
                                 replace=(fill_count > len(all_fp)))
                fp_sel = all_fp[sel]
                fn_sel = self._quick_normals(fp_sel)
                pts = np.vstack([pts, fp_sel]) if len(pts) else fp_sel
                nms = np.vstack([nms, fn_sel]) if len(nms) else fn_sel
        return pts, nms

    # ------------------------------------------------------- FPS helpers
    @staticmethod
    def _fps(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """Greedy farthest-point sampling (max-min-distance selection)."""
        N = len(points)
        n = int(n)
        if n <= 0 or N == 0:
            return np.zeros(0, dtype=np.int64)
        if n >= N:
            return np.arange(N, dtype=np.int64)
        sel = np.empty(n, dtype=np.int64)
        sel[0] = int(rng.integers(N))
        d2 = ((points - points[sel[0]]) ** 2).sum(1)
        for i in range(1, n):
            j = int(np.argmax(d2))
            sel[i] = j
            np.minimum(d2, ((points - points[j]) ** 2).sum(1), out=d2)
        return sel

    def _sample_shell_fps(self, shell, count, seed, rng):
        """FPS over an area-weighted candidate pool: density follows surface
        area (pool), spacing is even (FPS), normals stay exact face normals."""
        count = int(count)
        pool = max(count, min(len(shell.faces) * 3, count * self.fps_oversample))
        sp, fidx = trimesh.sample.sample_surface(shell, pool, seed=seed)
        keep = self._fps(sp, count, rng)
        return (sp[keep].astype(np.float64),
                shell.face_normals[fidx[keep]].astype(np.float64))

    # poisson
    def _even_sample(self, mesh, count, seed):
        """Poisson-disk (blue noise) + uniform padding to exact count."""
        if count <= 0 or len(mesh.faces) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        p, fi = trimesh.sample.sample_surface_even(mesh, count=int(count), seed=seed)
        if len(p) < count:
            p2, fi2 = trimesh.sample.sample_surface(
                mesh, count=int(count) - len(p), seed=seed + 1)
            p = np.concatenate([p, p2], axis=0)
            fi = np.concatenate([fi, fi2], axis=0)
        elif len(p) > count:
            p, fi = p[:count], fi[:count]
        return p.astype(np.float64), mesh.face_normals[fi].astype(np.float64)

    def _feature_faces(self, shell):
        """Open-boundary (fracture rim) faces + high-curvature faces."""
        feat = np.zeros(len(shell.faces), dtype=bool)
        try:
            fue = shell.faces_unique_edges
            cnt = np.bincount(fue.ravel(), minlength=len(shell.edges_unique))
            bnd = np.nonzero(cnt == 1)[0]
            if len(bnd):
                feat |= np.isin(fue, bnd).any(axis=1)
        except Exception:
            pass
        try:
            c = trimesh.curvature.discrete_mean_curvature_measure(
                shell, shell.triangles_center, radius=max(shell.scale, 1e-6) / 30.0)
            c = np.abs(np.nan_to_num(c))
            if c.max() > 1e-12:
                feat |= (c / c.max()) > self.wpd_curv_thresh
        except Exception:
            pass
        return feat

    # weighted poisson
    def _sample_shell_wpd(self, shell, count, seed):
        """Weighted Poisson-disk: base blue noise + denser blue noise on features."""
        if count <= 0 or len(shell.faces) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        feat = self._feature_faces(shell)
        n_feat = int(count * self.wpd_detail_ratio) if feat.sum() >= 16 else 0
        pts, nms = self._even_sample(shell, count - n_feat, seed)
        if n_feat > 0:
            sub = shell.submesh([np.flatnonzero(feat)], append=True)
            if sub is not None and len(sub.faces) > 0:
                p2, n2 = self._even_sample(sub, n_feat, seed + 7)
                pts = np.concatenate([pts, p2], axis=0)
                nms = np.concatenate([nms, n2], axis=0)
        return pts, nms

    def _enforce_total(self, pc_gt_list, n_gt_list, rng):
        """Force exactly num_points_to_sample points (collate can never see ragged)."""
        target = self.num_points_to_sample
        total = sum(len(p) for p in pc_gt_list)
        if total == target or not pc_gt_list:
            return pc_gt_list, n_gt_list
        i = int(np.argmax([len(p) for p in pc_gt_list]))
        diff = target - total
        if diff > 0:
            idx = rng.choice(len(pc_gt_list[i]), diff, replace=True)
            pc_gt_list[i] = np.vstack([pc_gt_list[i], pc_gt_list[i][idx]])
            n_gt_list[i] = np.vstack([n_gt_list[i], n_gt_list[i][idx]])
        else:
            keep = rng.choice(len(pc_gt_list[i]), len(pc_gt_list[i]) + diff, replace=False)
            pc_gt_list[i] = pc_gt_list[i][keep]
            n_gt_list[i] = n_gt_list[i][keep]
        return pc_gt_list, n_gt_list

    # ------------------------------------------------------------- transform
    def _pad(self, arr):
        d = np.array(arr)
        pad_shape = (self.max_parts,) + tuple(d.shape[1:])
        out = np.zeros(pad_shape, dtype=np.float32)
        out[: d.shape[0]] = d
        return out

    def _transform(self, pc_gt_list, n_gt_list, index, name, mesh_scale, graph, rng):
        num_parts = len(pc_gt_list)
        points_per_part = np.array([len(pc) for pc in pc_gt_list])
        offset = np.concatenate([[0], np.cumsum(points_per_part)])
        pointclouds_gt = np.concatenate(pc_gt_list)
        pointclouds_normals_gt = np.concatenate(n_gt_list)
        N_total = pointclouds_gt.shape[0]

        pointclouds, pointclouds_normals, quaternions, translations = [], [], [], []
        scale = []
        for part_idx in range(num_parts):
            s, e = offset[part_idx], offset[part_idx + 1]
            pointcloud, translation = recenter_pc(pointclouds_gt[s:e])
            pointcloud, pointcloud_normals, quaternion = rotate_pc(
                pointcloud, pointclouds_normals_gt[s:e])
            pointcloud, pointcloud_normals, _ = shuffle_pc(pointcloud, pointcloud_normals)
            current_scale = np.max(np.abs(pointcloud))
            if current_scale < 1e-8:
                current_scale = 1.0
            scale.append(current_scale)
            pointcloud /= current_scale
            pointclouds.append(pointcloud)
            pointclouds_normals.append(pointcloud_normals)
            quaternions.append(quaternion)
            translations.append(translation)

        pointclouds = np.concatenate(pointclouds).astype(np.float32)
        pointclouds_normals = np.concatenate(pointclouds_normals).astype(np.float32)
        quaternions = np.stack(quaternions).astype(np.float32)
        translations = np.stack(translations).astype(np.float32)
        scale = np.array(scale).astype(np.float32)

        points_per_part = self._pad(points_per_part).astype(np.int64)
        quaternions = self._pad(quaternions)
        translations = self._pad(translations)
        scale = self._pad(scale)

        ref_part = np.zeros(self.max_parts, dtype=np.float32)
        ref_idx = int(np.argmax(points_per_part[:num_parts]))
        if self.random_anchor:
            can = points_per_part[:num_parts] > self.num_points_to_sample * 0.05
            if can.any():
                ref_idx = int(rng.choice(np.where(can)[0], 1)[0])
        ref_part[ref_idx] = 1
        ref_part = ref_part.astype(bool)

        fracture_surface_gt = np.zeros(N_total, dtype=np.int8)

        return {
            "index": index,
            "name": name,
            "num_parts": num_parts,
            "pointclouds": pointclouds,
            "pointclouds_gt": pointclouds_gt.astype(np.float32),
            "pointclouds_normals": pointclouds_normals.astype(np.float32),
            "pointclouds_normals_gt": pointclouds_normals_gt.astype(np.float32),
            "fracture_surface_gt": fracture_surface_gt,
            "quaternions": quaternions,
            "translations": translations,
            "points_per_part": points_per_part.astype(np.int64),
            "graph": graph,
            "scale": scale[:, np.newaxis],
            "ref_part": ref_part,
            "removal": 0, "redundancy": 0,
            "removal_pieces": "", "redundant_pieces": "",
            "pieces": ",".join(f"frag_{i:03d}" for i in range(num_parts)),
            "mesh_scale": mesh_scale,
        }

    # ------------------------------------------------------------- getitem
    def __getitem__(self, index):
        mesh_path = self.mesh_list[index]
        name = mesh_path.stem
        seed = self._seed_for(index)
        rng = np.random.default_rng(seed)

        mesh = self._load_mesh(mesh_path)

        # bounded retry to guarantee >= min_fragments
        result = None
        for attempt in range(5):
            result = destruct.generate_fragments_in_memory(
                mesh, self.destruct_args, seed + attempt * 7919)
            if result is not None and result["num_fragments"] >= self.min_fragments:
                break
        if result is None or result["num_fragments"] < self.min_fragments:
            raise RuntimeError(
                f"{name}: cannot produce >= {self.min_fragments} fragments "
                f"after 5 attempts (mesh may be degenerate)")

        shells = result["shells"]
        fill = result["fill"]
        num_parts = result["num_fragments"]

        # global scale to unit extent
        all_verts = np.concatenate([s.vertices for s in shells])
        extents = all_verts.max(0) - all_verts.min(0)
        mesh_scale = float(max(1.0, extents.max()))

        # GARF B.1: area-weighted shell budget + proportional fill budget
        areas = np.array(result["areas"], dtype=np.float64)
        fill_ratio = self.fill_ratio if self.include_fill else 0.0
        fill_budget = int(self.num_points_to_sample * fill_ratio)
        shell_budget = self.num_points_to_sample - fill_budget
        shell_counts = self._ppp_area(areas, shell_budget)
        fill_avail = np.array(
            [sum(len(fp) for fp, _ in fill.get(i, [])) for i in range(num_parts)],
            dtype=np.float64)
        fill_counts = (self._allocate(fill_avail, fill_budget)
                       if fill_budget > 0 else np.zeros(num_parts, dtype=int))

        pc_gt_list, n_gt_list = [], []
        for i in range(num_parts):
            pts, nms = self._sample_fragment(
                shells[i], fill.get(i, []), shell_counts[i], int(fill_counts[i]), rng)
            pc_gt_list.append(pts / mesh_scale)
            n_gt_list.append(nms)

        # connectivity graph from fragment-index edge pairs
        graph = np.zeros((self.max_parts, self.max_parts), dtype=bool)
        for i, j in result["edges"]:
            graph[i, j] = graph[j, i] = True

        pc_gt_list, n_gt_list = self._enforce_total(pc_gt_list, n_gt_list, rng)
        out = self._transform(pc_gt_list, n_gt_list, index, name, mesh_scale, graph, rng)

        # memory cadence: return arenas to OS periodically
        self._n_items += 1
        if self._n_items % _TRIM_EVERY == 0:
            _return_memory_to_os()
        return out

    # ------------------------------------------------------------- collate
    @staticmethod
    def collate_fn(batch):
        collated = {}
        for key in batch[0].keys():
            if key == "meshes":
                collated[key] = [item[key] for item in batch]
            else:
                collated[key] = default_collate([item[key] for item in batch])
        return collated