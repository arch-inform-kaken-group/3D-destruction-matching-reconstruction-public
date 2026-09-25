# from typing import Optional

# import lightning as L
# from torch.utils.data import DataLoader

# from .jomon_dataset import JomonDataset

# class JomonDataModule(L.LightningDataModule):
#     def __init__(
#         self,
#         data_root: str,
#         min_parts: int = 2,
#         max_parts: int = 20,
#         num_points_to_sample: int = 5000,
#         min_points_per_part: int = 20,
#         batch_size: int = 4,
#         num_workers: int = 4,
#         held_out: int = 10,
#         multi_ref: bool = False,          # accepted for config compat (unused)
#         random_anchor: bool = False,
#         **kwargs,
#     ):
#         super().__init__()
#         self.save_hyperparameters()
#         self.train_dataset: Optional[JomonDataset] = None
#         self.test_dataset: Optional[JomonDataset] = None

#     def _kwargs(self):
#         p = self.hparams
#         return dict(
#             data_root=p.data_root,
#             min_parts=p.min_parts,
#             max_parts=p.max_parts,
#             num_points_to_sample=p.num_points_to_sample,
#             min_points_per_part=p.min_points_per_part,
#             held_out=p.held_out,
#             random_anchor=p.random_anchor,
#         )

#     def setup(self, stage):
#         if stage == "fit":
#             self.train_dataset = JomonDataset(split="train", **self._kwargs())
#             self.test_dataset = JomonDataset(split="test", **self._kwargs())
#         elif stage in ("test", "predict"):
#             self.test_dataset = JomonDataset(split="test", **self._kwargs())

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.hparams.batch_size,
#             num_workers=self.hparams.num_workers,
#             shuffle=True,
#             collate_fn=JomonDataset.collate_fn,
#             persistent_workers=False,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.hparams.batch_size,
#             num_workers=self.hparams.num_workers,
#             collate_fn=JomonDataset.collate_fn,
#             persistent_workers=False,
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.hparams.batch_size,
#             num_workers=self.hparams.num_workers,
#             collate_fn=JomonDataset.collate_fn,
#             persistent_workers=False,
#         )

from typing import Optional
import lightning as L
import numpy as np
import trimesh
from pathlib import Path
from torch.utils.data import DataLoader

from .jomon_dataset import JomonDataset


class JomonDataModule(L.LightningDataModule):
    def __init__(
        self,
        pottery_dir: str,
        min_parts: int = 2,
        max_parts: int = 20,
        num_points_to_sample: int = 5000,
        min_points_per_part: int = 20,
        batch_size: int = 4,
        num_workers: int = 4,
        held_out: int = 10,
        multi_ref: bool = False,
        random_anchor: bool = False,
        # on-the-fly destruct settings
        num_fragments: int = 8,
        destruct_num_samples: int = 20_000,
        destruct_stress_model: str = "fem_lite",
        destruct_grid_res: int = 40,
        destruct_jacobi_iters: int = 60,
        skip_fill: bool = False,
        min_fragments: int = 3,
        # GARF sampling settings
        mesh_sample_strategy: str = "poisson",   # "poisson" | "fps" | "wpd"
        wpd_detail_ratio: float = 0.30,
        wpd_curv_thresh: float = 0.5,
        include_fill: bool = True,
        fill_ratio: float = 0.15,
        fps_oversample: int = 12,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset: Optional[JomonDataset] = None
        self.test_dataset: Optional[JomonDataset] = None

    def _kwargs(self):
        p = self.hparams
        return dict(
            pottery_dir=p.pottery_dir,
            min_parts=p.min_parts,
            max_parts=p.max_parts,
            num_points_to_sample=p.num_points_to_sample,
            min_points_per_part=p.min_points_per_part,
            held_out=p.held_out,
            random_anchor=p.random_anchor,
            num_fragments=p.num_fragments,
            destruct_num_samples=p.destruct_num_samples,
            destruct_stress_model=p.destruct_stress_model,
            destruct_grid_res=p.destruct_grid_res,
            destruct_jacobi_iters=p.destruct_jacobi_iters,
            skip_fill=p.skip_fill,
            min_fragments=p.min_fragments,
            mesh_sample_strategy=p.mesh_sample_strategy,
            wpd_detail_ratio=p.wpd_detail_ratio,
            wpd_curv_thresh=p.wpd_curv_thresh,
            include_fill=p.include_fill,
            fill_ratio=p.fill_ratio,
            fps_oversample=p.fps_oversample,
        )

    def _save_sample_visuals(self, dataset, out_dir, n_samples=3):
        """Saves N sample point clouds as colored PLYs for visual verification."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Golden angle palette for coloring fragments
        def palette(n):
            n = max(1, int(n))
            h = (np.arange(n) * 0.618033988749895) % 1.0
            i = (h * 6).astype(int) % 6
            f = (h * 6) - np.floor(h * 6)
            p, q, t = 0.25, 0.95 * (1 - f * 0.75), 0.95 * (1 - (1 - f) * 0.75)
            val = 0.95
            r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [val, q, p, p, t, val])
            g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, val, val, q, p, p])
            b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, val, val, q])
            return np.clip(np.stack([r, g, b], -1), 0, 1)

        strategy = getattr(self.hparams, "mesh_sample_strategy", "unknown")
        print(f"[INFO] Saving {n_samples} sample point clouds for strategy '{strategy}' to {out_dir}")
        
        n_samples = min(n_samples, len(dataset))
        for i in range(n_samples):
            try:
                item = dataset[i]
                pts = item["pointclouds_gt"]
                ppp = item["points_per_part"]
                name = item["name"]
                
                colors = []
                num_parts = item["num_parts"]
                cols = palette(num_parts)
                
                for p_idx in range(num_parts):
                    count = int(ppp[p_idx])
                    if count > 0:
                        colors.append(np.tile(cols[p_idx], (count, 1)))
                
                if colors:
                    colors = np.vstack(colors)
                    rgba = np.column_stack([
                        (colors * 255).astype(np.uint8), 
                        np.full(len(pts), 255, dtype=np.uint8)
                    ])
                else:
                    rgba = np.full((len(pts), 4), 200, dtype=np.uint8)
                    
                pc = trimesh.PointCloud(pts, colors=rgba)
                out_path = out_dir / f"{name}_{strategy}.ply"
                pc.export(str(out_path))
            except Exception as e:
                print(f"[WARN] Failed to save sample {i}: {e}")

    def setup(self, stage):
        strategy = getattr(self.hparams, "mesh_sample_strategy", "unknown")
        print(f" ACTIVE SAMPLING STRATEGY: {strategy.upper()}")

        if stage == "fit":
            self.train_dataset = JomonDataset(split="train", **self._kwargs())
            self.test_dataset = JomonDataset(split="test", **self._kwargs())
            
            # Save visual samples to verify sampling strategy
            self._save_sample_visuals(self.train_dataset, f"logs/samples_{strategy}_train", n_samples=3)
            self._save_sample_visuals(self.test_dataset, f"logs/samples_{strategy}_test", n_samples=3)
            
        elif stage in ("test", "predict"):
            self.test_dataset = JomonDataset(split="test", **self._kwargs())
            self._save_sample_visuals(self.test_dataset, f"logs/samples_{strategy}_test", n_samples=3)

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        nw = int(self.hparams.num_workers)
        kw = dict(
            batch_size=self.hparams.batch_size,
            num_workers=nw,
            shuffle=shuffle,
            collate_fn=JomonDataset.collate_fn,
            pin_memory=True,
        )
        if nw > 0:   # spawn + persistent workers are only valid with num_workers > 0
            kw.update(
                persistent_workers=True,          # fork once, no per-epoch CoW spikes
                multiprocessing_context="spawn",  # workers start clean -> flat RSS
            )
        return DataLoader(dataset, **kw)

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)