import numpy as np
import trimesh
import open3d as o3d
from scipy.spatial import cKDTree
from collections import deque
import argparse
from pathlib import Path
import traceback


class PotteryDestroyer:

    def __init__(self,
                 mesh_path,
                 num_pieces=10,
                 randomness=0.5,
                 smoothness_weight=0.3):
        self.mesh_path = mesh_path
        self.num_pieces = num_pieces
        self.randomness = randomness
        self.smoothness_weight = smoothness_weight

        scene = trimesh.load(str(mesh_path), force="scene")
        self.mesh = trimesh.util.concatenate(scene.geometry.values())
        self.mesh.fix_normals()

        try:
            vertex_color_trimesh = self.mesh.visual.to_color().vertex_colors
            self.vertex_colors = vertex_color_trimesh[:, :3] / 255.0
            print(
                f"Loaded mesh: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces"
            )
            print(f"Extracted vertex colors: {self.vertex_colors.shape}")
        except Exception:
            print("Warning: Could not extract vertex colors. Using white.")
            self.vertex_colors = np.ones((len(self.mesh.vertices), 3))

    def generate_seed_points(self):
        vertices = self.mesh.vertices
        if self.randomness < 0.01:
            seeds = self._furthest_point_sampling(vertices, self.num_pieces)
        elif self.randomness > 0.99:
            indices = np.random.choice(len(vertices),
                                       self.num_pieces,
                                       replace=False)
            seeds = vertices[indices]
        else:
            fps_count = int(self.num_pieces * (1 - self.randomness))
            random_count = self.num_pieces - fps_count
            fps_seeds = self._furthest_point_sampling(vertices, fps_count)
            random_indices = np.random.choice(len(vertices),
                                              random_count,
                                              replace=False)
            random_seeds = vertices[random_indices]
            seeds = np.vstack([fps_seeds, random_seeds])
        return seeds

    def _furthest_point_sampling(self, points, n_samples):
        n_points = len(points)
        selected_indices = np.zeros(n_samples, dtype=int)
        distances = np.full(n_points, np.inf)
        selected_indices[0] = np.random.randint(n_points)
        for i in range(1, n_samples):
            last_selected_idx = selected_indices[i - 1]
            dist_to_last = np.linalg.norm(points - points[last_selected_idx],
                                          axis=1)
            distances = np.minimum(distances, dist_to_last)
            selected_indices[i] = np.argmax(distances)
        return points[selected_indices]

    def create_fractured_pieces(self, face_labels):
        pass

    def save_pieces(self, pieces, output_dir):
        pass

    def destroy(self, output_dir="output"):
        pass

def main():
    parser = argparse.ArgumentParser(
        description='Destroy a 3D model into solid fragments.')
    parser.add_argument(
        'input',
        type=str,
        help='Input .glb file or directory containing model files.')
    parser.add_argument('--pieces',
                        type=int,
                        default=15,
                        help='Number of pieces to create.')
    parser.add_argument('--randomness',
                        type=float,
                        default=0.5,
                        help='Randomness factor (0.0=uniform, 1.0=random).')
    parser.add_argument(
        '--smoothness',
        type=float,
        default=0.5,
        help='How much to prefer smooth break lines (0.0-1.0).')
    parser.add_argument('--output',
                        type=str,
                        default='output',
                        help='Output directory.')

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}")
        return

    input_files = []
    if input_path.is_dir():
        for ext in ('*.glb', '*.obj', '*.gltf', '*.ply'):
            input_files.extend(list(input_path.glob(f'**/{ext}')))
    else:
        input_files.append(input_path)

    if not input_files:
        print(f"No compatible model files found at: {args.input}")
        return

    for file_path in input_files:
        print(f"\n{'='*20} Starting: {file_path.name} {'='*20}")
        try:
            specific_output_dir = Path(args.output) / file_path.stem

            destroyer = PotteryDestroyer(str(file_path),
                                         num_pieces=args.pieces,
                                         randomness=args.randomness,
                                         smoothness_weight=args.smoothness)
            destroyer.destroy(specific_output_dir)

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            traceback.print_exc()
            continue

    print(
        f"\n{'='*60}\nAll files processed! Results saved to: {args.output}\n{'='*60}"
    )


if __name__ == "__main__":
    main()
