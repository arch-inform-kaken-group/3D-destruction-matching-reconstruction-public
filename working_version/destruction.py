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
        self.smoothness_weight = smoothness_weight  # How much to prefer smooth boundaries

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

    def _calculate_edge_smoothness(self, face_idx1, face_idx2):
        """Calculate how smooth the edge is between two faces (lower = smoother)"""
        normal1 = self.mesh.face_normals[face_idx1]
        normal2 = self.mesh.face_normals[face_idx2]

        # Dot product: 1 = parallel, -1 = opposite, 0 = perpendicular
        dot = np.dot(normal1, normal2)

        # Convert to smoothness score: 0 = smooth (parallel), 2 = sharp (opposite)
        smoothness = 1.0 - dot
        return smoothness

    def assign_faces_to_regions_with_smoothness(self, seed_points):
        """Flood fill that prefers to create smooth boundaries"""
        n_faces = len(self.mesh.faces)
        face_labels = -np.ones(n_faces, dtype=int)

        # Build adjacency
        adjacency = [[] for _ in range(n_faces)]
        for i, j in self.mesh.face_adjacency:
            adjacency[i].append(j)
            adjacency[j].append(i)

        # Find initial seed faces
        face_centers = self.mesh.triangles_center
        seed_tree = cKDTree(seed_points)
        _, closest_seed_indices = seed_tree.query(face_centers, k=1)

        # Priority queue: (priority, face_idx, label)
        # Lower priority = processed first
        # Priority = distance_to_seed * (1 - smoothness_weight) + edge_sharpness * smoothness_weight
        from heapq import heappush, heappop
        pq = []

        # Initialize with seed faces
        for seed_idx in range(len(seed_points)):
            start_faces = np.where(closest_seed_indices == seed_idx)[0]
            for face_idx in start_faces:
                if face_labels[face_idx] == -1:
                    face_labels[face_idx] = seed_idx
                    dist = np.linalg.norm(face_centers[face_idx] -
                                          seed_points[seed_idx])
                    priority = dist * (1.0 - self.smoothness_weight)
                    heappush(pq, (priority, face_idx, seed_idx))

        # Flood fill with smoothness preference
        while pq:
            current_priority, current_face, current_label = heappop(pq)

            # Process neighbors
            for neighbor_face in adjacency[current_face]:
                if face_labels[neighbor_face] == -1:
                    face_labels[neighbor_face] = current_label

                    # Calculate priority for this face
                    dist = np.linalg.norm(face_centers[neighbor_face] -
                                          seed_points[current_label])
                    edge_sharpness = self._calculate_edge_smoothness(
                        current_face, neighbor_face)

                    priority = (dist * (1.0 - self.smoothness_weight) +
                                edge_sharpness * self.smoothness_weight)

                    heappush(pq, (priority, neighbor_face, current_label))

        # Handle any unlabeled faces
        unlabeled = np.where(face_labels == -1)[0]
        if len(unlabeled) > 0:
            _, labels = seed_tree.query(face_centers[unlabeled], k=1)
            face_labels[unlabeled] = labels

        return face_labels

    def create_fractured_pieces(self, face_labels):
        """Create open mesh pieces"""
        pieces = []

        for label in range(self.num_pieces):
            piece_face_indices = np.where(face_labels == label)[0]
            if len(piece_face_indices) == 0:
                continue

            # Extract faces for this piece
            piece_faces = self.mesh.faces[piece_face_indices]
            unique_verts_old = np.unique(piece_faces.flatten())
            old_to_new = {
                old_idx: new_idx
                for new_idx, old_idx in enumerate(unique_verts_old)
            }
            new_faces_flat = [old_to_new[v] for v in piece_faces.flatten()]
            new_faces = np.array(new_faces_flat).reshape((-1, 3))
            new_vertices = self.mesh.vertices[unique_verts_old]
            new_colors = self.vertex_colors[unique_verts_old]

            # Create open mesh
            piece_mesh = trimesh.Trimesh(vertices=new_vertices,
                                         faces=new_faces,
                                         process=False)

            print(
                f"  Piece {label}: {len(piece_mesh.vertices)} verts, {len(piece_mesh.faces)} faces"
            )

            pieces.append((piece_mesh, new_colors))

        return pieces

    def save_pieces(self, pieces, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for i, (piece_mesh, piece_colors) in enumerate(pieces):
            filename = output_path / f"piece_{i:03d}.ply"
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(piece_mesh.vertices)
            mesh_o3d.triangles = o3d.utility.Vector3iVector(piece_mesh.faces)
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(piece_colors)
            o3d.io.write_triangle_mesh(str(filename),
                                       mesh_o3d,
                                       write_ascii=True)
            print(f"Saved: {filename}")

    def destroy(self, output_dir="output"):
        print("\n=== Starting Pottery Destruction ===")
        print(f"Target pieces: {self.num_pieces}")
        print(f"Randomness: {self.randomness}")
        print(f"Smoothness weight: {self.smoothness_weight}")
        print("\n1. Generating seed points...")
        seeds = self.generate_seed_points()
        print(
            "\n2. Assigning faces to regions (with smoothness preference)...")
        face_labels = self.assign_faces_to_regions_with_smoothness(seeds)
        print("\n3. Creating fractured pieces...")
        pieces = self.create_fractured_pieces(face_labels)
        print(f"\n4. Saving {len(pieces)} pieces...")
        self.save_pieces(pieces, output_dir)
        print(f"\n=== Complete! Saved pieces to {output_dir} ===")
        return pieces


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
