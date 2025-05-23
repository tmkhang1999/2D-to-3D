import open3d as o3d
import sys
from pathlib import Path


def visualize_file(file_path):
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return

    # Determine file type and load accordingly
    if file_path.suffix.lower() == '.ply':
        if 'mesh' in file_path.name:
            # Load as mesh
            geometry = o3d.io.read_triangle_mesh(Path(file_path))
            geometry.compute_vertex_normals()
            print(f"Loaded mesh with {len(geometry.vertices)} vertices and {len(geometry.triangles)} triangles")
        else:
            # Load as point cloud
            geometry = o3d.io.read_point_cloud(Path(file_path))
            print(f"Loaded point cloud with {len(geometry.points)} points")
    else:
        print(f"Unsupported file format: {file_path.suffix}")
        return

    # Visualize
    print(f"Visualizing {file_path.name}...")
    o3d.visualization.draw_geometries([geometry])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        visualize_file(sys.argv[1])
    else:
        # Default: visualize all results
        results_dir = Path("dataset/results")

        for file_path in results_dir.glob("*.ply"):
            visualize_file(file_path)