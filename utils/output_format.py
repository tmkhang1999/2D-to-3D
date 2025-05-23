from pathlib import Path
import open3d as o3d


# ETH3D expects specific file formats and naming conventions
def save_for_eth3d_evaluation(cloud, mesh, scene_name, method_name):
    """Save reconstruction in ETH3D evaluation format"""
    eval_dir = Path(f"eth3d_results/{method_name}/{scene_name}")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Point cloud in PLY format (specific coordinate system)
    o3d.io.write_point_cloud(Path(eval_dir / "pointcloud.ply"), cloud)

    # Mesh in PLY format
    if mesh is not None:
        o3d.io.write_triangle_mesh(Path(eval_dir / "mesh.ply"), mesh)