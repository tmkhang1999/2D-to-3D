import numpy as np
import open3d as o3d


def create_mesh(cloud: o3d.geometry.PointCloud,
                depth: int = 10,
                trim: float = 0.15,
                target_triangles: int = 500_000):
    """
    Builds a watertight mesh, trims very low-support faces, and
    simplifies to a reasonable triangle budget.
    """
    # make sure normals exist
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                radius=0.02, max_nn=30))
    cloud.orient_normals_consistent_tangent_plane(50)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                          cloud, depth=depth, linear_fit=True)

    densities = np.asarray(densities)
    verts_to_remove = densities < np.quantile(densities, trim)
    mesh.remove_vertices_by_mask(verts_to_remove)

    # crop to cloud’s AABB to kill floating artefacts
    mesh = mesh.crop(cloud.get_axis_aligned_bounding_box())

    # simplify
    mesh = mesh.simplify_quadric_decimation(target_triangles)
    mesh.compute_vertex_normals()

    return mesh
