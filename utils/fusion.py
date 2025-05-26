import numpy as np
import open3d as o3d
from utils.stereo import rectify_pair, compute_filtered_disparity, disparity_to_cloud



def clean_cloud(pcl, colors, voxel_size=0.01, nb_neighbors=20, std_ratio=2.0):
    """
    Additional point cloud cleaning function
    """
    import open3d as o3d

    if len(pcl) == 0:
        print("Warning: Empty point cloud provided for cleaning")
        return o3d.geometry.PointCloud()

    # Create Open3D point cloud
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pcl)

    if colors is not None and len(colors) > 0:
        # Normalize colors to [0,1] range
        if colors.max() > 1.0:
            colors = colors.astype(np.float32) / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors)

    print(f"Original cloud: {len(cloud.points)} points")

    # Voxel downsampling
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size)
        print(f"After voxel downsampling: {len(cloud.points)} points")

    # Statistical outlier removal
    if len(cloud.points) > nb_neighbors:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors, std_ratio)
        print(f"After statistical filtering: {len(cloud.points)} points")

    # Remove isolated points
    if len(cloud.points) > 50:
        cloud, _ = cloud.remove_radius_outlier(nb_points=10, radius=0.05)
        print(f"After radius filtering: {len(cloud.points)} points")

    return cloud

# def clean_cloud(pcl, colors=None, voxel=0.002):
#     """
#     Filter outliers and regularize point cloud for ETH3D datasets.
#
#     Args:
#         pcl: Nx3 array of points or Open3D point cloud
#         colors: Optional Nx3 array of RGB colors
#         voxel: Voxel size for downsampling (default: 0.002 for indoor scenes)
#
#     Returns:
#         o3d.geometry.PointCloud: Cleaned point cloud
#     """
#     # Handle different input types
#     if isinstance(pcl, np.ndarray):
#         cloud = o3d.geometry.PointCloud()
#         cloud.points = o3d.utility.Vector3dVector(pcl)
#         if colors is not None:
#             colors_float = colors.astype(np.float64) / 255.0
#             cloud.colors = o3d.utility.Vector3dVector(colors_float)
#     else:
#         cloud = pcl
#
#     # Early exit if empty
#     if len(cloud.points) == 0:
#         return cloud
#
#     # Remove NaN and infinite points
#     cloud = cloud.remove_non_finite_points()
#
#     if len(cloud.points) == 0:
#         return cloud
#
#     # Statistical outlier removal - balanced for ETH3D indoor scenes
#     cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)
#
#     if len(cloud.points) == 0:
#         return cloud
#
#     # Radius outlier removal - tuned for indoor scene density
#     cloud, _ = cloud.remove_radius_outlier(nb_points=25, radius=0.04)
#
#     if len(cloud.points) == 0:
#         return cloud
#
#     # Voxel downsampling - good balance for indoor scene detail
#     cloud = cloud.voxel_down_sample(voxel)
#
#     # Estimate normals for subsequent meshing
#     cloud.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
#     cloud.orient_normals_consistent_tangent_plane(40)
#
#     return cloud


def ensure_normals(cloud, radius=0.05, max_nn=50):
    """Ensure point cloud has oriented normals.

    Args:
        cloud: Open3D point cloud
        radius: Search radius for normal estimation
        max_nn: Maximum nearest neighbors for normal estimation
    """
    if not cloud.has_normals() or len(cloud.normals) == 0:
        try:
            # Estimate normals
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius, max_nn=max_nn))

            # Orient normals towards camera center (assuming camera at origin)
            cloud.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
        except Exception as e:
            print(f"Warning: Normal estimation failed - {str(e)}")
            # Create dummy normals pointing up
            pts = np.asarray(cloud.points)
            normals = np.tile([0, 0, 1], (len(pts), 1))
            cloud.normals = o3d.utility.Vector3dVector(normals)


def fuse_multi_view_eth3d(ref_id, neighbor_ids, imgs, K, T_wc,
                          scene_dir, resize=0.5,
                          voxel=0.002, icp_thresh=0.01,
                          matchers=None):
    """
    Fuse multiple views into a single point cloud, optimized for ETH3D indoor scenes.

    This function implements:
    1. Camera-aware point cloud generation via rectify_pair_eth3d
    2. Progressive multi-scale ICP registration
    3. Robust outlier handling with adaptive thresholds
    4. Visibility-consistent normal orientation
    5. Multi-stage cleaning for optimal mesh generation

    Args:
        ref_id: ID of reference view
        neighbor_ids: List of neighbor view IDs
        imgs: Dictionary of image objects
        K: Dictionary of camera matrices
        T_wc: Dictionary of world-to-camera transforms
        scene_dir: Path to scene directory
        resize: Image resize factor
        voxel: Voxel size for downsampling
        icp_thresh: Base ICP distance threshold
        matchers: Optional tuple of (left_matcher, right_matcher, wls_filter)

    Returns:
        o3d.geometry.PointCloud: Fused point cloud in world coordinates
    """
    if not neighbor_ids:
        print("No neighbor views provided for fusion")
        return None

    def cloud_from_pair(id1, id2):
        # Use optimized rectification function for ETH3D
        rect1, rect2, Q, rect_info = rectify_pair(
            id1, id2, imgs, K, T_wc, scene_dir, resize)

        if matchers is None:
            # Create matchers if not provided
            from utils.stereo import sgbm_with_consistency
            l_m, r_m, wls = sgbm_with_consistency()
        else:
            l_m, r_m, wls = matchers

        # Compute disparity
        disp, mask = compute_filtered_disparity(rect1, rect2, l_m, r_m, wls)

        # Convert to point cloud with enhanced filtering
        pts, rgb = disparity_to_cloud(disp, Q, mask, rect1,
                                      min_depth=0.5, max_depth=30.0)

        # Early exit if no valid points
        if pts.size == 0 or len(pts) < 100:
            return None

        # Create Open3D point cloud
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        if rgb is not None:
            pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float32) / 255.0)

        # Transform to world frame
        T_w1 = np.linalg.inv(T_wc[id1])
        pc.transform(T_w1)

        # Store camera center for normal orientation
        cam_center = T_w1[:3, 3]

        # Clean the individual cloud before fusion
        clean_pc = clean_cloud(pc, voxel=voxel)

        # Orient normals toward camera center for better registration
        if len(clean_pc.points) > 0:
            clean_pc.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
            clean_pc.orient_normals_towards_camera_location(cam_center)

        return clean_pc

    # Process reference view with first neighbor
    print(f"Processing reference view {ref_id} with neighbor {neighbor_ids[0]}")
    ref_cloud = cloud_from_pair(ref_id, neighbor_ids[0])

    # Check if reference cloud is valid
    if ref_cloud is None or len(ref_cloud.points) == 0:
        print(f"Warning: No valid points in reference cloud (views {ref_id}-{neighbor_ids[0]})")
        if len(neighbor_ids) > 1:
            print(f"Trying next neighbor {neighbor_ids[1]} as reference")
            ref_cloud = cloud_from_pair(ref_id, neighbor_ids[1])
            if ref_cloud is None or len(ref_cloud.points) == 0:
                print("Failed to create initial reference cloud. Aborting fusion.")
                return None
            neighbor_ids = neighbor_ids[2:] if len(neighbor_ids) > 2 else []
        else:
            return None

    # Iterate through remaining neighbors
    from tqdm import tqdm
    for i, nb_id in enumerate(tqdm(neighbor_ids[1:], desc="Fusing views")):
        nb_cloud = cloud_from_pair(ref_id, nb_id)

        if nb_cloud is None or len(nb_cloud.points) == 0:
            print(f"Warning: No points from neighbor {nb_id} - skipping")
            continue

        # Only use ICP for clouds with sufficient points
        if len(nb_cloud.points) < 200:
            print(f"Warning: Too few points from neighbor {nb_id} - using direct merge")
            ref_cloud += nb_cloud
            continue

        # Multi-scale ICP for robust alignment
        current_thresh = icp_thresh * 4
        current_trans = np.eye(4)

        for scale in range(3):  # Coarse to fine registration
            # Adjust criteria based on cloud size
            max_iter = 50 if scale == 0 else (80 if scale == 1 else 100)

            reg = o3d.pipelines.registration.registration_icp(
                nb_cloud, ref_cloud,
                current_thresh,
                current_trans,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=max_iter, relative_fitness=1e-6, relative_rmse=1e-6)
            )

            current_trans = reg.transformation
            current_thresh /= 2

            # Early termination if registration is already good
            if reg.fitness > 0.95:
                break

        # Apply transformation
        nb_cloud.transform(current_trans)

        # Merge with reference cloud
        ref_cloud += nb_cloud

        # Intermediate processing every few iterations to keep cloud size manageable
        if i % 3 == 2 or i == len(neighbor_ids) - 2:
            ref_cloud = ref_cloud.voxel_down_sample(voxel)
            ref_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
            ref_cloud.orient_normals_consistent_tangent_plane(40)

    # Final light cleaning before returning
    if len(ref_cloud.points) > 0:
        ref_cloud = ref_cloud.voxel_down_sample(voxel)
        # Simple outlier removal to remove obvious artifacts
        ref_cloud, _ = ref_cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

    return ref_cloud


def prepare_for_meshing(cloud, nb_neighbors=50, std_ratio=1.8,
                        radius=0.05, voxel=0.002):
    """
    Prepares a point cloud for mesh generation with ETH3D-specific parameters.

    Args:
        cloud: Input point cloud
        nb_neighbors: Number of neighbors for statistical outlier removal
        std_ratio: Standard deviation ratio for statistical outlier removal
        radius: Radius for radius outlier removal
        voxel: Voxel size for downsampling

    Returns:
        Prepared point cloud ready for meshing
    """
    if len(cloud.points) == 0:
        return cloud

    # Apply multi-stage filtering with parameters tuned for ETH3D indoor scenes
    cloud = cloud.voxel_down_sample(voxel)
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    cloud, _ = cloud.remove_radius_outlier(nb_points=30, radius=radius)

    # Final normal estimation for meshing
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
    cloud.orient_normals_consistent_tangent_plane(40)

    return cloud


def create_mesh(cloud: o3d.geometry.PointCloud,
                depth: int = 10,
                trim: float = 0.15,
                target_triangles: int = 500_000,
                prepare_cloud: bool = True):
    """
    Builds a watertight mesh, trims very low-support faces, and
    simplifies to a reasonable triangle budget.

    Args:
        cloud: Input point cloud
        depth: Depth parameter for Poisson reconstruction
        trim: Trimming factor for low-density vertices
        target_triangles: Target number of triangles after simplification
        prepare_cloud: Whether to run preparation steps (set to False if already prepared)

    Returns:
        o3d.geometry.TriangleMesh: Reconstructed mesh or None if reconstruction fails
    """
    try:
        # Prepare the cloud for meshing if requested
        if prepare_cloud:
            cloud = prepare_for_meshing(cloud)

        # Ensure normals are available and consistently oriented
        if not cloud.has_normals() or len(cloud.normals) == 0:
            cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.02, max_nn=30))
            cloud.orient_normals_consistent_tangent_plane(50)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud, depth=depth, linear_fit=True)

        densities = np.asarray(densities)
        verts_to_remove = densities < np.quantile(densities, trim)
        mesh.remove_vertices_by_mask(verts_to_remove)

        # crop to cloud's AABB to kill floating artefacts
        mesh = mesh.crop(cloud.get_axis_aligned_bounding_box())

        # simplify
        mesh = mesh.simplify_quadric_decimation(target_triangles)
        mesh.compute_vertex_normals()

        return mesh
    except Exception as e:
        print(f"Mesh reconstruction failed: {str(e)}")
        return None
