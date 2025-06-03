import numpy as np
import open3d as o3d
from utils.stereo import rectify_pair, compute_filtered_disparity, disparity_to_cloud
import cv2


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


def prepare_for_meshing(cloud, nb_neighbors=50, std_ratio=2.0,
                        radius=0.06, voxel=0.002):
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
        print("Warning: Empty cloud provided to prepare_for_meshing")
        return cloud

    # Apply multi-stage filtering with parameters tuned for ETH3D indoor scenes
    cloud = cloud.voxel_down_sample(voxel)
    cloud, ind = cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"prepare_for_meshing: Removed {len(ind)} statistical outliers.")

    cloud, ind = cloud.remove_radius_outlier(nb_points=30, radius=radius)
    print(f"prepare_for_meshing: Removed {len(ind)} radius outliers.")

    # Final normal estimation for meshing
    if len(cloud.points) > 100:
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
        cloud.orient_normals_consistent_tangent_plane(40)
    else:
        print("Warning: Not enough points after filtering to estimate normals for meshing.")

    return cloud


def create_mesh(cloud: o3d.geometry.PointCloud,
                depth: int = 8,
                trim: float = 0.25,
                target_triangles: int = 500_000,
                prepare_cloud: bool = True):
    """
    Builds a mesh using ball pivoting.
    """
    try:
        # Prepare the cloud for meshing if requested
        if prepare_cloud:
            cloud = prepare_for_meshing(cloud)

        if len(cloud.points) < 1000:
            print(f"Warning: Too few points ({len(cloud.points)}) for reliable meshing.")
            return None

        # Ensure normals are available and consistently oriented
        if not cloud.has_normals() or len(cloud.normals) == 0:
            print("Warning: Normals not available, estimating for meshing.")
            cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.02, max_nn=30))
            cloud.orient_normals_consistent_tangent_plane(50)

        # Use ball pivoting for mesh creation
        radii = [0.005, 0.01, 0.02, 0.04, 0.08]  # Use increasing radii
        print("Attempting ball pivoting mesh creation...")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            cloud, o3d.utility.DoubleVector(radii))

        # Clean up the mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        print(f"Ball pivoting created mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")

        # Final cleanup and simplification
        if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
            mesh = mesh.crop(cloud.get_axis_aligned_bounding_box())
            # Simplify to a reasonable number of triangles if it's very large
            if len(mesh.triangles) > target_triangles * 1.5:
                print(f"Simplifying mesh from {len(mesh.triangles)} to {target_triangles} triangles.")
                mesh = mesh.simplify_quadric_decimation(target_triangles)
            mesh.compute_vertex_normals()
            print(f"Final mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles after cleanup.")
        else:
            print("Warning: Final mesh is empty after cleanup/simplification.")
            return None  # Return None if the mesh becomes empty

        return mesh
    except Exception as e:
        print(f"Mesh reconstruction failed completely in create_mesh: {str(e)}")
        return None


def create_textured_mesh(mesh, images, K, T_wc, scene_dir):
    """Create textured mesh from multiple views with blending"""
    print("Creating textured mesh with blending...")
    
    # Create UV map (this part is often done before texturing, but ensuring normals here)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    # Initialize texture atlas
    texture_size = 2048
    texture_atlas = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    uv_coords = np.zeros((len(mesh.vertices), 2))
    
    # Project vertices to each image, sample colors, and blend
    vertices = np.asarray(mesh.vertices)
    vertex_colors = np.zeros((len(vertices), 3), dtype=np.float32) # Use float for blending
    vertex_weights = np.zeros(len(vertices), dtype=np.float32)
    
    for i, vertex in enumerate(vertices):
        visible_views_data = [] # Store (score, color) for visible views
        
        for img_id, (K_i, T_i) in zip(images.keys(), zip(K.values(), T_wc.values())):
            # Project vertex to image
            vertex_h = np.append(vertex, 1)
            cam_vertex = T_i @ vertex_h
            if cam_vertex[2] <= 1e-6:  # Skip if behind or very close to camera
                continue
                
            proj = K_i @ cam_vertex[:3]
            
            # Avoid division by zero or near-zero depth
            if abs(proj[2]) < 1e-6:
                 continue

            proj = proj[:2] / proj[2]
            
            # Check if projection is within image bounds
            h_img, w_img = images[img_id].shape[:2]
            if 0 <= proj[0] < w_img and 0 <= proj[1] < h_img:
                # Compute view score based on viewing angle (dot product of normal and view direction)
                normal = np.asarray(mesh.vertex_normals[i])
                view_dir = -cam_vertex[:3] / np.linalg.norm(cam_vertex[:3])
                score = np.maximum(0.0, np.dot(normal, view_dir)) # Use np.maximum to ensure non-negative score
                
                if score > 0.05: # Only consider views with a reasonable viewing angle
                    # Get color from image (bilinear interpolation for smoother result)
                    x, y = proj[0], proj[1]
                    # Clamp coordinates to prevent issues near boundaries
                    x = np.clip(x, 0, w_img - 1.001)
                    y = np.clip(y, 0, h_img - 1.001)
                    
                    # Simple nearest neighbor for now, can be upgraded to bilinear
                    color = images[img_id][int(y), int(x)] # BGR order from cv2.imread
                    visible_views_data.append((score, color))

        # Blend colors from visible views
        if visible_views_data:
            total_score = sum(data[0] for data in visible_views_data)
            if total_score > 1e-6: # Avoid division by zero
                 blended_color = np.sum([data[0] * data[1] for data in visible_views_data], axis=0) / total_score
                 vertex_colors[i] = blended_color
                 vertex_weights[i] = total_score

    # Only keep vertices that were visible in at least one view
    valid_vertices_mask = vertex_weights > 0
    valid_vertices = vertices[valid_vertices_mask]
    valid_vertex_colors = vertex_colors[valid_vertices_mask]

    # If no vertices were visible, return the original mesh (or an empty one)
    if not valid_vertices_mask.any():
         print("Warning: No mesh vertices were visible in any image for texturing.")
         return mesh # Or o3d.geometry.TriangleMesh() if you prefer an empty mesh

    # --- Texture Atlas Creation and UV Mapping ---
    # Simple packing: divide texture into grid based on number of valid vertices
    num_valid_vertices = len(valid_vertices)
    grid_size = int(np.ceil(np.sqrt(num_valid_vertices))) # Use num_valid_vertices for grid size
    cell_size = texture_size // grid_size
    
    # Create a new texture atlas with blended colors
    texture_atlas_blended = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    uv_coords_blended = np.zeros((num_valid_vertices, 2))

    for i in range(num_valid_vertices):
        row = i // grid_size
        col = i % grid_size
        y_start = row * cell_size
        x_start = col * cell_size
        
        # Ensure blended_color is in uint8 [0, 255] range
        color_uint8 = np.clip(valid_vertex_colors[i], 0, 255).astype(np.uint8)
        
        # Fill a small square in the texture atlas with the blended color
        texture_atlas_blended[y_start:y_start + cell_size, x_start:x_start + cell_size] = color_uint8
        
        # Update UV coordinates to point to the center of this cell
        uv_coords_blended[i] = np.array([
            (x_start + cell_size/2) / texture_size,
            (y_start + cell_size/2) / texture_size
        ])
    
    # Save texture atlas
    texture_path = scene_dir / "texture_atlas_blended.png"
    cv2.imwrite(str(texture_path), cv2.cvtColor(texture_atlas_blended, cv2.COLOR_RGB2BGR))
    
    # Create a new mesh with only the valid vertices and assign UVs
    # Need to remap triangles to the new vertex indices
    # This part is complex and might require a different approach (like Open3D's built-in texturing) 
    # For now, let's simplify and just color the original mesh vertices with blended colors
    # and assign the simple grid UVs to the original mesh vertices (will work visually but UV layout isn't optimal)

    # Assign blended colors directly to mesh vertices (for visualization in Open3D)
    # Ensure vertex_colors is in the range [0, 1] for Open3D visualization
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(vertex_colors / 255.0, 0.0, 1.0))

    # Assign simple grid UVs to the original mesh vertices (will work with the texture atlas)
    # The UV coordinates should correspond to the original vertices index.
    # We need a UV coordinate for EVERY vertex in the original mesh, even if it wasn't visible.
    # For non-visible vertices, their UVs can point to a black/transparent area in the atlas.

    # Create UVs for all original vertices
    uv_coords_all = np.zeros((len(vertices), 2)) # Default to (0,0) or similar for non-visible
    # Fill in UVs only for valid vertices using the blended UVs
    # This requires mapping the valid_vertices_mask back to the original vertex indices
    
    # Let's simplify for now: Assign simple grid UVs to all vertices based on their original index
    # and use the blended texture atlas. This is not ideal but a starting point.
    num_all_vertices = len(vertices)
    grid_size_all = int(np.ceil(np.sqrt(num_all_vertices)))
    cell_size_all = texture_size // grid_size_all

    for i in range(num_all_vertices):
        row = i // grid_size_all
        col = i % grid_size_all
        x_center = (col * cell_size_all + cell_size_all / 2.0) / texture_size
        y_center = (row * cell_size_all + cell_size_all / 2.0) / texture_size
        uv_coords_all[i] = np.array([x_center, y_center])
        
        # Fill the texture atlas with the blended color if the vertex was visible
        if valid_vertices_mask[i]:
             y_start = row * cell_size_all
             x_start = col * cell_size_all
             color_uint8 = np.clip(vertex_colors[i], 0, 255).astype(np.uint8)
             texture_atlas_blended[y_start:y_start + cell_size_all, x_start:x_start + cell_size_all] = color_uint8
        # else: The cell remains black (initialized with zeros)


    # Save the updated texture atlas
    texture_path = scene_dir / "texture_atlas_blended.png"
    cv2.imwrite(str(texture_path), cv2.cvtColor(texture_atlas_blended, cv2.COLOR_RGB2BGR)) # Save as BGR for cv2 compatibility

    # Assign UV coordinates and material to the original mesh
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_coords_all)
    mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(mesh.triangles), dtype=np.int32)) # Assuming one material
    
    # Open3D visualization might not use this texture directly, but the PLY export should

    # Note: Proper texture mapping with UV unwrapping and atlas packing is complex.
    # This implementation uses a simplified grid packing based on vertex index.
    # A more advanced approach would involve proper UV unwrapping based on mesh topology.

    return mesh


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
        if pts is None or len(pts) < 100:
            return None

        # Clean the individual cloud before fusion (pass numpy arrays, not PointCloud)
        clean_pc = clean_cloud(pts, rgb, voxel_size=voxel)

        # Transform to world frame
        T_w1 = np.linalg.inv(T_wc[id1])
        clean_pc.transform(T_w1)

        # Store camera center for normal orientation
        cam_center = T_w1[:3, 3]

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
