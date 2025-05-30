import numpy as np
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from utils import (load_middlebury_calib, sgbm_with_consistency, compute_filtered_disparity,
                   disparity_to_cloud, create_gt_cloud, clean_cloud,
                   compute_metrics, read_pfm_file, compute_cloud_metrics,
                   prepare_for_meshing, create_mesh)
from utils.visualization import save_rectified_pair


def align_point_clouds(source, target, voxel_size=0.005, max_iterations=50):
    """Align point clouds using multi-scale ICP"""
    print("Aligning point clouds...")
    
    # Downsample for faster alignment
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    # Estimate normals for point-to-plane ICP
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # Multi-scale ICP
    current_transformation = np.eye(4)
    for scale in range(3):
        iter_count = max_iterations // (scale + 1)
        distance_threshold = voxel_size * (2 ** scale)

        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, distance_threshold,
            current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=iter_count,
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )
        )
        current_transformation = result.transformation

        if result.fitness > 0.95:  # Early termination if alignment is good
            break

    # Apply transformation to full resolution cloud
    source.transform(current_transformation)
    return source, result.fitness


def create_textured_mesh(mesh, images, K, T_wc, scene_dir):
    """Create textured mesh from multiple views"""
    print("Creating textured mesh...")

    # Create UV map
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # Initialize texture atlas
    texture_size = 2048
    texture_atlas = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    uv_coords = np.zeros((len(mesh.vertices), 2))

    # Project vertices to each image and find best view
    vertices = np.asarray(mesh.vertices)
    for i, vertex in enumerate(vertices):
        best_score = -1
        best_uv = None

        for img_id, (K_i, T_i) in zip(images.keys(), zip(K.values(), T_wc.values())):
            # Project vertex to image
            vertex_h = np.append(vertex, 1)
            cam_vertex = T_i @ vertex_h
            if cam_vertex[2] <= 0:  # Skip if behind camera
                continue

            proj = K_i @ cam_vertex[:3]
            proj = proj[:2] / proj[2]

            # Check if projection is within image bounds
            if 0 <= proj[0] < images[img_id].shape[1] and 0 <= proj[1] < images[img_id].shape[0]:
                # Compute view score based on viewing angle
                normal = np.asarray(mesh.vertex_normals[i])
                view_dir = -cam_vertex[:3] / np.linalg.norm(cam_vertex[:3])
                score = np.abs(np.dot(normal, view_dir))

                if score > best_score:
                    best_score = score
                    # Convert to UV coordinates
                    u = proj[0] / images[img_id].shape[1]
                    v = proj[1] / images[img_id].shape[0]
                    best_uv = np.array([u, v])

        if best_uv is not None:
            uv_coords[i] = best_uv

    # Create texture atlas
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_coords)
    mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(mesh.triangles), dtype=np.int32))

    return mesh


# --- Chessboard pose estimation parameters ---
CHESSBOARD_SIZE = (7, 7)  # (corners_per_row, corners_per_col)
SQUARE_SIZE = 0.01  # meters (adjust to your chessboard)

# Prepare 3D model points for the chessboard (z=0 plane)
objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # Scale the object points by SQUARE_SIZE

# Store transformations for each scene
scene_to_chess_T = {}

if __name__ == "__main__":
    # Process all three chess scenes
    scenes = ["chess_1", "chess_2", "chess_3"]
    all_clouds = []
    all_gt_clouds = []
    all_metrics = []
    all_images = {}  # Store images for texture mapping
    pair_metrics = []  # Store metrics for each pair

    for scene_name in scenes:
        scene = Path(f"dataset/chess/{scene_name}")
        print(f"\nProcessing {scene_name}...")

        # 1. Load calibration and images
        print("1. Loading data...")
        K0, K1, doffs, baseline, w, h, ndisp, vmin, vmax = load_middlebury_calib(scene / "calib.txt")
        imgL = cv2.imread(str(scene / "im0.png"))
        imgR = cv2.imread(str(scene / "im1.png"))
        all_images[scene_name] = imgL  # Store left image for texture mapping

        baseline_m = baseline / 1000.0  # mm → m
        f = K0[0, 0]  # focal length in px
        cx = K0[0, 2]
        cy = K0[1, 2]
        print(f"Baseline: {baseline_m} m, Focal length: {f} px, Center: ({cx}, {cy}) px")

        print("Intrinsic Matrix K0:")
        print(K0)
        
        Q = np.array([
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, f],
            [0, 0, 1 / baseline_m, -doffs / baseline_m]
        ], dtype=np.float32)

        # Save rectified pair
        save_path = scene / "rectified_pair.png"
        save_rectified_pair(imgL, imgR, save_path)
        print(f"Saved rectified image pair to {save_path}")

        # 2. Compute disparity
        print("2. Computing disparity...")
        min_disp = vmin - doffs
        raw_span = vmax - min_disp
        num_disp = ((raw_span + 15) // 16) * 16
        left_matcher, right_matcher, wls_filter = sgbm_with_consistency(min_disp=int(min_disp), num_disp=int(num_disp))
        filtered_disp, valid_mask = compute_filtered_disparity(
            imgL, imgR, left_matcher, right_matcher, wls_filter
        )

        # Save disparity map
        filtered_disp_gray = (filtered_disp * 255 / np.max(filtered_disp)).astype(np.uint8)
        disparity_map_path = scene / f"disp_raw.png"
        cv2.imwrite(str(disparity_map_path), filtered_disp_gray)

        # 3. Generate point clouds
        print("3. Generating point clouds...")
        # Predicted cloud
        pts, colors = disparity_to_cloud(filtered_disp, Q, valid_mask, imgL)
        cloud = clean_cloud(pts, colors)

        # Ground truth cloud
        disp_gt, _ = read_pfm_file(scene / "disp0.pfm")
        cloud_gt = create_gt_cloud(disp_gt, Q, valid_mask, imgL)

        # Save individual clouds
        cloud_path = scene / f"cloud.ply"
        cloud_gt_path = scene / f"cloud_gt.ply"
        o3d.io.write_point_cloud(Path(cloud_path), cloud)
        o3d.io.write_point_cloud(Path(cloud_gt_path), cloud_gt)

        # Compute cloud-to-cloud metrics for this pair
        print(f"\nComputing cloud-to-cloud metrics for {scene_name}...")
        cloud_metrics = compute_cloud_metrics(cloud, cloud_gt)
        pair_metrics.append({
            'scene': scene_name,
            'chamfer_dist': cloud_metrics['chamfer_dist'],
            'accuracy': cloud_metrics['accuracy'],
            'completeness': cloud_metrics['completeness'],
            'f1_score': cloud_metrics['f1_score']
        })
        print(f"Chamfer Distance: {cloud_metrics['chamfer_dist']:.6f} m")
        print(f"Accuracy: {cloud_metrics['accuracy']:.4f}")
        print(f"Completeness: {cloud_metrics['completeness']:.4f}")
        print(f"F1 Score: {cloud_metrics['f1_score']:.4f}")

        all_clouds.append(cloud)
        all_gt_clouds.append(cloud_gt)

        # 4. Evaluate disparity
        print("4. Evaluating disparity...")
        mae, rms, bad1, bad2, bad4 = compute_metrics(filtered_disp, disp_gt, valid_mask)
        metrics = {
            'scene': scene_name,
            'mae': mae,
            'rms': rms,
            'bad1': bad1,
            'bad2': bad2,
            'bad4': bad4
        }
        all_metrics.append(metrics)
        print(f"Metrics for {scene_name}:")
        print(f"MAE: {mae:.3f}, RMS: {rms:.3f}")
        print(f"Bad1%: {bad1:.2f}, Bad2%: {bad2:.2f}, Bad4%: {bad4:.2f}")

        # --- 1. Detect chessboard corners in left image ---
        gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

        manual_corners_path = scene / "corners_manual.npy"
        if manual_corners_path.exists():
            print(f"Loading manual corners for {scene_name}...")
            corners_2d_all = np.load(str(manual_corners_path))
            if corners_2d_all.shape[0] != CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1]:
                print(f"[ERROR] Manual corners file {manual_corners_path} should contain {CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1]} points, but found {corners_2d_all.shape[0]}.")
                print("Please re-run annotation for this scene.")
                ret = False
            else:
                ret = True
                corners_2d = corners_2d_all # Use loaded points
        else:
            print(f"Manual corners not found, attempting automatic detection for {scene_name}...")
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners_2d = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags)

        # Save visualization of detected/loaded corners
        vis = imgL.copy()
        if ret and corners_2d is not None:
             cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners_2d, ret)
        else:
             print(f"Warning: Could not visualize corners for {scene_name} as detection/loading failed.")
        cv2.imwrite(str(scene / f"corners_vis.png"), vis)

        if not ret or corners_2d is None or len(corners_2d) != CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1]:
            print(f"[ERROR] Valid chessboard corners not available for pose estimation in {scene_name}")
            continue

        # --- 2. Reproject 2D corners to 3D using disparity and Q ---
        # Ensure corners_2d is float32 and has shape (N, 1, 2) for reprojectImageTo3D
        corners_2d_reproj = corners_2d.reshape(-1, 1, 2).astype(np.float32)

        # Get the 3D points for the 2D corner locations
        # Ensure indices are integers
        corner_indices_y = np.round(corners_2d_reproj[:, 0, 1]).astype(int)
        corner_indices_x = np.round(corners_2d_reproj[:, 0, 0]).astype(int)

        # Clamp indices to image bounds to prevent IndexError
        h, w = filtered_disp.shape[:2]
        corner_indices_y = np.clip(corner_indices_y, 0, h - 1)
        corner_indices_x = np.clip(corner_indices_x, 0, w - 1)

        points_3d_all = cv2.reprojectImageTo3D(filtered_disp, Q)[corner_indices_y, corner_indices_x]

        # Filter out invalid 3D points and corresponding 2D points and object points
        valid_mask_3d = np.isfinite(points_3d_all).all(axis=1)
        corners_3d = points_3d_all[valid_mask_3d]
        objp_valid = objp[valid_mask_3d]
        corners_2d_valid = corners_2d[valid_mask_3d]

        # --- 3. Estimate pose using solvePnP ---
        if len(corners_3d) >= 6:
            # Use camera intrinsics for solvePnP
            camera_matrix = K0
            dist_coeffs = np.zeros(5)  # Assume no distortion
            # Use corners_2d_valid as image points input to solvePnP
            success, rvec, tvec = cv2.solvePnP(objp_valid, corners_2d_valid, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                print(f"[ERROR] solvePnP failed for {scene_name}")
                continue
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3,:3] = R
            T[:3,3] = tvec.ravel()
            scene_to_chess_T[scene_name] = T

            print(f"Estimated Chessboard to Camera Transform (T) for {scene_name}:")
            print(T)

        else:
            print(f"[ERROR] Not enough valid (3D reprojected) corners for pose estimation in {scene_name} (found {len(corners_3d)} valid points)")
            continue

    # 5. Merge and refine point clouds
    print("\n5. Merging and refining point clouds...")
    merged_chess_cloud = o3d.geometry.PointCloud()
    merged_chess_gt_cloud = o3d.geometry.PointCloud()

    # Assuming the first scene's chessboard pose as the common frame reference
    # Alternatively, one could define an arbitrary world frame and transform all to it.

    # Transform each cloud to chessboard frame and merge
    for i, scene_name in enumerate(scenes):
        if scene_name in scene_to_chess_T:
            cloud = all_clouds[i]
            cloud_gt = all_gt_clouds[i]
            T = scene_to_chess_T[scene_name]

            # Transform predicted and GT clouds to chessboard frame
            # Scale the transformation matrix by SQUARE_SIZE
            T_scaled = T.copy()
            T_scaled[:3, 3] *= SQUARE_SIZE  # Scale the translation component
            
            cloud_chess = cloud.transform(np.linalg.inv(T_scaled))
            cloud_gt_chess = cloud_gt.transform(np.linalg.inv(T_scaled))

            merged_chess_cloud += cloud_chess
            merged_chess_gt_cloud += cloud_gt_chess

    # Downsample and clean merged clouds
    # Scale voxel size by SQUARE_SIZE to maintain relative scale
    voxel_size = 0.005 * SQUARE_SIZE
    merged_chess_cloud = merged_chess_cloud.voxel_down_sample(voxel_size=voxel_size)
    merged_chess_cloud, _ = merged_chess_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    merged_chess_gt_cloud = merged_chess_gt_cloud.voxel_down_sample(voxel_size=voxel_size)
    merged_chess_gt_cloud, _ = merged_chess_gt_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # --- ICP Refinement of Merged Cloud ---
    print("\nRefining merged cloud alignment with ICP...")
    # Downsample for faster ICP
    merged_cloud_down = merged_chess_cloud.voxel_down_sample(voxel_size=voxel_size * 2)
    # Estimate normals (needed for point-to-plane ICP)
    merged_cloud_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30))

    # Perform ICP (aligning against itself after downsampling/cleaning helps internal consistency)
    # Scale ICP distance threshold by SQUARE_SIZE
    icp_distance_threshold = 0.02 * SQUARE_SIZE  # meters (adjust as needed)
    icp_result = o3d.pipelines.registration.registration_icp(
        merged_cloud_down, merged_cloud_down, icp_distance_threshold,
        np.eye(4), # Initial transformation is identity
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

    # Apply the ICP transformation to both the full resolution merged clouds
    merged_chess_cloud.transform(icp_result.transformation)
    merged_chess_gt_cloud.transform(icp_result.transformation)
    print(f"ICP refinement applied. Fitness: {icp_result.fitness:.4f}, RMSE: {icp_result.inlier_rmse:.4f}")

    # Save merged chessboard-frame cloud
    o3d.io.write_point_cloud("dataset/chess/merged_chess_cloud.ply", merged_chess_cloud)
    o3d.io.write_point_cloud("dataset/chess/merged_chess_gt_cloud.ply", merged_chess_gt_cloud)

    # 6. Compute cloud-to-cloud metrics
    print("\n6. Computing cloud-to-cloud metrics...")
    cloud_metrics = compute_cloud_metrics(merged_chess_cloud, merged_chess_gt_cloud)
    print("\nCloud-to-cloud metrics:")
    print(f"Chamfer Distance: {cloud_metrics['chamfer_dist']:.6f} m")
    print(f"Accuracy: {cloud_metrics['accuracy']:.4f}")
    print(f"Completeness: {cloud_metrics['completeness']:.4f}")
    print(f"F1 Score: {cloud_metrics['f1_score']:.4f}")

    # # 7. Generate meshes
    # print("\n7. Generating meshes...")
    # # Prepare point clouds for meshing
    # merged_chess_cloud_mesh = prepare_for_meshing(merged_chess_cloud)
    # merged_chess_gt_cloud_mesh = prepare_for_meshing(merged_chess_gt_cloud)
    #
    # # Create mesh
    # mesh = create_mesh(merged_chess_cloud_mesh, depth=9, trim=0.1)
    # mesh_gt = create_mesh(merged_chess_gt_cloud_mesh, depth=9, trim=0.1)
    #
    # if mesh is not None:
    #     # Create camera parameters for texture mapping
    #     K_dict = {scene: K0 for scene in scenes}  # Use K0 for all views
    #     T_wc_dict = {scene: np.eye(4) for scene in scenes}  # Identity transforms
    #
    #     # Add texture to mesh
    #     textured_mesh = create_textured_mesh(mesh, all_images, K_dict, T_wc_dict, Path("dataset/chess"))
    #
    #     # Save mesh
    #     mesh_path = Path("dataset/chess/merged_chess_mesh.ply")
    #     o3d.io.write_triangle_mesh(str(mesh_path), textured_mesh)
    #
    #     # Compute mesh-to-mesh metrics
    #     print("\n8. Computing mesh-to-mesh metrics...")
    #     # Convert mesh to point cloud for comparison
    #     mesh_cloud = mesh.sample_points_uniformly(number_of_points=100000)
    #     mesh_gt_cloud = mesh_gt.sample_points_uniformly(number_of_points=100000)
    #
    #     mesh_metrics = compute_cloud_metrics(mesh_cloud, mesh_gt_cloud)
    #     print("\nMesh-to-mesh metrics:")
    #     print(f"Chamfer Distance: {mesh_metrics['chamfer_dist']:.6f} m")
    #     print(f"Accuracy: {mesh_metrics['accuracy']:.4f}")
    #     print(f"Completeness: {mesh_metrics['completeness']:.4f}")
    #     print(f"F1 Score: {mesh_metrics['f1_score']:.4f}")

    # 9. Print summary metrics
    print("\n9. Summary metrics:")
    print("Scene\tMAE\tRMS\tBad1%\tBad2%\tBad4%")
    print("-" * 50)
    for metrics in all_metrics:
        print(f"{metrics['scene']}\t{metrics['mae']:.3f}\t{metrics['rms']:.3f}\t{metrics['bad1']:.2f}\t{metrics['bad2']:.2f}\t{metrics['bad4']:.2f}")
    
    # Compute averages
    avg_metrics = {
        'mae': np.mean([m['mae'] for m in all_metrics]),
        'rms': np.mean([m['rms'] for m in all_metrics]),
        'bad1': np.mean([m['bad1'] for m in all_metrics]),
        'bad2': np.mean([m['bad2'] for m in all_metrics]),
        'bad4': np.mean([m['bad4'] for m in all_metrics])
    }
    print("-" * 50)
    print(f"Average\t{avg_metrics['mae']:.3f}\t{avg_metrics['rms']:.3f}\t{avg_metrics['bad1']:.2f}\t{avg_metrics['bad2']:.2f}\t{avg_metrics['bad4']:.2f}")

    # Print summary of pair-wise metrics
    print("\nSummary of Cloud-to-Cloud Metrics for Each Pair:")
    print("-" * 80)
    print(f"{'Scene':<10} {'Chamfer (m)':<12} {'Accuracy':<10} {'Completeness':<12} {'F1 Score':<10}")
    print("-" * 80)
    
    for metrics in pair_metrics:
        print(f"{metrics['scene']:<10} {metrics['chamfer_dist']:.6f} {metrics['accuracy']:.4f} {metrics['completeness']:.4f} {metrics['f1_score']:.4f}")
    
    # Compute averages
    avg_metrics = {
        'chamfer_dist': np.mean([m['chamfer_dist'] for m in pair_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in pair_metrics]),
        'completeness': np.mean([m['completeness'] for m in pair_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in pair_metrics])
    }
    
    print("-" * 80)
    print(f"{'Average':<10} {avg_metrics['chamfer_dist']:.6f} {avg_metrics['accuracy']:.4f} {avg_metrics['completeness']:.4f} {avg_metrics['f1_score']:.4f}")
