from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from utils import (load_middlebury_calib, sgbm_with_consistency, compute_filtered_disparity,
                   disparity_to_cloud, create_gt_cloud, clean_cloud,
                   compute_metrics, read_pfm_file, compute_cloud_metrics,
                   prepare_for_meshing, create_mesh, create_textured_mesh)
from utils.visualization import save_rectified_pair

# --- Chessboard pose estimation parameters ---
CHESSBOARD_SIZE = (7, 7)  # (corners_per_row, corners_per_col)
SQUARE_SIZE = 0.01  # meters (adjust to your chessboard)

# Prepare 3D model points for the chessboard (z=0 plane)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # Scale the object points by SQUARE_SIZE

# Store transformations for each scene
scene_to_chess_T = {}


def process_scene(scene_name, scene_path):
    """Process a single chess scene"""
    print(f"\n{'=' * 20} PROCESSING {scene_name} {'=' * 20}")

    # 1. Load calibration and images
    print("\n[1] Loading data and calibration...")
    K0, K1, doffs, baseline, w, h, ndisp, vmin, vmax = load_middlebury_calib(scene_path / "calib.txt")
    imgL = cv2.imread(str(scene_path / "im0.png"))
    imgR = cv2.imread(str(scene_path / "im1.png"))

    baseline_m = baseline / 1000.0  # mm → m
    f = K0[0, 0]  # focal length in px
    cx = K0[0, 2]
    cy = K0[1, 2]
    print(f"Baseline: {baseline_m} m, Focal length: {f} px, Center: ({cx}, {cy}) px")

    Q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, f],
        [0, 0, 1 / baseline_m, -doffs / baseline_m]
    ], dtype=np.float32)

    # Save rectified pair
    save_path = scene_path / "rectified_pair.png"
    save_rectified_pair(imgL, imgR, save_path)
    print(f"Saved rectified image pair to {save_path}")

    # 2. Compute disparity
    print("\n[2] Computing disparity map...")
    min_disp = vmin - doffs
    raw_span = vmax - min_disp
    num_disp = ((raw_span + 15) // 16) * 16
    left_matcher, right_matcher, wls_filter = sgbm_with_consistency(min_disp=int(min_disp), num_disp=int(num_disp))
    filtered_disp, valid_mask = compute_filtered_disparity(
        imgL, imgR, left_matcher, right_matcher, wls_filter
    )

    # Save disparity map
    filtered_disp_gray = (filtered_disp * 255 / np.max(filtered_disp)).astype(np.uint8)
    disparity_map_path = scene_path / "disp_raw.png"
    cv2.imwrite(str(disparity_map_path), filtered_disp_gray)

    # 3. Generate point clouds
    print("\n[3] Generating point clouds...")
    # Predicted cloud
    pts, colors = disparity_to_cloud(filtered_disp, Q, valid_mask, imgL)
    cloud = clean_cloud(pts, colors)

    # Ground truth cloud
    disp_gt, _ = read_pfm_file(scene_path / "disp0.pfm")
    cloud_gt = create_gt_cloud(disp_gt, Q, valid_mask, imgL)

    # Save individual clouds
    cloud_path = scene_path / "cloud.ply"
    cloud_gt_path = scene_path / "cloud_gt.ply"
    o3d.io.write_point_cloud(str(cloud_path), cloud)
    o3d.io.write_point_cloud(str(cloud_gt_path), cloud_gt)

    # Compute cloud-to-cloud metrics
    print("\n[4] Computing point cloud metrics...")
    cloud_metrics = compute_cloud_metrics(cloud, cloud_gt)
    metrics_data = {
        'scene': scene_name,
        'chamfer_dist': cloud_metrics['chamfer_dist'],
        'accuracy': cloud_metrics['accuracy'],
        'completeness': cloud_metrics['completeness'],
        'f1_score': cloud_metrics['f1_score']
    }

    print(f"Chamfer Distance: {cloud_metrics['chamfer_dist']:.6f} m")
    print(f"Accuracy: {cloud_metrics['accuracy']:.4f}")
    print(f"Completeness: {cloud_metrics['completeness']:.4f}")
    print(f"F1 Score: {cloud_metrics['f1_score']:.4f}")

    # 5. Evaluate disparity
    print("\n[5] Evaluating disparity metrics...")
    mae, rms, bad1, bad2, bad4 = compute_metrics(filtered_disp, disp_gt, valid_mask)
    disparity_metrics = {
        'scene': scene_name,
        'mae': mae,
        'rms': rms,
        'bad1': bad1,
        'bad2': bad2,
        'bad4': bad4
    }

    print(f"MAE: {mae:.3f}, RMS: {rms:.3f}")
    print(f"Bad1%: {bad1:.2f}, Bad2%: {bad2:.2f}, Bad4%: {bad4:.2f}")

    # 6. Detect chessboard corners and estimate pose
    print("\n[6] Detecting chessboard and estimating pose...")
    gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

    # Try to load manual corners or detect them
    manual_corners_path = scene_path / "corners_manual.npy"
    if manual_corners_path.exists():
        print("Loading manual corners...")
        corners_2d_all = np.load(str(manual_corners_path))
        if corners_2d_all.shape[0] != CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1]:
            print(
                f"[ERROR] Manual corners file should contain {CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1]} points, but found {corners_2d_all.shape[0]}.")
            ret = False
        else:
            ret = True
            corners_2d = corners_2d_all
    else:
        print("Attempting automatic corner detection...")
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners_2d = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags)

    # Save visualization of detected corners
    vis = imgL.copy()
    if ret and corners_2d is not None:
        cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners_2d, ret)
    cv2.imwrite(str(scene_path / "corners_vis.png"), vis)

    if not ret or corners_2d is None or len(corners_2d) != CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1]:
        print("[ERROR] Valid chessboard corners not available for pose estimation")
        return None, None, None, disparity_metrics, metrics_data

    # Reproject 2D corners to 3D using disparity
    corners_2d_reproj = corners_2d.reshape(-1, 1, 2).astype(np.float32)

    # Get coordinates and ensure they're within bounds
    corner_indices_y = np.round(corners_2d_reproj[:, 0, 1]).astype(int)
    corner_indices_x = np.round(corners_2d_reproj[:, 0, 0]).astype(int)

    h, w = filtered_disp.shape[:2]
    corner_indices_y = np.clip(corner_indices_y, 0, h - 1)
    corner_indices_x = np.clip(corner_indices_x, 0, w - 1)

    points_3d_all = cv2.reprojectImageTo3D(filtered_disp, Q)[corner_indices_y, corner_indices_x]

    # Filter out invalid points
    valid_mask_3d = np.isfinite(points_3d_all).all(axis=1)
    corners_3d = points_3d_all[valid_mask_3d]
    objp_valid = objp[valid_mask_3d]
    corners_2d_valid = corners_2d[valid_mask_3d]

    # Estimate pose using solvePnP
    chessboard_transform = None
    if len(corners_3d) >= 6:
        camera_matrix = K0
        dist_coeffs = np.zeros(5)  # Assume no distortion
        success, rvec, tvec = cv2.solvePnP(objp_valid, corners_2d_valid, camera_matrix, dist_coeffs,
                                           flags=cv2.SOLVEPNP_ITERATIVE)

        if success:
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.ravel()
            chessboard_transform = T

            print("Successfully estimated chessboard pose:")
            print(T)
        else:
            print("[ERROR] solvePnP failed")
    else:
        print(f"[ERROR] Not enough valid corners for pose estimation (found {len(corners_3d)} valid points)")

    return K0, imgL, cloud, cloud_gt, disparity_metrics, metrics_data, chessboard_transform


def print_summary_metrics(all_metrics, pair_metrics, cloud_metrics):
    """Print summary of all metrics"""
    print("\n[9] Summary metrics:")

    print("\nDisparity Metrics by Scene:")
    print("=" * 50)
    print("Scene\tMAE\tRMS\tBad1%\tBad2%\tBad4%")
    print("-" * 50)
    for metrics in all_metrics:
        print(
            f"{metrics['scene']}\t{metrics['mae']:.3f}\t{metrics['rms']:.3f}\t{metrics['bad1']:.2f}\t{metrics['bad2']:.2f}\t{metrics['bad4']:.2f}")

    # Compute averages
    avg_metrics = {
        'mae': np.mean([m['mae'] for m in all_metrics]),
        'rms': np.mean([m['rms'] for m in all_metrics]),
        'bad1': np.mean([m['bad1'] for m in all_metrics]),
        'bad2': np.mean([m['bad2'] for m in all_metrics]),
        'bad4': np.mean([m['bad4'] for m in all_metrics])
    }
    print("-" * 50)
    print(
        f"Average\t{avg_metrics['mae']:.3f}\t{avg_metrics['rms']:.3f}\t{avg_metrics['bad1']:.2f}\t{avg_metrics['bad2']:.2f}\t{avg_metrics['bad4']:.2f}")

    # Print summary of pair-wise metrics
    print("\nCloud-to-Cloud Metrics by Scene:")
    print("=" * 80)
    print(f"{'Scene':<10} {'Chamfer (m)':<12} {'Accuracy':<10} {'Completeness':<12} {'F1 Score':<10}")
    print("-" * 80)

    for metrics in pair_metrics:
        print(
            f"{metrics['scene']:<10} {metrics['chamfer_dist']:.6f} {metrics['accuracy']:.4f} {metrics['completeness']:.4f} {metrics['f1_score']:.4f}")

    # Compute averages
    avg_metrics = {
        'chamfer_dist': np.mean([m['chamfer_dist'] for m in pair_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in pair_metrics]),
        'completeness': np.mean([m['completeness'] for m in pair_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in pair_metrics])
    }

    print("-" * 80)
    print(
        f"{'Average':<10} {avg_metrics['chamfer_dist']:.6f} {avg_metrics['accuracy']:.4f} {avg_metrics['completeness']:.4f} {avg_metrics['f1_score']:.4f}")

    # Print merged cloud metrics
    print("\nMerged Cloud Metrics:")
    print("=" * 80)
    print(f"{'Metric':<15} {'Value':<10}")
    print("-" * 80)
    print(f"{'Chamfer (m)':<15} {cloud_metrics['chamfer_dist']:.6f}")
    print(f"{'Accuracy':<15} {cloud_metrics['accuracy']:.4f}")
    print(f"{'Completeness':<15} {cloud_metrics['completeness']:.4f}")
    print(f"{'F1 Score':<15} {cloud_metrics['f1_score']:.4f}")

    # Visualize distance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, density=True)
    plt.title('Distribution of Point-to-Point Distances among ground_truth and estimated merged clouds')
    plt.xlabel('Distance (m)')
    plt.ylabel('Density')
    plt.savefig(str(out_dir / 'distance_distribution.png'))
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("CHESS SCENES STEREO RECONSTRUCTION PIPELINE")
    print("=" * 50)

    # Process all three chess scenes
    scenes = ["chess_1", "chess_2", "chess_3"]
    all_clouds = []
    all_gt_clouds = []
    all_metrics = []
    all_images = {}  # Store images for texture mapping
    pair_metrics = []  # Store metrics for each pair

    # Reference points for alignment
    reference_point = np.array([0.23, 0.59, -0.28])
    point2 = np.array([0.56, 0.4, -0.28])
    point3 = np.array([0.74, 0.17, -0.28])

    # Calculate translation vectors
    translation2 = reference_point - point2
    translation3 = reference_point - point3

    # Process each scene
    for scene_name in scenes:
        scene_path = Path(f"dataset/chess/{scene_name}")
        K0, img, cloud, cloud_gt, disp_metrics, cloud_metrics, transform = process_scene(scene_name, scene_path)

        if img is not None:
            all_images[scene_name] = img
        if cloud is not None and cloud_gt is not None:
            all_clouds.append(cloud)
            all_gt_clouds.append(cloud_gt)
        if disp_metrics is not None:
            all_metrics.append(disp_metrics)
        if cloud_metrics is not None:
            pair_metrics.append(cloud_metrics)
        if transform is not None:
            scene_to_chess_T[scene_name] = transform

    # Merge point clouds
    print("\n[7] Merging and aligning point clouds...")
    merged_chess_cloud = o3d.geometry.PointCloud()
    merged_chess_gt_cloud = o3d.geometry.PointCloud()
    transformed_clouds = []

    # Transform and merge clouds
    for i, scene_name in enumerate(scenes):
        if scene_name in scene_to_chess_T:
            cloud = all_clouds[i]
            cloud_gt = all_gt_clouds[i]
            T = np.copy(scene_to_chess_T[scene_name])

            # Apply appropriate transformation based on scene index
            translation_matrix = np.eye(4)
            if i == 1:
                translation_matrix[:3, 3] = translation2
            elif i == 2:
                translation_matrix[:3, 3] = translation3

            new_T = np.matmul(translation_matrix, np.linalg.inv(T))
            print(f"Transforming {scene_name} with T_old:\n{np.array2string(T, precision=5, suppress_small=True)}")
            print(
                f"Transforming {scene_name} with T:\n{np.array2string(np.linalg.inv(new_T), precision=5, suppress_small=True)}")
            cloud_chess = cloud.transform(new_T)
            cloud_gt_chess = cloud_gt.transform(new_T)

            transformed_clouds.append((scene_name, cloud_chess))
            merged_chess_cloud += cloud_chess
            merged_chess_gt_cloud += cloud_gt_chess

    # Check correlation between transformed clouds
    print("\n[8] Checking correlation between transformed clouds...")
    voxel_size = 0.005 * SQUARE_SIZE
    for i in range(len(transformed_clouds)):
        for j in range(i + 1, len(transformed_clouds)):
            scene1, cloud1 = transformed_clouds[i]
            scene2, cloud2 = transformed_clouds[j]

            # Compute point-to-point distances
            pcd1 = cloud1.voxel_down_sample(voxel_size=voxel_size)
            pcd2 = cloud2.voxel_down_sample(voxel_size=voxel_size)

            distances = pcd1.compute_point_cloud_distance(pcd2)
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)

            print(f"\nCorrelation between {scene1} and {scene2}:")
            print(f"Mean distance: {mean_dist:.6f} m")
            print(f"Std distance: {std_dist:.6f} m")

            # Visualize the two clouds
            output_path = Path(f"dataset/chess/correlation_{scene1}_{scene2}.png")

    # Clean and downsample merged clouds
    print("\n[9] Cleaning and refining merged clouds...")
    voxel_size = 0.005 * SQUARE_SIZE
    merged_chess_cloud = merged_chess_cloud.voxel_down_sample(voxel_size=voxel_size)
    merged_chess_cloud, _ = merged_chess_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    merged_chess_gt_cloud = merged_chess_gt_cloud.voxel_down_sample(voxel_size=voxel_size)
    merged_chess_gt_cloud, _ = merged_chess_gt_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # ICP Refinement
    print("Refining merged cloud alignment with ICP...")
    merged_cloud_down = merged_chess_cloud.voxel_down_sample(voxel_size=voxel_size * 2)
    merged_cloud_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30))

    icp_distance_threshold = 0.02 * SQUARE_SIZE
    icp_result = o3d.pipelines.registration.registration_icp(
        merged_cloud_down, merged_cloud_down, icp_distance_threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

    # Apply the ICP transformation
    merged_chess_cloud.transform(icp_result.transformation)
    merged_chess_gt_cloud.transform(icp_result.transformation)
    print(f"ICP refinement applied. Fitness: {icp_result.fitness:.4f}, RMSE: {icp_result.inlier_rmse:.4f}")

    # Save merged clouds
    out_dir = Path("dataset/chess")
    o3d.io.write_point_cloud(str(out_dir / "merged_chess_cloud.ply"), merged_chess_cloud)
    o3d.io.write_point_cloud(str(out_dir / "merged_chess_gt_cloud.ply"), merged_chess_gt_cloud)

    # Compute detailed metrics between merged clouds
    print("\n[10] Computing detailed metrics for merged clouds...")
    distances = np.asarray(merged_chess_cloud.compute_point_cloud_distance(merged_chess_gt_cloud))
    gt_distances = np.asarray(merged_chess_gt_cloud.compute_point_cloud_distance(merged_chess_cloud))

    # Calculate metrics
    threshold = 0.03  # 2cm threshold for accuracy/completeness
    accuracy = np.mean(distances < threshold)
    completeness = np.mean(gt_distances < threshold)
    f1_score = 2 * (accuracy * completeness) / (accuracy + completeness) if (accuracy + completeness) > 0 else 0
    chamfer_dist = (np.mean(distances) + np.mean(gt_distances)) / 2

    cloud_metrics = {
        'chamfer_dist': chamfer_dist,
        'accuracy': accuracy,
        'completeness': completeness,
        'f1_score': f1_score
    }

    # Visualize distance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, density=True)
    plt.title('Distribution of Point-to-Point Distances')
    plt.xlabel('Distance (m)')
    plt.ylabel('Density')
    plt.savefig(str(out_dir / 'distance_distribution.png'))
    plt.close()

    # Generate meshes
    print("\n[11] Generating meshes...")
    mesh_button = False
    if mesh_button:
        merged_chess_cloud_mesh = prepare_for_meshing(merged_chess_cloud)
        merged_chess_gt_cloud_mesh = prepare_for_meshing(merged_chess_gt_cloud)

        mesh = create_mesh(merged_chess_cloud_mesh, depth=9, trim=0.1)
        mesh_gt = create_mesh(merged_chess_gt_cloud_mesh, depth=9, trim=0.1)

        if mesh is not None:
            # Create camera parameters for texture mapping
            K_dict = {scene: K0 for scene in scenes}  # Use K0 for all views
            T_wc_dict = {scene: np.eye(4) for scene in scenes}  # Identity transforms

            # Add texture to mesh
            textured_mesh = create_textured_mesh(mesh, all_images, K_dict, T_wc_dict, out_dir)

            # Save mesh
            mesh_path = out_dir / "merged_chess_mesh.ply"
            o3d.io.write_triangle_mesh(str(mesh_path), textured_mesh)

            # Compute mesh-to-mesh metrics
            print("\n[12] Computing mesh-to-mesh metrics...")
            mesh_cloud = mesh.sample_points_uniformly(number_of_points=100000)
            mesh_gt_cloud = mesh_gt.sample_points_uniformly(number_of_points=100000)

            mesh_metrics = compute_cloud_metrics(mesh_cloud, mesh_gt_cloud)
            print("\nMesh-to-mesh metrics:")
            print(f"Chamfer Distance: {mesh_metrics['chamfer_dist']:.6f} m")
            print(f"Accuracy: {mesh_metrics['accuracy']:.4f}")
            print(f"Completeness: {mesh_metrics['completeness']:.4f}")
            print(f"F1 Score: {mesh_metrics['f1_score']:.4f}")

    # Print summary metrics
    print_summary_metrics(all_metrics, pair_metrics, cloud_metrics)

    print("\nPipeline completed successfully!")