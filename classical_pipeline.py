"""
ETH-3D stereo → dense indoor 3-D reconstruction.
Classical geometry baseline plus optional multi-view fusion.

1. Calibration load - Uses pycolmap to load camera and image data
2. Image choice - Automated selection of optimal stereo pairs based on baseline and overlap
3. Rectification - Implements down-sampling during rectification, keeping homography info for potential up-sampling
4. Stereo matching - Utilizes SGBM with left-right consistency check and WLS filtering
5. Disparity to depth - Clamps small disparities and uses the validity mask
6. Outlier filtering - Includes statistical filtering, voxel down-sampling, and plane-fitting
7. Multi-view fusion - Added framework for fusing multiple views
8. Meshing - Implemented Poisson surface reconstruction with Laplacian smoothing
"""

from pathlib import Path
import open3d as o3d
from utils import *
import numpy as np


if __name__ == "__main__":
    scene = Path("dataset/delivery_area")
    save_dir = Path("dataset/results")

    # Load COLMAP data
    print("1. Loading COLMAP data...")
    imgs, K, T_wc, points3D = load_colmap(scene)

    # Compute baselines and find best pairs
    print("2. Finding optimal stereo pairs...")
    B = compute_baselines(imgs)
    top_pairs = best_pairs(imgs, B, k=3, min_B=0.05, max_B=0.9, min_overlap=0.6)

    if top_pairs:
        print(f"Top pairs: {top_pairs}")
    else:
        print("No suitable stereo pairs found. Check baseline/overlap criteria.")
        exit(1)

    # Choose primary pair
    ref_id, second_id = top_pairs[0]
    print(f"Primary stereo pair: {ref_id} - {second_id}")
    print(f"Locations: {imgs[ref_id].name} - {imgs[second_id].name}")

    # Find neighboring views for fusion
    neighbor_ids = find_neighbor_views(imgs, B, ref_id, k=10, min_B=0.05, max_B=2.0, min_overlap=0.3)
    print(f"Neighboring views for fusion: {neighbor_ids}")

    # 1: Process single pair
    print("3. Processing primary stereo pair...")
    rect_img1, rect_img2, Q, rect_info = rectify_pair(ref_id, second_id, imgs, K, T_wc, scene, resize=0.5)

    # Compute and filter disparity
    print("4. Computing disparity...")
    left_matcher, right_matcher, wls_filter = sgbm_with_consistency()
    filtered_disp, valid_mask = compute_filtered_disparity(
        rect_img1, rect_img2, left_matcher, right_matcher, wls_filter
    )

    # Generate point cloud
    print("5. Generating depth from disparity...")
    pcl, colors = disparity_to_cloud(filtered_disp, Q, valid_mask, rect_img1)
    print("6. Filtering point cloud...")
    cloud = clean_cloud(pcl, colors)

    # convert ref-camera cloud to world coordinate system
    cloud.transform(np.linalg.inv(T_wc[ref_id]))
    o3d.io.write_point_cloud(Path(save_dir) / "cloud_single_pair.ply", cloud)

    # 2: Multi-view fusion
    print("7. Performing multi-view fusion...")
    matchers = sgbm_with_consistency()
    fused_cloud = fuse_multi_view(ref_id, neighbor_ids,
                                  imgs, K, T_wc, scene,
                                  resize=0.5,
                                  matchers=matchers)

    if fused_cloud is not None:
        fused_cloud.transform(np.linalg.inv(T_wc[ref_id]))  # world frame safeguard
        o3d.io.write_point_cloud(Path(save_dir) / "cloud_fused.ply", fused_cloud)

        # Create mesh
        # print("8. Creating mesh...")
        # mesh = create_mesh(fused_cloud, depth=11, target_triangles=300_000)
        # o3d.io.write_triangle_mesh(Path(save_dir) / "mesh.ply", mesh)

    print("Processing complete!")
