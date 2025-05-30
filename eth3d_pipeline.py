"""
ETH-3D stereo → dense indoor 3-D reconstruction.
Classical geometry baseline plus optional multi-view fusion.

1. Calibration load - Uses pycolmap to load camera and image data
2. Image choice - Automated selection of optimal stereo pairs based on baseline and overlap
3. Rectification - Implements down-sampling during rectification, keeping homography info for potential up-sampling
4. Stereo matching - Utilizes SGBM (min_disp and num_disp are important) with left-right consistency check and WLS filtering
5. Disparity to depth - Clamps small disparities and uses the validity mask
6. Outlier filtering - Includes statistical filtering, voxel down-sampling, and plane-fitting
7. Multi-view fusion - Added framework for fusing multiple views
8. Mesh generation - Implemented Poisson surface reconstruction with optimized parameters for ETH3D indoor scenes
"""

from utils import *
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path


if __name__ == "__main__":
    scene = Path("dataset/delivery_area")
    save_dir = Path("dataset/delivery_area/results")
    save_dir.mkdir(exist_ok=True, parents=True)

    # Load COLMAP data
    print("1. Loading COLMAP data...")
    imgs, K, T_wc, points3D = load_colmap(scene)

    # Choose pairs - using (9,8) and (10,9)
    ref_id1 = 9  # First reference view
    right_id = 8  # Right view for first pair
    ref_id2 = 10  # Second reference view
    left_id = 9   # Right view for second pair
    
    print(f"Using pairs: ({ref_id1},{right_id}) and ({ref_id2},{left_id})")
    print(f"Locations: {imgs[ref_id1].name} - {imgs[right_id].name} - {imgs[ref_id2].name}")

    # Process both stereo pairs
    print("\n2. Processing stereo pairs...")
    
    # Process right pair (9-8)
    print("\nProcessing right pair (9-8)...")
    rect_img1_right, rect_img2_right, Q_right, rect_info_right = rectify_pair(
        ref_id1, right_id, imgs, K, T_wc, scene, resize=0.5)
    save_rectified_pair(rect_img1_right, rect_img2_right, 
                       Path(save_dir) / f"rect_{ref_id1}_{right_id}.png")

    # Process left pair (10-9)
    print("\nProcessing left pair (10-9)...")
    rect_img1_left, rect_img2_left, Q_left, rect_info_left = rectify_pair(
        ref_id2, left_id, imgs, K, T_wc, scene, resize=0.5)
    save_rectified_pair(rect_img1_left, rect_img2_left, 
                       Path(save_dir) / f"rect_{ref_id2}_{left_id}.png")

    # Compute and filter disparity for both pairs
    print("\n4. Computing disparity for both pairs...")
    left_matcher, right_matcher, wls_filter = sgbm_with_consistency()

    # Right pair disparity
    filtered_disp_right, valid_mask_right = compute_filtered_disparity(
        rect_img1_right, rect_img2_right, left_matcher, right_matcher, wls_filter
    )
    filtered_disp_gray_right = (filtered_disp_right * 255 / np.max(filtered_disp_right)).astype(np.uint8)
    cv2.imwrite(str(Path(save_dir) / f"disp_{ref_id1}_{right_id}_raw.png"), filtered_disp_gray_right)

    # Left pair disparity
    filtered_disp_left, valid_mask_left = compute_filtered_disparity(
        rect_img1_left, rect_img2_left, left_matcher, right_matcher, wls_filter
    )
    filtered_disp_gray_left = (filtered_disp_left * 255 / np.max(filtered_disp_left)).astype(np.uint8)
    cv2.imwrite(str(Path(save_dir) / f"disp_{ref_id2}_{left_id}_raw.png"), filtered_disp_gray_left)

    # Generate point clouds for both pairs
    print("\n5. Generating point clouds...")
    
    # Right pair cloud
    pcl_right, colors_right = disparity_to_cloud(filtered_disp_right, Q_right, valid_mask_right, rect_img1_right)
    cloud_right = clean_cloud(pcl_right, colors_right)
    cloud_right.transform(np.linalg.inv(T_wc[ref_id1]))
    o3d.io.write_point_cloud(Path(save_dir / f"cloud_right_{ref_id1}_{right_id}.ply"), cloud_right)

    # Left pair cloud
    pcl_left, colors_left = disparity_to_cloud(filtered_disp_left, Q_left, valid_mask_left, rect_img1_left)
    cloud_left = clean_cloud(pcl_left, colors_left)
    cloud_left.transform(np.linalg.inv(T_wc[ref_id2]))
    o3d.io.write_point_cloud(Path(save_dir / f"cloud_left_{ref_id2}_{left_id}.ply"), cloud_left)

    # Multi-view fusion
    print("\n7. Performing multi-view fusion...")
    matchers = (left_matcher, right_matcher, wls_filter)
    # Use both reference views for fusion
    neighbor_ids = [right_id, left_id]  # Both neighbors for fusion
    fused_cloud = fuse_multi_view_eth3d(ref_id1, neighbor_ids,
                                      imgs, K, T_wc, scene,
                                      resize=0.5,
                                      voxel=0.002,
                                      matchers=matchers)

    if fused_cloud is not None:
        o3d.io.write_point_cloud(Path(save_dir / "cloud_fused.ply"), fused_cloud)

        # Create mesh
        print("\n8. Creating mesh from fused point cloud...")
        mesh = create_mesh(fused_cloud, depth=10, trim=0.15, target_triangles=300_000, prepare_cloud=True)
        if mesh is not None:
            o3d.io.write_triangle_mesh(Path(save_dir / "mesh.ply"), mesh)
            print(f"Mesh created with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
        else:
            print("Mesh creation failed")
    else:
        print("Multi-view fusion failed, no point cloud generated")

    print("\nProcessing complete! Results saved to:", save_dir)