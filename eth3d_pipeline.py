"""
ETH-3D stereo → dense indoor 3-D reconstruction.
Classical geometry baseline plus optional multi-view fusion.

1. Calibration load - Uses pycolmap to load camera and image data
2. Image choice - Automated selection of optimal stereo pairs based on baseline and overlap
3. Rectification - Implements down-sampling during rectification, keeping homography info for potential up-sampling

Find texture
adaptive normalization (increase contrast in pixels with low texture)

4. Stereo matching - Utilizes SGBM with left-right consistency check and WLS filtering
min_disp and num_disp important

oculusion
selective filtering
insanity checking

The disparity map is bad here (0 -> z infinity, but we remove it later, so there're many holes in 3D point cloud)
interpolation?

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
    scene = Path("dataset/office")
    save_dir = Path("dataset/office/results")
    save_dir.mkdir(exist_ok=True, parents=True)

    # Load COLMAP data
    print("1. Loading COLMAP data...")
    imgs, K, T_wc, points3D = load_colmap(scene)
    B = compute_baselines(imgs, T_wc)

    # Choose primary pair
    ref_id, second_id = (22, 21)
    pair = (ref_id, second_id)
    print(f"Primary stereo pair: {ref_id} - {second_id}")
    print(f"Locations: {imgs[ref_id].name} - {imgs[second_id].name}")

    # Find neighboring views for fusion
    neighbor_ids = find_neighbor_views(imgs, B, ref_id, k=10, min_B=0.05, max_B=2.0, min_overlap=0.3)
    print(f"Neighboring views for fusion: {neighbor_ids}")

    # 1: Process single pair
    print("2. Processing primary stereo pair...")
    rect_img1, rect_img2, Q, rect_info = rectify_pair(ref_id, second_id, imgs, K, T_wc, scene, resize=0.5)

    # Save rectified images
    h = min(rect_img1.shape[0], rect_img2.shape[0])
    img1 = rect_img1[:h]
    img2 = rect_img2[:h]

    # Draw horizontal lines
    combined = np.hstack((img1, img2))
    for y in range(0, h, 150):
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 2)
    # Save the combined image
    combined_image_path = Path(save_dir) / f"rect_{ref_id}_{second_id}.png"
    cv2.imwrite(str(combined_image_path), combined)

    # Compute vmin/vmax from sparse COLMAP depths
    # print("3. Computing depth range from COLMAP points...")
    # baseline = B.get(pair, B.get((second_id, ref_id)))
    # print(f"Baseline for pair {pair}: {baseline}")
    #
    # vmin, vmax = compute_depth_range_from_colmap(points3D, imgs, K, T_wc, ref_id, second_id, baseline)
    # print(f"Depth range from COLMAP: vmin={vmin}, vmax={vmax}")

    # Compute and filter disparity
    print("4. Computing disparity...")
    # min_disp = int(np.floor(vmin))
    # raw_span = vmax - min_disp
    # num_disp = ((raw_span + 15) // 16) * 16

    left_matcher, right_matcher, wls_filter = sgbm_with_consistency()
    filtered_disp, valid_mask = compute_filtered_disparity(
        rect_img1, rect_img2, left_matcher, right_matcher, wls_filter
    )

    # Save raw disparity map
    filtered_disp_gray = (filtered_disp * 255 / np.max(filtered_disp)).astype(np.uint8)
    disparity_map_path = Path(save_dir) / f"disp_{ref_id}_{second_id}_raw.png"
    cv2.imwrite(str(disparity_map_path), filtered_disp_gray)

    # Generate point cloud
    print("5. Generating depth from disparity...")
    print(f"Valid mask sum: {np.sum(valid_mask)}")
    print(f"Filtered disparity range: {filtered_disp[filtered_disp > 0].min()} to {filtered_disp.max()}")
    print(f"Q matrix structure: Q[2,3]={Q[2, 3]}, Q[3,2]={Q[3, 2]}")

    # Modified version that adapts to your actual depth range
    pcl, colors = disparity_to_cloud(filtered_disp, Q, valid_mask, rect_img1)

    print("6. Filtering point cloud...")
    cloud = clean_cloud(pcl, colors)

    # convert ref-camera cloud to world coordinate system
    cloud.transform(np.linalg.inv(T_wc[ref_id]))
    o3d.io.write_point_cloud(Path(save_dir / f"cloud_single_pair_{ref_id}_{second_id}.ply"), cloud)

    # # 2: Multi-view fusion
    # print("7. Performing multi-view fusion...")
    # matchers = (left_matcher, right_matcher, wls_filter)
    # fused_cloud = fuse_multi_view_eth3d(ref_id, neighbor_ids,
    #                                     imgs, K, T_wc, scene,
    #                                     resize=0.5,
    #                                     voxel=0.002,
    #                                     matchers=matchers)
    #
    # if fused_cloud is not None:
    #     o3d.io.write_point_cloud(Path(save_dir / "cloud_fused.ply"), fused_cloud)
    #
    #     # Create mesh
    #     print("8. Creating mesh from fused point cloud...")
    #     mesh = create_mesh(fused_cloud, depth=10, trim=0.15, target_triangles=300_000, prepare_cloud=True)
    #     if mesh is not None:
    #         o3d.io.write_triangle_mesh(Path(save_dir / "mesh.ply"), mesh)
    #         print(f"Mesh created with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    #     else:
    #         print("Mesh creation failed")
    # else:
    #     print("Multi-view fusion failed, no point cloud generated")
    #
    # print("Processing complete! Results saved to:", save_dir)