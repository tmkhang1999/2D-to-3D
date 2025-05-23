# --------------------------------------------------------------------------- #
#  MULTI-VIEW FUSION (reference + k neighbours, with ICP + voxel merge)       #
# --------------------------------------------------------------------------- #
import open3d as o3d
from tqdm import tqdm
import numpy as np

from utils.dense_disparity import compute_filtered_disparity
from utils.depth_estimator import disparity_to_cloud
from utils.outlier_filter import clean_cloud
from utils.rectification import rectify_pair


def fuse_multi_view(ref_id, neighbor_ids, imgs, K, T_wc,
                    scene_dir, resize=0.5,
                    voxel=0.002, icp_thresh=0.02,
                    matchers=None):
    """
    Returns an Open3D point-cloud in WORLD coordinates that combines
    the reference view and its neighbours.
    `matchers` is the (left_matcher, right_matcher, wls_filter) triple
    you already built once in main().
    """

    # helper ---------------------------------------------------------------
    def _ensure_normals(cloud, r=0.02, k=30):
        if not cloud.has_normals():
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=k))
            cloud.orient_normals_consistent_tangent_plane(50)

    def cloud_from_pair(id1, id2):
        rect1, rect2, Q, _ = rectify_pair(id1, id2, imgs, K, T_wc,
                                          scene_dir, resize)
        l_m, r_m, wls = matchers
        disp, mask = compute_filtered_disparity(rect1, rect2, l_m, r_m, wls)
        pts, rgb = disparity_to_cloud(disp, Q, mask, rect1)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float32)/255.0)
        # bring to *world* frame
        pc.transform(np.linalg.inv(T_wc[id1]))
        return pc

    # build reference cloud ----------------------------------------------
    ref_cloud = cloud_from_pair(ref_id, neighbor_ids[0])   # use first pair
    ref_cloud = clean_cloud(np.asarray(ref_cloud.points),
                            np.asarray(ref_cloud.colors)*255, voxel)
    _ensure_normals(ref_cloud)

    # iterate neighbours --------------------------------------------------
    for nb_id in tqdm(neighbor_ids, desc="ICP fusing"):
        nb_cloud = cloud_from_pair(ref_id, nb_id)
        _ensure_normals(nb_cloud)

        # coarse → fine ICP (point-to-plane)
        reg = o3d.pipelines.registration.registration_icp(
                nb_cloud, ref_cloud,
                icp_thresh,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
        nb_cloud.transform(reg.transformation)
        ref_cloud += nb_cloud # merge
        ref_cloud = ref_cloud.voxel_down_sample(voxel)
        _ensure_normals(ref_cloud)

    return ref_cloud

from utils.data_loader import compute_baselines
from utils.stereo_matcher import best_pairs


def complete_scene_reconstruction(imgs, K, T_wc, scene_dir, matchers):
    """Reconstruct complete scene using all available views"""
    all_clouds = []

    # Process all viable stereo pairs
    B = compute_baselines(imgs)
    all_pairs = best_pairs(imgs, B, k=20, min_B=0.05, max_B=2.0, min_overlap=0.3)

    for ref_id, sec_id in all_pairs:
        try:
            # Generate point cloud for each pair
            rect1, rect2, Q, _ = rectify_pair(ref_id, sec_id, imgs, K, T_wc, scene_dir)
            l_m, r_m, wls = matchers
            disp, mask = compute_filtered_disparity(rect1, rect2, l_m, r_m, wls)
            pts, colors = disparity_to_cloud(disp, Q, mask, rect1)

            # Convert to world coordinates
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pts)
            cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)
            cloud.transform(np.linalg.inv(T_wc[ref_id]))

            all_clouds.append(cloud)
        except Exception as e:
            print(f"Failed to process pair {ref_id}-{sec_id}: {e}")
            continue

    # Merge all clouds with global registration
    if len(all_clouds) > 1:
        merged_cloud = all_clouds[0]
        for cloud in all_clouds[1:]:
            merged_cloud += cloud
        merged_cloud = merged_cloud.voxel_down_sample(0.001)
    else:
        merged_cloud = all_clouds[0] if all_clouds else None

    return merged_cloud
