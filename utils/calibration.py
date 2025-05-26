from pathlib import Path

import numpy as np
import pycolmap


def load_colmap(scene_dir: Path):
    """Load camera intrinsics and poses from COLMAP format files"""

    # Verify the path contains required files
    reconstruction_path = scene_dir / "dslr_calibration_undistorted"
    if not all((reconstruction_path / f).exists()
               for f in ["cameras.txt", "images.txt", "points3D.txt"]):
        raise FileNotFoundError(
            f"Could not find all required COLMAP files in {reconstruction_path}.\n"
            "Expected: cameras.txt, images.txt, points3D.txt"
        )

    # Correct way to read COLMAP data with pycolmap
    reconstruction = pycolmap.Reconstruction(str(reconstruction_path))

    # Extract cameras, images and points
    cams = {cam.camera_id: cam for cam in reconstruction.cameras.values()}
    imgs = {img.image_id: img for img in reconstruction.images.values()}
    points3D = reconstruction.points3D

    # Intrinsics per image
    K = {}
    for im in imgs.values():
        cam = cams[im.camera_id]
        K[im.image_id] = cam.calibration_matrix()  # Get calibration matrix directly

    # World -> cam SE(3)
    T_wc = {}
    for im in imgs.values():
        # Get rotation matrix directly from the pose object
        R = im.cam_from_world.rotation.matrix()

        # Get translation vector
        t = im.cam_from_world.translation

        # Construct transform matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        T_wc[im.image_id] = T

    return imgs, K, T_wc, points3D


def compute_baselines(imgs, T_wc):
    ids = sorted(imgs.keys())
    baselines = {}

    # Get camera centers
    centers = {}
    for img_id in ids:
        # Extract camera center from transform
        T = T_wc[img_id]
        R = T[:3, :3]
        t = T[:3, 3]
        # C = -R^T * t (inverse transform)
        center = -R.T @ t
        centers[img_id] = center

    # Compute pairwise distances
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id1, id2 = ids[i], ids[j]
            dist = np.linalg.norm(centers[id1] - centers[id2])
            baselines[(id1, id2)] = dist

    return baselines


def shared_track_ratio(img1, img2):
    # Get sets of 3D point IDs
    pts1 = {pt.point3D_id for pt in img1.points2D if pt.point3D_id != -1}
    pts2 = {pt.point3D_id for pt in img2.points2D if pt.point3D_id != -1}

    # Compute intersection over union
    if not pts1 or not pts2:
        return 0.0

    shared = len(pts1 & pts2)
    total = len(pts1 | pts2)

    return shared / total


def best_pairs(imgs, pair_B, k=5, min_B=0.15, max_B=0.6, min_overlap=0.7):
    """Find top-k pairs with good baseline and overlap"""
    pairs = []
    for (i, j), B in pair_B.items():
        if not (min_B <= B <= max_B):
            continue
        ov = shared_track_ratio(imgs[i], imgs[j])
        if ov < min_overlap:
            continue
        score = B * ov
        pairs.append(((i, j), score))

    # Sort by score and return top-k pairs
    pairs.sort(key=lambda x: x[1], reverse=True)
    if not pairs:
        return []
    print(f"Score of {pairs[:k][0][0]}: {pairs[:k][0][1]}")

    return [p[0] for p in pairs[:k]]


def find_neighbor_views(imgs, pair_B, ref_id, k=3, min_B=0.1, max_B=1.1, min_overlap=0.5):
    """Find best neighbor views for stereo matching."""
    # Get all pairs with reference view
    pairs = []
    for (id1, id2), B in pair_B.items():
        if id1 == ref_id:
            other_id = id2
        elif id2 == ref_id:
            other_id = id1
            B = pair_B.get((ref_id, other_id)) or pair_B[(other_id, ref_id)]
        else:
            continue

        # Check baseline constraints
        if not (min_B <= B <= max_B):
            continue

        # Check overlap
        overlap = shared_track_ratio(imgs[ref_id], imgs[other_id])
        if overlap < min_overlap:
            continue

        pairs.append((other_id, B, overlap))

    # Sort by overlap (prefer more overlap when baselines similar)
    pairs.sort(key=lambda x: (-x[2], x[1]))

    return [id for id, _, _ in pairs[:k]]


import numpy as np


def compute_depth_range_from_colmap(points3D, imgs, K, T_wc, ref_id, second_id, baseline):
    valid_pids = []
    for pt in imgs[ref_id].points2D:
        pid = pt.point3D_id
        # 1) try casting to int
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            continue
        # 2) skip negative IDs (COLMAP uses –1 for “no point”)
        if pid < 0:
            continue
        # 3) skip if it's missing from the points3D dict
        if pid not in points3D:
            continue
        valid_pids.append(pid)

    if not valid_pids:
        raise RuntimeError(f"No valid 3D points found for image {ref_id}")

    # Stack their world‐coordinates
    pts_world = np.vstack([points3D[pid].xyz for pid in set(valid_pids)])

    # Transform points to the camera coordinate frame
    T = T_wc[ref_id]
    R, t = T[:3, :3], T[:3, 3]
    pts_cam = (R @ pts_world.T).T + t

    # Get depth range
    Znear, Zfar = pts_cam[:, 2].min(), pts_cam[:, 2].max()

    # Compute disparity bounds
    f = K[ref_id][0, 0]  # focal length in pixels
    vmax = (f * baseline) / Znear  # max disparity (closest)
    vmin = (f * baseline) / Zfar   # min disparity (farthest)

    return vmin, vmax

