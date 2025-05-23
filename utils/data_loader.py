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


def compute_baselines(imgs):
    """Compute Euclidean baselines between all camera pairs"""
    ids = list(imgs.keys())
    t = np.array([imgs[i].cam_from_world.translation for i in ids])
    pair_B = {(ids[i], ids[j]): np.linalg.norm(t[i] - t[j])
              for i in range(len(ids)) for j in range(i + 1, len(ids))}
    return pair_B


def shared_track_ratio(im_i, im_j):
    """Compute view overlap as ratio of shared 3D points"""
    s_i = {p.point3D_id for p in im_i.points2D if p.point3D_id != -1}
    s_j = {p.point3D_id for p in im_j.points2D if p.point3D_id != -1}
    if not s_i:
        return 0
    return len(s_i & s_j) / len(s_i | s_j)


def recover_absolute_scale(imgs, T_wc, points3D):
    """Recover absolute scale using COLMAP's 3D points"""
    # Use COLMAP's reconstructed 3D points as scale reference
    colmap_points = np.array([p.xyz for p in points3D.values()])

    # Compute characteristic scale from COLMAP reconstruction
    centroid = np.mean(colmap_points, axis=0)
    distances = np.linalg.norm(colmap_points - centroid, axis=1)
    scale_reference = np.median(distances)

    return scale_reference


def apply_scale_correction(cloud, target_scale, current_scale):
    """Apply scale correction to match ETH3D requirements"""
    scale_factor = target_scale / current_scale
    cloud.scale(scale_factor, center=cloud.get_center())
    return cloud
