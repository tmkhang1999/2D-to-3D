import cv2
import numpy as np


def rectify_pair(img_id1, img_id2, imgs, K, T_wc, scene_dir, resize=0.5):
    """Rectify an image pair for stereo matching"""
    K1, K2 = K[img_id1], K[img_id2]
    T1, T2 = T_wc[img_id1], T_wc[img_id2]
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]

    # Use the correct image path from the dslr_images_undistorted folder
    img_path1 = scene_dir / imgs[img_id1].name
    img_path2 = scene_dir / imgs[img_id2].name

    img1 = cv2.imread(str(img_path1))
    img2 = cv2.imread(str(img_path2))

    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Failed to load images at {img_path1} or {img_path2}")

    orig_size = img1.shape[:2]

    # Downscale for processing
    if resize != 1.0:
        img1_small = cv2.resize(img1, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
        img2_small = cv2.resize(img2, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
        K1_small = K1.copy()
        K2_small = K2.copy()
        K1_small[:2] *= resize
        K2_small[:2] *= resize
    else:
        img1_small, img2_small = img1, img2
        K1_small, K2_small = K1, K2

    # Calculate rectification transforms
    R, T, _, _, Q, roi1, roi2 = cv2.stereoRectify(
        K1_small, None, K2_small, None,
        img1_small.shape[:2][::-1],
        R2 @ R1.T,
        R1 @ (t1 - t2),
        flags=cv2.CALIB_ZERO_DISPARITY
    )

    # Compute rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(
        K1_small, None, R, K1_small, img1_small.shape[:2][::-1], cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        K2_small, None, R, K2_small, img2_small.shape[:2][::-1], cv2.CV_32FC1
    )

    # Apply rectification
    rect_img1 = cv2.remap(img1_small, map1x, map1y, cv2.INTER_LINEAR)
    rect_img2 = cv2.remap(img2_small, map2x, map2y, cv2.INTER_LINEAR)

    # Store homographies for later upsampling if needed
    H1 = np.eye(3)  # To be computed if needed for full-resolution processing
    H2 = np.eye(3)

    return rect_img1, rect_img2, Q, (orig_size, resize, H1, H2)