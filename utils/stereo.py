import cv2
import numpy as np
import open3d as o3d


def sgbm_with_consistency(min_disp=None, num_disp=None):
    if min_disp is None:
        min_disp = 0
        max_disp = 192
        # must be divisible by 16
        num_disp = (max_disp - min_disp) // 16 * 16

    print(f"Using SGBM with min_disp={min_disp}, num_disp={num_disp}")
    block_size = 3
    left_matcher = cv2.StereoSGBM.create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,  # 3 channels, increased regularization
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,  # Slightly relaxed for more matches
        uniquenessRatio=5,  # Increased for better uniqueness
        speckleWindowSize=50,  # Increased to remove more noise
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(12000.0)  # stronger smoothness
    wls_filter.setSigmaColor(2)  # preserve edges

    return left_matcher, right_matcher, wls_filter


def compute_filtered_disparity(left_img, right_img, left_matcher, right_matcher, wls_filter):
    """Compute disparity with consistency check and filtering"""
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE for contrast enhancement (Adaptive Histogram Equalization) cv2.createCLAHE(
    # CLAHE helps boost faint gradients on walls/floors - increase contrast in low-texture areas
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(32, 32))
    gray_left = clahe.apply(gray_left)
    gray_right = clahe.apply(gray_right)

    # 2. Bilateral filter to reduce noise while preserving edges
    gray_left = cv2.bilateralFilter(gray_left, 5, 50, 50)
    gray_right = cv2.bilateralFilter(gray_right, 5, 50, 50)

    # Compute both left and right disparities
    left_disp = left_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
    right_disp = right_matcher.compute(gray_right, gray_left).astype(np.float32) / 16.0

    # Apply WLS filter for consistency check
    filtered_disp = wls_filter.filter(left_disp, gray_left, disparity_map_right=right_disp)

    # Improved validity mask
    conf_map = wls_filter.getConfidenceMap()
    valid_mask = (conf_map > 0) & (filtered_disp > 1)  # Lower threshold to keep more data

    filtered_disp = cv2.inpaint(filtered_disp, (~valid_mask).astype(np.uint8), 3, cv2.INPAINT_TELEA)

    return filtered_disp, valid_mask


def disparity_to_cloud(disp, Q, mask=None, color_img=None, min_depth=0.1, max_depth=30.0):
    """
    Convert disparity to point cloud with optional color and depth filtering
    """
    # Clamp small disparities to avoid far-plane explosions
    disp_valid = disp.copy().astype(np.float32)
    disp_valid[disp_valid <= 0] = 0
    disp_valid[np.isnan(disp_valid)] = 0
    disp_valid[np.isinf(disp_valid)] = 0

    # Apply initial mask if provided
    if mask is not None:
        mask = mask & (disp_valid > 0)
    else:
        mask = disp_valid > 0

    # Reproject to 3D
    pts = cv2.reprojectImageTo3D(disp_valid, Q, handleMissingValues=True)

    # Apply depth limits and remove outliers
    depths = np.linalg.norm(pts, axis=2)
    depth_mask = (depths > min_depth) & (depths < max_depth)
    mask = mask & depth_mask

    # Remove points with large depth discontinuities
    depth_grad = np.gradient(depths)
    grad_mask = (np.abs(depth_grad[0]) < 1.0) & (np.abs(depth_grad[1]) < 1.0)
    mask = mask & grad_mask

    # Extract valid points
    pcl = pts[mask]

    # Add color if available
    colors = None
    if color_img is not None:
        colors = color_img[mask]
        if colors.shape[1] == 3:
            colors = colors[:, [2, 1, 0]]  # BGR to RGB

    return pcl, colors


def create_gt_cloud(disp_gt, Q, valid_mask, color_img, R=None, t=None):
    """Create point cloud from ground truth disparity map

    Args:
        disp_gt: Ground truth disparity map
        Q: Reprojection matrix
        valid_mask: Validity mask
        color_img: Color image
        R: Optional rotation matrix
        t: Optional translation vector

    Returns:
        o3d.geometry.PointCloud: Point cloud in world coordinates
    """
    # Create mask for valid GT points
    mask = (disp_gt > 0) & np.isfinite(disp_gt) & valid_mask

    # Generate point cloud
    pts, colors = disparity_to_cloud(disp_gt, Q, mask, color_img)

    # Create Open3D point cloud
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)

    # Transform to world frame if extrinsics provided
    if R is not None and t is not None:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()
        cloud.transform(T)

    return cloud


def rectify_pair(img_id1, img_id2, imgs, K, T_wc, scene_dir, resize=0.5):
    """
    Rectify an image pair for stereo matching, optimized for ETH3D dataset.

    This function:
    1. Properly handles image paths for ETH3D directory structure
    2. Computes relative pose correctly
    3. Uses view-specific intrinsics when available
    4. Handles ROI information to crop invalid regions
    5. Performs resizing after rectification (better for precision)
    6. Properly adjusts Q matrix for both cropping and resizing
    """
    # Load images (check both possible path structures)
    img_dir = scene_dir / "dslr_images_undistorted"

    # Get image paths, handling different possible structures
    name1 = imgs[img_id1].name
    name2 = imgs[img_id2].name

    # Remove prefix if present
    name1 = name1.replace("dslr_images_undistorted/", "")
    name2 = name2.replace("dslr_images_undistorted/", "")

    # Try different possible paths
    if (img_dir / name1).exists():
        img1 = cv2.imread(str(img_dir / name1))
        img2 = cv2.imread(str(img_dir / name2))
    else:
        # Fall back to direct paths
        img1 = cv2.imread(str(scene_dir / imgs[img_id1].name))
        img2 = cv2.imread(str(scene_dir / imgs[img_id2].name))

    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Could not load images for IDs {img_id1}, {img_id2}")

    h, w = img1.shape[:2]

    # Get camera-specific intrinsics (if available, otherwise use first camera)
    K1 = K.get(img_id1, K[next(iter(K))])
    K2 = K.get(img_id2, K[next(iter(K))])

    # Get relative pose - world to camera transformations
    T1, T2 = T_wc[img_id1], T_wc[img_id2]
    # Relative pose from cam1 to cam2
    T_21 = T2 @ np.linalg.inv(T1)
    R = T_21[:3, :3]
    t = T_21[:3, 3]

    # Compute stereo rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, None,  # No distortion
        K2, None,  # No distortion
        (w, h), R, t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0  # Crop to valid region
    )

    # Get rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, None, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(
        K2, None, R2, P2, (w, h), cv2.CV_32FC1)

    # Apply rectification with border handling
    rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT)
    rect2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT)

    # Find common valid region
    y_min = max(roi1[1], roi2[1])
    y_max = min(roi1[1] + roi1[3], roi2[1] + roi2[3])
    x_min = max(roi1[0], roi2[0])
    x_max = min(roi1[0] + roi1[2], roi2[0] + roi2[2])

    # Ensure valid region exists
    if x_min >= x_max or y_min >= y_max:
        # If no valid region, use defaults
        y_min, y_max = 0, h
        x_min, x_max = 0, w

    # Crop to common region
    rect1 = rect1[y_min:y_max, x_min:x_max]
    rect2 = rect2[y_min:y_max, x_min:x_max]

    # Update Q matrix for cropped region
    Q_adjusted = Q.copy()
    Q_adjusted[3, 2] -= x_min  # Adjust principal point x
    Q_adjusted[3, 3] -= y_min  # Adjust principal point y

    # Store original size before resize
    orig_size = rect1.shape[:2]

    # Resize if requested (after rectification)
    if resize != 1.0:
        h_rect, w_rect = rect1.shape[:2]
        new_size = (int(w_rect * resize), int(h_rect * resize))
        rect1 = cv2.resize(rect1, new_size, interpolation=cv2.INTER_AREA)
        rect2 = cv2.resize(rect2, new_size, interpolation=cv2.INTER_AREA)

        # Adjust Q matrix for resize
        Q_adjusted[0, 0] *= resize
        Q_adjusted[1, 1] *= resize
        Q_adjusted[0, 3] *= resize
        Q_adjusted[1, 3] *= resize

    # Verify images are same size
    assert rect1.shape == rect2.shape, "Rectified images must have same dimensions"

    # Return rectification info for potential up-sampling
    rect_info = {
        "original_size": (h, w),
        "crop_region": (y_min, y_max, x_min, x_max),
        "rectified_size": orig_size,
        "resize_factor": resize
    }

    return rect1, rect2, Q_adjusted, rect_info
