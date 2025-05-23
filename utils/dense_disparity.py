import cv2
import numpy as np


def sgbm_with_consistency(w=5, num_disp=128):
    """Create SGBM matcher with left-right consistency check"""
    left_matcher = cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=w,
        P1=8 * 3 * w * w,
        P2=32 * 3 * w * w,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # Configure disparity filter
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.5)

    return left_matcher, right_matcher, wls_filter


def compute_filtered_disparity(left_img, right_img, left_matcher, right_matcher, wls_filter):
    """Compute disparity with consistency check and filtering"""
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Compute both left and right disparities
    left_disp = left_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
    right_disp = right_matcher.compute(gray_right, gray_left).astype(np.float32) / 16.0

    # Apply WLS filter for consistency check
    filtered_disp = wls_filter.filter(left_disp, gray_left, disparity_map_right=right_disp)

    # Create confidence/validity mask
    conf_map = wls_filter.getConfidenceMap()
    valid_mask = (conf_map > 0) & (filtered_disp > 4)

    # mask BEFORE smoothing
    filtered_disp[~valid_mask] = 0

    # Apply median + bilateral filter to keep edges
    filtered_disp = cv2.medianBlur(filtered_disp, 3)
    filtered_disp = cv2.bilateralFilter(filtered_disp, 9, 75, 75)

    return filtered_disp, valid_mask
