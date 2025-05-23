import cv2


def disparity_to_cloud(disp, Q, mask=None, color_img=None):
    """Convert disparity to point cloud with optional color"""
    # Clamp small disparities to avoid far-plane explosions
    disp_valid = disp.copy()
    disp_valid[disp_valid < 4] = 0

    # Apply mask if provided
    if mask is not None:
        mask = mask & (disp_valid > 0)
    else:
        mask = disp_valid > 0

    # Reproject to 3D
    pts = cv2.reprojectImageTo3D(disp_valid, Q, handleMissingValues=True)
    pcl = pts[mask]

    # Add color if available
    colors = None
    if color_img is not None:
        colors = color_img[mask]

    return pcl, colors
