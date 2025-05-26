import re
from pathlib import Path

import numpy as np

from utils import *


def load_middlebury_calib(calib_path: Path):
    s = calib_path.read_text()
    m0 = re.search(r"cam0=\[([^\]]+)\]", s).group(1)
    m1 = re.search(r"cam1=\[([^\]]+)\]", s).group(1)

    # replace semicolons (row separators) with spaces
    m0 = m0.replace(';', ' ')
    m1 = m1.replace(';', ' ')

    # now safely parse into 9 floats and reshape
    K0 = np.fromstring(m0, sep=' ').reshape(3, 3)
    K1 = np.fromstring(m1, sep=' ').reshape(3, 3)

    doffs = float(re.search(r"doffs=([-\d.]+)", s).group(1))
    baseline = float(re.search(r"baseline=([-\d.]+)", s).group(1))
    width = int(re.search(r"width=(\d+)", s).group(1))
    height = int(re.search(r"height=(\d+)", s).group(1))
    ndisp = int(re.search(r"ndisp=(\d+)", s).group(1))
    vmin = int(re.search(r"vmin=(\d+)", s).group(1))
    vmax = int(re.search(r"vmax=(\d+)", s).group(1))

    return K0, K1, doffs, baseline, width, height, ndisp, vmin, vmax


if __name__ == "__main__":
    scene = Path("dataset/chess")
    print("1. Loading data...")
    K0, K1, doffs, baseline, w, h, ndisp, vmin, vmax = load_middlebury_calib(scene / "calib.txt")
    # read the two rectified views
    print("2. Processing stereo pair...")
    imgL = cv2.imread(str(scene / "im0.png"))
    imgR = cv2.imread(str(scene / "im1.png"))

    baseline_m = baseline / 1000.0  # mm → m
    f = K0[0, 0]  # focal length in px
    cx = K0[0, 2]
    cy = K0[1, 2]
    print(f"Baseline: {baseline_m} m, Focal length: {f} px, Center: ({cx}, {cy}) px")

    Q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, f],
        [0, 0, 1 / baseline_m, -doffs / baseline_m]
    ], dtype=np.float32)
    # With this Q you’ll get 𝑍 = (𝑓 × baseline_m) / (𝑑 − doffs) all in meters, and positive for positive disparities

    # 3. Rectify pair
    print("3. Saving rectified images...")
    combined = np.hstack((imgL, imgR))

    # Draw horizontal lines
    for y in range(0, h, 150):
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 2)
    # Save the combined image
    combined_image_path = Path(scene) / f"rect_img.png"
    cv2.imwrite(str(combined_image_path), combined)

    # 4. Compute & filter disparity
    print("4. Computing disparity...")
    min_disp = vmin - doffs
    raw_span = vmax - min_disp
    num_disp = ((raw_span + 15) // 16) * 16  # → 192 px
    left_matcher, right_matcher, wls_filter = sgbm_with_consistency(min_disp=int(min_disp), num_disp=int(num_disp))
    filtered_disp, valid_mask = compute_filtered_disparity(
        imgL, imgR, left_matcher, right_matcher, wls_filter
    )

    # Save raw disparity map
    filtered_disp_gray = (filtered_disp * 255 / np.max(filtered_disp)).astype(np.uint8)
    disparity_map_path = Path(scene) / f"disp_raw.png"
    cv2.imwrite(str(disparity_map_path), filtered_disp_gray)

    # 5. Reproject to 3D
    print("5. Generating depth from disparity...")
    pts, colors = disparity_to_cloud(filtered_disp, Q, valid_mask, imgL)

    # 6. Clean & save
    print("6. Filtering point cloud...")
    cloud = clean_cloud(pts, colors)
    o3d.io.write_point_cloud(scene / "chess_cloud.ply", cloud)
