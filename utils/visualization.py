import argparse
from pathlib import Path

"""
Visualization utilities for stereo matching and reconstruction results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def draw_disparity(disparity, mask=None):
    """Draw disparity map as color image.

    Args:
        disparity: Disparity map
        mask: Optional validity mask

    Returns:
        np.ndarray: Color visualization
    """
    if mask is not None:
        disparity = disparity.copy()
        disparity[~mask] = 0

    valid_disp = disparity[disparity > 0]
    if valid_disp.size == 0:
        print("Warning: No valid disparity values found")
        return np.zeros((*disparity.shape, 3), dtype=np.uint8)

    norm = plt.Normalize(vmin=valid_disp.min(), vmax=valid_disp.max())
    cmap = plt.get_cmap('magma')
    vis = cmap(norm(disparity))[:, :, :3]
    return (vis * 255).astype(np.uint8)


def draw_matches(img1, img2, pts1, pts2, matches, mask=None):
    """Draw matching points between image pairs.

    Args:
        img1: First image
        img2: Second image
        pts1: Keypoints in first image
        pts2: Keypoints in second image
        matches: List of DMatch objects
        mask: Optional mask for good matches

    Returns:
        np.ndarray: Image with drawn matches
    """
    # Convert to cv2 keypoints
    kp1 = [cv2.KeyPoint(x=p[0], y=p[1], size=1) for p in pts1]
    kp2 = [cv2.KeyPoint(x=p[0], y=p[1], size=1) for p in pts2]

    # Draw matches
    img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                          matchColor=(0, 255, 0),
                          singlePointColor=(255, 0, 0),
                          matchesMask=mask,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img


def draw_epipolar_lines(img1, img2, pts1, lines2):
    """Draw epipolar lines and corresponding points.

    Args:
        img1: First image
        img2: Second image
        pts1: Points in first image
        lines2: Corresponding epipolar lines in second image

    Returns:
        tuple: (image1 with points, image2 with lines)
    """
    h, w = img1.shape[:2]
    img1_copy = img1.copy()
    img2_copy = img2.copy()

    for pt, line in zip(pts1, lines2):
        # Draw point in first image
        x, y = map(int, pt)
        cv2.circle(img1_copy, (x, y), 5, (0, 255, 0), -1)

        # Draw corresponding line in second image
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [w, -(line[2] + line[0] * w) / line[1]])
        cv2.line(img2_copy, (x0, y0), (x1, y1), (255, 0, 0), 2)

    return img1_copy, img2_copy


def visualize_point_cloud(cloud, window_name="Point Cloud"):
    """Visualize point cloud using Open3D with point picking capability."""
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name)
    vis.add_geometry(cloud)

    # Set view and rendering options
    vis.get_view_control().set_zoom(0.8)
    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = np.array([0, 0, 0])

    print("\nPoint Picking Instructions:")
    print("1. Hold Shift + Left mouse button to select points")
    print("2. Selected points will be highlighted in red")
    print("3. Point coordinates will be printed in real-time")
    print("4. Press 'Q' or close the window to exit")

    vis.run()
    vis.destroy_window()


def visualize_mesh(mesh, window_name="Reconstructed Mesh"):
    """Visualize triangle mesh using Open3D.

    Args:
        mesh: Open3D triangle mesh
        window_name: Title for visualization window
    """
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name)

    # Add geometry
    vis.add_geometry(mesh)

    # Set rendering options
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.mesh_show_wireframe = True

    # Set default camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])

    # Run visualizer
    vis.run()
    vis.destroy_window()


def draw_rectification_check(rect1, rect2, step=150):
    """Draw horizontal lines to check rectification quality.

    Args:
        rect1: First rectified image
        rect2: Second rectified image
        step: Vertical spacing between check lines

    Returns:
        tuple: (image1 with lines, image2 with lines)
    """
    img1 = rect1.copy()
    img2 = rect2.copy()

    h, w = img1.shape[:2]

    for y in range(0, h, step):
        cv2.line(img1, (0, y), (w, y), (0, 255, 0), 1)
        cv2.line(img2, (0, y), (w, y), (0, 255, 0), 1)

    return img1, img2


def save_rectified_pair(rect_img1, rect_img2, save_path):
    h = min(rect_img1.shape[0], rect_img2.shape[0])
    img1 = rect_img1[:h]
    img2 = rect_img2[:h]

    # Draw horizontal lines
    combined = np.hstack((img1, img2))
    for y in range(0, h, 150):
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 2)
    cv2.imwrite(save_path, combined)


def visualize_file(file_path):
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return

    # Determine file type and load accordingly
    if file_path.suffix.lower() == '.ply':
        if 'mesh' in file_path.name:
            # Load as mesh
            geometry = o3d.io.read_triangle_mesh(Path(file_path))
            geometry.compute_vertex_normals()
            print(f"Loaded mesh with {len(geometry.vertices)} vertices and {len(geometry.triangles)} triangles")
        else:
            # Load as point cloud
            geometry = o3d.io.read_point_cloud(Path(file_path))
            print(f"Loaded point cloud with {len(geometry.points)} points")
    else:
        print(f"Unsupported file format: {file_path.suffix}")
        return

    # Visualize
    print(f"Visualizing {file_path.name}...")
    o3d.visualization.draw_geometries([geometry])



import sys
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a .ply file from a specified path.")
    parser.add_argument("--path", type=str, required=False,
                       default="./dataset/office/results/cloud_single_pair_21_22.ply",
                       help="Path to the .ply file")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_file():
        print(f"Error: The specified file does not exist: {path}")
        sys.exit(1)

    cloud = o3d.io.read_point_cloud(str(path))
    if not cloud.has_points():
        print(f"Error: Could not load point cloud from {path}")
        sys.exit(1)
    
    print(f"Loaded point cloud with {len(cloud.points)} points")
    visualize_file(path)