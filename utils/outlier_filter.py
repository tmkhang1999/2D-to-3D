import numpy as np
import open3d as o3d


def clean_cloud(pcl, colors=None, voxel=0.001):
    """Filter outliers and regularize point cloud"""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pcl)
    if colors is not None:
        colors_float = colors.astype(np.float64) / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors_float)

    # Statistical outlier removal
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Voxel downsampling
    cloud = cloud.voxel_down_sample(voxel)

    # RANSAC plane fitting for regularization (in patches)
    # This is a simplified version; a full implementation would segment the cloud
    # into patches and fit planes to each patch
    # _, _ = cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

    return cloud
