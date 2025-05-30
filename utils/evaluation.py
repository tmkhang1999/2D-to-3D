import numpy as np


def compute_metrics(disp_est, disp_gt, mask):
    """Compute MAE, RMS, Bad1%, Bad2%, Bad4% metrics"""
    # Only evaluate on valid pixels
    valid = (disp_gt > 0) & np.isfinite(disp_gt) & mask
    if not np.any(valid):
        return 0, 0, 0, 0, 0

    # Compute absolute error
    abs_err = np.abs(disp_est[valid] - disp_gt[valid])

    # MAE
    mae = np.mean(abs_err)

    # RMS
    rms = np.sqrt(np.mean(abs_err ** 2))

    # Bad pixel percentages
    bad1 = np.mean(abs_err > 1.0) * 100
    bad2 = np.mean(abs_err > 2.0) * 100
    bad4 = np.mean(abs_err > 4.0) * 100

    return mae, rms, bad1, bad2, bad4


def compute_cloud_metrics(cloud_pred, cloud_gt, threshold=0.05):
    """Compute cloud-to-cloud metrics: Chamfer distance, Accuracy, Completeness, F1 Score

    Args:
        cloud_pred: Predicted point cloud
        cloud_gt: Ground truth point cloud
        threshold: Distance threshold for accuracy/completeness (in meters)

    Returns:
        dict: Dictionary containing metrics
    """
    # Compute point-to-point distances
    distances_pred_to_gt = np.asarray(cloud_pred.compute_point_cloud_distance(cloud_gt))
    distances_gt_to_pred = np.asarray(cloud_gt.compute_point_cloud_distance(cloud_pred))

    # Chamfer distance
    chamfer_dist = np.mean(distances_pred_to_gt) + np.mean(distances_gt_to_pred)

    # Accuracy: percentage of points in predicted cloud within threshold of GT
    accuracy = np.mean(distances_pred_to_gt < threshold)

    # Completeness: percentage of points in GT cloud within threshold of predicted
    completeness = np.mean(distances_gt_to_pred < threshold)

    # F1 Score
    f1_score = 2 * (accuracy * completeness) / (accuracy + completeness) if (accuracy + completeness) > 0 else 0

    return {
        'chamfer_dist': chamfer_dist,
        'accuracy': accuracy,
        'completeness': completeness,
        'f1_score': f1_score
    }