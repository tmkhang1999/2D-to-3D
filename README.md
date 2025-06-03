# 3D Multi-View Stereo Reconstruction

This project implements a full pipeline for dense 3D reconstruction from multiple calibrated images, evaluated on the ETH3D and Middlebury Chess (Mobile 2021) datasets.

## Overview

The project focuses on creating accurate 3D reconstructions from multiple 2D images using classical computer vision techniques. The pipeline has been tested and evaluated on two challenging datasets:

1. **ETH3D Dataset**: A comprehensive dataset for multi-view stereo reconstruction with high-resolution images and ground truth 3D scans. The dataset includes both indoor and outdoor scenes with varying complexity.
   - [Download ETH3D Dataset](https://www.eth3d.net/datasets)
   - Features:
     - High-resolution multi-view images
     - Ground truth 3D scans
     - Camera calibration data
     - Occlusion masks
     - Depth maps
   - Dataset Structure:
     ```
     dataset/
     ├── delivery_area/                    # Example scene
     │   ├── dslr_calibration_undistorted/ # COLMAP calibration files
     │   │   ├── cameras.txt              # Camera parameters
     │   │   ├── images.txt               # Image poses and features
     │   │   └── points3D.txt             # 3D point cloud
     │   ├── images/                      # Input images
     │   │   ├── 0000.jpg
     │   │   ├── 0001.jpg
     │   │   └── ...
     │   ├── results/                     # Output directory
     │   │   ├── rect_9_8.png            # Rectified image pair
     │   │   ├── disp_9_8_raw.png        # Disparity map
     │   │   ├── cloud_right_9_8.ply      # Point cloud
     │   │   ├── cloud_fused.ply          # Fused point cloud
     │   │   └── mesh.ply                 # Final mesh
     │   ├── depth/                       # Ground truth depth maps
     │   └── occlusion/                   # Occlusion masks
     └── other_scenes/                    # Additional scenes
     ```

2. **Middlebury Chess Dataset (Mobile 2021)**: A stereo dataset featuring a chessboard pattern for accurate calibration and evaluation.
   - [Download Middlebury Chess Dataset](https://vision.middlebury.edu/stereo/data/scenes2021/)
   - Features:
     - High-quality stereo pairs
     - Ground truth disparity maps
     - Camera calibration data
     - Chessboard pattern for pose estimation
   - Dataset Structure:
     ```
     dataset/
     ├── chess_scene/               # Example scene
     │   ├── im0.png               # Left image
     │   ├── im1.png               # Right image
     │   ├── disp0.pfm             # Ground truth disparity
     │   ├── calib.txt             # Camera calibration
     │   ├── corners_manual.npy    # Optional manual corner annotations
     │   ├── results/              # Output directory
     │   │   ├── rectified_pair.png    # Rectified image pair
     │   │   ├── disp_raw.png         # Disparity map
     │   │   ├── corners_vis.png       # Chessboard corner visualization
     │   │   ├── cloud.ply            # Generated point cloud
     │   │   ├── cloud_gt.ply         # Ground truth point cloud
     │   │   └── mesh.ply             # Final mesh
     └── other_scenes/             # Additional scenes
     ```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd 2D-to-3D
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- numpy ~= 2.2.6
- open3d ~= 0.19.0
- pycolmap ~= 3.11.1
- opencv-python ~= 4.11.0.86

## Usage

### Middlebury Chess Dataset

The pipeline processes the Middlebury Chess dataset with the following steps:

```python
from chess_pipeline import process_scene

# Process a single scene
scene_name = "chess_scene"
scene_path = Path("dataset/chess_scene")
K0, imgL, cloud, cloud_gt, disparity_metrics, metrics_data, chessboard_transform = process_scene(scene_name, scene_path)
```

### ETH3D Dataset

For ETH3D dataset processing:

```python
from eth3d_pipeline import process_eth3d_scene

# Process ETH3D scene
scene_path = Path("dataset/delivery_area")
save_dir = Path("dataset/delivery_area/results")
process_eth3d_scene(scene_path, save_dir)
```

## Pipeline Components

### 1. Stereo Calibration & Rectification
- Camera parameter extraction using COLMAP (ETH3D) or calibration files (Middlebury)
- Image rectification for accurate correspondence
- Support for down-sampling during rectification

### 2. Preprocessing
- Adaptive histogram equalization (CLAHE)
- Enhancement of low-texture regions
- Image resizing and normalization

### 3. Stereo Matching & Disparity Estimation
- Semi-Global Block Matching (SGBM)
- Left-right consistency checking
- WLS filtering for improved disparity maps
- Configurable disparity range and window size

### 4. 3D Point Cloud Generation
- Disparity map reprojection to 3D
- Filtering and outlier removal
- Point cloud optimization
- Statistical outlier removal
- Voxel down-sampling

### 5. Multi-View Fusion
- Point cloud alignment using Iterative Closest Point (ICP)
- Merging of multiple views into unified models
- Support for ETH3D multi-view fusion
- Chessboard-based pose estimation for Middlebury dataset

### 6. Mesh Generation
- Surface mesh creation using ball-pivoting algorithm
- Mesh optimization and refinement
- Poisson surface reconstruction for ETH3D scenes
- Configurable mesh parameters (depth, trim factor, target triangles)

## Evaluation Metrics

### Pixel-wise Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMS)
- Bad pixel percentages (Bad1%, Bad2%, Bad4%)

### 3D Metrics
- Chamfer distance
- Accuracy
- Completeness
- F1 Score

## Key Findings

- Accurate calibration and parameter tuning are crucial for quality reconstructions
- Classical methods face challenges with low-texture regions and occlusions (ETH3D dataset)
- High-quality datasets (Middlebury Chess) yield better results
- Manual extrinsic estimation and translation correction are necessary for multi-view fusion without ground-truth poses
- Quantitative and visual evaluations identify areas for improvement

## Future Work

- Automation of hyperparameter optimization
- Integration of advanced texture enhancement techniques
- Implementation of deep learning-based stereo matching
- Incorporation of bundle adjustment
- Development of improved multi-view fusion methods

## Contributors

- **Minh Khang Tran**
  - Pipeline implementation
  - Dataset preparation
  - Experiments
  - Visualization
  - Reporting

- **Anirban Das**
  - Literature review
  - Methodology design
  - Parameter selection
  - Debugging
