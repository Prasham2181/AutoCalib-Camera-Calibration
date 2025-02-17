# Camera Calibration using Zhang's Method

A Python implementation of Zhang's camera calibration technique using a planar checkerboard pattern. This project provides tools for estimating both intrinsic and extrinsic camera parameters through checkerboard corner detection, homography computation, and non-linear optimization.

## Features

- Automatic checkerboard corner detection and sub-pixel refinement
- Intrinsic camera parameter estimation (focal length, principal point)
- Extrinsic parameter computation (rotation, translation) for each calibration image
- Lens distortion correction (radial distortion coefficients k₁, k₂)
- Visualization tools for calibration quality assessment
- Image undistortion capabilities

## Overview

The calibration pipeline follows these key steps:

1. **Image Acquisition and Corner Detection**
   - Process multiple JPEG calibration images
   - Automatic checkerboard corner detection using OpenCV
   - Sub-pixel corner refinement for improved accuracy

2. **Initial Parameter Estimation**
   - Homography computation using Direct Linear Transform (DLT)
   - Intrinsic parameter extraction from homography constraints
   - Per-image extrinsic parameter estimation (R, t)

3. **Non-Linear Parameter Refinement**
   - Joint optimization of all parameters
   - Minimization of reprojection error
   - Estimation of lens distortion coefficients
   - Bundle adjustment for improved accuracy

4. **Visualization and Validation**
   - Image undistortion using calibrated parameters
   - Overlay of detected vs. reprojected corners
   - Reprojection error visualization and statistics

## Installation

### Prerequisites
- Python 3.x
- OpenCV
- NumPy
- SciPy

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/camera-calibration.git
cd camera-calibration

# Install required packages
pip install opencv-python numpy scipy
```

## Usage

```python
# Example code snippet for running calibration
from calibration import CameraCalibration

# Initialize calibrator
calibrator = CameraCalibration()

# Add calibration images
calibrator.add_images('path/to/calibration/images/*.jpg')

# Run calibration
calibrator.calibrate()

# Get calibration results
K, dist_coeffs = calibrator.get_parameters()

# Undistort an image
undistorted = calibrator.undistort_image(image)
```

## Results

### Original Checkerboard Images
<p align="center">
  <table>
    <tr>
      <td> <img src="media/output_img_1.jpg" alt="Checkerboard 1" style="width: 250px;"/> </td>
      <td> <img src="media/output_img_2.jpg" alt="Checkerboard 2" style="width: 250px;"/> </td>
    </tr>
    <tr>
      <td align="center">Checkerboard 1</td>
      <td align="center">Checkerboard 2</td>
    </tr>
  </table>
</p>

### Corner Detection and Reprojection
<p align="center">
  <table>
    <tr>
      <td> <img src="media/Reprojection_Visualization_1.jpg" alt="Reprojection 1" style="width: 250px;"/> </td>
      <td> <img src="media/Reprojection_Visualization_2.jpg" alt="Reprojection 2" style="width: 250px;"/> </td>
    </tr>
    <tr>
      <td align="center">Reprojection 1</td>
      <td align="center">Reprojection 2</td>
    </tr>
  </table>
</p>

### Undistorted Results
<p align="center">
  <table>
    <tr>
      <td> <img src="media/Undistorted_Reprojection_Visualization_1.jpg" alt="Undistorted 1" style="width: 250px;"/> </td>
      <td> <img src="media/Undistorted_Reprojection_Visualization_2.jpg" alt="Undistorted 2" style="width: 250px;"/> </td>
    </tr>
    <tr>
      <td align="center">Undistorted Image 1</td>
      <td align="center">Undistorted Image 2</td>
    </tr>
  </table>
</p>

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this implementation in your research, please cite:
```bibtex
@article{zhang2000flexible,
  title={A flexible new technique for camera calibration},
  author={Zhang, Zhengyou},
  journal={IEEE Transactions on pattern analysis and machine intelligence},
  volume={22},
  number={11},
  pages={1330--1334},
  year={2000},
  publisher={IEEE}
}
```
