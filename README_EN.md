# TSIC/CR-ICS01 Camera and Robot Arm Integration Control Software

English | [繁體中文](README.md)

Professional camera calibration toolkit using checkerboard patterns.

## Features

- **Intrinsic Calibration**: Compute camera matrix and distortion coefficients from multiple checkerboard images
- **Extrinsic Calibration**: Estimate camera pose using various PnP algorithms (P3P, AP3P, EPnP, IPPE, etc.)
- **Coordinate Transform**: Convert between pixel, camera, and world coordinates
- **Chessboard Coordinate Generation**: Auto-generate world coordinates based on robot arm coordinate system
- **Real-time Visualization**: Coordinate system alignment visualization with zoom and pan support
- **Multiple Export Formats**:
  - HDF5 (.h5) - Cross-platform scientific data format
  - MAT (.mat) - MATLAB/Octave compatible
  - JSON (.json) - Human-readable format

## Installation

```bash
# Install from source
git clone https://github.com/TSIC-LAB/VisionCalibrationSoftware.git
cd VisionCalibrationSoftware
pip install -e .
```

### Dependencies

- Python >= 3.10
- NumPy >= 1.24.0
- OpenCV >= 4.8.0
- PySide6 >= 6.5.0
- h5py >= 3.9.0
- SciPy >= 1.11.0
- Matplotlib >= 3.7.0

## Quick Start

### GUI Application

```bash
python main.py
```

### Python API

```python
from vision_calib.core.intrinsic import IntrinsicCalibrator, IntrinsicCalibrationConfig
from vision_calib.core.types import CheckerboardConfig
from vision_calib.io import CalibrationFile

# Configure calibration
config = IntrinsicCalibrationConfig(
    checkerboard=CheckerboardConfig(rows=12, cols=17, square_size_mm=10.0)
)

# Create calibrator and add images
calibrator = IntrinsicCalibrator(config)
for image_path in image_paths:
    calibrator.add_image(image_path)

# Run calibration
result = calibrator.calibrate()
print(result.summary())

# Save in multiple formats
CalibrationFile.save("calibration.h5", result)   # HDF5
CalibrationFile.save("calibration.mat", result)  # MAT (Octave/MATLAB)
CalibrationFile.save("calibration.json", result) # JSON
```

### Octave/MATLAB Usage

```matlab
% Load from MAT file
data = load('calibration.mat');
K = data.camera_matrix;
D = data.distortion_coeffs;

% Or load from HDF5
K = h5read('calibration.h5', '/intrinsic/camera_matrix');
D = h5read('calibration.h5', '/intrinsic/distortion_coeffs');
```

## Extrinsic Calibration Algorithms

This software supports multiple PnP algorithms:

| Algorithm | Min Points | Description |
|-----------|------------|-------------|
| PnP Iterative | ≥4 | Most versatile, recommended |
| EPnP | ≥4 | High efficiency, good for many points |
| P3P | =3 | Only needs 3 points, may have multiple solutions |
| AP3P | ≥3 | Improved P3P, better numerical stability |
| IPPE | ≥4 | For planar objects |
| IPPE_SQUARE | ≥4 | For square calibration boards |

## Export Format Details

### HDF5 Structure

```
calibration.h5
├── /intrinsic/
│   ├── camera_matrix      (3,3) float64
│   ├── distortion_coeffs  (n,) float64
│   ├── image_size         (2,) int32
│   └── reprojection_error scalar float64
├── /extrinsic/            (optional)
│   ├── rotation_vector    (3,) float64
│   ├── translation_vector (3,) float64
│   └── rotation_matrix    (3,3) float64
└── /metadata/
    ├── timestamp          string
    ├── checkerboard_*     various
    └── software_version   string
```

### MAT Variables

| Variable | Type | Description |
|----------|------|-------------|
| `camera_matrix` | (3,3) double | Camera intrinsic matrix K |
| `distortion_coeffs` | (1,n) double | Distortion coefficients |
| `image_size` | (1,2) double | [width, height] |
| `reprojection_error` | scalar | RMS error in pixels |
| `rotation_vector` | (3,1) double | Rotation vector (optional) |
| `translation_vector` | (3,1) double | Translation vector (optional) |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Developer

TSIC (Taiwan Smart Industrial Control)

## Acknowledgments

- OpenCV for the underlying calibration algorithms
- Zhang's calibration method for the theoretical foundation
