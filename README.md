# vision-calib

Professional camera calibration toolkit using checkerboard patterns.

## Features

- **Intrinsic Calibration**: Compute camera matrix and distortion coefficients from multiple checkerboard images
- **Extrinsic Calibration**: Estimate camera pose using Zhang's method
- **Coordinate Transform**: Convert between pixel, camera, and world coordinates
- **Multiple Export Formats**:
  - HDF5 (.h5) - Cross-platform scientific data format
  - MAT (.mat) - MATLAB/Octave compatible
  - JSON (.json) - Human-readable

## Installation

```bash
# From PyPI (coming soon)
pip install vision-calib

# From source
git clone https://github.com/yourusername/vision-calib.git
cd vision-calib
pip install -e .
```

### Dependencies

- Python >= 3.10
- NumPy >= 1.24.0
- OpenCV >= 4.8.0
- PySide6 >= 6.5.0
- h5py >= 3.9.0
- SciPy >= 1.11.0

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
    checkerboard=CheckerboardConfig(rows=5, cols=7, square_size_mm=30.0)
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

### Octave/MATLAB

```matlab
% Load from MAT file
data = load('calibration.mat');
K = data.camera_matrix;
D = data.distortion_coeffs;

% Or load from HDF5
K = h5read('calibration.h5', '/intrinsic/camera_matrix');
D = h5read('calibration.h5', '/intrinsic/distortion_coeffs');
```

See [examples/octave_example.m](examples/octave_example.m) for more details.

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

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## Acknowledgments

- OpenCV for the underlying calibration algorithms
- Zhang's calibration method for the theoretical foundation
