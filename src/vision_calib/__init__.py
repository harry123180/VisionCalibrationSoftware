"""
vision-calib: Camera Calibration Toolkit

A professional camera calibration tool for computing intrinsic and extrinsic
parameters using checkerboard patterns.

License: Apache 2.0
"""

__version__ = "1.0.0"
__author__ = "vision-calib contributors"

from vision_calib.core.types import (
    CameraIntrinsic,
    CameraExtrinsic,
    CalibrationResult,
    CheckerboardConfig,
)

__all__ = [
    "CameraIntrinsic",
    "CameraExtrinsic",
    "CalibrationResult",
    "CheckerboardConfig",
    "__version__",
]
