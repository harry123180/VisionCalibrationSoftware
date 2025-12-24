"""
TSIC/CR-ICS01: 相機與手臂整合控制教學軟體

A professional camera calibration tool for computing intrinsic and extrinsic
parameters using checkerboard patterns.

開發單位: TSIC
"""

__version__ = "1.0.0"
__author__ = "TSIC"

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
