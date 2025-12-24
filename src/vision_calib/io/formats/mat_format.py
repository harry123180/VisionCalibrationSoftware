"""
MAT format support for calibration data.

MAT format is native to MATLAB and fully compatible with GNU Octave.
This implementation uses scipy.io for reading/writing MAT files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
import scipy.io as sio

from vision_calib.core.types import (
    CalibrationResult,
    CameraIntrinsic,
    CameraExtrinsic,
    CheckerboardConfig,
    FileFormatError,
)
from vision_calib.utils.logging import get_logger

logger = get_logger("io.formats.mat")


class MATFormat:
    """MAT format reader/writer for calibration data.

    Creates MAT files compatible with MATLAB and GNU Octave.

    Variables in the MAT file:
        camera_matrix       (3,3) double - Camera intrinsic matrix K
        distortion_coeffs   (1,n) double - Distortion coefficients
        image_size          (1,2) double - Image dimensions [width, height]
        reprojection_error  (1,1) double - RMS reprojection error
        rotation_vector     (3,1) double - Rotation vector (if extrinsic available)
        translation_vector  (3,1) double - Translation vector (if extrinsic available)
        rotation_matrix     (3,3) double - Rotation matrix (if extrinsic available)
        checkerboard_size   (1,2) double - [cols, rows] (if available)
        square_size_mm      (1,1) double - Square size in mm (if available)
        num_images_used     (1,1) double - Number of calibration images
        timestamp           string       - Calibration timestamp
        software_version    string       - Software version

    Example (Octave/MATLAB):
        >> data = load('calibration.mat');
        >> K = data.camera_matrix;
        >> D = data.distortion_coeffs;
        >> disp(['Focal length: ', num2str(K(1,1))]);
    """

    EXTENSION = ".mat"
    VERSION = "1.0"

    @classmethod
    def save(
        cls,
        path: Union[str, Path],
        result: CalibrationResult,
        format_version: str = "7.3",
    ) -> None:
        """Save calibration result to MAT file.

        Args:
            path: Output file path.
            result: Calibration result to save.
            format_version: MAT file format version ('5', '7.3').
                '7.3' uses HDF5 internally and supports large arrays.
                '5' is more compatible with older MATLAB versions.
        """
        path = Path(path)
        if path.suffix.lower() != cls.EXTENSION:
            path = path.with_suffix(cls.EXTENSION)

        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving calibration to MAT: {path}")

        # Build data dictionary
        mdict: dict[str, Any] = {
            # Intrinsic parameters
            "camera_matrix": result.intrinsic.camera_matrix,
            "distortion_coeffs": result.intrinsic.distortion_coeffs.reshape(1, -1),
            "image_size": np.array(result.intrinsic.image_size, dtype=np.float64).reshape(1, 2),
            "reprojection_error": np.array([[result.intrinsic.reprojection_error]]),
            # Metadata
            "num_images_used": np.array([[result.num_images_used]]),
            "timestamp": result.timestamp.isoformat(),
            "software_version": result.software_version,
            "format_version": cls.VERSION,
        }

        # Extrinsic parameters (optional)
        if result.extrinsic is not None:
            mdict["rotation_vector"] = result.extrinsic.rotation_vector
            mdict["translation_vector"] = result.extrinsic.translation_vector
            mdict["rotation_matrix"] = result.extrinsic.rotation_matrix

        # Checkerboard config (optional)
        if result.checkerboard_config is not None:
            mdict["checkerboard_size"] = np.array([
                [result.checkerboard_config.cols, result.checkerboard_config.rows]
            ], dtype=np.float64)
            mdict["square_size_mm"] = np.array([[result.checkerboard_config.square_size_mm]])

        # Per-image errors (optional)
        if result.per_image_errors is not None:
            mdict["per_image_errors"] = np.array(result.per_image_errors).reshape(1, -1)

        # Notes (optional)
        if result.notes:
            mdict["notes"] = result.notes

        # Save file
        # Note: do_compression=True is only supported for format 7.3 (HDF5-based)
        sio.savemat(path, mdict, do_compression=True)

        logger.info(f"Saved calibration to: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> CalibrationResult:
        """Load calibration result from MAT file.

        Args:
            path: Input file path.

        Returns:
            CalibrationResult loaded from file.

        Raises:
            FileFormatError: If file format is invalid.
        """
        path = Path(path)

        if not path.exists():
            raise FileFormatError(f"File not found: {path}")

        logger.info(f"Loading calibration from MAT: {path}")

        try:
            # Load MAT file
            # squeeze_me=True: Convert single-element arrays to scalars
            # struct_as_record=False: Return structs as objects
            data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        except Exception as e:
            raise FileFormatError(f"Failed to load MAT file: {e}")

        # Validate required fields
        if "camera_matrix" not in data:
            raise FileFormatError("Missing camera_matrix in MAT file")
        if "distortion_coeffs" not in data:
            raise FileFormatError("Missing distortion_coeffs in MAT file")

        # Load intrinsic
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        distortion_coeffs = np.array(data["distortion_coeffs"], dtype=np.float64).ravel()

        # Image size (may be stored as [width, height] or [height, width])
        if "image_size" in data:
            image_size = tuple(int(x) for x in np.array(data["image_size"]).ravel()[:2])
        else:
            image_size = (0, 0)

        # Reprojection error
        reprojection_error = 0.0
        if "reprojection_error" in data:
            reprojection_error = float(np.array(data["reprojection_error"]).ravel()[0])

        intrinsic = CameraIntrinsic(
            camera_matrix=camera_matrix,
            distortion_coeffs=distortion_coeffs,
            image_size=image_size,
            reprojection_error=reprojection_error,
        )

        # Load extrinsic (optional)
        extrinsic = None
        if "rotation_vector" in data and "translation_vector" in data:
            extrinsic = CameraExtrinsic(
                rotation_vector=np.array(data["rotation_vector"], dtype=np.float64),
                translation_vector=np.array(data["translation_vector"], dtype=np.float64),
            )

        # Load checkerboard config (optional)
        checkerboard_config = None
        if "checkerboard_size" in data:
            cb_size = np.array(data["checkerboard_size"]).ravel()
            if len(cb_size) >= 2:
                square_size = 0.0
                if "square_size_mm" in data:
                    square_size = float(np.array(data["square_size_mm"]).ravel()[0])
                checkerboard_config = CheckerboardConfig(
                    cols=int(cb_size[0]),
                    rows=int(cb_size[1]),
                    square_size_mm=square_size,
                )

        # Load metadata
        num_images_used = 0
        if "num_images_used" in data:
            num_images_used = int(np.array(data["num_images_used"]).ravel()[0])

        software_version = str(data.get("software_version", "unknown"))
        notes = str(data.get("notes", ""))

        # Parse timestamp
        from datetime import datetime
        timestamp_str = str(data.get("timestamp", ""))
        try:
            timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
        except ValueError:
            timestamp = datetime.now()

        # Per-image errors (optional)
        per_image_errors = None
        if "per_image_errors" in data:
            per_image_errors = list(np.array(data["per_image_errors"]).ravel())

        return CalibrationResult(
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            timestamp=timestamp,
            checkerboard_config=checkerboard_config,
            num_images_used=num_images_used,
            software_version=software_version,
            notes=notes,
            per_image_errors=per_image_errors,
        )

    @classmethod
    def is_valid_file(cls, path: Union[str, Path]) -> bool:
        """Check if a file is a valid calibration MAT file.

        Args:
            path: File path to check.

        Returns:
            True if file is valid, False otherwise.
        """
        path = Path(path)
        if not path.exists():
            return False

        try:
            data = sio.loadmat(path, squeeze_me=True)
            return "camera_matrix" in data and "distortion_coeffs" in data
        except Exception:
            return False
