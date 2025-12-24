"""
Camera intrinsic calibration.

This module provides camera intrinsic parameter estimation
using multiple checkerboard images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from numpy.typing import NDArray

from vision_calib.core.types import (
    CalibrationResult,
    CameraIntrinsic,
    CheckerboardConfig,
    InsufficientImagesError,
    CornerDetectionError,
)
from vision_calib.core.corner_detector import CornerDetector, CornerDetectionResult
from vision_calib.io.image_loader import ImageLoader
from vision_calib.utils.logging import get_logger

logger = get_logger("core.intrinsic")


# Minimum number of images required for calibration
MIN_IMAGES_FOR_CALIBRATION = 3


@dataclass
class IntrinsicCalibrationConfig:
    """Configuration for intrinsic calibration.

    Attributes:
        checkerboard: Checkerboard pattern configuration.
        fix_principal_point: Fix principal point at image center.
        fix_aspect_ratio: Fix fx/fy aspect ratio.
        zero_tangent_dist: Assume zero tangential distortion.
        use_rational_model: Use rational distortion model (more coefficients).
    """
    checkerboard: CheckerboardConfig
    fix_principal_point: bool = False
    fix_aspect_ratio: bool = False
    zero_tangent_dist: bool = False
    use_rational_model: bool = False

    def get_calibration_flags(self) -> int:
        """Get OpenCV calibration flags from config."""
        flags = 0
        if self.fix_principal_point:
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        if self.fix_aspect_ratio:
            flags |= cv2.CALIB_FIX_ASPECT_RATIO
        if self.zero_tangent_dist:
            flags |= cv2.CALIB_ZERO_TANGENT_DIST
        if self.use_rational_model:
            flags |= cv2.CALIB_RATIONAL_MODEL
        return flags


class IntrinsicCalibrator:
    """Camera intrinsic parameter calibrator.

    Estimates camera intrinsic matrix and distortion coefficients
    from multiple checkerboard images.

    Example:
        >>> config = IntrinsicCalibrationConfig(
        ...     checkerboard=CheckerboardConfig(rows=5, cols=7, square_size_mm=30.0)
        ... )
        >>> calibrator = IntrinsicCalibrator(config)
        >>>
        >>> # Add images
        >>> for path in image_paths:
        ...     calibrator.add_image(path)
        >>>
        >>> # Run calibration
        >>> result = calibrator.calibrate()
        >>> print(f"Reprojection error: {result.intrinsic.reprojection_error:.4f}")
    """

    def __init__(self, config: IntrinsicCalibrationConfig):
        """Initialize the calibrator.

        Args:
            config: Calibration configuration.
        """
        self.config = config
        self._corner_detector = CornerDetector(config.checkerboard)
        self._image_loader = ImageLoader()

        # Storage for calibration data
        self._object_points: list[NDArray[np.float32]] = []
        self._image_points: list[NDArray[np.float32]] = []
        self._image_size: Optional[tuple[int, int]] = None
        self._detection_results: list[CornerDetectionResult] = []

        # Object points (same for all images)
        self._objp = config.checkerboard.generate_object_points()

    def add_image(
        self,
        image: Union[str, Path, NDArray],
    ) -> CornerDetectionResult:
        """Add an image for calibration.

        Args:
            image: Image path or numpy array.

        Returns:
            CornerDetectionResult indicating success/failure.
        """
        result = self._corner_detector.detect(image)
        self._detection_results.append(result)

        if result.success and result.corners is not None:
            self._object_points.append(self._objp)
            self._image_points.append(result.corners)

            # Set image size from first successful detection
            if self._image_size is None:
                self._image_size = result.image_size

            logger.info(
                f"Added image: {result.image_path or 'array'} "
                f"({len(self._image_points)} total)"
            )
        else:
            logger.warning(
                f"Failed to detect corners: {result.image_path or 'array'} - "
                f"{result.error_message}"
            )

        return result

    def add_images(
        self,
        images: list[Union[str, Path, NDArray]],
        progress_callback: Optional[callable] = None,
    ) -> list[CornerDetectionResult]:
        """Add multiple images for calibration.

        Args:
            images: List of image paths or arrays.
            progress_callback: Optional callback(current, total, message).

        Returns:
            List of CornerDetectionResult for each image.
        """
        results = []
        total = len(images)

        for i, image in enumerate(images):
            if progress_callback:
                name = image if isinstance(image, (str, Path)) else f"image_{i}"
                progress_callback(i, total, f"Processing: {name}")

            result = self.add_image(image)
            results.append(result)

        if progress_callback:
            progress_callback(total, total, "Processing complete")

        return results

    @property
    def num_valid_images(self) -> int:
        """Number of images with successfully detected corners."""
        return len(self._image_points)

    @property
    def can_calibrate(self) -> bool:
        """Check if enough images are available for calibration."""
        return self.num_valid_images >= MIN_IMAGES_FOR_CALIBRATION

    def clear(self) -> None:
        """Clear all added images and reset calibrator."""
        self._object_points.clear()
        self._image_points.clear()
        self._detection_results.clear()
        self._image_size = None
        logger.info("Calibrator cleared")

    def calibrate(
        self,
        progress_callback: Optional[callable] = None,
    ) -> CalibrationResult:
        """Run camera calibration.

        Args:
            progress_callback: Optional callback(current, total, message).

        Returns:
            CalibrationResult with intrinsic parameters.

        Raises:
            InsufficientImagesError: If not enough valid images.
            CornerDetectionError: If no corners were detected.
        """
        if not self.can_calibrate:
            raise InsufficientImagesError(
                f"Need at least {MIN_IMAGES_FOR_CALIBRATION} images, "
                f"have {self.num_valid_images}"
            )

        if self._image_size is None:
            raise CornerDetectionError("No image size available (no successful detections)")

        if progress_callback:
            progress_callback(0, 100, "Starting calibration...")

        logger.info(
            f"Starting calibration with {self.num_valid_images} images, "
            f"image size: {self._image_size}"
        )

        # Get calibration flags
        flags = self.config.get_calibration_flags()

        if progress_callback:
            progress_callback(10, 100, "Running OpenCV calibration...")

        # Run OpenCV calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self._object_points,
            self._image_points,
            self._image_size,
            None,
            None,
            flags=flags,
        )

        if progress_callback:
            progress_callback(80, 100, "Computing per-image errors...")

        # Compute per-image reprojection errors
        per_image_errors = []
        for i in range(len(self._object_points)):
            projected, _ = cv2.projectPoints(
                self._object_points[i],
                rvecs[i],
                tvecs[i],
                camera_matrix,
                dist_coeffs,
            )
            error = cv2.norm(
                self._image_points[i], projected, cv2.NORM_L2
            ) / len(projected)
            per_image_errors.append(error)

        if progress_callback:
            progress_callback(100, 100, "Calibration complete")

        logger.info(f"Calibration complete, RMS error: {ret:.4f} pixels")

        # Create result
        intrinsic = CameraIntrinsic(
            camera_matrix=camera_matrix,
            distortion_coeffs=dist_coeffs,
            image_size=self._image_size,
            reprojection_error=ret,
        )

        return CalibrationResult(
            intrinsic=intrinsic,
            checkerboard_config=self.config.checkerboard,
            num_images_used=self.num_valid_images,
            per_image_errors=per_image_errors,
        )

    def get_detection_results(self) -> list[CornerDetectionResult]:
        """Get all corner detection results."""
        return self._detection_results.copy()

    def get_successful_detections(self) -> list[CornerDetectionResult]:
        """Get only successful corner detections."""
        return [r for r in self._detection_results if r.success]


def calibrate_from_images(
    image_paths: list[Union[str, Path]],
    checkerboard_rows: int,
    checkerboard_cols: int,
    square_size_mm: float,
    progress_callback: Optional[callable] = None,
) -> CalibrationResult:
    """Convenience function for quick calibration.

    Args:
        image_paths: List of calibration image paths.
        checkerboard_rows: Number of inner corner rows.
        checkerboard_cols: Number of inner corner columns.
        square_size_mm: Size of each square in mm.
        progress_callback: Optional progress callback.

    Returns:
        CalibrationResult with intrinsic parameters.
    """
    config = IntrinsicCalibrationConfig(
        checkerboard=CheckerboardConfig(
            rows=checkerboard_rows,
            cols=checkerboard_cols,
            square_size_mm=square_size_mm,
        )
    )

    calibrator = IntrinsicCalibrator(config)
    calibrator.add_images(image_paths, progress_callback)
    return calibrator.calibrate()
