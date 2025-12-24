"""
Core data types for camera calibration.

This module defines the fundamental data structures used throughout
the vision-calib library, including camera parameters and calibration results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class CheckerboardConfig:
    """Configuration for checkerboard calibration target.

    Attributes:
        rows: Number of inner corners in the vertical direction.
        cols: Number of inner corners in the horizontal direction.
        square_size_mm: Size of each square in millimeters.

    Example:
        A standard 8x6 checkerboard (with 7x5 inner corners):
        >>> config = CheckerboardConfig(rows=5, cols=7, square_size_mm=30.0)
    """
    rows: int
    cols: int
    square_size_mm: float

    def __post_init__(self) -> None:
        if self.rows < 2:
            raise ValueError(f"rows must be >= 2, got {self.rows}")
        if self.cols < 2:
            raise ValueError(f"cols must be >= 2, got {self.cols}")
        if self.square_size_mm <= 0:
            raise ValueError(f"square_size_mm must be > 0, got {self.square_size_mm}")

    @property
    def pattern_size(self) -> tuple[int, int]:
        """OpenCV-compatible pattern size (cols, rows)."""
        return (self.cols, self.rows)

    @property
    def num_corners(self) -> int:
        """Total number of inner corners."""
        return self.rows * self.cols

    def generate_object_points(self) -> NDArray[np.float32]:
        """Generate 3D object points for the checkerboard.

        Returns:
            Array of shape (rows*cols, 3) with Z=0 for all points.
        """
        objp = np.zeros((self.num_corners, 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
        objp *= self.square_size_mm
        return objp


@dataclass
class CameraIntrinsic:
    """Camera intrinsic parameters.

    Attributes:
        camera_matrix: 3x3 camera matrix K containing fx, fy, cx, cy.
        distortion_coeffs: Distortion coefficients [k1, k2, p1, p2, k3, ...].
        image_size: Image dimensions as (width, height).
        reprojection_error: RMS reprojection error in pixels.
    """
    camera_matrix: NDArray[np.float64]
    distortion_coeffs: NDArray[np.float64]
    image_size: tuple[int, int]
    reprojection_error: float = 0.0

    def __post_init__(self) -> None:
        # Validate camera matrix shape
        if self.camera_matrix.shape != (3, 3):
            raise ValueError(
                f"camera_matrix must be 3x3, got {self.camera_matrix.shape}"
            )
        # Ensure correct dtype
        self.camera_matrix = self.camera_matrix.astype(np.float64)
        self.distortion_coeffs = self.distortion_coeffs.astype(np.float64).ravel()

    @property
    def fx(self) -> float:
        """Focal length in x direction (pixels)."""
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        """Focal length in y direction (pixels)."""
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        """Principal point x coordinate (pixels)."""
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        """Principal point y coordinate (pixels)."""
        return float(self.camera_matrix[1, 2])

    @property
    def k1(self) -> float:
        """First radial distortion coefficient."""
        return float(self.distortion_coeffs[0]) if len(self.distortion_coeffs) > 0 else 0.0

    @property
    def k2(self) -> float:
        """Second radial distortion coefficient."""
        return float(self.distortion_coeffs[1]) if len(self.distortion_coeffs) > 1 else 0.0

    @property
    def p1(self) -> float:
        """First tangential distortion coefficient."""
        return float(self.distortion_coeffs[2]) if len(self.distortion_coeffs) > 2 else 0.0

    @property
    def p2(self) -> float:
        """Second tangential distortion coefficient."""
        return float(self.distortion_coeffs[3]) if len(self.distortion_coeffs) > 3 else 0.0

    @property
    def k3(self) -> float:
        """Third radial distortion coefficient."""
        return float(self.distortion_coeffs[4]) if len(self.distortion_coeffs) > 4 else 0.0

    def undistort_points(
        self, points: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Undistort 2D image points.

        Args:
            points: Array of shape (N, 2) with distorted image coordinates.

        Returns:
            Array of shape (N, 2) with undistorted coordinates.
        """
        points = points.reshape(-1, 1, 2).astype(np.float64)
        undistorted = cv2.undistortPoints(
            points, self.camera_matrix, self.distortion_coeffs, P=self.camera_matrix
        )
        return undistorted.reshape(-1, 2)

    def undistort_image(self, image: NDArray) -> NDArray:
        """Undistort an image.

        Args:
            image: Input image (distorted).

        Returns:
            Undistorted image.
        """
        return cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)


@dataclass
class CameraExtrinsic:
    """Camera extrinsic parameters (pose).

    Attributes:
        rotation_vector: 3x1 Rodrigues rotation vector.
        translation_vector: 3x1 translation vector.
    """
    rotation_vector: NDArray[np.float64]
    translation_vector: NDArray[np.float64]

    def __post_init__(self) -> None:
        # Ensure correct shapes and dtypes
        self.rotation_vector = self.rotation_vector.astype(np.float64).reshape(3, 1)
        self.translation_vector = self.translation_vector.astype(np.float64).reshape(3, 1)

    @property
    def rotation_matrix(self) -> NDArray[np.float64]:
        """3x3 rotation matrix converted from rotation vector."""
        R, _ = cv2.Rodrigues(self.rotation_vector)
        return R

    @property
    def transformation_matrix(self) -> NDArray[np.float64]:
        """4x4 homogeneous transformation matrix [R|t]."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self.translation_vector.ravel()
        return T

    @property
    def camera_position(self) -> NDArray[np.float64]:
        """Camera position in world coordinates."""
        R = self.rotation_matrix
        t = self.translation_vector.ravel()
        return -R.T @ t

    @classmethod
    def from_rotation_matrix(
        cls,
        rotation_matrix: NDArray[np.float64],
        translation_vector: NDArray[np.float64],
    ) -> CameraExtrinsic:
        """Create from rotation matrix instead of rotation vector.

        Args:
            rotation_matrix: 3x3 rotation matrix.
            translation_vector: 3x1 translation vector.

        Returns:
            CameraExtrinsic instance.
        """
        rvec, _ = cv2.Rodrigues(rotation_matrix)
        return cls(rotation_vector=rvec, translation_vector=translation_vector)


@dataclass
class CalibrationResult:
    """Complete camera calibration result.

    Contains intrinsic parameters, optional extrinsic parameters,
    and metadata about the calibration process.
    """
    intrinsic: CameraIntrinsic
    extrinsic: Optional[CameraExtrinsic] = None

    # Calibration metadata
    timestamp: datetime = field(default_factory=datetime.now)
    checkerboard_config: Optional[CheckerboardConfig] = None
    num_images_used: int = 0
    software_version: str = "1.0.0"
    notes: str = ""

    # Per-image data (for detailed analysis)
    per_image_errors: Optional[list[float]] = None

    @property
    def has_extrinsic(self) -> bool:
        """Check if extrinsic parameters are available."""
        return self.extrinsic is not None

    def summary(self) -> str:
        """Generate a human-readable summary of the calibration."""
        lines = [
            "=" * 50,
            "Camera Calibration Result",
            "=" * 50,
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Software Version: {self.software_version}",
            "",
            "Intrinsic Parameters:",
            f"  Image Size: {self.intrinsic.image_size[0]} x {self.intrinsic.image_size[1]}",
            f"  Focal Length: fx={self.intrinsic.fx:.2f}, fy={self.intrinsic.fy:.2f}",
            f"  Principal Point: cx={self.intrinsic.cx:.2f}, cy={self.intrinsic.cy:.2f}",
            f"  Reprojection Error: {self.intrinsic.reprojection_error:.4f} pixels",
            "",
            "Distortion Coefficients:",
            f"  k1={self.intrinsic.k1:.6f}, k2={self.intrinsic.k2:.6f}",
            f"  p1={self.intrinsic.p1:.6f}, p2={self.intrinsic.p2:.6f}",
            f"  k3={self.intrinsic.k3:.6f}",
        ]

        if self.checkerboard_config:
            lines.extend([
                "",
                "Checkerboard Configuration:",
                f"  Pattern: {self.checkerboard_config.cols} x {self.checkerboard_config.rows}",
                f"  Square Size: {self.checkerboard_config.square_size_mm} mm",
            ])

        if self.has_extrinsic:
            lines.extend([
                "",
                "Extrinsic Parameters:",
                f"  Rotation Vector: {self.extrinsic.rotation_vector.ravel()}",
                f"  Translation Vector: {self.extrinsic.translation_vector.ravel()}",
                f"  Camera Position: {self.extrinsic.camera_position}",
            ])

        lines.extend([
            "",
            f"Images Used: {self.num_images_used}",
            "=" * 50,
        ])

        return "\n".join(lines)


# Custom exceptions
class CalibrationError(Exception):
    """Base exception for calibration errors."""
    pass


class InsufficientImagesError(CalibrationError):
    """Raised when there are not enough images for calibration."""
    pass


class CornerDetectionError(CalibrationError):
    """Raised when corner detection fails."""
    pass


class InvalidParameterError(CalibrationError):
    """Raised when invalid parameters are provided."""
    pass


class FileFormatError(CalibrationError):
    """Raised when file format is invalid or unsupported."""
    pass
