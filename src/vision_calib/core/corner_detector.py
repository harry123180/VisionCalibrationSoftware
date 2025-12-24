"""
Checkerboard corner detection.

This module provides robust corner detection for checkerboard
calibration patterns using OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from numpy.typing import NDArray

from vision_calib.core.types import CheckerboardConfig, CornerDetectionError
from vision_calib.io.image_loader import ImageLoader
from vision_calib.utils.logging import get_logger

logger = get_logger("core.corner_detector")


@dataclass
class CornerDetectionResult:
    """Result of corner detection on a single image.

    Attributes:
        success: Whether corners were successfully detected.
        corners: Detected corner points (N, 1, 2) or None if failed.
        image_path: Path to the source image.
        image_size: Image dimensions (width, height).
        error_message: Error description if detection failed.
    """
    success: bool
    corners: Optional[NDArray[np.float32]]
    image_path: Optional[Path]
    image_size: tuple[int, int]
    error_message: str = ""

    @property
    def num_corners(self) -> int:
        """Number of detected corners."""
        if self.corners is None:
            return 0
        return len(self.corners)

    def get_corners_2d(self) -> Optional[NDArray[np.float32]]:
        """Get corners as (N, 2) array."""
        if self.corners is None:
            return None
        return self.corners.reshape(-1, 2)


class CornerDetector:
    """Checkerboard corner detector.

    Detects inner corners of a checkerboard pattern in images.
    Uses OpenCV's findChessboardCorners with optional subpixel refinement.

    Example:
        >>> config = CheckerboardConfig(rows=5, cols=7, square_size_mm=30.0)
        >>> detector = CornerDetector(config)
        >>> result = detector.detect("calibration_image.jpg")
        >>> if result.success:
        ...     print(f"Found {result.num_corners} corners")
    """

    # Corner refinement criteria
    SUBPIX_CRITERIA = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,  # max iterations
        0.001,  # epsilon
    )

    # Subpixel refinement window size
    SUBPIX_WINDOW_SIZE = (11, 11)
    SUBPIX_ZERO_ZONE = (-1, -1)

    # Detection flags
    DEFAULT_FLAGS = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )

    def __init__(
        self,
        config: CheckerboardConfig,
        refine_corners: bool = True,
        detection_flags: Optional[int] = None,
    ):
        """Initialize the corner detector.

        Args:
            config: Checkerboard configuration.
            refine_corners: Whether to refine corners with subpixel accuracy.
            detection_flags: OpenCV detection flags (default: adaptive + normalize + fast).
        """
        self.config = config
        self.refine_corners = refine_corners
        self.detection_flags = detection_flags or self.DEFAULT_FLAGS
        self._image_loader = ImageLoader()

    def detect(
        self,
        image: Union[str, Path, NDArray],
    ) -> CornerDetectionResult:
        """Detect checkerboard corners in an image.

        Args:
            image: Image path or numpy array.

        Returns:
            CornerDetectionResult with detection outcome.
        """
        # Load image if path provided
        image_path: Optional[Path] = None
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            loaded = self._image_loader.load(image_path)
            if loaded is None:
                return CornerDetectionResult(
                    success=False,
                    corners=None,
                    image_path=image_path,
                    image_size=(0, 0),
                    error_message=f"Failed to load image: {image_path}",
                )
            image = loaded

        # Get image dimensions
        height, width = image.shape[:2]
        image_size = (width, height)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect corners
        pattern_size = self.config.pattern_size
        success, corners = cv2.findChessboardCorners(
            gray, pattern_size, flags=self.detection_flags
        )

        if not success or corners is None:
            logger.debug(f"Corner detection failed for: {image_path or 'array'}")
            return CornerDetectionResult(
                success=False,
                corners=None,
                image_path=image_path,
                image_size=image_size,
                error_message="No checkerboard pattern found",
            )

        # Verify corner count
        expected_corners = self.config.num_corners
        if len(corners) != expected_corners:
            return CornerDetectionResult(
                success=False,
                corners=None,
                image_path=image_path,
                image_size=image_size,
                error_message=f"Expected {expected_corners} corners, found {len(corners)}",
            )

        # Refine corners with subpixel accuracy
        if self.refine_corners:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                self.SUBPIX_WINDOW_SIZE,
                self.SUBPIX_ZERO_ZONE,
                self.SUBPIX_CRITERIA,
            )

        logger.debug(f"Detected {len(corners)} corners in: {image_path or 'array'}")

        return CornerDetectionResult(
            success=True,
            corners=corners,
            image_path=image_path,
            image_size=image_size,
        )

    def detect_batch(
        self,
        images: list[Union[str, Path, NDArray]],
        progress_callback: Optional[callable] = None,
    ) -> list[CornerDetectionResult]:
        """Detect corners in multiple images.

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

            result = self.detect(image)
            results.append(result)

        if progress_callback:
            progress_callback(total, total, "Detection complete")

        # Log summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"Corner detection: {successful}/{total} images successful")

        return results

    def draw_corners(
        self,
        image: NDArray,
        corners: NDArray[np.float32],
        success: bool = True,
    ) -> NDArray:
        """Draw detected corners on an image.

        Args:
            image: Input image (will be copied).
            corners: Detected corners.
            success: Whether detection was successful (affects color).

        Returns:
            Image with drawn corners.
        """
        output = image.copy()
        cv2.drawChessboardCorners(
            output, self.config.pattern_size, corners, success
        )
        return output

    def visualize_result(
        self,
        image: Union[str, Path, NDArray],
        result: CornerDetectionResult,
    ) -> NDArray:
        """Create a visualization of the detection result.

        Args:
            image: Original image.
            result: Detection result.

        Returns:
            Image with visualization.
        """
        # Load image if needed
        if isinstance(image, (str, Path)):
            loaded = self._image_loader.load(image)
            if loaded is None:
                raise CornerDetectionError(f"Failed to load image: {image}")
            image = loaded

        output = image.copy()

        if result.success and result.corners is not None:
            # Draw corners
            output = self.draw_corners(output, result.corners, True)

            # Add corner indices
            corners_2d = result.get_corners_2d()
            if corners_2d is not None:
                for i, (x, y) in enumerate(corners_2d):
                    cv2.putText(
                        output,
                        str(i),
                        (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),
                        1,
                    )
        else:
            # Draw failure indicator
            h, w = output.shape[:2]
            cv2.putText(
                output,
                "Detection Failed",
                (w // 4, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
            )
            if result.error_message:
                cv2.putText(
                    output,
                    result.error_message,
                    (w // 4, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        return output
