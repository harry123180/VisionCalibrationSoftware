"""
Image loading utilities with Unicode path support.

OpenCV's imread() has issues with non-ASCII paths on Windows.
This module provides a cross-platform solution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from numpy.typing import NDArray

from vision_calib.utils.logging import get_logger

logger = get_logger("io.image_loader")


class ImageLoader:
    """Cross-platform image loader with Unicode path support.

    OpenCV's imread() doesn't handle Unicode paths properly on Windows.
    This class provides a workaround using numpy file reading.

    Example:
        >>> loader = ImageLoader()
        >>> image = loader.load("C:/圖片/test.jpg")
        >>> if image is not None:
        ...     print(f"Loaded image: {image.shape}")
    """

    # Supported image extensions
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def load(
        self,
        path: Union[str, Path],
        flags: int = cv2.IMREAD_COLOR,
    ) -> Optional[NDArray]:
        """Load an image from file.

        Args:
            path: Path to the image file (supports Unicode).
            flags: OpenCV imread flags (default: IMREAD_COLOR).

        Returns:
            Image as numpy array, or None if loading failed.
        """
        path = Path(path)

        if not path.exists():
            logger.error(f"Image file not found: {path}")
            return None

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported image format: {path.suffix}")

        try:
            # Read file as binary
            with open(path, "rb") as f:
                data = f.read()

            # Decode using OpenCV
            nparr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(nparr, flags)

            if image is None:
                logger.error(f"Failed to decode image: {path}")
                return None

            logger.debug(f"Loaded image: {path} ({image.shape})")
            return image

        except OSError as e:
            logger.error(f"Failed to read image file: {path} - {e}")
            return None
        except cv2.error as e:
            logger.error(f"OpenCV error decoding image: {path} - {e}")
            return None

    def load_grayscale(self, path: Union[str, Path]) -> Optional[NDArray]:
        """Load an image as grayscale.

        Args:
            path: Path to the image file.

        Returns:
            Grayscale image as numpy array, or None if loading failed.
        """
        return self.load(path, flags=cv2.IMREAD_GRAYSCALE)

    def load_batch(
        self,
        paths: list[Union[str, Path]],
        flags: int = cv2.IMREAD_COLOR,
    ) -> list[tuple[Path, Optional[NDArray]]]:
        """Load multiple images.

        Args:
            paths: List of image paths.
            flags: OpenCV imread flags.

        Returns:
            List of (path, image) tuples. Image is None if loading failed.
        """
        results = []
        for path in paths:
            path = Path(path)
            image = self.load(path, flags)
            results.append((path, image))
        return results

    @staticmethod
    def save(
        path: Union[str, Path],
        image: NDArray,
        params: Optional[list[int]] = None,
    ) -> bool:
        """Save an image to file (with Unicode path support).

        Args:
            path: Output path.
            image: Image to save.
            params: Optional imwrite parameters.

        Returns:
            True if successful, False otherwise.
        """
        path = Path(path)

        try:
            # Encode image
            ext = path.suffix.lower()
            success, encoded = cv2.imencode(ext, image, params or [])

            if not success:
                logger.error(f"Failed to encode image for: {path}")
                return False

            # Write to file
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                f.write(encoded.tobytes())

            logger.debug(f"Saved image: {path}")
            return True

        except OSError as e:
            logger.error(f"Failed to save image: {path} - {e}")
            return False

    @staticmethod
    def get_image_info(path: Union[str, Path]) -> Optional[dict]:
        """Get image information without loading the full image.

        Args:
            path: Path to the image file.

        Returns:
            Dictionary with image info, or None if failed.
        """
        path = Path(path)
        loader = ImageLoader()
        image = loader.load(path)

        if image is None:
            return None

        info = {
            "path": str(path),
            "filename": path.name,
            "width": image.shape[1],
            "height": image.shape[0],
            "channels": image.shape[2] if len(image.shape) > 2 else 1,
            "dtype": str(image.dtype),
            "size_bytes": path.stat().st_size,
        }

        return info


# Convenience function
def load_image(
    path: Union[str, Path],
    grayscale: bool = False,
) -> Optional[NDArray]:
    """Load an image (convenience function).

    Args:
        path: Path to the image file.
        grayscale: If True, load as grayscale.

    Returns:
        Image as numpy array, or None if loading failed.
    """
    loader = ImageLoader()
    if grayscale:
        return loader.load_grayscale(path)
    return loader.load(path)
