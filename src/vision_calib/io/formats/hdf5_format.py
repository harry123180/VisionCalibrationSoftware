"""
HDF5 format support for calibration data.

HDF5 is a cross-platform, widely supported format for scientific data.
It can be read by Python (h5py), MATLAB, Octave, Julia, R, and many other tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import numpy as np

from vision_calib.core.types import (
    CalibrationResult,
    CameraIntrinsic,
    CameraExtrinsic,
    CheckerboardConfig,
    FileFormatError,
)
from vision_calib.utils.logging import get_logger

logger = get_logger("io.formats.hdf5")


class HDF5Format:
    """HDF5 format reader/writer for calibration data.

    File structure:
        /intrinsic/
            camera_matrix      (3,3) float64
            distortion_coeffs  (n,) float64
            image_size         (2,) int32
            reprojection_error scalar float64
        /extrinsic/           (optional)
            rotation_vector    (3,) float64
            translation_vector (3,) float64
        /metadata/
            timestamp          string
            checkerboard_rows  scalar int32
            checkerboard_cols  scalar int32
            square_size_mm     scalar float64
            num_images_used    scalar int32
            software_version   string
            notes              string
        /per_image_errors/    (optional)
            errors             (n,) float64

    Example (Python):
        >>> import h5py
        >>> with h5py.File('calibration.h5', 'r') as f:
        ...     K = f['/intrinsic/camera_matrix'][:]
        ...     D = f['/intrinsic/distortion_coeffs'][:]

    Example (Octave/MATLAB):
        K = h5read('calibration.h5', '/intrinsic/camera_matrix');
        D = h5read('calibration.h5', '/intrinsic/distortion_coeffs');
    """

    EXTENSION = ".h5"
    VERSION = "1.0"

    @classmethod
    def save(
        cls,
        path: Union[str, Path],
        result: CalibrationResult,
        compression: str = "gzip",
    ) -> None:
        """Save calibration result to HDF5 file.

        Args:
            path: Output file path.
            result: Calibration result to save.
            compression: Compression algorithm ('gzip', 'lzf', or None).
        """
        path = Path(path)
        if path.suffix.lower() != cls.EXTENSION:
            path = path.with_suffix(cls.EXTENSION)

        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving calibration to HDF5: {path}")

        with h5py.File(path, "w") as f:
            # File-level attributes
            f.attrs["format_version"] = cls.VERSION
            f.attrs["format_type"] = "vision-calib"

            # Intrinsic group
            intrinsic_grp = f.create_group("intrinsic")
            intrinsic_grp.create_dataset(
                "camera_matrix",
                data=result.intrinsic.camera_matrix,
                compression=compression,
            )
            intrinsic_grp.create_dataset(
                "distortion_coeffs",
                data=result.intrinsic.distortion_coeffs,
                compression=compression,
            )
            intrinsic_grp.create_dataset(
                "image_size",
                data=np.array(result.intrinsic.image_size, dtype=np.int32),
            )
            intrinsic_grp.attrs["reprojection_error"] = result.intrinsic.reprojection_error

            # Extrinsic group (optional)
            if result.extrinsic is not None:
                extrinsic_grp = f.create_group("extrinsic")
                extrinsic_grp.create_dataset(
                    "rotation_vector",
                    data=result.extrinsic.rotation_vector.ravel(),
                )
                extrinsic_grp.create_dataset(
                    "translation_vector",
                    data=result.extrinsic.translation_vector.ravel(),
                )
                # Also save rotation matrix for convenience
                extrinsic_grp.create_dataset(
                    "rotation_matrix",
                    data=result.extrinsic.rotation_matrix,
                    compression=compression,
                )

            # Metadata group
            metadata_grp = f.create_group("metadata")
            metadata_grp.attrs["timestamp"] = result.timestamp.isoformat()
            metadata_grp.attrs["num_images_used"] = result.num_images_used
            metadata_grp.attrs["software_version"] = result.software_version
            metadata_grp.attrs["notes"] = result.notes

            if result.checkerboard_config is not None:
                metadata_grp.attrs["checkerboard_rows"] = result.checkerboard_config.rows
                metadata_grp.attrs["checkerboard_cols"] = result.checkerboard_config.cols
                metadata_grp.attrs["square_size_mm"] = result.checkerboard_config.square_size_mm

            # Per-image errors (optional)
            if result.per_image_errors is not None:
                f.create_dataset(
                    "per_image_errors",
                    data=np.array(result.per_image_errors),
                    compression=compression,
                )

        logger.info(f"Saved calibration to: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> CalibrationResult:
        """Load calibration result from HDF5 file.

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

        logger.info(f"Loading calibration from HDF5: {path}")

        with h5py.File(path, "r") as f:
            # Verify format
            format_type = f.attrs.get("format_type", "")
            if format_type != "vision-calib":
                logger.warning(f"Unknown format type: {format_type}")

            # Load intrinsic
            if "intrinsic" not in f:
                raise FileFormatError("Missing intrinsic group in HDF5 file")

            intrinsic_grp = f["intrinsic"]
            intrinsic = CameraIntrinsic(
                camera_matrix=intrinsic_grp["camera_matrix"][:],
                distortion_coeffs=intrinsic_grp["distortion_coeffs"][:],
                image_size=tuple(intrinsic_grp["image_size"][:]),
                reprojection_error=intrinsic_grp.attrs.get("reprojection_error", 0.0),
            )

            # Load extrinsic (optional)
            extrinsic = None
            if "extrinsic" in f:
                extrinsic_grp = f["extrinsic"]
                extrinsic = CameraExtrinsic(
                    rotation_vector=extrinsic_grp["rotation_vector"][:],
                    translation_vector=extrinsic_grp["translation_vector"][:],
                )

            # Load metadata
            checkerboard_config = None
            metadata_grp = f.get("metadata", {})

            if isinstance(metadata_grp, h5py.Group):
                attrs = metadata_grp.attrs
                if "checkerboard_rows" in attrs and "checkerboard_cols" in attrs:
                    checkerboard_config = CheckerboardConfig(
                        rows=int(attrs["checkerboard_rows"]),
                        cols=int(attrs["checkerboard_cols"]),
                        square_size_mm=float(attrs.get("square_size_mm", 0.0)),
                    )

                timestamp_str = attrs.get("timestamp", "")
                num_images_used = int(attrs.get("num_images_used", 0))
                software_version = attrs.get("software_version", "unknown")
                notes = attrs.get("notes", "")
            else:
                timestamp_str = ""
                num_images_used = 0
                software_version = "unknown"
                notes = ""

            # Load per-image errors (optional)
            per_image_errors = None
            if "per_image_errors" in f:
                per_image_errors = list(f["per_image_errors"][:])

            # Parse timestamp
            from datetime import datetime
            try:
                timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
            except ValueError:
                timestamp = datetime.now()

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
        """Check if a file is a valid vision-calib HDF5 file.

        Args:
            path: File path to check.

        Returns:
            True if file is valid, False otherwise.
        """
        path = Path(path)
        if not path.exists():
            return False

        try:
            with h5py.File(path, "r") as f:
                return "intrinsic" in f and "camera_matrix" in f["intrinsic"]
        except (OSError, KeyError):
            return False
