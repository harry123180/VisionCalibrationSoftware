"""
Unified calibration file interface.

Provides a single entry point for saving/loading calibration data
in multiple formats (HDF5, MAT, JSON).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Union

from vision_calib.core.types import CalibrationResult, FileFormatError
from vision_calib.io.formats.hdf5_format import HDF5Format
from vision_calib.io.formats.mat_format import MATFormat
from vision_calib.io.formats.json_format import JSONFormat
from vision_calib.utils.logging import get_logger

logger = get_logger("io.calibration_file")


class CalibrationFileFormat(Enum):
    """Supported calibration file formats."""
    HDF5 = "hdf5"
    MAT = "mat"
    JSON = "json"

    @classmethod
    def from_extension(cls, ext: str) -> "CalibrationFileFormat":
        """Get format from file extension.

        Args:
            ext: File extension (with or without leading dot).

        Returns:
            CalibrationFileFormat enum value.

        Raises:
            ValueError: If extension is not recognized.
        """
        ext = ext.lower().lstrip(".")
        mapping = {
            "h5": cls.HDF5,
            "hdf5": cls.HDF5,
            "mat": cls.MAT,
            "json": cls.JSON,
        }
        if ext not in mapping:
            raise ValueError(f"Unknown file extension: {ext}")
        return mapping[ext]


class CalibrationFile:
    """Unified interface for calibration file operations.

    Supports multiple formats with automatic format detection.

    Example:
        >>> # Save in different formats
        >>> CalibrationFile.save("calibration.h5", result)   # HDF5
        >>> CalibrationFile.save("calibration.mat", result)  # MAT (Octave)
        >>> CalibrationFile.save("calibration.json", result) # JSON
        >>>
        >>> # Load (format auto-detected from extension)
        >>> result = CalibrationFile.load("calibration.h5")
    """

    # Format handlers
    _handlers = {
        CalibrationFileFormat.HDF5: HDF5Format,
        CalibrationFileFormat.MAT: MATFormat,
        CalibrationFileFormat.JSON: JSONFormat,
    }

    @classmethod
    def save(
        cls,
        path: Union[str, Path],
        result: CalibrationResult,
        format: CalibrationFileFormat | None = None,
    ) -> Path:
        """Save calibration result to file.

        Args:
            path: Output file path.
            result: Calibration result to save.
            format: File format (auto-detected from extension if None).

        Returns:
            Path to saved file.

        Raises:
            ValueError: If format cannot be determined.
        """
        path = Path(path)

        # Determine format
        if format is None:
            try:
                format = CalibrationFileFormat.from_extension(path.suffix)
            except ValueError:
                # Default to HDF5 if no extension
                format = CalibrationFileFormat.HDF5
                path = path.with_suffix(".h5")

        # Get handler and save
        handler = cls._handlers[format]
        handler.save(path, result)

        logger.info(f"Saved calibration to {format.value}: {path}")
        return path

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        format: CalibrationFileFormat | None = None,
    ) -> CalibrationResult:
        """Load calibration result from file.

        Args:
            path: Input file path.
            format: File format (auto-detected from extension if None).

        Returns:
            CalibrationResult loaded from file.

        Raises:
            FileFormatError: If file cannot be loaded.
            ValueError: If format cannot be determined.
        """
        path = Path(path)

        if not path.exists():
            raise FileFormatError(f"File not found: {path}")

        # Determine format
        if format is None:
            try:
                format = CalibrationFileFormat.from_extension(path.suffix)
            except ValueError:
                # Try to detect format from content
                format = cls._detect_format(path)

        # Get handler and load
        handler = cls._handlers[format]
        result = handler.load(path)

        logger.info(f"Loaded calibration from {format.value}: {path}")
        return result

    @classmethod
    def _detect_format(cls, path: Path) -> CalibrationFileFormat:
        """Detect file format from content.

        Args:
            path: File path.

        Returns:
            Detected format.

        Raises:
            FileFormatError: If format cannot be detected.
        """
        # Try each format
        for format, handler in cls._handlers.items():
            if handler.is_valid_file(path):
                return format

        raise FileFormatError(f"Could not detect format of: {path}")

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Get list of supported file extensions."""
        return [".h5", ".hdf5", ".mat", ".json"]

    @classmethod
    def save_all_formats(
        cls,
        base_path: Union[str, Path],
        result: CalibrationResult,
    ) -> dict[str, Path]:
        """Save calibration result in all supported formats.

        Useful for maximum compatibility.

        Args:
            base_path: Base path without extension.
            result: Calibration result to save.

        Returns:
            Dictionary mapping format names to saved paths.
        """
        base_path = Path(base_path)
        base_name = base_path.stem
        base_dir = base_path.parent

        saved_paths = {}

        for format in CalibrationFileFormat:
            ext = {
                CalibrationFileFormat.HDF5: ".h5",
                CalibrationFileFormat.MAT: ".mat",
                CalibrationFileFormat.JSON: ".json",
            }[format]

            output_path = base_dir / f"{base_name}{ext}"
            cls.save(output_path, result, format=format)
            saved_paths[format.value] = output_path

        return saved_paths
