"""
JSON format support for calibration data.

JSON format is human-readable and widely supported.
It's useful for configuration, debugging, and interoperability.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import numpy as np

from vision_calib.core.types import (
    CalibrationResult,
    CameraIntrinsic,
    CameraExtrinsic,
    CheckerboardConfig,
    FileFormatError,
)
from vision_calib.utils.logging import get_logger

logger = get_logger("io.formats.json")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class JSONFormat:
    """JSON format reader/writer for calibration data.

    Creates human-readable JSON files with calibration data.
    Arrays are stored as nested lists.

    JSON structure:
        {
            "format_version": "1.0",
            "format_type": "vision-calib",
            "intrinsic": {
                "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                "distortion_coeffs": [k1, k2, p1, p2, k3],
                "image_size": [width, height],
                "reprojection_error": 0.5
            },
            "extrinsic": {  // optional
                "rotation_vector": [rx, ry, rz],
                "translation_vector": [tx, ty, tz]
            },
            "metadata": {
                "timestamp": "2024-01-01T12:00:00",
                "checkerboard": {
                    "rows": 5,
                    "cols": 7,
                    "square_size_mm": 30.0
                },
                "num_images_used": 10,
                "software_version": "1.0.0",
                "notes": ""
            },
            "per_image_errors": [0.3, 0.4, ...]  // optional
        }

    Example (Python):
        >>> import json
        >>> with open('calibration.json', 'r') as f:
        ...     data = json.load(f)
        >>> K = np.array(data['intrinsic']['camera_matrix'])
    """

    EXTENSION = ".json"
    VERSION = "1.0"

    @classmethod
    def save(
        cls,
        path: Union[str, Path],
        result: CalibrationResult,
        indent: int = 2,
        ensure_ascii: bool = False,
    ) -> None:
        """Save calibration result to JSON file.

        Args:
            path: Output file path.
            result: Calibration result to save.
            indent: JSON indentation level.
            ensure_ascii: If True, escape non-ASCII characters.
        """
        path = Path(path)
        if path.suffix.lower() != cls.EXTENSION:
            path = path.with_suffix(cls.EXTENSION)

        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving calibration to JSON: {path}")

        # Build data dictionary
        data: dict[str, Any] = {
            "format_version": cls.VERSION,
            "format_type": "vision-calib",
            "intrinsic": {
                "camera_matrix": result.intrinsic.camera_matrix.tolist(),
                "distortion_coeffs": result.intrinsic.distortion_coeffs.tolist(),
                "image_size": list(result.intrinsic.image_size),
                "reprojection_error": result.intrinsic.reprojection_error,
                # Convenience fields
                "fx": result.intrinsic.fx,
                "fy": result.intrinsic.fy,
                "cx": result.intrinsic.cx,
                "cy": result.intrinsic.cy,
            },
            "metadata": {
                "timestamp": result.timestamp.isoformat(),
                "num_images_used": result.num_images_used,
                "software_version": result.software_version,
                "notes": result.notes,
            },
        }

        # Extrinsic (optional)
        if result.extrinsic is not None:
            data["extrinsic"] = {
                "rotation_vector": result.extrinsic.rotation_vector.ravel().tolist(),
                "translation_vector": result.extrinsic.translation_vector.ravel().tolist(),
                "rotation_matrix": result.extrinsic.rotation_matrix.tolist(),
                "camera_position": result.extrinsic.camera_position.tolist(),
            }

        # Checkerboard config (optional)
        if result.checkerboard_config is not None:
            data["metadata"]["checkerboard"] = {
                "rows": result.checkerboard_config.rows,
                "cols": result.checkerboard_config.cols,
                "square_size_mm": result.checkerboard_config.square_size_mm,
            }

        # Per-image errors (optional)
        if result.per_image_errors is not None:
            data["per_image_errors"] = result.per_image_errors

        # Write file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, cls=NumpyEncoder)

        logger.info(f"Saved calibration to: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> CalibrationResult:
        """Load calibration result from JSON file.

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

        logger.info(f"Loading calibration from JSON: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise FileFormatError(f"Invalid JSON format: {e}")

        # Validate required fields
        if "intrinsic" not in data:
            raise FileFormatError("Missing 'intrinsic' section in JSON file")

        intrinsic_data = data["intrinsic"]
        if "camera_matrix" not in intrinsic_data:
            raise FileFormatError("Missing 'camera_matrix' in intrinsic section")

        # Load intrinsic
        intrinsic = CameraIntrinsic(
            camera_matrix=np.array(intrinsic_data["camera_matrix"], dtype=np.float64),
            distortion_coeffs=np.array(
                intrinsic_data.get("distortion_coeffs", [0, 0, 0, 0, 0]),
                dtype=np.float64,
            ),
            image_size=tuple(intrinsic_data.get("image_size", [0, 0])),
            reprojection_error=float(intrinsic_data.get("reprojection_error", 0.0)),
        )

        # Load extrinsic (optional)
        extrinsic = None
        if "extrinsic" in data:
            extrinsic_data = data["extrinsic"]
            if "rotation_vector" in extrinsic_data and "translation_vector" in extrinsic_data:
                extrinsic = CameraExtrinsic(
                    rotation_vector=np.array(extrinsic_data["rotation_vector"], dtype=np.float64),
                    translation_vector=np.array(
                        extrinsic_data["translation_vector"], dtype=np.float64
                    ),
                )

        # Load metadata
        metadata = data.get("metadata", {})

        # Checkerboard config (optional)
        checkerboard_config = None
        if "checkerboard" in metadata:
            cb_data = metadata["checkerboard"]
            checkerboard_config = CheckerboardConfig(
                rows=int(cb_data.get("rows", 0)),
                cols=int(cb_data.get("cols", 0)),
                square_size_mm=float(cb_data.get("square_size_mm", 0.0)),
            )

        # Parse timestamp
        timestamp_str = metadata.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
        except ValueError:
            timestamp = datetime.now()

        # Per-image errors (optional)
        per_image_errors = data.get("per_image_errors")

        return CalibrationResult(
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            timestamp=timestamp,
            checkerboard_config=checkerboard_config,
            num_images_used=int(metadata.get("num_images_used", 0)),
            software_version=str(metadata.get("software_version", "unknown")),
            notes=str(metadata.get("notes", "")),
            per_image_errors=per_image_errors,
        )

    @classmethod
    def is_valid_file(cls, path: Union[str, Path]) -> bool:
        """Check if a file is a valid calibration JSON file.

        Args:
            path: File path to check.

        Returns:
            True if file is valid, False otherwise.
        """
        path = Path(path)
        if not path.exists():
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return "intrinsic" in data and "camera_matrix" in data.get("intrinsic", {})
        except (json.JSONDecodeError, KeyError):
            return False

    @classmethod
    def to_string(cls, result: CalibrationResult, indent: int = 2) -> str:
        """Convert calibration result to JSON string.

        Args:
            result: Calibration result.
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        import io
        from pathlib import Path
        import tempfile

        # Use save to generate the dict, then serialize
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        cls.save(temp_path, result, indent=indent)

        with open(temp_path, "r", encoding="utf-8") as f:
            json_str = f.read()

        temp_path.unlink()  # Clean up temp file

        return json_str
