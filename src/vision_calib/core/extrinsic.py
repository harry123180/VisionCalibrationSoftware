"""
外參標定模組

使用 solvePnP 計算相機相對於世界座標系的位姿。
世界座標系原點定義在棋盤格的一個角點上。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from vision_calib.core.types import (
    CalibrationError,
    CameraExtrinsic,
    CameraIntrinsic,
    CheckerboardConfig,
)
from vision_calib.io.image_loader import ImageLoader
from vision_calib.utils.logging import get_logger

logger = get_logger("core.extrinsic")


@dataclass
class ExtrinsicCalibrationResult:
    """外參標定結果"""

    extrinsic: CameraExtrinsic
    reprojection_error: float
    image_path: str
    num_points: int

    @property
    def rotation_matrix(self) -> np.ndarray:
        """旋轉矩陣 (3x3)"""
        return self.extrinsic.rotation_matrix

    @property
    def camera_position_world(self) -> np.ndarray:
        """相機在世界座標系中的位置"""
        R = self.rotation_matrix
        t = self.extrinsic.translation_vector
        return -R.T @ t

    def summary(self) -> str:
        """生成結果摘要"""
        R = self.rotation_matrix
        t = self.extrinsic.translation_vector.flatten()
        rvec = self.extrinsic.rotation_vector.flatten()
        cam_pos = self.camera_position_world.flatten()

        return f"""外參標定結果
════════════════════════════════════════

旋轉向量 (Rodrigues):
    rx = {rvec[0]:+.6f} rad ({np.degrees(rvec[0]):+.2f}°)
    ry = {rvec[1]:+.6f} rad ({np.degrees(rvec[1]):+.2f}°)
    rz = {rvec[2]:+.6f} rad ({np.degrees(rvec[2]):+.2f}°)

平移向量 (mm):
    tx = {t[0]:+.2f}
    ty = {t[1]:+.2f}
    tz = {t[2]:+.2f}

旋轉矩陣:
    ┌ {R[0,0]:+.6f}  {R[0,1]:+.6f}  {R[0,2]:+.6f} ┐
    │ {R[1,0]:+.6f}  {R[1,1]:+.6f}  {R[1,2]:+.6f} │
    └ {R[2,0]:+.6f}  {R[2,1]:+.6f}  {R[2,2]:+.6f} ┘

相機位置 (世界座標, mm):
    X = {cam_pos[0]:+.2f}
    Y = {cam_pos[1]:+.2f}
    Z = {cam_pos[2]:+.2f}

重投影誤差: {self.reprojection_error:.4f} pixels
使用點數: {self.num_points}
"""


class ExtrinsicCalibrator:
    """
    外參標定器

    使用單張圖像計算相機相對於棋盤格座標系的外參。
    棋盤格定義世界座標系：原點在左上角，X 向右，Y 向下，Z 垂直紙面向外。
    """

    def __init__(
        self,
        intrinsic: CameraIntrinsic,
        checkerboard: CheckerboardConfig,
    ):
        """
        初始化外參標定器

        Args:
            intrinsic: 內參標定結果
            checkerboard: 棋盤格配置
        """
        self.intrinsic = intrinsic
        self.checkerboard = checkerboard

        # 生成世界座標點 (Z=0 平面)
        self.object_points = checkerboard.generate_object_points()

    def calibrate(
        self,
        image_path: str,
        corners: Optional[np.ndarray] = None,
        method: int = cv2.SOLVEPNP_ITERATIVE,
    ) -> ExtrinsicCalibrationResult:
        """
        執行外參標定

        Args:
            image_path: 圖像路徑
            corners: 預先偵測的角點 (可選)。如果為 None，會自動偵測。
            method: solvePnP 方法

        Returns:
            ExtrinsicCalibrationResult 外參標定結果

        Raises:
            CalibrationError: 標定失敗
        """
        # 如果沒有提供角點，自動偵測
        if corners is None:
            corners = self._detect_corners(image_path)

        if corners is None:
            raise CalibrationError(f"無法在圖像中偵測到角點: {image_path}")

        # 確保角點格式正確
        image_points = corners.reshape(-1, 2).astype(np.float32)
        object_points = self.object_points.reshape(-1, 3).astype(np.float32)

        # 執行 solvePnP
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.intrinsic.camera_matrix,
            self.intrinsic.distortion_coeffs,
            flags=method,
        )

        if not success:
            raise CalibrationError("solvePnP 失敗")

        # 計算重投影誤差
        reprojected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.intrinsic.camera_matrix,
            self.intrinsic.distortion_coeffs,
        )
        reprojected = reprojected.reshape(-1, 2)
        error = np.sqrt(np.mean(np.sum((image_points - reprojected) ** 2, axis=1)))

        # 建立結果
        extrinsic = CameraExtrinsic(
            rotation_vector=rvec,
            translation_vector=tvec,
        )

        result = ExtrinsicCalibrationResult(
            extrinsic=extrinsic,
            reprojection_error=error,
            image_path=image_path,
            num_points=len(image_points),
        )

        logger.info(f"外參標定完成，重投影誤差: {error:.4f} pixels")
        return result

    def _detect_corners(self, image_path: str) -> Optional[np.ndarray]:
        """偵測棋盤格角點"""
        img = ImageLoader.load_grayscale(image_path)
        if img is None:
            return None

        pattern_size = (self.checkerboard.cols, self.checkerboard.rows)
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK
        )

        found, corners = cv2.findChessboardCorners(img, pattern_size, flags)

        if not found:
            return None

        # 亞像素精化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)

        return corners

    def calibrate_multi_pose(
        self,
        image_paths: list[str],
        corners_list: Optional[list[np.ndarray]] = None,
    ) -> list[ExtrinsicCalibrationResult]:
        """
        對多張圖像分別計算外參

        這不是聯合優化，而是對每張圖像獨立計算其外參。
        用於分析相機在不同位置的姿態。

        Args:
            image_paths: 圖像路徑列表
            corners_list: 對應的角點列表 (可選)

        Returns:
            外參結果列表
        """
        results = []

        for i, path in enumerate(image_paths):
            corners = corners_list[i] if corners_list else None
            try:
                result = self.calibrate(path, corners)
                results.append(result)
            except CalibrationError as e:
                logger.warning(f"圖像 {path} 外參標定失敗: {e}")

        return results


def calibrate_extrinsic(
    intrinsic: CameraIntrinsic,
    checkerboard: CheckerboardConfig,
    image_path: str,
    corners: Optional[np.ndarray] = None,
) -> ExtrinsicCalibrationResult:
    """
    便捷函數：執行外參標定

    Args:
        intrinsic: 內參標定結果
        checkerboard: 棋盤格配置
        image_path: 圖像路徑
        corners: 預先偵測的角點 (可選)

    Returns:
        ExtrinsicCalibrationResult
    """
    calibrator = ExtrinsicCalibrator(intrinsic, checkerboard)
    return calibrator.calibrate(image_path, corners)
