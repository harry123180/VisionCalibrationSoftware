"""
座標轉換工具函數

提供簡化的像素↔世界座標轉換功能

預設標定檔案位置:
    - calibration/calibration.h5 (內參)
    - calibration/extrinsic.npy (外參)
"""

import numpy as np
import cv2
import h5py
from pathlib import Path
from typing import Tuple, Optional

# 取得專案根目錄 (examples 的上一層)
PROJECT_ROOT = Path(__file__).parent.parent

# 預設標定檔案路徑
DEFAULT_INTRINSIC = PROJECT_ROOT / "calibration" / "calibration.h5"
DEFAULT_EXTRINSIC = PROJECT_ROOT / "calibration" / "extrinsic.npy"


class CoordinateTransform:
    """座標轉換類別 - 簡化版"""

    def __init__(self,
                 intrinsic_h5_path: Optional[str] = None,
                 extrinsic_npy_path: Optional[str] = None):
        """
        初始化座標轉換器

        Args:
            intrinsic_h5_path: 內參檔案路徑 (.h5)，預設使用 calibration/calibration.h5
            extrinsic_npy_path: 外參檔案路徑 (.npy)，預設使用 calibration/extrinsic.npy
        """
        # 使用預設路徑
        if intrinsic_h5_path is None:
            intrinsic_h5_path = DEFAULT_INTRINSIC
        if extrinsic_npy_path is None:
            extrinsic_npy_path = DEFAULT_EXTRINSIC

        # 載入內參
        with h5py.File(intrinsic_h5_path, 'r') as f:
            self.camera_matrix = f['/intrinsic/camera_matrix'][:]
            self.distortion_coeffs = f['/intrinsic/distortion_coeffs'][:]

        # 載入外參
        data = np.load(extrinsic_npy_path, allow_pickle=True).item()
        self.rvec = data['rvec']
        self.tvec = data['tvec'].reshape(3, 1)
        self.rotation_matrix = data['rotation_matrix']

        # 預計算常用矩陣
        self.R_inv = self.rotation_matrix.T
        self.camera_pos_world = -self.R_inv @ self.tvec

    def pixel_to_world(self, u: float, v: float, z: float = 0.0) -> Tuple[float, float]:
        """
        像素座標 → 世界座標

        Args:
            u: 像素 U 座標
            v: 像素 V 座標
            z: 世界座標 Z 平面 (預設 0)

        Returns:
            (x, y): 世界座標 (mm)
        """
        # 去畸變
        pixel_point = np.array([[[u, v]]], dtype=np.float64)
        normalized = cv2.undistortPoints(pixel_point, self.camera_matrix, self.distortion_coeffs)
        normalized = normalized.reshape(2)

        # 射線方向
        ray_camera = np.array([normalized[0], normalized[1], 1.0])
        ray_world = self.R_inv @ ray_camera.reshape(3, 1)

        # 與 Z 平面求交
        t_param = (z - self.camera_pos_world[2, 0]) / ray_world[2, 0]
        intersection = self.camera_pos_world + t_param * ray_world

        return float(intersection[0, 0]), float(intersection[1, 0])

    def world_to_pixel(self, x: float, y: float, z: float = 0.0) -> Tuple[float, float]:
        """
        世界座標 → 像素座標

        Args:
            x: 世界 X 座標 (mm)
            y: 世界 Y 座標 (mm)
            z: 世界 Z 座標 (mm，預設 0)

        Returns:
            (u, v): 像素座標
        """
        world_point = np.array([[x, y, z]], dtype=np.float64)
        projected, _ = cv2.projectPoints(
            world_point,
            self.rvec,
            self.tvec,
            self.camera_matrix,
            self.distortion_coeffs
        )
        return float(projected[0, 0, 0]), float(projected[0, 0, 1])

    def batch_pixel_to_world(self, pixels: np.ndarray, z: float = 0.0) -> np.ndarray:
        """
        批量像素座標 → 世界座標

        Args:
            pixels: 像素座標陣列 (N, 2)
            z: 世界座標 Z 平面

        Returns:
            世界座標陣列 (N, 2)
        """
        results = []
        for u, v in pixels:
            x, y = self.pixel_to_world(u, v, z)
            results.append([x, y])
        return np.array(results)


# ============ 使用範例 ============

if __name__ == '__main__':
    print("=" * 60)
    print("座標轉換工具 - 使用範例")
    print("=" * 60)
    print()
    print(f"預設內參檔案: {DEFAULT_INTRINSIC}")
    print(f"預設外參檔案: {DEFAULT_EXTRINSIC}")
    print()

    # 檢查檔案是否存在
    if DEFAULT_INTRINSIC.exists() and DEFAULT_EXTRINSIC.exists():
        print("標定檔案存在，執行範例轉換...")
        print("-" * 60)

        # 初始化（使用預設路徑）
        tf = CoordinateTransform()

        # 測試像素座標
        test_pixels = [(578, 217), (389, 385), (618, 400)]

        for u, v in test_pixels:
            x, y = tf.pixel_to_world(u, v, z=0)
            print(f"像素 ({u:>6.1f}, {v:>6.1f}) → 世界 (X={x:>10.2f}mm, Y={y:>10.2f}mm)")

        print("-" * 60)
        print()
        print("反向驗證 (世界 → 像素):")
        x, y = tf.pixel_to_world(640, 480, z=0)
        u_back, v_back = tf.world_to_pixel(x, y, z=0)
        print(f"原始像素: (640, 480)")
        print(f"世界座標: ({x:.2f}, {y:.2f})")
        print(f"反向像素: ({u_back:.2f}, {v_back:.2f})")

    else:
        print("找不到標定檔案，請先執行標定。")
        print()
        print("使用方式:")
        print("  from coordinate_transform_utils import CoordinateTransform")
        print()
        print("  # 使用預設路徑初始化")
        print("  tf = CoordinateTransform()")
        print()
        print("  # 或指定自訂路徑")
        print("  tf = CoordinateTransform('my_calib.h5', 'my_ext.npy')")
        print()
        print("  # 像素 → 世界 (假設 Z=0)")
        print("  x, y = tf.pixel_to_world(u=640, v=480)")
        print()
        print("  # 世界 → 像素")
        print("  u, v = tf.world_to_pixel(x=100, y=200)")
