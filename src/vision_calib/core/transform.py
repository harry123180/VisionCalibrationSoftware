"""
座標轉換模組

提供像素座標、相機座標、世界座標之間的轉換。

座標系定義：
- 像素座標 (u, v): 圖像左上角為原點，u 向右，v 向下
- 相機座標 (Xc, Yc, Zc): 相機光心為原點，Zc 沿光軸向前
- 世界座標 (Xw, Yw, Zw): 由棋盤格定義，原點在角點，Zw 垂直於棋盤格平面
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from vision_calib.core.types import CameraExtrinsic, CameraIntrinsic
from vision_calib.utils.logging import get_logger

logger = get_logger("core.transform")


@dataclass
class TransformResult:
    """座標轉換結果"""

    input_coords: np.ndarray
    output_coords: np.ndarray
    transform_type: str


class CoordinateTransformer:
    """
    座標轉換器

    提供像素、相機、世界座標系之間的雙向轉換。
    """

    def __init__(
        self,
        intrinsic: CameraIntrinsic,
        extrinsic: Optional[CameraExtrinsic] = None,
    ):
        """
        初始化座標轉換器

        Args:
            intrinsic: 內參標定結果
            extrinsic: 外參標定結果 (如果需要世界座標轉換)
        """
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic

        # 快取常用矩陣
        self._K = intrinsic.camera_matrix
        self._K_inv = np.linalg.inv(self._K)
        self._dist = intrinsic.distortion_coeffs

        if extrinsic is not None:
            self._R = extrinsic.rotation_matrix
            self._R_inv = self._R.T
            self._t = extrinsic.translation_vector.reshape(3, 1)

    def set_extrinsic(self, extrinsic: CameraExtrinsic) -> None:
        """設置或更新外參"""
        self.extrinsic = extrinsic
        self._R = extrinsic.rotation_matrix
        self._R_inv = self._R.T
        self._t = extrinsic.translation_vector.reshape(3, 1)

    # ==================== 像素 ↔ 相機 ====================

    def pixel_to_normalized(
        self,
        pixel_coords: np.ndarray,
        undistort: bool = True,
    ) -> np.ndarray:
        """
        像素座標 → 歸一化相機座標

        Args:
            pixel_coords: 像素座標 (N, 2) 或 (2,)
            undistort: 是否進行去畸變

        Returns:
            歸一化座標 (N, 2)，在 Z=1 平面上
        """
        pts = np.asarray(pixel_coords, dtype=np.float64)
        single_point = pts.ndim == 1
        if single_point:
            pts = pts.reshape(1, 2)

        if undistort:
            # 使用 OpenCV 去畸變
            pts_undist = cv2.undistortPoints(
                pts.reshape(-1, 1, 2),
                self._K,
                self._dist,
            )
            result = pts_undist.reshape(-1, 2)
        else:
            # 僅應用 K^(-1)
            pts_homo = np.hstack([pts, np.ones((len(pts), 1))])
            normalized = (self._K_inv @ pts_homo.T).T
            result = normalized[:, :2]

        return result[0] if single_point else result

    def normalized_to_pixel(
        self,
        normalized_coords: np.ndarray,
        distort: bool = True,
    ) -> np.ndarray:
        """
        歸一化相機座標 → 像素座標

        Args:
            normalized_coords: 歸一化座標 (N, 2) 或 (2,)
            distort: 是否應用畸變

        Returns:
            像素座標 (N, 2)
        """
        pts = np.asarray(normalized_coords, dtype=np.float64)
        single_point = pts.ndim == 1
        if single_point:
            pts = pts.reshape(1, 2)

        if distort:
            # 應用畸變模型
            pts_3d = np.hstack([pts, np.ones((len(pts), 1))])
            projected, _ = cv2.projectPoints(
                pts_3d,
                np.zeros(3),  # 無旋轉
                np.zeros(3),  # 無平移
                self._K,
                self._dist,
            )
            result = projected.reshape(-1, 2)
        else:
            # 僅應用 K
            pts_homo = np.hstack([pts, np.ones((len(pts), 1))])
            pixel = (self._K @ pts_homo.T).T
            result = pixel[:, :2]

        return result[0] if single_point else result

    def pixel_to_camera_ray(
        self,
        pixel_coords: np.ndarray,
        undistort: bool = True,
    ) -> np.ndarray:
        """
        像素座標 → 相機座標系中的射線方向

        Args:
            pixel_coords: 像素座標 (N, 2) 或 (2,)
            undistort: 是否去畸變

        Returns:
            歸一化射線方向 (N, 3)，從相機原點出發
        """
        normalized = self.pixel_to_normalized(pixel_coords, undistort)

        single_point = normalized.ndim == 1
        if single_point:
            normalized = normalized.reshape(1, 2)

        # 射線方向: [x, y, 1]，然後歸一化
        rays = np.hstack([normalized, np.ones((len(normalized), 1))])
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)

        return rays[0] if single_point else rays

    # ==================== 相機 ↔ 世界 ====================

    def camera_to_world(self, camera_coords: np.ndarray) -> np.ndarray:
        """
        相機座標 → 世界座標

        Args:
            camera_coords: 相機座標 (N, 3) 或 (3,)

        Returns:
            世界座標 (N, 3)

        公式: P_world = R^T @ (P_camera - t)
        """
        self._check_extrinsic()

        pts = np.asarray(camera_coords, dtype=np.float64)
        single_point = pts.ndim == 1
        if single_point:
            pts = pts.reshape(1, 3)

        # P_world = R^(-1) @ (P_camera - t)
        result = (self._R_inv @ (pts.T - self._t)).T

        return result[0] if single_point else result

    def world_to_camera(self, world_coords: np.ndarray) -> np.ndarray:
        """
        世界座標 → 相機座標

        Args:
            world_coords: 世界座標 (N, 3) 或 (3,)

        Returns:
            相機座標 (N, 3)

        公式: P_camera = R @ P_world + t
        """
        self._check_extrinsic()

        pts = np.asarray(world_coords, dtype=np.float64)
        single_point = pts.ndim == 1
        if single_point:
            pts = pts.reshape(1, 3)

        # P_camera = R @ P_world + t
        result = (self._R @ pts.T + self._t).T

        return result[0] if single_point else result

    # ==================== 像素 ↔ 世界 ====================

    def pixel_to_world(
        self,
        pixel_coords: np.ndarray,
        z_world: float = 0.0,
    ) -> np.ndarray:
        """
        像素座標 → 世界座標 (假設在 Z=z_world 平面上)

        這是工業視覺中最常用的轉換：將圖像上的點轉換為
        棋盤格平面（或平行平面）上的實際位置。

        Args:
            pixel_coords: 像素座標 (N, 2) 或 (2,)
            z_world: 目標平面的世界 Z 座標 (預設 0，即棋盤格平面)

        Returns:
            世界座標 (N, 2) 或 (N, 3)，(X, Y) 或 (X, Y, Z)

        原理：
        1. 計算從相機穿過像素點的射線
        2. 將射線與 Z=z_world 平面求交
        """
        self._check_extrinsic()

        pts = np.asarray(pixel_coords, dtype=np.float64)
        single_point = pts.ndim == 1
        if single_point:
            pts = pts.reshape(1, 2)

        # 取得歸一化座標
        normalized = self.pixel_to_normalized(pts, undistort=True)

        results = []
        for norm_pt in normalized:
            # 相機座標系中的射線方向 (Z=1 平面上的點)
            ray_camera = np.array([norm_pt[0], norm_pt[1], 1.0])

            # 轉換到世界座標系
            # 射線起點: 相機位置 (世界座標)
            camera_pos_world = -self._R_inv @ self._t

            # 射線方向 (世界座標)
            ray_world = self._R_inv @ ray_camera.reshape(3, 1)

            # 與 Z = z_world 平面求交
            # camera_pos + t * ray_dir = [x, y, z_world]
            # 解: t = (z_world - camera_pos[2]) / ray_dir[2]
            t_param = (z_world - camera_pos_world[2, 0]) / ray_world[2, 0]

            intersection = camera_pos_world + t_param * ray_world
            results.append(intersection.flatten())

        result = np.array(results)

        if single_point:
            return result[0]
        return result

    def world_to_pixel(self, world_coords: np.ndarray) -> np.ndarray:
        """
        世界座標 → 像素座標

        Args:
            world_coords: 世界座標 (N, 3) 或 (3,)

        Returns:
            像素座標 (N, 2)
        """
        self._check_extrinsic()

        pts = np.asarray(world_coords, dtype=np.float64)
        single_point = pts.ndim == 1
        if single_point:
            pts = pts.reshape(1, 3)

        # 使用 projectPoints
        projected, _ = cv2.projectPoints(
            pts,
            self.extrinsic.rotation_vector,
            self.extrinsic.translation_vector,
            self._K,
            self._dist,
        )

        result = projected.reshape(-1, 2)
        return result[0] if single_point else result

    # ==================== 便捷方法 ====================

    def get_world_xy(
        self,
        pixel_coords: np.ndarray,
        z_world: float = 0.0,
    ) -> Tuple[float, float]:
        """
        取得像素對應的世界 XY 座標 (標量版本)

        Args:
            pixel_coords: 像素座標 [u, v]
            z_world: 世界 Z 座標

        Returns:
            (X_world, Y_world) 元組
        """
        world = self.pixel_to_world(pixel_coords, z_world)
        return (float(world[0]), float(world[1]))

    def get_pixel_uv(
        self,
        world_coords: np.ndarray,
    ) -> Tuple[float, float]:
        """
        取得世界座標對應的像素座標 (標量版本)

        Args:
            world_coords: 世界座標 [X, Y, Z]

        Returns:
            (u, v) 元組
        """
        pixel = self.world_to_pixel(world_coords)
        return (float(pixel[0]), float(pixel[1]))

    def create_pixel_to_world_map(
        self,
        z_world: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        建立整張圖像的像素→世界座標映射表

        用於快速查詢或視覺化座標系。

        Args:
            z_world: 世界 Z 座標

        Returns:
            (X_map, Y_map): 與圖像同尺寸的陣列，
                           X_map[v, u] = 該像素對應的世界 X 座標
                           Y_map[v, u] = 該像素對應的世界 Y 座標
        """
        self._check_extrinsic()

        h, w = self.intrinsic.image_size[1], self.intrinsic.image_size[0]

        # 建立像素網格
        u_coords = np.arange(w)
        v_coords = np.arange(h)
        uu, vv = np.meshgrid(u_coords, v_coords)

        # 展平並轉換
        pixels = np.stack([uu.flatten(), vv.flatten()], axis=1)
        world = self.pixel_to_world(pixels, z_world)

        # 重塑為圖像尺寸
        X_map = world[:, 0].reshape(h, w)
        Y_map = world[:, 1].reshape(h, w)

        return X_map, Y_map

    def _check_extrinsic(self) -> None:
        """檢查外參是否已設置"""
        if self.extrinsic is None:
            raise ValueError("需要外參才能進行世界座標轉換。請先執行外參標定。")


# ==================== 便捷函數 ====================


def create_transformer(
    intrinsic: CameraIntrinsic,
    extrinsic: Optional[CameraExtrinsic] = None,
) -> CoordinateTransformer:
    """建立座標轉換器"""
    return CoordinateTransformer(intrinsic, extrinsic)


def pixel_to_world_simple(
    pixel: Tuple[float, float],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    z_world: float = 0.0,
) -> Tuple[float, float]:
    """
    簡易版像素→世界座標轉換

    不需要建立完整的轉換器物件。

    Args:
        pixel: (u, v) 像素座標
        camera_matrix: 3x3 內參矩陣
        dist_coeffs: 畸變係數
        rvec: 3x1 旋轉向量
        tvec: 3x1 平移向量
        z_world: 世界 Z 座標

    Returns:
        (X_world, Y_world)
    """
    # 去畸變
    pts = np.array([[pixel]], dtype=np.float64)
    normalized = cv2.undistortPoints(pts, camera_matrix, dist_coeffs)
    norm_pt = normalized[0, 0]

    # 旋轉矩陣
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t = tvec.reshape(3, 1)

    # 相機位置
    camera_pos = -R_inv @ t

    # 射線方向
    ray_camera = np.array([[norm_pt[0]], [norm_pt[1]], [1.0]])
    ray_world = R_inv @ ray_camera

    # 與 Z 平面求交
    t_param = (z_world - camera_pos[2, 0]) / ray_world[2, 0]
    intersection = camera_pos + t_param * ray_world

    return (float(intersection[0, 0]), float(intersection[1, 0]))
