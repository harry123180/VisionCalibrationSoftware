"""
像素座標轉世界座標範例

使用方式:
    python examples/pixel_to_world_example.py --u 640 --v 480

輸入:
    - .h5 內參檔案 (預設: calibration/calibration.h5)
    - .npy 外參檔案 (預設: calibration/extrinsic.npy)
    - 像素座標 (u, v)

輸出:
    - 世界座標 (x, y)，假設 Z=0
"""

import argparse
import numpy as np
import cv2
import h5py
from pathlib import Path

# 取得專案根目錄 (examples 的上一層)
PROJECT_ROOT = Path(__file__).parent.parent

# 預設標定檔案路徑
DEFAULT_INTRINSIC = PROJECT_ROOT / "calibration" / "calibration.h5"
DEFAULT_EXTRINSIC = PROJECT_ROOT / "calibration" / "extrinsic.npy"


def load_intrinsic(h5_path: str) -> tuple:
    """
    從 .h5 檔案載入內參

    Returns:
        camera_matrix: 內參矩陣 (3x3)
        distortion_coeffs: 畸變係數
    """
    with h5py.File(h5_path, 'r') as f:
        camera_matrix = f['/intrinsic/camera_matrix'][:]
        distortion_coeffs = f['/intrinsic/distortion_coeffs'][:]

    return camera_matrix, distortion_coeffs


def load_extrinsic(npy_path: str) -> tuple:
    """
    從 .npy 檔案載入外參

    Returns:
        rvec: 旋轉向量 (3x1)
        tvec: 平移向量 (3x1)
        rotation_matrix: 旋轉矩陣 (3x3)
    """
    data = np.load(npy_path, allow_pickle=True).item()

    rvec = data['rvec']
    tvec = data['tvec']
    rotation_matrix = data['rotation_matrix']

    return rvec, tvec, rotation_matrix


def pixel_to_world(u: float, v: float,
                   camera_matrix: np.ndarray,
                   distortion_coeffs: np.ndarray,
                   rotation_matrix: np.ndarray,
                   tvec: np.ndarray,
                   z_world: float = 0.0) -> tuple:
    """
    像素座標轉世界座標 (假設在 Z=z_world 平面上)

    Args:
        u, v: 像素座標
        camera_matrix: 內參矩陣 (3x3)
        distortion_coeffs: 畸變係數
        rotation_matrix: 旋轉矩陣 (3x3)
        tvec: 平移向量 (3x1)
        z_world: 世界座標 Z 平面 (預設 0)

    Returns:
        (x, y): 世界座標
    """
    # 1. 去畸變並轉換為歸一化相機座標
    pixel_point = np.array([[[u, v]]], dtype=np.float64)
    normalized = cv2.undistortPoints(pixel_point, camera_matrix, distortion_coeffs)
    normalized = normalized.reshape(2)

    # 2. 相機座標系中的射線方向 (Z=1 平面上的點)
    ray_camera = np.array([normalized[0], normalized[1], 1.0])

    # 3. 計算相機在世界座標系中的位置
    R_inv = rotation_matrix.T
    t = tvec.reshape(3, 1)
    camera_pos_world = -R_inv @ t

    # 4. 射線方向轉換到世界座標系
    ray_world = R_inv @ ray_camera.reshape(3, 1)

    # 5. 與 Z = z_world 平面求交
    # 射線方程: P = camera_pos + t * ray_dir
    # 求解 t 使得 P[2] = z_world
    t_param = (z_world - camera_pos_world[2, 0]) / ray_world[2, 0]

    # 6. 計算交點
    intersection = camera_pos_world + t_param * ray_world

    x = intersection[0, 0]
    y = intersection[1, 0]

    return x, y


def main():
    parser = argparse.ArgumentParser(
        description='像素座標轉世界座標 (假設 Z=0)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
    python examples/pixel_to_world_example.py --u 640 --v 480
    python examples/pixel_to_world_example.py -u 100 -v 200 --z 10
    python examples/pixel_to_world_example.py -i other.h5 -e other.npy -u 640 -v 480
        """
    )

    parser.add_argument('-i', '--intrinsic', default=str(DEFAULT_INTRINSIC),
                        help=f'內參檔案路徑 (預設: calibration/calibration.h5)')
    parser.add_argument('-e', '--extrinsic', default=str(DEFAULT_EXTRINSIC),
                        help=f'外參檔案路徑 (預設: calibration/extrinsic.npy)')
    parser.add_argument('-u', '--u', type=float, required=True,
                        help='像素 U 座標')
    parser.add_argument('-v', '--v', type=float, required=True,
                        help='像素 V 座標')
    parser.add_argument('-z', '--z', type=float, default=0.0,
                        help='世界座標 Z 平面 (預設: 0)')

    args = parser.parse_args()

    # 載入標定數據
    print(f"載入內參: {args.intrinsic}")
    camera_matrix, distortion_coeffs = load_intrinsic(args.intrinsic)

    print(f"載入外參: {args.extrinsic}")
    rvec, tvec, rotation_matrix = load_extrinsic(args.extrinsic)

    # 執行轉換
    x, y = pixel_to_world(
        args.u, args.v,
        camera_matrix, distortion_coeffs,
        rotation_matrix, tvec,
        z_world=args.z
    )

    # 輸出結果
    print()
    print("=" * 50)
    print(f"輸入像素座標: u={args.u:.2f}, v={args.v:.2f}")
    print(f"世界座標 Z 平面: {args.z:.2f} mm")
    print("-" * 50)
    print(f"輸出世界座標: X={x:.4f} mm, Y={y:.4f} mm")
    print("=" * 50)

    return x, y


if __name__ == '__main__':
    main()
