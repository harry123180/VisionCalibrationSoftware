# TSIC/CR-ICS01 相機與手臂整合控制教學軟體

[English](README_EN.md) | 繁體中文

專業的相機標定工具，使用棋盤格圖案進行相機校正。

## 功能特色

- **內參標定**：從多張棋盤格圖像計算相機矩陣與畸變係數
- **外參標定**：使用多種 PnP 算法估計相機姿態（支援 P3P、AP3P、EPnP、IPPE 等）
- **座標轉換**：在像素座標、相機座標與世界座標之間轉換
- **棋盤格座標生成**：根據機械臂座標系自動生成世界座標
- **即時可視化**：座標系對齊可視化，支援縮放與拖動
- **多種匯出格式**：
  - HDF5 (.h5) - 跨平台科學數據格式
  - MAT (.mat) - MATLAB/Octave 相容
  - JSON (.json) - 人類可讀格式

## 安裝

```bash
# 從原始碼安裝
git clone https://github.com/TSIC-LAB/VisionCalibrationSoftware.git
cd VisionCalibrationSoftware
pip install -e .
```

### 相依套件

- Python >= 3.10
- NumPy >= 1.24.0
- OpenCV >= 4.8.0
- PySide6 >= 6.5.0
- h5py >= 3.9.0
- SciPy >= 1.11.0
- Matplotlib >= 3.7.0

## 快速開始

### GUI 應用程式

```bash
python main.py
```

### Python API

```python
from vision_calib.core.intrinsic import IntrinsicCalibrator, IntrinsicCalibrationConfig
from vision_calib.core.types import CheckerboardConfig
from vision_calib.io import CalibrationFile

# 設定標定參數
config = IntrinsicCalibrationConfig(
    checkerboard=CheckerboardConfig(rows=12, cols=17, square_size_mm=10.0)
)

# 建立標定器並加入圖像
calibrator = IntrinsicCalibrator(config)
for image_path in image_paths:
    calibrator.add_image(image_path)

# 執行標定
result = calibrator.calibrate()
print(result.summary())

# 儲存為多種格式
CalibrationFile.save("calibration.h5", result)   # HDF5
CalibrationFile.save("calibration.mat", result)  # MAT (Octave/MATLAB)
CalibrationFile.save("calibration.json", result) # JSON
```

### Octave/MATLAB 使用

```matlab
% 從 MAT 檔案載入
data = load('calibration.mat');
K = data.camera_matrix;
D = data.distortion_coeffs;

% 或從 HDF5 載入
K = h5read('calibration.h5', '/intrinsic/camera_matrix');
D = h5read('calibration.h5', '/intrinsic/distortion_coeffs');
```

## 外參標定算法

本軟體支援多種 PnP 算法：

| 算法 | 最少點數 | 說明 |
|------|---------|------|
| PnP 迭代法 | ≥4 | 最通用，推薦使用 |
| EPnP | ≥4 | 高效率，適合點數較多的情況 |
| P3P | =3 | 只需 3 個點，可能有多解 |
| AP3P | ≥3 | P3P 改進版，數值更穩定 |
| IPPE | ≥4 | 平面物體專用 |
| IPPE_SQUARE | ≥4 | 正方形標定板專用 |

## 匯出格式詳情

### HDF5 結構

```
calibration.h5
├── /intrinsic/
│   ├── camera_matrix      (3,3) float64
│   ├── distortion_coeffs  (n,) float64
│   ├── image_size         (2,) int32
│   └── reprojection_error scalar float64
├── /extrinsic/            (選用)
│   ├── rotation_vector    (3,) float64
│   ├── translation_vector (3,) float64
│   └── rotation_matrix    (3,3) float64
└── /metadata/
    ├── timestamp          string
    ├── checkerboard_*     various
    └── software_version   string
```

### MAT 變數

| 變數名稱 | 型別 | 說明 |
|----------|------|------|
| `camera_matrix` | (3,3) double | 相機內參矩陣 K |
| `distortion_coeffs` | (1,n) double | 畸變係數 |
| `image_size` | (1,2) double | [寬度, 高度] |
| `reprojection_error` | scalar | RMS 重投影誤差（像素） |
| `rotation_vector` | (3,1) double | 旋轉向量（選用） |
| `translation_vector` | (3,1) double | 平移向量（選用） |

## 開發

```bash
# 安裝開發相依套件
pip install -e ".[dev]"

# 執行測試
pytest

# 型別檢查
mypy src/

# 程式碼檢查
ruff check src/
```

## 授權條款

Apache License 2.0 - 詳見 [LICENSE](LICENSE)

## 開發單位

TSIC (Taiwan Smart Industrial Control)

## 致謝

- OpenCV 提供底層標定算法
- 張正友標定法的理論基礎
