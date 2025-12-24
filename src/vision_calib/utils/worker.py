"""
背景工作執行緒

將耗時的計算任務移至背景執行緒，避免阻塞 GUI。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from PySide6.QtCore import QThread, Signal


@dataclass
class CornerDetectionTask:
    """角點偵測任務"""
    image_path: str
    index: int


@dataclass
class CornerDetectionResult:
    """角點偵測結果"""
    index: int
    image_path: str
    success: bool
    corners: Optional[any] = None  # numpy array of corners
    message: str = ""


class CornerDetectionWorker(QThread):
    """角點偵測背景工作執行緒"""

    # 訊號
    progress = Signal(int, int, str)  # (current, total, message)
    single_result = Signal(object)  # CornerDetectionResult
    finished = Signal(int, int)  # (success_count, total_count)
    error = Signal(str)

    def __init__(
        self,
        image_paths: List[str],
        checkerboard_config,
        parent=None,
    ):
        super().__init__(parent)
        self.image_paths = image_paths
        self.checkerboard_config = checkerboard_config
        self._is_cancelled = False

    def cancel(self):
        """取消任務"""
        self._is_cancelled = True

    def run(self):
        """執行角點偵測"""
        from vision_calib.core.corner_detector import CornerDetector

        try:
            detector = CornerDetector(self.checkerboard_config)
            total = len(self.image_paths)
            success_count = 0

            for i, path in enumerate(self.image_paths):
                if self._is_cancelled:
                    break

                self.progress.emit(i, total, f"正在偵測 ({i + 1}/{total})...")

                result = detector.detect(path)

                detection_result = CornerDetectionResult(
                    index=i,
                    image_path=path,
                    success=result.success,
                    corners=result.corners if result.success else None,
                    message="" if result.success else "偵測失敗",
                )

                if result.success:
                    success_count += 1

                self.single_result.emit(detection_result)

            self.progress.emit(total, total, "偵測完成")
            self.finished.emit(success_count, total)

        except Exception as e:
            self.error.emit(str(e))


class CalibrationWorker(QThread):
    """標定計算背景工作執行緒"""

    # 訊號
    progress = Signal(int, int, str)  # (current, total, message)
    finished = Signal(object)  # CalibrationResult
    error = Signal(str)

    def __init__(
        self,
        image_paths: List[str],
        calibration_config,
        parent=None,
    ):
        super().__init__(parent)
        self.image_paths = image_paths
        self.calibration_config = calibration_config
        self._is_cancelled = False

    def cancel(self):
        """取消任務"""
        self._is_cancelled = True

    def run(self):
        """執行標定計算"""
        from vision_calib.core.intrinsic import IntrinsicCalibrator

        try:
            calibrator = IntrinsicCalibrator(self.calibration_config)

            # 載入圖像
            total_images = len(self.image_paths)
            for i, path in enumerate(self.image_paths):
                if self._is_cancelled:
                    return

                self.progress.emit(i, total_images + 1, f"載入圖像 ({i + 1}/{total_images})...")
                calibrator.add_image(path)

            if self._is_cancelled:
                return

            # 檢查是否可以標定
            if not calibrator.can_calibrate:
                self.error.emit(
                    f"有效圖像不足，至少需要 3 張。目前僅有 {calibrator.num_valid_images} 張。"
                )
                return

            # 執行標定
            self.progress.emit(total_images, total_images + 1, "正在計算標定參數...")

            def progress_callback(current, total, message):
                if not self._is_cancelled:
                    self.progress.emit(current, total, message)

            result = calibrator.calibrate(progress_callback=progress_callback)

            if not self._is_cancelled:
                self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))
