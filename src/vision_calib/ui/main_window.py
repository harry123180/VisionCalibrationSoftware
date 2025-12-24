"""
主應用程式視窗

提供 vision-calib 的主要圖形介面，整合所有標定功能。
採用 Google Material Design 3 設計語言。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QSize, Signal, Slot
from PySide6.QtGui import QAction, QIcon, QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QSpacerItem,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from vision_calib import __version__
from vision_calib.core.types import CalibrationResult
from vision_calib.io import CalibrationFile
from vision_calib.ui.styles.theme import Theme, ThemeManager
from vision_calib.utils.logging import get_logger, setup_logging

logger = get_logger("ui.main_window")


class MainWindow(QMainWindow):
    """主應用程式視窗"""

    def __init__(self):
        super().__init__()

        # 主題管理器
        self.theme_manager = ThemeManager()

        self.setWindowTitle(f"Vision Calib v{__version__}")
        self.setMinimumSize(800, 500)
        self.resize(1280, 800)

        # 當前標定結果
        self._result: Optional[CalibrationResult] = None

        # 背景工作執行緒
        self._corner_worker = None
        self._calib_worker = None

        # 角點偵測結果快取 {image_path: corners}
        self._corner_cache: dict = {}

        # 外參標定結果
        self._extrinsic_result = None

        # 座標轉換器
        self._transformer = None

        # 設置 UI
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()

        # 套用主題
        self.theme_manager.apply_current_theme()

        logger.info("主視窗初始化完成")

    def _setup_ui(self):
        """設置主要 UI 佈局"""
        # 中央組件
        central = QWidget()
        self.setCentralWidget(central)

        # 主佈局
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)

        # 左側面板 - 控制區
        left_panel = self._create_control_panel()
        splitter.addWidget(left_panel)

        # 右側面板 - 標籤頁
        right_panel = self._create_tab_panel()
        splitter.addWidget(right_panel)

        # 設置分割比例
        splitter.setSizes([320, 1080])
        splitter.setStretchFactor(0, 0)  # 左側固定
        splitter.setStretchFactor(1, 1)  # 右側可伸縮
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        main_layout.addWidget(splitter)

    def _create_control_panel(self) -> QWidget:
        """建立左側控制面板（含捲動支援）"""
        from PySide6.QtWidgets import (
            QGroupBox,
            QFormLayout,
            QPushButton,
            QSpinBox,
            QDoubleSpinBox,
            QListWidget,
            QScrollArea,
        )

        # 外層容器
        container = QFrame()
        container.setObjectName("controlPanel")
        container.setMinimumWidth(260)
        container.setMaximumWidth(380)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # 捲動區域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)

        # 內容面板
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # ===== 標題 =====
        title_label = QLabel("相機標定工具")
        title_label.setProperty("heading", True)
        layout.addWidget(title_label)

        subtitle_label = QLabel("使用棋盤格圖案進行相機校正")
        subtitle_label.setProperty("subheading", True)
        subtitle_label.setWordWrap(True)
        layout.addWidget(subtitle_label)

        layout.addSpacing(4)

        # ===== 棋盤格設定 =====
        cb_group = QGroupBox("棋盤格參數")
        cb_layout = QFormLayout(cb_group)
        cb_layout.setSpacing(8)
        cb_layout.setContentsMargins(12, 20, 12, 12)

        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(2, 50)
        self.cols_spin.setValue(17)
        self.cols_spin.setToolTip("棋盤格內部角點的水平數量（與原工具「寬度」相同）")
        cb_layout.addRow("寬度(角點數)：", self.cols_spin)

        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(2, 50)
        self.rows_spin.setValue(12)
        self.rows_spin.setToolTip("棋盤格內部角點的垂直數量（與原工具「高度」相同）")
        cb_layout.addRow("高度(角點數)：", self.rows_spin)

        self.square_size_spin = QDoubleSpinBox()
        self.square_size_spin.setRange(0.1, 100.0)
        self.square_size_spin.setValue(1.0)
        self.square_size_spin.setDecimals(2)
        self.square_size_spin.setSuffix(" cm")
        self.square_size_spin.setToolTip("棋盤格每個方格的實際邊長（公分）")
        cb_layout.addRow("方格邊長：", self.square_size_spin)

        layout.addWidget(cb_group)

        # ===== 圖像列表 =====
        img_group = QGroupBox("標定圖像")
        img_layout = QVBoxLayout(img_group)
        img_layout.setContentsMargins(12, 20, 12, 12)
        img_layout.setSpacing(8)

        self.image_list = QListWidget()
        self.image_list.setMinimumHeight(100)
        self.image_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_list.setToolTip("已載入的標定圖像列表，點擊可預覽")
        self.image_list.currentItemChanged.connect(self._on_image_selected)
        img_layout.addWidget(self.image_list)

        # 圖像操作按鈕
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self.add_images_btn = QPushButton("新增圖像")
        self.add_images_btn.clicked.connect(self._on_add_images)
        self.add_images_btn.setToolTip("選擇標定用的圖像檔案")
        btn_layout.addWidget(self.add_images_btn)

        self.clear_images_btn = QPushButton("清除全部")
        self.clear_images_btn.setProperty("secondary", True)
        self.clear_images_btn.clicked.connect(self._on_clear_images)
        self.clear_images_btn.setToolTip("移除所有已載入的圖像")
        btn_layout.addWidget(self.clear_images_btn)

        img_layout.addLayout(btn_layout)
        layout.addWidget(img_group, 1)  # stretch factor 1

        # ===== 操作按鈕 =====
        action_group = QGroupBox("執行操作")
        action_layout = QVBoxLayout(action_group)
        action_layout.setContentsMargins(12, 20, 12, 12)
        action_layout.setSpacing(8)

        self.detect_btn = QPushButton("偵測角點")
        self.detect_btn.clicked.connect(self._on_detect_corners)
        self.detect_btn.setToolTip("在所有圖像中偵測棋盤格角點")
        action_layout.addWidget(self.detect_btn)

        self.calibrate_btn = QPushButton("執行標定")
        self.calibrate_btn.clicked.connect(self._on_calibrate)
        self.calibrate_btn.setToolTip("計算相機內參矩陣與畸變係數")
        action_layout.addWidget(self.calibrate_btn)

        self.export_btn = QPushButton("匯出結果")
        self.export_btn.clicked.connect(self._on_export)
        self.export_btn.setEnabled(False)
        self.export_btn.setToolTip("將標定結果儲存為檔案")
        action_layout.addWidget(self.export_btn)

        layout.addWidget(action_group)

        # ===== 進度條 =====
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        # 設置捲動區域
        scroll.setWidget(panel)
        container_layout.addWidget(scroll)

        return container

    def _create_tab_panel(self) -> QWidget:
        """建立右側標籤頁面板"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self._create_intrinsic_tab(), "內參標定")
        self.tab_widget.addTab(self._create_extrinsic_tab(), "外參計算")
        self.tab_widget.addTab(self._create_transform_tab(), "座標轉換")

        layout.addWidget(self.tab_widget)
        return panel

    def _create_intrinsic_tab(self) -> QWidget:
        """建立內參標定頁籤"""
        from PySide6.QtWidgets import QTextEdit, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
        from PySide6.QtGui import QPixmap

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 上下分割
        splitter = QSplitter(Qt.Vertical)

        # ===== 上方：圖像預覽 =====
        preview_container = QFrame()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(4)

        preview_header = QLabel("圖像預覽")
        preview_header.setProperty("subheading", True)
        preview_layout.addWidget(preview_header)

        # 圖像檢視器
        self.image_scene = QGraphicsScene()
        self.image_view = QGraphicsView(self.image_scene)
        self.image_view.setMinimumHeight(200)
        self.image_view.setStyleSheet("""
            QGraphicsView {
                border: 1px solid #dadce0;
                border-radius: 8px;
                background-color: #f8f9fa;
            }
        """)
        from PySide6.QtGui import QPainter
        self.image_view.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        self.image_pixmap_item = None
        preview_layout.addWidget(self.image_view)

        self.image_info_label = QLabel("點擊左側列表中的圖像進行預覽")
        self.image_info_label.setProperty("subheading", True)
        self.image_info_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.image_info_label)

        splitter.addWidget(preview_container)

        # ===== 下方：標定結果 =====
        result_container = QFrame()
        result_layout = QVBoxLayout(result_container)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(4)

        result_header = QLabel("標定結果")
        result_header.setProperty("subheading", True)
        result_layout.addWidget(result_header)

        self.intrinsic_view = QTextEdit()
        self.intrinsic_view.setReadOnly(True)
        self.intrinsic_view.setPlaceholderText(
            "請依照以下步驟進行相機標定：\n\n"
            "① 設定棋盤格參數（行數、列數、方格邊長）\n"
            "② 點擊「新增圖像」載入標定照片\n"
            "③ 點擊「偵測角點」自動識別\n"
            "④ 點擊「執行標定」計算參數\n"
            "⑤ 點擊「匯出結果」儲存"
        )
        result_layout.addWidget(self.intrinsic_view)

        splitter.addWidget(result_container)

        # 設置分割比例 (60% 圖像, 40% 結果)
        splitter.setSizes([400, 250])

        layout.addWidget(splitter)
        return widget

    def _create_extrinsic_tab(self) -> QWidget:
        """建立外參計算頁籤"""
        from PySide6.QtWidgets import (
            QTextEdit, QGroupBox, QComboBox, QPushButton,
            QFormLayout, QGraphicsView, QGraphicsScene,
        )

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 說明
        header = QLabel("外參計算")
        header.setProperty("heading", True)
        layout.addWidget(header)

        desc = QLabel("選擇一張圖像作為世界座標系原點（棋盤格左上角）")
        desc.setProperty("subheading", True)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # 設定區
        settings_group = QGroupBox("外參設定")
        settings_layout = QFormLayout(settings_group)
        settings_layout.setContentsMargins(12, 20, 12, 12)
        settings_layout.setSpacing(10)

        # 圖像選擇下拉選單
        self.ext_image_combo = QComboBox()
        self.ext_image_combo.setToolTip("選擇要用於外參計算的圖像")
        self.ext_image_combo.addItem("-- 請先完成內參標定 --")
        settings_layout.addRow("定位圖像：", self.ext_image_combo)

        # 執行按鈕
        self.ext_calibrate_btn = QPushButton("計算外參")
        self.ext_calibrate_btn.clicked.connect(self._on_calibrate_extrinsic)
        self.ext_calibrate_btn.setEnabled(False)
        settings_layout.addRow("", self.ext_calibrate_btn)

        layout.addWidget(settings_group)

        # 結果顯示區
        result_group = QGroupBox("外參結果")
        result_layout = QVBoxLayout(result_group)
        result_layout.setContentsMargins(12, 20, 12, 12)

        self.extrinsic_view = QTextEdit()
        self.extrinsic_view.setReadOnly(True)
        self.extrinsic_view.setPlaceholderText(
            "外參標定流程：\n\n"
            "① 先完成內參標定\n"
            "② 選擇一張圖像定義世界座標系\n"
            "③ 點擊「計算外參」\n\n"
            "結果將包含：\n"
            "• 旋轉向量 / 旋轉矩陣\n"
            "• 平移向量\n"
            "• 相機在世界座標系中的位置"
        )
        result_layout.addWidget(self.extrinsic_view)

        layout.addWidget(result_group, 1)

        return widget

    def _create_transform_tab(self) -> QWidget:
        """建立座標轉換頁籤"""
        from PySide6.QtWidgets import (
            QTextEdit, QGroupBox, QDoubleSpinBox, QPushButton,
            QFormLayout, QGridLayout,
        )

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QLabel("座標轉換")
        header.setProperty("heading", True)
        layout.addWidget(header)

        # ===== 像素 → 世界 =====
        p2w_group = QGroupBox("像素 → 世界座標")
        p2w_layout = QFormLayout(p2w_group)
        p2w_layout.setContentsMargins(12, 20, 12, 12)
        p2w_layout.setSpacing(10)

        # 輸入像素座標
        pixel_input_layout = QHBoxLayout()
        self.pixel_u_spin = QDoubleSpinBox()
        self.pixel_u_spin.setRange(0, 10000)
        self.pixel_u_spin.setDecimals(1)
        self.pixel_u_spin.setSuffix(" px")
        pixel_input_layout.addWidget(QLabel("U:"))
        pixel_input_layout.addWidget(self.pixel_u_spin)

        self.pixel_v_spin = QDoubleSpinBox()
        self.pixel_v_spin.setRange(0, 10000)
        self.pixel_v_spin.setDecimals(1)
        self.pixel_v_spin.setSuffix(" px")
        pixel_input_layout.addWidget(QLabel("V:"))
        pixel_input_layout.addWidget(self.pixel_v_spin)
        p2w_layout.addRow("像素座標：", pixel_input_layout)

        # Z 平面
        self.world_z_spin = QDoubleSpinBox()
        self.world_z_spin.setRange(-10000, 10000)
        self.world_z_spin.setDecimals(2)
        self.world_z_spin.setValue(0)
        self.world_z_spin.setSuffix(" mm")
        self.world_z_spin.setToolTip("目標平面的 Z 座標（0 = 棋盤格平面）")
        p2w_layout.addRow("世界 Z 平面：", self.world_z_spin)

        # 轉換按鈕
        self.p2w_btn = QPushButton("轉換為世界座標")
        self.p2w_btn.clicked.connect(self._on_pixel_to_world)
        self.p2w_btn.setEnabled(False)
        p2w_layout.addRow("", self.p2w_btn)

        # 結果
        self.p2w_result = QLabel("X: -- mm, Y: -- mm")
        self.p2w_result.setStyleSheet("font-weight: bold; font-size: 16px; padding: 8px;")
        p2w_layout.addRow("結果：", self.p2w_result)

        layout.addWidget(p2w_group)

        # ===== 世界 → 像素 =====
        w2p_group = QGroupBox("世界 → 像素座標")
        w2p_layout = QFormLayout(w2p_group)
        w2p_layout.setContentsMargins(12, 20, 12, 12)
        w2p_layout.setSpacing(10)

        # 輸入世界座標
        world_input_layout = QHBoxLayout()
        self.world_x_spin = QDoubleSpinBox()
        self.world_x_spin.setRange(-10000, 10000)
        self.world_x_spin.setDecimals(2)
        self.world_x_spin.setSuffix(" mm")
        world_input_layout.addWidget(QLabel("X:"))
        world_input_layout.addWidget(self.world_x_spin)

        self.world_y_spin = QDoubleSpinBox()
        self.world_y_spin.setRange(-10000, 10000)
        self.world_y_spin.setDecimals(2)
        self.world_y_spin.setSuffix(" mm")
        world_input_layout.addWidget(QLabel("Y:"))
        world_input_layout.addWidget(self.world_y_spin)

        self.world_z_input_spin = QDoubleSpinBox()
        self.world_z_input_spin.setRange(-10000, 10000)
        self.world_z_input_spin.setDecimals(2)
        self.world_z_input_spin.setValue(0)
        self.world_z_input_spin.setSuffix(" mm")
        world_input_layout.addWidget(QLabel("Z:"))
        world_input_layout.addWidget(self.world_z_input_spin)
        w2p_layout.addRow("世界座標：", world_input_layout)

        # 轉換按鈕
        self.w2p_btn = QPushButton("轉換為像素座標")
        self.w2p_btn.clicked.connect(self._on_world_to_pixel)
        self.w2p_btn.setEnabled(False)
        w2p_layout.addRow("", self.w2p_btn)

        # 結果
        self.w2p_result = QLabel("U: -- px, V: -- px")
        self.w2p_result.setStyleSheet("font-weight: bold; font-size: 16px; padding: 8px;")
        w2p_layout.addRow("結果：", self.w2p_result)

        layout.addWidget(w2p_group)

        # 狀態提示
        self.transform_status = QLabel("請先完成內參和外參標定")
        self.transform_status.setProperty("subheading", True)
        self.transform_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.transform_status)

        layout.addStretch()

        return widget

    def _setup_menu(self):
        """設置選單列"""
        menubar = self.menuBar()

        # ===== 檔案選單 =====
        file_menu = menubar.addMenu("檔案(&F)")

        open_action = QAction("開啟標定檔(&O)...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open)
        file_menu.addAction(open_action)

        save_action = QAction("儲存標定檔(&S)...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_export)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("結束(&X)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ===== 檢視選單 =====
        view_menu = menubar.addMenu("檢視(&V)")

        self.theme_action = QAction("切換深色模式", self)
        self.theme_action.setShortcut("Ctrl+T")
        self.theme_action.triggered.connect(self._on_toggle_theme)
        view_menu.addAction(self.theme_action)

        # ===== 說明選單 =====
        help_menu = menubar.addMenu("說明(&H)")

        about_action = QAction("關於(&A)", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self):
        """設置工具列"""
        toolbar = QToolBar("主工具列")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(toolbar)

        # 工具列按鈕
        add_action = toolbar.addAction("新增圖像")
        add_action.setToolTip("載入標定用圖像 (Ctrl+I)")
        add_action.triggered.connect(self._on_add_images)

        toolbar.addSeparator()

        detect_action = toolbar.addAction("偵測角點")
        detect_action.setToolTip("偵測棋盤格角點")
        detect_action.triggered.connect(self._on_detect_corners)

        calibrate_action = toolbar.addAction("執行標定")
        calibrate_action.setToolTip("計算相機參數")
        calibrate_action.triggered.connect(self._on_calibrate)

        toolbar.addSeparator()

        export_action = toolbar.addAction("匯出")
        export_action.setToolTip("匯出標定結果")
        export_action.triggered.connect(self._on_export)

        # 彈性空間 - 將主題指示器推到右側
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        # 主題指示器（僅顯示，從選單切換）
        self.theme_indicator = QLabel("🌙")
        self.theme_indicator.setToolTip("目前為明亮模式（從「檢視」選單切換）")
        self.theme_indicator.setStyleSheet("font-size: 18px; padding: 4px 12px;")
        toolbar.addWidget(self.theme_indicator)

    def _setup_statusbar(self):
        """設置狀態列"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("就緒")

    # ===== 事件處理 =====

    @Slot()
    def _on_toggle_theme(self):
        """切換主題"""
        self.theme_manager.toggle_theme()

        if self.theme_manager.is_dark:
            self.theme_indicator.setText("☀️")
            self.theme_indicator.setToolTip("目前為深色模式（從「檢視」選單切換）")
            self.theme_action.setText("切換明亮模式")
        else:
            self.theme_indicator.setText("🌙")
            self.theme_indicator.setToolTip("目前為明亮模式（從「檢視」選單切換）")
            self.theme_action.setText("切換深色模式")

    @Slot()
    def _on_add_images(self):
        """處理新增圖像"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "選擇標定圖像",
            "",
            "圖像檔案 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有檔案 (*)",
        )

        if files:
            for f in files:
                self.image_list.addItem(Path(f).name)
                item = self.image_list.item(self.image_list.count() - 1)
                item.setData(Qt.UserRole, f)

            self.statusbar.showMessage(f"已載入 {len(files)} 張圖像")
            logger.info(f"載入 {len(files)} 張圖像")

    @Slot()
    def _on_clear_images(self):
        """清除所有圖像"""
        self.image_list.clear()
        self._corner_cache.clear()
        self._clear_image_preview()
        self.statusbar.showMessage("已清除所有圖像")

    @Slot()
    def _on_image_selected(self, current, previous):
        """處理圖像選擇變更"""
        if current is None:
            self._clear_image_preview()
            return

        image_path = current.data(Qt.UserRole)
        if image_path:
            self._display_image(image_path)

    def _display_image(self, image_path: str):
        """顯示圖像（含角點標記）"""
        from PySide6.QtGui import QPixmap, QImage
        import cv2
        import numpy as np

        try:
            # 讀取圖像（支援中文路徑）
            with open(image_path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)

            if img is None:
                self.image_info_label.setText("無法載入圖像")
                return

            # 如果有角點資料，繪製角點
            if image_path in self._corner_cache:
                corners = self._corner_cache[image_path]
                if corners is not None:
                    cv2.drawChessboardCorners(
                        img,
                        (self.cols_spin.value(), self.rows_spin.value()),
                        corners,
                        True
                    )

            # 轉換為 QPixmap
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            # 顯示在場景中
            self.image_scene.clear()
            self.image_pixmap_item = self.image_scene.addPixmap(pixmap)
            self.image_scene.setSceneRect(pixmap.rect().toRectF())

            # 自適應縮放
            self.image_view.fitInView(
                self.image_scene.sceneRect(),
                Qt.KeepAspectRatio
            )

            # 更新資訊標籤
            filename = Path(image_path).name
            corner_status = "（已偵測角點）" if image_path in self._corner_cache else ""
            self.image_info_label.setText(f"{filename} - {w}×{h} {corner_status}")

        except Exception as e:
            self.image_info_label.setText(f"載入失敗：{e}")
            logger.error(f"載入圖像失敗：{e}")

    def _clear_image_preview(self):
        """清除圖像預覽"""
        self.image_scene.clear()
        self.image_pixmap_item = None
        self.image_info_label.setText("點擊左側列表中的圖像進行預覽")

    @Slot()
    def _on_detect_corners(self):
        """偵測角點（背景執行緒）"""
        if self.image_list.count() == 0:
            QMessageBox.warning(self, "提示", "請先載入圖像")
            return

        from vision_calib.core.types import CheckerboardConfig
        from vision_calib.utils.worker import CornerDetectionWorker

        config = CheckerboardConfig(
            rows=self.rows_spin.value(),
            cols=self.cols_spin.value(),
            square_size_mm=self.square_size_spin.value() * 10,  # cm → mm
        )

        # 取得圖像路徑
        paths = []
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            paths.append(item.data(Qt.UserRole))

        # 禁用按鈕
        self._set_buttons_enabled(False)

        # 顯示進度條
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(paths))
        self.progress_bar.setValue(0)

        # 建立並啟動工作執行緒
        self._corner_worker = CornerDetectionWorker(paths, config, self)
        self._corner_worker.progress.connect(self._on_corner_progress)
        self._corner_worker.single_result.connect(self._on_corner_single_result)
        self._corner_worker.finished.connect(self._on_corner_finished)
        self._corner_worker.error.connect(self._on_corner_error)
        self._corner_worker.start()

    @Slot(int, int, str)
    def _on_corner_progress(self, current: int, total: int, message: str):
        """角點偵測進度更新"""
        self.progress_bar.setValue(current)
        self.statusbar.showMessage(message)

    @Slot(object)
    def _on_corner_single_result(self, result):
        """單張圖像角點偵測結果"""
        item = self.image_list.item(result.index)
        if item:
            filename = Path(result.image_path).name
            if result.success:
                item.setText(f"✓ {filename}")
                # 儲存角點到快取
                self._corner_cache[result.image_path] = result.corners
            else:
                item.setText(f"✗ {filename}")
                self._corner_cache[result.image_path] = None

            # 如果當前選中的是這張圖，更新預覽
            current_item = self.image_list.currentItem()
            if current_item and current_item.data(Qt.UserRole) == result.image_path:
                self._display_image(result.image_path)

    @Slot(int, int)
    def _on_corner_finished(self, success_count: int, total_count: int):
        """角點偵測完成"""
        self.progress_bar.setVisible(False)
        self._set_buttons_enabled(True)
        self.statusbar.showMessage(f"角點偵測完成：{success_count}/{total_count} 張成功")
        self._corner_worker = None

    @Slot(str)
    def _on_corner_error(self, error_msg: str):
        """角點偵測錯誤"""
        self.progress_bar.setVisible(False)
        self._set_buttons_enabled(True)
        QMessageBox.critical(self, "錯誤", f"角點偵測失敗：{error_msg}")
        self._corner_worker = None

    @Slot()
    def _on_calibrate(self):
        """執行標定（背景執行緒）"""
        if self.image_list.count() == 0:
            QMessageBox.warning(self, "提示", "請先載入圖像")
            return

        from vision_calib.core.intrinsic import IntrinsicCalibrationConfig
        from vision_calib.core.types import CheckerboardConfig
        from vision_calib.utils.worker import CalibrationWorker

        config = IntrinsicCalibrationConfig(
            checkerboard=CheckerboardConfig(
                rows=self.rows_spin.value(),
                cols=self.cols_spin.value(),
                square_size_mm=self.square_size_spin.value() * 10,  # cm → mm
            )
        )

        # 取得圖像路徑
        paths = []
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            paths.append(item.data(Qt.UserRole))

        # 禁用按鈕
        self._set_buttons_enabled(False)

        # 顯示進度條
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.statusbar.showMessage("正在計算標定參數...")

        # 建立並啟動工作執行緒
        self._calib_worker = CalibrationWorker(paths, config, self)
        self._calib_worker.progress.connect(self._on_calib_progress)
        self._calib_worker.finished.connect(self._on_calib_finished)
        self._calib_worker.error.connect(self._on_calib_error)
        self._calib_worker.start()

    @Slot(int, int, str)
    def _on_calib_progress(self, current: int, total: int, message: str):
        """標定進度更新"""
        pct = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.statusbar.showMessage(message)

    @Slot(object)
    def _on_calib_finished(self, result):
        """標定完成"""
        self.progress_bar.setVisible(False)
        self._set_buttons_enabled(True)
        self.statusbar.showMessage("就緒")
        self._calib_worker = None

        self._result = result
        self._display_calibration_result(result)
        self.export_btn.setEnabled(True)

        # 更新外參標定的圖像選擇下拉選單
        self._update_extrinsic_image_combo()

        QMessageBox.information(
            self,
            "標定完成",
            f"重投影誤差：{result.intrinsic.reprojection_error:.4f} 像素\n\n"
            f"相機矩陣和畸變係數已計算完成。\n"
            f"請點擊「匯出結果」儲存標定資料。",
        )

    @Slot(str)
    def _on_calib_error(self, error_msg: str):
        """標定錯誤"""
        self.progress_bar.setVisible(False)
        self._set_buttons_enabled(True)
        self.statusbar.showMessage("就緒")
        QMessageBox.critical(self, "錯誤", f"標定失敗：{error_msg}")
        logger.error(f"標定失敗：{error_msg}")
        self._calib_worker = None

    def _set_buttons_enabled(self, enabled: bool):
        """啟用/禁用操作按鈕"""
        self.detect_btn.setEnabled(enabled)
        self.calibrate_btn.setEnabled(enabled)
        self.add_images_btn.setEnabled(enabled)
        self.clear_images_btn.setEnabled(enabled)
        if enabled and self._result is not None:
            self.export_btn.setEnabled(True)
        elif not enabled:
            self.export_btn.setEnabled(False)

    def _display_calibration_result(self, result: CalibrationResult):
        """顯示標定結果"""
        intrinsic = result.intrinsic

        text = f"""標定完成！

══════════════════════════════════════
　相機內參矩陣 (Camera Matrix K)
══════════════════════════════════════

　　┌　{intrinsic.camera_matrix[0,0]:12.4f}　{intrinsic.camera_matrix[0,1]:12.4f}　{intrinsic.camera_matrix[0,2]:12.4f}　┐
　　│　{intrinsic.camera_matrix[1,0]:12.4f}　{intrinsic.camera_matrix[1,1]:12.4f}　{intrinsic.camera_matrix[1,2]:12.4f}　│
　　└　{intrinsic.camera_matrix[2,0]:12.4f}　{intrinsic.camera_matrix[2,1]:12.4f}　{intrinsic.camera_matrix[2,2]:12.4f}　┘

══════════════════════════════════════
　相機參數
══════════════════════════════════════

　　焦距 (fx)：{intrinsic.fx:.2f} pixels
　　焦距 (fy)：{intrinsic.fy:.2f} pixels
　　主點 (cx)：{intrinsic.cx:.2f} pixels
　　主點 (cy)：{intrinsic.cy:.2f} pixels

══════════════════════════════════════
　畸變係數 (Distortion Coefficients)
══════════════════════════════════════

　　k1 = {intrinsic.distortion_coeffs[0]:+.6f}
　　k2 = {intrinsic.distortion_coeffs[1]:+.6f}
　　p1 = {intrinsic.distortion_coeffs[2]:+.6f}
　　p2 = {intrinsic.distortion_coeffs[3]:+.6f}
　　k3 = {intrinsic.distortion_coeffs[4]:+.6f}

══════════════════════════════════════
　標定品質
══════════════════════════════════════

　　重投影誤差：{intrinsic.reprojection_error:.4f} pixels
　　圖像尺寸：{intrinsic.image_size[0]} × {intrinsic.image_size[1]}
　　使用圖像數：{result.num_images_used}
"""
        self.intrinsic_view.setText(text)

    @Slot()
    def _on_export(self):
        """匯出標定結果"""
        if self._result is None:
            QMessageBox.warning(self, "提示", "尚無標定結果可匯出")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "匯出標定結果",
            "calibration",
            "HDF5 檔案 (*.h5);;MAT 檔案 (*.mat);;JSON 檔案 (*.json);;所有檔案 (*)",
        )

        if file_path:
            try:
                CalibrationFile.save(file_path, self._result)
                self.statusbar.showMessage(f"已匯出至：{file_path}")
                QMessageBox.information(
                    self,
                    "匯出成功",
                    f"標定結果已儲存至：\n{file_path}",
                )
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"匯出失敗：{e}")

    @Slot()
    def _on_open(self):
        """開啟標定檔案"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "開啟標定檔案",
            "",
            "標定檔案 (*.h5 *.mat *.json);;所有檔案 (*)",
        )

        if file_path:
            try:
                self._result = CalibrationFile.load(file_path)
                self._display_calibration_result(self._result)
                self.export_btn.setEnabled(True)
                self.statusbar.showMessage(f"已載入：{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"無法載入檔案：{e}")

    @Slot()
    def _on_about(self):
        """顯示關於對話框"""
        QMessageBox.about(
            self,
            "關於 Vision Calib",
            f"""<h2>Vision Calib</h2>
            <p>版本 {__version__}</p>
            <hr>
            <p>專業相機標定工具</p>
            <p>使用棋盤格圖案進行相機內外參標定，<br>
            支援像素、相機、世界座標系之間的轉換。</p>
            <hr>
            <p><b>授權條款：</b>Apache License 2.0</p>
            <p><b>原始碼：</b><a href='https://github.com/yourusername/vision-calib'>GitHub</a></p>
            """,
        )

    def resizeEvent(self, event):
        """視窗縮放時調整圖像顯示"""
        super().resizeEvent(event)
        # 重新調整圖像檢視器的縮放
        if hasattr(self, 'image_scene') and self.image_scene.items():
            self.image_view.fitInView(
                self.image_scene.sceneRect(),
                Qt.KeepAspectRatio
            )

    def _update_extrinsic_image_combo(self):
        """更新外參標定的圖像選擇下拉選單"""
        self.ext_image_combo.clear()

        # 僅列出偵測到角點的圖像
        valid_images = []
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            image_path = item.data(Qt.UserRole)
            if image_path in self._corner_cache and self._corner_cache[image_path] is not None:
                valid_images.append((Path(image_path).name, image_path))

        if not valid_images:
            self.ext_image_combo.addItem("-- 無有效圖像 --")
            self.ext_calibrate_btn.setEnabled(False)
            return

        for name, path in valid_images:
            self.ext_image_combo.addItem(name, path)

        self.ext_calibrate_btn.setEnabled(True)
        self.statusbar.showMessage(f"可使用 {len(valid_images)} 張圖像進行外參標定")

    @Slot()
    def _on_calibrate_extrinsic(self):
        """執行外參標定"""
        if self._result is None:
            QMessageBox.warning(self, "提示", "請先完成內參標定")
            return

        # 取得選中的圖像
        image_path = self.ext_image_combo.currentData()
        if not image_path:
            QMessageBox.warning(self, "提示", "請選擇一張定位圖像")
            return

        # 取得角點
        corners = self._corner_cache.get(image_path)
        if corners is None:
            QMessageBox.warning(self, "錯誤", "該圖像沒有有效的角點資料")
            return

        from vision_calib.core.extrinsic import ExtrinsicCalibrator
        from vision_calib.core.transform import CoordinateTransformer
        from vision_calib.core.types import CheckerboardConfig

        try:
            self.statusbar.showMessage("正在計算外參...")

            # 建立棋盤格設定
            checkerboard = CheckerboardConfig(
                rows=self.rows_spin.value(),
                cols=self.cols_spin.value(),
                square_size_mm=self.square_size_spin.value() * 10,  # cm → mm
            )

            # 執行外參標定
            calibrator = ExtrinsicCalibrator(
                intrinsic=self._result.intrinsic,
                checkerboard=checkerboard,
            )
            self._extrinsic_result = calibrator.calibrate(image_path, corners)

            # 建立座標轉換器
            self._transformer = CoordinateTransformer(
                intrinsic=self._result.intrinsic,
                extrinsic=self._extrinsic_result.extrinsic,
            )

            # 顯示結果
            self._display_extrinsic_result()

            # 啟用座標轉換功能
            self._enable_transform_buttons()

            self.statusbar.showMessage("外參標定完成")
            QMessageBox.information(
                self,
                "外參標定完成",
                f"重投影誤差：{self._extrinsic_result.reprojection_error:.4f} 像素\n\n"
                f"現在可以使用座標轉換功能。",
            )

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"外參標定失敗：{e}")
            logger.error(f"外參標定失敗：{e}")

    def _display_extrinsic_result(self):
        """顯示外參標定結果"""
        if self._extrinsic_result is None:
            return

        result = self._extrinsic_result
        self.extrinsic_view.setText(result.summary())

    def _enable_transform_buttons(self):
        """啟用座標轉換按鈕"""
        self.p2w_btn.setEnabled(True)
        self.w2p_btn.setEnabled(True)
        self.transform_status.setText("座標轉換功能已就緒")

    @Slot()
    def _on_pixel_to_world(self):
        """像素座標 → 世界座標"""
        if self._transformer is None:
            QMessageBox.warning(self, "提示", "請先完成外參標定")
            return

        import numpy as np

        try:
            # 取得輸入
            u = self.pixel_u_spin.value()
            v = self.pixel_v_spin.value()
            z_world = self.world_z_spin.value()

            # 轉換
            pixel = np.array([u, v])
            world = self._transformer.pixel_to_world(pixel, z_world)

            # 顯示結果
            x, y, z = world[0], world[1], world[2]
            self.p2w_result.setText(f"X: {x:.2f} mm, Y: {y:.2f} mm, Z: {z:.2f} mm")
            self.statusbar.showMessage(f"像素 ({u:.1f}, {v:.1f}) → 世界 ({x:.2f}, {y:.2f}, {z:.2f})")

        except Exception as e:
            self.p2w_result.setText(f"轉換失敗：{e}")
            logger.error(f"像素→世界轉換失敗：{e}")

    @Slot()
    def _on_world_to_pixel(self):
        """世界座標 → 像素座標"""
        if self._transformer is None:
            QMessageBox.warning(self, "提示", "請先完成外參標定")
            return

        import numpy as np

        try:
            # 取得輸入
            x = self.world_x_spin.value()
            y = self.world_y_spin.value()
            z = self.world_z_input_spin.value()

            # 轉換
            world = np.array([x, y, z])
            pixel = self._transformer.world_to_pixel(world)

            # 顯示結果
            u, v = pixel[0], pixel[1]
            self.w2p_result.setText(f"U: {u:.1f} px, V: {v:.1f} px")
            self.statusbar.showMessage(f"世界 ({x:.2f}, {y:.2f}, {z:.2f}) → 像素 ({u:.1f}, {v:.1f})")

        except Exception as e:
            self.w2p_result.setText(f"轉換失敗：{e}")
            logger.error(f"世界→像素轉換失敗：{e}")


def main():
    """應用程式入口點"""
    setup_logging()

    app = QApplication(sys.argv)
    app.setApplicationName("Vision Calib")
    app.setApplicationVersion(__version__)

    # 設置預設字型
    font = QFont()
    font.setFamily("Microsoft JhengHei UI")
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    logger.info("應用程式啟動")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
