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

        self.setWindowTitle(f"TSIC/CR-ICS01 相機與手臂整合控制教學軟體 v{__version__}")
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

        # 點位數據 [id, image_x, image_y, world_x, world_y]
        self._point_data: list = []

        # 生成的世界座標
        self._generated_world_coords: list = []

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
        self.cols_spin.valueChanged.connect(self._on_global_param_changed)
        cb_layout.addRow("寬度(角點數)：", self.cols_spin)

        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(2, 50)
        self.rows_spin.setValue(12)
        self.rows_spin.setToolTip("棋盤格內部角點的垂直數量（與原工具「高度」相同）")
        self.rows_spin.valueChanged.connect(self._on_global_param_changed)
        cb_layout.addRow("高度(角點數)：", self.rows_spin)

        self.square_size_spin = QDoubleSpinBox()
        self.square_size_spin.setRange(0.1, 100.0)
        self.square_size_spin.setValue(1.0)
        self.square_size_spin.setDecimals(2)
        self.square_size_spin.setSuffix(" cm")
        self.square_size_spin.setToolTip("棋盤格每個方格的實際邊長（公分）")
        self.square_size_spin.valueChanged.connect(self._on_global_param_changed)
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
        self.tab_widget.addTab(self._create_points_tab(), "點位數據")
        self.tab_widget.addTab(self._create_chessboard_gen_tab(), "棋盤生成")
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

    def _create_points_tab(self) -> QWidget:
        """建立點位數據管理頁籤"""
        from PySide6.QtWidgets import (
            QTextEdit, QGroupBox, QPushButton,
            QFormLayout, QTableWidget, QTableWidgetItem, QHeaderView,
            QDoubleSpinBox, QSpinBox, QScrollArea,
        )

        # 外層容器
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # 捲動區域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 說明
        header = QLabel("點位數據管理")
        header.setProperty("heading", True)
        layout.addWidget(header)

        desc = QLabel("管理像素座標與世界座標的對應關係，用於外參計算")
        desc.setProperty("subheading", True)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # ===== 手動添加點位 =====
        add_group = QGroupBox("添加點位")
        add_layout = QFormLayout(add_group)
        add_layout.setContentsMargins(12, 20, 12, 12)
        add_layout.setSpacing(8)
        add_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # 點位 ID
        self.point_id_spin = QSpinBox()
        self.point_id_spin.setRange(1, 9999)
        self.point_id_spin.setValue(1)
        self.point_id_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        add_layout.addRow("點位 ID：", self.point_id_spin)

        # 像素座標
        pixel_layout = QHBoxLayout()
        pixel_layout.setSpacing(8)
        self.point_img_x_spin = QDoubleSpinBox()
        self.point_img_x_spin.setRange(0, 10000)
        self.point_img_x_spin.setDecimals(2)
        self.point_img_x_spin.setSuffix(" px")
        self.point_img_x_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        pixel_layout.addWidget(QLabel("X:"))
        pixel_layout.addWidget(self.point_img_x_spin, 1)
        self.point_img_y_spin = QDoubleSpinBox()
        self.point_img_y_spin.setRange(0, 10000)
        self.point_img_y_spin.setDecimals(2)
        self.point_img_y_spin.setSuffix(" px")
        self.point_img_y_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        pixel_layout.addWidget(QLabel("Y:"))
        pixel_layout.addWidget(self.point_img_y_spin, 1)
        add_layout.addRow("像素座標：", pixel_layout)

        # 世界座標
        world_layout = QHBoxLayout()
        world_layout.setSpacing(8)
        self.point_world_x_spin = QDoubleSpinBox()
        self.point_world_x_spin.setRange(-10000, 10000)
        self.point_world_x_spin.setDecimals(2)
        self.point_world_x_spin.setSuffix(" mm")
        self.point_world_x_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        world_layout.addWidget(QLabel("X:"))
        world_layout.addWidget(self.point_world_x_spin, 1)
        self.point_world_y_spin = QDoubleSpinBox()
        self.point_world_y_spin.setRange(-10000, 10000)
        self.point_world_y_spin.setDecimals(2)
        self.point_world_y_spin.setSuffix(" mm")
        self.point_world_y_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        world_layout.addWidget(QLabel("Y:"))
        world_layout.addWidget(self.point_world_y_spin, 1)
        add_layout.addRow("世界座標：", world_layout)

        # 添加按鈕
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        add_point_btn = QPushButton("添加點位")
        add_point_btn.clicked.connect(self._on_add_point)
        add_point_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_layout.addWidget(add_point_btn)

        clear_points_btn = QPushButton("清除全部")
        clear_points_btn.setProperty("secondary", True)
        clear_points_btn.clicked.connect(self._on_clear_points)
        clear_points_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_layout.addWidget(clear_points_btn)
        add_layout.addRow("", btn_layout)

        layout.addWidget(add_group)

        # ===== 點位列表 =====
        list_group = QGroupBox(f"當前點位 (共 0 個)")
        self.points_list_group = list_group
        list_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        list_layout = QVBoxLayout(list_group)
        list_layout.setContentsMargins(12, 20, 12, 12)

        # 表格
        self.points_table = QTableWidget()
        self.points_table.setColumnCount(5)
        self.points_table.setHorizontalHeaderLabels(["ID", "像素X", "像素Y", "世界X", "世界Y"])
        self.points_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.points_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.points_table.setMinimumHeight(150)
        list_layout.addWidget(self.points_table)

        # 匯入/匯出
        io_layout = QHBoxLayout()
        io_layout.setSpacing(8)
        import_csv_btn = QPushButton("匯入CSV")
        import_csv_btn.clicked.connect(self._on_import_points_csv)
        import_csv_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        io_layout.addWidget(import_csv_btn)

        export_csv_btn = QPushButton("匯出CSV")
        export_csv_btn.clicked.connect(self._on_export_points_csv)
        export_csv_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        io_layout.addWidget(export_csv_btn)

        load_from_corners_btn = QPushButton("從角點載入")
        load_from_corners_btn.clicked.connect(self._on_load_corners_to_points)
        load_from_corners_btn.setToolTip("將偵測到的角點載入為像素座標")
        load_from_corners_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        io_layout.addWidget(load_from_corners_btn)

        list_layout.addLayout(io_layout)
        layout.addWidget(list_group, 1)

        scroll.setWidget(widget)
        container_layout.addWidget(scroll)

        return container

    def _create_chessboard_gen_tab(self) -> QWidget:
        """建立棋盤格世界座標生成頁籤 - 簡化版：使用左側全局棋盤格參數"""
        from PySide6.QtWidgets import (
            QTextEdit, QGroupBox, QPushButton,
            QFormLayout, QDoubleSpinBox, QSpinBox, QComboBox,
            QCheckBox,
        )

        # 主容器 - 上下分割：上方可視化，下方控制
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # ========== 上方：可視化區域（佔主要空間）==========
        vis_container = QWidget()
        vis_layout = QVBoxLayout(vis_container)
        vis_layout.setContentsMargins(0, 0, 0, 0)
        vis_layout.setSpacing(8)

        # 標題行
        header_layout = QHBoxLayout()
        header = QLabel("座標系可視化")
        header.setProperty("heading", True)
        header_layout.addWidget(header)
        header_layout.addStretch()

        # 顯示控制 - 使用明顯的切換按鈕樣式
        toggle_frame = QFrame()
        toggle_frame.setStyleSheet("""
            QFrame {
                background-color: #f1f3f4;
                border-radius: 8px;
                padding: 4px;
            }
        """)
        toggle_layout = QHBoxLayout(toggle_frame)
        toggle_layout.setContentsMargins(4, 4, 4, 4)
        toggle_layout.setSpacing(4)

        toggle_style = """
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
                color: #5f6368;
            }
            QPushButton:checked {
                background-color: #1a73e8;
                color: white;
            }
            QPushButton:hover:!checked {
                background-color: #e8eaed;
            }
        """

        self.show_image_coords_btn = QPushButton("視覺座標")
        self.show_image_coords_btn.setCheckable(True)
        self.show_image_coords_btn.setChecked(True)
        self.show_image_coords_btn.setStyleSheet(toggle_style)
        self.show_image_coords_btn.clicked.connect(self._update_chessboard_vis)
        toggle_layout.addWidget(self.show_image_coords_btn)

        self.show_world_coords_btn = QPushButton("世界座標")
        self.show_world_coords_btn.setCheckable(True)
        self.show_world_coords_btn.setChecked(True)
        self.show_world_coords_btn.setStyleSheet(toggle_style)
        self.show_world_coords_btn.clicked.connect(self._update_chessboard_vis)
        toggle_layout.addWidget(self.show_world_coords_btn)

        self.show_corners_btn = QPushButton("偵測角點")
        self.show_corners_btn.setCheckable(True)
        self.show_corners_btn.setChecked(False)
        self.show_corners_btn.setStyleSheet(toggle_style)
        self.show_corners_btn.clicked.connect(self._update_chessboard_vis)
        toggle_layout.addWidget(self.show_corners_btn)

        header_layout.addWidget(toggle_frame)
        vis_layout.addLayout(header_layout)

        # Matplotlib 畫布
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            import matplotlib
            matplotlib.use('QtAgg')

            # 設置中文字體
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False

            self.chessboard_fig = Figure(figsize=(10, 6), dpi=100)
            self.chessboard_ax = self.chessboard_fig.add_subplot(111)
            self.chessboard_canvas = FigureCanvas(self.chessboard_fig)
            self.chessboard_canvas.setMinimumHeight(300)

            # 啟用滾輪縮放和拖動
            self._chessboard_vis_pressed = False
            self._chessboard_vis_last_x = 0
            self._chessboard_vis_last_y = 0
            self.chessboard_canvas.mpl_connect('scroll_event', self._on_chessboard_scroll)
            self.chessboard_canvas.mpl_connect('button_press_event', self._on_chessboard_press)
            self.chessboard_canvas.mpl_connect('button_release_event', self._on_chessboard_release)
            self.chessboard_canvas.mpl_connect('motion_notify_event', self._on_chessboard_motion)

            vis_layout.addWidget(self.chessboard_canvas, 1)

        except ImportError:
            fallback_label = QLabel("需要安裝 matplotlib 才能顯示座標可視化\npip install matplotlib")
            fallback_label.setAlignment(Qt.AlignCenter)
            fallback_label.setStyleSheet("color: #666; font-size: 14px; padding: 40px;")
            vis_layout.addWidget(fallback_label, 1)
            self.chessboard_canvas = None

        main_layout.addWidget(vis_container, 1)  # stretch=1，佔主要空間

        # ========== 下方：控制區域（精簡）==========
        ctrl_container = QWidget()
        ctrl_layout = QHBoxLayout(ctrl_container)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(16)

        # --- 機械臂原點設定 ---
        origin_group = QGroupBox("機械臂原點")
        origin_layout = QFormLayout(origin_group)
        origin_layout.setContentsMargins(12, 16, 12, 12)
        origin_layout.setSpacing(6)

        # 原點座標 (X, Y)
        origin_coord_layout = QHBoxLayout()
        origin_coord_layout.setSpacing(4)
        self.origin_x_spin = QDoubleSpinBox()
        self.origin_x_spin.setRange(-10000, 10000)
        self.origin_x_spin.setDecimals(1)
        self.origin_x_spin.setValue(0.0)
        self.origin_x_spin.setSuffix(" mm")
        self.origin_x_spin.valueChanged.connect(self._update_chessboard_vis)
        origin_coord_layout.addWidget(QLabel("X:"))
        origin_coord_layout.addWidget(self.origin_x_spin)
        self.origin_y_spin = QDoubleSpinBox()
        self.origin_y_spin.setRange(-10000, 10000)
        self.origin_y_spin.setDecimals(1)
        self.origin_y_spin.setValue(0.0)
        self.origin_y_spin.setSuffix(" mm")
        self.origin_y_spin.valueChanged.connect(self._update_chessboard_vis)
        origin_coord_layout.addWidget(QLabel("Y:"))
        origin_coord_layout.addWidget(self.origin_y_spin)
        origin_layout.addRow("座標：", origin_coord_layout)

        # 對應角點編號
        self.robot_origin_point_spin = QSpinBox()
        self.robot_origin_point_spin.setRange(1, 9999)
        self.robot_origin_point_spin.setValue(204)
        self.robot_origin_point_spin.setToolTip("棋盤格上對應機械臂原點的角點編號")
        self.robot_origin_point_spin.valueChanged.connect(self._update_chessboard_vis)
        origin_layout.addRow("角點 ID：", self.robot_origin_point_spin)

        ctrl_layout.addWidget(origin_group)

        # --- 座標軸方向 ---
        dir_group = QGroupBox("座標軸對應")
        dir_layout = QFormLayout(dir_group)
        dir_layout.setContentsMargins(12, 16, 12, 12)
        dir_layout.setSpacing(6)

        self.robot_x_dir_combo = QComboBox()
        self.robot_x_dir_combo.addItem("圖像Y+ (向下)", "image_y_positive")
        self.robot_x_dir_combo.addItem("圖像Y- (向上)", "image_y_negative")
        self.robot_x_dir_combo.addItem("圖像X+ (向右)", "image_x_positive")
        self.robot_x_dir_combo.addItem("圖像X- (向左)", "image_x_negative")
        self.robot_x_dir_combo.currentIndexChanged.connect(self._update_chessboard_vis)
        dir_layout.addRow("機械臂 X+：", self.robot_x_dir_combo)

        self.robot_y_dir_combo = QComboBox()
        self.robot_y_dir_combo.addItem("圖像X+ (向右)", "image_x_positive")
        self.robot_y_dir_combo.addItem("圖像X- (向左)", "image_x_negative")
        self.robot_y_dir_combo.addItem("圖像Y+ (向下)", "image_y_positive")
        self.robot_y_dir_combo.addItem("圖像Y- (向上)", "image_y_negative")
        self.robot_y_dir_combo.currentIndexChanged.connect(self._update_chessboard_vis)
        dir_layout.addRow("機械臂 Y+：", self.robot_y_dir_combo)

        ctrl_layout.addWidget(dir_group)

        # --- 操作按鈕 + 結果 ---
        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout(action_group)
        action_layout.setContentsMargins(12, 16, 12, 12)
        action_layout.setSpacing(6)

        gen_btn = QPushButton("生成世界座標")
        gen_btn.clicked.connect(self._on_generate_world_coords)
        gen_btn.setToolTip("使用左側「棋盤格參數」生成世界座標")
        action_layout.addWidget(gen_btn)

        load_btn = QPushButton("載入到點位數據")
        load_btn.clicked.connect(self._on_load_world_coords_to_points)
        load_btn.setToolTip("合併角點與世界座標")
        action_layout.addWidget(load_btn)

        self.chessboard_gen_status = QLabel("使用左側棋盤格參數")
        self.chessboard_gen_status.setProperty("subheading", True)
        self.chessboard_gen_status.setWordWrap(True)
        action_layout.addWidget(self.chessboard_gen_status)

        ctrl_layout.addWidget(action_group)

        # --- 結果摘要 ---
        result_group = QGroupBox("生成結果")
        result_layout = QVBoxLayout(result_group)
        result_layout.setContentsMargins(12, 16, 12, 12)

        self.chessboard_gen_result = QTextEdit()
        self.chessboard_gen_result.setReadOnly(True)
        self.chessboard_gen_result.setMaximumHeight(80)
        self.chessboard_gen_result.setPlaceholderText("點擊「生成世界座標」查看結果")
        result_layout.addWidget(self.chessboard_gen_result)

        ctrl_layout.addWidget(result_group, 1)

        main_layout.addWidget(ctrl_container)

        # 初始繪製
        if hasattr(self, 'chessboard_canvas') and self.chessboard_canvas:
            self._init_chessboard_vis()

        return container

    def _on_global_param_changed(self):
        """當左側全局棋盤格參數變更時更新可視化"""
        if hasattr(self, 'chessboard_ax') and self.chessboard_ax is not None:
            self._update_chessboard_vis()

    def _init_chessboard_vis(self):
        """初始化棋盤格可視化"""
        if not hasattr(self, 'chessboard_ax') or self.chessboard_ax is None:
            return
        self._update_chessboard_vis()

    def _update_chessboard_vis(self):
        """更新棋盤格座標可視化"""
        if not hasattr(self, 'chessboard_ax') or self.chessboard_ax is None:
            return

        import numpy as np

        self.chessboard_ax.clear()

        # 獲取當前設置（使用左側全局棋盤格參數）
        try:
            grid_x = self.cols_spin.value()  # 使用全局參數
            grid_y = self.rows_spin.value()  # 使用全局參數
            robot_point = self.robot_origin_point_spin.value()
            spacing = self.square_size_spin.value() * 10  # cm → mm
        except:
            grid_x, grid_y, robot_point, spacing = 17, 12, 204, 10.0

        # ===== 顯示視覺座標系（模擬的棋盤格點位）=====
        if self.show_image_coords_btn.isChecked():
            points_x = []
            points_y = []
            for j in range(grid_y):
                for i in range(grid_x):
                    x = 200 + i * 40
                    y = 150 + j * 40
                    points_x.append(x)
                    points_y.append(y)

            self.chessboard_ax.scatter(points_x, points_y, c='lightblue', s=60, alpha=0.7,
                                       edgecolor='blue', label='視覺座標系點位', marker='s')

            # 標出機械臂原點在視覺座標系中的位置
            if 1 <= robot_point <= len(points_x):
                robot_idx = robot_point - 1
                robot_x = points_x[robot_idx]
                robot_y = points_y[robot_idx]

                self.chessboard_ax.scatter([robot_x], [robot_y], c='red', s=200, marker='*',
                                           edgecolor='darkred', linewidth=2, label=f'機械臂原點 (P{robot_point})')

                # 繪製機械臂座標軸方向
                robot_x_dir = self.robot_x_dir_combo.currentData()
                robot_y_dir = self.robot_y_dir_combo.currentData()

                arrow_length = 80
                robot_x_dx, robot_x_dy = self._get_direction_vector(robot_x_dir, arrow_length)
                robot_y_dx, robot_y_dy = self._get_direction_vector(robot_y_dir, arrow_length)

                self.chessboard_ax.arrow(robot_x, robot_y, robot_x_dx, robot_x_dy,
                                         head_width=12, head_length=15, fc='green', ec='green', linewidth=3)
                self.chessboard_ax.arrow(robot_x, robot_y, robot_y_dx, robot_y_dy,
                                         head_width=12, head_length=15, fc='orange', ec='orange', linewidth=3)

                self.chessboard_ax.text(robot_x + robot_x_dx + 15, robot_y + robot_x_dy + 15, '機械臂X+',
                                        fontsize=10, color='green', weight='bold')
                self.chessboard_ax.text(robot_x + robot_y_dx + 15, robot_y + robot_y_dy + 15, '機械臂Y+',
                                        fontsize=10, color='orange', weight='bold')

        # ===== 顯示真實世界座標系 =====
        if self.show_world_coords_btn.isChecked() and self._generated_world_coords:
            world_coords = np.array(self._generated_world_coords)
            world_x = world_coords[:, 1]
            world_y = world_coords[:, 2]

            self.chessboard_ax.scatter(world_x, world_y, c='lightgreen', s=60, alpha=0.7,
                                       edgecolor='darkgreen', label='真實世界座標', marker='o')

            # 標出機械臂原點在世界座標系中的位置
            if 1 <= robot_point <= len(world_coords):
                robot_world_x = world_coords[robot_point - 1, 1]
                robot_world_y = world_coords[robot_point - 1, 2]

                self.chessboard_ax.scatter([robot_world_x], [robot_world_y], c='darkred', s=200, marker='*',
                                           edgecolor='red', linewidth=2, label='機械臂原點 (世界座標)')

                # 世界座標系中的軸
                self.chessboard_ax.arrow(robot_world_x, robot_world_y, spacing * 2, 0,
                                         head_width=spacing * 0.5, head_length=spacing * 0.6, fc='cyan', ec='cyan', linewidth=3)
                self.chessboard_ax.arrow(robot_world_x, robot_world_y, 0, spacing * 2,
                                         head_width=spacing * 0.5, head_length=spacing * 0.6, fc='magenta', ec='magenta', linewidth=3)

                self.chessboard_ax.text(robot_world_x + spacing * 2 + 5, robot_world_y + 5, '世界X+',
                                        fontsize=10, color='cyan', weight='bold')
                self.chessboard_ax.text(robot_world_x + 5, robot_world_y + spacing * 2 + 5, '世界Y+',
                                        fontsize=10, color='magenta', weight='bold')

        # ===== 顯示偵測到的角點 =====
        if self.show_corners_btn.isChecked() and self._corner_cache:
            # 使用第一張圖的角點作為示例
            first_corners = list(self._corner_cache.values())[0] if self._corner_cache else None
            if first_corners is not None:
                corners = np.array(first_corners).reshape(-1, 2)
                self.chessboard_ax.scatter(corners[:, 0], corners[:, 1],
                                           c='red', s=100, marker='+', alpha=0.8, linewidths=3,
                                           label='偵測到的角點')

                # 標註部分角點編號
                step = max(1, len(corners) // 20)
                for i in range(0, len(corners), step):
                    corner = corners[i]
                    self.chessboard_ax.annotate(f'P{i + 1}', (corner[0], corner[1]),
                                                xytext=(5, 5), textcoords='offset points',
                                                fontsize=8, color='darkred', weight='bold',
                                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # 設置圖表屬性
        self.chessboard_ax.set_title('棋盤格座標系設置', fontsize=14, fontweight='bold')
        self.chessboard_ax.set_xlabel('X座標')
        self.chessboard_ax.set_ylabel('Y座標')
        self.chessboard_ax.legend(loc='upper right', fontsize=9)
        self.chessboard_ax.grid(True, alpha=0.3)
        self.chessboard_ax.axis('equal')

        self.chessboard_canvas.draw()

    def _get_direction_vector(self, direction: str, length: float) -> tuple:
        """根據方向字串返回向量"""
        vectors = {
            "image_x_positive": (length, 0),
            "image_x_negative": (-length, 0),
            "image_y_positive": (0, length),
            "image_y_negative": (0, -length),
        }
        return vectors.get(direction, (0, 0))

    def _on_chessboard_scroll(self, event):
        """棋盤格可視化滾輪縮放"""
        if event.inaxes != self.chessboard_ax:
            return

        xlim = self.chessboard_ax.get_xlim()
        ylim = self.chessboard_ax.get_ylim()

        scale_factor = 1.2 if event.step > 0 else 1 / 1.2

        x_center = event.xdata if event.xdata else (xlim[0] + xlim[1]) / 2
        y_center = event.ydata if event.ydata else (ylim[0] + ylim[1]) / 2

        x_range = (xlim[1] - xlim[0]) / scale_factor
        y_range = (ylim[1] - ylim[0]) / scale_factor

        self.chessboard_ax.set_xlim([x_center - x_range / 2, x_center + x_range / 2])
        self.chessboard_ax.set_ylim([y_center - y_range / 2, y_center + y_range / 2])
        self.chessboard_canvas.draw()

    def _on_chessboard_press(self, event):
        """棋盤格可視化滑鼠按下"""
        if event.button == 3 and event.inaxes == self.chessboard_ax:
            self._chessboard_vis_pressed = True
            self._chessboard_vis_last_x = event.xdata
            self._chessboard_vis_last_y = event.ydata

    def _on_chessboard_release(self, event):
        """棋盤格可視化滑鼠釋放"""
        if event.button == 3:
            self._chessboard_vis_pressed = False

    def _on_chessboard_motion(self, event):
        """棋盤格可視化滑鼠拖動"""
        if self._chessboard_vis_pressed and event.inaxes == self.chessboard_ax and event.xdata and event.ydata:
            dx = event.xdata - self._chessboard_vis_last_x
            dy = event.ydata - self._chessboard_vis_last_y

            xlim = self.chessboard_ax.get_xlim()
            ylim = self.chessboard_ax.get_ylim()

            self.chessboard_ax.set_xlim([xlim[0] - dx, xlim[1] - dx])
            self.chessboard_ax.set_ylim([ylim[0] - dy, ylim[1] - dy])

            self._chessboard_vis_last_x = event.xdata
            self._chessboard_vis_last_y = event.ydata

            self.chessboard_canvas.draw()

    def _create_extrinsic_tab(self) -> QWidget:
        """建立外參計算頁籤"""
        from PySide6.QtWidgets import (
            QTextEdit, QGroupBox, QComboBox, QPushButton,
            QFormLayout, QRadioButton, QButtonGroup, QScrollArea,
        )

        # 外層容器
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # 捲動區域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 說明
        header = QLabel("外參計算")
        header.setProperty("heading", True)
        layout.addWidget(header)

        desc = QLabel("使用 solvePnP 計算相機相對於世界座標系的位姿")
        desc.setProperty("subheading", True)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # ===== 計算方式選擇 =====
        method_group = QGroupBox("計算方式")
        method_layout = QVBoxLayout(method_group)
        method_layout.setContentsMargins(12, 20, 12, 12)

        self.ext_method_group = QButtonGroup(self)

        # 方式一：使用點位數據
        self.ext_use_points_radio = QRadioButton("使用點位數據（推薦）")
        self.ext_use_points_radio.setChecked(True)
        self.ext_use_points_radio.setToolTip("使用「點位數據」頁籤中的像素-世界座標對應")
        self.ext_method_group.addButton(self.ext_use_points_radio, 0)
        method_layout.addWidget(self.ext_use_points_radio)

        self.ext_points_status = QLabel("點位數據：0 個點")
        self.ext_points_status.setProperty("subheading", True)
        self.ext_points_status.setStyleSheet("margin-left: 24px;")
        method_layout.addWidget(self.ext_points_status)

        # 方式二：使用圖像角點
        self.ext_use_corners_radio = QRadioButton("使用圖像角點")
        self.ext_use_corners_radio.setToolTip("選擇一張圖像，使用棋盤格角點計算")
        self.ext_method_group.addButton(self.ext_use_corners_radio, 1)
        method_layout.addWidget(self.ext_use_corners_radio)

        # 圖像選擇下拉選單
        corners_layout = QHBoxLayout()
        corners_layout.setContentsMargins(24, 0, 0, 0)
        self.ext_image_combo = QComboBox()
        self.ext_image_combo.setToolTip("選擇要用於外參計算的圖像")
        self.ext_image_combo.addItem("-- 請先載入圖像 --")
        corners_layout.addWidget(QLabel("定位圖像："))
        corners_layout.addWidget(self.ext_image_combo, 1)
        method_layout.addLayout(corners_layout)

        layout.addWidget(method_group)

        # ===== 算法選擇 =====
        algo_group = QGroupBox("算法選擇")
        algo_layout = QVBoxLayout(algo_group)
        algo_layout.setContentsMargins(12, 20, 12, 12)
        algo_layout.setSpacing(8)

        algo_select_layout = QFormLayout()
        self.ext_algo_combo = QComboBox()
        self.ext_algo_combo.addItem("PnP 迭代法 (≥4點, 推薦)", "SOLVEPNP_ITERATIVE")
        self.ext_algo_combo.addItem("EPnP 算法 (≥4點, 高效)", "SOLVEPNP_EPNP")
        self.ext_algo_combo.addItem("P3P 算法 (=3點)", "SOLVEPNP_P3P")
        self.ext_algo_combo.addItem("AP3P 算法 (≥3點)", "SOLVEPNP_AP3P")
        self.ext_algo_combo.addItem("IPPE 算法 (≥4點, 平面)", "SOLVEPNP_IPPE")
        self.ext_algo_combo.addItem("IPPE_SQUARE (≥4點, 正方形)", "SOLVEPNP_IPPE_SQUARE")
        self.ext_algo_combo.currentIndexChanged.connect(self._on_algo_changed)
        algo_select_layout.addRow("PnP 算法：", self.ext_algo_combo)
        algo_layout.addLayout(algo_select_layout)

        # 算法說明
        self.algo_desc_label = QLabel(
            "迭代法：最通用，適合大多數情況，需≥4個點"
        )
        self.algo_desc_label.setProperty("subheading", True)
        self.algo_desc_label.setWordWrap(True)
        self.algo_desc_label.setStyleSheet("color: #5f6368; font-size: 12px;")
        algo_layout.addWidget(self.algo_desc_label)

        layout.addWidget(algo_group)

        # ===== 執行按鈕 =====
        btn_layout = QHBoxLayout()

        self.ext_calibrate_btn = QPushButton("計算外參")
        self.ext_calibrate_btn.clicked.connect(self._on_calibrate_extrinsic)
        btn_layout.addWidget(self.ext_calibrate_btn)

        self.ext_export_btn = QPushButton("匯出外參")
        self.ext_export_btn.clicked.connect(self._on_export_extrinsic)
        self.ext_export_btn.setEnabled(False)
        btn_layout.addWidget(self.ext_export_btn)

        layout.addLayout(btn_layout)

        # ===== 結果顯示區 =====
        result_group = QGroupBox("外參結果")
        result_layout = QVBoxLayout(result_group)
        result_layout.setContentsMargins(12, 20, 12, 12)

        self.extrinsic_view = QTextEdit()
        self.extrinsic_view.setReadOnly(True)
        self.extrinsic_view.setPlaceholderText(
            "外參標定流程：\n\n"
            "方式一（推薦）：使用點位數據\n"
            "① 在「內參標定」完成內參計算或載入\n"
            "② 在「點位數據」設定像素-世界座標對應\n"
            "③ 在此頁選擇「使用點位數據」\n"
            "④ 點擊「計算外參」\n\n"
            "方式二：使用圖像角點\n"
            "① 在「內參標定」完成內參計算\n"
            "② 選擇一張圖像定義世界座標系\n"
            "③ 在此頁選擇「使用圖像角點」\n"
            "④ 點擊「計算外參」"
        )
        result_layout.addWidget(self.extrinsic_view)

        layout.addWidget(result_group, 1)

        scroll.setWidget(widget)
        container_layout.addWidget(scroll)

        return container

    def _create_transform_tab(self) -> QWidget:
        """建立座標轉換頁籤"""
        from PySide6.QtWidgets import (
            QTextEdit, QGroupBox, QDoubleSpinBox, QPushButton,
            QFormLayout, QGridLayout, QScrollArea,
        )

        # 外層容器
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # 捲動區域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

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

        scroll.setWidget(widget)
        container_layout.addWidget(scroll)

        return container

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

            # 如果已有內參，更新外參圖像下拉選單
            if self._result is not None:
                self._update_extrinsic_image_combo()

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

        # 如果已有內參數據，更新外參圖像下拉選單
        if self._result is not None:
            self._update_extrinsic_image_combo()

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
                self.statusbar.showMessage(f"已載入內參：{file_path}")
                logger.info(f"成功載入標定檔案，內參已就緒")

                # 更新外參頁面狀態提示
                self.ext_image_combo.clear()
                self.ext_image_combo.addItem("-- 請新增圖像並偵測角點 --")
                self.ext_calibrate_btn.setEnabled(False)

                # 顯示提示訊息
                QMessageBox.information(
                    self,
                    "載入成功",
                    "內參已載入！\n\n"
                    "若要計算外參，請：\n"
                    "① 新增定位圖像\n"
                    "② 偵測角點\n"
                    "③ 切換到「外參計算」頁籤選擇圖像",
                )
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"無法載入檔案：{e}")
                logger.error(f"載入標定檔案失敗：{e}")

    @Slot()
    def _on_about(self):
        """顯示關於對話框"""
        QMessageBox.about(
            self,
            "關於 TSIC/CR-ICS01",
            f"""<h2>TSIC/CR-ICS01</h2>
            <h3>相機與手臂整合控制教學軟體</h3>
            <p>版本 {__version__}</p>
            <hr>
            <p>專業相機標定與座標轉換工具</p>
            <p>使用棋盤格圖案進行相機內外參標定，<br>
            支援像素、相機、世界座標系之間的轉換。</p>
            <hr>
            <p><b>開發單位：</b>TSIC</p>
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

        # 列出所有已載入的圖像（優先顯示已偵測角點的）
        all_images = []
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            image_path = item.data(Qt.UserRole)
            has_corners = image_path in self._corner_cache and self._corner_cache[image_path] is not None
            all_images.append((Path(image_path).name, image_path, has_corners))

        if not all_images:
            self.ext_image_combo.addItem("-- 請先新增圖像 --")
            self.ext_calibrate_btn.setEnabled(False)
            return

        # 先加已偵測角點的，再加未偵測的
        for name, path, has_corners in all_images:
            if has_corners:
                self.ext_image_combo.addItem(f"✓ {name}", path)
            else:
                self.ext_image_combo.addItem(f"○ {name} (需偵測)", path)

        self.ext_calibrate_btn.setEnabled(True)

        detected_count = sum(1 for _, _, has in all_images if has)
        self.statusbar.showMessage(f"共 {len(all_images)} 張圖像，{detected_count} 張已偵測角點")

    def _on_algo_changed(self, index: int):
        """當 PnP 算法選擇變更時更新說明"""
        algo_descriptions = {
            "SOLVEPNP_ITERATIVE": "迭代法：最通用，適合大多數情況，需≥4個點",
            "SOLVEPNP_EPNP": "EPnP：高效率算法，點數較多(>10)時推薦，需≥4個點",
            "SOLVEPNP_P3P": "P3P：只需剛好3個點，可能有多解，適合點數極少的情況",
            "SOLVEPNP_AP3P": "AP3P：P3P改進版，數值穩定性更好，需≥3個點",
            "SOLVEPNP_IPPE": "IPPE：專為平面物體設計，適合棋盤格等平面標定，需≥4個點",
            "SOLVEPNP_IPPE_SQUARE": "IPPE_SQUARE：IPPE改進版，專為正方形標定板優化，需≥4個點",
        }
        algo_data = self.ext_algo_combo.currentData()
        desc = algo_descriptions.get(algo_data, "")
        if hasattr(self, 'algo_desc_label'):
            self.algo_desc_label.setText(desc)

    @Slot()
    def _on_calibrate_extrinsic(self):
        """執行外參標定"""
        if self._result is None:
            QMessageBox.warning(
                self,
                "尚無內參數據",
                "請先載入內參標定檔案（.h5/.mat/.json）\n"
                "或執行內參標定。"
            )
            return

        import cv2
        import numpy as np
        from vision_calib.core.types import CameraExtrinsic
        from vision_calib.core.extrinsic import ExtrinsicCalibrationResult
        from vision_calib.core.transform import CoordinateTransformer

        # 取得 PnP 算法
        algo_data = self.ext_algo_combo.currentData()
        algo_flags = {
            "SOLVEPNP_ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
            "SOLVEPNP_EPNP": cv2.SOLVEPNP_EPNP,
            "SOLVEPNP_P3P": cv2.SOLVEPNP_P3P,
            "SOLVEPNP_AP3P": cv2.SOLVEPNP_AP3P,
            "SOLVEPNP_IPPE": cv2.SOLVEPNP_IPPE,
            "SOLVEPNP_IPPE_SQUARE": cv2.SOLVEPNP_IPPE_SQUARE,
        }
        pnp_flag = algo_flags.get(algo_data, cv2.SOLVEPNP_ITERATIVE)

        # 確定最少需要的點數
        min_points = 3 if algo_data in ("SOLVEPNP_P3P", "SOLVEPNP_AP3P") else 4

        try:
            # 根據選擇的方式進行計算
            if self.ext_use_points_radio.isChecked():
                # 方式一：使用點位數據
                if len(self._point_data) < min_points:
                    QMessageBox.warning(
                        self,
                        "點位不足",
                        f"選用的算法至少需要 {min_points} 個點位！\n"
                        f"目前只有 {len(self._point_data)} 個點位。\n\n"
                        "請前往「點位數據」頁籤添加點位。"
                    )
                    return

                self.statusbar.showMessage("正在使用點位數據計算外參...")

                # 從 point_data 提取座標
                object_points = np.array(
                    [[p[3], p[4], 0.0] for p in self._point_data],
                    dtype=np.float32
                )
                image_points = np.array(
                    [[p[1], p[2]] for p in self._point_data],
                    dtype=np.float32
                )

                # 執行 solvePnP
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    self._result.intrinsic.camera_matrix,
                    self._result.intrinsic.distortion_coeffs,
                    flags=pnp_flag,
                )

                if not success:
                    QMessageBox.critical(self, "失敗", "solvePnP 計算失敗！請檢查點位數據。")
                    return

                # 計算重投影誤差
                projected, _ = cv2.projectPoints(
                    object_points,
                    rvec,
                    tvec,
                    self._result.intrinsic.camera_matrix,
                    self._result.intrinsic.distortion_coeffs,
                )
                projected = projected.reshape(-1, 2)
                error = np.sqrt(np.mean(np.sum((image_points - projected) ** 2, axis=1)))

                # 建立結果
                extrinsic = CameraExtrinsic(
                    rotation_vector=rvec,
                    translation_vector=tvec,
                )
                self._extrinsic_result = ExtrinsicCalibrationResult(
                    extrinsic=extrinsic,
                    reprojection_error=error,
                    image_path="point_data",
                    num_points=len(self._point_data),
                )

            else:
                # 方式二：使用圖像角點
                image_path = self.ext_image_combo.currentData()
                if not image_path:
                    QMessageBox.warning(
                        self,
                        "無可用圖像",
                        "請先新增定位圖像，然後從下拉選單選擇。"
                    )
                    return

                # 取得角點
                corners = self._corner_cache.get(image_path)
                if corners is None:
                    # 自動偵測角點
                    self.statusbar.showMessage("正在偵測角點...")
                    from vision_calib.core.corner_detector import CornerDetector
                    from vision_calib.core.types import CheckerboardConfig

                    config = CheckerboardConfig(
                        rows=self.rows_spin.value(),
                        cols=self.cols_spin.value(),
                        square_size_mm=self.square_size_spin.value() * 10,
                    )
                    detector = CornerDetector(config)
                    result = detector.detect(image_path)

                    if not result.success:
                        QMessageBox.warning(
                            self,
                            "角點偵測失敗",
                            f"無法偵測到 {self.cols_spin.value()}×{self.rows_spin.value()} 棋盤格角點。"
                        )
                        return

                    corners = result.corners
                    self._corner_cache[image_path] = corners
                    self._update_extrinsic_image_combo()

                self.statusbar.showMessage("正在使用圖像角點計算外參...")

                from vision_calib.core.extrinsic import ExtrinsicCalibrator
                from vision_calib.core.types import CheckerboardConfig

                checkerboard = CheckerboardConfig(
                    rows=self.rows_spin.value(),
                    cols=self.cols_spin.value(),
                    square_size_mm=self.square_size_spin.value() * 10,
                )

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

            # 啟用按鈕
            self._enable_transform_buttons()
            self.ext_export_btn.setEnabled(True)

            self.statusbar.showMessage("外參計算完成")
            QMessageBox.information(
                self,
                "外參計算完成",
                f"重投影誤差：{self._extrinsic_result.reprojection_error:.4f} 像素\n\n"
                f"使用點位數：{self._extrinsic_result.num_points}\n"
                f"現在可以使用座標轉換功能。",
            )

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"外參計算失敗：{e}")
            logger.error(f"外參計算失敗：{e}")

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

    # ===== 點位數據管理 =====

    @Slot()
    def _on_add_point(self):
        """添加單個點位"""
        point_id = self.point_id_spin.value()
        img_x = self.point_img_x_spin.value()
        img_y = self.point_img_y_spin.value()
        world_x = self.point_world_x_spin.value()
        world_y = self.point_world_y_spin.value()

        self._point_data.append([point_id, img_x, img_y, world_x, world_y])
        self._update_points_table()

        # 自動遞增 ID
        self.point_id_spin.setValue(point_id + 1)
        self.statusbar.showMessage(f"已添加點位 {point_id}")

    @Slot()
    def _on_clear_points(self):
        """清除所有點位數據"""
        if self._point_data:
            reply = QMessageBox.question(
                self, "確認清除",
                f"確定要清除所有 {len(self._point_data)} 個點位嗎？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._point_data.clear()
                self._update_points_table()
                self.statusbar.showMessage("已清除所有點位")

    def _update_points_table(self):
        """更新點位數據表格"""
        from PySide6.QtWidgets import QTableWidgetItem

        self.points_table.setRowCount(len(self._point_data))
        for i, point in enumerate(self._point_data):
            for j, val in enumerate(point):
                item = QTableWidgetItem(f"{val:.2f}" if isinstance(val, float) else str(int(val)))
                self.points_table.setItem(i, j, item)

        # 更新標題
        self.points_list_group.setTitle(f"當前點位 (共 {len(self._point_data)} 個)")

        # 更新外參頁面狀態
        self.ext_points_status.setText(f"點位數據：{len(self._point_data)} 個點")

    @Slot()
    def _on_import_points_csv(self):
        """匯入點位 CSV"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "匯入點位數據",
            "", "CSV 檔案 (*.csv);;所有檔案 (*)",
        )
        if not file_path:
            return

        try:
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    # 支持多種欄位名稱
                    point_id = float(row.get('id', row.get('ID', count + 1)))
                    img_x = float(row.get('image_x', row.get('img_x', row.get('x', 0))))
                    img_y = float(row.get('image_y', row.get('img_y', row.get('y', 0))))
                    world_x = float(row.get('world_x', row.get('X', 0)))
                    world_y = float(row.get('world_y', row.get('Y', 0)))
                    self._point_data.append([point_id, img_x, img_y, world_x, world_y])
                    count += 1

            self._update_points_table()
            self.statusbar.showMessage(f"已匯入 {count} 個點位")
            QMessageBox.information(self, "匯入成功", f"已匯入 {count} 個點位")
        except Exception as e:
            QMessageBox.critical(self, "匯入失敗", f"無法讀取 CSV：{e}")

    @Slot()
    def _on_export_points_csv(self):
        """匯出點位 CSV"""
        if not self._point_data:
            QMessageBox.warning(self, "無數據", "目前沒有點位數據可匯出")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "匯出點位數據",
            "point_data.csv", "CSV 檔案 (*.csv);;所有檔案 (*)",
        )
        if not file_path:
            return

        try:
            import csv
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'image_x', 'image_y', 'world_x', 'world_y'])
                for point in self._point_data:
                    writer.writerow(point)

            self.statusbar.showMessage(f"已匯出至：{file_path}")
            QMessageBox.information(self, "匯出成功", f"已匯出 {len(self._point_data)} 個點位")
        except Exception as e:
            QMessageBox.critical(self, "匯出失敗", f"無法寫入 CSV：{e}")

    @Slot()
    def _on_load_corners_to_points(self):
        """從角點快取載入像素座標到點位數據"""
        # 找到第一個有角點的圖像
        corners_data = None
        for path, corners in self._corner_cache.items():
            if corners is not None:
                corners_data = corners
                break

        if corners_data is None:
            QMessageBox.warning(
                self, "無角點數據",
                "請先偵測圖像角點。\n\n"
                "① 載入圖像\n"
                "② 設定棋盤格參數\n"
                "③ 點擊「偵測角點」"
            )
            return

        # 如果有世界座標，合併；否則只載入像素座標
        if self._generated_world_coords:
            if len(corners_data) != len(self._generated_world_coords):
                QMessageBox.warning(
                    self, "數量不匹配",
                    f"角點數量 ({len(corners_data)}) 與世界座標數量 "
                    f"({len(self._generated_world_coords)}) 不匹配！\n\n"
                    "請確保棋盤格參數一致。"
                )
                return

            self._point_data.clear()
            for i, (corner, world) in enumerate(zip(corners_data, self._generated_world_coords)):
                point_id = world[0]
                img_x, img_y = corner[0], corner[1]
                world_x, world_y = world[1], world[2]
                self._point_data.append([point_id, img_x, img_y, world_x, world_y])
        else:
            # 只載入像素座標，世界座標設為 0
            self._point_data.clear()
            for i, corner in enumerate(corners_data):
                self._point_data.append([i + 1, corner[0], corner[1], 0.0, 0.0])

        self._update_points_table()
        self.statusbar.showMessage(f"已載入 {len(self._point_data)} 個點位")
        QMessageBox.information(
            self, "載入成功",
            f"已載入 {len(self._point_data)} 個點位到數據表。\n\n"
            + ("已合併世界座標。" if self._generated_world_coords else "世界座標尚未設定，請使用「棋盤生成」功能。")
        )

    # ===== 棋盤格世界座標生成 =====

    def _get_direction_coeff(self, direction: str) -> tuple:
        """獲取方向在網格座標系中的係數"""
        if direction == "image_x_positive":
            return (1, 0)
        elif direction == "image_x_negative":
            return (-1, 0)
        elif direction == "image_y_positive":
            return (0, 1)
        elif direction == "image_y_negative":
            return (0, -1)
        return (1, 0)

    @Slot()
    def _on_generate_world_coords(self):
        """生成棋盤格世界座標（使用左側全局棋盤格參數）"""
        try:
            origin_x = self.origin_x_spin.value()
            origin_y = self.origin_y_spin.value()
            grid_x = self.cols_spin.value()  # 使用全局參數
            grid_y = self.rows_spin.value()  # 使用全局參數
            spacing = self.square_size_spin.value() * 10  # cm → mm
            robot_point = self.robot_origin_point_spin.value()

            robot_x_dir = self.robot_x_dir_combo.currentData()
            robot_y_dir = self.robot_y_dir_combo.currentData()

            total_points = grid_x * grid_y
            if robot_point < 1 or robot_point > total_points:
                QMessageBox.warning(
                    self, "參數錯誤",
                    f"機械臂原點編號必須在 1 到 {total_points} 之間"
                )
                return

            # 計算機械臂原點在棋盤格中的位置
            robot_idx = robot_point - 1
            robot_grid_i = robot_idx % grid_x
            robot_grid_j = robot_idx // grid_x

            # 獲取座標軸方向係數
            robot_x_coeff = self._get_direction_coeff(robot_x_dir)
            robot_y_coeff = self._get_direction_coeff(robot_y_dir)

            # 計算行列式
            det = robot_x_coeff[0] * robot_y_coeff[1] - robot_y_coeff[0] * robot_x_coeff[1]
            if abs(det) < 1e-10:
                QMessageBox.warning(self, "參數錯誤", "座標軸方向不能平行！")
                return

            # 逆矩陣
            inv_11 = robot_y_coeff[1] / det
            inv_12 = -robot_y_coeff[0] / det
            inv_21 = -robot_x_coeff[1] / det
            inv_22 = robot_x_coeff[0] / det

            # 生成座標
            self._generated_world_coords.clear()
            point_id = 1
            for j in range(grid_y):
                for i in range(grid_x):
                    grid_offset_i = i - robot_grid_i
                    grid_offset_j = j - robot_grid_j

                    robot_x_offset = inv_11 * grid_offset_i + inv_12 * grid_offset_j
                    robot_y_offset = inv_21 * grid_offset_i + inv_22 * grid_offset_j

                    world_x = origin_x + robot_x_offset * spacing
                    world_y = origin_y + robot_y_offset * spacing

                    self._generated_world_coords.append([point_id, world_x, world_y])
                    point_id += 1

            # 顯示結果
            import numpy as np
            coords_array = np.array(self._generated_world_coords)

            result_text = f"""座標生成完成！

總計 {len(self._generated_world_coords)} 個點

棋盤格規格: {grid_x} × {grid_y}
格子間距: {spacing:.2f} mm
機械臂原點: P{robot_point} = ({origin_x:.1f}, {origin_y:.1f}) mm

座標範圍:
  X: {coords_array[:, 1].min():.1f} ~ {coords_array[:, 1].max():.1f} mm
  Y: {coords_array[:, 2].min():.1f} ~ {coords_array[:, 2].max():.1f} mm

關鍵點位:
  P1: ({self._generated_world_coords[0][1]:.1f}, {self._generated_world_coords[0][2]:.1f}) mm
  P{robot_point}: ({origin_x:.1f}, {origin_y:.1f}) mm
  P{total_points}: ({self._generated_world_coords[-1][1]:.1f}, {self._generated_world_coords[-1][2]:.1f}) mm
"""
            self.chessboard_gen_result.setText(result_text)
            self.statusbar.showMessage(f"已生成 {len(self._generated_world_coords)} 個世界座標")

            # 更新可視化
            self._update_chessboard_vis()

        except Exception as e:
            QMessageBox.critical(self, "生成失敗", f"生成座標失敗：{e}")
            logger.error(f"生成世界座標失敗：{e}")

    @Slot()
    def _on_load_world_coords_to_points(self):
        """將生成的世界座標載入到點位數據"""
        if not self._generated_world_coords:
            QMessageBox.warning(self, "無數據", "請先生成世界座標")
            return

        # 找到角點數據
        corners_data = None
        for path, corners in self._corner_cache.items():
            if corners is not None:
                corners_data = corners
                break

        if corners_data is None:
            QMessageBox.warning(
                self, "無角點數據",
                "請先偵測圖像角點。\n\n"
                "① 回到「內參標定」頁籤\n"
                "② 載入圖像\n"
                "③ 點擊「偵測角點」"
            )
            return

        if len(corners_data) != len(self._generated_world_coords):
            QMessageBox.warning(
                self, "數量不匹配",
                f"角點數量 ({len(corners_data)}) 與世界座標數量 "
                f"({len(self._generated_world_coords)}) 不匹配！\n\n"
                "請確保棋盤格參數一致。"
            )
            return

        # 合併數據
        self._point_data.clear()
        for corner, world in zip(corners_data, self._generated_world_coords):
            point_id = world[0]
            img_x, img_y = corner[0], corner[1]
            world_x, world_y = world[1], world[2]
            self._point_data.append([point_id, img_x, img_y, world_x, world_y])

        self._update_points_table()
        self.statusbar.showMessage(f"已載入 {len(self._point_data)} 個點位到數據表")

        # 切換到點位數據頁籤
        self.tab_widget.setCurrentIndex(1)  # 點位數據頁籤

        QMessageBox.information(
            self, "載入成功",
            f"已將 {len(self._point_data)} 個點位載入到數據表！\n\n"
            "現在可以切換到「外參計算」頁籤進行外參計算。"
        )

    # ===== 匯出外參 =====

    @Slot()
    def _on_export_extrinsic(self):
        """匯出外參數據"""
        if self._extrinsic_result is None:
            QMessageBox.warning(self, "無數據", "請先完成外參計算")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "匯出外參",
            "extrinsic", "NPY 檔案 (*.npy);;JSON 檔案 (*.json);;所有檔案 (*)",
        )
        if not file_path:
            return

        try:
            import numpy as np
            import json

            ext = self._extrinsic_result.extrinsic
            rvec = ext.rotation_vector.flatten()
            tvec = ext.translation_vector.flatten()

            if file_path.endswith('.npy'):
                # 儲存為結構化陣列
                data = {
                    'rvec': ext.rotation_vector,
                    'tvec': ext.translation_vector,
                    'rotation_matrix': ext.rotation_matrix,
                }
                np.save(file_path, data)
            else:
                # 儲存為 JSON
                data = {
                    'rvec': rvec.tolist(),
                    'tvec': tvec.tolist(),
                    'rotation_matrix': ext.rotation_matrix.tolist(),
                    'reprojection_error': self._extrinsic_result.reprojection_error,
                }
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)

            self.statusbar.showMessage(f"已匯出至：{file_path}")
            QMessageBox.information(self, "匯出成功", f"外參已儲存至：\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "匯出失敗", f"無法儲存檔案：{e}")


def main():
    """應用程式入口點"""
    setup_logging()

    app = QApplication(sys.argv)
    app.setApplicationName("TSIC/CR-ICS01")
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
