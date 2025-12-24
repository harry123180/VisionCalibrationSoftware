import customtkinter as ctk
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import pandas as pd
from datetime import datetime

class CameraCalibrationTool:
    def __init__(self):
        # 設置 CustomTkinter 主題
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # 初始化主窗口
        self.root = ctk.CTk()
        self.root.title("Camera Calibration Adjuster")
        self.root.geometry("1600x1000")
        
        # 設置中文字體
        try:
            # 嘗試設置系統中文字體
            import platform
            system = platform.system()
            if system == "Darwin":  # macOS
                self.font_family = "PingFang SC"
            elif system == "Windows":
                self.font_family = "Microsoft YaHei"
            else:  # Linux
                self.font_family = "DejaVu Sans"
                
            # 設置matplotlib中文字體
            plt.rcParams['font.sans-serif'] = [self.font_family, 'SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
        except:
            self.font_family = "Arial"
        
        # 初始化數據
        self.init_default_data()
        
        # 可視化控制變量
        self.show_image_coords = tk.BooleanVar(value=False)
        self.show_world_coords = tk.BooleanVar(value=True)
        self.show_transformed_coords = tk.BooleanVar(value=True)
        self.show_error_lines = tk.BooleanVar(value=True)
        
        # 新增功能的變量
        self.loaded_image = None
        self.detected_corners = None
        self.canvas_zoom = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        
        # 鼠標狀態
        self.mouse_pressed = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # 創建界面
        self.create_widgets()
        
        # 初始計算和顯示
        self.calculate_transformation()
        
    def init_default_data(self):
        """初始化默認數據"""
        # 默認內參矩陣
        self.K = np.array([
            [5527.91522, 0.0, 1249.56097],
            [0.0, 5523.37409, 997.41524],
            [0.0, 0.0, 1.0]
        ])
        
        # 默認畸變參數
        self.D = np.array([-0.06833483, 0.00056340, 0.00137019, 0.00055740, 4.80949681])
        
        # 默認外參
        self.rvec = np.array([[-2.17796294], [-2.24565035], [0.02621215]])
        self.tvec = np.array([[330.20053861], [48.63793437], [533.5402696]])
        
        # 點位數據
        self.image_coords = np.array([])
        self.world_coords = np.array([])
        self.point_data = []  # 存儲點位數據 [id, image_x, image_y, world_x, world_y]
        
        # 算法選項
        self.estimation_algorithm = tk.StringVar(value="PnP_ITERATIVE")
        
    def create_widgets(self):
        """創建主界面"""
        # 主框架
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 左側控制面板
        left_panel = ctk.CTkFrame(main_frame, width=450)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # 右側可視化面板
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True)
        
        self.create_control_panel(left_panel)
        self.create_visualization_panel(right_panel)
        
    def create_control_panel(self, parent):
        """創建控制面板"""
        # 標題
        title = ctk.CTkLabel(parent, text="相機參數調整", 
                           font=ctk.CTkFont(family=self.font_family, size=20, weight="bold"))
        title.pack(pady=(10, 20))
        
        # 創建分頁按鈕框架
        tab_frame = ctk.CTkFrame(parent)
        tab_frame.pack(fill="x", padx=10, pady=5)
        
        # 分頁按鈕
        self.tab_buttons = {}
        self.current_tab = "intrinsic"
        
        tab_names = {
            "intrinsic": "內參",
            "extrinsic": "外參", 
            "points": "點位",
            "algorithm": "算法",
            "corner_detect": "角點檢測",  # 新增
            "chessboard_gen": "棋盤生成",  # 新增
            "file": "文件",
            "view": "顯示",
            "help": "說明"
        }
        
        # 創建兩行按鈕布局
        for i, (tab_id, tab_text) in enumerate(tab_names.items()):
            if i < 5:  # 第一行
                row, col = 0, i
            else:  # 第二行
                row, col = 1, i - 5
                
            btn = ctk.CTkButton(
                tab_frame, 
                text=tab_text, 
                width=80,
                height=30,
                font=ctk.CTkFont(family=self.font_family, size=10),
                command=lambda tid=tab_id: self.switch_tab(tid)
            )
            btn.grid(row=row, column=col, padx=1, pady=1, sticky="ew")
            self.tab_buttons[tab_id] = btn
            tab_frame.grid_columnconfigure(col, weight=1)
        
        # 內容框架
        self.content_frame = ctk.CTkScrollableFrame(parent)
        self.content_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 創建所有分頁內容
        self.create_all_tab_contents()
        
        # 顯示默認分頁
        self.switch_tab("intrinsic")
        
    def create_all_tab_contents(self):
        """創建所有分頁內容"""
        # 內參調整內容
        self.intrinsic_content = ctk.CTkFrame(self.content_frame)
        self.create_intrinsic_controls(self.intrinsic_content)
        
        # 外參調整內容
        self.extrinsic_content = ctk.CTkFrame(self.content_frame)
        self.create_extrinsic_controls(self.extrinsic_content)
        
        # 點位數據內容
        self.points_content = ctk.CTkFrame(self.content_frame)
        self.create_points_controls(self.points_content)
        
        # 算法選擇內容
        self.algorithm_content = ctk.CTkFrame(self.content_frame)
        self.create_algorithm_controls(self.algorithm_content)
        
        # 角點檢測內容
        self.corner_detect_content = ctk.CTkFrame(self.content_frame)
        self.create_corner_detect_controls(self.corner_detect_content)
        
        # 棋盤生成內容
        self.chessboard_gen_content = ctk.CTkFrame(self.content_frame)
        self.create_chessboard_gen_controls(self.chessboard_gen_content)
        
        # 文件操作內容
        self.file_content = ctk.CTkFrame(self.content_frame)
        self.create_file_controls(self.file_content)
        
        # 顯示控制內容
        self.view_content = ctk.CTkFrame(self.content_frame)
        self.create_view_controls(self.view_content)
        
        # 說明頁面內容
        self.help_content = ctk.CTkFrame(self.content_frame)
        self.create_help_controls(self.help_content)
        
        # 存儲所有分頁內容
        self.tab_contents = {
            "intrinsic": self.intrinsic_content,
            "extrinsic": self.extrinsic_content,
            "points": self.points_content,
            "algorithm": self.algorithm_content,
            "corner_detect": self.corner_detect_content,
            "chessboard_gen": self.chessboard_gen_content,
            "file": self.file_content,
            "view": self.view_content,
            "help": self.help_content
        }
    
    def create_corner_detect_controls(self, parent):
        """創建角點檢測控制面板"""
        ctk.CTkLabel(parent, text="角點檢測:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(10, 5))
        
        # 圖片導入按鈕
        img_frame = ctk.CTkFrame(parent)
        img_frame.pack(fill="x", pady=5)
        
        ctk.CTkButton(img_frame, text="導入圖片", 
                     font=ctk.CTkFont(family=self.font_family),
                     command=self.import_image).pack(side="left", padx=5)
        ctk.CTkButton(img_frame, text="重置視圖",
                     font=ctk.CTkFont(family=self.font_family),
                     command=self.reset_image_view).pack(side="left", padx=5)
        
        # 棋盤格參數設置
        ctk.CTkLabel(parent, text="棋盤格參數:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(20, 5))
        
        # 棋盤格尺寸
        size_frame = ctk.CTkFrame(parent)
        size_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(size_frame, text="內角點數 (寬x高):", width=120,
                    font=ctk.CTkFont(family=self.font_family)).pack(side="left", padx=2)
        self.chessboard_width_entry = ctk.CTkEntry(size_frame, width=50, placeholder_text="7",
                                                  font=ctk.CTkFont(family=self.font_family))
        self.chessboard_width_entry.pack(side="left", padx=2)
        self.chessboard_width_entry.insert(0, "7")
        
        ctk.CTkLabel(size_frame, text="x", width=20,
                    font=ctk.CTkFont(family=self.font_family)).pack(side="left", padx=2)
        
        self.chessboard_height_entry = ctk.CTkEntry(size_frame, width=50, placeholder_text="5",
                                                   font=ctk.CTkFont(family=self.font_family))
        self.chessboard_height_entry.pack(side="left", padx=2)
        self.chessboard_height_entry.insert(0, "5")
        
        # 檢測按鈕
        detect_frame = ctk.CTkFrame(parent)
        detect_frame.pack(fill="x", pady=10)
        
        ctk.CTkButton(detect_frame, text="檢測角點", 
                     font=ctk.CTkFont(family=self.font_family),
                     command=self.detect_corners).pack(side="left", padx=5)
        ctk.CTkButton(detect_frame, text="導出角點數據",
                     font=ctk.CTkFont(family=self.font_family),
                     command=self.export_corner_points).pack(side="left", padx=5)
        
        # 檢測結果顯示
        ctk.CTkLabel(parent, text="檢測結果:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(20, 5))
        
        self.corner_result_text = ctk.CTkTextbox(parent, height=150,
                                               font=ctk.CTkFont(family=self.font_family, size=11))
        self.corner_result_text.pack(fill="x", pady=5)
        
        # 視圖控制說明
        control_info = ctk.CTkLabel(parent, 
                                  text="圖片控制：\n滾輪縮放、右鍵拖動平移\n左鍵點擊查看點位信息",
                                  font=ctk.CTkFont(family=self.font_family, size=11),
                                  justify="left")
        control_info.pack(anchor="w", pady=(20, 5))
    
    def create_chessboard_gen_controls(self, parent):
        """創建棋盤生成控制面板"""
        ctk.CTkLabel(parent, text="棋盤格世界座標生成:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(10, 5))
        
        # 原點設置
        origin_frame = ctk.CTkFrame(parent)
        origin_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(origin_frame, text="機械臂原點座標 (mm):", width=140,
                    font=ctk.CTkFont(family=self.font_family, size=11)).pack(side="left", padx=2)
        self.origin_x_entry = ctk.CTkEntry(origin_frame, width=60, placeholder_text="0.0",
                                          font=ctk.CTkFont(family=self.font_family, size=11))
        self.origin_x_entry.pack(side="left", padx=1)
        self.origin_x_entry.insert(0, "0.0")
        
        ctk.CTkLabel(origin_frame, text=",", width=10,
                    font=ctk.CTkFont(family=self.font_family)).pack(side="left", padx=1)
        
        self.origin_y_entry = ctk.CTkEntry(origin_frame, width=60, placeholder_text="0.0",
                                          font=ctk.CTkFont(family=self.font_family, size=11))
        self.origin_y_entry.pack(side="left", padx=1)
        self.origin_y_entry.insert(0, "0.0")
        
        # 棋盤格規格
        grid_frame = ctk.CTkFrame(parent)
        grid_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(grid_frame, text="X方向格數:", width=70,
                    font=ctk.CTkFont(family=self.font_family, size=11)).pack(side="left", padx=2)
        self.grid_x_entry = ctk.CTkEntry(grid_frame, width=50,
                                        font=ctk.CTkFont(family=self.font_family, size=11))
        self.grid_x_entry.pack(side="left", padx=1)
        self.grid_x_entry.insert(0, "17")
        
        ctk.CTkLabel(grid_frame, text="Y方向格數:", width=70,
                    font=ctk.CTkFont(family=self.font_family, size=11)).pack(side="left", padx=2)
        self.grid_y_entry = ctk.CTkEntry(grid_frame, width=50,
                                        font=ctk.CTkFont(family=self.font_family, size=11))
        self.grid_y_entry.pack(side="left", padx=1)
        self.grid_y_entry.insert(0, "12")
        
        # 格子間距和機械臂原點
        spacing_frame = ctk.CTkFrame(parent)
        spacing_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(spacing_frame, text="格子間距:", width=60,
                    font=ctk.CTkFont(family=self.font_family, size=11)).pack(side="left", padx=2)
        self.spacing_entry = ctk.CTkEntry(spacing_frame, width=50,
                                         font=ctk.CTkFont(family=self.font_family, size=11))
        self.spacing_entry.pack(side="left", padx=1)
        self.spacing_entry.insert(0, "30.0")
        
        ctk.CTkLabel(spacing_frame, text="機械臂原點對應點位:", width=120,
                    font=ctk.CTkFont(family=self.font_family, size=11)).pack(side="left", padx=5)
        self.robot_point_entry = ctk.CTkEntry(spacing_frame, width=50,
                                             font=ctk.CTkFont(family=self.font_family, size=11))
        self.robot_point_entry.pack(side="left", padx=1)
        self.robot_point_entry.insert(0, "204")
        
        # 座標系方向設置 - 分成四行顯示，每行兩個選項
        ctk.CTkLabel(parent, text="機械臂座標系方向:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(8, 2))
        
        # 機械臂X+方向 - 第一行
        robot_x_frame1 = ctk.CTkFrame(parent)
        robot_x_frame1.pack(fill="x", pady=1)
        ctk.CTkLabel(robot_x_frame1, text="機械臂X+:", width=70,
                    font=ctk.CTkFont(family=self.font_family, size=11)).pack(side="left", padx=2)
        
        self.robot_x_direction = tk.StringVar(value="image_y_positive")
        
        ctk.CTkRadioButton(robot_x_frame1, text="圖像Y+", variable=self.robot_x_direction, 
                          value="image_y_positive", font=ctk.CTkFont(family=self.font_family, size=10)).pack(side="left", padx=5)
        ctk.CTkRadioButton(robot_x_frame1, text="圖像Y-", variable=self.robot_x_direction, 
                          value="image_y_negative", font=ctk.CTkFont(family=self.font_family, size=10)).pack(side="left", padx=5)
        
        # 機械臂X+方向 - 第二行
        robot_x_frame2 = ctk.CTkFrame(parent)
        robot_x_frame2.pack(fill="x", pady=1)
        ctk.CTkLabel(robot_x_frame2, text="", width=70).pack(side="left", padx=2)  # 空白標籤對齊
        
        ctk.CTkRadioButton(robot_x_frame2, text="圖像X+", variable=self.robot_x_direction, 
                          value="image_x_positive", font=ctk.CTkFont(family=self.font_family, size=10)).pack(side="left", padx=5)
        ctk.CTkRadioButton(robot_x_frame2, text="圖像X-", variable=self.robot_x_direction, 
                          value="image_x_negative", font=ctk.CTkFont(family=self.font_family, size=10)).pack(side="left", padx=5)
        
        # 機械臂Y+方向 - 第三行
        robot_y_frame1 = ctk.CTkFrame(parent)
        robot_y_frame1.pack(fill="x", pady=1)
        ctk.CTkLabel(robot_y_frame1, text="機械臂Y+:", width=70,
                    font=ctk.CTkFont(family=self.font_family, size=11)).pack(side="left", padx=2)
        
        self.robot_y_direction = tk.StringVar(value="image_x_positive")
        
        ctk.CTkRadioButton(robot_y_frame1, text="圖像X+", variable=self.robot_y_direction, 
                          value="image_x_positive", font=ctk.CTkFont(family=self.font_family, size=10)).pack(side="left", padx=5)
        ctk.CTkRadioButton(robot_y_frame1, text="圖像X-", variable=self.robot_y_direction, 
                          value="image_x_negative", font=ctk.CTkFont(family=self.font_family, size=10)).pack(side="left", padx=5)
        
        # 機械臂Y+方向 - 第四行
        robot_y_frame2 = ctk.CTkFrame(parent)
        robot_y_frame2.pack(fill="x", pady=1)
        ctk.CTkLabel(robot_y_frame2, text="", width=70).pack(side="left", padx=2)  # 空白標籤對齊
        
        ctk.CTkRadioButton(robot_y_frame2, text="圖像Y+", variable=self.robot_y_direction, 
                          value="image_y_positive", font=ctk.CTkFont(family=self.font_family, size=10)).pack(side="left", padx=5)
        ctk.CTkRadioButton(robot_y_frame2, text="圖像Y-", variable=self.robot_y_direction, 
                          value="image_y_negative", font=ctk.CTkFont(family=self.font_family, size=10)).pack(side="left", padx=5)
        
        # 角點檢測結果導入區域
        ctk.CTkLabel(parent, text="角點檢測結果:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(8, 2))
        
        corner_import_frame = ctk.CTkFrame(parent)
        corner_import_frame.pack(fill="x", pady=2)
        
        ctk.CTkButton(corner_import_frame, text="導入檢測到的角點", width=120,
                     font=ctk.CTkFont(family=self.font_family, size=11),
                     command=self.import_detected_corners).pack(side="left", padx=2)
        
        # 座標系顯示控制
        ctk.CTkLabel(parent, text="座標系顯示控制:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(8, 2))
        
        display_control_frame = ctk.CTkFrame(parent)
        display_control_frame.pack(fill="x", pady=2)
        
        # 顯示控制選項
        self.show_detected_corners = tk.BooleanVar(value=False)
        self.show_world_coordinates = tk.BooleanVar(value=True)
        self.show_image_coordinates = tk.BooleanVar(value=True)
        
        ctk.CTkCheckBox(display_control_frame, text="檢測角點", variable=self.show_detected_corners,
                       font=ctk.CTkFont(family=self.font_family, size=10),
                       command=self.update_chessboard_visualization).pack(side="left", padx=3)
        
        ctk.CTkCheckBox(display_control_frame, text="真實世界座標", variable=self.show_world_coordinates,
                       font=ctk.CTkFont(family=self.font_family, size=10),
                       command=self.update_chessboard_visualization).pack(side="left", padx=3)
        
        ctk.CTkCheckBox(display_control_frame, text="視覺座標系", variable=self.show_image_coordinates,
                       font=ctk.CTkFont(family=self.font_family, size=10),
                       command=self.update_chessboard_visualization).pack(side="left", padx=3)
        
        # 預覽和生成按鈕
        preview_frame = ctk.CTkFrame(parent)
        preview_frame.pack(fill="x", pady=5)
        
        ctk.CTkButton(preview_frame, text="預覽座標系", width=100,
                     font=ctk.CTkFont(family=self.font_family, size=11),
                     command=self.preview_coordinate_system).pack(side="left", padx=3)
        ctk.CTkButton(preview_frame, text="生成座標", width=100,
                     font=ctk.CTkFont(family=self.font_family, size=11),
                     command=self.generate_chessboard_coords).pack(side="left", padx=3)
        
        # 導出按鈕
        gen_frame = ctk.CTkFrame(parent)
        gen_frame.pack(fill="x", pady=2)
        
        ctk.CTkButton(gen_frame, text="導出世界座標", width=100,
                     font=ctk.CTkFont(family=self.font_family, size=11),
                     command=self.export_world_coords).pack(side="left", padx=3)
        ctk.CTkButton(gen_frame, text="載入到點位", width=100,
                     font=ctk.CTkFont(family=self.font_family, size=11),
                     command=self.load_coords_to_points).pack(side="left", padx=3)
        
        # 生成結果顯示
        ctk.CTkLabel(parent, text="生成結果:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(5, 2))
        
        self.chessboard_result_text = ctk.CTkTextbox(parent, height=80,
                                                   font=ctk.CTkFont(family=self.font_family, size=10))
        self.chessboard_result_text.pack(fill="x", pady=2)
    
    def switch_tab(self, tab_id):
        """切換分頁"""
        # 隱藏所有分頁內容
        for content in self.tab_contents.values():
            content.pack_forget()
        
        # 顯示選中的分頁內容
        self.tab_contents[tab_id].pack(fill="both", expand=True, padx=5, pady=5)
        
        # 更新按鈕樣式
        for btn_id, btn in self.tab_buttons.items():
            if btn_id == tab_id:
                btn.configure(fg_color=("gray75", "gray25"))
            else:
                btn.configure(fg_color=("gray84", "gray25"))
        
        self.current_tab = tab_id
        
        # 如果切換到角點檢測頁面，更新可視化
        if tab_id == "corner_detect" and self.loaded_image is not None:
            self.update_corner_visualization()
    
    def import_image(self):
        """導入圖片"""
        file_path = filedialog.askopenfilename(
            title="選擇圖片文件",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # 讀取圖片
                self.loaded_image = cv2.imread(file_path)
                if self.loaded_image is None:
                    raise ValueError("無法讀取圖片文件")
                
                # 重置視圖參數
                self.canvas_zoom = 1.0
                self.canvas_offset_x = 0
                self.canvas_offset_y = 0
                self.detected_corners = None
                
                # 更新顯示
                self.update_corner_visualization()
                
                # 顯示圖片信息
                height, width = self.loaded_image.shape[:2]
                info_text = f"圖片導入成功！\n尺寸: {width} x {height} 像素\n文件: {file_path.split('/')[-1]}"
                self.corner_result_text.delete("1.0", "end")
                self.corner_result_text.insert("1.0", info_text)
                
                messagebox.showinfo("成功", "圖片導入成功！")
                
            except Exception as e:
                messagebox.showerror("錯誤", f"導入圖片失敗: {str(e)}")
    
    def detect_corners(self):
        """檢測棋盤格角點"""
        if self.loaded_image is None:
            messagebox.showwarning("警告", "請先導入圖片！")
            return
        
        try:
            # 獲取棋盤格參數
            pattern_width = int(self.chessboard_width_entry.get())
            pattern_height = int(self.chessboard_height_entry.get())
            pattern_size = (pattern_width, pattern_height)
            
            # 轉換為灰度圖
            gray = cv2.cvtColor(self.loaded_image, cv2.COLOR_BGR2GRAY)
            
            # 檢測角點
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            if ret:
                # 細化角點
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                self.detected_corners = corners.reshape(-1, 2)
                
                # 更新可視化
                self.update_corner_visualization()
                
                # 顯示檢測結果
                result_text = f"角點檢測成功！\n"
                result_text += f"檢測到 {len(self.detected_corners)} 個角點\n"
                result_text += f"棋盤格尺寸: {pattern_width} x {pattern_height}\n\n"
                result_text += f"角點座標預覽:\n"
                
                for i, corner in enumerate(self.detected_corners[:10]):  # 只顯示前10個
                    result_text += f"點 {i+1}: ({corner[0]:.2f}, {corner[1]:.2f})\n"
                
                if len(self.detected_corners) > 10:
                    result_text += f"... 還有 {len(self.detected_corners) - 10} 個點\n"
                
                self.corner_result_text.delete("1.0", "end")
                self.corner_result_text.insert("1.0", result_text)
                
                messagebox.showinfo("成功", f"檢測到 {len(self.detected_corners)} 個角點！")
                
            else:
                messagebox.showerror("失敗", "未檢測到棋盤格角點！\n請檢查圖片和參數設置。")
                
        except Exception as e:
            messagebox.showerror("錯誤", f"角點檢測失敗: {str(e)}")
    
    def export_corner_points(self):
        """導出角點數據"""
        if self.detected_corners is None or len(self.detected_corners) == 0:
            messagebox.showwarning("警告", "請先檢測角點！")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存角點數據",
            defaultextension=".npy",
            initialfile="26_corner_points.npy",  # 使用 initialfile 而不是 initialvalue
            filetypes=[("NPY files", "*.npy"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # 詢問用戶ID從0還是1開始
                start_from_zero = messagebox.askyesno(
                    "ID編號選擇", 
                    "角點ID編號:\n是 - 從0開始 (0,1,2...)\n否 - 從1開始 (1,2,3...)"
                )
                
                # 準備數據：[id, x, y] 格式，確保數據類型正確
                corner_data = []
                for i, corner in enumerate(self.detected_corners):
                    point_id = i if start_from_zero else (i + 1)
                    corner_data.append([
                        float(point_id),  # ID
                        float(corner[0]),  # X座標
                        float(corner[1])   # Y座標
                    ])
                
                corner_array = np.array(corner_data, dtype=np.float32)
                
                # 驗證數據格式
                print(f"導出數據形狀: {corner_array.shape}")
                print(f"導出數據類型: {corner_array.dtype}")
                print(f"前3行數據:\n{corner_array[:3]}")
                
                if file_path.endswith('.csv'):
                    # 導出為CSV格式
                    df = pd.DataFrame(corner_array, columns=['id', 'x', 'y'])
                    # 確保ID是整數格式顯示
                    df['id'] = df['id'].astype(int)
                    df.to_csv(file_path, index=False)
                    messagebox.showinfo("成功", f"角點數據已導出到 {file_path} (CSV格式)")
                else:
                    # 導出為NPY格式
                    np.save(file_path, corner_array)
                    messagebox.showinfo("成功", f"角點數據已導出到 {file_path} (NPY格式)\n共 {len(corner_data)} 個角點\n格式: [id, x, y]")
                
                # 顯示導出的數據統計
                id_start = "0" if start_from_zero else "1"
                id_end = len(corner_data) - 1 if start_from_zero else len(corner_data)
                
                result_text = f"導出完成！\n"
                result_text += f"文件: {file_path.split('/')[-1]}\n"
                result_text += f"格式: {'CSV' if file_path.endswith('.csv') else 'NPY'}\n"
                result_text += f"數據形狀: {corner_array.shape}\n"
                result_text += f"角點數量: {len(corner_data)}\n"
                result_text += f"ID範圍: {id_start} ~ {id_end}\n\n"
                result_text += f"座標範圍:\n"
                result_text += f"  X: {corner_array[:, 1].min():.2f} ~ {corner_array[:, 1].max():.2f}\n"
                result_text += f"  Y: {corner_array[:, 2].min():.2f} ~ {corner_array[:, 2].max():.2f}\n"
                
                self.corner_result_text.delete("1.0", "end")
                self.corner_result_text.insert("1.0", result_text)
                
            except Exception as e:
                messagebox.showerror("錯誤", f"導出失敗: {str(e)}")
                print(f"導出錯誤詳情: {e}")  # 用於調試
    
    def generate_chessboard_coords(self):
        """生成棋盤格世界座標"""
        try:
            # 獲取參數
            origin_x = float(self.origin_x_entry.get())
            origin_y = float(self.origin_y_entry.get())
            grid_x = int(self.grid_x_entry.get())
            grid_y = int(self.grid_y_entry.get())
            spacing = float(self.spacing_entry.get())
            robot_point = int(self.robot_point_entry.get())
            
            robot_x_dir = self.robot_x_direction.get()
            robot_y_dir = self.robot_y_direction.get()
            
            # 驗證機械臂原點範圍
            total_points = grid_x * grid_y
            if robot_point < 1 or robot_point > total_points:
                messagebox.showerror("錯誤", f"機械臂原點必須在1到{total_points}之間！")
                return
            
            # 計算機械臂原點在棋盤格中的位置（從1開始的編號轉換為0開始的索引）
            robot_idx = robot_point - 1
            robot_grid_i = robot_idx % grid_x  # 在grid中的i位置（列）
            robot_grid_j = robot_idx // grid_x  # 在grid中的j位置（行）
            
            # 生成座標
            self.generated_world_coords = []
            
            # 計算機械臂座標軸相對於棋盤格網格的方向係數
            robot_x_coeff_i, robot_x_coeff_j = self.get_grid_direction_coeff(robot_x_dir)
            robot_y_coeff_i, robot_y_coeff_j = self.get_grid_direction_coeff(robot_y_dir)
            
            point_id = 1
            for j in range(grid_y):
                for i in range(grid_x):
                    # 計算相對於機械臂原點的網格偏移
                    grid_offset_i = i - robot_grid_i
                    grid_offset_j = j - robot_grid_j
                    
                    # 將網格偏移轉換為機械臂座標系偏移
                    # 這裡需要解一個2x2的線性方程組
                    # [robot_x_offset] = [robot_x_coeff_i, robot_y_coeff_i] [grid_offset_i]
                    # [robot_y_offset]   [robot_x_coeff_j, robot_y_coeff_j] [grid_offset_j]
                    
                    # 使用逆矩陣求解
                    det = robot_x_coeff_i * robot_y_coeff_j - robot_y_coeff_i * robot_x_coeff_j
                    if abs(det) < 1e-10:
                        messagebox.showerror("錯誤", "座標軸方向設置錯誤：兩個軸不能平行！")
                        return
                    
                    # 逆矩陣計算
                    inv_11 = robot_y_coeff_j / det
                    inv_12 = -robot_y_coeff_i / det
                    inv_21 = -robot_x_coeff_j / det
                    inv_22 = robot_x_coeff_i / det
                    
                    robot_x_offset = inv_11 * grid_offset_i + inv_12 * grid_offset_j
                    robot_y_offset = inv_21 * grid_offset_i + inv_22 * grid_offset_j
                    
                    # 計算世界座標
                    world_x = origin_x + robot_x_offset * spacing
                    world_y = origin_y + robot_y_offset * spacing
                    
                    self.generated_world_coords.append([point_id, world_x, world_y])
                    point_id += 1
            
            # 顯示結果
            result_text = f"座標生成完成！\n"
            result_text += f"總計 {len(self.generated_world_coords)} 個點\n"
            result_text += f"機械臂原點: ({origin_x}, {origin_y}) mm\n"
            result_text += f"棋盤格規格: {grid_x} x {grid_y}\n"
            result_text += f"格子間距: {spacing} mm\n"
            result_text += f"機械臂原點對應: P{robot_point}\n"
            result_text += f"機械臂X+ → {self.get_direction_name(robot_x_dir)}\n"
            result_text += f"機械臂Y+ → {self.get_direction_name(robot_y_dir)}\n\n"
            
            # 顯示關鍵點位座標
            result_text += "關鍵點位座標:\n"
            key_points = [1, robot_point, len(self.generated_world_coords)]
            for point_id in key_points:
                if 1 <= point_id <= len(self.generated_world_coords):
                    coord = self.generated_world_coords[point_id - 1]
                    result_text += f"  P{point_id}: ({coord[1]:.1f}, {coord[2]:.1f}) mm\n"
            
            # 顯示座標範圍
            coords_array = np.array(self.generated_world_coords)
            result_text += f"\n座標範圍:\n"
            result_text += f"  X: {coords_array[:, 1].min():.1f} ~ {coords_array[:, 1].max():.1f} mm\n"
            result_text += f"  Y: {coords_array[:, 2].min():.1f} ~ {coords_array[:, 2].max():.1f} mm\n"
            
            self.chessboard_result_text.delete("1.0", "end")
            self.chessboard_result_text.insert("1.0", result_text)
            
            messagebox.showinfo("成功", f"生成了 {len(self.generated_world_coords)} 個世界座標點！")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"生成座標失敗: {str(e)}")
    
    def get_direction_name(self, direction):
        """獲取方向的中文名稱"""
        direction_names = {
            "image_x_positive": "圖像X+",
            "image_x_negative": "圖像X-", 
            "image_y_positive": "圖像Y+",
            "image_y_negative": "圖像Y-"
        }
        return direction_names.get(direction, "未知")
    
    def get_direction_vector(self, direction, length):
        """根據方向獲取箭頭向量"""
        if direction == "image_x_positive":
            return length, 0
        elif direction == "image_x_negative":
            return -length, 0
        elif direction == "image_y_positive":
            return 0, length
        elif direction == "image_y_negative":
            return 0, -length
        else:
            return length, 0
    
    def get_grid_direction_coeff(self, direction):
        """獲取方向在網格座標系中的係數"""
        if direction == "image_x_positive":  # 圖像X+方向（網格i+方向）
            return 1, 0
        elif direction == "image_x_negative":  # 圖像X-方向（網格i-方向）
            return -1, 0
        elif direction == "image_y_positive":  # 圖像Y+方向（網格j+方向）
            return 0, 1
        elif direction == "image_y_negative":  # 圖像Y-方向（網格j-方向）
            return 0, -1
        else:
            return 1, 0
    
    def export_world_coords(self):
        """導出世界座標"""
        if not hasattr(self, 'generated_world_coords') or not self.generated_world_coords:
            messagebox.showwarning("警告", "請先生成座標！")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存世界座標數據",
            defaultextension=".npy",
            initialfile="world_points.npy",  # 使用 initialfile 而不是 initialvalue
            filetypes=[("NPY files", "*.npy"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # 確保數據格式正確：[id, x, y]
                world_array = np.array(self.generated_world_coords, dtype=np.float32)
                
                # 驗證數據格式
                print(f"導出世界座標形狀: {world_array.shape}")
                print(f"導出世界座標類型: {world_array.dtype}")
                print(f"前3行世界座標:\n{world_array[:3]}")
                
                if file_path.endswith('.csv'):
                    # CSV格式
                    df = pd.DataFrame(world_array, columns=['id', 'world_x', 'world_y'])
                    # 確保ID是整數格式顯示
                    df['id'] = df['id'].astype(int)
                    df.to_csv(file_path, index=False)
                    messagebox.showinfo("成功", f"世界座標已導出到 {file_path} (CSV格式)")
                else:
                    # NPY格式
                    np.save(file_path, world_array)
                    messagebox.showinfo("成功", f"世界座標已導出到 {file_path} (NPY格式)\n共 {len(world_array)} 個點\n格式: [id, x, y]")
                
                # 更新結果顯示
                result_text = f"導出完成！\n"
                result_text += f"文件: {file_path.split('/')[-1]}\n"
                result_text += f"格式: {'CSV' if file_path.endswith('.csv') else 'NPY'}\n"
                result_text += f"數據形狀: {world_array.shape}\n"
                result_text += f"座標點數量: {len(world_array)}\n\n"
                result_text += f"世界座標範圍:\n"
                result_text += f"  X: {world_array[:, 1].min():.2f} ~ {world_array[:, 1].max():.2f} mm\n"
                result_text += f"  Y: {world_array[:, 2].min():.2f} ~ {world_array[:, 2].max():.2f} mm\n"
                
                self.chessboard_result_text.delete("1.0", "end")
                self.chessboard_result_text.insert("1.0", result_text)
                
            except Exception as e:
                messagebox.showerror("錯誤", f"導出失敗: {str(e)}")
                print(f"世界座標導出錯誤: {e}")
    
    def load_coords_to_points(self):
        """載入座標到點位數據"""
        if not hasattr(self, 'generated_world_coords') or not self.generated_world_coords:
            messagebox.showwarning("警告", "請先生成世界座標！")
            return
        
        if self.detected_corners is None:
            messagebox.showwarning("警告", "請先檢測角點！")
            return
        
        if len(self.generated_world_coords) != len(self.detected_corners):
            messagebox.showerror("錯誤", 
                f"角點數量 ({len(self.detected_corners)}) 與世界座標數量 ({len(self.generated_world_coords)}) 不匹配！")
            return
        
        try:
            # 清除現有點位數據
            self.point_data.clear()
            
            # 載入新數據
            for i, (world_coord, image_corner) in enumerate(zip(self.generated_world_coords, self.detected_corners)):
                point_id = world_coord[0]
                world_x = world_coord[1]
                world_y = world_coord[2]
                image_x = image_corner[0]
                image_y = image_corner[1]
                
                self.point_data.append([point_id, image_x, image_y, world_x, world_y])
            
            # 更新顯示
            self.update_points_display()
            self.update_coordinate_arrays()
            self.calculate_transformation()
            
            messagebox.showinfo("成功", f"已載入 {len(self.point_data)} 個點位到標定數據中！")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"載入失敗: {str(e)}")
    
    def preview_coordinate_system(self):
        """預覽座標系設置"""
        try:
            # 獲取參數
            grid_x = int(self.grid_x_entry.get())
            grid_y = int(self.grid_y_entry.get())
            robot_point = int(self.robot_point_entry.get())
            
            robot_x_dir = self.robot_x_direction.get()
            robot_y_dir = self.robot_y_direction.get()
            
            # 切換到可視化視圖
            if self.current_tab != "chessboard_gen":
                return
                
            # 清除圖表
            self.ax.clear()
            
            # 創建棋盤格點位圖示
            total_points = grid_x * grid_y
            
            # 計算棋盤格點位的圖像座標（模擬）
            points_x = []
            points_y = []
            point_ids = []
            
            for j in range(grid_y):
                for i in range(grid_x):
                    # 模擬圖像座標（假設棋盤格在圖像中央）
                    x = 100 + i * 50  # 假設每格50像素
                    y = 100 + j * 50
                    point_id = j * grid_x + i + 1  # 從1開始編號
                    
                    points_x.append(x)
                    points_y.append(y)
                    point_ids.append(point_id)
            
            # 繪製棋盤格點位
            scatter = self.ax.scatter(points_x, points_y, c='lightblue', s=100, alpha=0.7, edgecolor='blue')
            
            # 標出機械臂原點對應的點位
            if 1 <= robot_point <= len(points_x):
                robot_idx = robot_point - 1
                robot_x = points_x[robot_idx]
                robot_y = points_y[robot_idx]
                
                # 高亮機械臂原點
                self.ax.scatter([robot_x], [robot_y], c='red', s=200, marker='*', 
                               edgecolor='darkred', linewidth=2, label=f'機械臂原點 (P{robot_point})')
                
                # 繪製機械臂座標軸箭頭
                arrow_length = 100
                
                # 機械臂X+方向箭頭
                robot_x_dx, robot_x_dy = self.get_direction_vector(robot_x_dir, arrow_length)
                self.ax.arrow(robot_x, robot_y, robot_x_dx, robot_x_dy, 
                             head_width=15, head_length=20, fc='green', ec='green', linewidth=3,
                             label='機械臂X+')
                
                # 機械臂Y+方向箭頭
                robot_y_dx, robot_y_dy = self.get_direction_vector(robot_y_dir, arrow_length)
                self.ax.arrow(robot_x, robot_y, robot_y_dx, robot_y_dy, 
                             head_width=15, head_length=20, fc='orange', ec='orange', linewidth=3,
                             label='機械臂Y+')
                
                # 添加文字說明
                self.ax.text(robot_x + robot_x_dx + 10, robot_y + robot_x_dy + 10, 'X+', 
                           fontsize=12, color='green', weight='bold')
                self.ax.text(robot_x + robot_y_dx + 10, robot_y + robot_y_dy + 10, 'Y+', 
                           fontsize=12, color='orange', weight='bold')
            
            # 添加圖像座標軸參考
            img_x_start = min(points_x) - 50
            img_y_start = max(points_y) + 50
            
            # 圖像X+方向（向右）
            self.ax.arrow(img_x_start, img_y_start, 80, 0, 
                         head_width=10, head_length=15, fc='blue', ec='blue', linewidth=2)
            self.ax.text(img_x_start + 90, img_y_start, '圖像X+', fontsize=10, color='blue')
            
            # 圖像Y+方向（向下）
            self.ax.arrow(img_x_start, img_y_start, 0, 80, 
                         head_width=10, head_length=15, fc='purple', ec='purple', linewidth=2)
            self.ax.text(img_x_start - 20, img_y_start + 90, '圖像Y+', fontsize=10, color='purple')
            
            # 標出第一個和最後一個點位
            if len(points_x) > 0:
                self.ax.annotate('P1', (points_x[0], points_y[0]), 
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=10, color='darkblue', weight='bold')
                self.ax.annotate(f'P{len(points_x)}', (points_x[-1], points_y[-1]), 
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=10, color='darkblue', weight='bold')
            
            self.ax.set_title(f'機械臂座標系預覽 - {grid_x}x{grid_y}棋盤格', fontsize=14, fontweight='bold')
            self.ax.set_xlabel('圖像X座標')
            self.ax.set_ylabel('圖像Y座標') 
            self.ax.legend(loc='upper right')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_aspect('equal')
            
            # 反轉Y軸（圖像座標系）
            self.ax.invert_yaxis()
            
            self.canvas.draw()
            
            # 顯示預覽信息
            preview_text = f"座標系預覽:\n\n"
            preview_text += f"棋盤格規格: {grid_x} x {grid_y} ({grid_x * grid_y} 個點)\n"
            preview_text += f"機械臂原點: P{robot_point}\n"
            preview_text += f"機械臂X+ → {self.get_direction_name(robot_x_dir)}\n"
            preview_text += f"機械臂Y+ → {self.get_direction_name(robot_y_dir)}\n\n"
            preview_text += f"綠色箭頭: 機械臂X+方向\n"
            preview_text += f"橙色箭頭: 機械臂Y+方向\n"
            preview_text += f"紅色星號: 機械臂原點位置\n"
            
            self.chessboard_result_text.delete("1.0", "end")
            self.chessboard_result_text.insert("1.0", preview_text)
            
        except Exception as e:
            messagebox.showerror("錯誤", f"預覽失敗: {str(e)}")
    
    def get_direction_vector(self, direction, length):
        """根據方向獲取箭頭向量"""
        if direction == "image_x_positive":
            return length, 0
        elif direction == "image_x_negative":
            return -length, 0
        elif direction == "image_y_positive":
            return 0, length
        elif direction == "image_y_negative":
            return 0, -length
        else:
            return length, 0
    
    def update_corner_visualization(self):
        """更新角點可視化"""
        if self.loaded_image is None:
            return
        
        # 清除圖表
        self.ax.clear()
        
        # 顯示圖片
        self.ax.imshow(cv2.cvtColor(self.loaded_image, cv2.COLOR_BGR2RGB))
        
        # 如果有檢測到的角點，繪製它們
        if self.detected_corners is not None:
            # 繪製角點
            self.ax.scatter(self.detected_corners[:, 0], self.detected_corners[:, 1], 
                          c='red', s=100, marker='+', linewidths=3, alpha=0.8)
            
            # 添加編號
            for i, corner in enumerate(self.detected_corners):
                self.ax.annotate(str(i+1), (corner[0], corner[1]), 
                               xytext=(8, 8), textcoords='offset points',
                               fontsize=10, color='yellow', weight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        self.ax.set_title('角點檢測結果 (滾輪縮放，右鍵拖動)', fontsize=12, pad=20)
        
        # 設置坐標軸
        self.ax.set_xlabel('X (像素)')
        self.ax.set_ylabel('Y (像素)')
        
        self.canvas.draw()
    
    def import_detected_corners(self):
        """導入檢測到的角點"""
        if self.detected_corners is None or len(self.detected_corners) == 0:
            messagebox.showwarning("警告", "請先在角點檢測頁面檢測角點！")
            return
        
        # 將檢測到的角點保存到棋盤生成頁面使用
        self.imported_corners = self.detected_corners.copy()
        messagebox.showinfo("成功", f"已導入 {len(self.imported_corners)} 個檢測到的角點！")
        
        # 更新可視化
        self.update_chessboard_visualization()
    
    def update_chessboard_visualization(self):
        """更新棋盤生成頁面的可視化"""
        if self.current_tab != "chessboard_gen":
            return
        
        # 清除圖表
        self.ax.clear()
        
        # 獲取當前設置
        try:
            grid_x = int(self.grid_x_entry.get())
            grid_y = int(self.grid_y_entry.get())
            robot_point = int(self.robot_point_entry.get())
        except:
            grid_x, grid_y, robot_point = 17, 12, 204
        
        # 如果顯示視覺座標系（模擬的棋盤格點位）
        if self.show_image_coordinates.get():
            # 創建棋盤格點位的模擬圖像座標
            points_x = []
            points_y = []
            
            for j in range(grid_y):
                for i in range(grid_x):
                    # 模擬圖像座標
                    x = 200 + i * 40
                    y = 150 + j * 40
                    points_x.append(x)
                    points_y.append(y)
            
            # 繪製棋盤格點位
            self.ax.scatter(points_x, points_y, c='lightblue', s=60, alpha=0.7, 
                          edgecolor='blue', label='視覺座標系點位', marker='s')
            
            # 標出機械臂原點在視覺座標系中的位置
            if 1 <= robot_point <= len(points_x):
                robot_idx = robot_point - 1
                robot_x = points_x[robot_idx]
                robot_y = points_y[robot_idx]
                
                self.ax.scatter([robot_x], [robot_y], c='red', s=200, marker='*', 
                              edgecolor='darkred', linewidth=2, label=f'機械臂原點 (P{robot_point})')
                
                # 繪製機械臂座標軸在視覺座標系中的方向
                robot_x_dir = self.robot_x_direction.get()
                robot_y_dir = self.robot_y_direction.get()
                
                arrow_length = 80
                robot_x_dx, robot_x_dy = self.get_direction_vector(robot_x_dir, arrow_length)
                robot_y_dx, robot_y_dy = self.get_direction_vector(robot_y_dir, arrow_length)
                
                self.ax.arrow(robot_x, robot_y, robot_x_dx, robot_x_dy, 
                             head_width=12, head_length=15, fc='green', ec='green', linewidth=3)
                self.ax.arrow(robot_x, robot_y, robot_y_dx, robot_y_dy, 
                             head_width=12, head_length=15, fc='orange', ec='orange', linewidth=3)
                
                self.ax.text(robot_x + robot_x_dx + 15, robot_y + robot_x_dy + 15, '機械臂X+', 
                           fontsize=10, color='green', weight='bold')
                self.ax.text(robot_x + robot_y_dx + 15, robot_y + robot_y_dy + 15, '機械臂Y+', 
                           fontsize=10, color='orange', weight='bold')
        
        # 如果顯示真實世界座標系
        if self.show_world_coordinates.get() and hasattr(self, 'generated_world_coords') and self.generated_world_coords:
            # 提取世界座標
            world_coords = np.array(self.generated_world_coords)
            world_x = world_coords[:, 1]
            world_y = world_coords[:, 2]
            
            # 繪製世界座標點位
            self.ax.scatter(world_x, world_y, c='lightgreen', s=60, alpha=0.7, 
                          edgecolor='darkgreen', label='真實世界座標', marker='o')
            
            # 標出機械臂原點在世界座標系中的位置
            if 1 <= robot_point <= len(world_coords):
                robot_world_x = world_coords[robot_point - 1, 1]
                robot_world_y = world_coords[robot_point - 1, 2]
                
                self.ax.scatter([robot_world_x], [robot_world_y], c='darkred', s=200, marker='*', 
                              edgecolor='red', linewidth=2, label=f'機械臂原點 (世界座標)')
                
                # 在世界座標系中繪製機械臂座標軸
                spacing = float(self.spacing_entry.get()) if self.spacing_entry.get() else 30.0
                
                # 世界座標系中的軸向量（固定方向）
                self.ax.arrow(robot_world_x, robot_world_y, spacing, 0, 
                             head_width=spacing*0.3, head_length=spacing*0.4, fc='cyan', ec='cyan', linewidth=3)
                self.ax.arrow(robot_world_x, robot_world_y, 0, spacing, 
                             head_width=spacing*0.3, head_length=spacing*0.4, fc='magenta', ec='magenta', linewidth=3)
                
                self.ax.text(robot_world_x + spacing + 5, robot_world_y + 5, '世界X+', 
                           fontsize=10, color='cyan', weight='bold')
                self.ax.text(robot_world_x + 5, robot_world_y + spacing + 5, '世界Y+', 
                           fontsize=10, color='magenta', weight='bold')
        
        # 如果有導入的角點且選擇顯示
        if (hasattr(self, 'imported_corners') and self.imported_corners is not None and 
            self.show_detected_corners.get()):
            
            # 顯示檢測到的角點
            self.ax.scatter(self.imported_corners[:, 0], self.imported_corners[:, 1], 
                          c='red', s=100, marker='+', alpha=0.8, linewidths=3,
                          label='檢測到的角點')
            
            # 添加角點編號（只顯示部分以避免過於擁擠）
            step = max(1, len(self.imported_corners) // 20)  # 最多顯示20個編號
            for i in range(0, len(self.imported_corners), step):
                corner = self.imported_corners[i]
                self.ax.annotate(f'P{i+1}', (corner[0], corner[1]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, color='darkred', weight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        # 設置圖表屬性
        self.ax.set_title('棋盤格座標系設置', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('X座標')
        self.ax.set_ylabel('Y座標')
        self.ax.legend(loc='upper right', fontsize=9)
        self.ax.grid(True, alpha=0.3)
        
        # 自動調整視圖範圍
        self.ax.axis('equal')
        
        self.canvas.draw()
    
    def on_scroll(self, event):
        """滾輪縮放事件 - 支援所有頁面"""
        if event.inaxes == self.ax:
            # 獲取當前視圖範圍
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # 縮放係數
            scale_factor = 1.2 if event.step > 0 else 1/1.2
            
            # 計算縮放中心（鼠標位置）
            x_center = event.xdata if event.xdata else (xlim[0] + xlim[1]) / 2
            y_center = event.ydata if event.ydata else (ylim[0] + ylim[1]) / 2
            
            # 計算新的視圖範圍
            x_range = (xlim[1] - xlim[0]) / scale_factor
            y_range = (ylim[1] - ylim[0]) / scale_factor
            
            new_xlim = [x_center - x_range/2, x_center + x_range/2]
            new_ylim = [y_center - y_range/2, y_center + y_range/2]
            
            # 應用新的視圖範圍
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.canvas.draw()
    
    def on_button_press(self, event):
        """鼠標按下事件 - 支援所有頁面"""
        if event.button == 3 and event.inaxes == self.ax:  # 右鍵
            self.mouse_pressed = True
            self.last_mouse_x = event.xdata
            self.last_mouse_y = event.ydata
        elif event.button == 1 and event.inaxes == self.ax:  # 左鍵
            # 只在角點檢測頁面顯示點位信息
            if self.current_tab == "corner_detect":
                self.on_click_point(event)
    
    def on_motion(self, event):
        """鼠標移動事件 - 支援所有頁面"""
        if self.mouse_pressed and event.inaxes == self.ax and event.xdata and event.ydata:
            # 計算移動距離
            dx = event.xdata - self.last_mouse_x
            dy = event.ydata - self.last_mouse_y
            
            # 獲取當前視圖範圍
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # 更新視圖範圍
            self.ax.set_xlim([xlim[0] - dx, xlim[1] - dx])
            self.ax.set_ylim([ylim[0] - dy, ylim[1] - dy])
            
            self.last_mouse_x = event.xdata
            self.last_mouse_y = event.ydata
            
            self.canvas.draw()
    
    def preview_coordinate_system(self):
        """預覽座標系設置"""
        try:
            # 獲取參數
            grid_x = int(self.grid_x_entry.get())
            grid_y = int(self.grid_y_entry.get())
            robot_point = int(self.robot_point_entry.get())
            
            robot_x_dir = self.robot_x_direction.get()
            robot_y_dir = self.robot_y_direction.get()
            
            # 更新可視化
            self.update_chessboard_visualization()
            
            # 顯示預覽信息
            preview_text = f"座標系預覽:\n\n"
            preview_text += f"棋盤格規格: {grid_x} x {grid_y} ({grid_x * grid_y} 個點)\n"
            preview_text += f"機械臂原點: P{robot_point}\n"
            preview_text += f"機械臂X+ → {self.get_direction_name(robot_x_dir)}\n"
            preview_text += f"機械臂Y+ → {self.get_direction_name(robot_y_dir)}\n\n"
            
            preview_text += f"圖例說明:\n"
            if self.show_image_coordinates.get():
                preview_text += f"• 藍色方塊: 視覺座標系點位\n"
                preview_text += f"• 綠色/橙色箭頭: 機械臂座標軸在視覺座標系中的方向\n"
            if self.show_world_coordinates.get():
                preview_text += f"• 綠色圓點: 真實世界座標\n"
                preview_text += f"• 青色/洋紅箭頭: 世界座標軸\n"
            if self.show_detected_corners.get():
                preview_text += f"• 紅色十字: 檢測到的角點\n"
            preview_text += f"• 紅色星號: 機械臂原點位置\n"
            
            self.chessboard_result_text.delete("1.0", "end")
            self.chessboard_result_text.insert("1.0", preview_text)
            
        except Exception as e:
            messagebox.showerror("錯誤", f"預覽失敗: {str(e)}")
    
    def enable_plot_interaction(self):
        """為圖表啟用交互功能（縮放和平移）"""
        # 這個方法現在不需要做特別的事情，因為交互功能已經在事件處理器中實現
        pass
        """更新角點可視化"""
        if self.loaded_image is None:
            return
        
        # 清除圖表
        self.ax.clear()
        
        # 顯示圖片
        self.ax.imshow(cv2.cvtColor(self.loaded_image, cv2.COLOR_BGR2RGB))
        
        # 如果有檢測到的角點，繪製它們
        if self.detected_corners is not None:
            # 繪製角點
            self.ax.scatter(self.detected_corners[:, 0], self.detected_corners[:, 1], 
                          c='red', s=100, marker='+', linewidths=3, alpha=0.8)
            
            # 添加編號
            for i, corner in enumerate(self.detected_corners):
                self.ax.annotate(str(i+1), (corner[0], corner[1]), 
                               xytext=(8, 8), textcoords='offset points',
                               fontsize=10, color='yellow', weight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        self.ax.set_title('角點檢測結果 (滾輪縮放，右鍵拖動)', fontsize=12, pad=20)
        
        # 設置坐標軸
        self.ax.set_xlabel('X (像素)')
        self.ax.set_ylabel('Y (像素)')
        
        self.canvas.draw()
    
    def reset_image_view(self):
        """重置圖片視圖"""
        if self.loaded_image is not None:
            # 重置到圖片的完整視圖
            height, width = self.loaded_image.shape[:2]
            self.ax.set_xlim(0, width)
            self.ax.set_ylim(height, 0)  # Y軸翻轉，因為圖像坐標系原點在左上角
            self.canvas.draw()
    
    def create_intrinsic_controls(self, parent):
        """創建內參控制面板"""
        # 內參矩陣輸入
        ctk.CTkLabel(parent, text="內參矩陣 K:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(10, 5))
        
        self.intrinsic_entries = {}
        intrinsic_labels = [
            ["fx", "skew", "cx"],
            ["0", "fy", "cy"],
            ["0", "0", "1"]
        ]
        
        for i in range(3):
            row_frame = ctk.CTkFrame(parent)
            row_frame.pack(fill="x", pady=2)
            for j in range(3):
                if intrinsic_labels[i][j] in ["0", "1"]:
                    label = ctk.CTkLabel(row_frame, text=intrinsic_labels[i][j], width=80,
                                       font=ctk.CTkFont(family=self.font_family))
                    label.pack(side="left", padx=2)
                else:
                    entry = ctk.CTkEntry(row_frame, width=80, 
                                       font=ctk.CTkFont(family=self.font_family))
                    entry.pack(side="left", padx=2)
                    entry.insert(0, str(self.K[i, j]))
                    entry.bind("<KeyRelease>", self.on_parameter_change)
                    self.intrinsic_entries[f"K_{i}_{j}"] = entry
        
        # 畸變參數
        ctk.CTkLabel(parent, text="畸變參數 D:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(20, 5))
        
        self.distortion_entries = {}
        distortion_labels = ["k1", "k2", "p1", "p2", "k3"]
        
        for i, label in enumerate(distortion_labels):
            row_frame = ctk.CTkFrame(parent)
            row_frame.pack(fill="x", pady=2)
            
            ctk.CTkLabel(row_frame, text=f"{label}:", width=30,
                        font=ctk.CTkFont(family=self.font_family)).pack(side="left", padx=2)
            entry = ctk.CTkEntry(row_frame, width=120,
                               font=ctk.CTkFont(family=self.font_family))
            entry.pack(side="left", padx=2)
            entry.insert(0, str(self.D[i]))
            entry.bind("<KeyRelease>", self.on_parameter_change)
            self.distortion_entries[f"D_{i}"] = entry
            
        # 導入內參按鈕
        import_intrinsic_frame = ctk.CTkFrame(parent)
        import_intrinsic_frame.pack(fill="x", pady=20)
        
        ctk.CTkButton(import_intrinsic_frame, text="導入相機內參(.npy)", 
                     font=ctk.CTkFont(family=self.font_family),
                     command=self.import_intrinsic_npy).pack(side="left", padx=5)
        ctk.CTkButton(import_intrinsic_frame, text="導入畸變係數(.npy)",
                     font=ctk.CTkFont(family=self.font_family),
                     command=self.import_distortion_npy).pack(side="left", padx=5)
            
    def create_extrinsic_controls(self, parent):
        """創建外參控制面板"""
        # 旋轉向量
        ctk.CTkLabel(parent, text="旋轉向量 rvec:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(10, 5))
        
        self.rvec_entries = {}
        rvec_labels = ["rx", "ry", "rz"]
        
        for i, label in enumerate(rvec_labels):
            row_frame = ctk.CTkFrame(parent)
            row_frame.pack(fill="x", pady=2)
            
            ctk.CTkLabel(row_frame, text=f"{label}:", width=30,
                        font=ctk.CTkFont(family=self.font_family)).pack(side="left", padx=2)
            entry = ctk.CTkEntry(row_frame, width=120,
                               font=ctk.CTkFont(family=self.font_family))
            entry.pack(side="left", padx=2)
            entry.insert(0, str(self.rvec[i, 0]))
            entry.bind("<KeyRelease>", self.on_parameter_change)
            self.rvec_entries[f"rvec_{i}"] = entry
            
            # 微調按鈕
            btn_frame = ctk.CTkFrame(row_frame)
            btn_frame.pack(side="right", padx=5)
            
            ctk.CTkButton(btn_frame, text="-", width=30, 
                         font=ctk.CTkFont(family=self.font_family),
                         command=lambda idx=i: self.adjust_rvec(idx, -0.01)).pack(side="left", padx=1)
            ctk.CTkButton(btn_frame, text="+", width=30,
                         font=ctk.CTkFont(family=self.font_family),
                         command=lambda idx=i: self.adjust_rvec(idx, 0.01)).pack(side="left", padx=1)
        
        # 平移向量
        ctk.CTkLabel(parent, text="平移向量 tvec:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(20, 5))
        
        self.tvec_entries = {}
        tvec_labels = ["tx", "ty", "tz"]
        
        for i, label in enumerate(tvec_labels):
            row_frame = ctk.CTkFrame(parent)
            row_frame.pack(fill="x", pady=2)
            
            ctk.CTkLabel(row_frame, text=f"{label}:", width=30,
                        font=ctk.CTkFont(family=self.font_family)).pack(side="left", padx=2)
            entry = ctk.CTkEntry(row_frame, width=120,
                               font=ctk.CTkFont(family=self.font_family))
            entry.pack(side="left", padx=2)
            entry.insert(0, str(self.tvec[i, 0]))
            entry.bind("<KeyRelease>", self.on_parameter_change)
            self.tvec_entries[f"tvec_{i}"] = entry
            
            # 微調按鈕
            btn_frame = ctk.CTkFrame(row_frame)
            btn_frame.pack(side="right", padx=5)
            
            step = 1.0 if i < 2 else 5.0  # x,y用1.0，z用5.0
            ctk.CTkButton(btn_frame, text="-", width=30,
                         font=ctk.CTkFont(family=self.font_family),
                         command=lambda idx=i, s=step: self.adjust_tvec(idx, -s)).pack(side="left", padx=1)
            ctk.CTkButton(btn_frame, text="+", width=30,
                         font=ctk.CTkFont(family=self.font_family),
                         command=lambda idx=i, s=step: self.adjust_tvec(idx, s)).pack(side="left", padx=1)
        
        # 外參操作按鈕
        extrinsic_btn_frame = ctk.CTkFrame(parent)
        extrinsic_btn_frame.pack(fill="x", pady=20)
        
        ctk.CTkButton(extrinsic_btn_frame, text="重置外參", 
                     font=ctk.CTkFont(family=self.font_family),
                     command=self.reset_extrinsic).pack(side="left", padx=5)
        ctk.CTkButton(extrinsic_btn_frame, text="導入外參(.npy)",
                     font=ctk.CTkFont(family=self.font_family),
                     command=self.import_extrinsic_npy).pack(side="left", padx=5)
        
    def create_points_controls(self, parent):
        """創建點位數據控制面板"""
        # 添加單個點位
        ctk.CTkLabel(parent, text="添加點位:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(10, 5))
        
        # 點位ID
        id_frame = ctk.CTkFrame(parent)
        id_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(id_frame, text="ID:", width=50,
                    font=ctk.CTkFont(family=self.font_family)).pack(side="left", padx=2)
        self.point_id_entry = ctk.CTkEntry(id_frame, width=80,
                                          font=ctk.CTkFont(family=self.font_family))
        self.point_id_entry.pack(side="left", padx=2)
        
        # 圖像座標
        img_frame = ctk.CTkFrame(parent)
        img_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(img_frame, text="圖像座標:", width=80,
                    font=ctk.CTkFont(family=self.font_family)).pack(side="left", padx=2)
        self.img_x_entry = ctk.CTkEntry(img_frame, width=60, placeholder_text="x",
                                       font=ctk.CTkFont(family=self.font_family))
        self.img_x_entry.pack(side="left", padx=2)
        self.img_y_entry = ctk.CTkEntry(img_frame, width=60, placeholder_text="y",
                                       font=ctk.CTkFont(family=self.font_family))
        self.img_y_entry.pack(side="left", padx=2)
        
        # 世界座標
        world_frame = ctk.CTkFrame(parent)
        world_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(world_frame, text="世界座標:", width=80,
                    font=ctk.CTkFont(family=self.font_family)).pack(side="left", padx=2)
        self.world_x_entry = ctk.CTkEntry(world_frame, width=60, placeholder_text="x",
                                         font=ctk.CTkFont(family=self.font_family))
        self.world_x_entry.pack(side="left", padx=2)
        self.world_y_entry = ctk.CTkEntry(world_frame, width=60, placeholder_text="y",
                                         font=ctk.CTkFont(family=self.font_family))
        self.world_y_entry.pack(side="left", padx=2)
        
        # 添加和清除按鈕
        btn_frame = ctk.CTkFrame(parent)
        btn_frame.pack(fill="x", pady=10)
        ctk.CTkButton(btn_frame, text="添加點位", 
                     font=ctk.CTkFont(family=self.font_family),
                     command=self.add_point).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="清除所有",
                     font=ctk.CTkFont(family=self.font_family),
                     command=self.clear_points).pack(side="left", padx=5)
        
        # 點位列表
        ctk.CTkLabel(parent, text="當前點位:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(20, 5))
        
        # 創建表格框架
        table_frame = ctk.CTkFrame(parent)
        table_frame.pack(fill="both", expand=True, pady=5)
        
        # 表格標題
        header_frame = ctk.CTkFrame(table_frame)
        header_frame.pack(fill="x", padx=5, pady=5)
        
        headers = ["ID", "圖像X", "圖像Y", "世界X", "世界Y", "操作"]
        widths = [40, 50, 50, 50, 50, 60]
        
        for header, width in zip(headers, widths):
            ctk.CTkLabel(header_frame, text=header, width=width, 
                        font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(side="left", padx=1)
        
        # 滾動框架用於點位列表
        self.points_scroll_frame = ctk.CTkScrollableFrame(table_frame, height=200)
        self.points_scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
    def create_algorithm_controls(self, parent):
        """創建算法選擇控制面板"""
        ctk.CTkLabel(parent, text="外參估算算法:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(10, 5))
        
        # 算法選擇
        algorithm_frame = ctk.CTkFrame(parent)
        algorithm_frame.pack(fill="x", pady=5)
        
        algorithms = [
            ("PnP_ITERATIVE", "PnP迭代法"),
            ("PnP_EPNP", "EPnP算法"),
            ("PnP_P3P", "P3P算法"),
            ("PnP_AP3P", "AP3P算法"),
            ("PnP_IPPE", "IPPE算法"),
            ("PnP_IPPE_SQUARE", "IPPE_SQUARE算法")
        ]
        
        self.algorithm_combo = ctk.CTkComboBox(
            algorithm_frame,
            values=[f"{alg[1]} ({alg[0]})" for alg in algorithms],
            font=ctk.CTkFont(family=self.font_family),
            width=300
        )
        self.algorithm_combo.pack(padx=5, pady=5)
        self.algorithm_combo.set("PnP迭代法 (PnP_ITERATIVE)")
        
        # 算法說明
        algo_desc = ctk.CTkTextbox(parent, height=150, wrap="word",
                                  font=ctk.CTkFont(family=self.font_family, size=11))
        algo_desc.pack(fill="x", pady=10)
        
        algo_text = """
算法說明:

• PnP迭代法: 最常用的方法，通過迭代優化求解，適合大多數情況
• EPnP算法: 效率較高，適用於點數較多的情況  
• P3P算法: 只需3個點，但可能有多解，適合點數少的情況
• AP3P算法: P3P的改進版本，數值穩定性更好
• IPPE算法: 適用於平面物體的姿態估計
• IPPE_SQUARE算法: IPPE的改進版本，適用於正方形標定板

建議：
- 點數>=4時推薦使用PnP迭代法
- 點數較多(>10)時可使用EPnP
- 平面標定時可嘗試IPPE算法
        """
        algo_desc.insert("1.0", algo_text)
        algo_desc.configure(state="disabled")
        
        # 執行按鈕
        execute_frame = ctk.CTkFrame(parent)
        execute_frame.pack(fill="x", pady=20)
        
        ctk.CTkButton(execute_frame, text="計算外參", 
                     font=ctk.CTkFont(family=self.font_family),
                     command=self.estimate_extrinsic).pack(side="left", padx=5)
        ctk.CTkButton(execute_frame, text="導出估算外參",
                     font=ctk.CTkFont(family=self.font_family),
                     command=self.export_estimated_extrinsic).pack(side="left", padx=5)
        
        # 結果顯示
        ctk.CTkLabel(parent, text="估算結果:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(20, 5))
        
        self.estimation_result = ctk.CTkTextbox(parent, height=200,
                                               font=ctk.CTkFont(family=self.font_family, size=11))
        self.estimation_result.pack(fill="x", pady=5)
        
    def create_view_controls(self, parent):
        """創建顯示控制面板"""
        ctk.CTkLabel(parent, text="顯示控制:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(10, 5))
        
        # 顯示選項
        view_options = [
            ("show_image_coords", "顯示圖像座標", self.show_image_coords),
            ("show_world_coords", "顯示真實世界座標", self.show_world_coords),
            ("show_transformed_coords", "顯示轉換後座標", self.show_transformed_coords),
            ("show_error_lines", "顯示誤差線", self.show_error_lines)
        ]
        
        for option_id, text, var in view_options:
            checkbox = ctk.CTkCheckBox(
                parent, 
                text=text, 
                variable=var,
                font=ctk.CTkFont(family=self.font_family),
                command=self.update_visualization
            )
            checkbox.pack(anchor="w", pady=5, padx=10)
        
        # 圖表設置
        ctk.CTkLabel(parent, text="圖表設置:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(20, 5))
        
        # 點大小調整
        size_frame = ctk.CTkFrame(parent)
        size_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(size_frame, text="點大小:", font=ctk.CTkFont(family=self.font_family)).pack(side="left", padx=5)
        self.point_size_slider = ctk.CTkSlider(size_frame, from_=50, to=200, number_of_steps=15)
        self.point_size_slider.pack(side="right", padx=5, fill="x", expand=True)
        self.point_size_slider.set(100)
        self.point_size_slider.configure(command=self.update_visualization)
        
        # 線寬調整
        width_frame = ctk.CTkFrame(parent)
        width_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(width_frame, text="線寬:", font=ctk.CTkFont(family=self.font_family)).pack(side="left", padx=5)
        self.line_width_slider = ctk.CTkSlider(width_frame, from_=0.5, to=3.0, number_of_steps=25)
        self.line_width_slider.pack(side="right", padx=5, fill="x", expand=True)
        self.line_width_slider.set(1.0)
        self.line_width_slider.configure(command=self.update_visualization)
        
        # 刷新按鈕
        ctk.CTkButton(parent, text="刷新圖表", 
                     font=ctk.CTkFont(family=self.font_family),
                     command=self.update_visualization).pack(pady=20)
        
    def create_file_controls(self, parent):
        """創建文件操作控制面板"""
        ctk.CTkLabel(parent, text="文件操作:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(10, 5))
        
        # 導入數據
        import_frame = ctk.CTkFrame(parent)
        import_frame.pack(fill="x", pady=5)
        
        ctk.CTkButton(import_frame, text="導入CSV點位", 
                     font=ctk.CTkFont(family=self.font_family),
                     command=self.import_csv).pack(side="left", padx=5)
        ctk.CTkButton(import_frame, text="導入NPY數據",
                     font=ctk.CTkFont(family=self.font_family),
                     command=self.import_npy).pack(side="left", padx=5)
        
        # 導出數據
        export_frame = ctk.CTkFrame(parent)
        export_frame.pack(fill="x", pady=5)
        
        ctk.CTkButton(export_frame, text="導出參數",
                     font=ctk.CTkFont(family=self.font_family),
                     command=self.export_params).pack(side="left", padx=5)
        ctk.CTkButton(export_frame, text="導出點位",
                     font=ctk.CTkFont(family=self.font_family),
                     command=self.export_points).pack(side="left", padx=5)
        
        # 批量輸入區域
        ctk.CTkLabel(parent, text="批量輸入 (格式: id,img_x,img_y,world_x,world_y):", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", pady=(20, 5))
        
        self.batch_text = ctk.CTkTextbox(parent, height=150,
                                        font=ctk.CTkFont(family=self.font_family))
        self.batch_text.pack(fill="x", pady=5)
        self.batch_text.insert("1.0", "1,100,200,10.5,20.3\n2,150,250,15.2,25.1\n3,200,300,20.8,30.5")
        
        ctk.CTkButton(parent, text="批量添加點位",
                     font=ctk.CTkFont(family=self.font_family),
                     command=self.batch_add_points).pack(pady=10)
        
    def create_help_controls(self, parent):
        """創建說明頁面"""
        # 創建滾動文本框用於顯示說明
        help_text = ctk.CTkTextbox(parent, height=600, wrap="word",
                                  font=ctk.CTkFont(family="PingFang SC", size=12))
        help_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 插入說明內容
        help_content = """
🎯 相機標定調整工具使用說明

═══════════════════════════════════════════════════════════════

📖 工具概述
────────────────────────────────────────────────────────────────
本工具用於精細調整相機內參和外參矩陣，通過已知的圖像座標點和對應的世界座標點，
實現從相機座標系到世界座標系的精確轉換。特別適用於需要將相機捕獲的二維圖像
座標轉換為三維世界座標的應用場景。

🔧 功能模塊說明
────────────────────────────────────────────────────────────────

📌 1. 內參矩陣 (Intrinsic Matrix)
• fx, fy: 相機焦距（像素單位）
• cx, cy: 主點座標（光軸與像平面交點）
• 畸變係數: k1,k2,k3（徑向畸變），p1,p2（切向畸變）
• 支援導入.npy格式的內參和畸變係數檔案

📌 2. 外參矩陣 (Extrinsic Matrix)
• 旋轉向量 (rvec): 相機座標系相對世界座標系的旋轉
• 平移向量 (tvec): 相機座標系原點在世界座標系中的位置
• 微調按鈕: 點擊 +/- 進行精細調整
• 支援導入算法估算的外參檔案

📌 3. 點位數據 (Point Data)
• 手動添加: 逐個輸入對應點對
• 批量添加: 使用 CSV 格式批量導入
• 格式: id,image_x,image_y,world_x,world_y

📌 4. 算法估算 (Algorithm Estimation)
• 多種PnP算法可選擇
• 自動計算初始外參估值
• 可導出估算結果進行微調

📌 5. 角點檢測 (Corner Detection) 🆕
• 導入棋盤格圖片進行自動角點檢測
• 可視化顯示檢測到的角點位置和編號
• 支持圖片縮放和平移查看
• 導出26_corner_points.npy格式的角點數據
• 可設定棋盤格內角點數量參數

📌 6. 棋盤生成 (Chessboard Generation) 🆕
• 基於機械臂座標系生成標準棋盤格世界座標
• 設定原點位置和座標系方向
• 可調整格子數量和間距
• 自動生成對應的世界座標點
• 支持直接載入到點位數據進行標定

📌 7. 顯示控制 (View Control)
• 可選擇顯示/隱藏不同類型的座標點
• 圖像座標、世界座標、轉換座標獨立控制
• 可調整點大小和線寬

📌 8. 文件操作 (File Operations)
• 導入CSV: 支持多種列名格式
• 導入NPY: 兼容numpy數組格式
• 導出參數: JSON格式保存相機參數
• 導出點位: CSV格式保存點位數據

🎯 操作指南
────────────────────────────────────────────────────────────────

🚀 方法一：使用角點檢測功能
1. 切換到「角點檢測」頁面
2. 點擊「導入圖片」選擇棋盤格圖片
3. 設定棋盤格內角點數（如7x5）
4. 點擊「檢測角點」自動識別角點
5. 在可視化窗口中查看檢測結果（可縮放平移）
6. 點擊「導出角點數據」保存26_corner_points.npy

🚀 方法二：使用棋盤生成功能
1. 切換到「棋盤生成」頁面
2. 設定原點座標（機械臂座標系中的位置）
3. 輸入X、Y方向的格子數量
4. 設定格子間距（實際物理距離mm）
5. 選擇座標系方向（與機械臂座標軸對應）
6. 點擊「生成座標」計算世界座標
7. 點擊「導出世界座標」或「載入到點位」

⚠️ 注意事項
────────────────────────────────────────────────────────────────
• 本工具假設所有世界座標點都在Z=0平面上
• 角點檢測需要清晰的棋盤格圖片，光照要均勻
• 棋盤格放置時要確保與設定的座標系方向一致
• 機械臂標定時建議使用棋盤格邊角的角點作為原點
• 點位分布應盡可能均勻覆蓋整個感興趣區域
        """
        
        help_text.insert("1.0", help_content)
        help_text.configure(state="disabled")  # 設為只讀
        
    def create_visualization_panel(self, parent):
        """創建可視化面板"""
        # 創建matplotlib圖形
        self.fig = Figure(figsize=(12, 9), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # 創建畫布
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        # 綁定鼠標事件（用於角點檢測頁面的圖片交互）
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # 誤差信息面板
        info_frame = ctk.CTkFrame(parent, height=100)
        info_frame.pack(fill="x", padx=10, pady=(0, 10))
        info_frame.pack_propagate(False)
        
        ctk.CTkLabel(info_frame, text="轉換誤差信息:", 
                    font=ctk.CTkFont(family=self.font_family, weight="bold")).pack(anchor="w", padx=10, pady=5)
        
        self.error_label = ctk.CTkLabel(info_frame, text="", justify="left",
                                       font=ctk.CTkFont(family=self.font_family))
        self.error_label.pack(anchor="w", padx=10, pady=5)
    
    def on_scroll(self, event):
        """滾輪縮放事件"""
        if self.current_tab == "corner_detect" and self.loaded_image is not None:
            if event.inaxes == self.ax:
                # 獲取當前視圖範圍
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                
                # 縮放係數
                scale_factor = 1.2 if event.step > 0 else 1/1.2
                
                # 計算縮放中心（鼠標位置）
                x_center = event.xdata if event.xdata else (xlim[0] + xlim[1]) / 2
                y_center = event.ydata if event.ydata else (ylim[0] + ylim[1]) / 2
                
                # 計算新的視圖範圍
                x_range = (xlim[1] - xlim[0]) / scale_factor
                y_range = (ylim[1] - ylim[0]) / scale_factor
                
                new_xlim = [x_center - x_range/2, x_center + x_range/2]
                new_ylim = [y_center - y_range/2, y_center + y_range/2]
                
                # 應用新的視圖範圍
                self.ax.set_xlim(new_xlim)
                self.ax.set_ylim(new_ylim)
                self.canvas.draw()
    
    def on_button_press(self, event):
        """鼠標按下事件"""
        if self.current_tab == "corner_detect" and self.loaded_image is not None:
            if event.button == 3 and event.inaxes == self.ax:  # 右鍵
                self.mouse_pressed = True
                self.last_mouse_x = event.xdata
                self.last_mouse_y = event.ydata
            elif event.button == 1 and event.inaxes == self.ax:  # 左鍵
                self.on_click_point(event)
    
    def on_button_release(self, event):
        """鼠標釋放事件"""
        if event.button == 3:  # 右鍵
            self.mouse_pressed = False
    
    def on_motion(self, event):
        """鼠標移動事件"""
        if self.current_tab == "corner_detect" and self.loaded_image is not None:
            if self.mouse_pressed and event.inaxes == self.ax and event.xdata and event.ydata:
                # 計算移動距離
                dx = event.xdata - self.last_mouse_x
                dy = event.ydata - self.last_mouse_y
                
                # 獲取當前視圖範圍
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                
                # 更新視圖範圍
                self.ax.set_xlim([xlim[0] - dx, xlim[1] - dx])
                self.ax.set_ylim([ylim[0] - dy, ylim[1] - dy])
                
                self.last_mouse_x = event.xdata
                self.last_mouse_y = event.ydata
                
                self.canvas.draw()
    
    def on_click_point(self, event):
        """點擊查看點位信息"""
        if self.detected_corners is not None and event.inaxes == self.ax:
            # 找到最近的角點
            click_x, click_y = event.xdata, event.ydata
            if click_x is not None and click_y is not None:
                distances = np.sqrt((self.detected_corners[:, 0] - click_x)**2 + 
                                  (self.detected_corners[:, 1] - click_y)**2)
                closest_idx = np.argmin(distances)
                
                if distances[closest_idx] < 20:  # 在20像素範圍內
                    corner = self.detected_corners[closest_idx]
                    info = f"角點 {closest_idx + 1}: ({corner[0]:.2f}, {corner[1]:.2f})"
                    messagebox.showinfo("角點信息", info)

    # 以下是所有其他方法的完整實現
    
    def import_intrinsic_npy(self):
        """導入內參矩陣.npy檔案"""
        file_path = filedialog.askopenfilename(
            title="選擇相機內參檔案 (camera_matrix.npy)",
            filetypes=[("NPY files", "*.npy"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                K = np.load(file_path)
                if K.shape == (3, 3):
                    self.K = K
                    # 更新界面顯示
                    self.intrinsic_entries["K_0_0"].delete(0, "end")
                    self.intrinsic_entries["K_0_0"].insert(0, str(K[0, 0]))
                    self.intrinsic_entries["K_0_2"].delete(0, "end")
                    self.intrinsic_entries["K_0_2"].insert(0, str(K[0, 2]))
                    self.intrinsic_entries["K_1_1"].delete(0, "end")
                    self.intrinsic_entries["K_1_1"].insert(0, str(K[1, 1]))
                    self.intrinsic_entries["K_1_2"].delete(0, "end")
                    self.intrinsic_entries["K_1_2"].insert(0, str(K[1, 2]))
                    
                    self.calculate_transformation()
                    messagebox.showinfo("成功", "內參矩陣導入成功！")
                else:
                    messagebox.showerror("錯誤", "內參矩陣格式不正確，應為3x3矩陣！")
            except Exception as e:
                messagebox.showerror("錯誤", f"導入失敗: {str(e)}")
    
    def import_distortion_npy(self):
        """導入畸變係數.npy檔案"""
        file_path = filedialog.askopenfilename(
            title="選擇畸變係數檔案 (dist_coeffs.npy)",
            filetypes=[("NPY files", "*.npy"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                D = np.load(file_path)
                # 處理不同的畸變係數格式
                if D.shape == (1, 5):
                    D = D.ravel()
                elif D.shape == (5,):
                    pass
                elif D.shape == (5, 1):
                    D = D.ravel()
                else:
                    raise ValueError("畸變係數格式不正確")
                
                self.D = D
                # 更新界面顯示
                for i in range(5):
                    self.distortion_entries[f"D_{i}"].delete(0, "end")
                    self.distortion_entries[f"D_{i}"].insert(0, str(D[i]))
                
                self.calculate_transformation()
                messagebox.showinfo("成功", "畸變係數導入成功！")
            except Exception as e:
                messagebox.showerror("錯誤", f"導入失敗: {str(e)}")
    
    def import_extrinsic_npy(self):
        """導入外參.npy檔案"""
        file_path = filedialog.askopenfilename(
            title="選擇外參檔案 (extrinsic.npy 或包含rvec,tvec的檔案)",
            filetypes=[("NPY files", "*.npy"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    # JSON格式
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if 'rotation_vector' in data and 'translation_vector' in data:
                        rvec = np.array(data['rotation_vector'])
                        tvec = np.array(data['translation_vector'])
                    else:
                        raise ValueError("JSON檔案格式不正確")
                else:
                    # NPY格式
                    data = np.load(file_path, allow_pickle=True)
                    
                    if isinstance(data, np.ndarray) and data.shape == ():
                        # 字典格式
                        data = data.item()
                        rvec = data['rvec']
                        tvec = data['tvec']
                    elif isinstance(data, dict):
                        rvec = data['rvec']
                        tvec = data['tvec']
                    else:
                        raise ValueError("NPY檔案格式不正確")
                
                # 確保形狀正確
                if rvec.shape == (3,):
                    rvec = rvec.reshape(3, 1)
                if tvec.shape == (3,):
                    tvec = tvec.reshape(3, 1)
                
                self.rvec = rvec
                self.tvec = tvec
                
                # 更新界面顯示
                for i in range(3):
                    self.rvec_entries[f"rvec_{i}"].delete(0, "end")
                    self.rvec_entries[f"rvec_{i}"].insert(0, str(rvec[i, 0]))
                    self.tvec_entries[f"tvec_{i}"].delete(0, "end")
                    self.tvec_entries[f"tvec_{i}"].insert(0, str(tvec[i, 0]))
                
                self.calculate_transformation()
                messagebox.showinfo("成功", "外參導入成功！")
            except Exception as e:
                messagebox.showerror("錯誤", f"導入失敗: {str(e)}")
    
    def estimate_extrinsic(self):
        """使用選定算法估算外參"""
        if len(self.point_data) < 4:
            messagebox.showwarning("警告", "至少需要4個點位進行外參估算！")
            return
        
        try:
            # 準備數據
            object_points = np.array([[p[3], p[4], 0.0] for p in self.point_data], dtype=np.float32)
            image_points = np.array([[p[1], p[2]] for p in self.point_data], dtype=np.float32)
            
            # 獲取選定的算法
            algo_text = self.algorithm_combo.get()
            if "PnP_ITERATIVE" in algo_text:
                flag = cv2.SOLVEPNP_ITERATIVE
            elif "PnP_EPNP" in algo_text:
                flag = cv2.SOLVEPNP_EPNP
            elif "PnP_P3P" in algo_text:
                flag = cv2.SOLVEPNP_P3P
            elif "PnP_AP3P" in algo_text:
                flag = cv2.SOLVEPNP_AP3P
            elif "PnP_IPPE" in algo_text:
                flag = cv2.SOLVEPNP_IPPE
            elif "PnP_IPPE_SQUARE" in algo_text:
                flag = cv2.SOLVEPNP_IPPE_SQUARE
            else:
                flag = cv2.SOLVEPNP_ITERATIVE
            
            # 執行PnP求解
            success, rvec_est, tvec_est = cv2.solvePnP(
                object_points, image_points, self.K, self.D, flags=flag
            )
            
            if success:
                self.estimated_rvec = rvec_est
                self.estimated_tvec = tvec_est
                
                # 計算重投影誤差
                projected_points, _ = cv2.projectPoints(
                    object_points, rvec_est, tvec_est, self.K, self.D
                )
                projected_points = projected_points.reshape(-1, 2)
                
                errors = []
                for i in range(len(image_points)):
                    error = np.linalg.norm(image_points[i] - projected_points[i])
                    errors.append(error)
                
                mean_error = np.mean(errors)
                max_error = np.max(errors)
                min_error = np.min(errors)
                
                # 顯示結果
                result_text = f"=== {algo_text} 估算結果 ===\n\n"
                result_text += f"旋轉向量 (rvec):\n"
                result_text += f"  rx = {rvec_est[0, 0]:.6f}\n"
                result_text += f"  ry = {rvec_est[1, 0]:.6f}\n"
                result_text += f"  rz = {rvec_est[2, 0]:.6f}\n\n"
                
                result_text += f"平移向量 (tvec):\n"
                result_text += f"  tx = {tvec_est[0, 0]:.6f}\n"
                result_text += f"  ty = {tvec_est[1, 0]:.6f}\n"
                result_text += f"  tz = {tvec_est[2, 0]:.6f}\n\n"
                
                result_text += f"重投影誤差統計:\n"
                result_text += f"  平均誤差: {mean_error:.4f} 像素\n"
                result_text += f"  最大誤差: {max_error:.4f} 像素\n"
                result_text += f"  最小誤差: {min_error:.4f} 像素\n\n"
                
                result_text += f"使用點位數量: {len(object_points)}\n"
                result_text += f"算法類型: {algo_text}\n"
                
                self.estimation_result.delete("1.0", "end")
                self.estimation_result.insert("1.0", result_text)
                
                messagebox.showinfo("成功", f"外參估算完成！\n平均重投影誤差: {mean_error:.4f} 像素")
                
            else:
                messagebox.showerror("錯誤", "外參估算失敗！請檢查點位數據是否正確。")
                
        except Exception as e:
            messagebox.showerror("錯誤", f"估算過程發生錯誤: {str(e)}")
    
    def export_estimated_extrinsic(self):
        """導出估算的外參"""
        if not hasattr(self, 'estimated_rvec') or not hasattr(self, 'estimated_tvec'):
            messagebox.showwarning("警告", "請先執行外參估算！")
            return
        
        # 選擇保存格式
        save_format = messagebox.askyesno("選擇格式", "選擇保存格式:\n是 - NPY格式\n否 - JSON格式")
        
        if save_format:  # NPY格式
            file_path = filedialog.asksaveasfilename(
                title="保存估算外參 (NPY格式)",
                defaultextension=".npy",
                filetypes=[("NPY files", "*.npy"), ("All files", "*.*")]
            )
            
            if file_path:
                try:
                    extrinsic_data = {
                        'rvec': self.estimated_rvec,
                        'tvec': self.estimated_tvec,
                        'algorithm': self.algorithm_combo.get(),
                        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                    }
                    np.save(file_path, extrinsic_data)
                    messagebox.showinfo("成功", "估算外參導出成功！(NPY格式)")
                except Exception as e:
                    messagebox.showerror("錯誤", f"導出失敗: {str(e)}")
        else:  # JSON格式
            file_path = filedialog.asksaveasfilename(
                title="保存估算外參 (JSON格式)",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if file_path:
                try:
                    extrinsic_data = {
                        'rotation_vector': self.estimated_rvec.tolist(),
                        'translation_vector': self.estimated_tvec.tolist(),
                        'algorithm': self.algorithm_combo.get(),
                        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                    }
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(extrinsic_data, f, indent=4, ensure_ascii=False)
                    messagebox.showinfo("成功", "估算外參導出成功！(JSON格式)")
                except Exception as e:
                    messagebox.showerror("錯誤", f"導出失敗: {str(e)}")
    
    def update_visualization(self, *args):
        """更新可視化"""
        self.calculate_transformation()
        
    def add_point(self):
        """添加單個點位"""
        try:
            point_id = int(self.point_id_entry.get())
            img_x = float(self.img_x_entry.get())
            img_y = float(self.img_y_entry.get())
            world_x = float(self.world_x_entry.get())
            world_y = float(self.world_y_entry.get())
            
            # 檢查ID是否已存在
            if any(p[0] == point_id for p in self.point_data):
                messagebox.showwarning("警告", f"點位ID {point_id} 已存在！")
                return
            
            self.point_data.append([point_id, img_x, img_y, world_x, world_y])
            self.update_points_display()
            self.update_coordinate_arrays()
            self.calculate_transformation()
            
            # 清空輸入框
            self.point_id_entry.delete(0, "end")
            self.img_x_entry.delete(0, "end")
            self.img_y_entry.delete(0, "end")
            self.world_x_entry.delete(0, "end")
            self.world_y_entry.delete(0, "end")
            
        except ValueError:
            messagebox.showerror("錯誤", "請輸入有效的數值！")
    
    def batch_add_points(self):
        """批量添加點位"""
        try:
            text_content = self.batch_text.get("1.0", "end-1c")
            lines = text_content.strip().split('\n')
            
            added_count = 0
            for line in lines:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) == 5:
                        point_id = int(parts[0])
                        img_x = float(parts[1])
                        img_y = float(parts[2])
                        world_x = float(parts[3])
                        world_y = float(parts[4])
                        
                        # 檢查ID是否已存在
                        if not any(p[0] == point_id for p in self.point_data):
                            self.point_data.append([point_id, img_x, img_y, world_x, world_y])
                            added_count += 1
            
            if added_count > 0:
                self.update_points_display()
                self.update_coordinate_arrays()
                self.calculate_transformation()
                messagebox.showinfo("成功", f"成功添加 {added_count} 個點位！")
            else:
                messagebox.showwarning("警告", "沒有有效的點位數據被添加！")
                
        except Exception as e:
            messagebox.showerror("錯誤", f"批量添加失敗: {str(e)}")
    
    def clear_points(self):
        """清除所有點位"""
        if messagebox.askyesno("確認", "確定要清除所有點位數據嗎？"):
            self.point_data.clear()
            self.update_points_display()
            self.update_coordinate_arrays()
            self.calculate_transformation()
    
    def delete_point(self, point_id):
        """刪除指定點位"""
        self.point_data = [p for p in self.point_data if p[0] != point_id]
        self.update_points_display()
        self.update_coordinate_arrays()
        self.calculate_transformation()
    
    def update_points_display(self):
        """更新點位顯示"""
        # 清除現有顯示
        for widget in self.points_scroll_frame.winfo_children():
            widget.destroy()
        
        # 顯示每個點位
        for point in sorted(self.point_data, key=lambda x: x[0]):
            point_frame = ctk.CTkFrame(self.points_scroll_frame)
            point_frame.pack(fill="x", pady=1)
            
            # 顯示數據
            ctk.CTkLabel(point_frame, text=str(point[0]), width=40,
                        font=ctk.CTkFont(family=self.font_family)).pack(side="left", padx=1)
            ctk.CTkLabel(point_frame, text=f"{point[1]:.1f}", width=50,
                        font=ctk.CTkFont(family=self.font_family)).pack(side="left", padx=1)
            ctk.CTkLabel(point_frame, text=f"{point[2]:.1f}", width=50,
                        font=ctk.CTkFont(family=self.font_family)).pack(side="left", padx=1)
            ctk.CTkLabel(point_frame, text=f"{point[3]:.1f}", width=50,
                        font=ctk.CTkFont(family=self.font_family)).pack(side="left", padx=1)
            ctk.CTkLabel(point_frame, text=f"{point[4]:.1f}", width=50,
                        font=ctk.CTkFont(family=self.font_family)).pack(side="left", padx=1)
            
            # 刪除按鈕
            ctk.CTkButton(point_frame, text="刪除", width=60, 
                         font=ctk.CTkFont(family=self.font_family),
                         command=lambda pid=point[0]: self.delete_point(pid)).pack(side="left", padx=1)
    
    def update_coordinate_arrays(self):
        """更新座標數組"""
        if len(self.point_data) > 0:
            sorted_points = sorted(self.point_data, key=lambda x: x[0])
            self.image_coords = np.array([[p[1], p[2]] for p in sorted_points], dtype=np.float32)
            self.world_coords = np.array([[p[3], p[4], 0.0] for p in sorted_points], dtype=np.float32)
        else:
            self.image_coords = np.array([])
            self.world_coords = np.array([])
    
    def adjust_rvec(self, index, delta):
        """微調旋轉向量"""
        current_value = float(self.rvec_entries[f"rvec_{index}"].get())
        new_value = current_value + delta
        self.rvec_entries[f"rvec_{index}"].delete(0, "end")
        self.rvec_entries[f"rvec_{index}"].insert(0, f"{new_value:.6f}")
        self.on_parameter_change()
    
    def adjust_tvec(self, index, delta):
        """微調平移向量"""
        current_value = float(self.tvec_entries[f"tvec_{index}"].get())
        new_value = current_value + delta
        self.tvec_entries[f"tvec_{index}"].delete(0, "end")
        self.tvec_entries[f"tvec_{index}"].insert(0, f"{new_value:.6f}")
        self.on_parameter_change()
    
    def reset_extrinsic(self):
        """重置外參到默認值"""
        default_rvec = np.array([[-2.17796294], [-2.24565035], [0.02621215]])
        default_tvec = np.array([[330.20053861], [48.63793437], [533.5402696]])
        
        for i in range(3):
            self.rvec_entries[f"rvec_{i}"].delete(0, "end")
            self.rvec_entries[f"rvec_{i}"].insert(0, str(default_rvec[i, 0]))
            self.tvec_entries[f"tvec_{i}"].delete(0, "end")
            self.tvec_entries[f"tvec_{i}"].insert(0, str(default_tvec[i, 0]))
        
        self.on_parameter_change()
    
    def on_parameter_change(self, event=None):
        """參數變化時的回調"""
        try:
            self.update_parameters_from_entries()
            self.calculate_transformation()
        except:
            pass  # 輸入不完整時忽略錯誤
    
    def update_parameters_from_entries(self):
        """從輸入框更新參數"""
        # 更新內參矩陣
        self.K[0, 0] = float(self.intrinsic_entries["K_0_0"].get())
        self.K[0, 2] = float(self.intrinsic_entries["K_0_2"].get())
        self.K[1, 1] = float(self.intrinsic_entries["K_1_1"].get())
        self.K[1, 2] = float(self.intrinsic_entries["K_1_2"].get())
        
        # 更新畸變參數
        for i in range(5):
            self.D[i] = float(self.distortion_entries[f"D_{i}"].get())
        
        # 更新外參
        for i in range(3):
            self.rvec[i, 0] = float(self.rvec_entries[f"rvec_{i}"].get())
            self.tvec[i, 0] = float(self.tvec_entries[f"tvec_{i}"].get())
    
    def calculate_transformation(self):
        """計算座標轉換並更新可視化"""
        if len(self.image_coords) == 0:
            self.plot_empty()
            return
        
        try:
            # 計算旋轉矩陣
            R, _ = cv2.Rodrigues(self.rvec)
            
            # 計算反投影世界座標
            transformed_points = []
            for uv in self.image_coords:
                # 去畸變
                undistorted_uv = cv2.undistortPoints(
                    uv.reshape(1, 1, 2), self.K, self.D, P=self.K).reshape(-1)
                
                # 轉換到相機座標系
                uv_hom = np.array([undistorted_uv[0], undistorted_uv[1], 1.0])
                cam_coords = np.linalg.inv(self.K) @ uv_hom
                
                # 計算Z=0平面上的點
                s = (0 - self.tvec[2, 0]) / (R[2] @ cam_coords)
                XYZ_cam = s * cam_coords
                
                # 轉換到世界座標系
                world_point = np.linalg.inv(R) @ (XYZ_cam - self.tvec.ravel())
                transformed_points.append(world_point[:2])
            
            self.transformed_points = np.array(transformed_points)
            
            # 計算誤差
            errors = []
            if len(self.world_coords) > 0:
                for i in range(len(self.transformed_points)):
                    error = np.linalg.norm(self.world_coords[i, :2] - self.transformed_points[i])
                    errors.append(error)
            
            # 更新可視化
            self.plot_results(errors)
            
        except Exception as e:
            print(f"計算錯誤: {e}")
            self.plot_empty()
    
    def plot_results(self, errors):
        """繪製結果"""
        # 如果在角點檢測頁面，使用角點可視化
        if self.current_tab == "corner_detect":
            self.update_corner_visualization()
            return
        elif self.current_tab == "chessboard_gen":
            # 如果在棋盤生成頁面且有預覽數據，顯示預覽
            if hasattr(self, 'chessboard_preview_mode') and self.chessboard_preview_mode:
                return
            self.update_chessboard_visualization()
            return
            
        self.ax.clear()
        
        if len(self.image_coords) == 0:
            self.ax.text(0.5, 0.5, '請添加點位數據', ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return
        
        # 獲取顯示參數
        point_size = self.point_size_slider.get() if hasattr(self, 'point_size_slider') else 100
        line_width = self.line_width_slider.get() if hasattr(self, 'line_width_slider') else 1.0
        
        # 創建顏色映射
        n_points = len(self.image_coords)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))
        
        # 繪製圖像座標（如果選擇顯示）
        if self.show_image_coords.get():
            # 將圖像座標縮放到合適的範圍進行顯示
            img_coords_scaled = self.image_coords / 10  # 簡單縮放
            self.ax.scatter(img_coords_scaled[:, 0], img_coords_scaled[:, 1],
                           c=colors, s=point_size, marker='o', edgecolor='black', alpha=0.8,
                           label='圖像座標 (縮放)')
        
        # 繪製真實世界座標
        if self.show_world_coords.get() and len(self.world_coords) > 0:
            self.ax.scatter(self.world_coords[:, 0], self.world_coords[:, 1],
                           c=colors, s=point_size, marker='s', edgecolor='black', alpha=0.8,
                           label='真實世界座標')
        
        # 繪製轉換後座標
        if self.show_transformed_coords.get():
            self.ax.scatter(self.transformed_points[:, 0], self.transformed_points[:, 1],
                           c=colors, s=point_size, marker='^', edgecolor='black', alpha=0.8,
                           label='轉換後座標')
        
        # 繪製誤差線
        if self.show_error_lines.get() and len(self.world_coords) > 0 and self.show_world_coords.get() and self.show_transformed_coords.get():
            for i in range(len(self.transformed_points)):
                self.ax.plot([self.world_coords[i, 0], self.transformed_points[i, 0]],
                           [self.world_coords[i, 1], self.transformed_points[i, 1]], 
                           'r--', alpha=0.6, linewidth=line_width)
        
        # 添加點位標籤
        for i, point in enumerate(self.point_data):
            point_id = point[0]
            
            # 圖像座標標籤
            if self.show_image_coords.get():
                img_coords_scaled = self.image_coords[i] / 10
                self.ax.annotate(f'I{point_id}', 
                               (img_coords_scaled[0], img_coords_scaled[1]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8, color='green')
            
            # 世界座標標籤
            if self.show_world_coords.get() and len(self.world_coords) > 0:
                self.ax.annotate(f'W{point_id}', 
                               (self.world_coords[i, 0], self.world_coords[i, 1]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8, color='blue')
            
            # 轉換座標標籤
            if self.show_transformed_coords.get():
                self.ax.annotate(f'T{point_id}', 
                               (self.transformed_points[i, 0], self.transformed_points[i, 1]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8, color='orange')
        
        self.ax.set_title('座標轉換結果對比', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.axis('equal')
        
        # 更新誤差信息
        if errors:
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            min_error = np.min(errors)
            std_error = np.std(errors)
            error_text = f"平均誤差: {mean_error:.2f} mm | 最大誤差: {max_error:.2f} mm | 最小誤差: {min_error:.2f} mm | 標準差: {std_error:.2f} mm"
            self.error_label.configure(text=error_text)
        else:
            self.error_label.configure(text="無誤差數據")
        
        # 為座標轉換結果對比添加交互功能
        self.enable_plot_interaction()
        
        self.canvas.draw()
    
    def import_detected_corners(self):
        """導入檢測到的角點"""
        if self.detected_corners is None or len(self.detected_corners) == 0:
            messagebox.showwarning("警告", "請先在角點檢測頁面檢測角點！")
            return
        
        # 將檢測到的角點保存到棋盤生成頁面使用
        self.imported_corners = self.detected_corners.copy()
        messagebox.showinfo("成功", f"已導入 {len(self.imported_corners)} 個檢測到的角點！")
        
        # 更新可視化
        self.update_chessboard_visualization()
    
    def preview_coordinate_system(self):
        """預覽座標系設置"""
        try:
            # 設置預覽模式
            self.chessboard_preview_mode = True
            
            # 獲取參數
            grid_x = int(self.grid_x_entry.get())
            grid_y = int(self.grid_y_entry.get())
            robot_point = int(self.robot_point_entry.get())
            
            robot_x_dir = self.robot_x_direction.get()
            robot_y_dir = self.robot_y_direction.get()
            
            # 更新可視化
            self.update_chessboard_visualization()
            
            # 顯示預覽信息
            preview_text = f"座標系預覽:\n\n"
            preview_text += f"棋盤格規格: {grid_x} x {grid_y} ({grid_x * grid_y} 個點)\n"
            preview_text += f"機械臂原點: P{robot_point}\n"
            preview_text += f"機械臂X+ → {self.get_direction_name(robot_x_dir)}\n"
            preview_text += f"機械臂Y+ → {self.get_direction_name(robot_y_dir)}\n\n"
            preview_text += f"綠色箭頭: 機械臂X+方向\n"
            preview_text += f"橙色箭頭: 機械臂Y+方向\n"
            preview_text += f"紅色星號: 機械臂原點位置\n"
            
            self.chessboard_result_text.delete("1.0", "end")
            self.chessboard_result_text.insert("1.0", preview_text)
            
        except Exception as e:
            messagebox.showerror("錯誤", f"預覽失敗: {str(e)}")
        finally:
            # 重置預覽模式
            self.chessboard_preview_mode = False
    
    def enable_plot_interaction(self):
        """為圖表啟用交互功能（縮放和平移）"""
        # 重複綁定事件處理器來確保在所有頁面都能使用交互功能
        if not hasattr(self, 'interaction_connected'):
            # 為所有可視化圖表啟用滾輪縮放和右鍵拖動功能
            self.interaction_connected = True
    
    def plot_empty(self):
        """繪製空圖表"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, '請添加點位數據\n或檢查參數設置', 
                    ha='center', va='center', transform=self.ax.transAxes,
                    fontsize=12)
        self.ax.set_title('座標轉換結果對比')
        self.canvas.draw()
        self.error_label.configure(text="")
    
    def import_csv(self):
        """導入CSV點位數據"""
        file_path = filedialog.askopenfilename(
            title="選擇CSV文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                df = pd.read_csv(file_path)
                
                # 嘗試不同的列名格式
                possible_columns = [
                    ['id', 'image_x', 'image_y', 'world_x', 'world_y'],
                    ['ID', 'img_x', 'img_y', 'world_x', 'world_y'],
                    ['point_id', 'pixel_x', 'pixel_y', 'real_x', 'real_y']
                ]
                
                columns_found = None
                for cols in possible_columns:
                    if all(col in df.columns for col in cols):
                        columns_found = cols
                        break
                
                if columns_found:
                    for _, row in df.iterrows():
                        point_id = int(row[columns_found[0]])
                        if not any(p[0] == point_id for p in self.point_data):
                            self.point_data.append([
                                point_id,
                                float(row[columns_found[1]]),
                                float(row[columns_found[2]]),
                                float(row[columns_found[3]]),
                                float(row[columns_found[4]])
                            ])
                    
                    self.update_points_display()
                    self.update_coordinate_arrays()
                    self.calculate_transformation()
                    messagebox.showinfo("成功", f"成功導入 {len(df)} 個點位！")
                else:
                    messagebox.showerror("錯誤", "CSV文件格式不正確！\n需要包含: id, image_x, image_y, world_x, world_y")
                    
            except Exception as e:
                messagebox.showerror("錯誤", f"導入失敗: {str(e)}")
    
    def import_npy(self):
        """導入NPY數據"""
        corner_file = filedialog.askopenfilename(
            title="選擇角點數據文件 (corner_points.npy)",
            filetypes=[("NPY files", "*.npy"), ("All files", "*.*")]
        )
        
        if corner_file:
            world_file = filedialog.askopenfilename(
                title="選擇世界座標數據文件 (world_points.npy)",
                filetypes=[("NPY files", "*.npy"), ("All files", "*.*")]
            )
            
            if world_file:
                try:
                    corner_data = np.load(corner_file)  # [id, x, y]
                    world_data = np.load(world_file)    # [id, x, y]
                    
                    print(f"角點數據形狀: {corner_data.shape}")
                    print(f"世界座標數據形狀: {world_data.shape}")
                    print(f"角點前3行:\n{corner_data[:3]}")
                    print(f"世界座標前3行:\n{world_data[:3]}")
                    
                    # 處理ID可能從0開始的情況
                    corner_dict = {}
                    world_dict = {}
                    
                    for row in corner_data:
                        point_id = int(row[0])
                        # 如果ID從0開始，調整為從1開始
                        adjusted_id = point_id + 1 if corner_data[:, 0].min() == 0 else point_id
                        corner_dict[adjusted_id] = row[1:]
                    
                    for row in world_data:
                        point_id = int(row[0])
                        # 如果ID從0開始，調整為從1開始
                        adjusted_id = point_id + 1 if world_data[:, 0].min() == 0 else point_id
                        world_dict[adjusted_id] = row[1:]
                    
                    # 找共同點
                    common_ids = sorted(set(corner_dict.keys()) & set(world_dict.keys()))
                    
                    print(f"共同ID數量: {len(common_ids)}")
                    print(f"前10個共同ID: {common_ids[:10]}")
                    
                    # 添加點位
                    added_count = 0
                    for point_id in common_ids:
                        if not any(p[0] == point_id for p in self.point_data):
                            corner_coords = corner_dict[point_id]
                            world_coords = world_dict[point_id]
                            
                            self.point_data.append([
                                point_id,
                                float(corner_coords[0]),  # image_x
                                float(corner_coords[1]),  # image_y
                                float(world_coords[0]),   # world_x
                                float(world_coords[1])    # world_y
                            ])
                            added_count += 1
                    
                    self.update_points_display()
                    self.update_coordinate_arrays()
                    self.calculate_transformation()
                    
                    # 顯示詳細信息
                    info_msg = f"成功導入 {added_count} 個點位！\n\n"
                    info_msg += f"角點數據範圍:\n"
                    info_msg += f"  X: {corner_data[:, 1].min():.1f} ~ {corner_data[:, 1].max():.1f}\n"
                    info_msg += f"  Y: {corner_data[:, 2].min():.1f} ~ {corner_data[:, 2].max():.1f}\n\n"
                    info_msg += f"世界座標範圍:\n"
                    info_msg += f"  X: {world_data[:, 1].min():.1f} ~ {world_data[:, 1].max():.1f}\n"
                    info_msg += f"  Y: {world_data[:, 2].min():.1f} ~ {world_data[:, 2].max():.1f}\n"
                    
                    messagebox.showinfo("導入成功", info_msg)
                    
                except Exception as e:
                    messagebox.showerror("錯誤", f"導入失敗: {str(e)}")
                    print(f"導入錯誤詳情: {e}")  # 調試信息
    
    def export_params(self):
        """導出參數"""
        file_path = filedialog.asksaveasfilename(
            title="保存參數文件",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                params = {
                    "intrinsic_matrix": self.K.tolist(),
                    "distortion_coefficients": self.D.tolist(),
                    "rotation_vector": self.rvec.tolist(),
                    "translation_vector": self.tvec.tolist(),
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(params, f, indent=4, ensure_ascii=False)
                
                messagebox.showinfo("成功", "參數導出成功！")
                
            except Exception as e:
                messagebox.showerror("錯誤", f"導出失敗: {str(e)}")
    
    def export_points(self):
        """導出點位數據"""
        if not self.point_data:
            messagebox.showwarning("警告", "沒有點位數據可導出！")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存點位數據",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                df = pd.DataFrame(self.point_data, 
                                columns=['id', 'image_x', 'image_y', 'world_x', 'world_y'])
                df.to_csv(file_path, index=False)
                messagebox.showinfo("成功", "點位數據導出成功！")
                
            except Exception as e:
                messagebox.showerror("錯誤", f"導出失敗: {str(e)}")
    
    def run(self):
        """運行應用"""
        self.root.mainloop()

# 使用範例
if __name__ == "__main__":
    # 創建並運行應用
    app = CameraCalibrationTool()
    app.run()