import customtkinter as ctk
import cv2
import numpy as np
import os
import glob
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import threading
from datetime import datetime
import math

class InteractiveImageCanvas:
    def __init__(self, parent, display_settings=None):
        self.parent = parent
        self.canvas = ctk.CTkCanvas(parent, bg="gray20")
        self.canvas.pack(fill="both", expand=True)
        
        # 圖像數據
        self.original_image = None
        self.display_image = None
        self.corners = None
        
        # 顯示設置
        self.display_settings = display_settings or {
            'corner_radius': 8,
            'cursor_style': 'circle',  # 'circle' or 'cross'
            'text_size': 12,
            'corner_color': 'lime',
            'border_color': 'red',
            'text_color': 'yellow'
        }
        
        # 縮放和平移
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.min_scale = 0.1
        self.max_scale = 5.0
        
        # 拖拽狀態
        self.dragging = False
        self.last_x = 0
        self.last_y = 0
        
        # 提示泡泡
        self.tooltip = None
        self.tooltip_visible = False
        
        # 綁定事件
        self.bind_events()
        
    def bind_events(self):
        # 滾輪縮放
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)  # Linux
        self.canvas.bind("<Button-5>", self.on_mousewheel)  # Linux
        
        # 拖拽平移
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # 滑鼠移動
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", self.on_mouse_leave)
        
        # 右鍵重置
        self.canvas.bind("<Button-3>", self.reset_view)
        
    def set_image(self, cv_image, corners=None):
        """設置要顯示的圖像和角點"""
        self.original_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.corners = corners
        self.reset_view()
        
    def reset_view(self, event=None):
        """重置視圖到初始狀態"""
        if self.original_image is None:
            return
            
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.canvas.after(100, self.reset_view)
            return
            
        img_height, img_width = self.original_image.shape[:2]
        
        # 計算適合的縮放比例
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self.scale = min(scale_x, scale_y) * 0.9
        
        # 居中
        self.offset_x = (canvas_width - img_width * self.scale) / 2
        self.offset_y = (canvas_height - img_height * self.scale) / 2
        
        self.update_display()
        
    def on_mousewheel(self, event):
        """處理滾輪縮放"""
        if self.original_image is None:
            return
            
        # 獲取滑鼠位置
        mouse_x = self.canvas.canvasx(event.x)
        mouse_y = self.canvas.canvasy(event.y)
        
        # 計算縮放因子
        if event.delta > 0 or event.num == 4:
            zoom_factor = 1.1
        else:
            zoom_factor = 0.9
            
        old_scale = self.scale
        self.scale *= zoom_factor
        self.scale = max(self.min_scale, min(self.max_scale, self.scale))
        
        if self.scale != old_scale:
            # 調整偏移以保持滑鼠位置不變
            scale_change = self.scale / old_scale
            self.offset_x = mouse_x - (mouse_x - self.offset_x) * scale_change
            self.offset_y = mouse_y - (mouse_y - self.offset_y) * scale_change
            
            self.update_display()
            
    def on_click(self, event):
        """處理點擊事件"""
        if self.original_image is None:
            return
            
        self.dragging = True
        self.last_x = event.x
        self.last_y = event.y
        
        # 檢查是否點擊在角點上
        if self.corners is not None:
            clicked_corner = self.get_corner_at_position(event.x, event.y)
            if clicked_corner is not None:
                self.show_corner_info(clicked_corner, event.x, event.y)
                return
                
        self.hide_tooltip()
        
    def on_drag(self, event):
        """處理拖拽事件"""
        if not self.dragging or self.original_image is None:
            return
            
        dx = event.x - self.last_x
        dy = event.y - self.last_y
        
        self.offset_x += dx
        self.offset_y += dy
        
        self.last_x = event.x
        self.last_y = event.y
        
        self.update_display()
        
    def on_release(self, event):
        """處理釋放事件"""
        self.dragging = False
        
    def on_mouse_move(self, event):
        """處理滑鼠移動"""
        if self.corners is None or self.dragging:
            return
            
        corner_idx = self.get_corner_at_position(event.x, event.y)
        if corner_idx is not None:
            self.canvas.configure(cursor="hand2")
        else:
            self.canvas.configure(cursor="")
            
    def on_mouse_leave(self, event):
        """滑鼠離開時隱藏提示"""
        self.hide_tooltip()
        self.canvas.configure(cursor="")
        
    def update_display_settings(self, settings):
        """更新顯示設置"""
        self.display_settings.update(settings)
        if self.original_image is not None:
            self.update_display()
        
    def get_corner_at_position(self, screen_x, screen_y):
        """獲取指定螢幕位置的角點索引"""
        if self.corners is None:
            return None
            
        corner_radius = self.display_settings['corner_radius']
        
        for i, corner in enumerate(self.corners):
            corner_x, corner_y = corner.ravel()
            
            # 轉換到螢幕座標
            screen_corner_x = corner_x * self.scale + self.offset_x
            screen_corner_y = corner_y * self.scale + self.offset_y
            
            # 檢查距離
            distance = math.sqrt((screen_x - screen_corner_x)**2 + (screen_y - screen_corner_y)**2)
            if distance <= corner_radius:
                return i
                
        return None
        
    def show_corner_info(self, corner_idx, screen_x, screen_y):
        """顯示角點信息泡泡"""
        self.hide_tooltip()
        
        corner_x, corner_y = self.corners[corner_idx].ravel()
        
        # 創建提示框
        self.tooltip = ctk.CTkToplevel(self.parent)
        self.tooltip.withdraw()  # 先隱藏
        self.tooltip.overrideredirect(True)  # 去除邊框
        self.tooltip.attributes('-topmost', True)
        
        # 提示內容
        info_text = f"點 {corner_idx}\n座標: ({corner_x:.2f}, {corner_y:.2f})"
        label = ctk.CTkLabel(self.tooltip, text=info_text, 
                           fg_color="gray10", corner_radius=8,
                           text_color="white")
        label.pack(padx=8, pady=4)
        
        # 計算位置
        root_x = self.canvas.winfo_rootx()
        root_y = self.canvas.winfo_rooty()
        
        tooltip_x = root_x + screen_x + 10
        tooltip_y = root_y + screen_y - 30
        
        self.tooltip.geometry(f"+{tooltip_x}+{tooltip_y}")
        self.tooltip.deiconify()  # 顯示
        self.tooltip_visible = True
        
        # 3秒後自動隱藏
        self.canvas.after(3000, self.hide_tooltip)
        
    def hide_tooltip(self):
        """隱藏提示泡泡"""
        if self.tooltip and self.tooltip_visible:
            self.tooltip.destroy()
            self.tooltip = None
            self.tooltip_visible = False
            
    def update_display(self):
        """更新顯示"""
        if self.original_image is None:
            return
            
        self.canvas.delete("all")
        
        # 調整圖像大小
        img_height, img_width = self.original_image.shape[:2]
        new_width = int(img_width * self.scale)
        new_height = int(img_height * self.scale)
        
        if new_width > 0 and new_height > 0:
            resized_img = cv2.resize(self.original_image, (new_width, new_height))
            
            # 轉換為PhotoImage
            pil_img = Image.fromarray(resized_img)
            self.photo = ImageTk.PhotoImage(pil_img)
            
            # 顯示圖像
            self.canvas.create_image(self.offset_x, self.offset_y, 
                                   anchor="nw", image=self.photo)
            
            # 繪製角點
            if self.corners is not None:
                self.draw_corners()
                
    def draw_corners(self):
        """繪製角點"""
        settings = self.display_settings
        corner_radius = settings['corner_radius']
        cursor_style = settings['cursor_style']
        text_size = settings['text_size']
        corner_color = settings['corner_color']
        border_color = settings['border_color']
        text_color = settings['text_color']
        
        for i, corner in enumerate(self.corners):
            corner_x, corner_y = corner.ravel()
            
            # 轉換到螢幕座標
            screen_x = corner_x * self.scale + self.offset_x
            screen_y = corner_y * self.scale + self.offset_y
            
            # 根據cursor_style繪製不同形狀
            if cursor_style == 'circle':
                # 繪製圓形光標
                self.canvas.create_oval(
                    screen_x - corner_radius, screen_y - corner_radius,
                    screen_x + corner_radius, screen_y + corner_radius,
                    fill=corner_color, outline=border_color, width=2
                )
            elif cursor_style == 'cross':
                # 繪製十字光標
                line_length = corner_radius
                self.canvas.create_line(
                    screen_x - line_length, screen_y,
                    screen_x + line_length, screen_y,
                    fill=corner_color, width=3
                )
                self.canvas.create_line(
                    screen_x, screen_y - line_length,
                    screen_x, screen_y + line_length,
                    fill=corner_color, width=3
                )
                # 添加中心點
                center_radius = max(2, corner_radius // 4)
                self.canvas.create_oval(
                    screen_x - center_radius, screen_y - center_radius,
                    screen_x + center_radius, screen_y + center_radius,
                    fill=border_color, outline=border_color
                )
            
            # 繪製編號
            font_size = max(8, int(text_size * min(self.scale, 1.5)))
            self.canvas.create_text(
                screen_x, screen_y - corner_radius - 8,
                text=str(i), fill=text_color, 
                font=("Arial", font_size, "bold")
            )

class CameraCalibrationTool:
    def __init__(self):
        # 設置 CustomTkinter 主題
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # 主窗口
        self.root = ctk.CTk()
        self.root.title("相機標定工具 - Camera Calibration Tool")
        self.root.geometry("1400x900")
        
        # 棋盤格參數（預設值）
        self.checkerboard_width = ctk.IntVar(value=17)
        self.checkerboard_height = ctk.IntVar(value=12)
        self.square_size = ctk.DoubleVar(value=1.0)  # 公分
        
        # 標定參數
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # 數據存儲
        self.image_paths = []
        self.processed_images = {}  # {path: {"corners": corners, "success": bool}}
        self.camera_matrix = None
        self.dist_coeffs = None
        self.reprojection_error = None
        
        # 當前顯示的圖像
        self.current_image_index = 0
        self.current_image_display = None
        
        # 顯示設置
        self.display_settings = {
            'corner_radius': 8,
            'cursor_style': 'circle',  # 'circle' or 'cross'
            'text_size': 12,
            'corner_color': 'lime',
            'border_color': 'red',
            'text_color': 'yellow'
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # 主框架
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 左側控制面板
        self.setup_control_panel(main_frame)
        
        # 右側顯示區域
        self.setup_display_area(main_frame)
        
    def setup_control_panel(self, parent):
        control_frame = ctk.CTkFrame(parent)
        control_frame.pack(side="left", fill="y", padx=(0, 10))
        
        # 標題
        title_label = ctk.CTkLabel(control_frame, text="相機標定工具", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=10)
        
        # 棋盤格設置
        self.setup_checkerboard_settings(control_frame)
        
        # 圖像管理
        self.setup_image_management(control_frame)
        
        # 顯示設置
        self.setup_display_settings(control_frame)
        
        # 標定控制
        self.setup_calibration_controls(control_frame)
        
        # 結果顯示
        self.setup_results_display(control_frame)
        
    def setup_checkerboard_settings(self, parent):
        settings_frame = ctk.CTkFrame(parent)
        settings_frame.pack(fill="x", pady=10, padx=10)
        
        ctk.CTkLabel(settings_frame, text="棋盤格設置", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        # 寬度設置
        width_frame = ctk.CTkFrame(settings_frame)
        width_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(width_frame, text="寬度(格數):").pack(side="left", padx=5)
        width_entry = ctk.CTkEntry(width_frame, textvariable=self.checkerboard_width, width=80)
        width_entry.pack(side="right", padx=5)
        
        # 高度設置
        height_frame = ctk.CTkFrame(settings_frame)
        height_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(height_frame, text="高度(格數):").pack(side="left", padx=5)
        height_entry = ctk.CTkEntry(height_frame, textvariable=self.checkerboard_height, width=80)
        height_entry.pack(side="right", padx=5)
        
        # 格子大小設置
        size_frame = ctk.CTkFrame(settings_frame)
        size_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(size_frame, text="格子大小(cm):").pack(side="left", padx=5)
        size_entry = ctk.CTkEntry(size_frame, textvariable=self.square_size, width=80)
        size_entry.pack(side="right", padx=5)
        
    def setup_image_management(self, parent):
        image_frame = ctk.CTkFrame(parent)
        image_frame.pack(fill="x", pady=10, padx=10)
        
        ctk.CTkLabel(image_frame, text="圖像管理", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        # 導入圖像按鈕
        import_btn = ctk.CTkButton(image_frame, text="導入圖像", command=self.import_images)
        import_btn.pack(fill="x", pady=2)
        
        # 清除圖像按鈕
        clear_btn = ctk.CTkButton(image_frame, text="清除所有圖像", command=self.clear_images)
        clear_btn.pack(fill="x", pady=2)
        
        # 圖像列表
        self.image_listbox = ctk.CTkScrollableFrame(image_frame, height=150)
        self.image_listbox.pack(fill="both", expand=True, pady=5)
        
        # 圖像導航
        nav_frame = ctk.CTkFrame(image_frame)
        nav_frame.pack(fill="x", pady=5)
        
        self.prev_btn = ctk.CTkButton(nav_frame, text="上一張", command=self.prev_image, width=70)
        self.prev_btn.pack(side="left", padx=2)
        
        self.next_btn = ctk.CTkButton(nav_frame, text="下一張", command=self.next_image, width=70)
        self.next_btn.pack(side="right", padx=2)
        
        self.image_info_label = ctk.CTkLabel(nav_frame, text="0/0")
        self.image_info_label.pack()
        
    def setup_display_settings(self, parent):
        settings_frame = ctk.CTkFrame(parent)
        settings_frame.pack(fill="x", pady=10, padx=10)
        
        ctk.CTkLabel(settings_frame, text="顯示設置", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        # 點尺寸設置
        size_frame = ctk.CTkFrame(settings_frame)
        size_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(size_frame, text="點尺寸:").pack(side="left", padx=5)
        self.corner_size_var = ctk.IntVar(value=self.display_settings['corner_radius'])
        size_slider = ctk.CTkSlider(size_frame, from_=4, to=20, number_of_steps=16,
                                   variable=self.corner_size_var, command=self.update_corner_size)
        size_slider.pack(side="right", padx=5, fill="x", expand=True)
        
        # 光標樣式設置
        cursor_frame = ctk.CTkFrame(settings_frame)
        cursor_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(cursor_frame, text="光標樣式:").pack(side="left", padx=5)
        self.cursor_style_var = ctk.StringVar(value=self.display_settings['cursor_style'])
        cursor_menu = ctk.CTkOptionMenu(cursor_frame, values=["circle", "cross"],
                                       variable=self.cursor_style_var, command=self.update_cursor_style)
        cursor_menu.pack(side="right", padx=5)
        
        # 編號大小設置
        text_frame = ctk.CTkFrame(settings_frame)
        text_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(text_frame, text="編號大小:").pack(side="left", padx=5)
        self.text_size_var = ctk.IntVar(value=self.display_settings['text_size'])
        text_slider = ctk.CTkSlider(text_frame, from_=8, to=24, number_of_steps=16,
                                   variable=self.text_size_var, command=self.update_text_size)
        text_slider.pack(side="right", padx=5, fill="x", expand=True)
        
        # 顏色設置
        color_frame = ctk.CTkFrame(settings_frame)
        color_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(color_frame, text="顏色方案:").pack(side="left", padx=5)
        self.color_scheme_var = ctk.StringVar(value="綠紅黃")
        color_menu = ctk.CTkOptionMenu(color_frame, values=["綠紅黃", "藍白橙", "紅白黃", "紫青白"],
                                      variable=self.color_scheme_var, command=self.update_color_scheme)
        color_menu.pack(side="right", padx=5)
        
    def update_corner_size(self, value):
        """更新角點尺寸"""
        self.display_settings['corner_radius'] = int(value)
        if hasattr(self, 'interactive_canvas'):
            self.interactive_canvas.update_display_settings(self.display_settings)
            
    def update_cursor_style(self, value):
        """更新光標樣式"""
        self.display_settings['cursor_style'] = value
        if hasattr(self, 'interactive_canvas'):
            self.interactive_canvas.update_display_settings(self.display_settings)
            
    def update_text_size(self, value):
        """更新文字大小"""
        self.display_settings['text_size'] = int(value)
        if hasattr(self, 'interactive_canvas'):
            self.interactive_canvas.update_display_settings(self.display_settings)
            
    def update_color_scheme(self, value):
        """更新顏色方案"""
        color_schemes = {
            "綠紅黃": {'corner_color': 'lime', 'border_color': 'red', 'text_color': 'yellow'},
            "藍白橙": {'corner_color': 'blue', 'border_color': 'white', 'text_color': 'orange'},
            "紅白黃": {'corner_color': 'red', 'border_color': 'white', 'text_color': 'yellow'},
            "紫青白": {'corner_color': 'purple', 'border_color': 'cyan', 'text_color': 'white'}
        }
        if value in color_schemes:
            self.display_settings.update(color_schemes[value])
            if hasattr(self, 'interactive_canvas'):
                self.interactive_canvas.update_display_settings(self.display_settings)
        
    def setup_calibration_controls(self, parent):
        calib_frame = ctk.CTkFrame(parent)
        calib_frame.pack(fill="x", pady=10, padx=10)
        
        ctk.CTkLabel(calib_frame, text="標定控制", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        # 檢測角點按鈕
        detect_btn = ctk.CTkButton(calib_frame, text="檢測所有角點", command=self.detect_all_corners)
        detect_btn.pack(fill="x", pady=2)
        
        # 執行標定按鈕
        calibrate_btn = ctk.CTkButton(calib_frame, text="執行相機標定", command=self.calibrate_camera)
        calibrate_btn.pack(fill="x", pady=2)
        
        # 導出結果按鈕
        export_btn = ctk.CTkButton(calib_frame, text="導出標定結果", command=self.export_calibration)
        export_btn.pack(fill="x", pady=2)
        
        # 進度條
        self.progress_label = ctk.CTkLabel(calib_frame, text="就緒")
        self.progress_label.pack(pady=5)
        
    def setup_results_display(self, parent):
        results_frame = ctk.CTkFrame(parent)
        results_frame.pack(fill="both", expand=True, pady=10, padx=10)
        
        ctk.CTkLabel(results_frame, text="標定結果", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        # 結果文本框
        self.results_text = ctk.CTkTextbox(results_frame, height=200)
        self.results_text.pack(fill="both", expand=True, pady=5)
        
        # 可視化按鈕
        viz_btn = ctk.CTkButton(results_frame, text="可視化精度", command=self.visualize_accuracy)
        viz_btn.pack(fill="x", pady=2)
        
    def setup_display_area(self, parent):
        display_frame = ctk.CTkFrame(parent)
        display_frame.pack(side="right", fill="both", expand=True)
        
        # 顯示區域標題和操作說明
        header_frame = ctk.CTkFrame(display_frame)
        header_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(header_frame, text="圖像顯示區域", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        
        help_text = "滾輪縮放 | 拖拽平移 | 點擊角點查看信息 | 右鍵重置視圖"
        ctk.CTkLabel(header_frame, text=help_text, 
                    font=ctk.CTkFont(size=10), text_color="gray70").pack(side="right")
        
        # 交互式圖像顯示
        image_display_frame = ctk.CTkFrame(display_frame)
        image_display_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.interactive_canvas = InteractiveImageCanvas(image_display_frame, self.display_settings)
        
        # 點位信息框
        info_frame = ctk.CTkFrame(display_frame)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(info_frame, text="點位統計信息", font=ctk.CTkFont(size=14, weight="bold")).pack()
        
        self.points_text = ctk.CTkTextbox(info_frame, height=100)
        self.points_text.pack(fill="x", pady=5)
        
    def import_images(self):
        filetypes = [
            ("圖像文件", "*.jpg *.jpeg *.png *.bmp"),
            ("JPEG文件", "*.jpg *.jpeg"),
            ("PNG文件", "*.png"),
            ("BMP文件", "*.bmp"),
            ("所有文件", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="選擇標定圖像",
            filetypes=filetypes
        )
        
        if files:
            self.image_paths.extend(files)
            self.update_image_list()
            self.update_navigation()
            
            # 如果是第一次導入，自動顯示第一張圖
            if len(self.image_paths) == len(files):
                self.current_image_index = 0
                self.view_current_image()
            
    def clear_images(self):
        self.image_paths.clear()
        self.processed_images.clear()
        self.current_image_index = 0
        self.update_image_list()
        self.update_navigation()
        self.interactive_canvas.set_image(None)
        self.points_text.delete("1.0", "end")
        
    def update_image_list(self):
        # 清除現有列表
        for widget in self.image_listbox.winfo_children():
            widget.destroy()
            
        # 添加圖像項目
        for i, path in enumerate(self.image_paths):
            filename = os.path.basename(path)
            status = "✓" if path in self.processed_images and self.processed_images[path]["success"] else "○"
            
            item_frame = ctk.CTkFrame(self.image_listbox)
            item_frame.pack(fill="x", pady=1)
            
            label = ctk.CTkLabel(item_frame, text=f"{status} {filename}")
            label.pack(side="left", padx=5)
            
            view_btn = ctk.CTkButton(item_frame, text="查看", width=50, 
                                   command=lambda idx=i: self.view_image(idx))
            view_btn.pack(side="right", padx=5)
            
    def update_navigation(self):
        total = len(self.image_paths)
        current = self.current_image_index + 1 if total > 0 else 0
        self.image_info_label.configure(text=f"{current}/{total}")
        
        # 修正: 確保按鈕狀態正確更新
        if total > 0:
            self.prev_btn.configure(state="normal" if self.current_image_index > 0 else "disabled")
            self.next_btn.configure(state="normal" if self.current_image_index < total - 1 else "disabled")
        else:
            self.prev_btn.configure(state="disabled")
            self.next_btn.configure(state="disabled")
        
    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.view_current_image()
            self.update_navigation()
            
    def next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.view_current_image()
            self.update_navigation()
            
    def view_image(self, index):
        if 0 <= index < len(self.image_paths):
            self.current_image_index = index
            self.view_current_image()
            self.update_navigation()
        
    def view_current_image(self):
        if not self.image_paths:
            return
            
        path = self.image_paths[self.current_image_index]
        
        try:
            # 讀取圖像 - 支持中文路徑
            img = self.imread_unicode(path)
            if img is None:
                messagebox.showerror("錯誤", f"無法讀取圖像: {path}")
                return
                
            corners = None
            
            # 檢查是否已處理
            if path in self.processed_images:
                data = self.processed_images[path]
                if data["success"] and data["corners"] is not None:
                    corners = data["corners"]
                    self.display_points_info(corners)
                else:
                    self.points_text.delete("1.0", "end")
                    self.points_text.insert("1.0", "此圖像未檢測到有效角點")
            else:
                self.points_text.delete("1.0", "end")
                self.points_text.insert("1.0", "此圖像尚未處理，請先執行角點檢測")
                
            # 使用交互式canvas顯示圖像
            self.interactive_canvas.set_image(img, corners)
            
        except Exception as e:
            messagebox.showerror("錯誤", f"顯示圖像時發生錯誤: {str(e)}")
            
    def display_points_info(self, corners):
        self.points_text.delete("1.0", "end")
        
        if corners is None:
            self.points_text.insert("1.0", "無角點數據")
            return
            
        info_text = f"檢測到的角點統計:\n"
        info_text += f"總計: {len(corners)} 個點\n"
        
        # 計算統計信息
        x_coords = [corner[0][0] for corner in corners]
        y_coords = [corner[0][1] for corner in corners]
        
        info_text += f"X座標範圍: {min(x_coords):.1f} ~ {max(x_coords):.1f}\n"
        info_text += f"Y座標範圍: {min(y_coords):.1f} ~ {max(y_coords):.1f}\n"
        info_text += f"\n提示: 點擊圖像中的綠色圓點查看詳細座標\n"
        
        self.points_text.insert("1.0", info_text)
        
    def imread_unicode(self, filename):
        """
        支持中文路徑的圖像讀取函數
        """
        try:
            # 使用numpy讀取文件
            with open(filename, 'rb') as f:
                data = f.read()
            
            # 轉換為numpy數組
            nparr = np.frombuffer(data, np.uint8)
            
            # 解碼圖像
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"讀取圖像失敗: {filename}, 錯誤: {e}")
            return None
        
    def detect_all_corners(self):
        if not self.image_paths:
            messagebox.showwarning("警告", "請先導入圖像")
            return
            
        def detect_corners():
            checkerboard = (self.checkerboard_width.get(), self.checkerboard_height.get())
            total = len(self.image_paths)
            success_count = 0
            
            for i, path in enumerate(self.image_paths):
                # 更新進度
                self.progress_label.configure(text=f"處理中... {i+1}/{total}")
                self.root.update()
                
                try:
                    img = self.imread_unicode(path)
                    if img is None:
                        continue
                        
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # 檢測角點
                    ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)
                    
                    if ret and corners.shape[0] == checkerboard[0] * checkerboard[1]:
                        # 精細化角點
                        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                        self.processed_images[path] = {"corners": corners2, "success": True}
                        success_count += 1
                    else:
                        self.processed_images[path] = {"corners": None, "success": False}
                        
                except Exception as e:
                    self.processed_images[path] = {"corners": None, "success": False}
                    
            self.progress_label.configure(text=f"完成! 成功: {success_count}/{total}")
            self.update_image_list()
            
            # 處理完成後，如果當前有圖像則重新顯示
            if self.image_paths:
                self.view_current_image()
            
        # 在新線程中執行
        threading.Thread(target=detect_corners, daemon=True).start()
        
    def calibrate_camera(self):
        if not self.processed_images:
            messagebox.showwarning("警告", "請先檢測角點")
            return
            
        # 收集成功的角點數據
        objpoints = []
        imgpoints = []
        
        checkerboard = (self.checkerboard_width.get(), self.checkerboard_height.get())
        square_size_mm = self.square_size.get() * 10  # 轉換為毫米
        
        # 創建世界坐標
        objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
        objp *= square_size_mm
        
        for path, data in self.processed_images.items():
            if data["success"] and data["corners"] is not None:
                objpoints.append(objp)
                imgpoints.append(data["corners"])
                
        if len(objpoints) < 3:
            messagebox.showwarning("警告", "至少需要3張成功檢測角點的圖像進行標定")
            return
            
        self.progress_label.configure(text="執行標定中...")
        
        def calibrate():
            try:
                # 獲取圖像尺寸
                first_image = self.imread_unicode(self.image_paths[0])
                gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
                image_size = gray.shape[::-1]
                
                # 執行標定
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, image_size, None, None
                )
                
                self.camera_matrix = mtx
                self.dist_coeffs = dist
                self.reprojection_error = ret
                
                # 顯示結果
                self.display_calibration_results(ret, mtx, dist, len(objpoints))
                self.progress_label.configure(text="標定完成!")
                
            except Exception as e:
                messagebox.showerror("錯誤", f"標定失敗: {str(e)}")
                self.progress_label.configure(text="標定失敗")
                
        threading.Thread(target=calibrate, daemon=True).start()
        
    def display_calibration_results(self, reprojection_error, camera_matrix, dist_coeffs, num_images):
        result_text = f"=== 相機標定結果 ===\n\n"
        result_text += f"使用圖像數量: {num_images}\n"
        result_text += f"重投影誤差: {reprojection_error:.4f} 像素\n\n"
        
        result_text += f"相機內參矩陣 (K):\n"
        for row in camera_matrix:
            result_text += f"  [{row[0]:10.2f} {row[1]:10.2f} {row[2]:10.2f}]\n"
            
        result_text += f"\n畸變係數 (D):\n"
        result_text += f"  {dist_coeffs.ravel()}\n\n"
        
        # 提取相機參數
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        result_text += f"焦距:\n"
        result_text += f"  fx = {fx:.2f} 像素\n"
        result_text += f"  fy = {fy:.2f} 像素\n"
        result_text += f"主點:\n"
        result_text += f"  cx = {cx:.2f} 像素\n"
        result_text += f"  cy = {cy:.2f} 像素\n"
        
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", result_text)
        
    def export_calibration(self):
        if self.camera_matrix is None:
            messagebox.showwarning("警告", "請先執行相機標定")
            return
            
        save_dir = filedialog.askdirectory(title="選擇保存目錄")
        if not save_dir:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存numpy格式
            np.save(os.path.join(save_dir, f"camera_matrix_{timestamp}.npy"), self.camera_matrix)
            np.save(os.path.join(save_dir, f"dist_coeffs_{timestamp}.npy"), self.dist_coeffs)
            
            # 保存JSON格式
            calib_data = {
                "camera_matrix": self.camera_matrix.tolist(),
                "distortion_coefficients": self.dist_coeffs.tolist(),
                "reprojection_error": float(self.reprojection_error),
                "checkerboard_size": [self.checkerboard_width.get(), self.checkerboard_height.get()],
                "square_size_cm": self.square_size.get(),
                "timestamp": timestamp,
                "num_images": len([d for d in self.processed_images.values() if d["success"]])
            }
            
            with open(os.path.join(save_dir, f"calibration_data_{timestamp}.json"), 'w') as f:
                json.dump(calib_data, f, indent=2)
                
            messagebox.showinfo("成功", f"標定結果已保存到:\n{save_dir}")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"保存失敗: {str(e)}")
            
    def visualize_accuracy(self):
        if not self.processed_images or self.camera_matrix is None:
            messagebox.showwarning("警告", "請先完成標定")
            return
            
        # 創建新窗口顯示精度可視化
        viz_window = ctk.CTkToplevel(self.root)
        viz_window.title("標定精度可視化")
        viz_window.geometry("800x600")
        
        # 計算重投影誤差
        objpoints = []
        imgpoints = []
        
        checkerboard = (self.checkerboard_width.get(), self.checkerboard_height.get())
        square_size_mm = self.square_size.get() * 10
        
        objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
        objp *= square_size_mm
        
        errors = []
        
        for path, data in self.processed_images.items():
            if data["success"] and data["corners"] is not None:
                objpoints.append(objp)
                imgpoints.append(data["corners"])
                
        # 計算每張圖像的重投影誤差
        for i in range(len(objpoints)):
            # 求解位姿
            _, rvec, tvec = cv2.solvePnP(objpoints[i], imgpoints[i], 
                                        self.camera_matrix, self.dist_coeffs)
            
            # 重投影
            projected_points, _ = cv2.projectPoints(objpoints[i], rvec, tvec, 
                                                   self.camera_matrix, self.dist_coeffs)
            
            # 計算誤差
            error = np.linalg.norm(imgpoints[i].reshape(-1, 2) - 
                                 projected_points.reshape(-1, 2), axis=1)
            errors.append(np.mean(error))
            
        # 創建圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 誤差分布直方圖
        ax1.hist(errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('重投影誤差 (像素)')
        ax1.set_ylabel('圖像數量')
        ax1.set_title('重投影誤差分布')
        ax1.grid(True, alpha=0.3)
        
        # 每張圖像的誤差
        ax2.bar(range(len(errors)), errors, color='lightcoral', alpha=0.7)
        ax2.axhline(y=np.mean(errors), color='red', linestyle='--', 
                   label=f'平均誤差: {np.mean(errors):.3f}')
        ax2.set_xlabel('圖像索引')
        ax2.set_ylabel('平均重投影誤差 (像素)')
        ax2.set_title('各圖像重投影誤差')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 將圖表嵌入到tkinter窗口
        canvas = FigureCanvasTkAgg(fig, viz_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = CameraCalibrationTool()
    app.run()