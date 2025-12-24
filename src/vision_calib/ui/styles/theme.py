"""
Material Design theme system for vision-calib.

Implements Google's Material Design 3 color scheme with light/dark modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


class Theme(Enum):
    """Available themes."""
    LIGHT = "light"
    DARK = "dark"


@dataclass
class ThemeColors:
    """Material Design 3 color palette."""
    # Primary
    primary: str
    on_primary: str
    primary_container: str
    on_primary_container: str

    # Secondary
    secondary: str
    on_secondary: str
    secondary_container: str
    on_secondary_container: str

    # Surface
    surface: str
    on_surface: str
    surface_variant: str
    on_surface_variant: str
    surface_container: str
    surface_container_high: str
    surface_container_low: str

    # Background
    background: str
    on_background: str

    # Outline
    outline: str
    outline_variant: str

    # Error
    error: str
    on_error: str
    error_container: str

    # Success
    success: str
    on_success: str

    # Shadow
    shadow: str


# Material Design 3 - Light Theme (Google Blue)
LIGHT_COLORS = ThemeColors(
    # Primary - Google Blue
    primary="#1a73e8",
    on_primary="#ffffff",
    primary_container="#d3e3fd",
    on_primary_container="#041e49",

    # Secondary - Teal
    secondary="#5f6368",
    on_secondary="#ffffff",
    secondary_container="#e8eaed",
    on_secondary_container="#1f1f1f",

    # Surface
    surface="#ffffff",
    on_surface="#1f1f1f",
    surface_variant="#f1f3f4",
    on_surface_variant="#444746",
    surface_container="#f8f9fa",
    surface_container_high="#f1f3f4",
    surface_container_low="#ffffff",

    # Background
    background="#ffffff",
    on_background="#1f1f1f",

    # Outline
    outline="#dadce0",
    outline_variant="#c4c7c5",

    # Error
    error="#d93025",
    on_error="#ffffff",
    error_container="#fce8e6",

    # Success
    success="#1e8e3e",
    on_success="#ffffff",

    # Shadow
    shadow="rgba(0, 0, 0, 0.15)",
)

# Material Design 3 - Dark Theme
DARK_COLORS = ThemeColors(
    # Primary - Google Blue (lighter for dark mode)
    primary="#8ab4f8",
    on_primary="#062e6f",
    primary_container="#1a73e8",
    on_primary_container="#d3e3fd",

    # Secondary
    secondary="#9aa0a6",
    on_secondary="#1f1f1f",
    secondary_container="#3c4043",
    on_secondary_container="#e8eaed",

    # Surface
    surface="#1f1f1f",
    on_surface="#e8eaed",
    surface_variant="#2d2d2d",
    on_surface_variant="#c4c7c5",
    surface_container="#282828",
    surface_container_high="#323232",
    surface_container_low="#1a1a1a",

    # Background
    background="#1f1f1f",
    on_background="#e8eaed",

    # Outline
    outline="#5f6368",
    outline_variant="#3c4043",

    # Error
    error="#f28b82",
    on_error="#1f1f1f",
    error_container="#8c1d18",

    # Success
    success="#81c995",
    on_success="#1f1f1f",

    # Shadow
    shadow="rgba(0, 0, 0, 0.4)",
)


def get_stylesheet(colors: ThemeColors) -> str:
    """Generate QSS stylesheet from theme colors."""
    return f"""
    /* ===== Global ===== */
    QMainWindow, QWidget {{
        background-color: {colors.background};
        color: {colors.on_background};
        font-family: "Microsoft JhengHei UI", "Segoe UI", "Roboto", sans-serif;
        font-size: 14px;
    }}

    /* ===== Menu Bar ===== */
    QMenuBar {{
        background-color: {colors.surface};
        color: {colors.on_surface};
        border-bottom: 1px solid {colors.outline};
        padding: 4px 0;
        font-size: 14px;
    }}

    QMenuBar::item {{
        padding: 8px 16px;
        border-radius: 4px;
        margin: 2px 4px;
    }}

    QMenuBar::item:selected {{
        background-color: {colors.surface_variant};
    }}

    QMenu {{
        background-color: {colors.surface_container};
        color: {colors.on_surface};
        border: 1px solid {colors.outline};
        border-radius: 8px;
        padding: 8px 0;
    }}

    QMenu::item {{
        padding: 10px 24px;
        margin: 2px 8px;
        border-radius: 4px;
    }}

    QMenu::item:selected {{
        background-color: {colors.primary_container};
        color: {colors.on_primary_container};
    }}

    /* ===== Tool Bar ===== */
    QToolBar {{
        background-color: {colors.surface};
        border-bottom: 1px solid {colors.outline};
        padding: 8px;
        spacing: 8px;
    }}

    QToolBar QToolButton {{
        background-color: transparent;
        color: {colors.on_surface};
        border: none;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 14px;
        font-weight: 500;
    }}

    QToolBar QToolButton:hover {{
        background-color: {colors.surface_variant};
    }}

    QToolBar QToolButton:pressed {{
        background-color: {colors.primary_container};
    }}

    /* ===== Status Bar ===== */
    QStatusBar {{
        background-color: {colors.surface};
        color: {colors.on_surface_variant};
        border-top: 1px solid {colors.outline};
        padding: 8px 16px;
        font-size: 13px;
    }}

    /* ===== Buttons ===== */
    QPushButton {{
        background-color: {colors.primary};
        color: {colors.on_primary};
        border: none;
        border-radius: 20px;
        padding: 12px 24px;
        font-size: 14px;
        font-weight: 500;
        min-height: 20px;
    }}

    QPushButton:hover {{
        background-color: {colors.primary};
        opacity: 0.92;
    }}

    QPushButton:pressed {{
        background-color: {colors.primary_container};
        color: {colors.on_primary_container};
    }}

    QPushButton:disabled {{
        background-color: {colors.surface_variant};
        color: {colors.on_surface_variant};
    }}

    /* Secondary Button */
    QPushButton[secondary="true"] {{
        background-color: transparent;
        color: {colors.primary};
        border: 1px solid {colors.outline};
    }}

    QPushButton[secondary="true"]:hover {{
        background-color: {colors.primary_container};
    }}

    /* ===== Group Box ===== */
    QGroupBox {{
        background-color: {colors.surface_container};
        border: 1px solid {colors.outline};
        border-radius: 12px;
        margin-top: 16px;
        padding: 20px 16px 16px 16px;
        font-size: 14px;
        font-weight: 500;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 16px;
        top: 4px;
        color: {colors.on_surface};
        background-color: {colors.surface_container};
        padding: 4px 8px;
        font-size: 13px;
        font-weight: 600;
    }}

    /* ===== Spin Box ===== */
    QSpinBox, QDoubleSpinBox {{
        background-color: {colors.surface_container};
        color: {colors.on_surface};
        border: 1px solid {colors.outline};
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 14px;
        min-height: 20px;
    }}

    QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 2px solid {colors.primary};
    }}

    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        background-color: transparent;
        border: none;
        width: 24px;
    }}

    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-bottom: 6px solid {colors.on_surface_variant};
    }}

    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {colors.on_surface_variant};
    }}

    /* ===== List Widget ===== */
    QListWidget {{
        background-color: {colors.surface_container};
        color: {colors.on_surface};
        border: 1px solid {colors.outline};
        border-radius: 8px;
        padding: 8px;
        font-size: 14px;
        outline: none;
    }}

    QListWidget::item {{
        padding: 10px 12px;
        border-radius: 6px;
        margin: 2px 0;
    }}

    QListWidget::item:selected {{
        background-color: {colors.primary_container};
        color: {colors.on_primary_container};
    }}

    QListWidget::item:hover {{
        background-color: {colors.surface_variant};
    }}

    /* ===== Tab Widget ===== */
    QTabWidget::pane {{
        background-color: {colors.surface};
        border: 1px solid {colors.outline};
        border-radius: 12px;
        padding: 8px;
    }}

    QTabBar::tab {{
        background-color: transparent;
        color: {colors.on_surface_variant};
        border: none;
        padding: 14px 24px;
        font-size: 14px;
        font-weight: 500;
        margin-right: 4px;
    }}

    QTabBar::tab:selected {{
        color: {colors.primary};
        border-bottom: 3px solid {colors.primary};
    }}

    QTabBar::tab:hover:!selected {{
        background-color: {colors.surface_variant};
        border-radius: 8px 8px 0 0;
    }}

    /* ===== Text Edit ===== */
    QTextEdit {{
        background-color: {colors.surface_container};
        color: {colors.on_surface};
        border: 1px solid {colors.outline};
        border-radius: 12px;
        padding: 16px;
        font-size: 14px;
        line-height: 1.6;
    }}

    QTextEdit:focus {{
        border: 2px solid {colors.primary};
    }}

    /* ===== Progress Bar ===== */
    QProgressBar {{
        background-color: {colors.surface_variant};
        border: none;
        border-radius: 4px;
        height: 8px;
        text-align: center;
    }}

    QProgressBar::chunk {{
        background-color: {colors.primary};
        border-radius: 4px;
    }}

    /* ===== Splitter ===== */
    QSplitter::handle {{
        background-color: {colors.outline};
        width: 1px;
        margin: 0 8px;
    }}

    QSplitter::handle:hover {{
        background-color: {colors.primary};
    }}

    /* ===== Scroll Area ===== */
    QScrollArea {{
        background-color: transparent;
        border: none;
    }}

    QScrollArea > QWidget > QWidget {{
        background-color: transparent;
    }}

    /* ===== Scroll Bar ===== */
    QScrollBar:vertical {{
        background-color: transparent;
        width: 12px;
        margin: 4px;
    }}

    QScrollBar::handle:vertical {{
        background-color: {colors.outline_variant};
        border-radius: 4px;
        min-height: 40px;
    }}

    QScrollBar::handle:vertical:hover {{
        background-color: {colors.outline};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}

    QScrollBar:horizontal {{
        background-color: transparent;
        height: 12px;
        margin: 4px;
    }}

    QScrollBar::handle:horizontal {{
        background-color: {colors.outline_variant};
        border-radius: 4px;
        min-width: 40px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background-color: {colors.outline};
    }}

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}

    /* ===== Message Box ===== */
    QMessageBox {{
        background-color: {colors.surface};
    }}

    QMessageBox QLabel {{
        color: {colors.on_surface};
        font-size: 14px;
        padding: 8px;
    }}

    /* ===== File Dialog ===== */
    QFileDialog {{
        background-color: {colors.surface};
    }}

    /* ===== Labels ===== */
    QLabel {{
        color: {colors.on_surface};
        font-size: 14px;
    }}

    QLabel[heading="true"] {{
        font-size: 20px;
        font-weight: 600;
        color: {colors.on_surface};
    }}

    QLabel[subheading="true"] {{
        font-size: 16px;
        font-weight: 500;
        color: {colors.on_surface_variant};
    }}

    /* ===== Form Layout ===== */
    QFormLayout {{
        spacing: 12px;
    }}

    /* ===== Tool Tip ===== */
    QToolTip {{
        background-color: {colors.surface_container_high};
        color: {colors.on_surface};
        border: 1px solid {colors.outline};
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 13px;
    }}

    /* ===== Combo Box ===== */
    QComboBox {{
        background-color: {colors.surface_container};
        color: {colors.on_surface};
        border: 1px solid {colors.outline};
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 14px;
        min-height: 20px;
    }}

    QComboBox:focus {{
        border: 2px solid {colors.primary};
    }}

    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}

    QComboBox::down-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {colors.on_surface_variant};
    }}

    QComboBox QAbstractItemView {{
        background-color: {colors.surface_container};
        color: {colors.on_surface};
        border: 1px solid {colors.outline};
        border-radius: 8px;
        padding: 8px;
        selection-background-color: {colors.primary_container};
        selection-color: {colors.on_primary_container};
        font-size: 14px;
    }}

    /* ===== Graphics View ===== */
    QGraphicsView {{
        background-color: {colors.surface_container};
        border: 1px solid {colors.outline};
        border-radius: 8px;
    }}
    """


class ThemeManager(QObject):
    """
    Manages application theme switching.

    Usage:
        manager = ThemeManager()
        manager.set_theme(Theme.DARK)
        manager.theme_changed.connect(on_theme_changed)
    """

    theme_changed = Signal(Theme)

    _instance: ThemeManager | None = None

    def __new__(cls) -> ThemeManager:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self._current_theme = Theme.LIGHT
        self._initialized = True

    @property
    def current_theme(self) -> Theme:
        """Get current theme."""
        return self._current_theme

    @property
    def colors(self) -> ThemeColors:
        """Get current theme colors."""
        return LIGHT_COLORS if self._current_theme == Theme.LIGHT else DARK_COLORS

    @property
    def is_dark(self) -> bool:
        """Check if dark theme is active."""
        return self._current_theme == Theme.DARK

    def set_theme(self, theme: Theme) -> None:
        """Set application theme."""
        if theme == self._current_theme:
            return

        self._current_theme = theme
        colors = LIGHT_COLORS if theme == Theme.LIGHT else DARK_COLORS

        app = QApplication.instance()
        if app:
            app.setStyleSheet(get_stylesheet(colors))

        self.theme_changed.emit(theme)

    def toggle_theme(self) -> None:
        """Toggle between light and dark themes."""
        new_theme = Theme.DARK if self._current_theme == Theme.LIGHT else Theme.LIGHT
        self.set_theme(new_theme)

    def apply_current_theme(self) -> None:
        """Apply current theme to application."""
        colors = LIGHT_COLORS if self._current_theme == Theme.LIGHT else DARK_COLORS
        app = QApplication.instance()
        if app:
            app.setStyleSheet(get_stylesheet(colors))
