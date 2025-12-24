#!/usr/bin/env python3
"""
vision-calib: Camera Calibration Toolkit

Main entry point for the GUI application.

Usage:
    python main.py
    python -m vision_calib

License: Apache 2.0
"""

import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))


def main():
    """Main entry point."""
    from vision_calib.utils.logging import setup_logging

    # Setup logging
    setup_logging()

    # Import and run GUI
    try:
        from vision_calib.ui.main_window import main as run_gui
        run_gui()
    except ImportError as e:
        print(f"Error: Could not import GUI module: {e}")
        print("Make sure PySide6 is installed: pip install PySide6")
        sys.exit(1)


if __name__ == "__main__":
    main()
