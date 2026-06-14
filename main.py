"""
main.py — Entry point duy nhất của ứng dụng FaceAttend.
Chạy: python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> None:
    """
    Chạy trước khi import bất kỳ module Qt hoặc AI nào.
    Đảm bảo thư mục data/ tồn tại và môi trường hợp lệ.
    """
    # ── 1. Thêm root vào sys.path để import src.* hoạt động ─────────────
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # ── 2. Tạo các thư mục cần thiết nếu chưa có ─────────────────────────
    #    (config.py cũng tự tạo, nhưng bootstrap tạo trước để an toàn)
    dirs = [
        root / "data" / "models",
        root / "data" / "database",
        root / "data" / "faiss_index",
        root / "data" / "exports",
        root / "logs",
        root / "assets" / "icons",
        root / "assets" / "images",
        root / "assets" / "sounds",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # ── 3. Kiểm tra model anti-spoofing ──────────────────────────────────
    model_path = root / "data" / "models" / "best.onnx"
    if not model_path.exists():
        print(
            "[WARN] Chưa có model anti-spoofing tại data/models/best.onnx\n"
            "       Chạy tools/export_onnx.py để xuất model trước.\n"
            "       Tính năng chống giả mạo sẽ bị vô hiệu."
        )

    # ── 4. Biến môi trường tối ưu cho ONNX + Qt trên Windows ────────────
    import os
    os.environ.setdefault("OMP_NUM_THREADS",        "4")   # Giới hạn thread ONNX
    os.environ.setdefault("MKL_NUM_THREADS",        "4")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")  # HiDPI


def _load_qss(app) -> None:
    """Load QSS từ file nếu có, fallback về inline dark theme."""
    from pathlib import Path
    qss_path = Path(__file__).parent / "assets" / "styles" / "dark.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
        return

    # ── Inline dark theme (slate palette) ────────────────────────────────
    app.setStyleSheet("""
        /* ── Global ─────────────────────────────────────── */
        QWidget {
            background: #0f172a;
            color: #e2e8f0;
            font-family: "Segoe UI", "Inter", sans-serif;
            font-size: 13px;
        }

        /* ── Scrollbars ──────────────────────────────────── */
        QScrollBar:vertical {
            background: #0f172a; width: 8px; margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #334155; border-radius: 4px; min-height: 24px;
        }
        QScrollBar::handle:vertical:hover { background: #475569; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

        QScrollBar:horizontal {
            background: #0f172a; height: 8px; margin: 0;
        }
        QScrollBar::handle:horizontal {
            background: #334155; border-radius: 4px; min-width: 24px;
        }
        QScrollBar::handle:horizontal:hover { background: #475569; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

        /* ── QMessageBox ────────────────────────────────── */
        QMessageBox {
            background: #1e293b;
        }
        QMessageBox QLabel {
            color: #e2e8f0; font-size: 13px;
        }
        QMessageBox QPushButton {
            background: #2563eb; color: #fff;
            border: none; border-radius: 6px;
            padding: 6px 20px; font-weight: 600;
            min-width: 80px;
        }
        QMessageBox QPushButton:hover  { background: #1d4ed8; }
        QMessageBox QPushButton:default { background: #2563eb; }

        /* ── QDialog ────────────────────────────────────── */
        QDialog { background: #1e293b; }

        /* ── QToolTip ───────────────────────────────────── */
        QToolTip {
            background: #1e293b; color: #cbd5e1;
            border: 1px solid #334155; border-radius: 4px;
            padding: 4px 8px; font-size: 12px;
        }

        /* ── QComboBox popup ────────────────────────────── */
        QAbstractItemView {
            background: #1e293b; color: #e2e8f0;
            selection-background-color: #2563eb;
            border: 1px solid #334155;
            outline: none;
        }

        /* ── Calendar popup ─────────────────────────────── */
        QCalendarWidget QWidget { background: #1e293b; color: #e2e8f0; }
        QCalendarWidget QAbstractItemView:enabled {
            background: #1e293b; color: #e2e8f0;
            selection-background-color: #2563eb;
        }
        QCalendarWidget QToolButton {
            background: transparent; color: #94a3b8;
            border: none; font-size: 12px;
        }
        QCalendarWidget QToolButton:hover { color: #60a5fa; }

        /* ── QFileDialog ────────────────────────────────── */
        QFileDialog { background: #1e293b; }
        QFileDialog QListView, QFileDialog QTreeView {
            background: #0f172a; color: #e2e8f0;
        }
    """)


def main() -> None:
    _bootstrap()

    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont
    from PySide6.QtCore import Qt

    app = QApplication(sys.argv)
    app.setApplicationName("FaceAttend")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("FaceAttend")

    # Fusion style là base tốt nhất để override bằng QSS tối
    app.setStyle("Fusion")

    # Font mặc định
    default_font = QFont("Segoe UI", 11)
    app.setFont(default_font)

    # Load QSS dark theme
    _load_qss(app)

    # Import sau bootstrap để đảm bảo sys.path đúng
    from src.gui.views.main_window import MainWindow

    window = MainWindow()
    window.showMaximized()

    logger.info("FaceAttend khởi động.")
    exit_code = app.exec()
    logger.info("FaceAttend thoát với code %d.", exit_code)
    sys.exit(exit_code)


# ── Logger cấp module (dùng trước khi src.utils.logger available) ────────────
import logging
logger = logging.getLogger("main")

if __name__ == "__main__":
    main()
