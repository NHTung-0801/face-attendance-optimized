"""
src/gui/views/main_window.py
MainWindow — cửa sổ chính với sidebar điều hướng, 5 view, status bar realtime.
Trang mặc định khi khởi động là HomeView (Dashboard tổng quan).
closeEvent đảm bảo tắt sạch camera, AI thread và animation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.core.employee_manager import EmployeeManager
from src.database.db_manager import DatabaseManager
from src.gui.views.attendance_view import AttendanceView
from src.gui.views.employee_list_view import EmployeeListView
from src.gui.views.enroll_view import EnrollView
from src.gui.views.history_view import HistoryView
from src.gui.views.home_view import HomeView
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Nav items: (icon, label, index) ─────────────────────────────────────────
_NAV_ITEMS = [
    ("🏠", "Trang Chủ",  0),
    ("🎯", "Điểm Danh",  1),
    ("➕", "Đăng Ký NV", 2),
    ("👥", "Nhân Viên",  3),
    ("📋", "Lịch Sử",    4),
]

_HOME_INDEX       = 0
_ATTENDANCE_INDEX = 1
_ENROLL_INDEX     = 2
_EMPLOYEE_INDEX   = 3
_HISTORY_INDEX    = 4


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._db      = DatabaseManager.instance()
        self._manager = EmployeeManager.instance()
        self._nav_buttons: list[QPushButton] = []

        self.setWindowTitle("FaceAttend — Hệ thống chấm công khuôn mặt")
        self.setMinimumSize(1280, 760)

        self._build_ui()
        self._setup_statusbar()
        self._setup_clock()
        self._update_statusbar()

    # ── Layout tổng thể ──────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_row = QHBoxLayout(central)
        main_row.setContentsMargins(0, 0, 0, 0)
        main_row.setSpacing(0)

        sidebar = self._build_sidebar()
        main_row.addWidget(sidebar)

        self._stack = QStackedWidget()
        main_row.addWidget(self._stack, stretch=1)

        # ── Khởi tạo 5 views ───────────────────────────────────────────────
        self._home_view          = HomeView(self)
        self._attendance_view    = AttendanceView(self)
        self._enroll_view        = EnrollView(self)
        self._employee_list_view = EmployeeListView(self)
        self._history_view       = HistoryView(self)

        self._stack.addWidget(self._home_view)           # index 0
        self._stack.addWidget(self._attendance_view)      # index 1
        self._stack.addWidget(self._enroll_view)          # index 2
        self._stack.addWidget(self._employee_list_view)   # index 3
        self._stack.addWidget(self._history_view)         # index 4

        # ── Kết nối Signals cross-view ────────────────────────────────────
        self._home_view.navigate_requested.connect(self._switch_page)
        self._enroll_view.enrolled.connect(self._on_enrolled)
        self._employee_list_view.employee_deleted.connect(self._on_employee_deleted)

        # Mặc định mở Trang Chủ
        self._switch_page(_HOME_INDEX)

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setFixedWidth(240)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #0b1326;
                border-right: 2px solid #2ca0ba;
            }
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Logo / Brand header ─────────────────────────────────────────
        brand = QFrame()
        brand.setFixedHeight(72)
        brand.setStyleSheet("border-bottom: 2px solid #2ca0ba; background: transparent;")

        b_layout = QVBoxLayout(brand)
        b_layout.setContentsMargins(24, 12, 20, 12)
        b_layout.setSpacing(2)

        app_name = QLabel("FaceAttend")
        app_name.setStyleSheet(
            "color:#4cd7f6; font-size:20px; font-weight:800; letter-spacing:1px; border:none;"
        )
        b_layout.addWidget(app_name)

        app_sub = QLabel("SecureFace AI Engine")
        app_sub.setStyleSheet("color:#8c909f; font-size:11px; border:none;")
        b_layout.addWidget(app_sub)

        layout.addWidget(brand)
        layout.addSpacing(24)

        # ── Nav buttons ───────────────────────────────────────────────────
        nav_container = QVBoxLayout()
        nav_container.setContentsMargins(16, 0, 16, 0)
        nav_container.setSpacing(12)

        for icon, label, idx in _NAV_ITEMS:
            btn = _NavButton(icon, label)
            btn.clicked.connect(lambda _, i=idx: self._switch_page(i))
            nav_container.addWidget(btn)
            self._nav_buttons.append(btn)

        layout.addLayout(nav_container)
        layout.addStretch()

        footer = QLabel("v1.0.0  •  ONNX Engine")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color:#2ca0ba; font-size:11px; padding:16px; border:none;")
        layout.addWidget(footer)

        return sidebar

    # ── Status bar ───────────────────────────────────────────────────────────
    def _setup_statusbar(self) -> None:
        bar = QStatusBar()
        bar.setStyleSheet("""
            QStatusBar {
                background:#0a1628;
                color:#475569;
                border-top:1px solid #1e293b;
                font-size:11px;
            }
            QStatusBar::item { border:none; }
        """)
        self.setStatusBar(bar)

        self._sb_employees = QLabel()
        self._sb_session   = QLabel()
        self._sb_clock     = QLabel()

        for lbl in (self._sb_employees, self._sb_session, self._sb_clock):
            lbl.setStyleSheet("color:#64748b; padding:0 12px;")

        bar.addWidget(self._sb_employees)
        bar.addWidget(_sb_sep())
        bar.addWidget(self._sb_session)
        bar.addPermanentWidget(self._sb_clock)

    def _setup_clock(self) -> None:
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)
        self._tick_clock()

    @Slot()
    def _tick_clock(self) -> None:
        time_str = datetime.now().strftime("%H:%M:%S  |  %d/%m/%Y")
        self._sb_clock.setText(f"🕐  {time_str}")

    def _update_statusbar(self) -> None:
        try:
            nv_count   = self._manager.get_registered_count()
            active_ses = self._db.get_active_session()
            ses_text   = (
                f"📌 Ca: {active_ses.session_name}" if active_ses
                else "Ca: không có ca nào đang mở"
            )
            self._sb_employees.setText(f"👤 FAISS: {nv_count} nhân viên")
            self._sb_session.setText(ses_text)
        except Exception:
            pass

    # ── Navigation ───────────────────────────────────────────────────────────
    def _switch_page(self, index: int) -> None:
        current_idx = self._stack.currentIndex()
        if current_idx == index:
            return   # Đã ở đúng tab, tránh restart camera/animation vô ích

        # ── Dọn dẹp trang đang rời ──────────────────────────────────────
        if current_idx == _HOME_INDEX:
            try:
                self._home_view.stop_animations()
            except Exception as exc:
                logger.warning("Lỗi khi dừng HomeView animation: %s", exc)

        elif current_idx == _ATTENDANCE_INDEX:
            try:
                self._attendance_view._on_stop()
            except Exception as exc:
                logger.warning("Lỗi khi dừng AttendanceView: %s", exc)

        elif current_idx == _ENROLL_INDEX:
            try:
                self._enroll_view._on_cancel()
                self._enroll_view._reset_form()
            except Exception as exc:
                logger.warning("Lỗi khi dừng EnrollView: %s", exc)

        # ── Chuyển tab ────────────────────────────────────────────────────
        self._stack.setCurrentIndex(index)

        for i, btn in enumerate(self._nav_buttons):
            btn.set_active(i == index)

        # ── Side-effect khi vào trang mới ───────────────────────────────
        if index == _HOME_INDEX:
            self._home_view.resume_animations()
        elif index == _HISTORY_INDEX:
            self._history_view.refresh()

        self._update_statusbar()

    # ── Cross-view slots ─────────────────────────────────────────────────────
    @Slot(str)
    def _on_enrolled(self, emp_code: str) -> None:
        logger.info("MainWindow: đăng ký mới '%s'", emp_code)
        self._update_statusbar()
        self._employee_list_view.load_data()

    @Slot(str)
    def _on_employee_deleted(self, emp_code: str) -> None:
        logger.info("MainWindow: xóa '%s'", emp_code)
        self._update_statusbar()

    # ── Shutdown sạch ────────────────────────────────────────────────────────
    def closeEvent(self, event) -> None:  # noqa: N802
        """
        Đảm bảo tắt hoàn toàn khi đóng cửa sổ:
        1. Dừng animation + timer của HomeView.
        2. Dừng AI Worker (QThread) của AttendanceView.
        3. Dừng SampleCollector (QThread) của EnrollView.
        4. Giải phóng cả 2 CameraStream.
        5. Dừng QTimer đồng hồ chính.
        """
        logger.info("MainWindow: đang đóng ứng dụng…")

        self._clock_timer.stop()

        try:
            self._home_view.stop_animations()
        except Exception as exc:
            logger.warning("closeEvent: HomeView stop lỗi — %s", exc)

        try:
            self._attendance_view._on_stop()
        except Exception as exc:
            logger.warning("closeEvent: AttendanceView stop lỗi — %s", exc)

        try:
            if self._enroll_view._collector and self._enroll_view._collector.isRunning():
                self._enroll_view._collector.stop()
            if self._enroll_view._camera_started:
                self._enroll_view._camera.stop()
        except Exception as exc:
            logger.warning("closeEvent: EnrollView stop lỗi — %s", exc)

        logger.info("MainWindow: đã dọn dẹp xong.")
        super().closeEvent(event)


# ── Custom nav button ────────────────────────────────────────────────────────
class _NavButton(QPushButton):
    _STYLE_NORMAL = """
        QPushButton {
            background-color: transparent;
            color: #8c909f;
            border: 1px solid transparent;
            border-radius: 12px;
            text-align: left;
            padding: 10px 16px;
            font-size: 14px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #161f2e;
            color: #4cd7f6;
            border: 1px solid #1d6475;
        }
    """
    _STYLE_ACTIVE = """
        QPushButton {
            background-color: #161f2e;
            color: #4cd7f6;
            border: 1px solid #2ca0ba;
            border-radius: 12px;
            text-align: left;
            padding: 10px 16px;
            font-size: 14px;
            font-weight: 700;
        }
    """

    def __init__(self, icon: str, label: str) -> None:
        super().__init__(f"   {icon}    {label}")
        self.setFixedHeight(48)
        self.setCursor(Qt.PointingHandCursor)
        self.set_active(False)

    def set_active(self, active: bool) -> None:
        self.setStyleSheet(self._STYLE_ACTIVE if active else self._STYLE_NORMAL)


# ── Status bar separator ─────────────────────────────────────────────────────
def _sb_sep() -> QLabel:
    lbl = QLabel("│")
    lbl.setStyleSheet("color:#1e293b; padding:0;")
    return lbl
