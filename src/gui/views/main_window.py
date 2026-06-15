"""
src/gui/views/main_window.py
MainWindow — cửa sổ chính với sidebar điều hướng, 4 view, status bar realtime.
closeEvent đảm bảo tắt sạch camera và AI thread.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
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
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Nav items: (icon, label, index) ─────────────────────────────────────────
_NAV_ITEMS = [
    ("🎯", "Điểm Danh",   0),
    ("➕", "Đăng Ký NV",  1),
    ("👥", "Nhân Viên",   2),
    ("📋", "Lịch Sử",     3),
]


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

        # ── Sidebar ────────────────────────────────────────────────────────
        sidebar = self._build_sidebar()
        main_row.addWidget(sidebar)

        # ── Separator line ────────────────────────────────────────────────
        sep = QWidget()
        sep.setFixedWidth(1)
        sep.setStyleSheet("background:#1e293b;")
        main_row.addWidget(sep)

        # ── Stacked pages ─────────────────────────────────────────────────
        self._stack = QStackedWidget()
        main_row.addWidget(self._stack, stretch=1)

        # Khởi tạo và add 4 views
        self._attendance_view   = AttendanceView(self)
        self._enroll_view       = EnrollView(self)
        self._employee_list_view = EmployeeListView(self)
        self._history_view      = HistoryView(self)

        self._stack.addWidget(self._attendance_view)     # index 0
        self._stack.addWidget(self._enroll_view)         # index 1
        self._stack.addWidget(self._employee_list_view)  # index 2
        self._stack.addWidget(self._history_view)        # index 3

        # ── Kết nối Signals cross-view ────────────────────────────────────
        self._enroll_view.enrolled.connect(self._on_enrolled)
        self._employee_list_view.employee_deleted.connect(self._on_employee_deleted)

        # Switch về tab đầu tiên
        self._switch_page(0)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet("background:#0a1628;")

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Logo / Brand header
        brand = QWidget()
        brand.setFixedHeight(72)
        brand.setStyleSheet("background:#0f1f3d; border-bottom:1px solid #1e3a6e;")
        b_layout = QVBoxLayout(brand)
        b_layout.setContentsMargins(16, 12, 16, 12)
        b_layout.setSpacing(2)

        app_name = QLabel("FaceAttend")
        app_name.setFont(QFont("Segoe UI", 14, QFont.Bold))
        app_name.setStyleSheet("color:#60a5fa; letter-spacing:1px;")
        b_layout.addWidget(app_name)

        app_sub = QLabel("Chấm công khuôn mặt")
        app_sub.setStyleSheet("color:#475569; font-size:10px;")
        b_layout.addWidget(app_sub)

        layout.addWidget(brand)
        layout.addSpacing(16)

        # Nav buttons
        for icon, label, idx in _NAV_ITEMS:
            btn = _NavButton(icon, label)
            btn.clicked.connect(lambda _, i=idx: self._switch_page(i))
            layout.addWidget(btn)
            self._nav_buttons.append(btn)

        layout.addStretch()

        # Footer
        footer = QLabel("v1.0.0  •  InsightFace + FAISS")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color:#1e3a6e; font-size:10px; padding:12px;")
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
    def _tick_clock(self):
        # Lấy chuỗi thời gian trước
        time_str = datetime.now().strftime("%H:%M:%S  |  %d/%m/%Y")
        # Nối cái icon đồng hồ vào sau bằng f-string
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

        # 1. ÉP TẮT CAMERA TRƯỚC KHI CHUYỂN TAB ĐỂ CHỐNG CRASH
        if current_idx == 0 and index != 0:
            try:
                self._attendance_view._on_stop()
            except Exception as e:
                logger.warning(f"Lỗi khi dừng AttendanceView: {e}")
                
        elif current_idx == 1 and index != 1:
            try:
                # Sửa lại thành gọi _on_cancel vì file enroll_view của bạn đang dùng hàm này
                self._enroll_view._on_cancel()
                self._enroll_view._reset_form()
            except Exception as e:
                logger.warning(f"Lỗi khi dừng EnrollView: {e}")

        # 2. Chuyển tab
        self._stack.setCurrentIndex(index)

        for i, btn in enumerate(self._nav_buttons):
            btn.set_active(i == index)

        if index == 3:
            self._history_view.refresh()

    # ── Cross-view slots ─────────────────────────────────────────────────────
    @Slot(str)
    def _on_enrolled(self, emp_code: str) -> None:
        logger.info("MainWindow: đăng ký mới '%s'", emp_code)
        self._update_statusbar()
        self._employee_list_view.load_data()   # Cập nhật danh sách NV

    @Slot(str)
    def _on_employee_deleted(self, emp_code: str) -> None:
        logger.info("MainWindow: xóa '%s'", emp_code)
        self._update_statusbar()

    # ── Shutdown sạch ────────────────────────────────────────────────────────
    def closeEvent(self, event) -> None:  # noqa: N802
        """
        Đảm bảo tắt hoàn toàn khi đóng cửa sổ:
        1. Dừng AI Worker (QThread) của AttendanceView.
        2. Dừng SampleCollector (QThread) của EnrollView.
        3. Giải phóng cả 2 CameraStream.
        4. Dừng QTimer đồng hồ.
        """
        logger.info("MainWindow: đang đóng ứng dụng…")

        self._clock_timer.stop()

        # AttendanceView
        try:
            self._attendance_view._on_stop()
        except Exception as exc:
            logger.warning("closeEvent: AttendanceView stop lỗi — %s", exc)

        # EnrollView
        try:
            if (
                self._enroll_view._collector
                and self._enroll_view._collector.isRunning()
            ):
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
        QPushButton {{
            background: transparent;
            color: #64748b;
            border: none;
            border-radius: 0;
            text-align: left;
            padding: 14px 20px;
            font-size: 13px;
            font-weight: 500;
            border-left: 3px solid transparent;
        }}
        QPushButton:hover {{
            background: #0f1f3d;
            color: #94a3b8;
            border-left: 3px solid #1e3a6e;
        }}
    """
    _STYLE_ACTIVE = """
        QPushButton {{
            background: #0f1f3d;
            color: #60a5fa;
            border: none;
            border-radius: 0;
            text-align: left;
            padding: 14px 20px;
            font-size: 13px;
            font-weight: 700;
            border-left: 3px solid #2563eb;
        }}
    """

    def __init__(self, icon: str, label: str) -> None:
        super().__init__(f"  {icon}   {label}")
        self.setFixedHeight(52)
        self.set_active(False)
        self.setCursor(Qt.PointingHandCursor)

    def set_active(self, active: bool) -> None:
        self.setStyleSheet(self._STYLE_ACTIVE if active else self._STYLE_NORMAL)


# ── Status bar separator ─────────────────────────────────────────────────────
def _sb_sep() -> QLabel:
    lbl = QLabel("│")
    lbl.setStyleSheet("color:#1e293b; padding:0;")
    return lbl
