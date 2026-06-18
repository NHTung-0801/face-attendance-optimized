"""
src/gui/views/attendance_view.py
AttendanceView — màn hình điểm danh realtime.
Phong cách đồng bộ với HomeView / MainWindow: SecureFace AI Engine (cyberpunk).

Token màu (đồng bộ toàn app):
    Background sâu    : #0b1326 / #060e20
    Surface            : #161f2e / #111828
    Border / glow      : #2ca0ba / #1d6475
    Accent chính (cyan): #4cd7f6
    Accent xanh lá     : #4edea3
    Accent đỏ (danger) : #f87171 / #dc2626
    Text chính         : #f8fafc
    Text phụ           : #8c909f
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.camera_stream import CameraStream
from src.database.db_manager import DatabaseManager
from src.gui.threads.ai_worker import AIResult, AIWorker
from src.utils.config import RECOGNITION_CONFIRM_COUNT
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CONFIRM_NEEDED = RECOGNITION_CONFIRM_COUNT   # Số frame liên tiếp để xác nhận


class AttendanceView(QWidget):
    """
    Layout (responsive, không còn camera đè lên panel):

    ┌──────────────────────────────────────────────────────────┐
    │  Header: 🎯 Điểm Danh Realtime         [trạng thái ca]    │
    ├───────────────────────────────────┬────────────────────────┤
    │                                     │  Status card           │
    │   Camera feed (co giãn theo cửa sổ)│  ─────────────         │
    │   khung viền cyan, bo góc          │  Bảng lịch sử ca       │
    │                                     │  ─────────────         │
    │   [● tắt/đang chạy]                │  [Bắt đầu] [Kết thúc]  │
    └───────────────────────────────────┴────────────────────────┘

    Cột trái dùng stretch=3, cột phải stretch=2, panel phải có
    minimumWidth cố định để không bị bóp méo khi camera video lớn.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._db      = DatabaseManager.instance()
        self._camera  = CameraStream()
        self._worker: Optional[AIWorker] = None

        self._confirm_counts: dict[str, int] = {}
        self._active_session_id: Optional[int] = None

        self._build_ui()

    # ── Xây dựng UI ─────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        self.setStyleSheet("background-color: #0b1326;")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(28, 24, 28, 28)
        body_layout.setSpacing(24)

        # Cột trái (camera) chiếm nhiều không gian hơn, co giãn được
        body_layout.addLayout(self._build_camera_column(), stretch=3)
        # Cột phải (panel) giữ độ rộng tối thiểu cố định, không bị bóp
        body_layout.addWidget(self._build_panel_column(), stretch=2)

        root.addWidget(body, stretch=1)

    # ── Header ───────────────────────────────────────────────────────────────
    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setFixedHeight(76)
        header.setStyleSheet("""
            QFrame {
                background-color: #0b1326;
                border-bottom: 2px solid #2ca0ba;
            }
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(32, 0, 32, 0)

        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        title_box.setAlignment(Qt.AlignVCenter)

        title = QLabel("🎯  Điểm Danh Realtime")
        title.setStyleSheet(
            "font-size: 22px; font-weight: 700; color: #f8fafc; letter-spacing: 0.3px;"
        )
        title_box.addWidget(title)

        sub_row = QHBoxLayout()
        sub_row.setSpacing(6)
        self._header_dot = QLabel("●")
        self._header_dot.setStyleSheet("color: #f87171; font-weight: bold; font-size: 11px;")
        self._header_sub = QLabel("Chưa bắt đầu ca làm việc")
        self._header_sub.setStyleSheet("color: #8c909f; font-size: 12px;")
        sub_row.addWidget(self._header_dot)
        sub_row.addWidget(self._header_sub)
        sub_row.addStretch()
        title_box.addLayout(sub_row)

        layout.addLayout(title_box)
        layout.addStretch()

        # Badge đếm số người đã điểm danh trong ca
        self._header_count_badge = QLabel("0 lượt")
        self._header_count_badge.setAlignment(Qt.AlignCenter)
        self._header_count_badge.setFixedHeight(30)
        self._header_count_badge.setStyleSheet("""
            background-color: #0e2a32; color: #4cd7f6;
            border: 1px solid #1d6475; border-radius: 15px;
            font-size: 12px; font-weight: 700; padding: 0 16px;
        """)
        layout.addWidget(self._header_count_badge)

        return header

    # ── Cột trái: camera ─────────────────────────────────────────────────────
    def _build_camera_column(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(14)

        # Khung camera — frame riêng biệt, bo góc, viền cyan, KHÔNG fixed size
        cam_frame = QFrame()
        cam_frame.setStyleSheet("""
            QFrame {
                background-color: #060e20;
                border: 2px solid #2ca0ba;
                border-radius: 16px;
            }
        """)
        cam_frame.setMinimumSize(480, 360)
        cam_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        cam_inner = QVBoxLayout(cam_frame)
        cam_inner.setContentsMargins(10, 10, 10, 10)

        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setMinimumSize(1, 1)        # Cho phép co nhỏ khi cửa sổ nhỏ
        self._video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video_label.setStyleSheet(
            "background-color: #060e20; border-radius: 10px; color: #475569;"
        )
        self._video_label.setText("📷\n\nCamera chưa bật")
        self._video_label.setFont(QFont("Segoe UI", 13))
        cam_inner.addWidget(self._video_label)

        col.addWidget(cam_frame, stretch=1)

        # Status bar nhỏ dưới camera (tách biệt hẳn, không đè)
        status_row = QFrame()
        status_row.setFixedHeight(44)
        status_row.setStyleSheet("""
            QFrame {
                background-color: #161f2e;
                border: 1px solid #1d6475;
                border-radius: 10px;
            }
        """)
        status_row_layout = QHBoxLayout(status_row)
        status_row_layout.setContentsMargins(16, 0, 16, 0)

        self._cam_status = QLabel("●  Camera: tắt")
        self._cam_status.setStyleSheet("color:#f87171; font-size:12px; font-weight:600;")
        status_row_layout.addWidget(self._cam_status)
        status_row_layout.addStretch()

        self._fps_label = QLabel("")
        self._fps_label.setStyleSheet("color:#4cd7f6; font-size:11px; font-family: Consolas, monospace;")
        status_row_layout.addWidget(self._fps_label)

        col.addWidget(status_row)
        return col

    # ── Cột phải: panel điều khiển ───────────────────────────────────────────
    def _build_panel_column(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(340)
        panel.setMaximumWidth(420)
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(18)

        # ── Status card (trạng thái nhận diện realtime) ───────────────────
        status_card = QFrame()
        status_card.setStyleSheet("""
            QFrame {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1, stop:0 #060e20, stop:1 #0f1e3c
                );
                border: 2px solid #2ca0ba;
                border-radius: 16px;
            }
        """)
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(20, 18, 20, 18)
        status_layout.setSpacing(8)

        status_title = QLabel("TRẠNG THÁI NHẬN DIỆN")
        status_title.setStyleSheet(
            "color: #4cd7f6; font-size: 11px; font-weight: 700; letter-spacing: 2px;"
        )
        status_layout.addWidget(status_title)

        self._status_label = QLabel("Chờ khuôn mặt…")
        self._status_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #f8fafc;")
        self._status_label.setMinimumHeight(44)
        status_layout.addWidget(self._status_label)

        col.addWidget(status_card)

        # ── Bảng lịch sử điểm danh ──────────────────────────────────────────
        table_frame = QFrame()
        table_frame.setStyleSheet("""
            QFrame {
                background-color: #161f2e;
                border: 1px solid #1d6475;
                border-radius: 16px;
            }
        """)
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(16, 16, 16, 16)
        table_layout.setSpacing(10)

        table_title = QLabel("Lịch Sử Trong Ca")
        table_title.setStyleSheet("font-size: 14px; font-weight: 700; color: #f8fafc;")
        table_layout.addWidget(table_title)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Mã NV", "Họ tên", "Giờ vào"])
        self._table.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                alternate-background-color: #0d1720;
                border: none;
                gridline-color: #102630;
                color: #dae2fd;
                font-size: 12px;
            }
            QTableWidget::item {
                border-bottom: 1px solid #102630;
                padding: 6px;
            }
            QTableWidget::item:selected {
                background-color: #1d6475;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #111828;
                color: #8c909f;
                font-weight: 700;
                border: none;
                border-bottom: 1px solid #102630;
                padding: 8px;
                font-size: 10px;
            }
        """)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setShowGrid(False)
        self._table.setAlternatingRowColors(True)
        self._table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

        table_layout.addWidget(self._table, stretch=1)

        self._empty_hint = QLabel("Chưa có ai điểm danh trong ca này.")
        self._empty_hint.setAlignment(Qt.AlignCenter)
        self._empty_hint.setStyleSheet("color: #475569; font-size: 11px; padding: 16px 0;")
        table_layout.addWidget(self._empty_hint)

        col.addWidget(table_frame, stretch=1)

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._btn_start = QPushButton("▶  Bắt đầu ca")
        self._btn_start.setFixedHeight(44)
        self._btn_start.setCursor(Qt.PointingHandCursor)
        self._btn_start.clicked.connect(self._on_start)
        btn_row.addWidget(self._btn_start)

        self._btn_stop = QPushButton("■  Kết thúc ca")
        self._btn_stop.setFixedHeight(44)
        self._btn_stop.setCursor(Qt.PointingHandCursor)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self._btn_stop)

        col.addLayout(btn_row)

        _apply_primary_style(self._btn_start)
        _apply_danger_style(self._btn_stop)

        return panel

    # ── Start / Stop ca ─────────────────────────────────────────────────────
    @Slot()
    def _on_start(self) -> None:
        session_name = f"Ca {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        db_session = self._db.create_session(session_name)
        self._active_session_id = db_session.id
        self._confirm_counts.clear()
        self._table.setRowCount(0)
        self._empty_hint.show()

        if not self._camera.start():
            QMessageBox.critical(self, "Lỗi", "Không thể mở camera!")
            return

        self._worker = AIWorker(self._camera, parent=self)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)

        self._cam_status.setText("●  Camera: đang chạy")
        self._cam_status.setStyleSheet("color:#4edea3; font-size:12px; font-weight:600;")

        self._header_dot.setStyleSheet("color: #4edea3; font-weight: bold; font-size: 11px;")
        self._header_sub.setText(f"Ca đang mở: {session_name}")
        self._header_count_badge.setText("0 lượt")

        self._set_status("Đang nhận diện…", "#4cd7f6")

        logger.info("AttendanceView: bắt đầu ca '%s' (id=%d)", session_name, db_session.id)

    @Slot()
    def _on_stop(self) -> None:
        if self._worker:
            self._worker.stop()
            self._worker = None

        self._camera.stop()

        if self._active_session_id:
            self._db.close_session(self._active_session_id)
            self._active_session_id = None

        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)

        self._cam_status.setText("●  Camera: tắt")
        self._cam_status.setStyleSheet("color:#f87171; font-size:12px; font-weight:600;")
        self._fps_label.setText("")

        self._video_label.setPixmap(QPixmap())
        self._video_label.setText("📷\n\nCamera chưa bật")

        self._header_dot.setStyleSheet("color: #f87171; font-weight: bold; font-size: 11px;")
        self._header_sub.setText("Chưa bắt đầu ca làm việc")

        self._set_status("Chờ khuôn mặt…", "#f8fafc")
        logger.info("AttendanceView: kết thúc ca.")

    # ── Slots nhận Signal từ AIWorker ────────────────────────────────────────
    @Slot(np.ndarray)
    def _on_frame(self, frame_bgr: np.ndarray) -> None:
        """Chuyển BGR numpy → QPixmap → hiển thị, co giãn theo kích thước QLabel hiện tại."""
        h, w, ch = frame_bgr.shape
        rgb = frame_bgr[:, :, ::-1].copy()
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self._video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._video_label.setPixmap(pixmap)

    @Slot(list)
    def _on_result(self, results: list[AIResult]) -> None:
        if self._active_session_id is None:
            return

        for r in results:
            if not r.is_real:
                self._set_status(f"⚠ Phát hiện giả mạo! ({r.spoof_conf:.0%})", "#f87171")
                self._confirm_counts.clear()
                continue

            if r.emp_code is None or r.emp_code == "UNKNOWN":
                self._set_status("Không nhận diện được khuôn mặt.", "#fbbf24")
                continue

            self._confirm_counts[r.emp_code] = self._confirm_counts.get(r.emp_code, 0) + 1

            if self._confirm_counts[r.emp_code] < _CONFIRM_NEEDED:
                self._set_status(
                    f"Đang xác nhận: {r.emp_code} "
                    f"({self._confirm_counts[r.emp_code]}/{_CONFIRM_NEEDED})",
                    "#60a5fa",
                )
                continue

            self._confirm_counts[r.emp_code] = 0
            success, msg = self._db.record_attendance(
                emp_id           = self._resolve_emp_id(r.emp_code),
                session_id       = self._active_session_id,
                confidence_score = r.similarity,
                is_spoofed       = False,
            )

            if success:
                self._set_status(f"✅ Đã chấm công: {r.emp_code}", "#4edea3")
                self._add_table_row(r.emp_code)
            else:
                self._set_status(f"ℹ {r.emp_code}: {msg}", "#8c909f")

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        logger.error("AIWorker error: %s", msg)
        self._set_status(f"Lỗi AI: {msg}", "#f87171")

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _set_status(self, text: str, color: str = "#f8fafc") -> None:
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color:{color};")

    def _resolve_emp_id(self, emp_code: str) -> int:
        emp = self._db.get_employee_by_code(emp_code)
        return emp.id if emp else -1

    def _add_table_row(self, emp_code: str) -> None:
        emp  = self._db.get_employee_by_code(emp_code)
        name = emp.name if emp else "—"
        now  = datetime.now().strftime("%H:%M:%S")

        self._empty_hint.hide()

        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, _make_cell(emp_code))
        self._table.setItem(row, 1, _make_cell(name))
        self._table.setItem(row, 2, _make_cell(now))
        self._table.scrollToBottom()

        self._header_count_badge.setText(f"{row + 1} lượt")

    # ── Dọn dẹp khi đóng widget ─────────────────────────────────────────────
    def closeEvent(self, event) -> None:  # noqa: N802
        self._on_stop()
        super().closeEvent(event)


# ── Widget Helpers ───────────────────────────────────────────────────────────
def _make_cell(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignCenter)
    return item


def _apply_primary_style(btn: QPushButton) -> None:
    btn.setStyleSheet("""
        QPushButton {
            background-color: #4cd7f6;
            color: #0b1326;
            border: none;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 700;
            padding: 0 12px;
        }
        QPushButton:hover { background-color: #6fe2ff; }
        QPushButton:pressed { background-color: #2ca0ba; }
        QPushButton:disabled { background-color: #1e293b; color: #475569; }
    """)


def _apply_danger_style(btn: QPushButton) -> None:
    btn.setStyleSheet("""
        QPushButton {
            background-color: transparent;
            color: #f87171;
            border: 1px solid #f87171;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 700;
            padding: 0 12px;
        }
        QPushButton:hover { background-color: #f8717122; }
        QPushButton:pressed { background-color: #f8717144; }
        QPushButton:disabled { background-color: transparent; color: #334155; border-color: #1e293b; }
    """)
