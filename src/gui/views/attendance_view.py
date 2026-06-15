"""
src/gui/views/attendance_view.py
AttendanceView — màn hình điểm danh realtime.
Nhận frame + AIResult từ AIWorker qua Signal, hiển thị lên QLabel.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
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
from src.utils.config import (
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    RECOGNITION_CONFIRM_COUNT,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CONFIRM_NEEDED = RECOGNITION_CONFIRM_COUNT   # Số frame liên tiếp để xác nhận


class AttendanceView(QWidget):
    """
    Layout:
    ┌─────────────────────────────────────┐
    │  [Camera feed — QLabel]  │  [Panel] │
    │                          │  Status  │
    │                          │  Table   │
    │                          │  Buttons │
    └─────────────────────────────────────┘
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._db      = DatabaseManager.instance()
        self._camera  = CameraStream()
        self._worker: Optional[AIWorker] = None

        # Bộ đếm xác nhận nhận diện (tránh ghi nhầm do 1 frame)
        self._confirm_counts: dict[str, int] = {}   # emp_code → count

        self._active_session_id: Optional[int] = None

        self._build_ui()

    # ── Xây dựng UI ─────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ── Cột trái: camera feed ──────────────────────────────────────────
        left = QVBoxLayout()

        self._video_label = QLabel("Camera chưa bật")
        self._video_label.setFixedSize(DISPLAY_WIDTH, DISPLAY_HEIGHT)
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setStyleSheet(
            "background:#111; color:#555; border:2px solid #333; border-radius:6px;"
        )
        left.addWidget(self._video_label)

        # FPS / trạng thái camera
        self._cam_status = QLabel("● Camera: tắt")
        self._cam_status.setStyleSheet("color:#f66; font-size:12px;")
        left.addWidget(self._cam_status)
        left.addStretch()

        root.addLayout(left)

        # ── Separator ─────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color:#333;")
        root.addWidget(sep)

        # ── Cột phải: panel điều khiển ─────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(10)

        # Tiêu đề
        title = QLabel("🎯  Điểm Danh")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        right.addWidget(title)

        # Trạng thái nhận diện
        self._status_label = QLabel("Chờ khuôn mặt…")
        self._status_label.setFont(QFont("Segoe UI", 11))
        self._status_label.setWordWrap(True)
        self._status_label.setFixedWidth(280)
        right.addWidget(self._status_label)

        # Bảng lịch sử điểm danh trong phiên
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Mã NV", "Họ tên", "Giờ vào"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setFixedWidth(320)
        self._table.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        right.addWidget(self._table)

        # Buttons
        btn_row = QHBoxLayout()

        self._btn_start = QPushButton("▶  Bắt đầu ca")
        self._btn_start.setFixedHeight(38)
        self._btn_start.clicked.connect(self._on_start)
        btn_row.addWidget(self._btn_start)

        self._btn_stop = QPushButton("■  Kết thúc ca")
        self._btn_stop.setFixedHeight(38)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self._btn_stop)

        right.addLayout(btn_row)

        _apply_button_style(self._btn_start, "#2563eb")
        _apply_button_style(self._btn_stop, "#dc2626")

        root.addLayout(right)

    # ── Start / Stop ca ─────────────────────────────────────────────────────
    @Slot()
    def _on_start(self) -> None:
        # Tạo session mới trong DB
        session_name = f"Ca {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        db_session = self._db.create_session(session_name)
        self._active_session_id = db_session.id
        self._confirm_counts.clear()
        self._table.setRowCount(0)

        # Khởi động camera
        if not self._camera.start():
            QMessageBox.critical(self, "Lỗi", "Không thể mở camera!")
            return

        # Khởi động AI worker
        self._worker = AIWorker(self._camera, parent=self)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._cam_status.setText("● Camera: đang chạy")
        self._cam_status.setStyleSheet("color:#4ade80; font-size:12px;")
        self._status_label.setText("Đang nhận diện…")

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
        self._cam_status.setText("● Camera: tắt")
        self._cam_status.setStyleSheet("color:#f66; font-size:12px;")
        self._video_label.setText("Camera chưa bật")
        self._video_label.setPixmap(QPixmap())
        self._status_label.setText("Chờ khuôn mặt…")
        logger.info("AttendanceView: kết thúc ca.")

    # ── Slots nhận Signal từ AIWorker ────────────────────────────────────────
    @Slot(np.ndarray)
    def _on_frame(self, frame_bgr: np.ndarray) -> None:
        """Chuyển BGR numpy → QPixmap → hiển thị lên QLabel."""
        h, w, ch = frame_bgr.shape
        rgb = frame_bgr[:, :, ::-1].copy()   # BGR → RGB (không dùng cvtColor để tiết kiệm)
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self._video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._video_label.setPixmap(pixmap)

    @Slot(list)
    def _on_result(self, results: list[AIResult]) -> None:
        """
        Nhận kết quả AI, kiểm tra confirm count rồi ghi điểm danh.
        """
        if self._active_session_id is None:
            return

        for r in results:
            # Nếu giả mạo → cảnh báo và bỏ qua
            if not r.is_real:
                self._set_status(
                    f"⚠ Phát hiện giả mạo! ({r.spoof_conf:.0%})",
                    color="#f87171",
                )
                self._confirm_counts.clear()
                continue

            if r.emp_code is None or r.emp_code == "UNKNOWN":
                self._set_status("Không nhận diện được khuôn mặt.", color="#fbbf24")
                continue

            # Tăng bộ đếm confirm
            self._confirm_counts[r.emp_code] = self._confirm_counts.get(r.emp_code, 0) + 1

            if self._confirm_counts[r.emp_code] < _CONFIRM_NEEDED:
                self._set_status(
                    f"Đang xác nhận: {r.emp_code} ({self._confirm_counts[r.emp_code]}/{_CONFIRM_NEEDED})",
                    color="#60a5fa",
                )
                continue

            # Đủ confirm → ghi điểm danh
            self._confirm_counts[r.emp_code] = 0   # Reset để tránh ghi liên tục
            success, msg = self._db.record_attendance(
                emp_id          = self._resolve_emp_id(r.emp_code),
                session_id      = self._active_session_id,
                confidence_score= r.similarity,
                is_spoofed      = False,
            )

            if success:
                self._set_status(f"✅ Đã chấm công: {r.emp_code}", color="#4ade80")
                self._add_table_row(r.emp_code)
            else:
                self._set_status(f"ℹ {r.emp_code}: {msg}", color="#94a3b8")

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        logger.error("AIWorker error: %s", msg)
        self._set_status(f"Lỗi AI: {msg}", color="#f87171")

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _set_status(self, text: str, color: str = "#e2e8f0") -> None:
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color:{color}; font-size:11px;")

    def _resolve_emp_id(self, emp_code: str) -> int:
        """Lấy employee.id từ emp_code (có thể cache nếu cần tối ưu hơn)."""
        emp = self._db.get_employee_by_code(emp_code)
        return emp.id if emp else -1

    def _add_table_row(self, emp_code: str) -> None:
        """Thêm dòng vào bảng lịch sử trong phiên."""
        emp = self._db.get_employee_by_code(emp_code)
        name = emp.name if emp else "—"
        now  = datetime.now().strftime("%H:%M:%S")

        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, _make_cell(emp_code))
        self._table.setItem(row, 1, _make_cell(name))
        self._table.setItem(row, 2, _make_cell(now))
        self._table.scrollToBottom()

    # ── Dọn dẹp khi đóng widget ─────────────────────────────────────────────
    def closeEvent(self, event) -> None:  # noqa: N802
        self._on_stop()
        super().closeEvent(event)


# ── Widget Helpers ───────────────────────────────────────────────────────────
def _make_cell(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignCenter)
    return item


def _apply_button_style(btn: QPushButton, hex_color: str) -> None:
    btn.setStyleSheet(f"""
        QPushButton {{
            background: {hex_color};
            color: #fff;
            border: none;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            padding: 0 12px;
        }}
        QPushButton:hover  {{ background: {hex_color}cc; }}
        QPushButton:disabled {{ background: #334155; color: #64748b; }}
    """)
