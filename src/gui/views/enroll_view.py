"""
src/gui/views/enroll_view.py
EnrollView — đăng ký khuôn mặt nhân viên mới.
Thu thập 15 mẫu qua CameraStream, validate qua FaceRecognizer, lưu qua EmployeeManager.
"""

from __future__ import annotations
from src.gui.components.video_frame import VideoFrame
import time
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.core.camera_stream import CameraStream
from src.core.employee_manager import EmployeeManager
from src.core.face_recognizer import FaceRecognizer
from src.utils.config import (
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    ENROLL_CAPTURE_COUNT,
    ENROLL_CAPTURE_INTERVAL,
    FACE_DET_SCORE_THRESHOLD,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TARGET_SAMPLES = ENROLL_CAPTURE_COUNT   # 15 mẫu


# ═══════════════════════════════════════════════════════════════════════════
# Worker thread thu thập mẫu
# ═══════════════════════════════════════════════════════════════════════════
class _SampleCollector(QThread):
    """
    Chạy trong thread riêng:
    - Liên tục đọc frame từ CameraStream.
    - Mỗi ENROLL_CAPTURE_INTERVAL frame, kiểm tra có khuôn mặt rõ không.
    - Phát tiếng bíp/delay nhẹ để nhịp độ thu thập rõ ràng.
    """

    preview_ready: Signal = Signal(np.ndarray)
    sample_captured: Signal = Signal(np.ndarray, int)
    finished: Signal = Signal(bool, str)

    def __init__(self, camera: CameraStream, parent=None) -> None:
        super().__init__(parent)
        self._camera     = camera
        self._recognizer = FaceRecognizer.instance()
        self._running    = False
        self._tick       = 0
        self._count      = 0

    def run(self) -> None:
        self._running = True
        self._tick    = 0
        self._count   = 0

        logger.info("SampleCollector: bat dau thu thap %d mau.", _TARGET_SAMPLES)

        while self._running and self._count < _TARGET_SAMPLES:
            frame = self._camera.get_frame(timeout=0.05)
            if frame is None:
                continue

            self._tick += 1
            annotated = frame.copy()

            # Chỉ đưa vào AI phân tích sau mỗi N frame
            if self._tick % ENROLL_CAPTURE_INTERVAL == 0:
                detections = self._recognizer.get_embeddings_from_frame(frame)

                if detections:
                    # Lấy khuôn mặt rõ nhất
                    detections.sort(key=lambda d: d[2], reverse=True)
                    _, bbox, det_score = detections[0]
                    x1, y1, x2, y2 = bbox

                    import cv2
                    # Vẽ khung nhận diện
                    color = (0, 255, 0) if det_score >= FACE_DET_SCORE_THRESHOLD else (0, 165, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    if det_score >= FACE_DET_SCORE_THRESHOLD:
                        # 1. TĂNG BIẾN ĐẾM (Chỉ khi đủ điều kiện)
                        self._count += 1
                        cv2.putText(
                            annotated,
                            f"Thanh cong: {self._count}/{_TARGET_SAMPLES}",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                        )

                        # Phát ảnh lên UI ngay để người dùng thấy khung xanh
                        self.preview_ready.emit(annotated)

                        # 2. GỬI FULL FRAME (Tuyệt đối không crop ảnh)
                        self.sample_captured.emit(frame.copy(), self._count)
                        logger.debug("SampleCollector: mau %d/%d", self._count, _TARGET_SAMPLES)

                        # 3. NHỊP DỪNG (Cooldown)
                        # Dừng 0.5 giây để người dùng kịp thay đổi góc mặt, 
                        # tạo ra nhịp chụp ảnh chắc chắn giống dự án cũ.
                        time.sleep(0.5)
                        continue  # Bỏ qua dòng emit preview ở cuối vì đã emit rồi
                    else:
                        cv2.putText(
                            annotated,
                            "Khuon mat chua ro",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                        )
                else:
                    import cv2
                    cv2.putText(
                        annotated,
                        "Khong tim thay khuon mat",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )

            # Phát ảnh liên tục để camera không bị giật
            self.preview_ready.emit(annotated)

        if self._count >= _TARGET_SAMPLES:
            self.finished.emit(True, f"Da thu thap du {_TARGET_SAMPLES} mau.")
        else:
            self.finished.emit(False, "Thu thap bi huy.")

        logger.info("SampleCollector: ket thuc (%d mau).", self._count)

    def stop(self) -> None:
        self._running = False
        self.wait(2000)


# ═══════════════════════════════════════════════════════════════════════════
# EnrollView
# ═══════════════════════════════════════════════════════════════════════════
class EnrollView(QWidget):
    """
    Layout:
    ┌─────────────────────────────────────────────┐
    │  [Camera preview — QLabel]  │  [Form panel] │
    │                             │  Mã NV        │
    │                             │  Họ Tên       │
    │                             │  Phòng Ban    │
    │                             │  ProgressBar  │
    │                             │  Status       │
    │                             │  [Nút]        │
    └─────────────────────────────────────────────┘
    """

    # Emit khi đăng ký thành công (để MainWindow cập nhật status bar)
    enrolled = Signal(str)   # emp_code

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._camera   = CameraStream()
        self._manager  = EmployeeManager.instance()
        self._collector: Optional[_SampleCollector] = None

        self._face_samples: list[np.ndarray] = []
        self._camera_started = False

        self._build_ui()

    # ── Xây dựng UI ─────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # ── Cột trái: camera preview ───────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(8)

        self._video_label = VideoFrame() # Đã dùng component mới
        left.addWidget(self._video_label)

        self._cam_hint = QLabel("Nhìn thẳng vào camera, giữ khuôn mặt trong khung hình")
        self._cam_hint.setAlignment(Qt.AlignCenter)
        self._cam_hint.setStyleSheet("color:#64748b; font-size:12px;")
        left.addWidget(self._cam_hint)
        left.addStretch()

        root.addLayout(left)

        # ── Separator ─────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color:#1e293b;")
        root.addWidget(sep)

        # ── Cột phải: form ─────────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(14)

        # Tiêu đề
        title = QLabel("➕  Đăng Ký Nhân Viên")
        title.setFont(QFont("Segoe UI", 15, QFont.Bold))
        title.setStyleSheet("color:#f1f5f9;")
        right.addWidget(title)

        right.addWidget(_divider())

        # Form fields
        self._emp_code_input = _labeled_input("Mã Nhân Viên *", "VD: NV001", right)
        self._name_input     = _labeled_input("Họ và Tên *",    "VD: Nguyễn Văn A", right)
        self._dept_input     = _labeled_input("Phòng Ban",      "VD: Kỹ thuật", right)

        right.addWidget(_divider())

        # Progress bar
        progress_label = QLabel("Tiến độ thu thập mẫu khuôn mặt:")
        progress_label.setStyleSheet("color:#94a3b8; font-size:12px;")
        right.addWidget(progress_label)

        self._progress = QProgressBar()
        self._progress.setMinimum(0)
        self._progress.setMaximum(_TARGET_SAMPLES)
        self._progress.setValue(0)
        self._progress.setFixedHeight(22)
        self._progress.setFormat(f"%v / {_TARGET_SAMPLES} mẫu")
        self._progress.setStyleSheet("""
            QProgressBar {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 4px;
                text-align: center;
                color: #cbd5e1;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2563eb, stop:1 #7c3aed);
                border-radius: 3px;
            }
        """)
        right.addWidget(self._progress)

        # Trạng thái
        self._status_label = QLabel("Điền thông tin và nhấn 'Bắt đầu lấy mẫu'.")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color:#94a3b8; font-size:12px; min-height:36px;")
        self._status_label.setFixedWidth(300)
        right.addWidget(self._status_label)

        right.addStretch()

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._btn_collect = QPushButton("▶  Bắt đầu lấy mẫu")
        self._btn_collect.setFixedHeight(40)
        self._btn_collect.clicked.connect(self._on_start_collect)
        _style_btn(self._btn_collect, "#2563eb")
        btn_row.addWidget(self._btn_collect)

        self._btn_cancel = QPushButton("✕  Hủy")
        self._btn_cancel.setFixedHeight(40)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._on_cancel)
        _style_btn(self._btn_cancel, "#475569")
        btn_row.addWidget(self._btn_cancel)

        right.addLayout(btn_row)

        self._btn_reset = QPushButton("↺  Làm lại")
        self._btn_reset.setFixedHeight(36)
        self._btn_reset.setEnabled(False)
        self._btn_reset.clicked.connect(self._reset_form)
        _style_btn(self._btn_reset, "#334155")
        right.addWidget(self._btn_reset)

        root.addLayout(right)

    # ── Slots ────────────────────────────────────────────────────────────────
    @Slot()
    def _on_start_collect(self) -> None:
        # Validate form
        emp_code = self._emp_code_input.text().strip().upper()
        name     = self._name_input.text().strip()

        if not emp_code:
            self._set_status("⚠ Vui lòng nhập Mã Nhân Viên.", "#f87171")
            self._emp_code_input.setFocus()
            return
        if not name:
            self._set_status("⚠ Vui lòng nhập Họ và Tên.", "#f87171")
            self._name_input.setFocus()
            return

        # Khởi động camera nếu chưa
        if not self._camera_started:
            if not self._camera.start():
                QMessageBox.critical(self, "Lỗi", "Không thể mở camera!")
                return
            self._camera_started = True

        # Reset mẫu
        self._face_samples.clear()
        self._progress.setValue(0)

        # Khởi động collector thread
        self._collector = _SampleCollector(self._camera, parent=self)
        self._collector.preview_ready.connect(self._on_preview)
        self._collector.sample_captured.connect(self._on_sample_captured)
        self._collector.finished.connect(self._on_collect_finished)
        self._collector.start()

        # Cập nhật UI
        self._btn_collect.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._btn_reset.setEnabled(False)
        self._emp_code_input.setEnabled(False)
        self._name_input.setEnabled(False)
        self._dept_input.setEnabled(False)
        self._set_status("📸 Đang thu thập mẫu — giữ khuôn mặt rõ ràng…", "#60a5fa")

    @Slot()
    def _on_cancel(self) -> None:
        if self._collector and self._collector.isRunning():
            self._collector.stop()
        self._restore_controls()
        self._set_status("Đã hủy thu thập mẫu.", "#fbbf24")

    @Slot(np.ndarray)
    def _on_preview(self, frame_bgr: np.ndarray) -> None:
        """Hiển thị frame lên QLabel."""
        h, w, ch = frame_bgr.shape
        rgb   = frame_bgr[:, :, ::-1].copy()
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix   = QPixmap.fromImage(q_img).scaled(
            self._video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._video_label.setPixmap(pix)

    @Slot(np.ndarray, int)
    def _on_sample_captured(self, face_crop: np.ndarray, count: int) -> None:
        self._face_samples.append(face_crop)
        self._progress.setValue(count)
        self._set_status(
            f"✅ Đã thu thập {count}/{_TARGET_SAMPLES} mẫu…",
            "#4ade80",
        )

    @Slot(bool, str)
    def _on_collect_finished(self, success: bool, msg: str) -> None:
        self._restore_controls()

        if not success:
            self._set_status(f"ℹ {msg}", "#94a3b8")
            return

        # Đủ mẫu → tự động enroll
        self._set_status("💾 Đang lưu vào hệ thống…", "#60a5fa")
        self._do_enroll()

    def _do_enroll(self) -> None:
        emp_code   = self._emp_code_input.text().strip().upper()
        name       = self._name_input.text().strip()
        department = self._dept_input.text().strip()

        result = self._manager.enroll_employee(
            emp_code     = emp_code,
            name         = name,
            face_samples = self._face_samples,
            department   = department,
        )

        if result.success:
            self._set_status(
                f"🎉 Đăng ký thành công!\n{result.message}",
                "#4ade80",
            )
            QMessageBox.information(
                self,
                "Thành công",
                f"Đã đăng ký nhân viên:\n\n"
                f"  Mã NV : {emp_code}\n"
                f"  Họ tên: {name}\n"
                f"  Phòng : {department or '—'}\n\n"
                f"{result.embeddings_added} vector khuôn mặt đã lưu vào FAISS.",
            )
            self.enrolled.emit(emp_code)
            self._btn_reset.setEnabled(True)
        else:
            self._set_status(f"❌ {result.message}", "#f87171")
            QMessageBox.warning(self, "Đăng ký thất bại", result.message)
            self._btn_reset.setEnabled(True)

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _restore_controls(self) -> None:
        self._btn_collect.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._emp_code_input.setEnabled(True)
        self._name_input.setEnabled(True)
        self._dept_input.setEnabled(True)

    def _reset_form(self) -> None:
        # 1. Đổi _worker thành _collector cho đúng với biến khai báo trong hàm _on_start_collect
        if self._collector and self._collector.isRunning():
            self._collector.stop()
        self._camera.stop()
        
        # 2. Dùng biến _video_label thay vì _video_frame
        if hasattr(self, '_video_label') and hasattr(self._video_label, 'clear_frame'):
            self._video_label.clear_frame() 

        self._emp_code_input.setEnabled(True)
        self._emp_code_input.clear()
        self._name_input.clear()
        self._dept_input.clear()
        self._progress.setValue(0)
        self._face_samples.clear()
        self._set_status("Điền thông tin và nhấn 'Bắt đầu lấy mẫu'.", "#94a3b8")
        self._btn_reset.setEnabled(False)
        self._emp_code_input.setFocus()

    def _set_status(self, text: str, color: str = "#94a3b8") -> None:
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f"color:{color}; font-size:12px; min-height:36px;"
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._collector and self._collector.isRunning():
            self._collector.stop()
        self._camera.stop()
        super().closeEvent(event)


# ── Widget helpers ───────────────────────────────────────────────────────────
def _labeled_input(label_text: str, placeholder: str, layout: QVBoxLayout) -> QLineEdit:
    lbl = QLabel(label_text)
    lbl.setStyleSheet("color:#cbd5e1; font-size:12px; font-weight:600;")
    layout.addWidget(lbl)

    inp = QLineEdit()
    inp.setPlaceholderText(placeholder)
    inp.setFixedHeight(36)
    layout.addWidget(inp)
    return inp


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("DividerLine")
    return line


def _style_btn(btn: QPushButton, variant: str = "primary") -> None:
    # Thay vì truyền mã màu Hex, ta dùng thuộc tính động (Dynamic Property) của Qt
    btn.setProperty("class", variant)
    btn.style().unpolish(btn)
    btn.style().polish(btn)
