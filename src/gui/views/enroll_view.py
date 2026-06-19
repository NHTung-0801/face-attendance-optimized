"""
src/gui/views/enroll_view.py
EnrollView — Đăng ký khuôn mặt nhân viên mới.
Thu thập 15 mẫu qua CameraStream, với bộ lọc realtime 3 lớp: Người thực -> Không giả mạo -> Không trùng lặp.
"""

from __future__ import annotations
import time
from typing import Optional

import numpy as np
import cv2
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
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

from src.gui.components.video_frame import VideoFrame
from src.core.camera_stream import CameraStream
from src.core.employee_manager import EmployeeManager
from src.core.face_recognizer import FaceRecognizer
from src.core.anti_spoofing import AntiSpoofing, SpoofResult
from src.utils.config import (
    ENROLL_CAPTURE_COUNT,
    ENROLL_CAPTURE_INTERVAL,
    FACE_DET_SCORE_THRESHOLD,
    DISPLAY_WIDTH,    # <-- THÊM DÒNG NÀY
    DISPLAY_HEIGHT,   # <-- THÊM DÒNG NÀY
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TARGET_SAMPLES = ENROLL_CAPTURE_COUNT   # Mặc định: 15 mẫu

# ── Helpers IoU để map khuôn mặt và kết quả Anti-Spoofing ──────────────────
def _calc_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union

def _find_matching_spoof(
    bbox: tuple[int, int, int, int],
    spoof_results: list[SpoofResult],
    iou_threshold: float = 0.3,
) -> Optional[SpoofResult]:
    best, best_iou = None, 0.0
    for sr in spoof_results:
        if sr.bbox is None:
            continue
        iou = _calc_iou(bbox, sr.bbox)
        if iou > best_iou:
            best_iou, best = iou, sr
    return best if best_iou >= iou_threshold else None

# ═══════════════════════════════════════════════════════════════════════════
# Worker thread thu thập mẫu với Bộ Lọc 3 Lớp
# ═══════════════════════════════════════════════════════════════════════════
class _SampleCollector(QThread):
    preview_ready: Signal = Signal(np.ndarray)
    sample_captured: Signal = Signal(np.ndarray, int)
    finished: Signal = Signal(bool, str)

    def __init__(self, camera: CameraStream, parent=None) -> None:
        super().__init__(parent)
        self._camera     = camera
        self._recognizer = FaceRecognizer.instance()
        self._spoofing   = AntiSpoofing.instance()
        self._running    = False
        self._tick       = 0
        self._count      = 0

    def run(self) -> None:
        self._running = True
        self._tick    = 0
        self._count   = 0
        last_capture  = 0.0

        # --- BỔ SUNG BỘ NHỚ ĐỆM (CACHE) KHUNG VIỀN ---
        last_bbox  = None
        last_text  = ""
        last_color = (0, 0, 0)

        logger.info("SampleCollector: Bắt đầu thu thập %d mẫu.", _TARGET_SAMPLES)

        while self._running and self._count < _TARGET_SAMPLES:
            frame = self._camera.get_frame(timeout=0.05)
            if frame is None:
                continue

            self._tick += 1
            annotated = frame.copy()
            now = time.time()

            is_cooldown = (now - last_capture) < 0.5
            run_ai = not is_cooldown and (self._tick % ENROLL_CAPTURE_INTERVAL == 0)

            # CHỈ CHẠY AI VÀ CẬP NHẬT CACHE MỖI 5 FRAME
            if run_ai:
                detections = self._recognizer.get_embeddings_from_frame(frame)

                if detections:
                    detections.sort(key=lambda d: d[2], reverse=True)
                    emb, bbox, det_score = detections[0]
                    last_bbox = bbox

                    if det_score < FACE_DET_SCORE_THRESHOLD:
                        last_text, last_color = "Khuon mat chua ro!", (0, 165, 255)
                    else:
                        spoof_results = self._spoofing.detect_spoof(frame)
                        matched_spoof = _find_matching_spoof(bbox, spoof_results)
                        is_real = matched_spoof.is_real if matched_spoof else True

                        if not is_real:
                            last_text, last_color = "CANH BAO GIA MAO!", (0, 0, 255)
                        else:
                            matches = self._recognizer.identify_face(emb)
                            is_duplicate = False
                            dup_code = ""
                            
                            if matches:
                                match_code, similarity = matches[0]
                                is_duplicate = True
                                dup_code = match_code

                            if is_duplicate:
                                last_text, last_color = f"Da ton tai ({dup_code})", (0, 0, 255)
                            else:
                                self._count += 1
                                last_capture = now
                                last_text, last_color = f"Hop le: {self._count}/{_TARGET_SAMPLES}", (0, 255, 0)
                                self.sample_captured.emit(frame.copy(), self._count)
                else:
                    last_bbox = None
                    last_text = "Khong tim thay khuon mat"
                    last_color = (0, 0, 255)

            # --- VẼ LÊN FRAME TỪ CACHE MỖI MILI-GIÂY (Khắc phục chớp tắt) ---
            if last_bbox is not None:
                x1, y1, x2, y2 = last_bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), last_color, 2)
                cv2.putText(annotated, last_text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, last_color, 2)
            elif last_text == "Khong tim thay khuon mat":
                cv2.putText(annotated, last_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, last_color, 2)

            # --- TỐI ƯU HIỆU NĂNG: Xử lý Letterbox ngầm ---
            h, w = annotated.shape[:2]
            scale = min(DISPLAY_WIDTH / w, DISPLAY_HEIGHT / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            resized = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            pad_x = (DISPLAY_WIDTH - new_w) // 2
            pad_y = (DISPLAY_HEIGHT - new_h) // 2
            canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
            
            rgb_frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            self.preview_ready.emit(rgb_frame)

        if self._count >= _TARGET_SAMPLES:
            self.finished.emit(True, f"Đã thu thập đủ {_TARGET_SAMPLES} mẫu.")
        else:
            self.finished.emit(False, "Thu thập bị hủy.")

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

# ═══════════════════════════════════════════════════════════════════════════
# EnrollView (Giao diện)
# ═══════════════════════════════════════════════════════════════════════════
class EnrollView(QWidget):
    enrolled = Signal(str)   # emp_code

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._camera   = CameraStream()
        self._manager  = EmployeeManager.instance()
        self._collector: Optional[_SampleCollector] = None

        self._face_samples: list[np.ndarray] = []
        self._camera_started = False

        self._build_ui()

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

        body_layout.addWidget(self._build_camera_card(), stretch=5)
        body_layout.addWidget(self._build_form_card(), stretch=4)

        root.addWidget(body, stretch=1)

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

        title = QLabel("➕  Đăng Ký Khuôn Mặt")
        title.setStyleSheet(
            "font-size: 22px; font-weight: 700; color: #f8fafc; letter-spacing: 0.3px;"
        )
        title_box.addWidget(title)

        sub = QLabel("Thu thập dữ liệu sinh trắc học và tạo hồ sơ nhân viên mới")
        sub.setStyleSheet("color: #8c909f; font-size: 12px;")
        title_box.addWidget(sub)

        layout.addLayout(title_box)
        layout.addStretch()
        return header

    def _build_camera_card(self) -> QFrame:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #161f2e;
                border: 1px solid #1d6475;
                border-radius: 16px;
            }
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("📷 TRẠNG THÁI CAMERA")
        title.setStyleSheet("color: #4cd7f6; font-size: 12px; font-weight: 700; letter-spacing: 1px; border: none;")
        layout.addWidget(title)

        self._video_label = VideoFrame()
        self._video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._video_label, stretch=1)

        self._cam_hint = QLabel("Nhìn thẳng vào camera, giữ khuôn mặt trong khung hình để hệ thống phân tích vector.")
        self._cam_hint.setAlignment(Qt.AlignCenter)
        self._cam_hint.setStyleSheet("color: #8c909f; font-size: 12px; border: none;")
        layout.addWidget(self._cam_hint)

        return card

    def _build_form_card(self) -> QFrame:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #161f2e;
                border: 1px solid #1d6475;
                border-radius: 16px;
            }
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(20)

        title = QLabel("📝 THÔNG TIN HỒ SƠ")
        title.setStyleSheet("color: #4cd7f6; font-size: 12px; font-weight: 700; letter-spacing: 1px; border: none;")
        layout.addWidget(title)

        self._emp_code_input = self._create_input_group("Mã Nhân Viên *", "VD: NV001", layout)
        self._name_input     = self._create_input_group("Họ và Tên *", "VD: Nguyễn Văn A", layout)
        self._dept_input     = self._create_input_group("Phòng Ban", "VD: Kỹ thuật", layout)

        prog_layout = QVBoxLayout()
        prog_layout.setSpacing(8)
        
        progress_label = QLabel("Tiến độ thu thập sinh trắc học:")
        progress_label.setStyleSheet("color: #8c909f; font-size: 12px; font-weight: 600; border: none;")
        prog_layout.addWidget(progress_label)

        self._progress = QProgressBar()
        self._progress.setMinimum(0)
        self._progress.setMaximum(_TARGET_SAMPLES)
        self._progress.setValue(0)
        self._progress.setFixedHeight(24)
        self._progress.setFormat(f"%v / {_TARGET_SAMPLES} mẫu")
        self._progress.setStyleSheet("""
            QProgressBar {
                background: #0d1720;
                border: 1px solid #1d6475;
                border-radius: 6px;
                text-align: center;
                color: #f8fafc;
                font-size: 11px;
                font-weight: 700;
            }
            QProgressBar::chunk {
                background-color: #4cd7f6;
                border-radius: 4px;
            }
        """)
        prog_layout.addWidget(self._progress)
        layout.addLayout(prog_layout)

        self._status_label = QLabel("Vui lòng điền thông tin và nhấn 'Bắt đầu lấy mẫu'.")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #94a3b8; font-size: 13px; min-height: 40px; border: none;")
        layout.addWidget(self._status_label)

        layout.addStretch()

        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(12)

        self._btn_collect = QPushButton("▶  Bắt đầu lấy mẫu")
        self._btn_collect.setFixedHeight(46)
        self._btn_collect.setCursor(Qt.PointingHandCursor)
        self._btn_collect.clicked.connect(self._on_start_collect)
        self._style_primary_btn(self._btn_collect)
        btn_layout.addWidget(self._btn_collect)

        # Đã loại bỏ hoàn toàn nút Hủy. Chỉ giữ lại nút Làm lại.
        self._btn_reset = QPushButton("↺  Làm lại")
        self._btn_reset.setFixedHeight(40)
        self._btn_reset.setCursor(Qt.PointingHandCursor)
        self._btn_reset.setEnabled(False)
        self._btn_reset.clicked.connect(self._reset_form)
        self._style_secondary_btn(self._btn_reset)
        btn_layout.addWidget(self._btn_reset)

        layout.addLayout(btn_layout)

        return card

    def _create_input_group(self, label_text: str, placeholder: str, parent_layout: QVBoxLayout) -> QLineEdit:
        group = QVBoxLayout()
        group.setSpacing(6)

        lbl = QLabel(label_text)
        lbl.setStyleSheet("color: #cbd5e1; font-size: 12px; font-weight: 600; border: none;")
        group.addWidget(lbl)

        inp = QLineEdit()
        inp.setPlaceholderText(placeholder)
        inp.setFixedHeight(42)
        inp.setStyleSheet("""
            QLineEdit {
                background-color: #0d1720;
                color: #f8fafc;
                border: 1px solid #1d6475;
                border-radius: 8px;
                padding: 0 14px;
                font-size: 13px;
            }
            QLineEdit:focus { border: 1px solid #4cd7f6; }
            QLineEdit::placeholder { color: #475569; }
            QLineEdit:disabled { background-color: #060e20; color: #475569; border: 1px solid #102630; }
        """)
        group.addWidget(inp)
        parent_layout.addLayout(group)
        return inp

    def _style_primary_btn(self, btn: QPushButton) -> None:
        btn.setStyleSheet("""
            QPushButton {
                background-color: #4cd7f6;
                color: #0b1326;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton:hover { background-color: #6fe2ff; }
            QPushButton:pressed { background-color: #2ca0ba; }
            QPushButton:disabled { background-color: #1e293b; color: #475569; }
        """)

    def _style_secondary_btn(self, btn: QPushButton) -> None:
        btn.setStyleSheet("""
            QPushButton {
                background-color: #0d1720;
                color: #4cd7f6;
                border: 1px solid #4cd7f6;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton:hover { background-color: #4cd7f6; color: #0b1326; }
            QPushButton:pressed { background-color: #2ca0ba; color: #0b1326; }
            QPushButton:disabled { background-color: transparent; border: 1px solid #334155; color: #475569; }
        """)

    @Slot()
    def _on_start_collect(self) -> None:
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

        if not self._camera_started:
            if not self._camera.start():
                QMessageBox.critical(self, "Lỗi", "Không thể mở camera!")
                return
            self._camera_started = True

        self._face_samples.clear()
        self._progress.setValue(0)

        self._collector = _SampleCollector(self._camera, parent=self)
        self._collector.preview_ready.connect(self._on_preview)
        self._collector.sample_captured.connect(self._on_sample_captured)
        self._collector.finished.connect(self._on_collect_finished)
        self._collector.start()

        self._btn_collect.setEnabled(False)
        self._btn_reset.setEnabled(False)
        self._emp_code_input.setEnabled(False)
        self._name_input.setEnabled(False)
        self._dept_input.setEnabled(False)
        self._set_status("📸 Đang thu thập mẫu sinh trắc học — vui lòng nhìn thẳng...", "#4cd7f6")

    @Slot(np.ndarray)
    def _on_preview(self, frame_bgr: np.ndarray) -> None:
        # Tối ưu resize ảnh: Sử dụng hàm update_frame của VideoFrame (chứa cv2.resize siêu tốc)
        if hasattr(self._video_label, 'update_frame'):
            self._video_label.update_frame(frame_bgr)

    @Slot(np.ndarray, int)
    def _on_sample_captured(self, face_crop: np.ndarray, count: int) -> None:
        self._face_samples.append(face_crop)
        self._progress.setValue(count)
        self._set_status(
            f"✅ Hệ thống đã ghi nhận {count}/{_TARGET_SAMPLES} vector...",
            "#4edea3",
        )

    @Slot(bool, str)
    def _on_collect_finished(self, success: bool, msg: str) -> None:
        self._restore_controls()
        
        # [FIX]: Tự động ngắt camera ngay khi đủ mẫu để không chạy ngầm
        self._camera.stop()
        self._camera_started = False

        if not success:
            self._set_status(f"ℹ {msg}", "#94a3b8")
            return

        self._set_status("💾 Đang mã hóa và lưu vector vào Database...", "#4cd7f6")
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
                f"🎉 Thiết lập thành công!\n{result.message}",
                "#4edea3",
            )
            QMessageBox.information(
                self,
                "Thành công",
                f"Đã đăng ký hồ sơ nhân viên:\n\n"
                f"  Mã NV : {emp_code}\n"
                f"  Họ tên: {name}\n"
                f"  Phòng : {department or '—'}\n\n"
                f"Hệ thống đã lưu {result.embeddings_added} vector sinh trắc học vào FAISS.",
            )
            self.enrolled.emit(emp_code)
            self._btn_reset.setEnabled(True)
        else:
            self._set_status(f"❌ {result.message}", "#f87171")
            QMessageBox.warning(self, "Đăng ký thất bại", result.message)
            self._btn_reset.setEnabled(True)

    def _restore_controls(self) -> None:
        self._btn_collect.setEnabled(True)
        self._emp_code_input.setEnabled(True)
        self._name_input.setEnabled(True)
        self._dept_input.setEnabled(True)

    def _reset_form(self) -> None:
        if self._collector and self._collector.isRunning():
            self._collector.stop()
        
        # [FIX]: Trả cờ về False dứt điểm để sửa lỗi liệt nút "Bắt đầu lấy mẫu"
        self._camera.stop()
        self._camera_started = False
        
        if hasattr(self, '_video_label') and hasattr(self._video_label, 'clear_frame'):
            self._video_label.clear_frame() 

        self._emp_code_input.setEnabled(True)
        self._emp_code_input.clear()
        self._name_input.clear()
        self._dept_input.clear()
        self._progress.setValue(0)
        self._face_samples.clear()
        self._set_status("Vui lòng điền thông tin và nhấn 'Bắt đầu lấy mẫu'.", "#94a3b8")
        self._btn_reset.setEnabled(False)
        self._emp_code_input.setFocus()

    @Slot()
    def _on_cancel(self) -> None:
        """
        Hàm giao tiếp để MainWindow gọi tới khi chuyển tab hoặc tắt hệ thống.
        Gọi _reset_form để đảm bảo camera được tắt và dọn dẹp sạch sẽ.
        """
        self._reset_form()

    def _set_status(self, text: str, color: str = "#94a3b8") -> None:
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f"color:{color}; font-size:13px; font-weight:600; min-height:40px; border:none;"
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        # [FIX]: Dọn dẹp an toàn khi tắt ứng dụng hoặc chuyển trang, không gọi _on_cancel sai lầm.
        try:
            if self._collector and self._collector.isRunning():
                self._collector.stop()
            self._camera.stop()
            self._camera_started = False
        except Exception as e:
            logger.warning(f"Lỗi khi dọn dẹp EnrollView: {e}")
            
        super().closeEvent(event)