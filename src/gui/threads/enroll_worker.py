"""
src/gui/threads/enroll_worker.py
Worker thread for capturing face samples during the enrollment process.
"""

from __future__ import annotations

import time
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from src.core.camera_stream import CameraStream
from src.core.face_recognizer import FaceRecognizer
from src.utils.config import FACE_DET_SCORE_THRESHOLD, ENROLL_CAPTURE_INTERVAL
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EnrollWorker(QThread):
    # ── Signals ──────────────────────────────────────────────────────────────
    # Phát mảng numpy (BGR) liên tục để GUI (VideoFrame component) tự render
    preview_ready = Signal(np.ndarray)
    
    # Phát mảng frame gốc (không vẽ Bounding Box) và số thứ tự sample đã chụp
    sample_captured = Signal(np.ndarray, int)
    
    # Phát tín hiệu (trạng thái thành công, thông báo) khi hoàn thành hoặc bị hủy
    finished = Signal(bool, str)

    def __init__(self, camera: CameraStream, parent=None) -> None:
        super().__init__(parent)
        self._camera = camera
        
        # Singleton pattern (hoặc lấy instance chuẩn) của FaceRecognizer
        self._recognizer = FaceRecognizer.instance() if hasattr(FaceRecognizer, 'instance') else FaceRecognizer()
        
        self._is_running = False
        self._target_samples = 15
        self._count = 0
        
        # Dùng để tạo nhịp điệu (cadence) 0.5s giữa các lần chụp mà không dùng sleep() gây giật lag UI
        self._last_capture_time = 0.0

    def run(self) -> None:
        """Main loop: Đọc frame liên tục, trích xuất đặc trưng và gửi tín hiệu UI."""
        self._is_running = True
        self._count = 0
        frame_counter = 0

        logger.info("EnrollWorker: Started capturing process.")

        while self._is_running:
            frame = self._camera.get_frame()
            if frame is None:
                # Đợi một chút nếu camera chưa trả về frame (tránh 100% CPU)
                QThread.msleep(10)
                continue

            annotated_frame = frame.copy()
            frame_counter += 1

            # Chỉ đưa vào model AI để nhận diện mỗi ENROLL_CAPTURE_INTERVAL frames
            if frame_counter % ENROLL_CAPTURE_INTERVAL == 0:
                faces = self._recognizer.get_embeddings_from_frame(frame)
                
                if faces:
                    # Lấy khuôn mặt rõ nhất (giả sử model đã sort hoặc lấy khuôn mặt đầu tiên)
                    clearest_face = faces[0]
                    bbox = clearest_face.bbox.astype(int)
                    score = clearest_face.det_score
                    x1, y1, x2, y2 = bbox

                    is_clear = score >= FACE_DET_SCORE_THRESHOLD
                    
                    if is_clear:
                        color = (0, 255, 0)  # Green (BGR)
                        text = f"Clear ({score:.2f}) - Capturing..."
                    else:
                        color = (0, 165, 255)  # Orange (BGR)
                        text = f"Low Score ({score:.2f}) - Keep still"

                    # Vẽ Bounding Box và Điểm số
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated_frame, text, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )

                    # Logic chụp (cadence 0.5 giây/ảnh)
                    current_time = time.time()
                    if is_clear and (current_time - self._last_capture_time >= 0.5):
                        self._count += 1
                        self._last_capture_time = current_time
                        
                        # Emit frame gốc (raw) để lưu trữ chất lượng cao nhất
                        raw_frame_copy = frame.copy()
                        self.sample_captured.emit(raw_frame_copy, self._count)
                        logger.info(f"EnrollWorker: Captured sample {self._count}/{self._target_samples}")

                        if self._count >= self._target_samples:
                            self.preview_ready.emit(annotated_frame)  # Emit frame cuối trước khi break
                            self.finished.emit(True, "Successfully collected all face samples.")
                            self._is_running = False
                            break
                else:
                    cv2.putText(
                        annotated_frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                    )

            # Emit preview liên tục để UI không bị đóng băng
            self.preview_ready.emit(annotated_frame)

        # Xử lý trường hợp vòng lặp bị break sớm (do người dùng stop)
        if self._count < self._target_samples:
            self.finished.emit(False, "Enrollment was canceled or stopped.")

        logger.info(f"EnrollWorker: Stopped ({self._count}/{self._target_samples} samples).")

    def stop(self) -> None:
        """Dừng worker an toàn từ bên ngoài."""
        if self._is_running:
            self._is_running = False
            self.wait()  # Chờ thread kết thúc hẳn vòng lặp