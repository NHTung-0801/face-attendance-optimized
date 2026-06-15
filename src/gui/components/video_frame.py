"""
src/gui/components/video_frame.py
Component hiển thị luồng Camera tái sử dụng.
"""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel

from src.utils.logger import get_logger

logger = get_logger(__name__)

class VideoFrame(QLabel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        
        # SỬ DỤNG PROPERTY THAY VÌ STYLESHEET HARDCODE
        # Đảm bảo bạn có CSS `.video-frame { background-color: #000; border-radius: 8px; }` trong dark.qss
        self.setProperty("class", "video-frame")
        
        self.setMinimumSize(640, 480)
        self.setText("🎥 Đang chờ tín hiệu Camera...")

    @Slot(np.ndarray)
    def update_frame(self, frame_bgr: np.ndarray) -> None:
        """Nhận np.ndarray (BGR) từ luồng Worker, chuyển đổi và tự động co giãn giữ tỉ lệ."""
        if frame_bgr is None or frame_bgr.size == 0:
            return

        try:
            # 1. Chuyển đổi màu từ BGR (OpenCV) sang RGB (PySide6)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w

            # 2. Tạo QImage
            image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 3. Tạo QPixmap và scale
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
            
        except Exception as e:
            logger.error(f"VideoFrame: Lỗi khi convert frame hiển thị - {e}")

    @Slot()
    def clear_frame(self) -> None:
        """Xóa khung hình hiện tại và đưa về trạng thái chờ."""
        self.clear()
        self.setText("🎥 Đang chờ tín hiệu Camera...")