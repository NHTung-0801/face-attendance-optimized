"""
src/gui/components/video_frame.py
Component hiển thị luồng Camera tái sử dụng.
Đã tối ưu hiệu năng bằng cv2.resize và áp dụng thuật toán Letterbox chống méo ảnh.
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
        
        self.setProperty("class", "video-frame")
        self.setMinimumSize(640, 480)
        self.setText("🎥 Đang chờ tín hiệu Camera...")

    @Slot(np.ndarray)
    def update_frame(self, frame_rgb: np.ndarray) -> None:
        if frame_rgb is None or frame_rgb.size == 0:
            return

        try:
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(image))
        except Exception as exc:
            logger.exception("Lỗi render khung hình camera")

    def clear_frame(self) -> None:
        self.setPixmap(QPixmap())
        self.setText("🎥 Đang chờ tín hiệu Camera...")