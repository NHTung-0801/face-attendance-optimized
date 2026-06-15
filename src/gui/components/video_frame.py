"""
src/gui/components/video_frame.py
Component hiển thị luồng Camera tái sử dụng.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel

class VideoFrame(QLabel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border-radius: 8px;
            }
        """)
        self.setMinimumSize(640, 480)
        self.setText("🎥 Đang chờ tín hiệu Camera...")

    @Slot(QImage)
    def update_frame(self, image: QImage) -> None:
        """Nhận QImage từ luồng AI và tự động co giãn giữ tỉ lệ."""
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)

    @Slot()
    def clear_frame(self) -> None:
        """Xóa khung hình hiện tại và đưa về trạng thái chờ."""
        self.clear()
        self.setText("🎥 Đang chờ tín hiệu Camera...")