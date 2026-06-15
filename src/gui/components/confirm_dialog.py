"""
src/gui/components/confirm_dialog.py
Modern UI Confirmation Dialog component for the FaceAttend project.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfirmDialog(QDialog):
    """
    A custom modern confirmation dialog replacing the standard QMessageBox.
    Uses global QSS classes for styling and supports frameless UI.
    """

    def __init__(
        self,
        parent: Optional[QWidget],
        title: str,
        message: str,
        danger: bool = False
    ) -> None:
        super().__init__(parent)

        # Remove default title bar for a modern look
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Make it modal
        self.setWindowModality(Qt.WindowModal)

        self._build_ui(title, message, danger)

    def _build_ui(self, title: str, message: str, danger: bool) -> None:
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Container frame to apply card-like styling from dark.qss
        container = QFrame()
        container.setProperty("class", "card")
        
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(24, 24, 24, 24)
        container_layout.setSpacing(16)

        # Title Label
        title_label = QLabel(title)
        title_label.setProperty("class", "text-h2")
        container_layout.addWidget(title_label)

        # Message Label
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setProperty("class", "text-muted")
        msg_label.setMinimumWidth(300)
        container_layout.addWidget(msg_label)

        # Buttons Layout
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.addStretch()

        # Cancel Button
        btn_cancel = QPushButton("Hủy")
        btn_cancel.setProperty("class", "secondary")
        btn_cancel.setCursor(Qt.PointingHandCursor)
        btn_cancel.setFixedHeight(36)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        # Confirm Button
        btn_confirm = QPushButton("Xác nhận")
        btn_confirm_class = "danger" if danger else "primary"
        btn_confirm.setProperty("class", btn_confirm_class)
        btn_confirm.setCursor(Qt.PointingHandCursor)
        btn_confirm.setFixedHeight(36)
        btn_confirm.clicked.connect(self.accept)
        btn_layout.addWidget(btn_confirm)

        container_layout.addLayout(btn_layout)
        layout.addWidget(container)

    @staticmethod
    def ask(
        parent: Optional[QWidget], 
        title: str, 
        message: str, 
        danger: bool = False
    ) -> bool:
        """
        Static helper to easily call the dialog.
        
        Args:
            parent: Parent widget
            title: Dialog title
            message: Confirmation message
            danger: If True, the confirm button will be styled as 'danger' (red).
            
        Returns:
            bool: True if 'Xác nhận' was clicked, False if 'Hủy' was clicked.
        """
        dialog = ConfirmDialog(parent, title, message, danger)
        result = dialog.exec()
        return result == QDialog.Accepted