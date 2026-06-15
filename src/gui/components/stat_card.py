"""
src/gui/components/stat_card.py
Modern UI Statistic Card component for the FaceAttend project.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from src.utils.logger import get_logger

logger = get_logger(__name__)


class StatCard(QFrame):
    """
    A reusable Statistic Card component to display key metrics.
    Styles are handled via global QSS using dynamic properties.
    """

    def __init__(
        self,
        title: str,
        value: str,
        icon_emoji: str = "",
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)

        # Assign the global card style class
        self.setProperty("class", "card")

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        # ── Top row: Icon and Title ──────────────────────────────────────────
        top_layout = QHBoxLayout()
        top_layout.setSpacing(8)
        top_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        if icon_emoji:
            icon_label = QLabel(icon_emoji)
            top_layout.addWidget(icon_label)

        title_label = QLabel(title)
        title_label.setProperty("class", "text-muted")
        top_layout.addWidget(title_label)
        top_layout.addStretch()  # Push icon and title to the left

        layout.addLayout(top_layout)

        # ── Bottom row: Value ────────────────────────────────────────────────
        self._value_label = QLabel(value)
        self._value_label.setProperty("class", "text-h2")
        self._value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self._value_label)

    def update_value(self, new_value: str) -> None:
        """
        Dynamically update the displayed numeric value or text.
        """
        self._value_label.setText(new_value)