"""
src/gui/components/badge.py
Modern UI Badge component for the FaceAttend project.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Badge(QLabel):
    """
    A reusable Badge component (QLabel subclass) to display statuses or tags.
    Supported variants: 'success', 'danger', 'warning', 'primary', 'neutral'.
    Styles are handled via global QSS using the dynamic property 'class'.
    """

    def __init__(
        self, 
        text: str, 
        variant: str = "neutral", 
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(text, parent)

        # Validate variant to ensure consistent QSS mapping
        valid_variants = {"success", "danger", "warning", "primary", "neutral"}
        if variant not in valid_variants:
            logger.warning(f"Invalid badge variant '{variant}'. Falling back to 'neutral'.")
            variant = "neutral"

        # Apply structural settings (alignment and padding)
        self.setAlignment(Qt.AlignCenter)
        self.setContentsMargins(10, 4, 10, 4)
        
        # We explicitly avoid inline CSS for colors. We set structural properties here
        # or rely entirely on the global dark.qss
        self.setFixedHeight(24)

        # Set dynamic class property for global QSS targeting
        self.set_variant(variant)

    def set_variant(self, variant: str) -> None:
        """
        Dynamically update the badge variant at runtime.
        Forces the Qt style engine to re-evaluate the widget.
        """
        self.setProperty("class", f"badge-{variant}")
        
        # Repolish to apply the new QSS rules immediately
        self.style().unpolish(self)
        self.style().polish(self)