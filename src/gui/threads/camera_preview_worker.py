"""
src/gui/threads/camera_preview_worker.py
Lightweight worker thread to continuously fetch and emit raw camera frames to the UI.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from src.core.camera_stream import CameraStream
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CameraPreviewWorker(QThread):
    # ── Signals ──────────────────────────────────────────────────────────────
    # Emits the raw numpy array (BGR) continuously for UI rendering
    frame_ready = Signal(np.ndarray)

    def __init__(self, camera: CameraStream, parent=None) -> None:
        super().__init__(parent)
        self._camera = camera
        self._running = False

    def run(self) -> None:
        """Main loop: Fetch frames without AI processing and emit them continuously."""
        self._running = True
        logger.info("CameraPreviewWorker: Started preview stream.")

        while self._running:
            try:
                # Attempt to fetch frame with a timeout to prevent thread blocking
                frame = self._camera.get_frame(timeout=0.05)
            except TypeError:
                # Fallback if the underlying CameraStream definition does not support 'timeout'
                frame = self._camera.get_frame()
                if frame is None:
                    QThread.msleep(50)  # Sleep roughly equivalent to 0.05s

            if frame is not None:
                self.frame_ready.emit(frame)
            else:
                # Yield thread execution slightly to prevent 100% CPU usage on empty frames
                QThread.msleep(10)

        logger.info("CameraPreviewWorker: Stopped.")

    def stop(self) -> None:
        """Stop the worker safely from the main thread."""
        if self._running:
            self._running = False
            self.wait()  # Block until the run() loop finishes completely