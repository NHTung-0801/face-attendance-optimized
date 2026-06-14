"""
src/gui/threads/enroll_worker.py
Worker thread for capturing face samples during the enrollment process.
"""

import time
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from src.core.camera_stream import CameraStream
from src.core.face_recognizer import FaceRecognizer
from src.utils.config import FACE_DET_SCORE_THRESHOLD, ENROLL_CAPTURE_INTERVAL
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EnrollWorker(QThread):
    # ── Signals ──────────────────────────────────────────────────────────────
    # Emits the live video feed converted to QImage for smooth UI rendering
    frame_ready = Signal(QImage)
    
    # Emits the cropped face array (BGR) and current sample count
    sample_captured = Signal(np.ndarray, int)
    
    # Emits (success_status, message) when target is reached or canceled
    finished_enrollment = Signal(bool, str)

    def __init__(
        self, 
        camera: CameraStream, 
        target_samples: int = 15, 
        margin: int = 20, 
        parent=None
    ):
        super().__init__(parent)
        self._camera = camera
        self._recognizer = FaceRecognizer.instance()
        self._target_samples = target_samples
        self._margin = margin
        
        self._running = False
        self._count = 0
        self._tick = 0

    def run(self) -> None:
        """Main active loop for the thread."""
        self._running = True
        self._count = 0
        self._tick = 0
        
        logger.info(f"EnrollWorker: Started collecting {self._target_samples} samples.")

        while self._running and self._count < self._target_samples:
            frame = self._camera.get_frame(timeout=0.05)
            if frame is None:
                continue

            self._tick += 1
            annotated_frame = frame.copy()

            # Process AI detection only at defined intervals to save CPU
            if self._tick % ENROLL_CAPTURE_INTERVAL == 0:
                detections = self._recognizer.get_embeddings_from_frame(frame)

                if detections:
                    # Filter out low-confidence detections
                    valid_faces = [d for d in detections if d[2] >= FACE_DET_SCORE_THRESHOLD]
                    
                    if valid_faces:
                        # Find the LARGEST face based on bounding box area
                        valid_faces.sort(
                            key=lambda d: (d[1][2] - d[1][0]) * (d[1][3] - d[1][1]), 
                            reverse=True
                        )
                        
                        _, bbox, det_score = valid_faces[0]
                        x1, y1, x2, y2 = bbox

                        # Draw bounding box for UI feedback
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Calculate crop coordinates with margin
                        h, w = frame.shape[:2]
                        cx1 = max(0, x1 - self._margin)
                        cy1 = max(0, y1 - self._margin)
                        cx2 = min(w, x2 + self._margin)
                        cy2 = min(h, y2 + self._margin)
                        
                        # Crop the original frame (not the annotated one)
                        face_crop = frame[cy1:cy2, cx1:cx2].copy()

                        self._count += 1
                        
                        # Emit UI text
                        cv2.putText(
                            annotated_frame,
                            f"Sample: {self._count}/{self._target_samples}",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                        )
                        
                        # Emit the captured crop
                        self.sample_captured.emit(face_crop, self._count)
                        logger.debug(f"EnrollWorker: Sample {self._count}/{self._target_samples} captured.")

                        # Emit the frame immediately to show the green box, then cool down
                        self._emit_frame(annotated_frame)
                        time.sleep(0.4)  # Delay to ensure distinct sample angles
                        continue
                    else:
                        cv2.putText(
                            annotated_frame, "Face not clear", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                        )
                else:
                    cv2.putText(
                        annotated_frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                    )

            # Constantly emit the live feed to keep the UI smooth
            self._emit_frame(annotated_frame)

        # Loop ended: check if target was reached
        if self._count >= self._target_samples:
            self.finished_enrollment.emit(True, f"Successfully collected {self._target_samples} samples.")
        else:
            self.finished_enrollment.emit(False, "Enrollment was canceled or stopped.")

        logger.info(f"EnrollWorker: Stopped ({self._count} samples collected).")

    def _emit_frame(self, frame_bgr: np.ndarray) -> None:
        """Helper to safely convert OpenCV BGR to QImage and emit."""
        try:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # Emit a copy to avoid memory corruption if the original array is overwritten
            self.frame_ready.emit(q_img.copy())
        except Exception as e:
            logger.error(f"EnrollWorker: Frame conversion failed - {e}")

    def stop(self) -> None:
        """Gracefully stops the worker thread."""
        self._running = False
        self.wait(2000)