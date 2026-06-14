"""
src/gui/threads/ai_worker.py
AIWorker — Thread chạy pipeline AI (Anti-spoofing + Recognition).
Sử dụng cache để tối ưu CPU.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from src.core.camera_stream import CameraStream
from src.core.anti_spoofing import AntiSpoofing
from src.core.face_recognizer import FaceRecognizer
from src.utils.config import (
    DETECT_INTERVAL, RECOGNITION_INTERVAL,
    COLOR_REAL, COLOR_SPOOF, COLOR_UNKNOWN, COLOR_RECOGNIZED,
    FACE_DET_SCORE_THRESHOLD, SPOOFING_CONFIDENCE_THRESHOLD
)
from src.utils.helpers import draw_bounding_box
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class AIResult:
    emp_code: str
    similarity: float
    is_real: bool
    spoof_conf: float
    bbox: tuple[int, int, int, int]

class AIWorker(QThread):
    frame_ready = Signal(np.ndarray)       # Frame đã vẽ bbox
    result_ready = Signal(list)            # List[AIResult]

    def __init__(self, camera: CameraStream) -> None:
        super().__init__()
        self._camera = camera
        self._spoofing = AntiSpoofing.instance()
        self._recognizer = FaceRecognizer.instance()
        self._running = False
        
        # Cache kết quả để dùng cho các frame bỏ qua
        self._cached_results: list[AIResult] = []
        self._tick = 0

    def run(self) -> None:
        self._running = True
        logger.info("AIWorker: bắt đầu.")

        while self._running:
            frame = self._camera.get_frame(timeout=0.05)
            if frame is None:
                continue

            self._tick += 1
            annotated = frame.copy()

            # Pipeline xử lý AI
            if self._tick % DETECT_INTERVAL == 0:
                self._cached_results = self._process_pipeline(frame)

            # Vẽ kết quả (dùng cache nếu frame bị skip)
            for res in self._cached_results:
                color = COLOR_REAL if res.is_real else COLOR_SPOOF
                if res.is_real and res.emp_code != "UNKNOWN":
                    color = COLOR_RECOGNIZED
                elif res.emp_code == "UNKNOWN":
                    color = COLOR_UNKNOWN

                label = f"{res.emp_code} ({res.similarity:.2f})" if res.is_real else "FAKE"
                annotated = draw_bounding_box(annotated, res.bbox, label, color)

            self.frame_ready.emit(annotated)
            self.result_ready.emit(self._cached_results)

        logger.info("AIWorker: đã dừng.")

    def _process_pipeline(self, frame: np.ndarray) -> list[AIResult]:
        results = []
        # 1. Phát hiện khuôn mặt (dùng FaceRecognizer để lấy cả emb và bbox)
        detections = self._recognizer.get_embeddings_from_frame(frame)
        
        for emb, bbox, _ in detections:
            # 2. Kiểm tra Spoofing (Real/Fake)
            is_real, conf = self._spoofing.predict(frame, bbox)
            
            emp_code = "UNKNOWN"
            sim = 0.0
            
            # 3. Chỉ nhận diện nếu là mặt thật
            if is_real and conf >= SPOOFING_CONFIDENCE_THRESHOLD:
                if self._tick % RECOGNITION_INTERVAL == 0:
                    matches = self._recognizer.identify_face(emb)
                    if matches:
                        emp_code, sim = matches[0]

            results.append(AIResult(emp_code, sim, is_real, conf, bbox))
            
        return results

    def stop(self) -> None:
        self._running = False
        self.wait()