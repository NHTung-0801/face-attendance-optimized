"""
src/gui/threads/ai_worker.py
AIWorker — QThread đọc frame từ CameraStream, chạy AntiSpoofing + FaceRecognizer,
emit kết quả về main thread qua Signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from src.core.anti_spoofing import AntiSpoofing, SpoofResult
from src.core.camera_stream import CameraStream
from src.core.face_recognizer import FaceRecognizer, RecognizeResult
from src.utils.config import (
    COLOR_RECOGNIZED,
    COLOR_REAL,
    COLOR_SPOOF,
    COLOR_UNKNOWN,
    DETECT_INTERVAL,
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    FONT_SCALE,
    FONT_THICKNESS,
    RECOGNITION_INTERVAL,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Payload emit ra ngoài ────────────────────────────────────────────────────
@dataclass
class AIResult:
    emp_code:   Optional[str]     # None nếu chưa nhận diện được
    similarity: float             # 0.0 nếu chưa nhận diện
    is_real:    bool              # Kết quả anti-spoofing
    spoof_conf: float
    bbox:       Optional[tuple[int, int, int, int]]


class AIWorker(QThread):
    """
    Signals:
        frame_ready(np.ndarray)     — frame BGR đã vẽ annotations, dùng để hiển thị
        result_ready(list[AIResult])— kết quả nhận diện mỗi lần inference
        error_occurred(str)         — thông báo lỗi runtime
    """

    frame_ready:    Signal = Signal(np.ndarray)
    result_ready:   Signal = Signal(list)
    error_occurred: Signal = Signal(str)

    def __init__(
        self,
        camera: CameraStream,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._camera   = camera
        self._spoofing = AntiSpoofing.instance()
        self._recognizer = FaceRecognizer.instance()
        self._running  = False
        self._tick     = 0          # Đếm frame để tách tần suất inference

    # ── Vòng lặp chính ──────────────────────────────────────────────────────
    def run(self) -> None:
        self._running = True
        logger.info("AIWorker: bắt đầu.")

        # Cache kết quả giữa các tick để vẽ liên tục dù không inference mỗi frame
        cached_spoof_results:  list[SpoofResult]    = []
        cached_recog_results:  list[RecognizeResult] = []

        while self._running:
            frame = self._camera.get_frame(timeout=0.05)
            if frame is None:
                continue

            self._tick += 1
            run_spoof  = (self._tick % DETECT_INTERVAL)   == 0
            run_recog  = (self._tick % RECOGNITION_INTERVAL) == 0

            try:
                # ── Anti-spoofing ────────────────────────────────────────
                if run_spoof:
                    cached_spoof_results = self._spoofing.detect_spoof(frame)

                # ── Nhận diện khuôn mặt ──────────────────────────────────
                if run_recog:
                    cached_recog_results = self._recognizer.recognize(frame)

                # ── Vẽ annotations ────────────────────────────────────────
                annotated = _draw_results(
                    frame.copy(),
                    cached_spoof_results,
                    cached_recog_results,
                )

                # ── Resize cho display ────────────────────────────────────
                display = cv2.resize(
                    annotated, (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )

                # ── Emit ──────────────────────────────────────────────────
                self.frame_ready.emit(display)

                if run_recog or run_spoof:
                    ai_results = _merge_results(
                        cached_spoof_results, cached_recog_results
                    )
                    self.result_ready.emit(ai_results)

            except Exception as exc:  # noqa: BLE001
                logger.exception("AIWorker: lỗi trong vòng lặp")
                self.error_occurred.emit(str(exc))

        logger.info("AIWorker: đã dừng.")

    def stop(self) -> None:
        """Dừng sạch vòng lặp và chờ thread kết thúc."""
        self._running = False
        self.wait(3000)   # Chờ tối đa 3 giây


# ── Helpers vẽ bounding box ─────────────────────────────────────────────────
def _draw_results(
    frame: np.ndarray,
    spoof_results: list[SpoofResult],
    recog_results: list[RecognizeResult],
) -> np.ndarray:
    """
    Vẽ bbox + label lên frame.
    Ưu tiên kết quả nhận diện nếu bbox gần nhau; fallback về spoof bbox.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Vẽ kết quả anti-spoofing
    for sr in spoof_results:
        if sr.bbox is None:
            continue
        x1, y1, x2, y2 = sr.bbox
        color = COLOR_REAL if sr.is_real else COLOR_SPOOF
        label = f"{sr.label} {sr.confidence:.0%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        _put_label(frame, label, x1, y1, color, font)

    # Vẽ kết quả nhận diện (đè lên nếu có bbox trùng)
    for rr in recog_results:
        if rr.bbox is None:
            continue
        x1, y1, x2, y2 = rr.bbox
        color = COLOR_RECOGNIZED if rr.emp_code != "UNKNOWN" else COLOR_UNKNOWN
        label = f"{rr.emp_code}  {rr.similarity:.0%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        _put_label(frame, label, x1, y2 + 20, color, font)   # Dưới bbox

    return frame


def _put_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font: int,
) -> None:
    """Vẽ text có nền đen để dễ đọc trên mọi nền."""
    (tw, th), _ = cv2.getTextSize(text, font, FONT_SCALE, FONT_THICKNESS)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 2, y - 2), font, FONT_SCALE, color, FONT_THICKNESS)


def _merge_results(
    spoof_results: list[SpoofResult],
    recog_results: list[RecognizeResult],
) -> list[AIResult]:
    """Gộp spoof + recog thành list AIResult để emit ra ngoài."""
    # Tạo lookup bbox spoof → result
    results: list[AIResult] = []

    if recog_results:
        for rr in recog_results:
            # Tìm spoof result gần nhất (cùng khuôn mặt)
            matched_spoof = _find_matching_spoof(rr.bbox, spoof_results)
            results.append(AIResult(
                emp_code   = rr.emp_code,
                similarity = rr.similarity,
                is_real    = matched_spoof.is_real    if matched_spoof else True,
                spoof_conf = matched_spoof.confidence if matched_spoof else 0.0,
                bbox       = rr.bbox,
            ))
    elif spoof_results:
        for sr in spoof_results:
            results.append(AIResult(
                emp_code   = None,
                similarity = 0.0,
                is_real    = sr.is_real,
                spoof_conf = sr.confidence,
                bbox       = sr.bbox,
            ))

    return results


def _find_matching_spoof(
    bbox: Optional[tuple[int, int, int, int]],
    spoof_results: list[SpoofResult],
    iou_threshold: float = 0.3,
) -> Optional[SpoofResult]:
    """Tìm SpoofResult có IoU cao nhất với bbox nhận diện."""
    if bbox is None or not spoof_results:
        return None
    best, best_iou = None, 0.0
    for sr in spoof_results:
        if sr.bbox is None:
            continue
        iou = _calc_iou(bbox, sr.bbox)
        if iou > best_iou:
            best_iou, best = iou, sr
    return best if best_iou >= iou_threshold else None


def _calc_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union
