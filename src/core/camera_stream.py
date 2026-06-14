"""
src/core/camera_stream.py
CameraStream — đọc webcam trong thread riêng, drop frame cũ qua Queue(maxsize=1).
AI thread gọi get_frame() bất kỳ lúc nào mà không bị block.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

from src.utils.config import (
    CAMERA_BUFFER,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MAX_RECONNECT = 3          # Số lần thử kết nối lại
_RECONNECT_DELAY = 2.0      # Giây chờ giữa các lần reconnect
_FRAME_INTERVAL = 1.0 / CAMERA_FPS   # Thời gian tối thiểu giữa 2 frame


class CameraStream:
    """
    Đọc webcam liên tục trong một thread nền.
    Queue size = 1: luôn giữ frame MỚI NHẤT, tự drop frame cũ.

    Sử dụng:
        cam = CameraStream()
        cam.start()

        while True:
            frame = cam.get_frame()
            if frame is not None:
                cv2.imshow("preview", frame)
            if cv2.waitKey(1) == ord("q"):
                break

        cam.stop()
    """

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
    ) -> None:
        self._index  = camera_index
        self._width  = width
        self._height = height
        self._fps    = fps

        # Queue size=1 → put_nowait sẽ replace frame cũ nếu queue đầy
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)

        self._cap:    Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None

        self._running   = threading.Event()   # Set = đang chạy
        self._connected = threading.Event()   # Set = camera đang mở thành công

        # Thống kê (tuỳ chọn, hữu ích khi debug)
        self._frame_count: int = 0
        self._drop_count:  int = 0

    # ── Khởi tạo camera ─────────────────────────────────────────────────────
    def _open_camera(self) -> bool:
        """Mở VideoCapture và áp cấu hình. Trả về True nếu thành công."""
        cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW if _is_windows() else cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS,          self._fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   CAMERA_BUFFER)  # Giảm buffer nội bộ của OpenCV

        self._cap = cap
        return True

    def _release_camera(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self._cap = None
        self._connected.clear()

    # ── Thread đọc camera ───────────────────────────────────────────────────
    def _capture_loop(self) -> None:
        """
        Vòng lặp chạy trong thread nền:
        - Đọc frame từ webcam liên tục.
        - Đẩy frame mới nhất vào Queue, drop frame cũ nếu Queue đầy.
        - Tự reconnect tối đa _MAX_RECONNECT lần nếu mất tín hiệu.
        """
        reconnect_attempts = 0

        while self._running.is_set():
            # ── Kết nối (hoặc kết nối lại) ──────────────────────────────
            if self._cap is None or not self._cap.isOpened():
                if reconnect_attempts >= _MAX_RECONNECT:
                    logger.error(
                        "Camera %d: vượt quá %d lần reconnect. Dừng thread.",
                        self._index, _MAX_RECONNECT,
                    )
                    self._running.clear()
                    break

                logger.warning(
                    "Camera %d: đang kết nối lại (lần %d/%d)…",
                    self._index, reconnect_attempts + 1, _MAX_RECONNECT,
                )
                self._release_camera()
                time.sleep(_RECONNECT_DELAY)

                if self._open_camera():
                    logger.info("Camera %d: kết nối thành công.", self._index)
                    self._connected.set()
                    reconnect_attempts = 0
                else:
                    reconnect_attempts += 1
                    continue

            # ── Đọc frame ────────────────────────────────────────────────
            t0 = time.monotonic()
            ret, frame = self._cap.read()

            if not ret or frame is None:
                logger.warning("Camera %d: read() thất bại, thử reconnect…", self._index)
                self._connected.clear()
                self._release_camera()
                reconnect_attempts += 1
                continue

            # Reset bộ đếm reconnect khi đọc thành công
            reconnect_attempts = 0
            self._frame_count += 1

            # ── Đẩy vào Queue (drop frame cũ nếu đầy) ───────────────────
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._queue.get_nowait()   # Bỏ frame cũ
                except queue.Empty:
                    pass
                self._queue.put_nowait(frame)
                self._drop_count += 1

            # ── Giữ đúng FPS (tránh spin 100% CPU) ──────────────────────
            elapsed = time.monotonic() - t0
            sleep_t = _FRAME_INTERVAL - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        self._release_camera()
        logger.info("Camera %d: capture thread đã dừng.", self._index)

    # ── API công khai ────────────────────────────────────────────────────────
    def start(self) -> bool:
        """
        Khởi động thread đọc camera.
        Trả về True nếu camera mở được trong vòng 5 giây.
        """
        if self._running.is_set():
            logger.warning("CameraStream đã đang chạy.")
            return True

        if not self._open_camera():
            logger.error("Không thể mở camera index=%d.", self._index)
            return False

        self._connected.set()
        self._running.set()

        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"CameraStream-{self._index}",
            daemon=True,   # Tự tắt khi main process thoát
        )
        self._thread.start()
        logger.info(
            "CameraStream khởi động: index=%d, %dx%d @ %dFPS",
            self._index, self._width, self._height, self._fps,
        )
        return True

    def stop(self) -> None:
        """Dừng thread và giải phóng camera."""
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._release_camera()
        # Flush queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        logger.info(
            "CameraStream dừng: %d frames đọc, %d frames dropped.",
            self._frame_count, self._drop_count,
        )

    def get_frame(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        """
        Lấy frame mới nhất từ Queue.

        Args:
            timeout: Giây chờ tối đa (mặc định 50ms).
                     Đặt 0 để non-blocking hoàn toàn.

        Returns:
            np.ndarray (BGR) hoặc None nếu chưa có frame.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_running(self) -> bool:
        return self._running.is_set()

    def is_connected(self) -> bool:
        return self._connected.is_set()

    @property
    def stats(self) -> dict[str, int]:
        return {
            "frames_captured": self._frame_count,
            "frames_dropped":  self._drop_count,
        }

    # ── Context manager ─────────────────────────────────────────────────────
    def __enter__(self) -> "CameraStream":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


# ── Util ────────────────────────────────────────────────────────────────────
def _is_windows() -> bool:
    import sys
    return sys.platform.startswith("win")
