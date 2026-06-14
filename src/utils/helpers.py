"""
src/utils/helpers.py
Hàm tiện ích dùng chung: vẽ bounding box, tính FPS, xử lý chuỗi, v.v.
"""

from __future__ import annotations

import time
import unicodedata
from typing import Optional

import cv2
import numpy as np


# ── Bounding box ─────────────────────────────────────────────────────────────

def draw_bounding_box(
    img:        np.ndarray,
    bbox:       tuple[int, int, int, int],
    label:      str                         = "",
    color:      tuple[int, int, int]        = (0, 255, 0),
    thickness:  int                         = 2,
    font_scale: float                       = 0.6,
    show_conf:  Optional[float]             = None,
) -> np.ndarray:
    """
    Vẽ bounding box + label có nền lên ảnh (in-place).

    Args:
        img:       Frame BGR (numpy array).
        bbox:      (x1, y1, x2, y2) pixel coordinates.
        label:     Tên hiển thị (vd: "NV001 - Nguyễn A").
        color:     Màu BGR của khung (mặc định xanh lá).
        thickness: Độ dày đường viền.
        font_scale: Cỡ chữ.
        show_conf: Nếu không None, append "  xx%" vào label.

    Returns:
        img đã được vẽ (cùng object, không copy).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Clamp trong biên ảnh
    h, w = img.shape[:2]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)

    # Khung chính
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Góc bo (accent 4 góc dài hơn)
    corner_len = max(12, (x2 - x1) // 8)
    ct = thickness + 1
    for cx, cy, dx, dy in [
        (x1, y1,  1,  1),
        (x2, y1, -1,  1),
        (x1, y2,  1, -1),
        (x2, y2, -1, -1),
    ]:
        cv2.line(img, (cx, cy), (cx + dx * corner_len, cy), color, ct)
        cv2.line(img, (cx, cy), (cx, cy + dy * corner_len), color, ct)

    # Label
    if label:
        display = label if show_conf is None else f"{label}  {show_conf:.0%}"
        _put_label_bg(img, display, x1, y1, color, font_scale, thickness)

    return img


def draw_spoof_overlay(
    img:      np.ndarray,
    bbox:     tuple[int, int, int, int],
    is_real:  bool,
    conf:     float,
) -> np.ndarray:
    """
    Overlay chuyên dụng cho anti-spoofing:
    - Xanh lá + "REAL" khi hợp lệ.
    - Đỏ + "FAKE" + vạch chéo khi giả mạo.
    """
    from src.utils.config import COLOR_REAL, COLOR_SPOOF

    color = COLOR_REAL if is_real else COLOR_SPOOF
    label = f"{'REAL' if is_real else 'FAKE ⚠'}  {conf:.0%}"
    draw_bounding_box(img, bbox, label, color)

    if not is_real:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Vạch chéo cảnh báo
        overlay = img.copy()
        cv2.line(overlay, (x1, y1), (x2, y2), COLOR_SPOOF, 2)
        cv2.line(overlay, (x2, y1), (x1, y2), COLOR_SPOOF, 2)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    return img


# ── FPS ──────────────────────────────────────────────────────────────────────

def calculate_fps(prev_time: float) -> tuple[float, float]:
    """
    Tính FPS từ timestamp lần trước.

    Args:
        prev_time: time.monotonic() của frame trước.

    Returns:
        (fps, current_time) — current_time để truyền vào lần gọi tiếp theo.

    Ví dụ:
        t = time.monotonic()
        while True:
            fps, t = calculate_fps(t)
            draw_fps(frame, fps)
    """
    now = time.monotonic()
    dt  = now - prev_time
    fps = 1.0 / dt if dt > 0 else 0.0
    return fps, now


def draw_fps(img: np.ndarray, fps: float, pos: tuple[int, int] = (12, 32)) -> np.ndarray:
    """Vẽ FPS counter lên góc trái trên của frame."""
    text  = f"FPS: {fps:.1f}"
    color = (0, 255, 0) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
    _put_label_bg(img, text, pos[0], pos[1], color, font_scale=0.55, thickness=1)
    return img


# ── String utils ─────────────────────────────────────────────────────────────

def normalize_string(text: str) -> str:
    """
    Chuẩn hóa chuỗi: strip, lower, bỏ dấu tiếng Việt.
    Dùng để so sánh tên/mã không phân biệt dấu.
    """
    text = text.strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text


def truncate_label(text: str, max_len: int = 20) -> str:
    """Rút gọn chuỗi dài để không tràn ra ngoài bbox."""
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


# ── Image utils ───────────────────────────────────────────────────────────────

def crop_face(
    frame: np.ndarray,
    bbox:  tuple[int, int, int, int],
    margin: int = 20,
) -> Optional[np.ndarray]:
    """
    Crop khuôn mặt từ frame với margin xung quanh.
    Trả về None nếu bbox không hợp lệ.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def resize_keep_ratio(
    img: np.ndarray,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    """Resize ảnh giữ tỉ lệ trong giới hạn (max_width, max_height)."""
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)   # Không phóng to
    if scale == 1.0:
        return img
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def bgr_to_qimage(frame_bgr: np.ndarray):
    """
    Chuyển BGR numpy array → QImage để hiển thị trong PySide6.
    Import PySide6 lazy để helpers.py dùng được cả ngoài GUI context.
    """
    from PySide6.QtGui import QImage
    h, w, ch = frame_bgr.shape
    rgb = frame_bgr[:, :, ::-1].copy()
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)


# ── Internal helper ───────────────────────────────────────────────────────────

def _put_label_bg(
    img:        np.ndarray,
    text:       str,
    x:          int,
    y:          int,
    color:      tuple[int, int, int],
    font_scale: float = 0.6,
    thickness:  int   = 1,
) -> None:
    """Vẽ text với nền đen bán trong suốt để dễ đọc trên mọi nền."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 4

    # Đảm bảo label không bị cắt ở đỉnh frame
    label_y = max(y - th - pad * 2 - baseline, th + pad)

    # Nền đen mờ
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x - pad, label_y - th - pad),
        (x + tw + pad, label_y + baseline + pad),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Chữ
    cv2.putText(img, text, (x, label_y), font, font_scale, color, thickness, cv2.LINE_AA)
