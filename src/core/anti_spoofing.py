"""
src/core/anti_spoofing.py
AntiSpoofing — chạy YOLO .onnx qua onnxruntime, KHÔNG dùng torch/ultralytics.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from src.utils.config import (
    ANTI_SPOOFING_MODEL_PATH,
    SPOOFING_CONFIDENCE_THRESHOLD,
    SPOOFING_INPUT_SIZE,
)


# ── Kết quả trả về ─────────────────────────────────────────────────────────
@dataclass
class SpoofResult:
    is_real:    bool
    confidence: float                          # confidence của class thắng
    label:      str                            # "Real" | "Fake"
    bbox:       Optional[tuple[int, int, int, int]]  # (x1, y1, x2, y2) pixel gốc | None


# ── Helpers ─────────────────────────────────────────────────────────────────
def _letterbox(
    img: np.ndarray,
    target: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize giữ tỉ lệ, pad phần còn lại bằng màu xám.
    Trả về (ảnh đã xử lý, scale, (pad_w, pad_h)).
    """
    h, w = img.shape[:2]
    tw, th = target
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((th, tw, 3), color, dtype=np.uint8)
    pad_x = (tw - nw) // 2
    pad_y = (th - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, scale, (pad_x, pad_y)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> list[int]:
    """NMS thuần numpy, trả về danh sách index giữ lại."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thr)[0] + 1]
    return keep


def _xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Chuyển [cx, cy, w, h] → [x1, y1, x2, y2]."""
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


# ── Class chính ─────────────────────────────────────────────────────────────
class AntiSpoofing:
    """
    Singleton.  Sử dụng:
        model = AntiSpoofing.instance()
        results = model.detect_spoof(frame_bgr)
    """

    _instance: Optional["AntiSpoofing"] = None
    _lock: threading.Lock = threading.Lock()

    # ── Singleton ──────────────────────────────────────────────────────────
    def __new__(cls) -> "AntiSpoofing":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    @classmethod
    def instance(cls) -> "AntiSpoofing":
        return cls()

    # ── Khởi tạo (lazy) ────────────────────────────────────────────────────
    def __init__(self) -> None:
        if self._initialized:
            return

        model_path = str(ANTI_SPOOFING_MODEL_PATH)
        if not ANTI_SPOOFING_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model: {model_path}\n"
                "Chạy tools/export_onnx.py để xuất model trước."
            )

        # Ưu tiên CPU EP; thêm CUDAExecutionProvider nếu có GPU
        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4   # tune theo số core CPU

        self._session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers
        )

        # Lấy metadata input/output
        inp = self._session.get_inputs()[0]
        self._input_name: str = inp.name
        # Tên output đầu tiên (YOLO thường có 1 output duy nhất)
        self._output_name: str = self._session.get_outputs()[0].name

        # Số class từ shape output: [1, num_det, 5+num_classes] hoặc [1, 5+nc, anchors]
        # Xác định tại runtime khi gọi lần đầu
        self._num_classes: Optional[int] = None

        self._conf_thr: float = SPOOFING_CONFIDENCE_THRESHOLD
        self._input_size: tuple[int, int] = SPOOFING_INPUT_SIZE  # (W, H)
        self._initialized = True

    # ── Tiền xử lý ─────────────────────────────────────────────────────────
    def _preprocess(
        self, frame_bgr: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        BGR → letterbox → RGB → float32 [0,1] → NCHW
        Trả về (blob, scale, (pad_x, pad_y)).
        """
        img, scale, pad = _letterbox(frame_bgr, self._input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        blob = np.transpose(img, (2, 0, 1))[np.newaxis, ...]   # (1, 3, H, W)
        blob = np.ascontiguousarray(blob)
        return blob, scale, pad

    # ── Hậu xử lý ──────────────────────────────────────────────────────────
    def _postprocess(
        self,
        output: np.ndarray,
        orig_shape: tuple[int, int],
        scale: float,
        pad: tuple[int, int],
    ) -> list[SpoofResult]:
        """
        output shape: (1, num_det, 5 + num_classes)  [YOLOv8 style]
                   hoặc (1, 5+nc, anchors)            [YOLOv5 style]
        Tự động phát hiện layout và xử lý.
        """
        orig_h, orig_w = orig_shape
        pad_x, pad_y = pad

        raw = output[0]  # (num_det, 5+nc) hoặc (5+nc, anchors)

        # YOLOv5/v8 detection đầu ra thường là (N, 5+nc)
        # Nếu chiều đầu nhỏ hơn chiều sau → cần transpose (YOLOv8 mới)
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T   # (anchors, 5+nc)

        if self._num_classes is None:
            self._num_classes = raw.shape[1] - 5

        nc = self._num_classes
        # Tách: cx cy w h obj_conf  class_scores...
        box_raw      = raw[:, :4]          # (N, 4) cx cy w h
        obj_conf     = raw[:, 4]           # (N,)
        class_scores = raw[:, 5 : 5 + nc] # (N, nc)

        # Confidence tổng hợp
        class_ids    = class_scores.argmax(axis=1)                     # (N,)
        class_confs  = class_scores[np.arange(len(class_ids)), class_ids]
        scores       = obj_conf * class_confs

        # Lọc theo threshold
        mask = scores >= self._conf_thr
        if not mask.any():
            return []

        boxes_filt  = box_raw[mask]
        scores_filt = scores[mask]
        ids_filt    = class_ids[mask]

        # cx cy w h → x1 y1 x2 y2  (trong tọa độ letterbox)
        boxes_xyxy = _xywh2xyxy(boxes_filt)

        # NMS
        keep = _nms(boxes_xyxy, scores_filt, iou_thr=0.45)

        results: list[SpoofResult] = []
        for i in keep:
            x1, y1, x2, y2 = boxes_xyxy[i]
            conf    = float(scores_filt[i])
            cls_id  = int(ids_filt[i])

            # Chuyển tọa độ letterbox → tọa độ ảnh gốc
            x1 = int((x1 - pad_x) / scale)
            y1 = int((y1 - pad_y) / scale)
            x2 = int((x2 - pad_x) / scale)
            y2 = int((y2 - pad_y) / scale)

            # Clamp trong biên ảnh
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))

            # Class 0 = Real, Class 1 = Fake (theo cách label dataset chuẩn)
            # Nếu dataset của bạn ngược lại, đổi điều kiện này.
            is_real = cls_id == 0
            label   = "Real" if is_real else "Fake"

            results.append(
                SpoofResult(
                    is_real=is_real,
                    confidence=conf,
                    label=label,
                    bbox=(x1, y1, x2, y2),
                )
            )

        return results

    # ── API công khai ───────────────────────────────────────────────────────
    def detect_spoof(self, frame_bgr: np.ndarray) -> list[SpoofResult]:
        """
        Nhận khung hình OpenCV (BGR), trả về danh sách SpoofResult.
        Mỗi phần tử ứng với một khuôn mặt được phát hiện.
        Trả về list rỗng nếu không phát hiện được khuôn mặt nào.

        Ví dụ:
            results = model.detect_spoof(frame)
            for r in results:
                print(r.label, r.confidence, r.bbox)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        orig_shape = frame_bgr.shape[:2]   # (H, W)
        blob, scale, pad = self._preprocess(frame_bgr)

        output = self._session.run(
            [self._output_name], {self._input_name: blob}
        )[0]

        return self._postprocess(output, orig_shape, scale, pad)

    def is_model_loaded(self) -> bool:
        return self._initialized and self._session is not None
