"""
src/core/anti_spoofing.py
AntiSpoofing — chạy YOLOv8 Detection .onnx qua onnxruntime.
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

@dataclass
class SpoofResult:
    is_real:    bool
    confidence: float                          
    label:      str                            
    bbox:       Optional[tuple[int, int, int, int]]  

def _letterbox(img: np.ndarray, target: tuple[int, int], color: tuple[int, int, int] = (114, 114, 114)) -> tuple[np.ndarray, float, tuple[int, int]]:
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
    if len(boxes) == 0: return []
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
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out

class AntiSpoofing:
    _instance: Optional["AntiSpoofing"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "AntiSpoofing":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    @classmethod
    def instance(cls) -> "AntiSpoofing":
        return cls()

    def __init__(self) -> None:
        if self._initialized: return
        model_path = str(ANTI_SPOOFING_MODEL_PATH)
        if not ANTI_SPOOFING_MODEL_PATH.exists():
            raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self._session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._conf_thr = SPOOFING_CONFIDENCE_THRESHOLD
        self._input_size = SPOOFING_INPUT_SIZE  
        self._initialized = True

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        img, scale, pad = _letterbox(frame_bgr, self._input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(img, (2, 0, 1))[np.newaxis, ...]   
        return np.ascontiguousarray(blob), scale, pad

    def _postprocess(self, output: np.ndarray, orig_shape: tuple[int, int], scale: float, pad: tuple[int, int]) -> list[SpoofResult]:
        """Giải mã Output của YOLOv8 Detection (Shape: [1, 4+nc, 8400]). YOLOv8 KHÔNG CÓ objectness confidence."""
        orig_h, orig_w = orig_shape
        pad_x, pad_y = pad

        raw = output[0]  
        # YOLOv8 trả về (4+nc, 8400) -> Transpose thành (8400, 4+nc)
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T  

        nc = raw.shape[1] - 4
        box_raw = raw[:, :4]
        class_scores = raw[:, 4: 4 + nc]

        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]

        mask = scores >= self._conf_thr
        if not mask.any(): return []

        boxes_filt = box_raw[mask]
        scores_filt = scores[mask]
        ids_filt = class_ids[mask]
        boxes_xyxy = _xywh2xyxy(boxes_filt)
        keep = _nms(boxes_xyxy, scores_filt, iou_thr=0.45)

        results: list[SpoofResult] = []
        for i in keep:
            x1, y1, x2, y2 = boxes_xyxy[i]
            x1 = max(0, min(int((x1 - pad_x) / scale), orig_w - 1))
            y1 = max(0, min(int((y1 - pad_y) / scale), orig_h - 1))
            x2 = max(0, min(int((x2 - pad_x) / scale), orig_w - 1))
            y2 = max(0, min(int((y2 - pad_y) / scale), orig_h - 1))

            is_real = int(ids_filt[i]) == 0
            results.append(
                SpoofResult(
                    is_real=is_real, 
                    confidence=float(scores_filt[i]), 
                    label="Real" if is_real else "Fake", 
                    bbox=(x1, y1, x2, y2)
                )
            )
        return results

    def detect_spoof(self, frame_bgr: np.ndarray) -> list[SpoofResult]:
        if frame_bgr is None or frame_bgr.size == 0: return []
        orig_shape = frame_bgr.shape[:2]
        blob, scale, pad = self._preprocess(frame_bgr)
        output = self._session.run([self._output_name], {self._input_name: blob})[0]
        return self._postprocess(output, orig_shape, scale, pad)