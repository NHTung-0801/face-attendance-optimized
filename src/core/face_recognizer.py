"""
src/core/face_recognizer.py
FaceRecognizer — InsightFace embedding + FAISS IndexFlatIP (cosine similarity).
Singleton, thread-safe.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import faiss
import numpy as np
from insightface.app import FaceAnalysis

from src.utils.config import (
    FACE_DET_SCORE_THRESHOLD,
    FACE_EMBEDDING_DIM,
    FACE_MIN_SIZE,
    FACE_RECOGNITION_THRESHOLD,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    INSIGHTFACE_MODEL_DIR,
)


# ── Kết quả nhận diện ──────────────────────────────────────────────────────
@dataclass
class RecognizeResult:
    emp_code:   str
    similarity: float                           # Độ giống nhau (Cosine Similarity)
    bbox:       Optional[tuple[int, int, int, int]]  # (x1, y1, x2, y2)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """Chuẩn hoá L2 vector về unit-length để dùng IndexFlatIP = cosine sim."""
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)


def _load_metadata(path: Path) -> dict[int, str]:
    """
    Đọc file JSON: { "0": "EMP001", "1": "EMP002", ... }
    Trả về dict {faiss_row_id: emp_code}.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, str] = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _save_metadata(meta: dict[int, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in meta.items()}, f, ensure_ascii=False, indent=2)


# ── Class chính ─────────────────────────────────────────────────────────────
class FaceRecognizer:
    """
    Singleton.  Sử dụng:
        fr = FaceRecognizer.instance()
        results = fr.recognize(frame_bgr)
    """

    _instance: Optional["FaceRecognizer"] = None
    _lock: threading.Lock = threading.Lock()

    # ── Singleton ──────────────────────────────────────────────────────────
    def __new__(cls) -> "FaceRecognizer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    @classmethod
    def instance(cls) -> "FaceRecognizer":
        return cls()

    # ── Khởi tạo ───────────────────────────────────────────────────────────
    def __init__(self) -> None:
        if self._initialized:
            return

        # InsightFace — buffalo_s nhanh hơn buffalo_l, phù hợp CPU
        self._app = FaceAnalysis(
            name="buffalo_s",
            root=str(INSIGHTFACE_MODEL_DIR),
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=(640, 640))

        # FAISS — IndexFlatIP (inner product) với vector đã L2-normalize = cosine
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(FACE_EMBEDDING_DIM)

        # Metadata: { faiss_row_id -> emp_code }
        self._meta: dict[int, str] = {}

        # Mutex riêng cho index để đọc/ghi đồng thời an toàn
        self._index_lock = threading.Lock()

        # Load index và metadata từ disk nếu đã có
        self.load_index()

        self._initialized = True

    # ═══════════════════════════════════════════════════════════════════════
    # EMBEDDING
    # ═══════════════════════════════════════════════════════════════════════

    def get_embedding(self, face_img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Nhận ảnh khuôn mặt đã crop (BGR).
        Trả về vector 512-dim đã L2-normalize, hoặc None nếu không detect được.
        """
        if face_img_bgr is None or face_img_bgr.size == 0:
            return None

        faces = self._app.get(face_img_bgr)
        if not faces:
            return None

        # Lấy khuôn mặt có det_score cao nhất
        face = max(faces, key=lambda f: f.det_score)
        if face.det_score < FACE_DET_SCORE_THRESHOLD:
            return None

        return _l2_normalize(face.embedding.astype(np.float32))

    def get_embeddings_from_frame(
        self, frame_bgr: np.ndarray
    ) -> list[tuple[np.ndarray, tuple[int, int, int, int], float]]:
        """
        Phát hiện TẤT CẢ khuôn mặt trong frame đầy đủ.
        Trả về list[(embedding, bbox, det_score)] đã lọc theo ngưỡng và kích thước.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        faces = self._app.get(frame_bgr)
        results = []
        for face in faces:
            if face.det_score < FACE_DET_SCORE_THRESHOLD:
                continue
            x1, y1, x2, y2 = face.bbox.astype(int)
            w, h = x2 - x1, y2 - y1
            if min(w, h) < FACE_MIN_SIZE:
                continue
            emb = _l2_normalize(face.embedding.astype(np.float32))
            results.append((emb, (x1, y1, x2, y2), float(face.det_score)))
        return results

    # ═══════════════════════════════════════════════════════════════════════
    # FAISS INDEX MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════

    def load_index(self) -> bool:
        """Load FAISS index + metadata từ disk. Trả về True nếu thành công."""
        with self._index_lock:
            if FAISS_INDEX_PATH.exists() and FAISS_METADATA_PATH.exists():
                try:
                    self._index = faiss.read_index(str(FAISS_INDEX_PATH))
                    self._meta  = _load_metadata(FAISS_METADATA_PATH)
                    return True
                except Exception as e:
                    print(f"[FaceRecognizer] Load index thất bại: {e}. Reset index mới.")
            # Khởi tạo index rỗng nếu không load được
            self._index = faiss.IndexFlatIP(FACE_EMBEDDING_DIM)
            self._meta  = {}
            return False

    def save_index(self) -> None:
        """Ghi FAISS index + metadata xuống disk (thread-safe)."""
        with self._index_lock:
            FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(FAISS_INDEX_PATH))
            _save_metadata(self._meta, FAISS_METADATA_PATH)

    def _rebuild_index(self, embeddings: np.ndarray, meta: dict[int, str]) -> None:
        """
        Build lại index từ đầu. Gọi nội bộ khi xóa nhân viên.
        Caller phải giữ self._index_lock trước khi gọi.
        """
        new_index = faiss.IndexFlatIP(FACE_EMBEDDING_DIM)
        if embeddings.shape[0] > 0:
            new_index.add(embeddings)
        self._index = new_index
        self._meta  = meta

    # ═══════════════════════════════════════════════════════════════════════
    # ADD / REMOVE
    # ═══════════════════════════════════════════════════════════════════════

    def add_face(self, emp_code: str, embeddings: list[np.ndarray]) -> bool:
        """
        Thêm một hoặc nhiều embedding của cùng một nhân viên vào FAISS.
        Tự động save index sau khi thêm.
        """
        if not embeddings:
            return False

        vectors = np.vstack([e.reshape(1, -1) for e in embeddings]).astype(np.float32)

        with self._index_lock:
            start_id = self._index.ntotal
            self._index.add(vectors)
            for i in range(len(embeddings)):
                self._meta[start_id + i] = emp_code

        self.save_index()
        return True

    def remove_face(self, emp_code: str) -> bool:
        """
        Xóa toàn bộ embedding của nhân viên và rebuild index.
        IndexFlatIP không hỗ trợ xóa trực tiếp → phải rebuild.
        """
        with self._index_lock:
            # Tìm tất cả row_id thuộc emp_code
            keep_ids = [rid for rid, code in self._meta.items() if code != emp_code]
            removed  = len(self._meta) - len(keep_ids)

            if removed == 0:
                return False  # Không tìm thấy

            if not keep_ids:
                # Không còn ai → index rỗng
                self._index = faiss.IndexFlatIP(FACE_EMBEDDING_DIM)
                self._meta  = {}
            else:
                # Lấy lại tất cả vector cần giữ từ index hiện tại
                all_vectors = faiss.rev_swig_ptr(
                    self._index.get_xb(), self._index.ntotal * FACE_EMBEDDING_DIM
                ).reshape(self._index.ntotal, FACE_EMBEDDING_DIM)

                keep_vectors = all_vectors[keep_ids].copy()
                new_meta = {new_id: self._meta[old_id]
                            for new_id, old_id in enumerate(keep_ids)}
                self._rebuild_index(keep_vectors, new_meta)

        self.save_index()
        return True

    # ═══════════════════════════════════════════════════════════════════════
    # NHẬN DIỆN
    # ═══════════════════════════════════════════════════════════════════════

    def identify_face(
        self, query_embedding: np.ndarray, top_k: int = 1
    ) -> list[tuple[str, float]]:
        """
        Tìm kiếm trong FAISS sử dụng Cosine Similarity nguyên bản.
        """
        with self._index_lock:
            if self._index.ntotal == 0:
                return []

            query = query_embedding.reshape(1, -1).astype(np.float32)
            k = min(top_k, self._index.ntotal)
            similarities, indices = self._index.search(query, k)

        results: list[tuple[str, float]] = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0:
                continue
            
            # [FIX LỚP 1] Lấy nguyên bản Cosine Similarity (không chia 2)
            score = float(sim)
            
            # Lọc sơ bộ những kết quả quá tệ
            if score < FACE_RECOGNITION_THRESHOLD:
                continue
                
            emp_code = self._meta.get(int(idx), "UNKNOWN")
            results.append((emp_code, score))

        return results

    def recognize(
        self, frame_bgr: np.ndarray, top_k: int = 1
    ) -> list[RecognizeResult]:
        """
        Pipeline đầy đủ: detect khuôn mặt trong frame → trích embedding → nhận diện.
        Đã bổ sung chốt chặn FACE_RECOGNITION_THRESHOLD để chống nhận diện nhầm (Bảo mật kép).
        """
        detections = self.get_embeddings_from_frame(frame_bgr)
        output: list[RecognizeResult] = []

        for emb, bbox, _det_score in detections:
            matches = self.identify_face(emb, top_k=top_k)
            
            if not matches:
                # Không tìm thấy ai trong DB (hoặc đã bị lọc ở bước trước)
                output.append(
                    RecognizeResult(
                        emp_code="UNKNOWN",
                        similarity=0.0,
                        bbox=bbox,
                    )
                )
                continue
            
            # FAISS trả về kết quả giống nhất (đứng đầu danh sách)
            emp_code, similarity = matches[0]
            
            # 🔒 CHỐT CHẶN (Bảo mật lớp 2): Chỉ công nhận nếu độ tin cậy vượt ngưỡng an toàn
            if similarity >= FACE_RECOGNITION_THRESHOLD:
                output.append(
                    RecognizeResult(
                        emp_code=emp_code,
                        similarity=similarity,
                        bbox=bbox,
                    )
                )
            else:
                # Đánh rớt kết quả vì độ tin cậy quá thấp (coi là người lạ)
                output.append(
                    RecognizeResult(
                        emp_code="UNKNOWN",
                        similarity=similarity,
                        bbox=bbox,
                    )
                )

        return output

    # ── Utils ───────────────────────────────────────────────────────────────
    @property
    def total_faces(self) -> int:
        """Tổng số vector đang lưu trong FAISS."""
        return self._index.ntotal

    @property
    def registered_employees(self) -> list[str]:
        """Danh sách emp_code đã đăng ký (không trùng)."""
        return list(set(self._meta.values()))

    def is_ready(self) -> bool:
        return self._initialized and self._index is not None