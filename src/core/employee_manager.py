"""
src/core/employee_manager.py
EmployeeManager — Controller điều phối giữa DatabaseManager và FaceRecognizer.
Singleton, thread-safe.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.core.face_recognizer import FaceRecognizer
from src.core.anti_spoofing import AntiSpoofing
from src.database.db_manager import DatabaseManager
from src.database.models import Employee
from src.utils.logger import get_logger
from src.utils.config import FACE_RECOGNITION_THRESHOLD

logger = get_logger(__name__)


# ── Kết quả trả về ──────────────────────────────────────────────────────────
@dataclass
class EnrollResult:
    success:     bool
    emp_code:    str
    message:     str
    embeddings_added: int = 0   # Số embedding thực sự đưa vào FAISS


@dataclass
class DeleteResult:
    success:  bool
    emp_code: str
    message:  str


# ── Class chính ─────────────────────────────────────────────────────────────
class EmployeeManager:
    """
    Singleton.  Sử dụng:
        em = EmployeeManager.instance()
        result = em.enroll_employee("NV001", "Nguyễn Văn A", face_samples)
    """

    _instance: Optional["EmployeeManager"] = None
    _lock: threading.Lock = threading.Lock()

    # ── Singleton ──────────────────────────────────────────────────────────
    def __new__(cls) -> "EmployeeManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    @classmethod
    def instance(cls) -> "EmployeeManager":
        return cls()

    # ── Khởi tạo ───────────────────────────────────────────────────────────
    def __init__(self) -> None:
        if self._initialized:
            return
        self._db = DatabaseManager.instance()
        self._recognizer = FaceRecognizer.instance()
        self._spoofing = AntiSpoofing.instance()  # Kích hoạt module Anti-Spoofing
        self._initialized = True

    # ── Nghiệp vụ ──────────────────────────────────────────────────────────
    def enroll_employee(
        self,
        emp_code: str,
        name: str,
        face_samples: list[np.ndarray],
        department: str = "",
    ) -> EnrollResult:
        """
        1. Quét bảo mật Liveness (Chống giả mạo).
        2. Chống trùng lặp danh tính.
        3. Tạo record Employee trong DB.
        4. Trích xuất embedding và Lưu vào FAISS.
        """
        if not emp_code or not name:
            return EnrollResult(False, emp_code, "Mã NV và Họ tên không được để trống.")

        # Thêm kiểm tra DB trước cho nhanh (để khỏi chạy AI tốn thời gian nếu trùng mã)
        existing = self._db.get_employee_by_code(emp_code)
        if existing:
            return EnrollResult(False, emp_code, f"Mã NV '{emp_code}' đã tồn tại trong DB.")

        with self._lock:
            try:
                # ── BỘ LỌC BẢO MẬT 3 LỚP ──
                embeddings = []
                for img in face_samples:
                    # 🔒 LỚP 1: Quét Liveness (Chống giả mạo)
                    spoof_results = self._spoofing.detect_spoof(img)
                    
                    # Nếu phát hiện bất kỳ khuôn mặt nào bị đánh dấu Fake -> Từ chối ảnh này
                    is_fake = any(not r.is_real for r in spoof_results)
                    if is_fake:
                        logger.warning(f"Enroll: Bỏ qua 1 mẫu do phát hiện GIẢ MẠO cho {emp_code}")
                        continue

                    # 🔒 LỚP 2: Trích xuất Vector đặc trưng
                    emb = self._recognizer.get_embedding(img)
                    if emb is not None:
                        
                        # 🔒 LỚP 3: Chống trùng lặp danh tính (Duplication check)
                        matches = self._recognizer.identify_face(emb, top_k=1)
                        if matches:
                            match_code, sim = matches[0]
                            # Nếu độ giống nhau vượt qua ngưỡng an toàn -> Người này đã từng đăng ký!
                            if sim >= FACE_RECOGNITION_THRESHOLD:
                                return EnrollResult(
                                    False, 
                                    emp_code, 
                                    f"Từ chối! Khuôn mặt này trùng khớp với nhân sự đã tồn tại ({match_code})."
                                )
                                
                        embeddings.append(emb)

                if not embeddings:
                    return EnrollResult(
                        False, 
                        emp_code, 
                        "Đăng ký thất bại! Mẫu không hợp lệ hoặc đã bị chặn bởi bộ lọc chống giả mạo."
                    )

                # ── LƯU DỮ LIỆU CHÍNH THỨC ──
                try:
                    self._db.add_employee(emp_code, name, department)
                except ValueError as e:
                    return EnrollResult(False, emp_code, str(e))

                # Lưu vào FAISS Vector DB
                success = self._recognizer.add_face(emp_code, embeddings)
                if not success:
                    self._safe_delete_from_db(emp_code)
                    return EnrollResult(False, emp_code, "Lỗi khi lưu vector vào FAISS. Đã rollback dữ liệu DB.")

                logger.info("Đã enroll thành công NV %s với %d vector.", emp_code, len(embeddings))
                return EnrollResult(True, emp_code, "Đăng ký hồ sơ sinh trắc học thành công.", len(embeddings))

            except Exception as exc:
                logger.exception("Lỗi hệ thống khi enroll_employee")
                self._safe_delete_from_db(emp_code)
                return EnrollResult(False, emp_code, f"Lỗi hệ thống: {exc}")

    def delete_employee(self, emp_code: str) -> DeleteResult:
        """Xóa NV hoàn toàn khỏi DB và FAISS (Hard-delete)."""
        with self._lock:
            # 1. Xóa trong FAISS
            success_faiss = self._recognizer.remove_face(emp_code)

            # 2. Xóa trong DB
            emp = self._db.get_employee_by_code(emp_code)
            if not emp:
                return DeleteResult(False, emp_code, f"Không tìm thấy NV '{emp_code}' trong DB.")

            self._db.delete_employee(emp.id)
            
            msg = "Đã xóa hoàn toàn khỏi DB và FAISS." if success_faiss else "Đã xóa khỏi DB (Không có vector trong FAISS)."
            return DeleteResult(True, emp_code, msg)

    def rebuild_faiss_from_db(self) -> tuple[bool, str]:
        """
        Ghi chú: Để build lại, hệ thống cần phải lưu trữ vector thô (raw embeddings)
        trong cơ sở dữ liệu SQLite ngay từ lúc đăng ký.
        """
        logger.warning(
            "rebuild_faiss_from_db: chức năng này cần embedding thô. "
            "Hiện tại chưa implement — cần thêm bảng lưu raw embedding vào DB."
        )
        return False, "Chưa implement. Cần bảng raw_embeddings trong DB."

    def get_registered_count(self) -> int:
        """Số nhân viên đang có vector trong FAISS."""
        return len(self._recognizer.registered_employees)

    def get_all_active_employees(self) -> list[Employee]:
        """Danh sách nhân viên active từ DB."""
        try:
            return self._db.get_all_employees(active_only=True)
        except Exception as exc:
            logger.exception("get_all_active_employees: lỗi")
            return []

    # ── Internal helpers ─────────────────────────────────────────────────────
    def _safe_delete_from_db(self, emp_code: str) -> None:
        """Rollback: xóa nhân viên khỏi DB theo emp_code (dùng khi FAISS thất bại)."""
        try:
            emp = self._db.get_employee_by_code(emp_code)
            if emp:
                self._db.delete_employee(emp.id)
                logger.info("Rollback DB: đã xóa %s", emp_code)
        except Exception:
            logger.exception("Rollback DB thất bại cho %s", emp_code)