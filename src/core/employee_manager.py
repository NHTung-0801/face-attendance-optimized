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
from src.database.db_manager import DatabaseManager
from src.database.models import Employee
from src.utils.logger import get_logger

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

    def __init__(self) -> None:
        if self._initialized:
            return
        self._db         = DatabaseManager.instance()
        self._recognizer = FaceRecognizer.instance()
        self._op_lock    = threading.Lock()   # Serialize enroll/delete
        self._initialized = True

    # ═══════════════════════════════════════════════════════════════════════
    # ENROLL
    # ═══════════════════════════════════════════════════════════════════════

    def enroll_employee(
        self,
        emp_code:     str,
        name:         str,
        face_samples: list[np.ndarray],   # Danh sách ảnh BGR đã crop khuôn mặt
        department:   str = "",
        use_mean_embedding: bool = True,
    ) -> EnrollResult:
        """
        Đăng ký nhân viên mới:
          1. Validate đầu vào.
          2. Trích embedding từ mỗi ảnh mẫu.
          3. Lưu thông tin text vào DB (Employee).
          4. Thêm embedding vào FAISS index.

        Args:
            emp_code:           Mã nhân viên (unique key).
            name:               Họ và tên.
            face_samples:       List ảnh khuôn mặt BGR (đã crop, không cần full frame).
            department:         Phòng ban (tuỳ chọn).
            use_mean_embedding: True  → tính vector trung bình rồi lưu 1 vector/nhân viên.
                                False → lưu tất cả vector riêng lẻ (nhận diện tốt hơn
                                        nhưng tốn bộ nhớ FAISS hơn).

        Returns:
            EnrollResult với success=True nếu cả DB lẫn FAISS đều thành công.
        """
        emp_code = emp_code.strip().upper()
        name     = name.strip()

        # ── 1. Validate ──────────────────────────────────────────────────
        if not emp_code:
            return EnrollResult(False, emp_code, "emp_code không được để trống.")
        if not name:
            return EnrollResult(False, emp_code, "Tên nhân viên không được để trống.")
        if not face_samples:
            return EnrollResult(False, emp_code, "Cần ít nhất 1 ảnh khuôn mặt.")

        with self._op_lock:
            # ── 2. Kiểm tra trùng mã ─────────────────────────────────────
            try:
                existing = self._db.get_employee_by_code(emp_code)
                if existing:
                    return EnrollResult(
                        False, emp_code,
                        f"Mã nhân viên '{emp_code}' đã tồn tại (id={existing.id}).",
                    )
            except Exception as exc:
                logger.exception("enroll_employee: lỗi kiểm tra trùng mã")
                return EnrollResult(False, emp_code, f"Lỗi kiểm tra DB: {exc}")

            # ── 3. Trích embedding ────────────────────────────────────────
            embeddings: list[np.ndarray] = []
            failed_count = 0

            for idx, img in enumerate(face_samples):
                try:
                    emb = self._recognizer.get_embedding(img)
                    if emb is not None:
                        embeddings.append(emb)
                    else:
                        failed_count += 1
                        logger.warning(
                            "enroll_employee '%s': ảnh %d không extract được embedding.",
                            emp_code, idx,
                        )
                except Exception as exc:
                    failed_count += 1
                    logger.warning(
                        "enroll_employee '%s': lỗi ảnh %d — %s", emp_code, idx, exc
                    )

            if not embeddings:
                return EnrollResult(
                    False, emp_code,
                    f"Không trích được embedding từ {len(face_samples)} ảnh. "
                    "Kiểm tra chất lượng ảnh đầu vào.",
                )

            # ── 4. Tính mean embedding (tuỳ chọn) ────────────────────────
            if use_mean_embedding:
                stack = np.vstack(embeddings)           # (N, 512)
                mean  = stack.mean(axis=0)
                norm  = np.linalg.norm(mean)
                embeddings_to_add = [mean / (norm + 1e-8)]
            else:
                embeddings_to_add = embeddings

            # ── 5. Lưu vào DB ─────────────────────────────────────────────
            try:
                self._db.add_employee(emp_code, name, department)
            except ValueError as exc:
                # add_employee raise ValueError nếu trùng (double-check)
                return EnrollResult(False, emp_code, str(exc))
            except Exception as exc:
                logger.exception("enroll_employee: lỗi ghi DB")
                return EnrollResult(False, emp_code, f"Lỗi ghi DB: {exc}")

            # ── 6. Thêm vào FAISS ─────────────────────────────────────────
            try:
                ok = self._recognizer.add_face(emp_code, embeddings_to_add)
                if not ok:
                    # Rollback DB nếu FAISS thất bại
                    self._safe_delete_from_db(emp_code)
                    return EnrollResult(
                        False, emp_code,
                        "Thêm vào FAISS thất bại. Đã rollback DB.",
                    )
            except Exception as exc:
                logger.exception("enroll_employee: lỗi FAISS")
                self._safe_delete_from_db(emp_code)
                return EnrollResult(False, emp_code, f"Lỗi FAISS (đã rollback DB): {exc}")

        msg = (
            f"Đăng ký thành công '{name}' ({emp_code}). "
            f"{len(embeddings_to_add)} vector lưu vào FAISS "
            f"(từ {len(embeddings)}/{len(face_samples)} ảnh hợp lệ)."
        )
        if failed_count:
            msg += f" {failed_count} ảnh bị bỏ qua."

        logger.info(msg)
        return EnrollResult(True, emp_code, msg, len(embeddings_to_add))

    # ═══════════════════════════════════════════════════════════════════════
    # DELETE
    # ═══════════════════════════════════════════════════════════════════════

    def delete_employee(self, emp_id: int) -> DeleteResult:
        """
        Xóa nhân viên hoàn toàn:
          1. Lấy emp_code từ DB để xoá FAISS.
          2. Xóa vector khỏi FAISS (rebuild index).
          3. Hard-delete khỏi DB (cascade xóa Attendance).

        Nếu FAISS xóa thất bại vẫn tiếp tục xóa DB nhưng log cảnh báo.
        """
        with self._op_lock:
            # ── 1. Lấy thông tin nhân viên ────────────────────────────────
            try:
                emp = self._db.get_employee_by_id(emp_id)
                if emp is None:
                    return DeleteResult(
                        False, "", f"Không tìm thấy nhân viên id={emp_id}."
                    )
                emp_code = emp.emp_code
                emp_name = emp.name
            except Exception as exc:
                logger.exception("delete_employee: lỗi truy vấn DB")
                return DeleteResult(False, "", f"Lỗi DB: {exc}")

            # ── 2. Xóa khỏi FAISS ─────────────────────────────────────────
            faiss_ok = False
            try:
                faiss_ok = self._recognizer.remove_face(emp_code)
                if not faiss_ok:
                    logger.warning(
                        "delete_employee: '%s' không tìm thấy trong FAISS index "
                        "(có thể chưa đăng ký khuôn mặt). Tiếp tục xóa DB.",
                        emp_code,
                    )
            except Exception as exc:
                logger.error(
                    "delete_employee: lỗi xóa FAISS cho '%s' — %s. Tiếp tục xóa DB.",
                    emp_code, exc,
                )

            # ── 3. Hard-delete khỏi DB ────────────────────────────────────
            try:
                deleted = self._db.delete_employee(emp_id)
                if not deleted:
                    return DeleteResult(
                        False, emp_code,
                        f"Xóa DB thất bại (id={emp_id} không còn tồn tại).",
                    )
            except Exception as exc:
                logger.exception("delete_employee: lỗi hard-delete DB")
                # Nếu DB xóa thất bại nhưng FAISS đã xóa → nguy hiểm
                # Log cảnh báo để admin xử lý thủ công
                if faiss_ok:
                    logger.critical(
                        "INCONSISTENCY: FAISS đã xóa '%s' nhưng DB xóa thất bại! "
                        "Cần rebuild FAISS từ DB thủ công.",
                        emp_code,
                    )
                return DeleteResult(False, emp_code, f"Lỗi hard-delete DB: {exc}")

        msg = f"Đã xóa '{emp_name}' ({emp_code}) khỏi DB và FAISS."
        logger.info(msg)
        return DeleteResult(True, emp_code, msg)

    def deactivate_employee(self, emp_id: int) -> DeleteResult:
        """
        Soft-delete: chỉ đặt status=False trong DB, KHÔNG xóa FAISS.
        Dùng khi muốn tạm nghỉ nhưng giữ lại lịch sử điểm danh.
        Lưu ý: nhân viên vẫn có thể bị nhận diện qua camera.
              Dùng delete_employee() nếu muốn ngăn hoàn toàn.
        """
        try:
            emp = self._db.get_employee_by_id(emp_id)
            if emp is None:
                return DeleteResult(False, "", f"Không tìm thấy id={emp_id}.")
            self._db.deactivate_employee(emp_id)
            msg = f"Đã vô hiệu hóa '{emp.name}' ({emp.emp_code})."
            logger.info(msg)
            return DeleteResult(True, emp.emp_code, msg)
        except Exception as exc:
            logger.exception("deactivate_employee: lỗi")
            return DeleteResult(False, "", f"Lỗi: {exc}")

    # ═══════════════════════════════════════════════════════════════════════
    # UTILS
    # ═══════════════════════════════════════════════════════════════════════

    def rebuild_faiss_from_db(self) -> tuple[bool, str]:
        """
        Utility: rebuild toàn bộ FAISS index từ dữ liệu DB.
        Dùng khi phát hiện inconsistency hoặc cần migrate.
        YÊU CẦU: mỗi nhân viên phải có embedding đã lưu riêng bằng cách
                  nào đó (vd: lưu file .npy). Hàm này chỉ là skeleton —
                  implement tuỳ theo cách lưu trữ embedding thô của dự án.
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
                logger.info("Rollback DB: đã xóa '%s'.", emp_code)
        except Exception as exc:
            logger.error("Rollback DB thất bại cho '%s': %s", emp_code, exc)
