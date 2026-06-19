"""
src/database/db_manager.py
Singleton DatabaseManager — thread-safe CRUD cho SQLite qua SQLAlchemy.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session as DBSession, sessionmaker

from src.database.models import Attendance, Base, Employee, Session as AttSession
from src.utils.config import DATABASE_URL, ATTENDANCE_COOLDOWN_SECONDS


class DatabaseManager:
    """
    Singleton thread-safe. Dùng:
        db = DatabaseManager.instance()
    """

    _instance:  Optional["DatabaseManager"] = None
    _lock: threading.Lock = threading.Lock()

    # ── Singleton ──────────────────────────────────────────────────────────
    def __new__(cls) -> "DatabaseManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    @classmethod
    def instance(cls) -> "DatabaseManager":
        return cls()

    # ── Init ───────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        if self._initialized:
            return
        self._engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},   # SQLite multi-thread
            echo=False,
        )
        self._SessionFactory = sessionmaker(
            bind=self._engine, expire_on_commit=False
        )
        Base.metadata.create_all(self._engine)
        self._initialized = True

    def _get_session(self) -> DBSession:
        return self._SessionFactory()

    # ═══════════════════════════════════════════════════════════════════════
    # EMPLOYEE
    # ═══════════════════════════════════════════════════════════════════════

    def add_employee(
        self,
        emp_code: str,
        name: str,
        department: str = "",
    ) -> Employee:
        """Thêm nhân viên mới. Raise ValueError nếu emp_code đã tồn tại."""
        with self._get_session() as s:
            if s.scalar(select(Employee).where(Employee.emp_code == emp_code)):
                raise ValueError(f"emp_code '{emp_code}' đã tồn tại.")
            emp = Employee(emp_code=emp_code, name=name, department=department)
            s.add(emp)
            s.commit()
            return emp

    def get_all_employees(self, active_only: bool = True) -> list[Employee]:
        with self._get_session() as s:
            q = select(Employee)
            if active_only:
                q = q.where(Employee.status == True)  # noqa: E712
            return list(s.scalars(q).all())

    def get_employee_by_code(self, emp_code: str) -> Optional[Employee]:
        with self._get_session() as s:
            return s.scalar(select(Employee).where(Employee.emp_code == emp_code))

    def get_employee_by_id(self, emp_id: int) -> Optional[Employee]:
        with self._get_session() as s:
            return s.get(Employee, emp_id)

    def deactivate_employee(self, emp_id: int) -> bool:
        """Soft-delete: đặt status = False."""
        with self._get_session() as s:
            s.execute(
                update(Employee).where(Employee.id == emp_id).values(status=False)
            )
            s.commit()
            return True

    def delete_employee(self, emp_id: int) -> bool:
        """Hard-delete + cascade xóa attendance."""
        with self._get_session() as s:
            emp = s.get(Employee, emp_id)
            if not emp:
                return False
            s.delete(emp)
            s.commit()
            return True

    # ═══════════════════════════════════════════════════════════════════════
    # SESSION (CA LÀM VIỆC)
    # ═══════════════════════════════════════════════════════════════════════

    def create_session(self, session_name: str, date: datetime | None = None) -> AttSession:
        with self._get_session() as s:
            att_session = AttSession(
                session_name=session_name,
                date=date or datetime.now(),
                is_active=True,
            )
            s.add(att_session)
            s.commit()
            return att_session

    def get_active_session(self) -> Optional[AttSession]:
        with self._get_session() as s:
            return s.scalar(
                select(AttSession)
                .where(AttSession.is_active == True)  # noqa: E712
                .order_by(AttSession.created_at.desc())
            )

    def close_session(self, session_id: int) -> bool:
        with self._get_session() as s:
            s.execute(
                update(AttSession)
                .where(AttSession.id == session_id)
                .values(is_active=False)
            )
            s.commit()
            return True

    def get_all_sessions(self) -> list[AttSession]:
        with self._get_session() as s:
            return list(s.scalars(select(AttSession).order_by(AttSession.date.desc())).all())

    # ═══════════════════════════════════════════════════════════════════════
    # ATTENDANCE
    # ═══════════════════════════════════════════════════════════════════════

    def record_attendance(
        self,
        emp_id: int,
        session_id: int,
        confidence_score: float = 0.0,
        is_spoofed: bool = False,
    ) -> tuple[bool, str]:
        """Ghi nhận chấm công với kiểm tra cooldown an toàn cho SQLite."""
        if is_spoofed:
            return False, "Phát hiện giả mạo khuôn mặt."

        now = datetime.now()

        with self._get_session() as s:
            # 1. Lấy bản ghi điểm danh GẦN NHẤT của nhân viên này
            last_att = (
                s.query(Attendance)
                .filter(Attendance.emp_id == emp_id)
                .order_by(Attendance.timestamp.desc())
                .first()
            )
            
            # 2. Xử lý logic Cooldown trực tiếp bằng Python (An toàn tuyệt đối cho SQLite)
            if last_att:
                last_time = last_att.timestamp
                # Parse string sang datetime nếu SQLite lưu dạng string
                if isinstance(last_time, str):
                    try:
                        last_time = datetime.strptime(last_time.split(".")[0], "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        last_time = now
                
                elapsed = (now - last_time).total_seconds()
                if elapsed < ATTENDANCE_COOLDOWN_SECONDS:
                    remaining = int(ATTENDANCE_COOLDOWN_SECONDS - elapsed)
                    hours = remaining // 3600
                    minutes = (remaining % 3600) // 60
                    return False, f"Điểm danh quá gần nhau. Cooldown còn {hours}h{minutes}p."

            # 3. Kiểm tra xem đã điểm danh trong ca này chưa (Chống trùng ca)
            existing_in_session = s.scalar(
                select(Attendance).where(
                    Attendance.emp_id == emp_id,
                    Attendance.session_id == session_id,
                )
            )
            if existing_in_session:
                # Xử lý format timestamp an toàn để in ra log
                ts = existing_in_session.timestamp
                if isinstance(ts, str):
                    ts = datetime.strptime(ts.split(".")[0], "%Y-%m-%d %H:%M:%S")
                return False, f"Đã điểm danh trong ca này lúc {ts.strftime('%H:%M:%S')}."

            # 4. Ghi nhận lượt điểm danh mới
            record = Attendance(
                emp_id=emp_id,
                session_id=session_id,
                timestamp=now,
                confidence_score=confidence_score,
                is_spoofed=False,
            )
            s.add(record)
            s.commit()
            return True, "Ghi nhận thành công."

    def get_attendance_by_session(self, session_id: int) -> list[Attendance]:
        with self._get_session() as s:
            return list(
                s.scalars(
                    select(Attendance)
                    .where(Attendance.session_id == session_id)
                    .order_by(Attendance.timestamp)
                ).all()
            )

    def get_attendance_by_employee(
        self,
        emp_id: int,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[Attendance]:
        with self._get_session() as s:
            q = select(Attendance).where(Attendance.emp_id == emp_id)
            if from_date:
                q = q.where(Attendance.timestamp >= from_date)
            if to_date:
                q = q.where(Attendance.timestamp <= to_date)
            return list(s.scalars(q.order_by(Attendance.timestamp)).all())
