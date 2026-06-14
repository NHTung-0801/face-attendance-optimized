"""
src/database/models.py
Định nghĩa schema CSDL bằng SQLAlchemy 2.x ORM.
"""

from datetime import datetime
from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey,
    Integer, String, UniqueConstraint, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────
class Employee(Base):
    __tablename__ = "employees"

    id:         Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    emp_code:   Mapped[str]      = mapped_column(String(32), unique=True, nullable=False, index=True)
    name:       Mapped[str]      = mapped_column(String(128), nullable=False)
    department: Mapped[str]      = mapped_column(String(64), nullable=True)
    status:     Mapped[bool]     = mapped_column(Boolean, default=True, nullable=False)   # True = active
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    attendances: Mapped[list["Attendance"]] = relationship(
        back_populates="employee", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Employee {self.emp_code} | {self.name}>"


# ─────────────────────────────────────────────
class Session(Base):
    """Phiên làm việc / ca chấm công (sáng, chiều, …)."""
    __tablename__ = "sessions"

    id:           Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_name: Mapped[str]      = mapped_column(String(64), nullable=False)
    date:         Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_active:    Mapped[bool]     = mapped_column(Boolean, default=True, nullable=False)
    created_at:   Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    attendances: Mapped[list["Attendance"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Session {self.session_name} | {self.date.date()}>"


# ─────────────────────────────────────────────
class Attendance(Base):
    __tablename__ = "attendances"
    __table_args__ = (
        UniqueConstraint("emp_id", "session_id", name="uq_emp_session"),
    )

    id:               Mapped[int]            = mapped_column(Integer, primary_key=True, autoincrement=True)
    emp_id:           Mapped[int]            = mapped_column(ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id:       Mapped[int]            = mapped_column(ForeignKey("sessions.id",   ondelete="CASCADE"), nullable=False, index=True)
    timestamp:        Mapped[datetime]       = mapped_column(DateTime, nullable=False)
    confidence_score: Mapped[float | None]   = mapped_column(Float, nullable=True)   # InsightFace similarity
    is_spoofed:       Mapped[bool]           = mapped_column(Boolean, default=False)  # Anti-spoofing flag

    employee: Mapped["Employee"] = relationship(back_populates="attendances")
    session:  Mapped["Session"]  = relationship(back_populates="attendances")

    def __repr__(self) -> str:
        return f"<Attendance emp={self.emp_id} session={self.session_id} @ {self.timestamp}>"
