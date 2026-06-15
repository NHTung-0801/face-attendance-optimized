"""
src/gui/views/employee_list_view.py
EmployeeListView — bảng danh sách nhân viên, tìm kiếm, xóa.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.employee_manager import EmployeeManager
from src.database.db_manager import DatabaseManager
from src.database.models import Employee
from src.utils.logger import get_logger

logger = get_logger(__name__)

_COLS = ["ID", "Mã NV", "Họ và Tên", "Phòng Ban", "Trạng Thái", "Ngày tạo", ""]


class EmployeeListView(QWidget):
    employee_deleted = Signal(str)   # emp_code → MainWindow cập nhật status bar

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._db      = DatabaseManager.instance()
        self._manager = EmployeeManager.instance()
        self._all_employees: list[Employee] = []
        self._build_ui()
        self.load_data()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Tiêu đề + toolbar
        top = QHBoxLayout()
        title = QLabel("👥  Danh Sách Nhân Viên")
        title.setFont(QFont("Segoe UI", 15, QFont.Bold))
        title.setStyleSheet("color:#f1f5f9;")
        top.addWidget(title)
        top.addStretch()

        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍  Tìm theo mã hoặc tên…")
        self._search.setFixedWidth(240)
        self._search.setFixedHeight(34)
        self._search.textChanged.connect(self._on_search)
        top.addWidget(self._search)

        btn_refresh = QPushButton("↺  Làm mới")
        btn_refresh.setFixedHeight(34)
        btn_refresh.clicked.connect(self.load_data)
        _style_btn(btn_refresh, "#334155")
        top.addWidget(btn_refresh)

        root.addLayout(top)

        # Thống kê nhanh
        self._stat_label = QLabel()
        self._stat_label.setStyleSheet("color:#64748b; font-size:12px;")
        root.addWidget(self._stat_label)

        # Bảng
        self._table = QTableWidget(0, len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._table.setColumnWidth(0, 50)    # ID
        self._table.setColumnWidth(1, 90)    # Mã NV
        self._table.setColumnWidth(4, 90)    # Trạng thái
        self._table.setColumnWidth(5, 130)   # Ngày tạo
        self._table.setColumnWidth(6, 70)    # Nút xóa
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        root.addWidget(self._table)

    # ── Data ────────────────────────────────────────────────────────────────
    @Slot()
    def load_data(self) -> None:
        try:
            self._all_employees = self._db.get_all_employees(active_only=False)
        except Exception as exc:
            logger.exception("EmployeeListView: lỗi load data")
            QMessageBox.warning(self, "Lỗi", f"Không tải được danh sách: {exc}")
            return

        self._render_rows(self._all_employees)
        total   = len(self._all_employees)
        active  = sum(1 for e in self._all_employees if e.status)
        in_faiss = EmployeeManager.instance().get_registered_count()
        self._stat_label.setText(
            f"Tổng: {total} nhân viên  •  Active: {active}  •  Trong FAISS: {in_faiss}"
        )

    def _render_rows(self, employees: list[Employee]) -> None:
        self._table.setRowCount(0)
        for emp in employees:
            self._add_row(emp)

    def _add_row(self, emp: Employee) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        status_text  = "✅ Active" if emp.status else "🚫 Inactive"
        status_color = "#4ade80"  if emp.status else "#f87171"
        created      = emp.created_at.strftime("%d/%m/%Y %H:%M") if emp.created_at else "—"

        values = [
            (str(emp.id),       "#94a3b8"),
            (emp.emp_code,      "#60a5fa"),
            (emp.name,          "#f1f5f9"),
            (emp.department or "—", "#cbd5e1"),
            (status_text,       status_color),
            (created,           "#94a3b8"),
        ]
        for col, (text, color) in enumerate(values):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            item.setForeground(Qt.GlobalColor.white)
            # Dùng setData để lưu màu rồi apply qua stylesheet per-cell nếu cần
            self._table.setItem(row, col, item)
            item.setForeground(_hex_to_qcolor(color))

        # Nút xóa
        btn = QPushButton("🗑 Xóa")
        btn.setFixedSize(60, 26)
        _style_btn(btn, "#7f1d1d")
        btn.clicked.connect(lambda _, eid=emp.id, ecode=emp.emp_code, ename=emp.name:
                            self._on_delete(eid, ecode, ename))
        self._table.setCellWidget(row, 6, btn)

    # ── Actions ─────────────────────────────────────────────────────────────
    @Slot(str)
    def _on_search(self, text: str) -> None:
        text = text.strip().lower()
        if not text:
            self._render_rows(self._all_employees)
            return
        filtered = [
            e for e in self._all_employees
            if text in e.emp_code.lower() or text in e.name.lower()
        ]
        self._render_rows(filtered)

    def _on_delete(self, emp_id: int, emp_code: str, emp_name: str) -> None:
        reply = QMessageBox.question(
            self,
            "Xác nhận xóa",
            f"Bạn có chắc muốn XÓA VĨNH VIỄN nhân viên:\n\n"
            f"  Mã NV : {emp_code}\n"
            f"  Họ tên: {emp_name}\n\n"
            "Hành động này sẽ xóa cả lịch sử chấm công và dữ liệu nhận diện!",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        result = self._manager.delete_employee(emp_id)
        if result.success:
            QMessageBox.information(self, "Thành công", result.message)
            self.employee_deleted.emit(emp_code)
            self.load_data()
        else:
            QMessageBox.critical(self, "Lỗi", result.message)


# ── Style helpers ────────────────────────────────────────────────────────────
def _hex_to_qcolor(hex_color: str):
    from PySide6.QtGui import QColor
    return QColor(hex_color)

def _style_btn(btn: QPushButton, variant: str) -> None:
    """
    variant: 'danger' (cho nút xóa), 'secondary' (cho nút làm mới)
    """
    btn.setProperty("class", variant)
    btn.style().unpolish(btn)
    btn.style().polish(btn)