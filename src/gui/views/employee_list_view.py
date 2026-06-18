"""
src/gui/views/employee_list_view.py
EmployeeListView — danh sách nhân viên, tìm kiếm, xóa.
Phong cách đồng bộ SecureFace AI Engine (cyberpunk) với toàn app.

Token màu (đồng bộ toàn app):
    Background sâu    : #0b1326 / #060e20
    Surface             : #161f2e / #111828
    Border / glow       : #2ca0ba / #1d6475
    Accent chính (cyan) : #4cd7f6
    Accent xanh lá      : #4edea3
    Accent đỏ (danger)  : #f87171
    Text chính          : #f8fafc
    Text phụ             : #8c909f
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
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

    # ── UI Root ──────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        self.setStyleSheet("background-color: #0b1326;")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(28, 24, 28, 28)
        body_layout.setSpacing(18)

        body_layout.addLayout(self._build_stat_row())
        body_layout.addWidget(self._build_table_card(), stretch=1)

        root.addWidget(body, stretch=1)

    # ── Header ───────────────────────────────────────────────────────────────
    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setFixedHeight(76)
        header.setStyleSheet("""
            QFrame {
                background-color: #0b1326;
                border-bottom: 2px solid #2ca0ba;
            }
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(32, 0, 32, 0)
        layout.setSpacing(16)

        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        title_box.setAlignment(Qt.AlignVCenter)

        title = QLabel("👥  Danh Sách Nhân Viên")
        title.setStyleSheet(
            "font-size: 22px; font-weight: 700; color: #f8fafc; letter-spacing: 0.3px;"
        )
        title_box.addWidget(title)

        sub = QLabel("Quản lý hồ sơ nhân viên và dữ liệu nhận diện khuôn mặt")
        sub.setStyleSheet("color: #8c909f; font-size: 12px;")
        title_box.addWidget(sub)

        layout.addLayout(title_box)
        layout.addStretch()

        # Search box trên header — dễ thấy, dễ truy cập
        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍  Tìm theo mã hoặc tên…")
        self._search.setFixedWidth(260)
        self._search.setFixedHeight(38)
        self._search.setStyleSheet("""
            QLineEdit {
                background-color: #161f2e;
                color: #f8fafc;
                border: 1px solid #1d6475;
                border-radius: 8px;
                padding: 0 14px;
                font-size: 13px;
            }
            QLineEdit:focus { border: 1px solid #4cd7f6; }
            QLineEdit::placeholder { color: #475569; }
        """)
        self._search.textChanged.connect(self._on_search)
        layout.addWidget(self._search)

        btn_refresh = QPushButton("↺  Làm mới")
        btn_refresh.setFixedHeight(38)
        btn_refresh.setCursor(Qt.PointingHandCursor)
        btn_refresh.clicked.connect(self.load_data)
        _style_secondary(btn_refresh)
        layout.addWidget(btn_refresh)

        return header

    # ── Stat row (3 thẻ thống kê nhanh) ─────────────────────────────────────
    def _build_stat_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(16)

        self._card_total  = self._build_stat_card("Tổng Nhân Viên", "—", "#4cd7f6")
        self._card_active = self._build_stat_card("Đang Hoạt Động", "—", "#4edea3")
        self._card_faiss  = self._build_stat_card("Đã Đăng Ký Khuôn Mặt", "—", "#4d8eff")

        row.addWidget(self._card_total)
        row.addWidget(self._card_active)
        row.addWidget(self._card_faiss)
        return row

    def _build_stat_card(self, title: str, value: str, value_color: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #161f2e;
                border: 1px solid #1d6475;
                border-radius: 14px;
            }
            QFrame:hover { border: 1px solid #2ca0ba; }
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(6)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #8c909f; font-size: 11px;")

        lbl_value = QLabel(value)
        lbl_value.setObjectName("stat_value")
        lbl_value.setStyleSheet(f"font-size: 22px; font-weight: 800; color: {value_color};")

        layout.addWidget(lbl_title)
        layout.addWidget(lbl_value)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return card

    def _set_stat(self, card: QFrame, text: str) -> None:
        lbl = card.findChild(QLabel, "stat_value")
        if lbl:
            lbl.setText(text)

    # ── Table card ───────────────────────────────────────────────────────────
    def _build_table_card(self) -> QFrame:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #161f2e;
                border: 1px solid #1d6475;
                border-radius: 16px;
            }
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self._table = QTableWidget(0, len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                alternate-background-color: #0d1720;
                border: none;
                gridline-color: #102630;
                color: #dae2fd;
                font-size: 12px;
            }
            QTableWidget::item {
                border-bottom: 1px solid #102630;
                padding: 8px;
            }
            QTableWidget::item:selected {
                background-color: #1d6475;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #111828;
                color: #8c909f;
                font-weight: 700;
                border: none;
                border-bottom: 1px solid #102630;
                padding: 10px;
                font-size: 10px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
        """)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(2, QHeaderView.Stretch)   # Họ và Tên
        header.setSectionResizeMode(3, QHeaderView.Stretch)   # Phòng Ban
        self._table.setColumnWidth(0, 50)    # ID
        self._table.setColumnWidth(1, 90)    # Mã NV
        self._table.setColumnWidth(4, 100)   # Trạng thái
        self._table.setColumnWidth(5, 130)   # Ngày tạo
        self._table.setColumnWidth(6, 80)    # Nút xóa
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(28)   # Để row height áp dụng đúng

        layout.addWidget(self._table, stretch=1)

        self._empty_hint = QLabel("Không có nhân viên nào phù hợp.")
        self._empty_hint.setAlignment(Qt.AlignCenter)
        self._empty_hint.setStyleSheet("color: #475569; font-size: 12px; padding: 28px 0;")
        self._empty_hint.hide()
        layout.addWidget(self._empty_hint)

        return card

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

        total    = len(self._all_employees)
        active   = sum(1 for e in self._all_employees if e.status)
        in_faiss = EmployeeManager.instance().get_registered_count()

        self._set_stat(self._card_total,  str(total))
        self._set_stat(self._card_active, str(active))
        self._set_stat(self._card_faiss,  str(in_faiss))

    def _render_rows(self, employees: list[Employee]) -> None:
        self._table.setRowCount(0)

        if not employees:
            self._table.hide()
            self._empty_hint.show()
            return

        self._table.show()
        self._empty_hint.hide()

        for emp in employees:
            self._add_row(emp)

    def _add_row(self, emp: Employee) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setRowHeight(row, 48)

        status_text  = "✅ Active"   if emp.status else "🚫 Inactive"
        status_color = "#4edea3"     if emp.status else "#f87171"
        created      = emp.created_at.strftime("%d/%m/%Y %H:%M") if emp.created_at else "—"

        values = [
            (str(emp.id),            "#8c909f"),
            (emp.emp_code,            "#4cd7f6"),
            (emp.name,                "#f8fafc"),
            (emp.department or "—",   "#cbd5e1"),
            (status_text,             status_color),
            (created,                 "#8c909f"),
        ]
        for col, (text, color) in enumerate(values):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            item.setForeground(QColor(color))
            self._table.setItem(row, col, item)

        btn = QPushButton("🗑 Xóa")
        btn.setFixedSize(64, 30)
        btn.setCursor(Qt.PointingHandCursor)
        _style_danger_small(btn)
        btn.clicked.connect(
            lambda _, eid=emp.id, ecode=emp.emp_code, ename=emp.name:
                self._on_delete(eid, ecode, ename)
        )

        # Wrapper để nút căn giữa trong cell (tránh dính sát viền)
        wrapper = QWidget()
        wrapper_layout = QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setAlignment(Qt.AlignCenter)
        wrapper_layout.addWidget(btn)
        self._table.setCellWidget(row, 6, wrapper)

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
def _style_secondary(btn: QPushButton) -> None:
    """Nút phụ — outline cyan, dùng cho 'Làm mới' và các action không phá hủy."""
    btn.setStyleSheet("""
        QPushButton {
            background-color: #0d1720;
            color: #4cd7f6;
            border: 1px solid #4cd7f6;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 700;
            padding: 0 16px;
        }
        QPushButton:hover {
            background-color: #4cd7f6;
            color: #0b1326;
        }
        QPushButton:pressed {
            background-color: #2ca0ba;
            color: #0b1326;
        }
    """)


def _style_danger_small(btn: QPushButton) -> None:
    """Nút xóa nhỏ trong bảng — đỏ nhạt, rõ ràng nhưng không quá gắt."""
    btn.setStyleSheet("""
        QPushButton {
            background-color: transparent;
            color: #f87171;
            border: 1px solid #7f1d1d;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #7f1d1d;
            color: #fecaca;
            border: 1px solid #f87171;
        }
        QPushButton:pressed {
            background-color: #450a0a;
        }
    """)
