"""
src/gui/views/history_view.py
HistoryView — xem lịch sử chấm công theo session, lọc theo ngày, export CSV.
"""

from __future__ import annotations

import csv
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.database.db_manager import DatabaseManager
from src.database.models import Attendance, Employee, Session as AttSession
from src.utils.config import EXPORT_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

_COLS = ["#", "Mã NV", "Họ và Tên", "Phòng Ban", "Ca làm việc", "Thời gian", "Độ tin cậy", "Giả mạo"]


class HistoryView(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._db = DatabaseManager.instance()
        # Cache: attendance_id → row data (dict) để export không cần query lại
        self._current_rows: list[dict] = []
        self._build_ui()
        self._load_sessions()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ── Tiêu đề ─────────────────────────────────────────────────────────
        title = QLabel("📋  Lịch Sử Chấm Công")
        title.setFont(QFont("Segoe UI", 15, QFont.Bold))
        title.setStyleSheet("color:#f1f5f9;")
        root.addWidget(title)

        # ── Toolbar lọc ─────────────────────────────────────────────────────
        filter_row = QHBoxLayout()
        filter_row.setSpacing(10)

        filter_row.addWidget(_lbl("Ca làm việc:"))
        self._session_combo = QComboBox()
        self._session_combo.setFixedWidth(260)
        self._session_combo.setFixedHeight(34)
        self._session_combo.setStyleSheet(_combo_style())
        self._session_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self._session_combo)

        filter_row.addWidget(_lbl("Từ ngày:"))
        self._date_from = QDateEdit(calendarPopup=True)
        self._date_from.setDate(date.today().replace(day=1))
        self._date_from.setFixedHeight(34)
        self._date_from.setStyleSheet(_input_style())
        self._date_from.setDisplayFormat("dd/MM/yyyy")
        self._date_from.dateChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self._date_from)

        filter_row.addWidget(_lbl("Đến ngày:"))
        self._date_to = QDateEdit(calendarPopup=True)
        self._date_to.setDate(date.today())
        self._date_to.setFixedHeight(34)
        self._date_to.setStyleSheet(_input_style())
        self._date_to.setDisplayFormat("dd/MM/yyyy")
        self._date_to.dateChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self._date_to)

        btn_refresh = QPushButton("↺  Làm mới")
        btn_refresh.setFixedHeight(34)
        btn_refresh.clicked.connect(self._on_filter_changed)
        _style_btn(btn_refresh, "#334155")
        filter_row.addWidget(btn_refresh)

        filter_row.addStretch()

        self._btn_export = QPushButton("⬇  Export CSV")
        self._btn_export.setFixedHeight(34)
        self._btn_export.clicked.connect(self._on_export_csv)
        _style_btn(self._btn_export, "#065f46")
        filter_row.addWidget(self._btn_export)

        root.addLayout(filter_row)

        # ── Stat bar ────────────────────────────────────────────────────────
        self._stat_label = QLabel("—")
        self._stat_label.setStyleSheet("color:#64748b; font-size:12px;")
        root.addWidget(self._stat_label)

        # ── Bảng ────────────────────────────────────────────────────────────
        self._table = QTableWidget(0, len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # Họ tên
        self._table.setColumnWidth(0, 40)    # #
        self._table.setColumnWidth(1, 80)    # Mã NV
        self._table.setColumnWidth(3, 110)   # Phòng ban
        self._table.setColumnWidth(4, 170)   # Ca
        self._table.setColumnWidth(5, 140)   # Thời gian
        self._table.setColumnWidth(6, 90)    # Độ tin cậy
        self._table.setColumnWidth(7, 70)    # Giả mạo
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setStyleSheet(_table_style())
        root.addWidget(self._table)

    # ── Load sessions vào ComboBox ───────────────────────────────────────────
    def _load_sessions(self) -> None:
        self._session_combo.blockSignals(True)
        self._session_combo.clear()
        self._session_combo.addItem("— Tất cả ca —", userData=None)

        try:
            sessions = self._db.get_all_sessions()
            for s in sessions:
                label = f"{s.session_name}  ({s.date.strftime('%d/%m/%Y')})"
                self._session_combo.addItem(label, userData=s.id)
        except Exception as exc:
            logger.exception("HistoryView: không load được sessions")

        self._session_combo.blockSignals(False)
        self._on_filter_changed()

    # ── Load + render data ───────────────────────────────────────────────────
    @Slot()
    def _on_filter_changed(self) -> None:
        session_id = self._session_combo.currentData()
        from_dt = datetime.combine(self._date_from.date().toPython(), datetime.min.time())
        to_dt   = datetime.combine(self._date_to.date().toPython(),   datetime.max.time())

        try:
            records = self._fetch_records(session_id, from_dt, to_dt)
        except Exception as exc:
            logger.exception("HistoryView: lỗi query")
            QMessageBox.warning(self, "Lỗi", f"Không tải được dữ liệu: {exc}")
            return

        self._render(records)

    def _fetch_records(
        self,
        session_id: Optional[int],
        from_dt: datetime,
        to_dt: datetime,
    ) -> list[dict]:
        """
        Truy vấn và join dữ liệu Attendance + Employee + Session.
        Trả về list dict để vừa render bảng vừa export CSV.
        """
        rows: list[dict] = []

        if session_id is not None:
            attendances = self._db.get_attendance_by_session(session_id)
        else:
            # Lấy tất cả theo khoảng ngày — query tổng hợp
            attendances = self._fetch_all_in_range(from_dt, to_dt)

        for att in attendances:
            if not (from_dt <= att.timestamp <= to_dt):
                continue
            emp = att.employee
            ses = att.session
            rows.append({
                "emp_code":        emp.emp_code if emp else "—",
                "name":            emp.name     if emp else "—",
                "department":      emp.department or "—" if emp else "—",
                "session_name":    ses.session_name if ses else "—",
                "timestamp":       att.timestamp.strftime("%d/%m/%Y %H:%M:%S"),
                "confidence":      f"{att.confidence_score:.1%}" if att.confidence_score else "—",
                "is_spoofed":      "⚠ Có" if att.is_spoofed else "Không",
                # Raw để export
                "_timestamp_raw":  att.timestamp,
                "_spoofed_raw":    att.is_spoofed,
            })

        return rows

    def _fetch_all_in_range(self, from_dt: datetime, to_dt: datetime) -> list[Attendance]:
        """Lấy tất cả Attendance từ DB có timestamp trong khoảng."""
        sessions = self._db.get_all_sessions()
        result: list[Attendance] = []
        for s in sessions:
            atts = self._db.get_attendance_by_session(s.id)
            result.extend(atts)
        return result

    def _render(self, rows: list[dict]) -> None:
        self._current_rows = rows
        self._table.setRowCount(0)

        spoofed_count = 0
        for i, row in enumerate(rows):
            r = self._table.rowCount()
            self._table.insertRow(r)

            is_spoof = row["_spoofed_raw"]
            if is_spoof:
                spoofed_count += 1

            cells = [
                (str(i + 1),           "#64748b"),
                (row["emp_code"],      "#60a5fa"),
                (row["name"],          "#f1f5f9"),
                (row["department"],    "#cbd5e1"),
                (row["session_name"],  "#94a3b8"),
                (row["timestamp"],     "#e2e8f0"),
                (row["confidence"],    "#4ade80"),
                (row["is_spoofed"],    "#f87171" if is_spoof else "#4ade80"),
            ]
            for col, (text, color) in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(_hex_color(color))
                self._table.setItem(r, col, item)

        self._stat_label.setText(
            f"Hiển thị {len(rows)} bản ghi  •  Phát hiện giả mạo: {spoofed_count}"
        )

    # ── Export CSV ───────────────────────────────────────────────────────────
    @Slot()
    def _on_export_csv(self) -> None:
        if not self._current_rows:
            QMessageBox.information(self, "Thông báo", "Không có dữ liệu để xuất.")
            return

        # Tạo tên file mặc định theo thời gian
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        default_name = EXPORT_DIR / f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu file CSV",
            str(default_name),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return   # User hủy dialog

        try:
            self._write_csv(Path(path))
            QMessageBox.information(
                self,
                "Export thành công",
                f"Đã xuất {len(self._current_rows)} bản ghi ra:\n{path}",
            )
            logger.info("HistoryView: export %d rows → %s", len(self._current_rows), path)
        except Exception as exc:
            logger.exception("HistoryView: export CSV thất bại")
            QMessageBox.critical(self, "Lỗi export", f"Không thể ghi file:\n{exc}")

    def _write_csv(self, path: Path) -> None:
        headers = ["STT", "Mã NV", "Họ và Tên", "Phòng Ban",
                   "Ca làm việc", "Thời gian chấm công", "Độ tin cậy", "Phát hiện giả mạo"]
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            # utf-8-sig → Excel Windows mở đúng tiếng Việt không bị lỗi font
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for i, row in enumerate(self._current_rows, start=1):
                writer.writerow({
                    "STT":                    i,
                    "Mã NV":                  row["emp_code"],
                    "Họ và Tên":              row["name"],
                    "Phòng Ban":              row["department"],
                    "Ca làm việc":            row["session_name"],
                    "Thời gian chấm công":    row["timestamp"],
                    "Độ tin cậy":             row["confidence"],
                    "Phát hiện giả mạo":      "Có" if row["_spoofed_raw"] else "Không",
                })

    # ── Public: gọi từ MainWindow khi tab được chọn ──────────────────────────
    def refresh(self) -> None:
        """Reload sessions và data (gọi khi switch sang tab này)."""
        self._load_sessions()


# ── Style helpers ─────────────────────────────────────────────────────────────
def _lbl(text: str) -> QLabel:
    l = QLabel(text)
    l.setStyleSheet("color:#94a3b8; font-size:12px;")
    return l

def _hex_color(hex_str: str):
    from PySide6.QtGui import QColor
    return QColor(hex_str)

def _combo_style() -> str:
    return """
        QComboBox {
            background:#1e293b; color:#f1f5f9;
            border:1px solid #334155; border-radius:6px;
            padding:0 10px; font-size:13px;
        }
        QComboBox::drop-down { border:none; }
        QComboBox QAbstractItemView {
            background:#1e293b; color:#f1f5f9;
            selection-background-color:#2563eb;
        }
    """

def _input_style() -> str:
    return """
        QDateEdit {
            background:#1e293b; color:#f1f5f9;
            border:1px solid #334155; border-radius:6px;
            padding:0 8px; font-size:13px;
        }
        QDateEdit::drop-down { border:none; }
    """

def _table_style() -> str:
    return """
        QTableWidget {
            background:#0f172a; color:#cbd5e1;
            gridline-color:#1e293b; border:none;
            alternate-background-color:#111827;
        }
        QHeaderView::section {
            background:#1e293b; color:#94a3b8;
            border:none; padding:6px;
            font-size:12px; font-weight:600;
        }
        QTableWidget::item:selected { background:#1d4ed8; color:#fff; }
    """

def _style_btn(btn: QPushButton, color: str) -> None:
    btn.setStyleSheet(f"""
        QPushButton {{
            background:{color}; color:#f1f5f9;
            border:none; border-radius:6px;
            font-size:12px; font-weight:600;
            padding:0 14px;
        }}
        QPushButton:hover {{ background:{color}cc; }}
    """)
