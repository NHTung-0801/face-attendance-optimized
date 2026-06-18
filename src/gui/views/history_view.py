from __future__ import annotations

import csv
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QFileDialog,
    QFrame,
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
from src.utils.config import EXPORT_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

_COLS = ["#", "Mã NV", "Họ và Tên", "Phòng Ban", "Ca làm việc", "Thời gian", "Độ tin cậy", "Giả mạo"]

class HistoryView(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._db = DatabaseManager.instance()
        self._current_rows: list[dict] = []
        self._build_ui()
        self._load_sessions()

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

        body_layout.addWidget(self._build_filter_card())
        body_layout.addWidget(self._build_table_card(), stretch=1)

        root.addWidget(body, stretch=1)

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

        title = QLabel("📋  Lịch Sử Chấm Công")
        title.setStyleSheet(
            "font-size: 22px; font-weight: 700; color: #f8fafc; letter-spacing: 0.3px;"
        )
        title_box.addWidget(title)

        sub = QLabel("Tra cứu, bộ lọc theo ca làm việc và xuất dữ liệu báo cáo")
        sub.setStyleSheet("color: #8c909f; font-size: 12px;")
        title_box.addWidget(sub)

        layout.addLayout(title_box)
        layout.addStretch()

        return header

    def _build_filter_card(self) -> QFrame:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #161f2e;
                border: 1px solid #1d6475;
                border-radius: 16px;
            }
            QLabel {
                color: #8c909f;
                font-size: 13px;
                font-weight: 600;
                border: none;
            }
        """)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        input_style = """
            QComboBox, QDateEdit {
                background-color: #0d1720;
                color: #f8fafc;
                border: 1px solid #1d6475;
                border-radius: 8px;
                padding: 0 12px;
                font-size: 13px;
            }
            QComboBox:focus, QDateEdit:focus {
                border: 1px solid #4cd7f6;
            }
            QComboBox::drop-down, QDateEdit::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background-color: #0d1720;
                color: #f8fafc;
                selection-background-color: #1d6475;
                border: 1px solid #1d6475;
                outline: none;
            }
        """

        layout.addWidget(QLabel("Ca làm việc:"))
        self._session_combo = QComboBox()
        self._session_combo.setFixedWidth(240)
        self._session_combo.setFixedHeight(38)
        self._session_combo.setStyleSheet(input_style)
        self._session_combo.currentIndexChanged.connect(self._on_filter_changed)
        layout.addWidget(self._session_combo)

        layout.addWidget(QLabel("Từ ngày:"))
        self._date_from = QDateEdit(calendarPopup=True)
        self._date_from.setDate(date.today().replace(day=1))
        self._date_from.setFixedHeight(38)
        self._date_from.setFixedWidth(130)
        self._date_from.setDisplayFormat("dd/MM/yyyy")
        self._date_from.setStyleSheet(input_style)
        self._date_from.dateChanged.connect(self._on_filter_changed)
        layout.addWidget(self._date_from)

        layout.addWidget(QLabel("Đến ngày:"))
        self._date_to = QDateEdit(calendarPopup=True)
        self._date_to.setDate(date.today())
        self._date_to.setFixedHeight(38)
        self._date_to.setFixedWidth(130)
        self._date_to.setDisplayFormat("dd/MM/yyyy")
        self._date_to.setStyleSheet(input_style)
        self._date_to.dateChanged.connect(self._on_filter_changed)
        layout.addWidget(self._date_to)

        btn_refresh = QPushButton("↺  Làm mới")
        btn_refresh.setFixedHeight(38)
        btn_refresh.setCursor(Qt.PointingHandCursor)
        btn_refresh.clicked.connect(self._on_filter_changed)
        _style_secondary(btn_refresh)
        layout.addWidget(btn_refresh)

        layout.addStretch()

        self._btn_export = QPushButton("⬇  Xuất CSV")
        self._btn_export.setFixedHeight(38)
        self._btn_export.setCursor(Qt.PointingHandCursor)
        self._btn_export.clicked.connect(self._on_export_csv)
        _style_success(self._btn_export)
        layout.addWidget(self._btn_export)

        return card

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

        header_row = QHBoxLayout()
        self._stat_label = QLabel("—")
        self._stat_label.setStyleSheet("color: #4cd7f6; font-size: 13px; font-weight: 700; border: none;")
        header_row.addWidget(self._stat_label)
        header_row.addStretch()
        layout.addLayout(header_row)

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
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        
        self._table.setColumnWidth(0, 40)
        self._table.setColumnWidth(1, 90)
        self._table.setColumnWidth(3, 110)
        self._table.setColumnWidth(5, 140)
        self._table.setColumnWidth(6, 100)
        self._table.setColumnWidth(7, 90)
        
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.verticalHeader().setVisible(False)
        
        layout.addWidget(self._table, stretch=1)

        self._empty_hint = QLabel("Không có dữ liệu phù hợp với bộ lọc.")
        self._empty_hint.setAlignment(Qt.AlignCenter)
        self._empty_hint.setStyleSheet("color: #475569; font-size: 12px; padding: 28px 0; border: none;")
        self._empty_hint.hide()
        layout.addWidget(self._empty_hint)

        return card

    def _load_sessions(self) -> None:
        self._session_combo.blockSignals(True)
        self._session_combo.clear()
        self._session_combo.addItem("— Tất cả ca —", userData=None)

        try:
            sessions = self._db.get_all_sessions()
            for item in sessions:
                s = item[0] if isinstance(item, tuple) else item
                s_date = s.date
                if isinstance(s_date, str):
                    try:
                        s_date = datetime.strptime(s_date.split()[0], "%Y-%m-%d").date()
                    except Exception:
                        s_date = datetime.now().date()
                label = f"{s.session_name}  ({s_date.strftime('%d/%m/%Y')})"
                self._session_combo.addItem(label, userData=s.id)
        except Exception as exc:
            logger.exception("HistoryView: không load được sessions")

        self._session_combo.blockSignals(False)
        self._on_filter_changed()

    @Slot()
    def _on_filter_changed(self) -> None:
        session_id = self._session_combo.currentData()
        from_dt = datetime.combine(self._date_from.date().toPython(), datetime.min.time())
        to_dt   = datetime.combine(self._date_to.date().toPython(),   datetime.max.time())

        try:
            records = self._fetch_records(session_id, from_dt, to_dt)
        except Exception as exc:
            logger.exception("HistoryView: lỗi query")
            QMessageBox.warning(self, "Lỗi", f"Không tải được dữ liệu:\n{exc}")
            return

        self._render(records)

    def _fetch_records(
        self,
        session_id: Optional[int],
        from_dt: datetime,
        to_dt: datetime,
    ) -> list[dict]:
        rows: list[dict] = []

        if session_id is not None:
            attendances = self._db.get_attendance_by_session(session_id)
        else:
            attendances = self._fetch_all_in_range(from_dt, to_dt)

        for item in attendances:
            att = item[0] if isinstance(item, tuple) else item
            att_ts = att.timestamp
            
            if isinstance(att_ts, str):
                try:
                    ts_str = att_ts.split(".")[0]
                    att_ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    att_ts = datetime.now()
            
            if not (from_dt <= att_ts <= to_dt):
                continue
                
            emp = att.employee
            ses = att.session
            dept = getattr(emp, 'department', "—") if emp else "—"
            if not dept: dept = "—"
            
            rows.append({
                "emp_code":        emp.emp_code if emp else "—",
                "name":            emp.name     if emp else "—",
                "department":      dept,
                "session_name":    ses.session_name if ses else "—",
                "timestamp":       att_ts.strftime("%d/%m/%Y %H:%M:%S"),
                "confidence":      f"{att.confidence_score:.1%}" if getattr(att, 'confidence_score', None) else "—",
                "is_spoofed":      "⚠ Cảnh báo" if getattr(att, 'is_spoofed', False) else "An toàn",
                "_timestamp_raw":  att_ts,
                "_spoofed_raw":    getattr(att, 'is_spoofed', False),
            })

        return rows

    def _fetch_all_in_range(self, from_dt: datetime, to_dt: datetime) -> list:
        sessions = self._db.get_all_sessions()
        result = []
        for item in sessions:
            s = item[0] if isinstance(item, tuple) else item
            atts = self._db.get_attendance_by_session(s.id)
            if atts:
                result.extend(atts)
        return result

    def _render(self, rows: list[dict]) -> None:
        self._current_rows = rows
        self._table.setRowCount(0)

        if not rows:
            self._table.hide()
            self._empty_hint.show()
            self._stat_label.setText("Không có bản ghi nào")
            return

        self._table.show()
        self._empty_hint.hide()

        spoofed_count = 0
        for i, row in enumerate(rows):
            r = self._table.rowCount()
            self._table.insertRow(r)
            self._table.setRowHeight(r, 42)

            is_spoof = row["_spoofed_raw"]
            if is_spoof:
                spoofed_count += 1

            cells = [
                (str(i + 1),           "#8c909f"),
                (row["emp_code"],      "#4cd7f6"),
                (row["name"],          "#f8fafc"),
                (row["department"],    "#cbd5e1"),
                (row["session_name"],  "#8c909f"),
                (row["timestamp"],     "#f8fafc"),
                (row["confidence"],    "#4edea3"),
                (row["is_spoofed"],    "#f87171" if is_spoof else "#4edea3"),
            ]
            
            for col, (text, color) in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(QColor(color))
                self._table.setItem(r, col, item)

        self._stat_label.setText(
            f"Hiển thị {len(rows)} bản ghi  •  Phát hiện {spoofed_count} trường hợp giả mạo"
        )

    @Slot()
    def _on_export_csv(self) -> None:
        if not self._current_rows:
            QMessageBox.information(self, "Thông báo", "Không có dữ liệu để xuất.")
            return

        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        default_name = EXPORT_DIR / f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu file CSV",
            str(default_name),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        try:
            self._write_csv(Path(path))
            QMessageBox.information(
                self,
                "Thành công",
                f"Đã xuất {len(self._current_rows)} bản ghi ra:\n{path}",
            )
            logger.info("HistoryView: export %d rows → %s", len(self._current_rows), path)
        except Exception as exc:
            logger.exception("HistoryView: export CSV thất bại")
            QMessageBox.critical(self, "Lỗi", f"Không thể ghi file:\n{exc}")

    def _write_csv(self, path: Path) -> None:
        headers = ["STT", "Mã NV", "Họ và Tên", "Phòng Ban",
                   "Ca làm việc", "Thời gian chấm công", "Độ tin cậy", "Phát hiện giả mạo"]
        with path.open("w", newline="", encoding="utf-8-sig") as f:
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

    def refresh(self) -> None:
        self._load_sessions()

def _style_secondary(btn: QPushButton) -> None:
    btn.setStyleSheet("""
        QPushButton {
            background-color: #0d1720;
            color: #4cd7f6;
            border: 1px solid #4cd7f6;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 700;
            padding: 0 16px;
        }
        QPushButton:hover { background-color: #4cd7f6; color: #0b1326; }
        QPushButton:pressed { background-color: #2ca0ba; color: #0b1326; }
    """)

def _style_success(btn: QPushButton) -> None:
    btn.setStyleSheet("""
        QPushButton {
            background-color: #064e3b;
            color: #4edea3;
            border: 1px solid #4edea3;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 700;
            padding: 0 16px;
        }
        QPushButton:hover { background-color: #4edea3; color: #064e3b; }
        QPushButton:pressed { background-color: #34d399; color: #064e3b; }
    """)