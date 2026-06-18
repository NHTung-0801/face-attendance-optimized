"""
src/gui/views/home_view.py
Trang chủ (Dashboard) — phong cách Cyberpunk / High-Tech "SecureFace AI Engine".

Token hệ thống (đồng bộ với main_window.py):
    Background sâu   : #0b1326 / #060e20
    Surface           : #161f2e / #111828
    Border / glow     : #2ca0ba / #1d6475
    Accent chính (cyan): #4cd7f6
    Accent phụ (xanh lá): #4edea3
    Accent phụ (blue)  : #4d8eff
    Text chính        : #f8fafc
    Text phụ          : #8c909f
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.employee_manager import EmployeeManager
from src.database.db_manager import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Signature element — Radar Scan Ring quanh icon mắt AI
# ═══════════════════════════════════════════════════════════════════════════
class _ScanRadar(QWidget):
    """
    Vòng quét radar xoay liên tục quanh icon trung tâm — đại diện cho việc
    AI Engine luôn "đang quan sát". Đây là signature element của trang chủ.
    """

    def __init__(self, size: int = 220, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._size  = size
        self._angle = 0
        self.setFixedSize(size, size)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate)
        self._timer.start(30)   # ~33 FPS

    def _rotate(self) -> None:
        self._angle = (self._angle + 2) % 360
        self.update()

    def stop(self) -> None:
        self._timer.stop()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        cx, cy = self._size // 2, self._size // 2
        r_outer = self._size // 2 - 4

        # ── Vòng tròn tĩnh mờ (3 lớp đồng tâm) ────────────────────────────
        for i, r_ratio in enumerate((1.0, 0.74, 0.48)):
            alpha = 28 if i == 0 else 18
            pen = QPen(QColor(44, 160, 186, alpha), 1.4)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            r = int(r_outer * r_ratio)
            painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)

        # ── Vòng quét sáng (gradient theo góc, giả lập bằng arc) ──────────
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(self._angle)

        sweep_pen = QPen(QColor(76, 215, 246, 180), 2.4)
        painter.setPen(sweep_pen)
        painter.drawArc(-r_outer, -r_outer, r_outer * 2, r_outer * 2, 0, 55 * 16)

        # Điểm sáng đầu vòng quét
        painter.setBrush(QColor(76, 215, 246))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(r_outer - 3, -3, 6, 6)
        painter.restore()

        # ── 4 tick nhỏ ở 0/90/180/270 độ ───────────────────────────────────
        painter.setPen(QPen(QColor(140, 144, 159, 90), 1.2))
        tick = 6
        painter.drawLine(cx, cy - r_outer, cx, cy - r_outer + tick)
        painter.drawLine(cx, cy + r_outer, cx, cy + r_outer - tick)
        painter.drawLine(cx - r_outer, cy, cx - r_outer + tick, cy)
        painter.drawLine(cx + r_outer, cy, cx + r_outer - tick, cy)

        painter.end()


# ═══════════════════════════════════════════════════════════════════════════
# HomeView
# ═══════════════════════════════════════════════════════════════════════════
class HomeView(QWidget):
    """
    Dashboard tổng quan: trạng thái AI Engine, thông số hệ thống,
    nhật ký điểm danh gần nhất (data thật từ DB, lọc CHUẨN theo NGÀY HÔM NAY).
    """

    navigate_requested = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._db      = DatabaseManager.instance()
        self._manager = EmployeeManager.instance()
        self._build_ui()
        self._setup_refresh_timer()
        self.refresh_data()

    # ── UI Root ──────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_layout.addWidget(self._build_header())

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(32, 24, 32, 32)
        content_layout.setSpacing(24)

        content_layout.addLayout(self._build_left_column(), stretch=2)
        content_layout.addLayout(self._build_right_column(), stretch=1)

        main_layout.addWidget(content, stretch=1)

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

        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        title_box.setAlignment(Qt.AlignVCenter)

        title_label = QLabel("Bảng Điều Khiển Hệ Thống")
        title_label.setStyleSheet(
            "font-size: 22px; font-weight: 700; color: #f8fafc; letter-spacing: 0.3px;"
        )
        title_box.addWidget(title_label)

        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet("color: #4edea3; font-weight: bold; font-size: 11px;")
        self._status_text = QLabel("Hệ thống hoạt động  •  Engine: ONNX Runtime")
        self._status_text.setStyleSheet("color: #8c909f; font-size: 12px;")
        status_row.addWidget(self._status_dot)
        status_row.addWidget(self._status_text)
        status_row.addStretch()
        title_box.addLayout(status_row)

        layout.addLayout(title_box)
        layout.addStretch()

        self._header_clock = QLabel()
        self._header_clock.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._header_clock.setStyleSheet(
            "color: #4cd7f6; font-weight: 600; font-size: 13px; font-family: Consolas, monospace;"
        )
        layout.addWidget(self._header_clock)

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_header_clock)
        self._clock_timer.start(1000)
        self._tick_header_clock()

        return header

    @Slot()
    def _tick_header_clock(self) -> None:
        now = datetime.now()
        self._header_clock.setText(now.strftime("%H:%M:%S  •  %d/%m/%Y"))

    # ── Cột trái: AI Core + Stat cards ──────────────────────────────────────
    def _build_left_column(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(24)

        col.addWidget(self._build_ai_core_card(), stretch=5)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(16)

        self._card_total_nv  = self._build_stat_card("Tổng Nhân Viên", "—", "#4cd7f6")
        self._card_today     = self._build_stat_card("Điểm Danh Hôm Nay", "—", "#4edea3")
        self._card_session   = self._build_stat_card("Ca Đang Mở", "—", "#4d8eff")

        stats_row.addWidget(self._card_total_nv)
        stats_row.addWidget(self._card_today)
        stats_row.addWidget(self._card_session)

        col.addLayout(stats_row, stretch=1)
        return col

    def _build_ai_core_card(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #060e20, stop:1 #0f1e3c
                );
                border: 2px solid #2ca0ba;
                border-radius: 16px;
            }
        """)
        outer = QVBoxLayout(frame)
        outer.setAlignment(Qt.AlignCenter)
        outer.setSpacing(18)
        outer.setContentsMargins(40, 32, 40, 32)

        radar_wrapper = QWidget()
        radar_layout  = QVBoxLayout(radar_wrapper)
        radar_layout.setAlignment(Qt.AlignCenter)
        radar_layout.setContentsMargins(0, 0, 0, 0)

        self._radar = _ScanRadar(size=200)
        radar_layout.addWidget(self._radar, alignment=Qt.AlignCenter)

        eye_label = QLabel("👁")
        eye_font = QFont()
        eye_font.setPointSize(40)
        eye_label.setFont(eye_font)
        eye_label.setAlignment(Qt.AlignCenter)
        eye_label.setStyleSheet("background: transparent;")
        eye_label.setFixedSize(200, 200)
        eye_label.setParent(radar_wrapper)
        eye_label.move(0, 0)
        eye_label.raise_()

        outer.addWidget(radar_wrapper, alignment=Qt.AlignCenter)

        core_title = QLabel("SECUREFACE AI ENGINE")
        core_title.setAlignment(Qt.AlignCenter)
        core_title.setStyleSheet(
            "color: #4cd7f6; font-size: 22px; font-weight: 800; letter-spacing: 4px;"
        )
        outer.addWidget(core_title)

        self._core_status = QLabel("[ TRẠNG THÁI: SẴN SÀNG NHẬN DIỆN ]")
        self._core_status.setStyleSheet("color: #4edea3; font-weight: 700; font-size: 13px;")
        self._core_status.setAlignment(Qt.AlignCenter)
        outer.addWidget(self._core_status)

        bars_wrap = QVBoxLayout()
        bars_wrap.setContentsMargins(40, 12, 40, 0)
        bars_wrap.setSpacing(10)

        bars_wrap.addWidget(self._create_loading_bar("Mô hình Anti-Spoofing (YOLOv8)", 100))
        bars_wrap.addWidget(self._create_loading_bar("Mô hình InsightFace", 100))
        bars_wrap.addWidget(self._create_loading_bar("Đồng bộ FAISS Vector DB", 100))

        outer.addLayout(bars_wrap)

        btn_go = QPushButton("▶  Bắt Đầu Điểm Danh")
        btn_go.setCursor(Qt.PointingHandCursor)
        btn_go.setFixedHeight(42)
        btn_go.setMinimumWidth(220)
        btn_go.setStyleSheet("""
            QPushButton {
                background-color: #4cd7f6;
                color: #0b1326;
                border: none;
                border-radius: 10px;
                font-weight: 700;
                font-size: 13px;
                padding: 0 24px;
            }
            QPushButton:hover { background-color: #6fe2ff; }
            QPushButton:pressed { background-color: #2ca0ba; }
        """)
        btn_go.clicked.connect(lambda: self.navigate_requested.emit(1))
        outer.addWidget(btn_go, alignment=Qt.AlignCenter)

        return frame

    def _build_stat_card(self, title: str, value: str, value_color: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #161f2e;
                border: 1px solid #1d6475;
                border-radius: 16px;
            }
            QFrame:hover {
                border: 1px solid #2ca0ba;
            }
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(6)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #8c909f; font-size: 12px;")

        lbl_value = QLabel(value)
        lbl_value.setObjectName("stat_value")
        lbl_value.setStyleSheet(f"font-size: 24px; font-weight: 800; color: {value_color};")

        layout.addWidget(lbl_title)
        layout.addWidget(lbl_value)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return card

    def _create_loading_bar(self, label_text: str, value: int) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        lbl = QLabel(label_text)
        lbl.setStyleSheet("color: #8c909f; font-size: 12px;")
        lbl.setFixedWidth(210)

        bar = QProgressBar()
        bar.setValue(value)
        bar.setTextVisible(False)
        bar.setFixedHeight(8)
        bar.setStyleSheet("""
            QProgressBar {
                background-color: #0e1627;
                border: 1px solid #122e38;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #4cd7f6;
                border-radius: 3px;
            }
        """)

        layout.addWidget(lbl)
        layout.addWidget(bar)
        return widget

    # ── Cột phải: Nhật ký điểm danh ──────────────────────────────────────────
    def _build_right_column(self) -> QVBoxLayout:
        col = QVBoxLayout()

        log_frame = QFrame()
        log_frame.setStyleSheet("""
            QFrame {
                background-color: #161f2e;
                border: 1px solid #1d6475;
                border-radius: 16px;
            }
        """)
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(20, 20, 20, 20)
        log_layout.setSpacing(16)

        title_row = QHBoxLayout()
        log_title = QLabel("Nhật Ký Hôm Nay")
        log_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #f8fafc;")
        title_row.addWidget(log_title)
        title_row.addStretch()

        self._log_count_badge = QLabel("0")
        self._log_count_badge.setAlignment(Qt.AlignCenter)
        self._log_count_badge.setFixedSize(28, 22)
        self._log_count_badge.setStyleSheet("""
            background-color: #0e2a32; color: #4cd7f6;
            border-radius: 11px; font-size: 11px; font-weight: 700;
        """)
        title_row.addWidget(self._log_count_badge)
        log_layout.addLayout(title_row)

        self._log_table = QTableWidget(0, 3)
        self._log_table.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                alternate-background-color: #0d1720;
                border: none;
                gridline-color: #102630;
                color: #dae2fd;
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
                font-size: 11px;
            }
        """)
        self._log_table.verticalHeader().setVisible(False)
        self._log_table.setHorizontalHeaderLabels(["", "Họ Tên / Mã NV", "Giờ"])
        self._log_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._log_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._log_table.setShowGrid(False)
        self._log_table.setAlternatingRowColors(True)

        header = self._log_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

        log_layout.addWidget(self._log_table)

        self._log_empty_label = QLabel("Chưa có lượt điểm danh nào trong hôm nay.")
        self._log_empty_label.setAlignment(Qt.AlignCenter)
        self._log_empty_label.setStyleSheet("color: #475569; font-size: 12px; padding: 24px 0;")
        self._log_empty_label.hide()
        log_layout.addWidget(self._log_empty_label)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        btn_history = QPushButton("Xem Toàn Bộ Lịch Sử")
        btn_history.setCursor(Qt.PointingHandCursor)
        btn_history.setMinimumHeight(40)
        btn_history.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #e2e8f0;
                border: 1px solid #475569;
                border-radius: 8px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #475569; }
        """)
        btn_history.clicked.connect(lambda: self.navigate_requested.emit(4))

        btn_refresh = QPushButton("↺  Làm Mới")
        btn_refresh.setCursor(Qt.PointingHandCursor)
        btn_refresh.setMinimumHeight(40)
        btn_refresh.setStyleSheet("""
            QPushButton {
                background-color: #0d1720;
                color: #4cd7f6;
                border: 1px solid #4cd7f6;
                border-radius: 8px;
                font-weight: 700;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #4cd7f6;
                color: #0b1326;
            }
        """)
        btn_refresh.clicked.connect(self.refresh_data)

        btn_layout.addWidget(btn_history)
        btn_layout.addWidget(btn_refresh)
        log_layout.addLayout(btn_layout)

        col.addWidget(log_frame)
        return col

    # ── Đọc và Xử lý Dữ liệu Realtime (Đã Fix Logic NGÀY HÔM NAY) ────────────
    def _setup_refresh_timer(self) -> None:
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh_data)
        self._refresh_timer.start(10_000)

    @Slot()
    def refresh_data(self) -> None:
        try:
            self._refresh_stat_cards()
            self._refresh_log_table()
        except Exception as exc:
            logger.exception("HomeView: lỗi tổng quát khi refresh dữ liệu")

    def _refresh_stat_cards(self) -> None:
        # Cập nhật số lượng nhân viên
        employees = self._db.get_all_employees(active_only=True)
        self._set_stat_value(self._card_total_nv, str(len(employees)))

        # Cập nhật trạng thái ca mở
        active_session = self._db.get_active_session()
        if active_session:
            self._set_stat_value(self._card_session, active_session.session_name)
            self._core_status.setText("[ TRẠNG THÁI: ĐANG ĐIỂM DANH ]")
            self._status_dot.setStyleSheet("color: #4cd7f6; font-weight: bold; font-size: 11px;")
            self._status_text.setText("Ca làm việc đang mở  •  Engine: ONNX Runtime")
        else:
            self._set_stat_value(self._card_session, "Không có")
            self._core_status.setText("[ TRẠNG THÁI: SẴN SÀNG NHẬN DIỆN ]")
            self._status_dot.setStyleSheet("color: #4edea3; font-weight: bold; font-size: 11px;")
            self._status_text.setText("Hệ thống hoạt động  •  Engine: ONNX Runtime")

        # ĐẾM ĐIỂM DANH HÔM NAY (Độc lập với việc có ca đang mở hay không)
        today = datetime.now().date()
        today_count = 0
        try:
            sessions = self._db.get_all_sessions()
            for item in sessions:
                s = item[0] if isinstance(item, tuple) else item
                atts = self._db.get_attendance_by_session(s.id)
                if atts:
                    for att in atts:
                        # Rút trích thời gian an toàn
                        att_ts = att.timestamp
                        if isinstance(att_ts, str):
                            try:
                                ts_str = att_ts.split(".")[0]
                                dt_obj = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                            except Exception:
                                continue # Bỏ qua bản ghi lỗi thay vì nhận diện sai thành hôm nay
                        else:
                            dt_obj = att_ts
                        
                        # Cộng dồn nếu đúng là ngày hôm nay
                        if dt_obj.date() == today:
                            today_count += 1
        except Exception as exc:
            logger.exception("HomeView: Lỗi đếm số lượng điểm danh hôm nay")

        self._set_stat_value(self._card_today, str(today_count))

    def _set_stat_value(self, card: QFrame, text: str) -> None:
        value_label = card.findChild(QLabel, "stat_value")
        if value_label:
            value_label.setText(text)

    def _refresh_log_table(self) -> None:
        """Thu thập dữ liệu từ tất cả các ca, CHỈ LỌC lấy dữ liệu của hôm nay."""
        rows: list[tuple[str, str, str]] = []
        all_attendances = []

        try:
            # Lấy tất cả lượt điểm danh để lọc
            sessions = self._db.get_all_sessions()
            for item in sessions:
                s = item[0] if isinstance(item, tuple) else item
                atts = self._db.get_attendance_by_session(s.id)
                if atts:
                    all_attendances.extend(atts)
        except Exception as exc:
            logger.exception("HomeView: Không thể đọc danh sách ca làm việc từ DB")

        today = datetime.now().date()
        parsed_records = []

        for att in all_attendances:
            emp = att.employee
            if not emp:
                continue

            # Rút trích thời gian an toàn
            att_ts = att.timestamp
            if isinstance(att_ts, str):
                try:
                    ts_str = att_ts.split(".")[0]
                    dt_obj = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue  # Bỏ qua dòng dữ liệu bị lỗi format
            else:
                dt_obj = att_ts

            # BỘ LỌC CỐT LÕI: Chỉ nạp dữ liệu của ngày hôm nay vào mảng
            if dt_obj.date() == today:
                parsed_records.append((dt_obj, emp.name, emp.emp_code))

        # Sắp xếp mới nhất lên đầu và lấy tối đa 8 dòng hiển thị trên Dashboard
        parsed_records.sort(key=lambda x: x[0], reverse=True)
        recent_records = parsed_records[:8]

        for dt_obj, name, code in recent_records:
            time_str = dt_obj.strftime("%H:%M:%S")
            rows.append(("👤", f"{name}\nID: {code}", time_str))

        self._log_count_badge.setText(str(len(rows)))
        self._log_table.setRowCount(0)

        # Xử lý trường hợp không có lượt điểm danh nào HÔM NAY
        if not rows:
            self._log_table.hide()
            self._log_empty_label.show()
            return

        self._log_table.show()
        self._log_empty_label.hide()

        # Hiển thị dữ liệu
        for row_idx, (icon, name_id, time_str) in enumerate(rows):
            self._log_table.insertRow(row_idx)
            self._log_table.setRowHeight(row_idx, 46) 
            
            self._log_table.setItem(row_idx, 0, QTableWidgetItem(icon))
            
            name_item = QTableWidgetItem(name_id)
            name_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self._log_table.setItem(row_idx, 1, name_item)
            
            time_item = QTableWidgetItem(time_str)
            time_item.setTextAlignment(Qt.AlignCenter)
            self._log_table.setItem(row_idx, 2, time_item)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    def stop_animations(self) -> None:
        self._radar.stop()
        self._clock_timer.stop()
        self._refresh_timer.stop()

    def resume_animations(self) -> None:
        self._radar._timer.start(30)
        self._clock_timer.start(1000)
        self._refresh_timer.start(10_000)
        self.refresh_data()