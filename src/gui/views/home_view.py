"""
src/gui/views/home_view.py
Trang chủ (Dashboard) mang phong cách Cyberpunk / High-Tech.
"""
from __future__ import annotations
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QVBoxLayout, 
    QWidget, QSizePolicy, QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QProgressBar
)

class HomeView(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # =========================================================
        # 1. TOP APP BAR (Header)
        # =========================================================
        self.header_frame = QFrame()
        self.header_frame.setFixedHeight(72)
        self.header_frame.setStyleSheet("""
            QFrame {
                background-color: #0b1326;
                border-bottom: 2px solid #2ca0ba;
            }
        """)
        
        header_layout = QHBoxLayout(self.header_frame)
        header_layout.setContentsMargins(32, 0, 32, 0)
        
        title_box = QVBoxLayout()
        title_box.setAlignment(Qt.AlignVCenter)
        title_label = QLabel("Bảng Điều Khiển Hệ Thống")
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #f8fafc;")
        
        status_layout = QHBoxLayout()
        status_dot = QLabel("●")
        status_dot.setStyleSheet("color: #4cd7f6; font-weight: bold;")
        status_text = QLabel("Hệ Thống Hoạt Động  |  FPS: 30  |  Engine: ONNX")
        status_text.setStyleSheet("color: #8c909f;")
        status_layout.addWidget(status_dot)
        status_layout.addWidget(status_text)
        status_layout.addStretch()
        
        title_box.addWidget(title_label)
        title_box.addLayout(status_layout)
        header_layout.addLayout(title_box)
        header_layout.addStretch()

        user_info = QLabel("Admin User\nID: 10294")
        user_info.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        user_info.setStyleSheet("color: #4cd7f6; font-weight: bold;")
        header_layout.addWidget(user_info)

        main_layout.addWidget(self.header_frame)

        # =========================================================
        # 2. KHU VỰC NỘI DUNG CHÍNH (Chia 2 cột)
        # =========================================================
        content_container = QWidget()
        content_layout = QHBoxLayout(content_container)
        content_layout.setContentsMargins(32, 24, 32, 32)
        content_layout.setSpacing(24)

        # ─── CỘT TRÁI (Tỷ lệ 2) ──────────────────────────────────
        left_col = QVBoxLayout()
        left_col.setSpacing(24)

        # A. AI Core Engine
        ai_core_frame = QFrame()
        ai_core_frame.setStyleSheet("""
            QFrame {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #060e20, stop:1 #0f1e3c);
                border: 2px solid #2ca0ba;
                border-radius: 16px;
            }
        """)
        ai_core_layout = QVBoxLayout(ai_core_frame)
        ai_core_layout.setAlignment(Qt.AlignCenter)
        ai_core_layout.setSpacing(20)

        core_icon = QLabel("👁️")
        font = core_icon.font()
        font.setPointSize(48)
        core_icon.setFont(font)
        core_icon.setAlignment(Qt.AlignCenter)
        
        core_title = QLabel("SECUREFACE AI ENGINE")
        core_title.setAlignment(Qt.AlignCenter)
        core_title.setStyleSheet("color: #4cd7f6; font-size: 24px; font-weight: bold; letter-spacing: 4px;")
        
        core_status = QLabel("[ TRẠNG THÁI: SẴN SÀNG NHẬN DIỆN ]")
        core_status.setStyleSheet("color: #4edea3; font-weight: bold;")
        core_status.setAlignment(Qt.AlignCenter)

        progress_layout = QVBoxLayout()
        progress_layout.setContentsMargins(40, 20, 40, 0)
        progress_layout.setSpacing(12)
        
        progress_layout.addWidget(self._create_loading_bar("Tải Mô Hình Anti-Spoofing (YOLOv8)", 100))
        progress_layout.addWidget(self._create_loading_bar("Tải Mô Hình InsightFace", 100))
        progress_layout.addWidget(self._create_loading_bar("Đồng Bộ FAISS Vector Database", 100))

        ai_core_layout.addStretch()
        ai_core_layout.addWidget(core_icon)
        ai_core_layout.addWidget(core_title)
        ai_core_layout.addWidget(core_status)
        ai_core_layout.addLayout(progress_layout)
        ai_core_layout.addStretch()
        
        left_col.addWidget(ai_core_frame, stretch=5)

        # B. Thanh thông số kỹ thuật
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(16)
        
        stats_layout.addWidget(self._create_stat_card("CPU Usage", "15%", "#4cd7f6"))
        stats_layout.addWidget(self._create_stat_card("RAM Allocation", "1.2 GB", "#4d8eff"))
        stats_layout.addWidget(self._create_stat_card("Latency", "12 ms", "#4edea3"))
        
        left_col.addLayout(stats_layout, stretch=1)

        # ─── CỘT PHẢI (Tỷ lệ 1) ──────────────────────────────────
        right_col = QVBoxLayout()
        
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

        log_title = QLabel("Nhật Ký Điểm Danh (Gần Đây)")
        log_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #f8fafc;")
        log_layout.addWidget(log_title)

        self.log_table = QTableWidget(4, 3) 
        self.log_table.setStyleSheet("""
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
                font-weight: bold;
                border: none;
                border-bottom: 1px solid #102630;
                padding: 10px;
                text-transform: uppercase;
            }
        """)
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.setHorizontalHeaderLabels(["NV", "Họ Tên / ID", "Giờ"])
        self.log_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.log_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.log_table.setShowGrid(False)

        header = self.log_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

        mock_data = [
            ("NV003", "Nguyễn Văn A", "08:15:22"),
            ("NV012", "Trần Thị B", "08:16:45"),
            ("NV045", "Lê Minh C", "08:20:10"),
            ("NV089", "Phạm Hoàng D", "08:22:30")
        ]
        for row, (uid, name, time_str) in enumerate(mock_data):
            self.log_table.setItem(row, 0, QTableWidgetItem("👤"))
            self.log_table.setItem(row, 1, QTableWidgetItem(f"{name}\nID: {uid}"))
            self.log_table.setItem(row, 2, QTableWidgetItem(time_str))

        log_layout.addWidget(self.log_table)

        btn_layout = QHBoxLayout()
        btn_stop = QPushButton("Kết Thúc Ca")
        btn_stop.setMinimumHeight(40)
        btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #e2e8f0;
                border: 1px solid #475569;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #475569; }
        """)
        
        btn_refresh = QPushButton("Làm Mới")
        btn_refresh.setMinimumHeight(40)
        btn_refresh.setStyleSheet("""
            QPushButton {
                background-color: #0d1720;
                color: #4cd7f6;
                border: 1px solid #4cd7f6;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4cd7f6;
                color: #0b1326;
            }
        """)
        
        btn_layout.addWidget(btn_stop)
        btn_layout.addWidget(btn_refresh)
        log_layout.addLayout(btn_layout)

        right_col.addWidget(log_frame)

        # ─── GỘP VÀO MAIN LAYOUT ─────────────────────────────────
        content_layout.addLayout(left_col, stretch=2)
        content_layout.addLayout(right_col, stretch=1)
        
        main_layout.addWidget(content_container, stretch=1)

    def _create_stat_card(self, title: str, value: str, value_color: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #161f2e;
                border: 1px solid #1d6475;
                border-radius: 16px;
            }
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #8c909f;")
        
        lbl_value = QLabel(value)
        lbl_value.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {value_color};")
        
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_value)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return card

    def _create_loading_bar(self, label_text: str, value: int) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        lbl = QLabel(label_text)
        lbl.setStyleSheet("color: #8c909f;")
        lbl.setFixedWidth(200)
        
        bar = QProgressBar()
        bar.setValue(value)
        bar.setStyleSheet("""
            QProgressBar {
                background-color: #0e1627;
                border: 1px solid #122e38;
                border-radius: 4px;
                text-align: center;
                color: transparent; 
                max-height: 8px;
            }
            QProgressBar::chunk {
                background-color: #4cd7f6; 
                border-radius: 3px;
            }
        """)
        
        layout.addWidget(lbl)
        layout.addWidget(bar)
        return widget