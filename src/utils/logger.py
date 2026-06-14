"""
src/utils/logger.py
Cấu hình logging tập trung: console (INFO) + file rotation (ERROR).
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

# Import lazy để tránh circular import khi config.py chưa sẵn sàng
def _get_log_config():
    try:
        from src.utils.config import LOG_DIR, LOG_LEVEL, LOG_MAX_BYTES, LOG_BACKUP_COUNT
        return LOG_DIR, LOG_LEVEL, LOG_MAX_BYTES, LOG_BACKUP_COUNT
    except ImportError:
        return Path("logs"), "INFO", 5 * 1024 * 1024, 3


_CONFIGURED = False   # Chỉ setup handler một lần duy nhất


def _setup() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    LOG_DIR, LOG_LEVEL, LOG_MAX_BYTES, LOG_BACKUP_COUNT = _get_log_config()
    LOG_DIR = Path(LOG_DIR)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOG_DIR / "app.log"
    level    = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    fmt_detailed = logging.Formatter(
        fmt     = "%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )
    fmt_console = logging.Formatter(
        fmt     = "%(asctime)s  [%(levelname)-8s]  %(message)s",
        datefmt = "%H:%M:%S",
    )

    # ── Console handler: INFO trở lên ──────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt_console)

    # ── File handler: ERROR trở lên, xoay vòng theo kích thước ────────
    file_handler = logging.handlers.RotatingFileHandler(
        filename    = log_file,
        maxBytes    = LOG_MAX_BYTES,
        backupCount = LOG_BACKUP_COUNT,
        encoding    = "utf-8",
    )
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(fmt_detailed)

    # ── Root logger ─────────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # Cho phép mọi level lọc tới handler
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Tắt propagation spam của thư viện bên thứ ba
    for noisy in ("insightface", "onnxruntime", "faiss", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Trả về logger theo tên module. Gọi một lần ở đầu mỗi file:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
    """
    _setup()
    return logging.getLogger(name)
