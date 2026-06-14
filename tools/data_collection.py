"""
tools/data_collection.py
Thu thập ảnh huấn luyện anti-spoofing qua webcam.

Phím tắt:
    r  — chụp ảnh REAL  → tools/dataset/real/
    f  — chụp ảnh FAKE  → tools/dataset/fake/
    s  — toggle show/hide hướng dẫn
    q  — thoát

Sử dụng:
    python tools/data_collection.py
    python tools/data_collection.py --cam 1 --size 320 --prefix session2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

# ── Thư mục gốc & dataset ────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "tools" / "dataset"
REAL_DIR    = DATASET_DIR / "real"
FAKE_DIR    = DATASET_DIR / "fake"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thu thập ảnh real/fake cho anti-spoofing")
    parser.add_argument("--cam",    type=int,   default=0,         help="Camera index (mặc định: 0)")
    parser.add_argument("--size",   type=int,   default=320,       help="Kích thước ảnh lưu (px vuông, mặc định: 320)")
    parser.add_argument("--prefix", type=str,   default="",        help="Tiền tố tên file (vd: session2 → session2_real_001.jpg)")
    parser.add_argument("--delay",  type=float, default=0.3,       help="Thời gian chờ tối thiểu giữa 2 lần chụp (giây)")
    return parser.parse_args()


def _count_existing(directory: Path) -> int:
    return len(list(directory.glob("*.jpg")))


def _save_image(
    frame:     "cv2.Mat",
    directory: Path,
    label:     str,
    prefix:    str,
    size:      int,
) -> Path:
    """Crop center-square, resize, lưu file, trả về Path."""
    h, w = frame.shape[:2]
    side  = min(h, w)
    y0    = (h - side) // 2
    x0    = (w - side) // 2
    crop  = frame[y0 : y0 + side, x0 : x0 + side]
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

    count   = _count_existing(directory) + 1
    pre     = f"{prefix}_" if prefix else ""
    fname   = f"{pre}{label}_{count:04d}.jpg"
    out_path = directory / fname
    cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out_path


def _draw_ui(
    frame:      "cv2.Mat",
    real_count: int,
    fake_count: int,
    show_help:  bool,
    last_msg:   str,
    msg_color:  tuple[int, int, int],
) -> "cv2.Mat":
    display = frame.copy()
    font    = cv2.FONT_HERSHEY_SIMPLEX

    # ── Center crop guide ────────────────────────────────────────────────
    h, w   = display.shape[:2]
    side   = min(h, w)
    x0, y0 = (w - side) // 2, (h - side) // 2
    cv2.rectangle(display, (x0, y0), (x0 + side, y0 + side), (80, 80, 80), 1)

    # ── Stat bar ─────────────────────────────────────────────────────────
    cv2.rectangle(display, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.putText(display, f"REAL: {real_count}", (10, 24),  font, 0.65, (80, 255, 80),  1)
    cv2.putText(display, f"FAKE: {fake_count}", (150, 24), font, 0.65, (80, 80, 255),  1)
    cv2.putText(display, "S: help  Q: quit",   (w - 190, 24), font, 0.55, (120, 120, 120), 1)

    # ── Flash message ─────────────────────────────────────────────────────
    if last_msg:
        (tw, th), _ = cv2.getTextSize(last_msg, font, 0.8, 2)
        mx = (w - tw) // 2
        my = h - 20
        cv2.rectangle(display, (mx - 8, my - th - 8), (mx + tw + 8, my + 8), (20, 20, 20), -1)
        cv2.putText(display, last_msg, (mx, my), font, 0.8, msg_color, 2)

    # ── Help overlay ──────────────────────────────────────────────────────
    if show_help:
        lines = [
            "  PHÍM TẮT  ",
            "",
            "  R  — Chụp ảnh REAL (khuôn mặt thật)",
            "  F  — Chụp ảnh FAKE (ảnh/video giả)",
            "  S  — Ẩn/hiện hướng dẫn này",
            "  Q  — Thoát",
            "",
            "  TIP: Giữ khuôn mặt trong ô vuông",
            "       Thay đổi góc, ánh sáng, khoảng cách",
        ]
        box_w, box_h = 360, len(lines) * 26 + 16
        bx, by = (w - box_w) // 2, (h - box_h) // 2
        overlay = display.copy()
        cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
        cv2.rectangle(display, (bx, by), (bx + box_w, by + box_h), (60, 60, 60), 1)
        for i, line in enumerate(lines):
            color = (200, 200, 200) if line.strip() else (80, 80, 80)
            if line.strip().startswith("R "):
                color = (80, 255, 80)
            elif line.strip().startswith("F "):
                color = (80, 80, 255)
            cv2.putText(display, line, (bx + 10, by + 24 + i * 26), font, 0.55, color, 1)

    return display


def collect(args: argparse.Namespace) -> None:
    # ── Tạo thư mục ──────────────────────────────────────────────────────
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    real_count = _count_existing(REAL_DIR)
    fake_count = _count_existing(FAKE_DIR)

    print(f"\n{'='*50}")
    print(f"  Data Collection — Anti-Spoofing")
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  REAL    : {real_count} ảnh có sẵn")
    print(f"  FAKE    : {fake_count} ảnh có sẵn")
    print(f"  Camera  : index {args.cam}")
    print(f"  Size    : {args.size}x{args.size}")
    print(f"{'='*50}\n")

    # ── Mở camera ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"❌ Không mở được camera index={args.cam}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    show_help  = True
    last_msg   = "Nhấn S để ẩn hướng dẫn"
    msg_color  = (180, 180, 180)
    msg_expire = time.monotonic() + 3.0
    last_shot  = 0.0

    print("🎥 Camera đang chạy. Nhấn S để xem phím tắt, Q để thoát.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Mất tín hiệu camera.")
            break

        now     = time.monotonic()
        expired = now > msg_expire

        display = _draw_ui(
            frame, real_count, fake_count,
            show_help,
            last_msg if not expired else "",
            msg_color,
        )
        cv2.imshow("Data Collection — FaceAttend", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            show_help = not show_help

        elif key in (ord("r"), ord("f")):
            if now - last_shot < args.delay:
                continue   # Cooldown chống chụp liên tục quá nhanh

            label     = "real" if key == ord("r") else "fake"
            directory = REAL_DIR if key == ord("r") else FAKE_DIR

            saved = _save_image(frame, directory, label, args.prefix, args.size)

            if key == ord("r"):
                real_count += 1
                last_msg    = f"✅ REAL #{real_count:04d} đã lưu"
                msg_color   = (80, 255, 80)
            else:
                fake_count += 1
                last_msg    = f"✅ FAKE #{fake_count:04d} đã lưu"
                msg_color   = (80, 80, 255)

            msg_expire = now + 1.5
            last_shot  = now
            print(f"  [{label.upper():4s}] {saved.name}")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"  Kết thúc thu thập.")
    print(f"  REAL: {real_count} ảnh  →  {REAL_DIR}")
    print(f"  FAKE: {fake_count} ảnh  →  {FAKE_DIR}")
    print(f"  Tổng: {real_count + fake_count} ảnh")
    print(f"{'='*50}\n")
    print("👉 Bước tiếp theo: chạy tools/train_yolo.py để huấn luyện model.\n")


if __name__ == "__main__":
    collect(parse_args())
