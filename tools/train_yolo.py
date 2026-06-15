"""
tools/train_yolo.py
Huấn luyện mô hình YOLO phân biệt khuôn mặt thật/giả (anti-spoofing).
Tự động tạo data.yaml từ dataset thu thập bằng data_collection.py.

Sử dụng:
    python tools/train_yolo.py
    python tools/train_yolo.py --epochs 100 --imgsz 320 --batch 16 --device cpu
    python tools/train_yolo.py --prepare-only   # Chỉ tạo data.yaml, không train
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

# ── Đường dẫn ────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
TOOLS_DIR   = ROOT / "tools"
DATASET_DIR = TOOLS_DIR / "dataset"
YOLO_DIR    = TOOLS_DIR / "yolo_dataset"   # Cấu trúc YOLO chuẩn
DATA_YAML   = TOOLS_DIR / "data.yaml"
RUNS_DIR    = ROOT / "data" / "models" / "runs"   # Output checkpoint


# ── Class labels ─────────────────────────────────────────────────────────────
CLASSES = ["real", "fake"]   # index 0=real, 1=fake (phải khớp với anti_spoofing.py)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO anti-spoofing model")
    parser.add_argument("--model",        type=str,   default="yolov8n.pt",
                        help="Model base (mặc định: yolov8n.pt — tự tải nếu chưa có)")
    parser.add_argument("--epochs",       type=int,   default=50,
                        help="Số epoch (mặc định: 50)")
    parser.add_argument("--imgsz",        type=int,   default=320,
                        help="Kích thước ảnh input (mặc định: 320, khớp SPOOFING_INPUT_SIZE)")
    parser.add_argument("--batch",        type=int,   default=16,
                        help="Batch size (mặc định: 16; giảm xuống 8 nếu RAM thấp)")
    parser.add_argument("--device",       type=str,   default="cpu",
                        help="Thiết bị: cpu | 0 | 0,1 (mặc định: cpu)")
    parser.add_argument("--val-split",    type=float, default=0.15,
                        help="Tỉ lệ ảnh dành cho validation (mặc định: 0.15)")
    parser.add_argument("--workers",      type=int,   default=2,
                        help="Số worker dataloader (mặc định: 2)")
    parser.add_argument("--patience",     type=int,   default=15,
                        help="Early stopping patience (mặc định: 15)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Chỉ chuẩn bị dataset + data.yaml, không train")
    parser.add_argument("--resume",       action="store_true",
                        help="Tiếp tục từ checkpoint gần nhất")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Bước 1: Chuyển dataset flat (real/ fake/) → cấu trúc YOLO classification
# ─────────────────────────────────────────────────────────────────────────────
def prepare_yolo_dataset(val_split: float) -> dict[str, int]:
    """
    Chuyển:
        tools/dataset/real/*.jpg
        tools/dataset/fake/*.jpg
    Thành:
        tools/yolo_dataset/
            train/real/*.jpg  train/fake/*.jpg
            val/real/*.jpg    val/fake/*.jpg

    YOLOv8 classification dùng cấu trúc ImageFolder (thư mục = class name).
    Trả về dict thống kê số ảnh mỗi split.
    """
    stats: dict[str, int] = {"train_real": 0, "train_fake": 0,
                              "val_real": 0,   "val_fake": 0}

    for split in ("train", "val"):
        for cls in CLASSES:
            (YOLO_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        src_dir = DATASET_DIR / cls
        if not src_dir.exists():
            print(f"  ⚠ Không tìm thấy thư mục: {src_dir}")
            continue

        images = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.png"))
        if not images:
            print(f"  ⚠ Không có ảnh trong: {src_dir}")
            continue

        random.shuffle(images)
        n_val   = max(1, int(len(images) * val_split))
        val_set = set(img.name for img in images[:n_val])

        for img in images:
            split    = "val" if img.name in val_set else "train"
            dst      = YOLO_DIR / split / cls / img.name
            shutil.copy2(img, dst)
            stats[f"{split}_{cls}"] += 1

        print(f"  {cls:4s}: {len(images)} ảnh  →  "
              f"train={len(images)-n_val}  val={n_val}")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Bước 2: Tạo data.yaml
# ─────────────────────────────────────────────────────────────────────────────
def write_data_yaml() -> Path:
    """Ghi file data.yaml theo định dạng YOLOv8 classification."""
    content = f"""\
# data.yaml — Anti-Spoofing dataset (YOLOv8 classification)
# Tạo tự động bởi tools/train_yolo.py

path: {YOLO_DIR}       # Thư mục gốc dataset
train: train           # Thư mục train (relative to path)
val:   val             # Thư mục val   (relative to path)

nc: {len(CLASSES)}             # Số class
names: {CLASSES}     # 0=real  1=fake
"""
    DATA_YAML.write_text(content, encoding="utf-8")
    return DATA_YAML


# ─────────────────────────────────────────────────────────────────────────────
# Bước 3: Huấn luyện
# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Chưa cài ultralytics:  pip install ultralytics")
        sys.exit(1)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if args.resume:
        # Tìm checkpoint last.pt gần nhất trong RUNS_DIR
        ckpts = sorted(RUNS_DIR.rglob("last.pt"))
        if not ckpts:
            print("❌ Không tìm thấy checkpoint để resume.")
            sys.exit(1)
        model_path = str(ckpts[-1])
        print(f"▶ Resume từ: {model_path}")
    else:
        model_path = args.model   # "yolov8n.pt" sẽ tự tải từ ultralytics hub

    model = YOLO(model_path)

    print(f"\n🚀 Bắt đầu huấn luyện…")
    print(f"   Model  : {model_path}")
    print(f"   Data   : {DATA_YAML}")
    print(f"   Epochs : {args.epochs}")
    print(f"   imgsz  : {args.imgsz}")
    print(f"   Batch  : {args.batch}")
    print(f"   Device : {args.device}")
    print(f"   Output : {RUNS_DIR}\n")

    t0 = time.time()

    results = model.train(
        task      = "classify",          # Phân loại ảnh (không phải detect)
        data      = str(DATA_YAML),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        device    = args.device,
        workers   = args.workers,
        patience  = args.patience,       # Early stopping
        project   = str(RUNS_DIR),
        name      = "anti_spoof",
        exist_ok  = True,
        resume    = args.resume,
        # Augmentation — quan trọng để model tổng quát hóa tốt
        flipud    = 0.0,                 # Không lật dọc (khuôn mặt luôn đứng)
        fliplr    = 0.5,                 # Lật ngang OK
        hsv_h     = 0.015,
        hsv_s     = 0.4,
        hsv_v     = 0.4,
        translate = 0.1,
        scale     = 0.3,
        # Optimizer
        optimizer = "AdamW",
        lr0       = 0.001,
        lrf       = 0.01,
        weight_decay = 0.0005,
        warmup_epochs = 3,
        # Lưu
        save         = True,
        save_period  = 10,              # Lưu checkpoint mỗi 10 epoch
        plots        = True,
        verbose      = True,
    )

    elapsed = time.time() - t0
    best_pt = RUNS_DIR / "anti_spoof" / "weights" / "best.pt"

    print(f"\n{'='*55}")
    print(f"  ✅ Huấn luyện hoàn tất sau {elapsed/60:.1f} phút")
    if best_pt.exists():
        size_mb = best_pt.stat().st_size / (1024 * 1024)
        print(f"  📁 Best model : {best_pt}  ({size_mb:.1f} MB)")
        print(f"\n  👉 Bước tiếp theo:")
        print(f"     python tools/export_onnx.py --model {best_pt} --imgsz {args.imgsz}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    print(f"\n{'='*55}")
    print(f"  FaceAttend — YOLO Anti-Spoofing Trainer")
    print(f"{'='*55}")

    # ── Kiểm tra dataset ─────────────────────────────────────────────────
    real_imgs = list((DATASET_DIR / "real").glob("*.jpg")) if (DATASET_DIR / "real").exists() else []
    fake_imgs = list((DATASET_DIR / "fake").glob("*.jpg")) if (DATASET_DIR / "fake").exists() else []
    total     = len(real_imgs) + len(fake_imgs)

    print(f"\n📂 Dataset nguồn: {DATASET_DIR}")
    print(f"   real: {len(real_imgs)} ảnh")
    print(f"   fake: {len(fake_imgs)} ảnh")
    print(f"   Tổng: {total} ảnh")

    if total == 0:
        print("\n❌ Chưa có ảnh nào. Chạy data_collection.py trước:")
        print("   python tools/data_collection.py")
        sys.exit(1)

    if len(real_imgs) < 20 or len(fake_imgs) < 20:
        print(f"\n⚠ Dataset quá nhỏ (khuyến nghị ≥ 200 ảnh/class để có kết quả tốt).")
        print("  Tiếp tục thu thập thêm hoặc nhấn Enter để train thử…")
        input()

    # ── Chuẩn bị dataset ─────────────────────────────────────────────────
    print(f"\n📦 Đang tổ chức lại dataset theo cấu trúc YOLO…")
    stats = prepare_yolo_dataset(args.val_split)

    # ── Tạo data.yaml ────────────────────────────────────────────────────
    yaml_path = write_data_yaml()
    print(f"\n📄 Đã tạo: {yaml_path}")
    print(f"   train: real={stats['train_real']}  fake={stats['train_fake']}")
    print(f"   val  : real={stats['val_real']}    fake={stats['val_fake']}")

    if args.prepare_only:
        print("\n✅ --prepare-only: dừng tại đây. Chạy lại không có flag để train.")
        return

    # ── Train ────────────────────────────────────────────────────────────
    train(args)


if __name__ == "__main__":
    main()
