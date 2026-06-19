"""
tools/train_yolo.py
Huấn luyện YOLOv8 Object Detection để khoanh vùng và phân biệt Real/Fake face.
"""

import argparse
import random
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "tools" / "dataset"
YOLO_DIR = ROOT / "tools" / "yolo_dataset"

def prepare_yolo_structure(val_split: float = 0.2):
    """Gom ảnh và txt từ thư mục real/fake, chia train/val theo chuẩn YOLO Detection."""
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
        
    for split in ["train", "val"]:
        for d_type in ["images", "labels"]:
            (YOLO_DIR / d_type / split).mkdir(parents=True, exist_ok=True)

    all_data = []
    # Quét tất cả file txt, nếu có file txt thì lấy file jpg tương ứng (bỏ qua ảnh lỗi không có khuôn mặt)
    for label_dir in [DATASET_DIR / "real", DATASET_DIR / "fake"]:
        if not label_dir.exists():
            continue
        for txt_file in label_dir.glob("*.txt"):
            jpg_file = txt_file.with_suffix(".jpg")
            if jpg_file.exists():
                all_data.append((jpg_file, txt_file))

    if not all_data:
        print("❌ Không tìm thấy dữ liệu hợp lệ (Cần cả file .jpg và .txt). Hãy chạy lại tool data_collection.py!")
        exit(1)

    random.shuffle(all_data)
    val_size = max(1, int(len(all_data) * val_split))
    
    splits = {
        "val": all_data[:val_size],
        "train": all_data[val_size:]
    }

    for split_name, files in splits.items():
        for img_src, txt_src in files:
            shutil.copy(img_src, YOLO_DIR / "images" / split_name / img_src.name)
            shutil.copy(txt_src, YOLO_DIR / "labels" / split_name / txt_src.name)
            
    return len(splits["train"]), len(splits["val"])

def write_yaml():
    yaml_data = {
        "path": str(YOLO_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": ["real", "fake"]
    }
    yaml_path = YOLO_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    return yaml_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()

    print("📦 Đang tổ chức lại Dataset sang cấu trúc YOLO Detection...")
    train_c, val_c = prepare_yolo_structure()
    print(f"📊 Dữ liệu: {train_c} Train | {val_c} Validation")
    
    yaml_path = write_yaml()
    print(f"📄 Đã tạo cấu hình: {yaml_path}")

    print("🚀 KHỞI ĐỘNG QUÁ TRÌNH HUẤN LUYỆN (OBJECT DETECTION)")
    # Train mô hình Detection (yolov8n.pt thay vì -cls.pt)
    model = YOLO("yolov8n.pt")
    
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(ROOT / "data" / "models"),
        name="anti_spoofing_det"
    )

if __name__ == "__main__":
    main()