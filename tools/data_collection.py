"""
tools/data_collection.py
Thu thập ảnh huấn luyện anti-spoofing và TỰ ĐỘNG GÁN NHÃN (Auto-Annotation) theo chuẩn YOLO.
Chỉ lưu ảnh khi phát hiện có khuôn mặt trong khung hình.
"""

import argparse
import time
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "tools" / "dataset"
REAL_DIR = DATASET_DIR / "real"
FAKE_DIR = DATASET_DIR / "fake"

for d in [REAL_DIR, FAKE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Khởi tạo bộ nhận diện khuôn mặt cơ bản của OpenCV (Nhanh, nhẹ, chạy CPU)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--delay", type=float, default=0.5, help="Cooldown giữa các lần chụp")
    return parser.parse_args()

def save_yolo_data(frame, face_box, directory: Path, label: str, count: int) -> bool:
    """Lưu ảnh và file .txt YOLO format."""
    x, y, w, h = face_box
    img_h, img_w = frame.shape[:2]
    
    # Chuẩn hóa tọa độ theo chuẩn YOLO (0.0 -> 1.0)
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    
    class_id = 0 if label == "real" else 1
    
    base_name = f"{label}_{count:04d}_{int(time.time())}"
    img_path = directory / f"{base_name}.jpg"
    txt_path = directory / f"{base_name}.txt"
    
    try:
        cv2.imwrite(str(img_path), frame)
        with open(txt_path, "w") as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi lưu file: {e}")
        return False

def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.cam)
    
    if not cap.isOpened():
        print("❌ Không thể mở camera.")
        return

    real_count = len(list(REAL_DIR.glob("*.jpg")))
    fake_count = len(list(FAKE_DIR.glob("*.jpg")))
    last_shot = 0.0

    print("🎥 HỆ THỐNG THU THẬP & TỰ ĐỘNG GÁN NHÃN ANTI-SPOOFING")
    print(" Bấm 'r' để thu thập ảnh THẬT (Real)")
    print(" Bấm 'f' để thu thập ảnh GIẢ (Fake)")
    print(" Bấm 'q' để thoát\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Nhận diện khuôn mặt
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        best_face = None
        
        if len(faces) > 0:
            # Ưu tiên lấy khuôn mặt to nhất trong khung hình
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            best_face = faces[0]
            x, y, w, h = best_face
            # Vẽ viền xanh báo hiệu đã lock được khuôn mặt
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display, "Face Locked", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(display, f"Real: {real_count} | Fake: {fake_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Data Collection (Auto-Annotation)", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in (ord('r'), ord('f')):
            now = time.time()
            if now - last_shot < args.delay:
                continue
                
            if best_face is None:
                print("⚠️  Không tìm thấy khuôn mặt! Vui lòng nhìn thẳng vào camera.")
                continue

            label = "real" if key == ord('r') else "fake"
            target_dir = REAL_DIR if label == "real" else FAKE_DIR
            current_count = real_count if label == "real" else fake_count
            
            if save_yolo_data(frame, best_face, target_dir, label, current_count + 1):
                if label == "real":
                    real_count += 1
                else:
                    fake_count += 1
                print(f"✅ Đã lưu {label.upper()} #{current_count + 1} (Kèm Bounding Box)")
                last_shot = now

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()