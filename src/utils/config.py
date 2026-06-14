"""
config.py - Cấu hình tập trung toàn hệ thống
Tất cả hằng số và đường dẫn được quản lý tại đây.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# ROOT & DATA PATHS
# ─────────────────────────────────────────────

# Thư mục gốc của dự án (hai cấp trên src/utils/)
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

DATA_DIR:        Path = ROOT_DIR / "data"
MODEL_DIR:       Path = DATA_DIR / "models"
DATABASE_DIR:    Path = DATA_DIR / "database"
FAISS_INDEX_DIR: Path = DATA_DIR / "faiss_index"
EXPORT_DIR:      Path = DATA_DIR / "exports"
LOG_DIR:         Path = ROOT_DIR / "logs"
ASSETS_DIR:      Path = ROOT_DIR / "assets"

# Tự động tạo nếu chưa tồn tại
for _dir in (MODEL_DIR, DATABASE_DIR, FAISS_INDEX_DIR, EXPORT_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────

# Anti-spoofing model (ONNX)
ANTI_SPOOFING_MODEL_PATH: Path = MODEL_DIR / "best.onnx"

# InsightFace model directory (buffalo_l hoặc buffalo_s)
INSIGHTFACE_MODEL_DIR: Path = MODEL_DIR / "insightface"

# SQLite database
DATABASE_PATH: Path = DATABASE_DIR / "attendance.db"
DATABASE_URL:  str  = f"sqlite:///{DATABASE_PATH}"

# FAISS index
FAISS_INDEX_PATH:    Path = FAISS_INDEX_DIR / "face_index.bin"
FAISS_METADATA_PATH: Path = FAISS_INDEX_DIR / "face_metadata.json"

# ─────────────────────────────────────────────
# CAMERA SETTINGS
# ─────────────────────────────────────────────

CAMERA_INDEX:   int = 0        # Index webcam (0 = mặc định)
CAMERA_WIDTH:   int = 1280     # Độ rộng khung hình (px)
CAMERA_HEIGHT:  int = 720      # Chiều cao khung hình (px)
CAMERA_FPS:     int = 30       # FPS mong muốn từ webcam
CAMERA_BUFFER:  int = 1        # Số frame buffered (giữ nhỏ để giảm độ trễ)

# ─────────────────────────────────────────────
# AI / DETECTION THRESHOLDS
# ─────────────────────────────────────────────

# Anti-spoofing (YOLO ONNX)
SPOOFING_CONFIDENCE_THRESHOLD: float = 0.75   # Dưới ngưỡng này → cảnh báo giả mạo
SPOOFING_INPUT_SIZE: tuple[int, int] = (320, 320)  # Kích thước input model YOLO

# Face recognition (InsightFace + FAISS)
FACE_DET_SCORE_THRESHOLD:   float = 0.60   # Ngưỡng tin cậy phát hiện khuôn mặt
FACE_RECOGNITION_THRESHOLD: float = 0.45   # Cosine distance — dưới ngưỡng này → khớp
FACE_EMBEDDING_DIM:         int   = 512    # Chiều vector đặc trưng (InsightFace buffalo)
FACE_MIN_SIZE:              int   = 60     # Kích thước khuôn mặt tối thiểu (px) để xử lý

# ─────────────────────────────────────────────
# PROCESSING INTERVALS
# ─────────────────────────────────────────────

# Chỉ chạy pipeline AI sau mỗi N frame (giảm tải CPU)
# Ví dụ: DETECT_INTERVAL=3 với 30 FPS → AI chạy ~10 lần/giây
DETECT_INTERVAL:       int = 3     # Khoảng cách frame giữa các lần inference
RECOGNITION_INTERVAL:  int = 5     # Khoảng cách frame riêng cho nhận diện danh tính

# ─────────────────────────────────────────────
# ATTENDANCE LOGIC
# ─────────────────────────────────────────────

# Thời gian tối thiểu (giây) giữa hai lần chấm công của cùng một nhân viên
ATTENDANCE_COOLDOWN_SECONDS: int = 30

# Số lần nhận diện thành công liên tiếp trước khi ghi log (chống nhận diện nhầm)
RECOGNITION_CONFIRM_COUNT: int = 3

# ─────────────────────────────────────────────
# ENROLL (ĐĂNG KÝ KHUÔN MẶT)
# ─────────────────────────────────────────────

ENROLL_CAPTURE_COUNT:    int = 10   # Số ảnh chụp khi đăng ký một nhân viên
ENROLL_CAPTURE_INTERVAL: int = 5    # Frame giữa mỗi lần chụp khi đăng ký

# ─────────────────────────────────────────────
# GUI DISPLAY
# ─────────────────────────────────────────────

DISPLAY_WIDTH:  int = 960   # Kích thước hiển thị video trong UI
DISPLAY_HEIGHT: int = 540

# Màu bounding box (BGR cho OpenCV)
COLOR_REAL:       tuple[int, int, int] = (0, 255, 0)    # Xanh lá — khuôn mặt thật
COLOR_SPOOF:      tuple[int, int, int] = (0, 0, 255)    # Đỏ — giả mạo
COLOR_UNKNOWN:    tuple[int, int, int] = (255, 165, 0)  # Cam — chưa nhận diện được
COLOR_RECOGNIZED: tuple[int, int, int] = (255, 255, 0)  # Vàng — đã nhận diện

FONT_SCALE:     float = 0.6
FONT_THICKNESS: int   = 2

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

LOG_LEVEL: str  = "INFO"   # DEBUG | INFO | WARNING | ERROR
LOG_MAX_BYTES:  int = 5 * 1024 * 1024   # 5 MB per log file
LOG_BACKUP_COUNT: int = 3               # Số file backup xoay vòng
