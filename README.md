# 🎓 Hệ Thống Chấm Công Nhận Diện Khuôn Mặt (Phiên bản Tối Ưu Hóa)
Dự án Hệ thống điểm danh tự động bằng khuôn mặt, được tái cấu trúc theo chuẩn Clean Architecture (MVC) nhằm mang lại trải nghiệm mượt mà, tối ưu hóa hiệu suất CPU và sẵn sàng mở rộng cho quy mô doanh nghiệp.

##  ✨ Điểm Mới & Tính Năng Nổi Bật
So với phiên bản cũ, kiến trúc mới tập trung giải quyết triệt để các vấn đề về nghẽn cổ chai hiệu năng:

- 🚀 Video Mượt Mà (No Lag): Tách biệt luồng Camera (hiển thị UI) và luồng AI (phân tích ảnh) bằng threading và QThread, đảm bảo video luôn chạy ở tốc độ 30 FPS.

- ⚡ Tăng Tốc AI với ONNX: Thay thế PyTorch nặng nề bằng onnxruntime, giúp mô hình YOLO Anti-spoofing chạy cực nhanh, giảm thời gian xử lý xuống mức thấp nhất trên CPU.

- 🔍 Tìm Kiếm Vector Siêu Tốc: Loại bỏ numpy.dot, tích hợp faiss-cpu để lập chỉ mục khuôn mặt. Tốc độ nhận diện không bị suy giảm ngay cả khi dữ liệu lên tới hàng ngàn nhân viên.

- 🛡️ Quản Lý Dữ Liệu An Toàn: Sử dụng SQLAlchemy (ORM) để tương tác với SQLite, chống lỗi "database locked" và hỗ trợ các thao tác CRUD chuẩn xác qua employee_manager.

- 🎨 Giao Diện Chuyên Nghiệp: Nâng cấp sang framework PySide6 với thiết kế tab-based hiện đại, chia rõ các màn hình điểm danh, thêm nhân viên và quản lý lịch sử.


## 💻 Công Nghệ Sử Dụng
- Core AI & Computer Vision: insightface, opencv-python, onnxruntime.

- Vector Database: faiss-cpu.

- Relational Database: SQLAlchemy + SQLite.

- GUI Framework: PySide6.

## 📁 Cấu Trúc Thư Mục (Clean Architecture)
Dự án được phân tách rõ ràng giữa Dữ Liệu (Data), Lõi Nghiệp Vụ (Core), Giao Diện (GUI) và Tiện Ích (Utils):

```
face-attendance-optimized/
│
├── data/                       # Thư mục chứa dữ liệu tĩnh (BẮT BUỘC thêm vào .gitignore)
│   ├── database/               # File CSDL cục bộ (vd: attendance.db)
│   ├── faiss_index/            # Chứa file index của thuật toán FAISS (tìm kiếm vector)
│   ├── models/                 # Chứa các file trọng số AI (best.onnx, insightface models)
│   └── exports/                # Nơi lưu file báo cáo CSV/Excel khi xuất ra từ phần mềm
│
├── src/                        # Thư mục chứa mã nguồn chính của ứng dụng
│   ├── __init__.py
│   ├── core/                   # Lớp xử lý nghiệp vụ lõi (AI & Xử lý ảnh)
│   │   ├── __init__.py
│   │   ├── camera_stream.py    # Xử lý đọc luồng Webcam đa luồng (chống giật lag)
│   │   ├── anti_spoofing.py    # Class chạy mô hình kiểm tra thật/giả (ONNX Runtime)
│   │   ├── face_recognizer.py  # Class trích xuất đặc trưng và so khớp (InsightFace + FAISS)
│   │   └── employee_manager.py # [MỚI] Class quản lý logic nhân viên (đồng bộ DB và FAISS khi thêm/xóa)
│   │
│   ├── database/               # Lớp tương tác Cơ sở dữ liệu (Database Layer)
│   │   ├── __init__.py
│   │   ├── models.py           # Định nghĩa các bảng CSDL (Employee, Attendance, Session) bằng SQLAlchemy
│   │   └── db_manager.py       # Chứa các hàm CRUD (Thêm/Sửa/Xóa nhân viên, Ghi log)
│   │
│   ├── gui/                    # Lớp Giao diện người dùng (Presentation Layer)
│   │   ├── __init__.py
│   │   ├── components/         # Các khối giao diện nhỏ dùng chung (Nút bấm, Khung Camera, Bảng table)
│   │   ├── threads/            # Các luồng (Worker Threads) để cập nhật UI từ kết quả AI
│   │   ├── views/              # Các màn hình chính của ứng dụng
│   │   │   ├── main_window.py        # Bộ khung cửa sổ chính (Điều hướng các tab)
│   │   │   ├── attendance_view.py    # Màn hình chức năng điểm danh
│   │   │   ├── enroll_view.py        # Màn hình thêm khuôn mặt nhân viên mới
│   │   │   ├── history_view.py       # Màn hình xem lịch sử và xuất báo cáo
│   │   │   └── employee_list_view.py # [MỚI] Màn hình hiển thị table, tìm kiếm và xóa danh sách nhân viên
│   │   └── styles/             # Các file định dạng giao diện (CSS/QSS)
│   │
│   └── utils/                  # Các hàm tiện ích và cấu hình hệ thống
│       ├── __init__.py
│       ├── config.py           # File lưu TẤT CẢ cấu hình: ngưỡng threshold, kích thước khung hình, path...
│       ├── logger.py           # Cấu hình ghi log lỗi của hệ thống
│       └── helpers.py          # Các hàm hỗ trợ (vẽ bounding box, tính FPS, chuẩn hóa chuỗi)
│
├── tools/                      # Các kịch bản chạy độc lập
│   ├── train_yolo.py           # Script huấn luyện lại mô hình phân biệt thật/giả
│   ├── export_onnx.py          # Script chuyển file best.pt sang best.onnx để tối ưu tốc độ
│   └── data_collection.py      # Script phụ để chụp ảnh thu thập dữ liệu
│
├── assets/                     # Tài nguyên hình ảnh, âm thanh cho giao diện
│   ├── icons/                  # Các icon dùng trong phần mềm (icon nhân viên, điểm danh...)
│   ├── images/                 # Logo hiển thị ở trang chủ
│   └── sounds/                 # Âm thanh thông báo (thành công, cảnh báo giả mạo)
│
├── logs/                       # Thư mục chứa các file log (.txt, .log) khi ứng dụng chạy
├── .gitignore                  # Khai báo những file không đẩy lên Github (như data/, logs/, venv/)
├── .env                        # Chứa các biến môi trường cấu hình động (tùy chọn)
├── requirements.txt            # Danh sách thư viện đã tối ưu (onnxruntime, faiss-cpu, sqlalchemy...)
└── main.py                     # File gốc duy nhất dùng để khởi chạy phần mềm
```


## 🚀 Hướng Dẫn Cài Đặt
### 1. Yêu Cầu Hệ Thống
- Python 3.10 trở lên.

- Webcam (Tích hợp hoặc cắm ngoài).

### 2. Khởi tạo môi trường
Khuyến nghị sử dụng môi trường ảo (Virtual Environment) để tránh xung đột thư viện.

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```
(Lưu ý: Nếu sử dụng InsightFace trên Windows, bạn có thể cần cài đặt gói .whl tương ứng với phiên bản Python của mình).

### 4. Cấu hình Dữ liệu và Mô hình
- Đảm bảo bạn đã đưa file mô hình YOLO (best.onnx) và các model của InsightFace vào thư mục data/models/.
```bash
pip install insightface-0.7.3-cp310-cp310-win_amd64.whl
```
- Hệ thống sẽ tự động khởi tạo file CSDL và FAISS index trong lần chạy đầu tiên.


### 5. Thu thập dữ liệu huấn luyện (Data Collection)
- Chạy lệnh sau trong Terminal:
```bash
python tools/data_collection.py
```
- Camera sẽ bật lên. Hãy làm theo hướng dẫn:
    - Nhìn thẳng vào camera, đảm bảo khung viền xanh lá (Face Locked) xuất hiện trên mặt bạn.

    - Thu thập mặt thật (Real): Bấm phím r nhiều lần ở nhiều góc độ khác nhau (cười, nghiêm túc, nghiêng đầu nhẹ...).

    - Thu thập mặt giả (Fake): Lấy điện thoại mở một bức ảnh khuôn mặt của bạn (hoặc ai đó) rồi giơ lên trước webcam sao cho nó bắt được khung viền xanh lá. Sau đó bấm phím f nhiều lần để chụp lại.

    - Số lượng: Để mô hình thông minh, bạn nên cố gắng bấm chụp khoảng 50 - 100 tấm Real và 50 - 100 tấm Fake.

- Bấm phím q để thoát camera sau khi thu thập đủ.

### 6. Huấn luyện và Đóng gói Mô hình AI (Train & Export)
- Bắt đầu huấn luyện: Chạy lệnh sau:
```bash
python tools/train_yolo.py
```
Tool này sẽ tự động tải mô hình YOLOv8n về, chia dữ liệu bạn vừa chụp và bắt đầu huấn luyện. Quá trình này có thể mất từ vài phút đến hơn mười phút tùy vào cấu hình máy (hiện tại trong file bạn set 50 epochs).

- Đóng gói ra file .onnx: Sau khi terminal báo Train xong, bạn chạy tiếp lệnh này để xuất file chạy nhẹ cho phần mềm:
```bash
python tools/export_onnx.py
```
Hệ thống sẽ chuyển đổi file và tự động lưu best.onnx vào đúng thư mục data/models/ cho phần mềm chính sử dụng.


### 7. Khởi chạy Ứng dụng
```bash
python main.py
```

## 🛠️ Kịch Bản Công Cụ (Tools)
Nếu bạn muốn tự huấn luyện lại AI hoặc chuyển đổi mô hình, hãy sử dụng các script trong thư mục tools/:

- python tools/train_yolo.py: Huấn luyện lại mô hình phát hiện khuôn mặt thật/giả.

- python tools/export_onnx.py: Chuyển đổi file trọng số .pt sang .onnx để tăng tốc suy luận.

## 👤 Tác Giả
- Sinh viên thực hiện: Nguyễn Hoàng Tùng

- Chuyên ngành: Công nghệ Thông tin

- Trường: Đại học Giao thông Vận tải Thành phố Hồ Chí Minh (UTH)
