# Hướng dẫn Sử dụng

## Mục lục

1. [Cài đặt](#1-cài-đặt)
2. [Kiểm tra môi trường](#2-kiểm-tra-môi-trường)
3. [Chạy ứng dụng](#3-chạy-ứng-dụng)
4. [Phím tắt](#4-phím-tắt)
5. [Tùy chọn CLI](#5-tùy-chọn-cli)
6. [Tinh chỉnh hiệu năng](#6-tinh-chỉnh-hiệu-năng)
7. [Web UI](#7-web-ui)
8. [Xử lý sự cố](#8-xử-lý-sự-cố)
9. [Cấu hình nâng cao](#9-cấu-hình-nâng-cao)

---

## 1. Cài đặt

### Yêu cầu

- **Python** 3.10 trở lên
- **Windows** (hỗ trợ chính), Linux, macOS
- **NVIDIA GPU** có CUDA (khuyến nghị)
- **Webcam** (tích hợp hoặc rời)

### Thiết lập

```powershell
# Clone
git clone <repo-url>
cd Real-Time-Face-and-Hand-Gesture-Recognition-System

# Môi trường ảo (khuyến nghị)
python -m venv venv
venv\Scripts\activate

# Cài thư viện
pip install -r requirements.txt
```

### Tùy chọn: Web UI

```powershell
pip install flask
```

Tất cả model YOLO và MediaPipe sẽ **tự động tải xuống** khi chạy lần đầu.

---

## 2. Kiểm tra môi trường

Kiểm tra cấu hình trước khi chạy:

```powershell
python scripts\system_check.py
```

Kết quả mong đợi:
```
[OK] Python
[OK] OpenCV
[OK] MediaPipe
[OK] Ultralytics
[OK] PyTorch
[OK] CUDA - NVIDIA GeForce RTX ...
[OK] Model exists: models/yolo11n.pt  (5 models)
[OK] Model exists: models/hand_landmarker.task
[OK] Device manager - selected=CUDA
[OK] Face detector init - YOLO11S
[OK] Hand detector init
[OK] Camera open
[OK] Camera frame read
```

### Chạy kiểm thử

```powershell
python -m pytest tests -v
```

Kết quả: **36 passed**

---

## 3. Chạy ứng dụng

### ⭐ Chế độ mặc định (Khuyến nghị)

```powershell
python main.py
```

Hệ thống sẽ:
1. Phát hiện GPU/VRAM → chọn model YOLO phù hợp nhất
2. Tự động tải model nếu thiếu
3. Tự động xuất TensorRT nếu có GPU (lần chạy đầu tiên)
4. Làm nóng model
5. Mở webcam và bắt đầu nhận diện

> **Kết quả:** YOLO11m ở 640px với ~49 FPS trên RTX 3050 Ti 4GB

### Chọn phiên bản model

```powershell
python main.py                              # Auto (mặc định, khuyến nghị)
python main.py --model-version m            # YOLO11m — cân bằng
python main.py --model-version x --imgsz 800  # YOLO11x — chính xác nhất
python main.py --model-version s            # YOLO11s — nhanh
python main.py --model-version n            # YOLO11n — nhẹ nhất
```

### Tracking & Lọc lớp

```powershell
# Bật tracking (ByteTrack)
python main.py --track

# Lọc lớp cụ thể (tên COCO)
python main.py --classes "person,cell phone,bottle"

# Kết hợp tracking + lọc
python main.py --track --classes "person,car"
```

### Tăng tốc TensorRT

```powershell
# Xuất thủ công
python main.py --optimize tensorrt

# Hoặc bật auto trong config.py:
# AUTO_TENSORRT = True
```

TensorRT xuất model sang định dạng `.engine` FP16 ở lần chạy đầu. Các lần sau dùng engine đã lưu, tăng FPS 2-3x.

### Ghi video

```powershell
python main.py --save-video
```

Video lưu tại `outputs/videos/result_YYYYMMDD_HHMMSS.mp4`.

### Đo hiệu năng

```powershell
python main.py --benchmark
```

Đo FPS cho tất cả tổ hợp model × imgsz, hiển thị bảng so sánh.

---

## 4. Phím tắt

| Phím | Chức năng |
|---|---|
| `Q` | Thoát ứng dụng |
| `S` | Chụp ảnh màn hình (lưu tại `outputs/screenshots/`) |

---

## 5. Tùy chọn CLI

### Tùy chọn `main.py`

```
--source              Chỉ số camera (0, 1, 2...) hoặc đường dẫn video
--model               Đường dẫn đến file model tùy chỉnh
--model-version       Phiên bản YOLO: auto, n, s, m, l, x
--device              Thiết bị: auto, cuda, cpu
--imgsz               Kích thước ảnh: auto, 416, 512, 640, 800
--conf                Ngưỡng tin cậy (mặc định: 0.5)
--track               Bật theo dõi đối tượng
--classes             Lọc lớp: "person,cell phone"
--optimize            Tối ưu: none, tensorrt
--save-video          Ghi lại video đầu ra
--benchmark           Đo FPS và thoát
--show                Hiển thị cửa sổ (mặc định: True)
```

### Tùy chọn `web_app.py`

```
--source              Chỉ số camera hoặc đường dẫn video
--model-version       Phiên bản YOLO: auto, n, s, m, l, x
--imgsz               Kích thước ảnh (mặc định: 640)
--conf                Ngưỡng tin cậy
--track               Bật theo dõi đối tượng
--classes             Lọc lớp
--port                Cổng web (mặc định: 5000)
```

---

## 6. Tinh chỉnh hiệu năng

### Tốc độ tối đa

```powershell
python main.py --model-version s --optimize tensorrt --imgsz 512
```

- YOLO11s + TensorRT: 100+ FPS
- Giảm độ phân giải để giảm tính toán

### Độ chính xác tối đa

```powershell
python main.py --model-version x --imgsz 800
```

- YOLO11x: model lớn nhất, chất lượng tốt nhất
- Độ phân giải cao hơn bắt được nhiều chi tiết

### Cân bằng

```powershell
python main.py
```

- YOLO11m tự động chọn
- ~49 FPS với độ chính xác tốt

---

## 7. Web UI

```powershell
python web_app.py
```

Mở [http://localhost:5000](http://localhost:5000) trong trình duyệt.

### Tính năng

- **Luồng video trực tiếp** với overlay nhận diện
- **Bật/tắt overlay**: bounding box, landmark, tay, cử chỉ, lịch sử
- **Bảng trạng thái thời gian thực**: FPS, số lượng vật thể, cử chỉ hiện tại
- **Tự động cập nhật**: Bảng trạng thái làm mới mỗi giây

### Cổng tùy chỉnh

```powershell
python web_app.py --port 8080
```

---

## 8. Xử lý sự cố

### Vấn đề Camera

| Vấn đề | Giải pháp |
|---|---|
| "Cannot open source: 0" | Thử `--source 1` hoặc `--source 2` |
| Không tìm thấy camera | Đóng ứng dụng đang dùng camera (Zoom, Teams) |
| Không có quyền | Kiểm tra Windows Settings → Privacy → Camera |

### Vấn đề CUDA / GPU

```powershell
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu `False`:
- Cài lại PyTorch với CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
- Cập nhật driver NVIDIA
- Hệ thống tự động dùng CPU dự phòng

### Vấn đề tải Model

Model tự động tải khi chạy lần đầu. Nếu thất bại:

```powershell
python scripts\system_check.py
```

Đường dẫn tải thủ công (đặt trong `models/`):
- Model YOLO: Tự động bởi Ultralytics
- Model tay: Tự động bởi MediaPipe

### FPS thấp

- Dùng model nhẹ hơn: `--model-version s`
- Giảm độ phân giải: `--imgsz 512`
- Bật TensorRT: `--optimize tensorrt`
- Tắt tracking: bỏ `--track`
- Đóng ứng dụng khác đang dùng GPU

### Ứng dụng bị treo

- Nhấn `Ctrl+C` trong terminal để thoát
- Hệ thống tự động giải phóng tài nguyên
- Nếu camera vẫn bị khóa, khởi động lại ứng dụng

---

## 9. Cấu hình nâng cao

Sửa [`config.py`](../config.py) để cấu hình cố định:

### Độ phân giải

```python
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
```

### Nhận diện

```python
CONFIDENCE_THRESHOLD = 0.5
YOLO_NMS_IoU = 0.45
```

### Tối ưu

```python
USE_FP16 = True           # Nửa độ chính xác trên GPU
AUTO_TENSORRT = True      # Tự động xuất TensorRT
OPTIMIZE = "none"         # Chế độ tối ưu
```

### Cử chỉ

```python
GESTURE_SMOOTHING_WINDOW = 5   # Số frame làm mịn
SHOW_GESTURE_CONFIDENCE = True # Hiện % độ tin cậy
SHOW_GESTURE_HISTORY = True    # Hiện lịch sử
MAX_GESTURE_HISTORY = 5        # Số cử chỉ lưu lại
```
