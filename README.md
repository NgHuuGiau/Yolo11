<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=200&color=gradient&text=Object%20%26%20Hand%20Detection&fontAlign=50&fontAlignY=40&desc=Nh%E1%BA%ADn%20di%E1%BB%87n%20v%E1%BA%ADt%20th%E1%BB%83%20%26%20b%C3%A0n%20tay%20th%E1%BB%9Di%20gian%20th%E1%BB%B1c&descAlign=50&descAlignY=60" width="100%"/>
</div>

<div align="center" style="margin-top: -20px;">

# Hệ thống Nhận diện Vật thể & Cử chỉ Bàn tay

**Nhận diện vật thể, theo dõi bàn tay và nhận dạng cử chỉ theo thời gian thực với YOLO11 + MediaPipe**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)]()
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4-FF6F00?style=flat-square&logo=yolo&logoColor=white)]()
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00C853?style=flat-square)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13-5C3EE8?style=flat-square&logo=opencv&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?style=flat-square&logo=nvidia&logoColor=white)]()
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask&logoColor=white)]()
[![TensorRT](https://img.shields.io/badge/TensorRT-10-76B900?style=flat-square)]()
[![License](https://img.shields.io/badge/Gi%E1%BA%A5y%20ph%C3%A9p-MIT-yellow?style=flat-square)]()
[![CI](https://img.shields.io/badge/CI-OK-4CAF50?style=flat-square)]()
[![Tests](https://img.shields.io/badge/Tests-36%2F36-4CAF50?style=flat-square)]()

---

[🚀 Bắt đầu nhanh](#-bắt-đầu-nhanh) •
[✨ Tính năng](#-tính-năng) •
[📦 Cấu trúc](#-cấu-trúc-dự-án) •
[⚙️ Cấu hình](#%EF%B8%8F-cấu-hình) •
[📊 Hiệu năng](#-hiệu-năng) •
[🌐 Web UI](#-web-ui) •
[📚 Tài liệu](docs/)

---

</div>

## 🚀 Bắt đầu nhanh

```powershell
# 1. Clone dự án
git clone <repo-url>
cd Real-Time-Face-and-Hand-Gesture-Recognition-System

# 2. Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate

# 3. Cài thư viện
pip install -r requirements.txt

# 4. Chạy (tự động detect GPU, tải model, tối ưu TensorRT)
python main.py
```

> **Không cần cấu hình gì thêm.** Hệ thống tự động phát hiện GPU/VRAM, chọn model YOLO phù hợp, tải model nếu thiếu, và tối ưu TensorRT nếu có hỗ trợ.

### ⚡ Lệnh nhanh

| Lệnh | Mô tả |
|---|---|
| `python main.py` | Chạy với cấu hình tối ưu (tự động) |
| `python main.py --benchmark` | Đo FPS tất cả model |
| `python main.py --model-version x --imgsz 800` | Độ chính xác cao nhất |
| `python main.py --track --classes "person"` | Tracking + lọc đối tượng |
| `python main.py --optimize tensorrt` | Tăng tốc TensorRT |
| `python web_app.py` | Mở giao diện Web |

---

## ✨ Tính năng

<div align="center">

| Nhóm | Tính năng | Mô tả |
|---|---|---|
| **Nhận diện vật thể** | YOLO11 (n/s/m/l/x) | 5 phiên bản từ siêu nhẹ đến siêu nặng |
| | Tự động detect phần cứng | Đo GPU/VRAM → chọn model phù hợp nhất |
| | FP16 | Tính toán nửa độ chính xác, tăng gấp đôi tốc độ |
| | TensorRT | Tự động xuất engine, tăng 2-3x FPS |
| | Làm nóng model | Khởi tạo CUDA kernels trước khi chạy |
| | Tracking vật thể | Gán ID cho từng đối tượng (ByteTrack) |
| | Lọc lớp đối tượng | Chỉ detect các lớp mong muốn trong 80 lớp COCO |
| **Nhận diện bàn tay** | 21 điểm landmark | Bộ xương bàn tay MediaPipe |
| | Phân biệt tay trái/phải | Kèm độ tin cậy |
| | Tự động thử lại | 3 lần nếu detect thất bại |
| **Nhận dạng cử chỉ** | 11 cử chỉ | Nắm đấm, Xòe tay, Chỉ trỏ, Hòa bình, Like, Gọi, Súng, Rock, Ngón giữa, Người nhện, đếm ngón tay |
| | Làm mịn tín hiệu | Cửa sổ 5 frame, loại nhiễu giật |
| | Độ tin cậy | Phần trăm tin cậy cho từng cử chỉ |
| | Lịch sử | 5 cử chỉ gần nhất |
| **Độ tin cậy** | Tự động kết nối lại camera | Backoff theo cấp số nhân khi mất tín hiệu |
| | Tắt ứng dụng an toàn | Giải phóng tài nguyên khi thoát |
| | Web UI | Giao diện trình duyệt, bật/tắt overlay |
| | Chụp ảnh | Nhấn phím `S` để chụp màn hình |

</div>

---

## 📦 Cấu trúc dự án

```
├── .github/workflows/     # CI pipeline (GitHub Actions)
├── docs/                  # Tài liệu hướng dẫn
│   ├── project_description.md   # Mô tả kiến trúc
│   └── user_guide.md            # Hướng dẫn sử dụng
├── models/                # File model (tự động tải)
│   ├── yolo11n.pt ~ yolo11x.pt
│   └── hand_landmarker.task
├── outputs/
│   ├── screenshots/       # Ảnh chụp màn hình (phím S)
│   └── videos/            # Video ghi lại
├── scripts/
│   └── system_check.py    # Kiểm tra môi trường
├── src/                   # Mã nguồn chính
│   ├── device_manager.py  # Chọn thiết bị CUDA/CPU
│   ├── face_detector.py   # Suy luận YOLO + tracking
│   ├── fps_counter.py     # Đo FPS
│   ├── gesture_recognizer.py  # Nhận dạng cử chỉ
│   ├── hand_detector.py   # Landmark bàn tay MediaPipe
│   ├── hardware_profiler.py  # Detect GPU + benchmark
│   └── utils.py           # Vẽ overlay, chụp ảnh
├── tests/
│   └── test_system.py     # 36 bài kiểm thử
├── config.py              # Cấu hình tập trung
├── main.py                # Giao diện OpenCV
├── web_app.py             # Giao diện Web Flask
├── pyproject.toml         # Cấu hình Python hiện đại
└── requirements.txt       # Thư viện phụ thuộc
```

---

## ⚙️ Cấu hình

Tất cả cấu hình trong [`config.py`](config.py):

### Thiết bị & Model

| Biến | Mặc định | Mô tả |
|---|---|---|
| `CAMERA_WIDTH` | `1280` | Độ rộng camera |
| `CAMERA_HEIGHT` | `720` | Độ cao camera |
| `DEVICE` | `"auto"` | `"auto"`, `"cuda"` hoặc `"cpu"` |
| `MODEL_TIER_KEY` | `"auto"` | Tự động hoặc chọn `n/s/m/l/x` |
| `CONFIDENCE_THRESHOLD` | `0.5` | Ngưỡng tin cậy YOLO |
| `YOLO_NMS_IoU` | `0.45` | Ngưỡng IoU khử trùng |

### Tối ưu

| Biến | Mặc định | Mô tả |
|---|---|---|
| `USE_FP16` | `True` | Tính toán nửa độ chính xác |
| `AUTO_TENSORRT` | `True` | Tự động xuất engine TensorRT |
| `OPTIMIZE` | `"none"` | Chế độ tối ưu: `"none"` / `"tensorrt"` |

### Hiển thị

| Biến | Mặc định | Mô tả |
|---|---|---|
| `SHOW_FACE_BOX` | `True` | Hiện khung bounding box |
| `SHOW_HAND_LANDMARKS` | `True` | Hiện bộ xương bàn tay |

### Cử chỉ

| Biến | Mặc định | Mô tả |
|---|---|---|
| `GESTURE_SMOOTHING_WINDOW` | `5` | Số frame làm mịn |
| `SHOW_GESTURE_CONFIDENCE` | `True` | Hiện % độ tin cậy |
| `SHOW_GESTURE_HISTORY` | `True` | Hiện lịch sử cử chỉ |
| `MAX_GESTURE_HISTORY` | `5` | Số cử chỉ lưu lại |

---

## 📊 Hiệu năng

Đo trên **NVIDIA GeForce RTX 3050 Ti (4GB VRAM)**:

| Model | imgsz | FPS | Mục đích |
|---|---|---|---|
| YOLO11n | 416 | ~67 | CPU / Tiết kiệm pin |
| YOLO11s | 640 | ~76 | Nhanh & nhẹ |
| **YOLO11m** | **640** | **~49** | **Cân bằng (mặc định)** |
| YOLO11l | 640 | ~46 | Chất lượng cao |
| YOLO11x | 640 | ~22 | Độ chính xác tối đa |
| YOLO11x | 800 | ~15 | Cực kỳ chính xác |

### Khuyến nghị

- **Cân bằng:** `python main.py` (YOLO11m, 49 FPS)
- **Nhanh nhất:** `python main.py --model-version s --optimize tensorrt` (100+ FPS)
- **Chính xác nhất:** `python main.py --model-version x --imgsz 800` (15 FPS)

---

## 🌐 Web UI

```powershell
pip install flask
python web_app.py
```

Mở trình duyệt tại [http://localhost:5000](http://localhost:5000)

- Luồng video MJPEG thời gian thực
- Bật/tắt overlay (bounding box, landmark, cử chỉ, lịch sử)
- Bảng trạng thái trực tiếp (FPS, số lượng vật thể, cử chỉ hiện tại)

---

## 🧪 Kiểm thử

```powershell
# Kiểm tra môi trường
python scripts\system_check.py

# Chạy 36 bài kiểm thử
python -m pytest tests -v

# Kiểm tra toàn diện (dùng ";" thay "&&" trong PowerShell)
python scripts\system_check.py; python -m pytest tests -v
```

---

## 🛠️ Tham khảo CLI

```powershell
python main.py [TÙY_CHỌN]

Tùy chọn:
  --source              Chỉ số camera hoặc đường dẫn video (mặc định: 0)
  --model-version       Phiên bản YOLO: auto, n, s, m, l, x
  --imgsz               Kích thước ảnh: auto, 416, 512, 640, 800
  --conf                Ngưỡng tin cậy (mặc định: 0.5)
  --track               Bật theo dõi đối tượng
  --classes             Lọc lớp: "person,cell phone"
  --optimize            Tối ưu: none, tensorrt
  --save-video          Ghi lại video
  --benchmark           Đo FPS và thoát
```

---

## 📄 Giấy phép

[MIT License](LICENSE)

---

<div align="center">
  <sub>Xây dựng với ❤️ bằng Python, YOLO11, MediaPipe và PyTorch</sub>
</div>
