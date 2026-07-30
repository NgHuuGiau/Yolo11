# Hệ thống Nhận diện Vật thể & Cử chỉ Bàn tay

**Nhận diện vật thể, theo dõi bàn tay và nhận dạng cử chỉ theo thời gian thực với YOLO11 + MediaPipe**

[![Python 3.10+](https://img.shields.io/badge/Python_3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLO11](https://img.shields.io/badge/YOLO11-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white)](https://ultralytics.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-00C853?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://nvidia.com)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![TensorRT](https://img.shields.io/badge/TensorRT-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![CI](https://img.shields.io/badge/CI_passing-4CAF50?style=for-the-badge&logo=githubactions&logoColor=white)]()
[![Tests](https://img.shields.io/badge/36_tests-4CAF50?style=for-the-badge&logo=pytest&logoColor=white)]()
[![License](https://img.shields.io/badge/MIT-e6e6e6?style=for-the-badge&logo=opensourceinitiative&logoColor=black)]()

---

## Bắt đầu nhanh

```powershell
git clone <repo-url>
cd Real-Time-Face-and-Hand-Gesture-Recognition-System
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Hệ thống tự động detect GPU, tải model, tối ưu TensorRT. Không cần cấu hình gì thêm.

| Lệnh | Mô tả |
|---|---|
| `python main.py` | Chạy mặc định (YOLO11m, ~49 FPS) |
| `python main.py --benchmark` | Đo FPS tất cả model |
| `python main.py --model-version x --imgsz 800` | Độ chính xác cao nhất |
| `python main.py --track --classes "person"` | Tracking + lọc đối tượng |
| `python main.py --optimize tensorrt` | Tăng tốc TensorRT |
| `python web_app.py` | Mở giao diện Web |

---

## Tính năng

- **YOLO11**: 5 phiên bản n/s/m/l/x, FP16, TensorRT, ByteTrack, lọc 80 lớp COCO
- **MediaPipe Hands**: 21 điểm landmark, phân biệt trái/phải, thử lại 3 lần
- **Cử chỉ**: 11 cử chỉ (Nắm đấm, Xòe tay, Chỉ trỏ, Hòa bình, Like, Gọi, Súng, Rock, Ngón giữa, Người nhện, đếm ngón), làm mịn 5 frame, độ tin cậy
- **Tự động**: Detect GPU/VRAM → chọn model, auto TensorRT, warmup, reconnect camera
- **Web UI**: Flask MJPEG stream, toggle overlay, bảng trạng thái

---

## Cấu trúc

```
├── src/          # Mã nguồn chính (7 modules)
├── models/       # Model weights (tự động tải)
├── tests/        # 36 bài kiểm thử
├── docs/         # Tài liệu hướng dẫn
├── scripts/      # system_check.py
├── outputs/      # screenshots + videos
├── config.py     # Cấu hình tập trung
├── main.py       # Giao diện OpenCV
├── web_app.py    # Giao diện Flask
└── pyproject.toml
```

---

## Hiệu năng (RTX 3050 Ti 4GB)

| Model | imgsz | FPS |
|---|---|---|
| YOLO11n | 416 | ~67 |
| YOLO11s | 640 | ~76 |
| **YOLO11m** | **640** | **~49** |
| YOLO11l | 640 | ~46 |
| YOLO11x | 640 | ~22 |
| YOLO11x | 800 | ~15 |

Khuyến nghị: `python main.py` (mặc định, YOLO11m 49 FPS)

---

## CLI

```
python main.py [TÙY_CHỌN]

--source              Chỉ số camera hoặc đường dẫn video
--model-version       YOLO: auto, n, s, m, l, x
--imgsz               Kích thước ảnh: auto, 416, 512, 640, 800
--conf                Ngưỡng tin cậy (mặc định: 0.5)
--track               Bật tracking
--classes             Lọc lớp: "person,cell phone"
--optimize            Tối ưu: none, tensorrt
--save-video          Ghi video
--benchmark           Đo FPS và thoát
```

---

## Kiểm thử

```powershell
python scripts\system_check.py     # Kiểm tra môi trường
python -m pytest tests -v          # 36 bài kiểm thử
```

## Giấy phép

MIT License
