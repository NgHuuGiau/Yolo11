# Hệ thống Nhận diện Vật thể & Cử chỉ Bàn tay

**Nhận diện vật thể, theo dõi bàn tay và nhận dạng cử chỉ theo thời gian thực với YOLO11 + MediaPipe**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Windows](https://img.shields.io/badge/Windows-11%2B-0078D6?logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![PowerShell](https://img.shields.io/badge/PowerShell-7%2B-5391FE?logo=powershell&logoColor=white)](https://learn.microsoft.com/powershell/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-111111)](https://www.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda)
[![TensorRT](https://img.shields.io/badge/TensorRT-10-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-00C853?logo=google&logoColor=white)](https://mediapipe.dev/)
[![NumPy](https://img.shields.io/badge/NumPy-Array%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Streaming-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?logo=opensourceinitiative&logoColor=white)](LICENSE)

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
|---|---|---|
| `python main.py` | Chạy mặc định — tự động chọn config tối ưu |
| `python main.py --benchmark` | Đo FPS tất cả model trên máy bạn |
| `python main.py --model-version x --imgsz 800` | Ép dùng model lớn nhất |
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
Yolo11/
├── .github/
│   ├── workflows/
│   │   └── ci.yml                    # CI pipeline (GitHub Actions)
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── config.yml
│   └── pull_request_template.md
├── docs/
│   ├── project_description.md        # Mô tả kiến trúc hệ thống
│   └── user_guide.md                 # Hướng dẫn sử dụng chi tiết
├── models/                           # Model weights (tự động tải)
│   ├── yolo11n.pt ~ yolo11x.pt       # 5 phiên bản YOLO11
│   └── hand_landmarker.task          # MediaPipe hand model
├── outputs/
│   ├── screenshots/                  # Ảnh chụp màn hình (phím S)
│   └── videos/                       # Video ghi lại
├── scripts/
│   └── system_check.py               # Kiểm tra môi trường
├── src/
│   ├── __init__.py
│   ├── device_manager.py             # Chọn thiết bị CUDA/CPU
│   ├── face_detector.py              # YOLO object detection + tracking + FP16
│   ├── fps_counter.py                # Đo FPS
│   ├── gesture_recognizer.py         # Nhận dạng cử chỉ + smoothing
│   ├── hand_detector.py              # MediaPipe hand landmarks
│   ├── hardware_profiler.py          # Detect GPU/VRAM + benchmark + TensorRT
│   └── utils.py                      # Vẽ overlay, chụp ảnh
├── tests/
│   ├── __init__.py
│   └── test_system.py                # 36 bài kiểm thử
├── config.py                         # Cấu hình tập trung
├── main.py                           # Giao diện OpenCV Desktop
├── web_app.py                        # Giao diện Web Flask
├── pyproject.toml                    # Cấu hình Python hiện đại
├── requirements.txt                  # Thư viện phụ thuộc
├── .gitignore
├── LICENSE
└── README.md
```

---

## Hiệu năng

Hệ thống tự động phát hiện GPU/VRAM và chọn model + image size phù hợp nhất với cấu hình máy bạn. Dùng lệnh `--benchmark` để đo FPS thực tế trên phần cứng của bạn.

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
