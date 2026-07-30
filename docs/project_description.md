# Mô tả Dự án

## Tổng quan

**Hệ thống Nhận diện Vật thể & Cử chỉ Bàn tay** là ứng dụng thị giác máy tính bằng Python xử lý video trực tiếp từ webcam để thực hiện:

- **Nhận diện vật thể** bằng YOLO11 (5 phiên bản: n/s/m/l/x)
- **Theo dõi bàn tay** bằng MediaPipe (21 điểm landmark + bộ xương)
- **Nhận dạng cử chỉ** bằng thuật toán rule-based kết hợp làm mịn tín hiệu (11 cử chỉ)
- **Theo dõi đối tượng** gán ID cố định cho từng vật thể
- **Lọc lớp đối tượng** từ 80 lớp COCO
- **Tăng tốc TensorRT** tăng tốc suy luận 2-3 lần

## Kiến trúc

```
┌─────────────────────────────────────────────────────────────┐
│                   main.py / web_app.py                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Hardware     │  │ Face         │  │ Hand             │   │
│  │ Profiler     │→│ Detector     │  │ Detector         │   │
│  │ (auto-detect)│  │ (YOLO11)     │  │ (MediaPipe)      │   │
│  └─────────────┘  └──────┬───────┘  └────────┬─────────┘   │
│                          │                     │            │
│                          ▼                     ▼            │
│                   ┌──────────────┐  ┌──────────────────┐   │
│                   │ Gesture      │  │ fps_counter       │   │
│                   │ Recognizer   │  │ + utils           │   │
│                   └──────────────┘  └──────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                 OpenCV / Flask Output                        │
└─────────────────────────────────────────────────────────────┘
```

### Luồng xử lý

```
Phát hiện phần cứng → Chọn model → Tải model (nếu thiếu)
     ↓
Xuất TensorRT (nếu có GPU) → Làm nóng model
     ↓
Mở camera (1280x720)
     ↓
┌─────────────── Vòng lặp chính ───────────────┐
│  Đọc frame                                    │
│  → YOLO Predict (FP16 + NMS + Track)         │
│  → MediaPipe Hands (21 landmark)             │
│  → Nhận dạng cử chỉ (làm mịn)               │
│  → Vẽ FPS + Tên model                        │
│  → Hiển thị / Stream                         │
└──────────────────────────────────────────────┘
```

## Module

| Module | Nhiệm vụ | Tính năng chính |
|---|---|---|
| `face_detector.py` | Nhận diện vật thể YOLO | FP16, NMS IoU, warmup, ByteTrack, lọc lớp |
| `hand_detector.py` | Theo dõi tay MediaPipe | 21 landmark, phân biệt trái/phải, thử lại 3 lần |
| `gesture_recognizer.py` | Phân loại cử chỉ | 11 cử chỉ, làm mịn 5 frame, độ tin cậy |
| `hardware_profiler.py` | Hồ sơ hệ thống | Detect GPU/VRAM, đề xuất tier, benchmark, TensorRT |
| `device_manager.py` | Chọn thiết bị | Tự chọn CUDA, dự phòng CPU |
| `fps_counter.py` | Đo hiệu năng | FPS trung bình động |
| `utils.py` | Vẽ overlay | FPS, chụp ảnh, resize |

## Cơ chế Nhận dạng Cử chỉ

Bộ nhận dạng cử chỉ sử dụng thuật toán **rule-based** phân tích 21 điểm MediaPipe:

1. **Xác định trạng thái ngón tay**: So sánh tọa độ Y của đầu ngón tay với khớp PIP để xác định ngón duỗi/gập. Ngón cái dùng so sánh tọa độ X với khớp MCP ngón trỏ.

2. **Vector trạng thái**: Mảng boolean 5 phần tử `[ngón_cái, trỏ, giữa, áp út, út]` biểu diễn tư thế bàn tay.

3. **So khớp mẫu**: Vector được so khớp với các mẫu cử chỉ đã biết:
   - `[0,0,0,0,0]` → Nắm đấm (Fist)
   - `[1,1,1,1,1]` → Xòe tay (Open Palm)
   - `[0,1,0,0,0]` → Chỉ trỏ (Pointing)
   - `[0,1,1,0,0]` → Hòa bình (Peace)
   - `[1,0,0,0,0]` → Like (Thumbs Up)
   - `[1,0,0,0,1]` → Gọi (Call Me)
   - `[1,1,0,0,0]` → Súng (Gun)
   - `[0,0,1,0,0]` → Ngón giữa (Middle Finger)
   - `[1,1,0,0,1]` → Rock
   - `[0,1,0,0,1]` → Người nhện (Spiderman)
   - Dự phòng đếm ngón tay cho 2-5 ngón

4. **Làm mịn tín hiệu**: Cửa sổ trượt 5 frame tính trung bình vector trạng thái để tránh giật.

5. **Độ tin cậy**: Tính bằng tỉ lệ `(số ngón duỗi / 5)`.

## Phân hạng Model

| Hạng | Model | VRAM | imgsz | Phần cứng mục tiêu |
|---|---|---|---|---|
| 1 | YOLO11n | ~1 GB | 416 | CPU / iGPU |
| 2 | YOLO11s | ~2 GB | 512 | GPU 2-4GB (GTX 1650, RTX 2050) |
| 3 | **YOLO11m** | ~4 GB | **640** | **GPU 4-6GB (RTX 3050 Ti, RTX 2060)** |
| 4 | YOLO11l | ~8 GB | 640 | GPU 6-10GB (RTX 3060-3080) |
| 5 | YOLO11x | ~12 GB | 640 | GPU >10GB (RTX 3090, 4090) |

## Tối ưu

- **FP16**: Mặc định bật trên GPU CUDA
- **TensorRT**: Tự động xuất engine khi phát hiện NVIDIA GPU
- **Làm nóng model**: Một lần suy luận giả để khởi tạo CUDA kernels
- **NMS IoU**: Ngưỡng cấu hình được trong `config.py`

## Xử lý lỗi

- **Frame rỗng/null**: Bỏ qua an toàn trong cả YOLO và MediaPipe
- **Lỗi nhận diện tay**: Thử lại 3 lần với cảnh báo
- **Mất camera**: Tự động kết nối lại với backoff (2s → 10s)
- **Tắt web server**: Xử lý SIGINT để giải phóng tài nguyên
- **Lỗi tải model**: Thông báo lỗi kèm hướng dẫn tải thủ công

## Điểm vào CLI

### `main.py` — Ứng dụng Desktop OpenCV

GUI đầy đủ với phím tắt (Q=thoát, S=chụp ảnh). Hỗ trợ tất cả cờ cấu hình.

### `web_app.py` — Máy chủ Web Flask

Giao diện trình duyệt tại `localhost:5000` với:
- Luồng video MJPEG
- Bật/tắt overlay
- Bảng trạng thái thời gian thực
- Hỗ trợ tất cả cờ camera và model

## Kiểm thử

36 bài kiểm thử bao gồm:
- Phát hiện phần cứng và đề xuất tier
- Tải và làm nóng model YOLO
- Khởi tạo hand detector và xử lý biên
- Tất cả 11 mẫu nhận dạng cử chỉ
- Xử lý frame rỗng/null
- Kiểm tra cấu hình
- Tính đúng đắn của hàm tiện ích
- Logic làm mịn cử chỉ

## Mở rộng

- **Thêm cử chỉ**: Sửa `src/gesture_recognizer.py`
- **Model tùy chỉnh**: Đặt file `.pt` trong `models/` và cập nhật config
- **Tính năng mới**: Thêm module trong `src/` và tích hợp vào `main.py`
- **Tùy chỉnh Web UI**: Sửa template trong `web_app.py`
