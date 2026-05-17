# Real-Time Face and Hand Gesture Recognition System

**Tên tiếng Việt:** Hệ thống nhận diện khuôn mặt và bàn tay thời gian thực

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Landmarker-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-red)

## Giới thiệu

Đây là dự án Python xử lý video theo thời gian thực từ webcam hoặc file video để:

- nhận diện khuôn mặt
- phát hiện bàn tay
- vẽ 21 điểm landmark và skeleton bàn tay
- hiển thị FPS, device, model và thanh tiến trình trực quan trên màn hình

Phiên bản hiện tại của dự án đã được chỉnh theo nhu cầu sử dụng thực tế:

- tập trung vào **khuôn mặt** thay vì nhận diện cử chỉ
- giữ lại phần **bàn tay** để hiển thị landmark
- toàn bộ **đường trên tay hiển thị màu xanh lá**
- có **system check** và **smoke test** để kiểm tra môi trường trước khi chạy

## Trạng thái hiện tại của dự án

Hiện tại code chạy theo hướng:

- nhận diện khuôn mặt bằng `OpenCV Haar Cascade` khi `FACE_ONLY_MODE = True`
- phát hiện bàn tay bằng `MediaPipe Hand Landmarker`
- sử dụng `PyTorch CUDA` nếu có GPU NVIDIA
- fallback sang CPU nếu CUDA không khả dụng
- hỗ trợ model `YOLO11s` và `YOLO11m` trong thư mục `models/` cho các mở rộng sau này

Lưu ý quan trọng:

- Trong cấu hình hiện tại, phần nhận diện khuôn mặt đang ưu tiên `Haar Cascade` để bám sát nhu cầu "xác định khuôn mặt, không cần cử chỉ".
- Hai model `YOLO11s` và `YOLO11m` vẫn được giữ trong `models/` vì chúng vẫn hữu ích nếu sau này bạn muốn bật lại chế độ dùng YOLO.

## Tính năng chính

- Nhận diện khuôn mặt theo thời gian thực
- Nhận diện một hoặc nhiều bàn tay
- Vẽ 21 landmarks bàn tay
- Vẽ skeleton bàn tay màu xanh lá
- Hiển thị tay trái/phải nếu MediaPipe trả về
- Hiển thị FPS realtime
- Hiển thị thiết bị xử lý: CUDA hoặc CPU
- Hiển thị thanh tiến trình trên frame
- Có kiểm tra hệ thống trước khi chạy
- Có smoke test tự động cho các thành phần quan trọng

## Công nghệ sử dụng

- Python
- OpenCV
- MediaPipe
- PyTorch
- CUDA
- NumPy
- Ultralytics YOLO11
- argparse
- pathlib

## Cấu trúc thư mục đúng và hợp lý

Đây là cấu trúc nên dùng cho dự án hiện tại:

```text
real-time-face-hand-gesture/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
├── data/
│   ├── datasets/
│   ├── images/
│   └── videos/
├── docs/
│   ├── project_description.md
│   └── user_guide.md
├── models/
│   ├── hand_landmarker.task
│   ├── yolo11s.pt
│   ├── yolo11m.pt
│   └── README.md
├── outputs/
│   ├── screenshots/
│   └── videos/
├── runs/
│   ├── detect/
│   └── train/
├── scripts/
│   └── system_check.py
├── src/
│   ├── __init__.py
│   ├── device_manager.py
│   ├── face_detector.py
│   ├── fps_counter.py
│   ├── gesture_recognizer.py
│   ├── hand_detector.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_system.py
├── .gitignore
├── config.py
├── CONTRIBUTING.md
├── LICENSE
├── main.py
├── README.md
└── requirements.txt
```

## Đánh giá lại cấu trúc hiện tại

Sau khi rà lại, đây là kết luận:

- `models/` là đúng chỗ để đặt `yolo11s.pt`, `yolo11m.pt`, `hand_landmarker.task`
- `tests/` là đúng chỗ cho các file kiểm thử
- `scripts/` là đúng chỗ cho `system_check.py`
- `src/` là đúng chỗ cho toàn bộ module nghiệp vụ
- `.github/` là đúng chỗ cho file liên quan GitHub
- `venv/` nên tồn tại cục bộ nhưng không nên đưa lên GitHub
- `__pycache__/` là file phát sinh, không nên giữ trong repo

Những gì đã được chỉnh lại:

- đã chuyển `system_check.py` từ thư mục gốc sang `scripts/system_check.py`
- đã xóa `yolo11s.pt` và `yolo11m.pt` bị dư ở thư mục gốc
- đã dọn `__pycache__/` phát sinh

## Ý nghĩa từng file quan trọng

### File gốc

- `main.py`: file chạy chính
- `config.py`: cấu hình mặc định của hệ thống
- `requirements.txt`: danh sách thư viện cần cài
- `README.md`: tài liệu giới thiệu và hướng dẫn sử dụng
- `LICENSE`: giấy phép MIT
- `CONTRIBUTING.md`: hướng dẫn đóng góp

### Thư mục `src/`

- `src/face_detector.py`: xử lý nhận diện khuôn mặt
- `src/hand_detector.py`: xử lý bàn tay bằng MediaPipe
- `src/fps_counter.py`: tính FPS
- `src/device_manager.py`: chọn CUDA hoặc CPU
- `src/utils.py`: hàm tiện ích vẽ panel, progress bar, resize, lưu file
- `src/gesture_recognizer.py`: hiện đang không dùng trong luồng chạy chính nhưng vẫn được giữ lại để mở rộng về sau

### Thư mục `scripts/`

- `scripts/system_check.py`: kiểm tra nhanh môi trường, model, camera, CUDA

### Thư mục `tests/`

- `tests/test_system.py`: smoke test cho các thành phần chính

## Cài đặt môi trường

### Bước 1. Tạo môi trường ảo

```powershell
python -m venv venv
```

### Bước 2. Kích hoạt môi trường ảo

```powershell
venv\Scripts\activate
```

### Bước 3. Cài thư viện

```powershell
pip install -r requirements.txt
```

## Model cần có

Thư mục `models/` nên chứa:

- `models/yolo11s.pt`
- `models/yolo11m.pt`
- `models/hand_landmarker.task`

Nếu thiếu, chương trình hoặc script kiểm tra hệ thống sẽ báo rõ.

## Cách chạy kiểm tra hệ thống

### Kiểm tra nhanh môi trường

```powershell
python scripts/system_check.py
```

Script này sẽ kiểm tra:

- import thư viện
- phiên bản PyTorch
- CUDA có hoạt động hay không
- model có tồn tại hay không
- detector có khởi tạo được không
- camera có mở được không

### Chạy smoke test

```powershell
python -m unittest tests.test_system
```

## Cách chạy chương trình

### Chạy webcam mặc định

```powershell
python main.py --source 0
```

### Chạy bằng GPU CUDA

```powershell
python main.py --source 0 --device cuda
```

### Chạy bằng CPU

```powershell
python main.py --source 0 --device cpu
```

### Chạy với video

```powershell
python main.py --source data/videos/test.mp4 --device cuda
```

### Chạy với model YOLO11m

```powershell
python main.py --model models/yolo11m.pt --source 0 --device cuda --imgsz 640
```

Lưu ý:

- nếu `FACE_ONLY_MODE = True` thì khuôn mặt đang dùng `Haar Cascade`
- khi đó tham số `--model` không phải là thành phần chính trong luồng nhận diện khuôn mặt hiện tại

## Những gì hiển thị trên màn hình

Khi chạy, chương trình sẽ hiển thị:

- `FPS`
- `Device`
- `Model`
- `Hand`
- bounding box khuôn mặt
- landmark bàn tay
- skeleton bàn tay màu xanh lá
- progress bar trực quan trên frame

Ví dụ:

```text
FPS: 32.5
Device: CUDA
Model: HAAR FACE
Hand: Right
Tracking: 78%
```

## Giải thích log bạn thường thấy khi chạy

Khi chạy MediaPipe, có thể xuất hiện một số dòng như:

- `Created TensorFlow Lite XNNPACK delegate for CPU`
- `inference_feedback_manager.cc`
- `portable_clearcut_uploader.cc`

Đây thường là log native từ MediaPipe hoặc TensorFlow Lite, không đồng nghĩa với việc chương trình bị lỗi.

Nếu bạn bấm `Ctrl + C` trong lúc chương trình đang chạy webcam, hiện tại chương trình đã được chỉnh để thoát gọn hơn thay vì quăng lỗi thô kiểu `KeyboardInterrupt`.

## Tối ưu cho RTX 3050 Ti 4GB VRAM

Khuyến nghị:

- dùng `--device cuda`
- giữ độ phân giải camera `640x480`
- nếu sau này bật lại chế độ YOLO cho face/object, ưu tiên `YOLO11s`
- chỉ dùng `YOLO11m` khi thực sự cần
- đóng các ứng dụng nặng đang chiếm GPU

## Các vấn đề thường gặp

### Không mở được webcam

- thử `--source 1`
- đóng ứng dụng đang chiếm camera
- kiểm tra quyền camera trên Windows

### CUDA không hoạt động

Kiểm tra:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

### Thiếu model

Kiểm tra thư mục `models/` có đủ 3 file cần thiết hay chưa.

### Chương trình chạy nhưng không hiện gì

- kiểm tra bạn có đang dùng `--show false` không
- nếu có, chương trình vẫn chạy nhưng không mở cửa sổ

## Gợi ý tổ chức repo về sau

Nếu dự án phát triển thêm, bạn có thể mở rộng theo hướng:

- thêm `assets/` nếu có hình minh họa hoặc logo
- thêm `notebooks/` nếu có notebook thử nghiệm
- thêm `benchmarks/` nếu muốn lưu script đo FPS, latency, VRAM
- thêm `configs/` nếu số cấu hình runtime tăng nhiều hơn hiện tại

## License

Dự án hiện dùng `MIT License`.

## Tài liệu liên quan

- [docs/project_description.md](docs/project_description.md)
- [docs/user_guide.md](docs/user_guide.md)
- [models/README.md](models/README.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)

