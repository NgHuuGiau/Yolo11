# Project Description

## Overview

Real-Time Face and Hand Gesture Recognition System là dự án Python xử lý video realtime từ webcam hoặc file video để:

- Nhận diện object/khuôn mặt bằng YOLO11.
- Nhận diện bàn tay bằng MediaPipe Hands.
- Suy luận cử chỉ cơ bản từ 21 điểm landmark.
- Hiển thị FPS và thiết bị suy luận trực tiếp trên màn hình.

## Design Goals

- Ưu tiên tính ổn định khi chạy realtime.
- Tận dụng CUDA nếu có GPU.
- Fallback CPU an toàn nếu không có CUDA.
- Tối ưu cho GPU RTX 3050 Ti 4GB VRAM bằng cách dùng `YOLO11s` làm mặc định.
- Tách module rõ ràng để dễ thay model, thêm gesture và bảo trì.

## Main Pipeline

1. Mở webcam hoặc video bằng OpenCV.
2. Resize frame về độ phân giải camera cấu hình.
3. Chạy YOLO11 để phát hiện object/face.
4. Chạy MediaPipe Hands để tìm landmark bàn tay.
5. Suy luận gesture từ landmark.
6. Cập nhật FPS.
7. Vẽ kết quả và hiển thị lên frame.
8. Tùy chọn lưu video đầu ra.

## Default Runtime Choices

- Default model: `yolo11s.pt`
- Optional accuracy model: `yolo11m.pt`
- Default `imgsz`: `640`
- Lighter `imgsz`: `480`
- Default camera resolution: `640x480`
- Default confidence threshold: `0.5`
- CUDA half precision khi có GPU

## Extensibility

Hệ thống hiện được thiết kế để dễ mở rộng theo các hướng:

- Thêm gesture nâng cao trong `src/gesture_recognizer.py`
- Thay model YOLO bằng model tùy biến trong `src/face_detector.py`
- Thêm lưu ảnh chụp hoặc log metadata
- Thêm chế độ benchmark FPS / latency
- Tích hợp nhận diện danh tính khuôn mặt

