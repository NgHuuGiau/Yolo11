# User Guide

## 1. Cài đặt

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Chuẩn bị model

Đặt file sau vào thư mục `models/` nếu bạn đã có sẵn:

- `models/yolo11s.pt`
- `models/yolo11m.pt`

Nếu chưa có, hệ thống sẽ cố gắng fallback sang tên model gốc để Ultralytics tự tải ở lần đầu.

## 3. Chạy webcam

```powershell
python main.py --source 0
```

## 4. Chạy GPU hoặc CPU

```powershell
python main.py --source 0 --device cuda
python main.py --source 0 --device cpu
python main.py --source 0 --device auto
```

## 5. Chạy video

```powershell
python main.py --source data/videos/test.mp4 --device cuda
```

## 6. Chọn model

```powershell
python main.py --model models/yolo11s.pt --source 0 --device cuda --imgsz 640
python main.py --model models/yolo11m.pt --source 0 --device cuda --imgsz 640
python main.py --model models/yolo11m.pt --source 0 --device cuda --imgsz 480
```

## 7. Lưu video đầu ra

```powershell
python main.py --source 0 --device cuda --save-video
```

File kết quả sẽ nằm trong `outputs/videos/`.

## 8. Nếu FPS thấp

- Giảm `--imgsz` từ `640` xuống `480`
- Dùng `YOLO11s`
- Tắt `--save-video`
- Giảm tải GPU từ ứng dụng khác
- Giữ webcam ở `640x480`

## 9. Nếu không mở được webcam

- Thử `--source 1`
- Đóng phần mềm đang chiếm camera
- Kiểm tra quyền camera trên Windows

## 10. Nếu CUDA không hoạt động

- Kiểm tra driver NVIDIA
- Kiểm tra bản cài PyTorch có hỗ trợ CUDA
- Chạy thử `python main.py --source 0 --device cpu` để xác nhận logic chung vẫn hoạt động

