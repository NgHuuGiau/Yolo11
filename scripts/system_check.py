from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import mediapipe as mp
import numpy as np
import torch
import ultralytics

import config
from src.device_manager import get_device
from src.face_detector import FaceDetector
from src.hand_detector import HandDetector


def print_status(name, ok, detail=""):
    state = "OK" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{state}] {name}{suffix}")


def main():
    print("=== SYSTEM CHECK ===")
    print_status("Python", True)
    print_status("OpenCV", True, cv2.__version__)
    print_status("MediaPipe", True, getattr(mp, "__version__", "unknown"))
    print_status("Ultralytics", True, ultralytics.__version__)
    print_status("PyTorch", True, torch.__version__)

    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU only"
    print_status("CUDA", cuda_ok, gpu_name)

    for model_path in [
        config.YOLO_MODEL_PATH,
        config.OPTIONAL_YOLO_MODEL_PATH,
        config.HAND_LANDMARKER_MODEL_PATH,
    ]:
        print_status(f"Model exists: {model_path}", Path(model_path).exists())

    device_info = get_device("auto")
    print_status("Device manager", True, f"selected={device_info['label']}")

    try:
        face_detector = FaceDetector(
            model_path=config.YOLO_MODEL_PATH,
            device_info=get_device("cpu"),
            conf=config.CONFIDENCE_THRESHOLD,
            imgsz=config.IMAGE_SIZE,
        )
        print_status("Face detector init", True, face_detector.model_name)
    except Exception as exc:
        print_status("Face detector init", False, str(exc))

    try:
        hand_detector = HandDetector()
        frame = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        hands = hand_detector.detect(frame)
        print_status("Hand detector init", True, f"hands={len(hands)} on blank frame")
        hand_detector.close()
    except Exception as exc:
        print_status("Hand detector init", False, str(exc))

    cap = cv2.VideoCapture(config.CAMERA_SOURCE)
    camera_ok = cap.isOpened()
    print_status("Camera open", camera_ok, f"source={config.CAMERA_SOURCE}")
    if camera_ok:
        ok, _ = cap.read()
        print_status("Camera frame read", ok)
    cap.release()

    print("=== CHECK COMPLETE ===")


if __name__ == "__main__":
    main()
