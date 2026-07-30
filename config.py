import logging
from pathlib import Path
from typing import Union

HAND_LANDMARKER_MODEL_PATH = "models/hand_landmarker.task"

CAMERA_SOURCE: Union[int, str] = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 640

DEVICE = "auto"
USE_GPU = True

MODEL_TIER_KEY = "auto"

LOGGING_LEVEL = "INFO"

MODEL_TIERS = {
    "n": {"key": "n", "name": "YOLO11n", "path": "models/yolo11n.pt", "vram_mb": 1024, "tier": 1},
    "s": {"key": "s", "name": "YOLO11s", "path": "models/yolo11s.pt", "vram_mb": 2048, "tier": 2},
    "m": {"key": "m", "name": "YOLO11m", "path": "models/yolo11m.pt", "vram_mb": 4096, "tier": 3},
    "l": {"key": "l", "name": "YOLO11l", "path": "models/yolo11l.pt", "vram_mb": 8192, "tier": 4},
    "x": {"key": "x", "name": "YOLO11x", "path": "models/yolo11x.pt", "vram_mb": 12288, "tier": 5},
}

YOLO_MODEL_PATH = "models/yolo11s.pt"

SHOW_FPS = True
SHOW_FACE_BOX = True
SHOW_HAND_LANDMARKS = True
SHOW_WINDOW = True

ENABLE_TRACKING = False
FILTER_CLASSES = ""

SAVE_VIDEO = False
OUTPUT_VIDEO_DIR = "outputs/videos"

YOLO_NMS_IoU = 0.45
GESTURE_SMOOTHING_WINDOW = 5

SHOW_HAND_BBOX = True
SHOW_GESTURE_CONFIDENCE = True
SHOW_GESTURE_HISTORY = True
MAX_GESTURE_HISTORY = 5

OPTIMIZE = "none"
AUTO_TENSORRT = True
USE_FP16 = True