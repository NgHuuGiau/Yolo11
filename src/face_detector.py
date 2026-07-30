import logging
from pathlib import Path
from typing import Any, Optional

import cv2
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)

YOLO_NAMES = {"yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"}

COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush",
}


class FaceDetector:
    def __init__(self, model_path: str, device_info: dict, conf: float = 0.5, imgsz: int = 640, classes: Optional[list] = None):
        self.model_path: str = model_path
        self.device_info: dict = device_info
        self.conf: float = conf
        self.imgsz: int = imgsz
        self.classes: Optional[list] = classes
        self.model: Optional[YOLO] = None
        self.model_name: str = Path(model_path).stem.upper()
        self._warmup_done: bool = False
        self.load_model()

    def load_model(self) -> None:
        path = Path(self.model_path)
        fallback_name = path.name
        if path.exists():
            model_source = str(path)
        elif fallback_name in YOLO_NAMES:
            logger.info("Downloading %s via Ultralytics...", fallback_name)
            model_source = fallback_name
        else:
            raise FileNotFoundError(f"Model not found: {path}. Valid names: {', '.join(sorted(YOLO_NAMES))}")
        self.model = YOLO(model_source)

    def warmup(self) -> None:
        if self._warmup_done:
            return
        import numpy as np
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model.predict(source=dummy, conf=self.conf, imgsz=self.imgsz, device=self.device_info["yolo_device"], half=self.device_info["use_half"], verbose=False, max_det=1)
        self._warmup_done = True

    def predict(self, frame: Any, track: bool = False) -> Any:
        if frame is None or frame.size == 0:
            return []
        use_half = self.device_info.get("use_half", config.USE_FP16)
        iou = config.YOLO_NMS_IoU
        kwargs = dict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device_info["yolo_device"],
            half=use_half,
            verbose=False,
            classes=self.classes,
            iou=iou,
        )
        if track:
            kwargs["persist"] = True
            return self.model.track(**kwargs)
        return self.model.predict(**kwargs)

    def draw_detections(self, frame: Any, results: Any) -> Any:
        for result in results:
            boxes = result.boxes
            names = result.names
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                label = names.get(cls_id, "Object") if isinstance(names, dict) else "Object"
                track_id = int(box.id[0]) if box.id is not None else None

                if track_id is not None:
                    display = f"#{track_id} {label}: {conf:.2f}"
                else:
                    display = f"{label}: {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, display, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        return frame