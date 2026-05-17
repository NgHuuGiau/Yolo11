from pathlib import Path

import cv2
from ultralytics import YOLO

import config


class FaceDetector:
    def __init__(self, model_path, device_info, conf=0.5, imgsz=640, face_only_mode=None):
        self.model_path = model_path
        self.device_info = device_info
        self.conf = conf
        self.imgsz = imgsz
        self.face_only_mode = config.FACE_ONLY_MODE if face_only_mode is None else face_only_mode
        self.model = None
        self.model_name = "HAAR FACE" if self.face_only_mode else Path(model_path).stem.upper()
        self.face_cascade = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
        if not self.face_only_mode:
            self.load_model()

    def load_model(self):
        path = Path(self.model_path)
        fallback_name = path.name

        if path.exists():
            model_source = str(path)
        elif fallback_name in {"yolo11s.pt", "yolo11m.pt"}:
            print(
                f"Warning: local model not found at '{path}'. "
                f"Trying Ultralytics auto-download with '{fallback_name}'."
            )
            model_source = fallback_name
        else:
            raise FileNotFoundError(
                f"Model file not found: {path}. Provide a valid model path or use yolo11s.pt / yolo11m.pt."
            )

        self.model = YOLO(model_source)

    def detect(self, frame):
        if self.face_only_mode:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
            return [{"box": (int(x), int(y), int(w), int(h)), "label": "Face"} for (x, y, w, h) in faces]

        return self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device_info["yolo_device"],
            half=self.device_info["use_half"],
            verbose=False,
        )

    def draw_detections(self, frame, results):
        if self.face_only_mode:
            for face in results:
                x, y, w, h = face["box"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                cv2.putText(
                    frame,
                    face["label"],
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            return frame

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

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(
                    frame,
                    f"{label}: {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
        return frame
