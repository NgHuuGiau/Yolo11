import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from flask import Flask, Response, jsonify, render_template_string

import config
from src.device_manager import get_device
from src.face_detector import FaceDetector
from src.fps_counter import FPSCounter
from src.gesture_recognizer import detect_gesture
from src.hand_detector import HandDetector
from src.hardware_profiler import (
    auto_optimize,
    detect_hardware,
    ensure_models_downloaded,
    export_tensorrt,
    print_hardware_report,
    recommend_tier,
)
from src.utils import draw_fps, draw_gesture, draw_gesture_history, draw_hand_bbox, draw_hand_label, draw_object_count, resize_frame

logging.basicConfig(level=getattr(logging, config.LOGGING_LEVEL.upper(), logging.INFO), format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Object & Hand Detection</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0a0a; color: #fff; font-family: sans-serif; text-align: center; padding: 20px; }
h1 { color: #0f0; margin-bottom: 10px; font-size: 1.5em; }
img { max-width: 100%; border: 2px solid #333; border-radius: 8px; }
.info { margin-top: 15px; color: #888; font-size: 0.9em; }
.status { margin-top: 10px; color: #0f0; font-size: 0.85em; font-family: monospace; }
.settings { margin-top: 15px; display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; }
.settings label { background: #1a1a1a; padding: 6px 12px; border-radius: 6px; font-size: 0.8em; color: #ccc; cursor: pointer; }
.settings input { margin-right: 4px; }
.footer { margin-top: 20px; color: #555; font-size: 0.75em; }
</style>
</head>
<body>
<h1>Object &amp; Hand Detection</h1>
<div class="status" id="status">Connecting...</div>
<img src="/video_feed" id="feed">
<div class="info">Streaming from webcam — detection running on server</div>
<div class="settings">
<label><input type="checkbox" checked id="chkBoxes" onchange="toggle('boxes')"> Boxes</label>
<label><input type="checkbox" checked id="chkLandmarks" onchange="toggle('landmarks')"> Landmarks</label>
<label><input type="checkbox" checked id="chkHandBbox" onchange="toggle('handbbox')"> Hand BBox</label>
<label><input type="checkbox" checked id="chkGesture" onchange="toggle('gesture')"> Gesture</label>
<label><input type="checkbox" checked id="chkHistory" onchange="toggle('history')"> History</label>
</div>
<div class="footer">Press Ctrl+C in terminal to stop</div>
<script>
function toggle(feature) {
  fetch('/toggle/' + feature).catch(function(){});
}
setInterval(function(){
  fetch('/status').then(function(r){ return r.json(); }).then(function(d){
    document.getElementById('status').textContent =
      'FPS: ' + (d.fps || '--') + ' | Objects: ' + (d.objects || 0) + ' | Gesture: ' + (d.gesture || '--');
  }).catch(function(){});
}, 1000);
</script>
</body>
</html>
"""

frame_buffer = None
buffer_lock = threading.Lock()

toggles = {"boxes": True, "landmarks": True, "handbbox": True, "gesture": True, "history": True}
status_data = {"fps": 0, "objects": 0, "gesture": ""}


def generate_frames() -> Any:
    while True:
        with buffer_lock:
            if frame_buffer is None:
                time.sleep(0.03)
                continue
            ret, jpeg = cv2.imencode(".jpg", frame_buffer, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                time.sleep(0.03)
                continue
            data = jpeg.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        time.sleep(0.03)


def capture_loop(source: Any, device: str, conf: float, classes_filter: Optional[list], track: bool, imgsz: int) -> None:
    global frame_buffer, status_data

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    if not cap.isOpened():
        logger.error("Cannot open source: %s", source)
        return

    device_info = get_device(device)
    model_path = config.YOLO_MODEL_PATH

    if config.OPTIMIZE == "tensorrt":
        model_path = export_tensorrt(model_path, imgsz=imgsz) or model_path
    elif config.AUTO_TENSORRT:
        model_path = auto_optimize(model_path, imgsz=imgsz) or model_path

    face_detector = FaceDetector(model_path=model_path, device_info=device_info, conf=conf, imgsz=imgsz, classes=classes_filter)
    face_detector.warmup()

    hand_detector = HandDetector()
    fps_counter = FPSCounter()

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.03)
            continue

        frame = resize_frame(frame, width=config.CAMERA_WIDTH, height=config.CAMERA_HEIGHT)

        yolo_results = face_detector.predict(frame, track=track)
        hands = hand_detector.detect(frame)

        if toggles.get("boxes", True):
            face_detector.draw_detections(frame, yolo_results)
        if toggles.get("landmarks", True):
            hand_detector.draw_landmarks(frame, hands)

        for hand in hands:
            if toggles.get("handbbox", True):
                draw_hand_bbox(frame, hand)
            handedness = hand.get("handedness")
            if handedness and hand["landmarks"]:
                wrist = hand["landmarks"][0]
                draw_hand_label(frame, handedness, wrist["px"], wrist["py"])

        fps_value = fps_counter.update()
        if toggles.get("gesture", True):
            draw_fps(frame, fps_value, face_detector.model_name)

        gestures = detect_gesture(hands)
        gesture_text = ""
        if gestures and toggles.get("gesture", True):
            draw_gesture(frame, gestures[0], hands[0] if hands else None)
            gesture_text = gestures[0].get("gesture", "")

        obj_count = 0
        for result in yolo_results:
            if result.boxes is not None:
                obj_count += len(result.boxes)
        if obj_count > 0:
            draw_object_count(frame, obj_count)

        with buffer_lock:
            frame_buffer = frame.copy()

        status_data = {"fps": round(fps_value, 1), "objects": obj_count, "gesture": gesture_text}


app = Flask(__name__)


@app.route("/")
def index() -> Any:
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed() -> Any:
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status_endpoint() -> Any:
    return jsonify(status_data)


@app.route("/toggle/<feature>")
def toggle_feature(feature: str) -> Any:
    if feature in toggles:
        toggles[feature] = not toggles[feature]
    return ("", 204)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Web UI for Object & Hand Detection")
    parser.add_argument("--source", default=config.CAMERA_SOURCE, type=lambda x: int(x) if x.isdigit() else x)
    parser.add_argument("--device", default=config.DEVICE, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model-version", default="auto", choices=["auto", "n", "s", "m", "l", "x"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD)
    parser.add_argument("--track", action="store_true", default=False)
    parser.add_argument("--classes", type=str, default="")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    args = parser.parse_args()

    print("=== OBJECT & HAND DETECTION - WEB UI ===")
    hw = detect_hardware()
    recommend_tier(hw)
    print_hardware_report(hw)
    ensure_models_downloaded()

    from src.face_detector import COCO_CLASSES
    classes_filter = None
    if args.classes:
        name_to_id = {v: k for k, v in COCO_CLASSES.items()}
        classes_filter = [name_to_id.get(c.strip().lower()) for c in args.classes.split(",") if c.strip().lower() in name_to_id]

    t = threading.Thread(target=capture_loop, args=(args.source, args.device, args.conf, classes_filter, args.track, args.imgsz), daemon=True)
    t.start()

    def handle_shutdown(signum, frame):
        print("\nShutting down web server...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    print(f"Web UI: http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()