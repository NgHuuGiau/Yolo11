import argparse
import logging
import time
from pathlib import Path
from typing import Any, Optional

import cv2

import config
from src.device_manager import get_device, print_device_info
from src.face_detector import FaceDetector
from src.fps_counter import FPSCounter
from src.gesture_recognizer import detect_gesture
from src.hand_detector import HandDetector
from src.hardware_profiler import (
    MODEL_TIERS,
    TIER_ORDER,
    auto_optimize,
    detect_hardware,
    ensure_models_downloaded,
    export_tensorrt,
    print_hardware_report,
    recommend_tier,
    run_benchmark,
)
from src.utils import create_folder, draw_fps, resize_frame

logging.basicConfig(level=getattr(logging, config.LOGGING_LEVEL.upper(), logging.INFO), format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_source(source_value: Any) -> Any:
    if isinstance(source_value, int):
        return source_value
    text = str(source_value).strip()
    return int(text) if text.isdigit() else text


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-Time Object and Hand Detection System")
    parser.add_argument("--source", default=config.CAMERA_SOURCE, help="Webcam index or video path")
    parser.add_argument("--model", default=config.YOLO_MODEL_PATH, help="Path to YOLO model")
    parser.add_argument("--device", default=config.DEVICE, choices=["auto", "cuda", "cpu"], help="Inference device")
    parser.add_argument("--model-version", default=config.MODEL_TIER_KEY, choices=["auto"] + TIER_ORDER, help="YOLO version: n (nano), s (small), m (medium), l (large), x (x-large), auto")
    parser.add_argument("--imgsz", type=str, default="auto", help="YOLO inference image size (auto, 416, 512, 640, 800)")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark all model tiers and imgsz combinations, then exit")
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--optimize", type=str, default="none", choices=["none", "tensorrt"], help="Inference optimization: none (default), tensorrt (FP16 .engine)")
    parser.add_argument("--track", action="store_true", default=config.ENABLE_TRACKING, help="Enable object tracking with persistent IDs")
    parser.add_argument("--classes", type=str, default=config.FILTER_CLASSES, help="Filter classes: comma-separated names (e.g. person,cell phone,bottle)")
    parser.add_argument("--save-video", nargs="?", const=True, default=config.SAVE_VIDEO, type=parse_bool, help="Save output video")
    parser.add_argument("--show", nargs="?", const=True, default=config.SHOW_WINDOW, type=parse_bool, help="Show display window")
    return parser


def _parse_class_filter(class_str: str) -> Optional[list]:
    if not class_str:
        return None
    from src.face_detector import COCO_CLASSES
    name_to_id = {v: k for k, v in COCO_CLASSES.items()}
    ids = []
    for name in class_str.split(","):
        name = name.strip().lower()
        if name in name_to_id:
            ids.append(name_to_id[name])
        elif name.isdigit():
            ids.append(int(name))
    return ids if ids else None


def create_video_writer(output_dir: str, frame_width: int, frame_height: int, fps: float) -> tuple:
    create_folder(output_dir)
    output_path = Path(output_dir) / f"result_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    safe_fps = fps if fps and fps > 0 else 20.0
    writer = cv2.VideoWriter(str(output_path), fourcc, safe_fps, (frame_width, frame_height))
    return writer, output_path


def resolve_model_path(tier_key: str, hardware_info: Optional[dict] = None) -> tuple:
    if tier_key == "auto":
        if hardware_info is None:
            hw = detect_hardware()
            recommend_tier(hw)
            return hw["tier_info"]["path"], hw["tier_info"]
        return hardware_info["tier_info"]["path"], hardware_info["tier_info"]
    if tier_key in MODEL_TIERS:
        return MODEL_TIERS[tier_key]["path"], MODEL_TIERS[tier_key]
    return config.YOLO_MODEL_PATH, MODEL_TIERS["s"]


def main() -> None:
    args = build_arg_parser().parse_args()
    source = parse_source(args.source)

    print("=== OBJECT & HAND DETECTION SYSTEM ===")

    hw = detect_hardware()
    recommend_tier(hw)
    print_hardware_report(hw)

    device_info = get_device(args.device)
    print_device_info(device_info)
    ensure_models_downloaded()

    if args.benchmark:
        print()
        run_benchmark(device_info)
        return

    model_path, tier_info = resolve_model_path(args.model_version, hardware_info=hw)
    if not args.model or args.model == config.YOLO_MODEL_PATH:
        effective_model_path = model_path
    else:
        effective_model_path = args.model

    if args.imgsz == "auto":
        effective_imgsz = tier_info["imgsz"]
    else:
        effective_imgsz = int(args.imgsz)

    print(f"\n=== CONFIG ===")
    print(f"Model: {tier_info['name']} ({effective_model_path})")
    print(f"Image size: {effective_imgsz}")
    print(f"Device: {device_info['label']}")
    print(f"Tracking: {'ON' if args.track else 'OFF'}")
    print(f"Optimize: {args.optimize.upper() if args.optimize != 'none' else 'OFF'}")
    if args.classes:
        print(f"Classes: {args.classes}")
    print()

    if args.optimize == "tensorrt":
        engine_path = export_tensorrt(effective_model_path, imgsz=effective_imgsz)
        if engine_path:
            effective_model_path = engine_path
            print(f"Using TensorRT engine: {engine_path}")
    elif config.AUTO_TENSORRT:
        opt_path = auto_optimize(effective_model_path, imgsz=effective_imgsz)
        if opt_path != effective_model_path:
            effective_model_path = opt_path
            print(f"Auto TensorRT enabled: {opt_path}")

    classes_filter = _parse_class_filter(args.classes)
    face_detector = FaceDetector(model_path=effective_model_path, device_info=device_info, conf=args.conf, imgsz=effective_imgsz, classes=classes_filter)
    face_detector.warmup()

    hand_detector = HandDetector()
    fps_counter = FPSCounter()

    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {source}. Check webcam index, file path, or camera permissions.")

    print("Press 'q' to quit.")
    print()

    writer = None
    output_path = None

    try:
        reconnect_delay = 2.0
        while True:
            success, frame = capture.read()
            if not success:
                if isinstance(source, int):
                    print(f"\nCamera lost. Reconnecting in {reconnect_delay}s... (Press Ctrl+C to quit)")
                    capture.release()
                    import time as _time
                    _time.sleep(reconnect_delay)
                    capture = cv2.VideoCapture(source)
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
                    if reconnect_delay < 10:
                        reconnect_delay = min(reconnect_delay * 1.5, 10)
                    continue
                else:
                    print("Info: reached end of video.")
                    break
            reconnect_delay = 2.0

            frame = resize_frame(frame, width=config.CAMERA_WIDTH, height=config.CAMERA_HEIGHT)

            yolo_results = face_detector.predict(frame, track=args.track)
            hands = hand_detector.detect(frame)

            if config.SHOW_FACE_BOX:
                face_detector.draw_detections(frame, yolo_results)
            if config.SHOW_HAND_LANDMARKS:
                hand_detector.draw_landmarks(frame, hands)

            fps_value = fps_counter.update()
            draw_fps(frame, fps_value, face_detector.model_name)

            if args.save_video:
                if writer is None:
                    stream_fps = capture.get(cv2.CAP_PROP_FPS)
                    height, width = frame.shape[:2]
                    writer, output_path = create_video_writer(config.OUTPUT_VIDEO_DIR, frame_width=width, frame_height=height, fps=stream_fps)
                writer.write(frame)

            if args.show:
                cv2.imshow("Object & Hand Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    from datetime import datetime
                    shot_dir = Path("outputs/screenshots")
                    shot_dir.mkdir(parents=True, exist_ok=True)
                    shot_path = shot_dir / f"shot_{datetime.now():%Y%m%d_%H%M%S}.png"
                    cv2.imwrite(str(shot_path), frame)
                    print(f"Screenshot: {shot_path}")

        if output_path is not None:
            print(f"Saved: {output_path}")
    except KeyboardInterrupt:
        print("\nInterrupted. Closing...")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        hand_detector.close()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()