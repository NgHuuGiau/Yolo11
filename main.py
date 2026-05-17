import argparse
import time
from pathlib import Path

import cv2

import config
from src.device_manager import get_device, print_device_info
from src.face_detector import FaceDetector
from src.fps_counter import FPSCounter
from src.hand_detector import HandDetector
from src.utils import create_folder, draw_progress_bar, draw_status_panel, resize_frame


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_source(source_value):
    if isinstance(source_value, int):
        return source_value
    text = str(source_value).strip()
    return int(text) if text.isdigit() else text


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Real-Time Face and Hand Gesture Recognition System"
    )
    parser.add_argument("--source", default=config.CAMERA_SOURCE, help="Webcam index or video path")
    parser.add_argument("--model", default=config.YOLO_MODEL_PATH, help="Path to YOLO model")
    parser.add_argument(
        "--device",
        default=config.DEVICE,
        choices=["auto", "cuda", "cpu"],
        help="Inference device",
    )
    parser.add_argument("--imgsz", type=int, default=config.IMAGE_SIZE, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument(
        "--save-video",
        nargs="?",
        const=True,
        default=config.SAVE_VIDEO,
        type=parse_bool,
        help="Enable or disable saving output video",
    )
    parser.add_argument(
        "--show",
        nargs="?",
        const=True,
        default=config.SHOW_WINDOW,
        type=parse_bool,
        help="Enable or disable output display window",
    )
    return parser


def create_video_writer(output_dir, frame_width, frame_height, fps):
    create_folder(output_dir)
    output_path = Path(output_dir) / f"result_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    safe_fps = fps if fps and fps > 0 else 20.0
    writer = cv2.VideoWriter(str(output_path), fourcc, safe_fps, (frame_width, frame_height))
    return writer, output_path


def print_startup_progress(step_name, percent):
    clamped = max(0, min(100, int(percent)))
    width = 30
    filled = int(width * clamped / 100)
    bar = "#" * filled + "-" * (width - filled)
    print(f"[{bar}] {clamped:3d}% - {step_name}")


def compute_runtime_progress(source, capture, frame_index, face_count, hand_count):
    if isinstance(source, int):
        score = 20
        if face_count > 0:
            score += 45
        if hand_count > 0:
            score += 35
        return "Tracking", score

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        return "Video", 0
    return "Video", (frame_index / total_frames) * 100.0


def main():
    args = build_arg_parser().parse_args()
    source = parse_source(args.source)

    print_startup_progress("Initializing system", 5)
    device_info = get_device(args.device)
    print_device_info(device_info)
    print_startup_progress("Device ready", 20)

    face_detector = FaceDetector(
        model_path=args.model,
        device_info=device_info,
        conf=args.conf,
        imgsz=args.imgsz,
    )
    print_startup_progress("Face detector ready", 45)
    hand_detector = HandDetector()
    print_startup_progress("Hand detector ready", 70)
    fps_counter = FPSCounter()

    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open source: {source}. Check webcam index, file path, or camera permissions."
        )
    print_startup_progress("Video source opened", 90)

    writer = None
    output_path = None
    frame_index = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                if isinstance(source, int):
                    print("Warning: webcam frame grab failed.")
                else:
                    print("Info: reached end of video or failed to read frame.")
                break

            frame_index += 1
            frame = resize_frame(frame, width=config.CAMERA_WIDTH, height=config.CAMERA_HEIGHT)

            yolo_results = face_detector.detect(frame)
            hands = hand_detector.detect(frame)
            face_count = len(yolo_results) if face_detector.face_only_mode else sum(
                len(result.boxes) for result in yolo_results if result.boxes is not None
            )

            handedness_label = None
            for hand in hands:
                handedness_label = hand.get("handedness", handedness_label)

            if config.SHOW_FACE_BOX:
                face_detector.draw_detections(frame, yolo_results)

            if config.SHOW_HAND_LANDMARKS:
                hand_detector.draw_landmarks(frame, hands)

            fps_value = fps_counter.update()
            draw_status_panel(
                frame=frame,
                fps=fps_value if config.SHOW_FPS else None,
                device=device_info["label"] if config.SHOW_DEVICE else None,
                gesture=None,
                model_name=face_detector.model_name,
                handedness=handedness_label,
            )

            if config.SHOW_PROGRESS_BAR:
                progress_label, progress_value = compute_runtime_progress(
                    source=source,
                    capture=capture,
                    frame_index=frame_index,
                    face_count=face_count,
                    hand_count=len(hands),
                )
                draw_progress_bar(frame, progress_label, progress_value)

            if args.save_video:
                if writer is None:
                    stream_fps = capture.get(cv2.CAP_PROP_FPS)
                    height, width = frame.shape[:2]
                    writer, output_path = create_video_writer(
                        config.OUTPUT_VIDEO_DIR,
                        frame_width=width,
                        frame_height=height,
                        fps=stream_fps,
                    )
                writer.write(frame)

            if args.show:
                cv2.imshow("Real-Time Face and Hand Gesture Recognition System", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if output_path is not None:
            print(f"Saved output video to: {output_path}")
    except KeyboardInterrupt:
        print("\nInfo: user interrupted execution. Closing application cleanly.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        hand_detector.close()
        print_startup_progress("Shutdown complete", 100)


if __name__ == "__main__":
    main()
