from pathlib import Path
from typing import Any, Optional

import cv2

from config import MAX_GESTURE_HISTORY, SHOW_GESTURE_CONFIDENCE, SHOW_GESTURE_HISTORY, SHOW_HAND_BBOX


def draw_text(frame: Any, text: str, position: tuple, color: tuple = (255, 255, 255), scale: float = 0.6, thickness: int = 2) -> Any:
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return frame


def resize_frame(frame: Any, width: Optional[int] = None, height: Optional[int] = None) -> Any:
    if width is None or height is None:
        return frame
    return cv2.resize(frame, (width, height))


def create_folder(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_screenshot(frame: Any, output_path: str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), frame)
    return output


def draw_fps(frame: Any, fps: float, model_name: Optional[str] = None) -> Any:
    cv2.putText(frame, f"FPS: {fps:.1f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
    if model_name:
        cv2.putText(frame, model_name, (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    return frame


def draw_hand_label(frame: Any, text: str, x: int, y: int) -> Any:
    cv2.putText(frame, text, (x + 15, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def draw_object_count(frame: Any, count: int) -> Any:
    cv2.putText(frame, f"Objects: {count}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    return frame


def draw_gesture(frame: Any, gesture: dict, hand: Optional[dict] = None) -> Any:
    if not gesture:
        return frame
    label = gesture.get("gesture", "")
    if not label:
        return frame
    handedness = gesture.get("handedness", "")
    text = f"{handedness}: {label}" if handedness else label
    conf = gesture.get("confidence")
    if conf is not None and SHOW_GESTURE_CONFIDENCE:
        text += f" ({conf:.0%})"
    cv2.putText(frame, text, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2, cv2.LINE_AA)
    return frame


def draw_hand_bbox(frame: Any, hand: dict) -> Any:
    if not SHOW_HAND_BBOX:
        return frame
    landmarks = hand.get("landmarks", [])
    if len(landmarks) < 21:
        return frame
    xs = [lm["px"] for lm in landmarks]
    ys = [lm["py"] for lm in landmarks]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    cv2.rectangle(frame, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (100, 255, 100), 1)
    return frame


def draw_gesture_history(frame: Any, history: list) -> Any:
    if not SHOW_GESTURE_HISTORY or not history:
        return frame
    h, w = frame.shape[:2]
    labels = []
    for h_item in history:
        g = h_item.get("gesture", "")
        hd = h_item.get("handedness", "")
        labels.append(f"{hd}: {g}" if hd else g)
    labels = labels[-MAX_GESTURE_HISTORY:]
    start_y = h - 40 - 22 * len(labels)
    for i, lbl in enumerate(labels):
        y = start_y + 22 * i
        cv2.putText(frame, lbl, (w - 250, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    return frame