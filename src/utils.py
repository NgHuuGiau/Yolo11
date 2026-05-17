from pathlib import Path

import cv2


def draw_text(frame, text, position, color=(255, 255, 255), scale=0.6, thickness=2):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return frame


def resize_frame(frame, width=None, height=None):
    if width is None or height is None:
        return frame
    return cv2.resize(frame, (width, height))


def create_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_screenshot(frame, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), frame)
    return output


def draw_progress_bar(
    frame,
    label,
    percent,
    position=(10, 150),
    size=(260, 20),
    bar_color=(0, 220, 120),
    background_color=(60, 60, 60),
    text_color=(255, 255, 255),
):
    x, y = position
    width, height = size
    clamped_percent = max(0, min(100, int(percent)))
    fill_width = int(width * (clamped_percent / 100.0))

    draw_text(frame, f"{label}: {clamped_percent}%", (x, y - 10), color=text_color, scale=0.55)
    cv2.rectangle(frame, (x, y), (x + width, y + height), background_color, -1)
    cv2.rectangle(frame, (x, y), (x + fill_width, y + height), bar_color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (220, 220, 220), 1)
    return frame


def draw_status_panel(frame, fps=None, device=None, gesture=None, model_name=None, handedness=None):
    lines = []
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")
    if device:
        lines.append(f"Device: {device}")
    if model_name:
        lines.append(f"Model: {model_name}")
    if gesture:
        lines.append(f"Gesture: {gesture}")
    if handedness:
        lines.append(f"Hand: {handedness}")

    if not lines:
        return frame

    panel_x = 10
    panel_y = 10
    line_height = 28
    panel_width = 240
    panel_height = 15 + len(lines) * line_height

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for index, line in enumerate(lines, start=1):
        draw_text(frame, line, (panel_x + 10, panel_y + index * line_height - 6), color=(0, 255, 255))

    return frame
