import time

import cv2


class FPSCounter:
    def __init__(self) -> None:
        self.prev_time = time.perf_counter()
        self.fps = 0.0

    def update(self) -> float:
        current_time = time.perf_counter()
        delta = current_time - self.prev_time
        if delta > 0:
            instant_fps = 1.0 / delta
            self.fps = instant_fps if self.fps == 0 else (0.9 * self.fps + 0.1 * instant_fps)
        self.prev_time = current_time
        return self.fps

    def draw(self, frame, position=(10, 30), color=(0, 255, 0)):
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        return frame

