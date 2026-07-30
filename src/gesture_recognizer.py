import logging
from collections import deque
from typing import Any

import config

logger = logging.getLogger(__name__)

FINGERTIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]
MCP_IDS = [2, 5, 9, 13, 17]
INDEX_MCP = 5


def finger_states(landmarks: list) -> list:
    tips = [landmarks[i] for i in FINGERTIP_IDS]
    pips = [landmarks[i] for i in PIP_IDS]
    mcp = landmarks[INDEX_MCP]

    extended = []
    for i in range(5):
        tip = tips[i]
        pip = pips[i]
        if i == 0:
            thumb_tip_x = tip["px"]
            thumb_ip_x = pip["px"]
            mcp_x = mcp["px"]
            extended.append(thumb_tip_x > thumb_ip_x if mcp_x < thumb_ip_x else thumb_tip_x < thumb_ip_x)
        else:
            extended.append(tip["py"] < pip["py"])
    return extended


def recognize_gesture(states: list) -> str:
    count = sum(states)
    if all(states):
        return "Open Palm"
    if not any(states):
        return "Fist"
    if states == [False, True, False, False, False]:
        return "Pointing"
    if states == [False, True, True, False, False]:
        return "Peace"
    if states == [True, False, False, False, True]:
        return "Call Me"
    if states == [True, False, False, False, False]:
        return "Thumbs Up"
    if states == [True, True, False, False, False]:
        return "Gun"
    if states == [False, False, True, False, False]:
        return "Middle Finger"
    if states == [True, True, False, False, True]:
        return "Rock"
    if states == [False, True, False, False, True]:
        return "Spiderman"
    if count == 2:
        return "Two Fingers"
    if count == 3:
        return "Three Fingers"
    if count == 4:
        return "Four Fingers"
    return f"{count} Fingers"


class GestureRecognizer:
    def __init__(self, window: int = 5):
        self.window = window
        self._hand_histories: dict = {}

    def _history_key(self, hand: dict) -> str:
        return hand.get("handedness", "unknown")

    def _smooth(self, hand: dict, raw_gesture: dict) -> dict:
        key = self._history_key(hand)
        if key not in self._hand_histories:
            self._hand_histories[key] = deque(maxlen=self.window)

        states = finger_states(hand.get("landmarks", []))
        self._hand_histories[key].append(states)

        if len(self._hand_histories[key]) < self.window:
            return raw_gesture

        smoothed = []
        for i in range(5):
            vals = [h[i] for h in self._hand_histories[key]]
            smoothed.append(sum(vals) > len(vals) // 2)

        smooth_gesture = recognize_gesture(smoothed)
        conf = sum(smoothed) / 5

        raw_gesture["gesture"] = smooth_gesture
        raw_gesture["confidence"] = conf
        return raw_gesture

    def detect(self, hands: list) -> Any:
        results = []
        for hand in hands:
            landmarks = hand.get("landmarks", [])
            if len(landmarks) < 21:
                continue
            states = finger_states(landmarks)
            raw_name = recognize_gesture(states)
            raw_conf = sum(states) / 5
            raw = {"gesture": raw_name, "handedness": hand.get("handedness", "Unknown"), "confidence": raw_conf}
            if self.window > 1:
                raw = self._smooth(hand, raw)
            results.append(raw)
        return results


_detector_instance = None


def detect_gesture(hands: list) -> Any:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = GestureRecognizer(window=config.GESTURE_SMOOTHING_WINDOW)
    return _detector_instance.detect(hands)