import math


class GestureRecognizer:
    FINGER_TIPS = {
        "thumb": 4,
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20,
    }

    FINGER_PIPS = {
        "thumb": 3,
        "index": 6,
        "middle": 10,
        "ring": 14,
        "pinky": 18,
    }

    def recognize(self, landmarks):
        if not landmarks or len(landmarks) != 21:
            return "Unknown"

        finger_states = self.count_fingers(landmarks, return_states=True)
        fingers_up = sum(finger_states.values())

        if self.detect_ok_gesture(landmarks):
            return "OK"
        if self.detect_like_gesture(landmarks, finger_states):
            return "Like"
        if self.detect_peace_gesture(finger_states):
            return "Peace"
        if fingers_up == 0:
            return "Fist"
        if fingers_up == 1 and finger_states["index"]:
            return "One Finger"
        if fingers_up == 2 and finger_states["index"] and finger_states["middle"]:
            return "Two Fingers"
        if self.detect_stop_gesture(landmarks, finger_states):
            return "Stop"
        if fingers_up >= 4:
            return "Open Hand"
        return "Unknown"

    def count_fingers(self, landmarks, return_states=False):
        states = {
            finger: self.is_finger_up(landmarks, finger)
            for finger in ["thumb", "index", "middle", "ring", "pinky"]
        }
        return states if return_states else sum(states.values())

    def is_finger_up(self, landmarks, finger):
        tip_idx = self.FINGER_TIPS[finger]
        pip_idx = self.FINGER_PIPS[finger]

        if finger == "thumb":
            wrist_x = landmarks[0]["x"]
            thumb_tip_x = landmarks[tip_idx]["x"]
            thumb_pip_x = landmarks[pip_idx]["x"]
            return abs(thumb_tip_x - wrist_x) > abs(thumb_pip_x - wrist_x)

        return landmarks[tip_idx]["y"] < landmarks[pip_idx]["y"]

    def detect_ok_gesture(self, landmarks):
        thumb_tip = landmarks[self.FINGER_TIPS["thumb"]]
        index_tip = landmarks[self.FINGER_TIPS["index"]]
        distance = self._distance(thumb_tip, index_tip)

        middle_up = self.is_finger_up(landmarks, "middle")
        ring_up = self.is_finger_up(landmarks, "ring")
        pinky_up = self.is_finger_up(landmarks, "pinky")

        return distance < 0.06 and middle_up and ring_up and pinky_up

    def detect_like_gesture(self, landmarks, finger_states):
        thumb_up = finger_states["thumb"]
        other_folded = not any(
            finger_states[finger] for finger in ["index", "middle", "ring", "pinky"]
        )
        thumb_tip = landmarks[self.FINGER_TIPS["thumb"]]
        wrist = landmarks[0]
        return thumb_up and other_folded and thumb_tip["y"] < wrist["y"]

    def detect_peace_gesture(self, finger_states):
        return (
            finger_states["index"]
            and finger_states["middle"]
            and not finger_states["ring"]
            and not finger_states["pinky"]
        )

    def detect_stop_gesture(self, landmarks, finger_states):
        if not all(finger_states.values()):
            return False
        index_tip = landmarks[self.FINGER_TIPS["index"]]
        pinky_tip = landmarks[self.FINGER_TIPS["pinky"]]
        spread = abs(index_tip["x"] - pinky_tip["x"])
        return spread > 0.18

    def _distance(self, point_a, point_b):
        return math.sqrt(
            (point_a["x"] - point_b["x"]) ** 2
            + (point_a["y"] - point_b["y"]) ** 2
        )

