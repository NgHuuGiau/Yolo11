import os
from pathlib import Path
from urllib.request import urlretrieve

import cv2

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import mediapipe as mp

import config


class HandDetector:
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    FINGERTIP_IDS = [4, 8, 12, 16, 20]
    GREEN = (0, 255, 0)
    DARK_GREEN = (0, 180, 0)

    def __init__(
        self,
        model_path=config.HAND_LANDMARKER_MODEL_PATH,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        min_presence_confidence=0.5,
    ):
        self.model_path = Path(model_path)
        self.max_num_hands = max_num_hands
        self.timestamp_ms = 0

        self.base_options_cls = mp.tasks.BaseOptions
        self.vision = mp.tasks.vision
        self.drawing_utils = mp.tasks.vision.drawing_utils
        self.drawing_styles = mp.tasks.vision.drawing_styles
        self.drawing_spec_cls = self.drawing_utils.DrawingSpec
        self.hand_connections = (
            self.vision.HandLandmarksConnections.HAND_PALM_CONNECTIONS
            + self.vision.HandLandmarksConnections.HAND_THUMB_CONNECTIONS
            + self.vision.HandLandmarksConnections.HAND_INDEX_FINGER_CONNECTIONS
            + self.vision.HandLandmarksConnections.HAND_MIDDLE_FINGER_CONNECTIONS
            + self.vision.HandLandmarksConnections.HAND_RING_FINGER_CONNECTIONS
            + self.vision.HandLandmarksConnections.HAND_PINKY_FINGER_CONNECTIONS
        )

        self._ensure_model_exists()

        options = self.vision.HandLandmarkerOptions(
            base_options=self.base_options_cls(model_asset_path=str(self.model_path.resolve())),
            running_mode=self.vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hands = self.vision.HandLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        if self.model_path.exists():
            return

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Info: '{self.model_path}' not found. Downloading MediaPipe hand model from {self.MODEL_URL}"
        )
        urlretrieve(self.MODEL_URL, str(self.model_path))

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.timestamp_ms += 33
        results = self.hands.detect_for_video(mp_image, self.timestamp_ms)
        detected_hands = []

        for index, hand_landmarks in enumerate(results.hand_landmarks):
            handedness = None
            if index < len(results.handedness) and results.handedness[index]:
                handedness = results.handedness[index][0].category_name

            landmarks = []
            for landmark in hand_landmarks:
                landmarks.append(
                    {
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "px": int(landmark.x * frame.shape[1]),
                        "py": int(landmark.y * frame.shape[0]),
                    }
                )

            detected_hands.append(
                {
                    "landmarks": landmarks,
                    "handedness": handedness,
                    "raw_landmarks": hand_landmarks,
                }
            )

        return detected_hands

    def draw_landmarks(self, frame, hands):
        landmark_style = {
            index: self.drawing_spec_cls(color=self.GREEN, thickness=2, circle_radius=3)
            for index in range(21)
        }
        for fingertip_id in self.FINGERTIP_IDS:
            landmark_style[fingertip_id] = self.drawing_spec_cls(
                color=self.GREEN, thickness=2, circle_radius=5
            )

        connection_style = {
            (connection.start, connection.end): self.drawing_spec_cls(
                color=self.DARK_GREEN, thickness=2, circle_radius=2
            )
            for connection in self.hand_connections
        }

        for hand in hands:
            self.drawing_utils.draw_landmarks(
                frame,
                hand["raw_landmarks"],
                self.hand_connections,
                landmark_style,
                connection_style,
            )

            label = hand.get("handedness")
            if label:
                wrist = hand["landmarks"][0]
                cv2.putText(
                    frame,
                    label,
                    (wrist["px"] + 10, max(20, wrist["py"] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    self.GREEN,
                    2,
                    cv2.LINE_AA,
                )
        return frame

    def close(self):
        self.hands.close()
