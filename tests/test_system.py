import unittest
from pathlib import Path

import numpy as np

import config
from src.device_manager import get_device
from src.face_detector import FaceDetector
from src.hand_detector import HandDetector


class SystemSmokeTests(unittest.TestCase):
    def test_required_model_files_exist(self):
        self.assertTrue(Path(config.HAND_LANDMARKER_MODEL_PATH).exists())
        self.assertTrue(Path(config.YOLO_MODEL_PATH).exists())

    def test_device_manager_returns_valid_structure(self):
        device_info = get_device("auto")
        self.assertIn("label", device_info)
        self.assertIn("torch_device", device_info)
        self.assertIn("yolo_device", device_info)
        self.assertIn(device_info["label"], {"CPU", "CUDA"})

    def test_face_detector_initializes(self):
        device_info = get_device("cpu")
        detector = FaceDetector(
            model_path=config.YOLO_MODEL_PATH,
            device_info=device_info,
            conf=config.CONFIDENCE_THRESHOLD,
            imgsz=config.IMAGE_SIZE,
        )
        self.assertIsNotNone(detector)

    def test_hand_detector_detect_runs_on_blank_frame(self):
        detector = HandDetector()
        frame = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        results = detector.detect(frame)
        detector.close()
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()

