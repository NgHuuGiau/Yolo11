import unittest
from collections import deque
from pathlib import Path

import numpy as np

import config
from src.device_manager import get_device
from src.face_detector import FaceDetector
from src.gesture_recognizer import GestureRecognizer, detect_gesture, finger_states, recognize_gesture
from src.hand_detector import HandDetector
from src.hardware_profiler import MODEL_TIERS, TIER_ORDER, detect_hardware, get_safe_fallback, recommend_tier
from src.utils import draw_fps, draw_gesture, draw_gesture_history, draw_hand_bbox, draw_object_count, resize_frame


class SystemSmokeTests(unittest.TestCase):
    def test_hand_landmarker_model_exists(self):
        if not Path(config.HAND_LANDMARKER_MODEL_PATH).exists():
            self.skipTest("Hand model not found (auto-downloaded on first run)")
        self.assertTrue(Path(config.HAND_LANDMARKER_MODEL_PATH).exists())

    def test_device_manager_returns_valid_structure(self):
        device_info = get_device("auto")
        self.assertIn("label", device_info)
        self.assertIn("torch_device", device_info)
        self.assertIn("yolo_device", device_info)
        self.assertIn(device_info["label"], {"CPU", "CUDA"})

    def test_face_detector_initializes(self):
        device_info = get_device("cpu")
        path = Path(config.YOLO_MODEL_PATH)
        if not path.exists():
            self.skipTest(f"Model not found: {path}")
        detector = FaceDetector(model_path=config.YOLO_MODEL_PATH, device_info=device_info, conf=config.CONFIDENCE_THRESHOLD, imgsz=config.IMAGE_SIZE)
        self.assertIsNotNone(detector)

    def test_face_detector_warmup(self):
        device_info = get_device("cpu")
        path = Path(config.YOLO_MODEL_PATH)
        if not path.exists():
            self.skipTest(f"Model not found: {path}")
        detector = FaceDetector(model_path=config.YOLO_MODEL_PATH, device_info=device_info, conf=config.CONFIDENCE_THRESHOLD, imgsz=config.IMAGE_SIZE)
        detector.warmup()
        self.assertTrue(detector._warmup_done)

    def test_face_detector_predict_empty_frame(self):
        device_info = get_device("cpu")
        path = Path(config.YOLO_MODEL_PATH)
        if not path.exists():
            self.skipTest(f"Model not found: {path}")
        detector = FaceDetector(model_path=config.YOLO_MODEL_PATH, device_info=device_info, conf=config.CONFIDENCE_THRESHOLD, imgsz=config.IMAGE_SIZE)
        results = detector.predict(None)
        self.assertEqual(results, [])
        results = detector.predict(np.zeros((0, 0, 3), dtype=np.uint8))
        self.assertEqual(results, [])

    def test_hand_detector_detect_runs_on_blank_frame(self):
        try:
            detector = HandDetector()
        except Exception as e:
            self.skipTest(f"HandDetector init failed: {e}")
        frame = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        results = detector.detect(frame)
        detector.close()
        self.assertIsInstance(results, list)

    def test_hand_detector_detect_none_frame(self):
        try:
            detector = HandDetector()
        except Exception as e:
            self.skipTest(f"HandDetector init failed: {e}")
        results = detector.detect(None)
        detector.close()
        self.assertEqual(results, [])

    def test_hardware_profiler_detects_structure(self):
        hw = detect_hardware()
        self.assertIn("platform", hw)
        self.assertIn("cpu_cores", hw)
        self.assertIn("ram_gb", hw)
        self.assertIn("has_cuda", hw)
        self.assertIn("gpu_name", hw)
        self.assertIn("gpu_vram_mb", hw)
        self.assertIn("tier_key", hw)
        self.assertGreater(hw["cpu_cores"], 0)
        self.assertGreater(hw["ram_gb"], 0)

    def test_hardware_profiler_recommend_tier(self):
        hw = detect_hardware()
        tier = recommend_tier(hw)
        self.assertIn(tier, TIER_ORDER)
        self.assertEqual(hw["tier_key"], tier)
        self.assertIsNotNone(hw["tier_info"])
        self.assertIn("name", hw["tier_info"])

    def test_hardware_profiler_all_tiers_have_unique_keys(self):
        keys = [t["key"] for t in MODEL_TIERS.values()]
        self.assertEqual(len(keys), len(set(keys)))

    def test_hardware_profiler_all_tiers_have_valid_paths(self):
        for tier_key in TIER_ORDER:
            t = MODEL_TIERS[tier_key]
            self.assertTrue(t["path"].endswith(".pt"))
            self.assertIn(t["key"], t["path"])

    def test_get_safe_fallback_fallback(self):
        fallback = get_safe_fallback("x")
        self.assertIn(fallback, TIER_ORDER)

    def test_gesture_recognize_fist(self):
        self.assertEqual(recognize_gesture([False, False, False, False, False]), "Fist")

    def test_gesture_recognize_open_palm(self):
        self.assertEqual(recognize_gesture([True, True, True, True, True]), "Open Palm")

    def test_gesture_recognize_pointing(self):
        self.assertEqual(recognize_gesture([False, True, False, False, False]), "Pointing")

    def test_gesture_recognize_peace(self):
        self.assertEqual(recognize_gesture([False, True, True, False, False]), "Peace")

    def test_gesture_recognize_thumbs_up(self):
        self.assertEqual(recognize_gesture([True, False, False, False, False]), "Thumbs Up")

    def test_gesture_recognize_call_me(self):
        self.assertEqual(recognize_gesture([True, False, False, False, True]), "Call Me")

    def test_gesture_recognize_middle_finger(self):
        self.assertEqual(recognize_gesture([False, False, True, False, False]), "Middle Finger")

    def test_gesture_recognize_gun(self):
        self.assertEqual(recognize_gesture([True, True, False, False, False]), "Gun")

    def test_gesture_recognize_rock(self):
        self.assertEqual(recognize_gesture([True, True, False, False, True]), "Rock")

    def test_gesture_recognize_spiderman(self):
        self.assertEqual(recognize_gesture([False, True, False, False, True]), "Spiderman")

    def test_gesture_recognize_count_based(self):
        self.assertEqual(recognize_gesture([False, True, True, False, True]), "Three Fingers")
        self.assertEqual(recognize_gesture([False, True, True, True, True]), "Four Fingers")

    def test_gesture_detect_on_empty_hands(self):
        self.assertEqual(detect_gesture([]), [])

    def test_gesture_detect_on_empty_hand_data(self):
        result = detect_gesture([{"landmarks": [], "handedness": "Left"}])
        self.assertEqual(result, [])

    def test_gesture_recognizer_smoothing(self):
        recognizer = GestureRecognizer(window=3)
        hands = []
        result = recognizer.detect(hands)
        self.assertEqual(result, [])

    def test_config_has_all_expected_keys(self):
        self.assertTrue(hasattr(config, "YOLO_NMS_IoU"))
        self.assertTrue(hasattr(config, "GESTURE_SMOOTHING_WINDOW"))
        self.assertTrue(hasattr(config, "AUTO_TENSORRT"))
        self.assertTrue(hasattr(config, "USE_FP16"))
        self.assertTrue(hasattr(config, "SHOW_HAND_BBOX"))
        self.assertTrue(hasattr(config, "SHOW_GESTURE_CONFIDENCE"))
        self.assertTrue(hasattr(config, "SHOW_GESTURE_HISTORY"))
        self.assertTrue(hasattr(config, "LOGGING_LEVEL"))

    def test_resize_frame(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = resize_frame(frame, width=640, height=360)
        self.assertEqual(result.shape, (360, 640, 3))

    def test_resize_frame_none(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = resize_frame(frame)
        self.assertIs(result, frame)

    def test_draw_fps_returns_frame(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = draw_fps(frame, 30.0, "YOLO11S")
        self.assertIs(result, frame)

    def test_draw_object_count_returns_frame(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = draw_object_count(frame, 5)
        self.assertIs(result, frame)

    def test_draw_gesture_empty(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = draw_gesture(frame, {}, None)
        self.assertIs(result, frame)

    def test_draw_gesture_with_label(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = draw_gesture(frame, {"gesture": "Fist", "handedness": "Left"}, None)
        self.assertIs(result, frame)

    def test_draw_hand_bbox_no_landmarks(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = draw_hand_bbox(frame, {"landmarks": []})
        self.assertIs(result, frame)

    def test_draw_gesture_history_empty(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = draw_gesture_history(frame, [])
        self.assertIs(result, frame)

    def test_draw_gesture_history_with_data(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = draw_gesture_history(frame, [{"gesture": "Fist", "handedness": "Left"}])
        self.assertIs(result, frame)


if __name__ == "__main__":
    unittest.main()