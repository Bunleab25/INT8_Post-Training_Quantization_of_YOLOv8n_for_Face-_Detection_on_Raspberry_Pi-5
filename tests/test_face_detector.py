"""Unit tests for pi.face_detector — all mocked, no model file needed."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from pi.face_detector import (
    FaceDetector,
    draw_status_banner,
    letterbox,
    nms,
    resolve_default_model_path,
    update_ema_fps,
)


class TestLetterbox(unittest.TestCase):
    def test_square_frame_unchanged_shape(self):
        img = np.zeros((416, 416, 3), dtype=np.uint8)
        out, ratio, (dw, dh) = letterbox(img, 416)
        self.assertEqual(out.shape, (416, 416, 3))
        self.assertEqual(ratio, 1.0)
        self.assertEqual(dw, 0)
        self.assertEqual(dh, 0)

    def test_wide_frame_pads_height(self):
        img = np.zeros((240, 640, 3), dtype=np.uint8)
        out, ratio, (dw, dh) = letterbox(img, 416)
        self.assertEqual(out.shape, (416, 416, 3))
        self.assertGreater(dh, 0)

    def test_tall_frame_pads_width(self):
        img = np.zeros((640, 240, 3), dtype=np.uint8)
        out, ratio, (dw, dh) = letterbox(img, 416)
        self.assertEqual(out.shape, (416, 416, 3))
        self.assertGreater(dw, 0)


class TestNMS(unittest.TestCase):
    def test_suppresses_overlapping_boxes(self):
        boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52], [100, 100, 150, 150]])
        scores = np.array([0.9, 0.8, 0.7])
        keep = nms(boxes, scores, iou_thresh=0.45)
        self.assertIn(0, keep)
        self.assertNotIn(1, keep)
        self.assertIn(2, keep)

    def test_keeps_non_overlapping_boxes(self):
        boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]])
        scores = np.array([0.9, 0.8])
        keep = nms(boxes, scores, iou_thresh=0.45)
        self.assertEqual(set(keep), {0, 1})


class TestDetectScaling(unittest.TestCase):
    def test_boxes_in_original_coords(self):
        # Fake ORT output: one box at centre of 416x416 letterbox space
        raw_output = np.zeros((1, 5, 1), dtype=np.float32)
        # cx=208, cy=208, w=100, h=100, conf=0.9
        raw_output[0, :, 0] = [208, 208, 100, 100, 0.9]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("onnxruntime.InferenceSession") as MockSession:
            mock_sess = MagicMock()
            mock_sess.run.return_value = [raw_output]
            mock_sess.get_inputs.return_value = [MagicMock(name="images")]
            MockSession.return_value = mock_sess
            det = FaceDetector("fake.onnx")
            results = det.detect(frame)

        self.assertGreaterEqual(len(results), 1)
        box = results[0]["box"]
        # Coordinates must stay within original frame bounds
        self.assertTrue(0 <= box[0] < 640)
        self.assertTrue(0 <= box[1] < 480)
        self.assertLessEqual(box[2], 640)
        self.assertLessEqual(box[3], 480)

    def test_detect_timed_returns_stage_breakdown_with_detections(self):
        raw_output = np.zeros((1, 5, 1), dtype=np.float32)
        raw_output[0, :, 0] = [208, 208, 100, 100, 0.9]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("onnxruntime.InferenceSession") as MockSession:
            mock_sess = MagicMock()
            mock_sess.run.return_value = [raw_output]
            mock_sess.get_inputs.return_value = [MagicMock(name="images", shape=[1, 3, 416, 416])]
            MockSession.return_value = mock_sess
            det = FaceDetector("fake.onnx")
            results, timings = det.detect_timed(frame)

        self.assertEqual(len(results), 1)
        for key in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"):
            self.assertIn(key, timings)
            self.assertGreaterEqual(timings[key], 0.0)

    def test_detect_timed_returns_stage_breakdown_when_no_detections(self):
        raw_output = np.zeros((1, 5, 1), dtype=np.float32)
        raw_output[0, :, 0] = [208, 208, 100, 100, 0.01]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("onnxruntime.InferenceSession") as MockSession:
            mock_sess = MagicMock()
            mock_sess.run.return_value = [raw_output]
            mock_sess.get_inputs.return_value = [MagicMock(name="images", shape=[1, 3, 416, 416])]
            MockSession.return_value = mock_sess
            det = FaceDetector("fake.onnx")
            results, timings = det.detect_timed(frame)

        self.assertEqual(results, [])
        for key in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"):
            self.assertIn(key, timings)
            self.assertGreaterEqual(timings[key], 0.0)

    def test_hud_helpers_and_default_model_path(self):
        fps1 = update_ema_fps(0.0, 0.05)
        self.assertAlmostEqual(fps1, 20.0, places=3)
        fps2 = update_ema_fps(fps1, 0.1)
        self.assertAlmostEqual(fps2, 0.9 * 20.0 + 0.1 * 10.0, places=3)

        frame = np.zeros((100, 300, 3), dtype=np.uint8)
        annotated = draw_status_banner(frame, "FPS: 20.0 | Faces: 1")
        self.assertEqual(annotated.shape, (100, 300, 3))
        self.assertTrue(np.any(annotated > 0))
        self.assertTrue(resolve_default_model_path().endswith("best_int8.onnx"))


if __name__ == "__main__":
    unittest.main()
