"""Smoke and unit tests for pi.detect_webcam CLI."""
import subprocess
import sys
import unittest

from pi.detect_webcam import parse_args, resolve_camera_source


class TestDetectWebcamCLI(unittest.TestCase):
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "pi/detect_webcam.py", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--model", result.stdout)
        self.assertIn("--camera", result.stdout)
        self.assertIn("--conf", result.stdout)

    def test_parse_args_defaults(self):
        args = parse_args([])
        self.assertEqual(args.conf, 0.25)
        self.assertEqual(args.iou, 0.45)
        self.assertEqual(args.threads, 4)
        self.assertEqual(args.camera, "0")
        self.assertFalse(args.headless)

    def test_parse_args_custom(self):
        args = parse_args([
            "--camera", "2",
            "--conf", "0.5",
            "--iou", "0.3",
            "--threads", "2",
            "--headless",
            "--max-frames", "10",
        ])
        self.assertEqual(args.camera, "2")
        self.assertEqual(args.conf, 0.5)
        self.assertEqual(args.iou, 0.3)
        self.assertEqual(args.threads, 2)
        self.assertTrue(args.headless)
        self.assertEqual(args.max_frames, 10)

    def test_resolve_camera_source(self):
        self.assertEqual(resolve_camera_source("0"), 0)
        self.assertEqual(resolve_camera_source("2"), 2)
        self.assertEqual(resolve_camera_source("/dev/video0"), "/dev/video0")
        self.assertEqual(
            resolve_camera_source("rtsp://192.168.1.100:554/live"),
            "rtsp://192.168.1.100:554/live",
        )


if __name__ == "__main__":
    unittest.main()
