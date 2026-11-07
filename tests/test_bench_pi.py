"""Unit tests for pi.bench_pi benchmark suite."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from pi.bench_pi import (
    benchmark_model,
    parse_args,
    parse_thread_list,
    read_cpu_temp_c,
)


class TestBenchPiHelpers(unittest.TestCase):
    def test_parse_thread_list_valid(self):
        self.assertEqual(parse_thread_list("1,2,4"), [1, 2, 4])
        self.assertEqual(parse_thread_list(" 4 "), [4])

    def test_parse_thread_list_invalid_raises(self):
        with self.assertRaises(ValueError):
            parse_thread_list("")
        with self.assertRaises(ValueError):
            parse_thread_list("0,2")
        with self.assertRaises(ValueError):
            parse_thread_list("abc")

    def test_read_cpu_temp_missing_returns_none(self):
        missing = Path("/tmp/nonexistent_thermal_zone_xyz_999")
        self.assertIsNone(read_cpu_temp_c(missing))

    def test_read_cpu_temp_valid_millidegrees(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as fh:
            fh.write("54250\n")
            tmp_path = Path(fh.name)
        try:
            temp = read_cpu_temp_c(tmp_path)
            self.assertAlmostEqual(temp, 54.25, places=2)
        finally:
            tmp_path.unlink(missing_ok=True)


class TestBenchmarkModel(unittest.TestCase):
    def test_benchmark_model_collects_stage_stats(self):
        with tempfile.NamedTemporaryFile("wb", suffix=".onnx", delete=False) as fh:
            fh.write(b"0" * (1024 * 512))  # 0.5 MB dummy file
            dummy_model = Path(fh.name)

        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        try:
            with patch("pi.bench_pi.FaceDetector") as MockDet:
                inst = MagicMock()
                inst.detect_timed.return_value = (
                    [{"box": [10.0, 10.0, 50.0, 50.0], "conf": 0.9}],
                    {
                        "preprocess_ms": 1.5,
                        "inference_ms": 12.0,
                        "postprocess_ms": 0.5,
                        "total_ms": 14.0,
                    },
                )
                MockDet.return_value = inst

                stats = benchmark_model(
                    model_path=dummy_model,
                    frames=frames,
                    num_threads=4,
                    warmup=2,
                    runs=5,
                    conf=0.25,
                    iou=0.45,
                )

            self.assertEqual(stats["threads"], 4)
            self.assertEqual(stats["runs"], 5)
            self.assertAlmostEqual(stats["preprocess_ms_median"], 1.5, places=2)
            self.assertAlmostEqual(stats["inference_ms_median"], 12.0, places=2)
            self.assertAlmostEqual(stats["postprocess_ms_median"], 0.5, places=2)
            self.assertAlmostEqual(stats["total_ms_median"], 14.0, places=2)
            self.assertGreater(stats["fps_median"], 70.0)
            self.assertAlmostEqual(stats["model_size_mb"], 0.5, places=2)
            self.assertGreaterEqual(stats["load_time_ms"], 0.0)
        finally:
            dummy_model.unlink(missing_ok=True)


class TestBenchPiCLI(unittest.TestCase):
    def test_parse_args_defaults(self):
        args = parse_args([])
        self.assertEqual(args.warmup, 30)
        self.assertEqual(args.runs, 150)
        self.assertEqual(args.threads, "1,2,3,4")

    def test_cli_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "pi/bench_pi.py", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--int8-model", result.stdout)
        self.assertIn("--fp32-model", result.stdout)
        self.assertIn("--threads", result.stdout)


if __name__ == "__main__":
    unittest.main()
