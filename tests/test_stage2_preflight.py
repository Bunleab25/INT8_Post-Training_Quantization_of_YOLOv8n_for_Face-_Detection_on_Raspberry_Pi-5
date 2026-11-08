import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.common import fmt, last_epoch, load_expected_checkpoint_hash, load_json, sha256_file, source_name
from scripts.eval_core import bucket_from_fraction, iou_matrix, load_gt, size_bucket
from scripts.stage2_preflight import (
    resolve_calibration_images,
    validate_calibration_manifest,
    validate_config,
)


def _write_checkpoint(root: Path) -> None:
    checkpoint = root / "models/stage1/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake-weights")
    digest = __import__("hashlib").sha256(b"fake-weights").hexdigest()
    (root / "models/stage1/best.sha256").write_text(
        f"{digest}  models/stage1/best.pt\n", encoding="utf-8"
    )


def _base_config(root: Path, manifest: Path, ood: Path) -> dict:
    return {
        "stage1_gate4_approved": True,
        "target_device": "NVIDIA GeForce RTX 3050 Laptop GPU",
        "runtime": "onnxruntime",
        "latency_target_ms": 10,
        "allowed_map50_95_drop": 0.02,
        "allowed_ood_map50_95_drop": 0.03,
        "imgsz": 416,
        "latency_definition": "model_only",
        "warmup_runs": 50,
        "bench_runs": 200,
        "calibration_count": 1,
        "ood_dir": str(ood),
        "calibration_manifest": str(manifest),
    }


class CalibrationManifestTests(unittest.TestCase):
    def test_accepts_existing_train_and_validation_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "datasets/face_v2_dedup/train/images/face.jpg"
            image.parent.mkdir(parents=True)
            image.touch()
            manifest = root / "calibration/images.json"
            manifest.parent.mkdir()
            manifest.write_text(json.dumps({
                "images": [{"path": str(image), "split": "train"}]
            }), encoding="utf-8")

            self.assertEqual(validate_calibration_manifest(manifest, root), [])

    def test_rejects_test_split_and_test_image_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "datasets/face_v2_dedup/test/images/face.jpg"
            image.parent.mkdir(parents=True)
            image.touch()
            manifest = root / "calibration/images.json"
            manifest.parent.mkdir()
            manifest.write_text(json.dumps({
                "images": [{"path": str(image), "split": "test"}]
            }), encoding="utf-8")

            problems = validate_calibration_manifest(manifest, root)
            self.assertEqual(len(problems), 2)
            self.assertTrue(any("disallowed split" in problem for problem in problems))
            self.assertTrue(any("frozen test split" in problem for problem in problems))

    def test_rejects_missing_image_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "calibration/images.json"
            manifest.parent.mkdir()
            manifest.write_text(json.dumps({
                "images": [{"path": "datasets/face_v2_dedup/train/images/missing.jpg",
                            "split": "train"}]
            }), encoding="utf-8")

            problems = validate_calibration_manifest(manifest, root)
            self.assertEqual(len(problems), 1)
            self.assertIn("does not exist", problems[0])


class ConfigFieldTests(unittest.TestCase):
    def test_requires_new_stage2_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_checkpoint(root)
            ood = root / "data/ood"
            ood.mkdir(parents=True)
            (ood / "sample.jpg").touch()
            image = root / "datasets/face_v2_dedup/train/images/face.jpg"
            image.parent.mkdir(parents=True)
            image.touch()
            manifest = root / "calibration/images.json"
            manifest.parent.mkdir()
            manifest.write_text(json.dumps({
                "images": [{"path": str(image), "split": "train"}]
            }), encoding="utf-8")

            incomplete = {
                "stage1_gate4_approved": True,
                "target_device": "NVIDIA GeForce RTX 3050 Laptop GPU",
                "runtime": "onnxruntime",
                "latency_target_ms": 10,
                "allowed_map50_95_drop": 0.02,
                "ood_dir": str(ood),
                "calibration_manifest": str(manifest),
            }
            blockers, _ = validate_config(incomplete, root)
            self.assertTrue(any("allowed_ood_map50_95_drop" in b for b in blockers))
            self.assertTrue(any("imgsz" in b for b in blockers))
            self.assertTrue(any("latency_definition" in b for b in blockers))
            self.assertTrue(any("warmup_runs" in b for b in blockers))
            self.assertTrue(any("bench_runs" in b for b in blockers))
            self.assertTrue(any("calibration_count" in b for b in blockers))

    def test_accepts_complete_config_when_runtime_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_checkpoint(root)
            ood = root / "data/ood"
            ood.mkdir(parents=True)
            (ood / "sample.jpg").touch()
            image = root / "datasets/face_v2_dedup/train/images/face.jpg"
            image.parent.mkdir(parents=True)
            image.touch()
            manifest = root / "calibration/images.json"
            manifest.parent.mkdir()
            manifest.write_text(json.dumps({
                "images": [{"path": str(image), "split": "train"}]
            }), encoding="utf-8")

            config = _base_config(root, manifest, ood)
            blockers, notes = validate_config(config, root)
            # onnxruntime may or may not be installed; if missing, that is the only blocker.
            if any("Runtime package" in b for b in blockers):
                self.assertTrue(all("Runtime package" in b for b in blockers))
            else:
                self.assertEqual(blockers, [])
                self.assertTrue(any("Frozen checkpoint verified" in n for n in notes))


class CommonAndEvalCoreTests(unittest.TestCase):
    def test_source_name_strips_roboflow_hash(self):
        p = Path("train/images/portrait_01.rf.a1b2c3d4e5.jpg")
        self.assertEqual(source_name(p), "portrait_01")
        self.assertEqual(source_name(Path("plain_face.jpg")), "plain_face")

    def test_sha256_and_expected_hash_lookup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_checkpoint(root)
            expected = load_expected_checkpoint_hash(
                root / "models/stage1/best.sha256",
                "models/stage1/best.pt",
            )
            self.assertEqual(expected, sha256_file(root / "models/stage1/best.pt"))

    def test_fmt_and_last_epoch(self):
        self.assertEqual(fmt(None), "—")
        self.assertEqual(fmt(0.123456, 4), "0.1235")
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "results.csv"
            csv_path.write_text("epoch, metrics/mAP50(B)\n0, 0.5\n46, 0.99\n", encoding="utf-8")
            self.assertEqual(last_epoch(csv_path), 47)

    def test_iou_matrix_and_size_buckets(self):
        a = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
        b = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], dtype=np.float32)
        ious = iou_matrix(a, b)
        self.assertEqual(ious.shape, (1, 2))
        self.assertAlmostEqual(float(ious[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(ious[0, 1]), 0.0, places=5)

        self.assertEqual(size_bucket(16 * 16), "small")
        self.assertEqual(size_bucket(64 * 64), "medium")
        self.assertEqual(size_bucket(128 * 128), "large")
        self.assertEqual(bucket_from_fraction(0.25, 416), "large")

    def test_load_gt_and_resolve_calibration_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lbl = root / "face.txt"
            lbl.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            boxes = load_gt(lbl, 400, 400)
            self.assertEqual(boxes.shape, (1, 4))
            np.testing.assert_allclose(boxes[0], [100.0, 100.0, 300.0, 300.0])

            img = root / "datasets/face_v2_dedup/train/images/f1.jpg"
            img.parent.mkdir(parents=True)
            img.touch()
            manifest = {"images": [{"path": str(img), "split": "train"}]}
            resolved = resolve_calibration_images(manifest, root)
            self.assertEqual(len(resolved), 1)
            self.assertEqual(resolved[0], img.resolve())


if __name__ == "__main__":
    unittest.main()
