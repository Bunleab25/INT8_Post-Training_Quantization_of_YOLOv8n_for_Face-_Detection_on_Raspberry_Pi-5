import json
import tempfile
import unittest
from pathlib import Path

from scripts.stage2_preflight import validate_calibration_manifest, validate_config


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


if __name__ == "__main__":
    unittest.main()
