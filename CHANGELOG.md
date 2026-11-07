# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] – 2026-11-07

### Added
- **Stage 1 – FP32 training** (YOLOv8n, 416 px, 100 epochs)
  - Cluster-aware deduplication via perceptual hashing (pHash / SSIM)
  - Gate 0–4 evidence: dataset audit, dedup split, three experiment runs (E0/E1/E2)
  - Champion: **E1 @ 416 px** — mAP50-95 = 0.6541, mAP50 = 0.9074
  - Frozen artifact: `models/stage1/best.pt` (SHA-256 verified)
- **Stage 2 – INT8 PTQ quantization** (ONNX Runtime, Conv-only MinMax QDQ)
  - Calibration on 500 deduplicated training images (never test-split)
  - Zero accuracy regression: INT8 mAP50-95 = 0.6549 (FP32 baseline 0.6541)
  - 3.5× size reduction: 12.1 MB FP32 → 3.5 MB INT8
  - Gate 5 latency check: median 15.0 ms on CPU (budget 20 ms) ✅
  - Frozen artifact: `models/stage2/best_int8.onnx` (SHA-256 verified)
- **Raspberry Pi 5 Edge Deployment (`pi/`)**
  - Standalone ONNX Runtime CPU detector (`pi/face_detector.py`) with pure-NumPy NMS and letterboxing (zero PyTorch/Ultralytics dependency)
  - Live webcam CLI (`pi/detect_webcam.py`) and Flask MJPEG web streaming dashboard (`pi/app_web.py`)
- Professional project structure: `README.md`, `LICENSE` (MIT), `Makefile`, `pyproject.toml`, `requirements.txt`
- CI workflow via GitHub Actions (`.github/workflows/ci.yml`)
- Architecture docs in `docs/`
