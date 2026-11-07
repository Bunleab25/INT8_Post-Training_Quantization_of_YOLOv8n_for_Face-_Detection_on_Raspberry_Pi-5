# Step-by-Step Guide: Benchmarking on Raspberry Pi 5 for the Technical Report

This guide walks you through running the automated benchmark suite ([`pi/bench_pi.py`](../pi/bench_pi.py)) on your **Raspberry Pi 5** and integrating the empirical results into the technical report ([`docs/technical_report.md`](technical_report.md)).

---

## Overview of What `pi/bench_pi.py` Measures

When executed on your Raspberry Pi 5, `pi/bench_pi.py` automatically collects:

1. **FP32 vs. INT8 ONNX Model Comparison** (`exports/stage2/onnxruntime/best.onnx` vs. `pi/best_int8.onnx`)
   - Model disk size (MB) and cold-start initialization time (ms)
   - Median, mean, p95, and p99 end-to-end latency (ms) and throughput (FPS)
2. **Stage-by-Stage Pipeline Breakdown** (via [`FaceDetector.detect_timed()`](../pi/face_detector.py))
   - **Stage 1 (Preprocess):** Aspect-ratio preserving `letterbox` (640×480 $\rightarrow$ 416×416) + BGR$\rightarrow$RGB + `float32` normalization
   - **Stage 2 (ONNX Runtime Inference):** `session.run()` CPU forward pass
   - **Stage 3 (Postprocess):** Anchor decoding + confidence filtering + pure-NumPy NMS + coordinate rescaling
3. **Multi-Core CPU Thread Scaling (`1, 2, 3, 4` Cortex-A76 Cores)**
   - Measures latency and FPS scaling efficiency across 1, 2, 3, and 4 `intra_op_num_threads`
4. **System Resource & Thermal Telemetry**
   - Peak process Resident Set Size (RSS) RAM usage (MB)
   - Raspberry Pi SoC temperature (`/sys/class/thermal/thermal_zone0/temp`) before and after each run
   - Firmware throttling flags (`vcgencmd get_throttled`) to verify zero thermal/under-voltage throttling

---

## Step 1: Commit and Push from Your Development PC

First, push this repository (which includes `pi/bench_pi.py`, `pi/best_int8.onnx`, and `exports/stage2/onnxruntime/best.onnx`) to GitHub:

```bash
git add .
git commit -m "feat(pi): add edge benchmark suite and technical report"
git push origin main
```

---

## Step 2: Clone and Set Up on Raspberry Pi 5

SSH into your Raspberry Pi 5 (running Raspberry Pi OS 64-bit Bookworm) and set up the lightweight virtual environment (no PyTorch or Ultralytics needed):

```bash
# 1. Install system OpenCV runtime libraries
sudo apt update
sudo apt install -y git python3-pip python3-venv libgl1 libglib2.0-0

# 2. Clone your repository
git clone https://github.com/<your-username>/face_yolov.git
cd face_yolov

# 3. Create lightweight virtual environment (~150 MB)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r pi/requirements.txt
```

---

## Step 3: Run the Standard Benchmark Suite on Raspberry Pi 5

Ensure the Pi is idle (close browser tabs or other camera streams), then run the benchmark script from the repository root:

```bash
python pi/bench_pi.py --warmup 30 --runs 150 --threads 1,2,3,4
```

### What happens during this run:
- Runs **30 warm-up iterations** + **150 timed iterations** on `pi/best_int8.onnx` for **1, 2, 3, and 4 CPU threads**.
- Runs **30 warm-up iterations** + **150 timed iterations** on `exports/stage2/onnxruntime/best.onnx` (FP32) at **4 CPU threads**.
- Automatically writes:
  - `reports/pi_benchmark.json` (raw machine-readable metrics)
  - `reports/pi_benchmark.md` (pre-formatted Markdown tables)

View the generated report directly in the terminal:

```bash
cat reports/pi_benchmark.md
```

---

## Step 4: (Optional) Run a 5-Minute Sustained Thermal Stability Test

To prove in your technical report that the Raspberry Pi 5 maintains real-time FPS under continuous load without thermal throttling, run a 1,000-iteration stress test on all 4 cores:

```bash
python pi/bench_pi.py \
  --warmup 50 \
  --runs 1000 \
  --threads 4 \
  --out-json reports/pi_thermal_stress.json \
  --out-md reports/pi_thermal_stress.md
```

Check `temp_start_c`, `temp_end_c`, and `throttled_state` (`0x0` means zero throttling occurred) in `reports/pi_thermal_stress.md`.

---

## Step 5: Measure Live Camera End-to-End FPS (Optional Qualitative Check)

If you have a USB webcam connected to the Raspberry Pi 5, run a 100-frame headless camera capture test to record real camera I/O + inference throughput:

```bash
python pi/detect_webcam.py \
  --camera 0 \
  --width 640 \
  --height 480 \
  --threads 4 \
  --headless \
  --max-frames 100 \
  --save-dir ./reports/pi_sample_frames
```

---

## Step 6: Commit Results & Update the Technical Report

Once `reports/pi_benchmark.md` and `reports/pi_benchmark.json` are generated on the Raspberry Pi 5:

1. **Copy the tables** from `reports/pi_benchmark.md` into **Section 5** of [`docs/technical_report.md`](technical_report.md), or
2. **Push the generated report files back to GitHub** from your Raspberry Pi:
   ```bash
   git add reports/pi_benchmark.json reports/pi_benchmark.md
   git commit -m "docs(reports): add empirical Raspberry Pi 5 benchmark results"
   git push origin main
   ```

---

## CLI Reference for `pi/bench_pi.py`

| Argument | Default | Description |
|---|---|---|
| `--int8-model` | `pi/best_int8.onnx` | Path to INT8 quantized ONNX model |
| `--fp32-model` | `exports/stage2/onnxruntime/best.onnx` | Path to FP32 baseline ONNX model |
| `--images-dir` | `data/ood/images` | Directory of `.jpg` frames for evaluation |
| `--threads` | `1,2,3,4` | Comma-separated CPU thread counts to test |
| `--warmup` | `30` | Warm-up runs before timing each configuration |
| `--runs` | `150` | Timed runs per configuration |
| `--conf` | `0.25` | Detection confidence threshold |
| `--iou` | `0.45` | NMS IoU overlap threshold |
| `--out-json` | `reports/pi_benchmark.json` | Output path for JSON metrics |
| `--out-md` | `reports/pi_benchmark.md` | Output path for Markdown tables |
