# Raspberry Pi 5 Webcam Face Detection

Lightweight, self-contained face detection optimized for **Raspberry Pi 5** using YOLOv8 INT8 ONNX and ONNX Runtime CPU.

- **No PyTorch or Ultralytics needed** on the Pi (~150 MB venv vs ~1 GB).
- **Pure NumPy NMS & letterboxing** for minimal overhead and zero compilation hurdles.
- **~15–25 FPS** expected on Raspberry Pi 5 quad-core Cortex-A76.

---

## 1. Quick Setup on Raspberry Pi 5

### System Prerequisites
Ensure your Pi has system video and GUI dependencies (Raspberry Pi OS 64-bit Bookworm):

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libgl1 libglib2.0-0
sudo usermod -aG video $USER  # Ensure access to /dev/video* devices
```
*(Log out and back in if you had to add yourself to the `video` group).*

### Virtual Environment & Dependencies
```bash
cd pi
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Choose How to Run

### Option A: Real-Time Web Stream (View on Phone / Laptop / PC)
Stream live video with detection boxes and interactive sliders to any browser on your network:

```bash
python app_web.py --camera 0 --port 5000
```
Open in browser:
```text
http://<raspberry-pi-ip>:5000
```
*(Find your Pi IP using `hostname -I`)*

### Option B: Local HDMI Display
If your Pi is connected directly to a monitor via HDMI:
```bash
python detect_webcam.py
```
*(Press **`q`** or **`ESC`** in the window to quit)*

### Option C: Headless Mode (Save to Folder)
```bash
python detect_webcam.py --headless --save-dir ./output_frames
```

### Option D: Hardware Benchmark Suite (FP32 vs. INT8 & Thread Scaling)
Measure stage-by-stage latency, CPU thread scaling (`1, 2, 3, 4` cores), memory footprint, and thermal stability:
```bash
python bench_pi.py --warmup 30 --runs 150
```
Outputs `reports/pi_benchmark.md` and `reports/pi_benchmark.json`.

### Options Reference (`detect_webcam.py`)
| Argument | Default | Description |
|---|---|---|
| `--model` | `best_int8.onnx` | Path to the ONNX model file |
| `--camera` | `0` | Camera device index (`0`, `2`, ...) or video path / RTSP URL |
| `--width` | `640` | Capture width |
| `--height` | `480` | Capture height |
| `--conf` | `0.25` | Confidence score threshold |
| `--iou` | `0.45` | NMS IoU overlap threshold |
| `--threads` | `4` | CPU inference threads (Pi 5 has 4 cores) |
| `--headless` | `False` | Run headless without opening display window |
| `--save-dir` | `None` | Directory to save annotated `.jpg` frames |
| `--max-frames`| `None` | Max frames to process before stopping |

---

## 3. Files in This Folder

- `best_int8.onnx` — Quantized INT8 face detection model (3.2 MB).
- `face_detector.py` — Self-contained detector (letterbox, ORT inference, NumPy NMS).
- `detect_webcam.py` — Main CLI entrypoint for live HDMI or headless webcam detection.
- `app_web.py` — Flask MJPEG web streaming server with live telemetry and interactive sliders.
- `bench_pi.py` — Automated edge CPU benchmark script (FP32 vs INT8, stage breakdown, thread scaling, thermals).
- `requirements.txt` — Minimal dependencies (`numpy`, `opencv-python`, `onnxruntime`, `flask`).
