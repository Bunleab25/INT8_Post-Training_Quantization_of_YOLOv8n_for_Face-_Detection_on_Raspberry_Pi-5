#!/usr/bin/env python3
"""Real-time Web Streaming Face Detection for Raspberry Pi 5.

Streams webcam video with real-time YOLOv8 INT8 face detection overlay to a web
browser over HTTP (MJPEG stream). Viewable on any phone, laptop, or PC on the same network.

Usage:
    # Run from repository root:
    python pi/app_web.py

    # Or inside pi/:
    python app_web.py --camera 0 --port 5000

    # Access in your browser:
    http://<raspberry-pi-ip>:5000
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

# Support running directly or as a package
try:
    from pi.face_detector import (
        FaceDetector,
        draw_detections,
        draw_status_banner,
        resolve_camera_source,
        resolve_default_model_path,
        update_ema_fps,
    )
except ImportError:
    from face_detector import (
        FaceDetector,
        draw_detections,
        draw_status_banner,
        resolve_camera_source,
        resolve_default_model_path,
        update_ema_fps,
    )

app = Flask(__name__)

# Global video streaming state
class CameraStream:
    def __init__(
        self,
        model_path: str,
        camera_id: int | str = 0,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        num_threads: int = 4,
        width: int = 640,
        height: int = 480,
    ):
        self.camera_id = camera_id
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.num_threads = num_threads
        self.width = width
        self.height = height

        print(f"[INFO] Initializing FaceDetector with model: {model_path}")
        self.detector = FaceDetector(
            model_path=model_path,
            conf_thresh=self.conf_thresh,
            iou_thresh=self.iou_thresh,
            num_threads=self.num_threads,
        )

        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera device {self.camera_id}")

        if self.width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Camera started: {self.actual_w}x{self.actual_h}")

        self.lock = threading.Lock()
        self.current_jpeg: bytes | None = None
        self.fps: float = 0.0
        self.face_count: int = 0
        self.infer_time_ms: float = 0.0
        self.running: bool = True

        # Start background capture thread
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def update_settings(self, conf: float | None = None, iou: float | None = None):
        with self.lock:
            if conf is not None:
                self.conf_thresh = conf
                self.detector.conf_thresh = conf
            if iou is not None:
                self.iou_thresh = iou
                self.detector.iou_thresh = iou

    def _update_loop(self):
        prev_time = time.perf_counter()
        fps_smooth = 0.0

        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            t0 = time.perf_counter()
            detections = self.detector.detect(frame)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            # Draw bounding boxes
            draw_detections(frame, detections)

            # Compute smoothed FPS and overlay stats badge
            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            fps_smooth = update_ema_fps(fps_smooth, dt)
            num_faces = len(detections)
            overlay_text = f"FPS: {fps_smooth:.1f} | Faces: {num_faces} | {infer_ms:.1f}ms"
            draw_status_banner(frame, overlay_text, org=(12, 28), font_scale=0.7)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                jpeg_bytes = buffer.tobytes()
                with self.lock:
                    self.current_jpeg = jpeg_bytes
                    self.fps = fps_smooth
                    self.face_count = num_faces
                    self.infer_time_ms = infer_ms

        self.cap.release()

    def get_frame(self) -> bytes | None:
        with self.lock:
            return self.current_jpeg

    def get_stats(self) -> dict:
        with self.lock:
            return {
                "fps": round(self.fps, 1),
                "faces": self.face_count,
                "latency_ms": round(self.infer_time_ms, 1),
                "conf": self.conf_thresh,
                "iou": self.iou_thresh,
            }

    def stop(self):
        self.running = False


camera_stream: CameraStream | None = None


# Web Interface HTML Template
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raspberry Pi 5 Face Detection Stream</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        header {
            text-align: center;
            margin-bottom: 20px;
        }
        header h1 {
            font-size: 1.8rem;
            color: #38bdf8;
            margin-bottom: 6px;
        }
        header p {
            color: #94a3b8;
            font-size: 0.95rem;
        }
        .main-card {
            background: #1e293b;
            border-radius: 12px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);
            overflow: hidden;
            max-width: 960px;
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .video-container {
            width: 100%;
            background: #000;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 360px;
        }
        .video-container img {
            width: 100%;
            height: auto;
            max-height: 600px;
            object-fit: contain;
            display: block;
        }
        .stats-bar {
            width: 100%;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 12px;
            padding: 16px;
            background: #0f172a;
            border-bottom: 1px solid #334155;
        }
        .stat-item {
            background: #1e293b;
            padding: 10px 14px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #334155;
        }
        .stat-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            color: #94a3b8;
            letter-spacing: 0.05em;
        }
        .stat-value {
            font-size: 1.4rem;
            font-weight: bold;
            color: #38bdf8;
            margin-top: 4px;
        }
        .controls-bar {
            width: 100%;
            padding: 16px 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: space-around;
            align-items: center;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .control-group label {
            font-size: 0.9rem;
            color: #cbd5e1;
        }
        .control-group input[type="range"] {
            accent-color: #38bdf8;
            cursor: pointer;
        }
        .val-badge {
            font-weight: bold;
            color: #38bdf8;
            min-width: 40px;
        }
        footer {
            margin-top: 20px;
            color: #64748b;
            font-size: 0.85rem;
        }
    </style>
</head>
<body>
    <header>
        <h1>Raspberry Pi 5 Real-Time Face Detection</h1>
        <p>YOLOv8 INT8 ONNX Engine &bull; Live Web Stream</p>
    </header>

    <div class="main-card">
        <div class="stats-bar">
            <div class="stat-item">
                <div class="stat-label">Frame Rate</div>
                <div class="stat-value" id="val-fps">-- FPS</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Faces Detected</div>
                <div class="stat-value" id="val-faces">0</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Inference Latency</div>
                <div class="stat-value" id="val-latency">-- ms</div>
            </div>
        </div>

        <div class="video-container">
            <img id="stream" src="/video_feed" alt="Camera Feed">
        </div>

        <div class="controls-bar">
            <div class="control-group">
                <label for="conf">Confidence:</label>
                <input type="range" id="conf" min="0.05" max="0.95" step="0.05" value="0.25">
                <span class="val-badge" id="conf-val">0.25</span>
            </div>
            <div class="control-group">
                <label for="iou">NMS IoU:</label>
                <input type="range" id="iou" min="0.10" max="0.90" step="0.05" value="0.45">
                <span class="val-badge" id="iou-val">0.45</span>
            </div>
        </div>
    </div>

    <footer>
        Raspberry Pi 5 &bull; Standalone Edge Inference &bull; No PyTorch Required
    </footer>

    <script>
        // Poll stats
        setInterval(async () => {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                document.getElementById('val-fps').innerText = data.fps.toFixed(1) + ' FPS';
                document.getElementById('val-faces').innerText = data.faces;
                document.getElementById('val-latency').innerText = data.latency_ms.toFixed(1) + ' ms';
            } catch (err) {}
        }, 1000);

        // Interactive threshold adjustments
        const confSlider = document.getElementById('conf');
        const iouSlider = document.getElementById('iou');
        const confVal = document.getElementById('conf-val');
        const iouVal = document.getElementById('iou-val');

        async function updateSettings() {
            const conf = parseFloat(confSlider.value);
            const iou = parseFloat(iouSlider.value);
            confVal.innerText = conf.toFixed(2);
            iouVal.innerText = iou.toFixed(2);
            await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ conf, iou })
            });
        }

        confSlider.addEventListener('input', updateSettings);
        iouSlider.addEventListener('input', updateSettings);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


def generate_frames():
    """Generator that streams JPEG frames for MJPEG multipart response."""
    while True:
        if camera_stream is not None:
            frame_bytes = camera_stream.get_frame()
            if frame_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
        time.sleep(0.02)  # ~50 max fps cap to save network bandwidth


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/stats")
def api_stats():
    if camera_stream is None:
        return jsonify({"fps": 0, "faces": 0, "latency_ms": 0})
    return jsonify(camera_stream.get_stats())


@app.route("/api/settings", methods=["POST"])
def api_settings():
    if camera_stream is None:
        return jsonify({"status": "error", "message": "Camera not initialized"}), 400
    data = request.get_json() or {}
    conf = float(data["conf"]) if "conf" in data else None
    iou = float(data["iou"]) if "iou" in data else None
    camera_stream.update_settings(conf=conf, iou=iou)
    return jsonify({"status": "ok", "conf": camera_stream.conf_thresh, "iou": camera_stream.iou_thresh})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi 5 Web Streaming Face Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_model = resolve_default_model_path()

    parser.add_argument("--model", type=str, default=default_model, help="Path to INT8 ONNX model")
    parser.add_argument("--camera", default="0", help="Camera device index (e.g. 0, 1, 2)")
    parser.add_argument("--width", type=int, default=640, help="Camera capture width")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height")
    parser.add_argument("--conf", type=float, default=0.25, help="Initial confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="Initial IoU threshold")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for ONNX Runtime")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server host IP")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")

    return parser.parse_args()


def main():
    global camera_stream
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"[ERROR] Model file not found: {model_path}", file=sys.stderr)
        return 1

    cam_id: int | str = resolve_camera_source(args.camera)

    print("==================================================")
    print(" Raspberry Pi 5 Web Face Detection Stream")
    print("==================================================")
    print(f"Server URL  : http://{args.host}:{args.port}")
    print(f"Model       : {model_path}")
    print(f"Camera      : {cam_id}")

    try:
        camera_stream = CameraStream(
            model_path=str(model_path),
            camera_id=cam_id,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
            num_threads=args.threads,
            width=args.width,
            height=args.height,
        )
    except Exception as e:
        print(f"[ERROR] Could not start camera stream: {e}", file=sys.stderr)
        return 1

    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    finally:
        if camera_stream:
            camera_stream.stop()


if __name__ == "__main__":
    main()
