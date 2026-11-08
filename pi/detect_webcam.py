#!/usr/bin/env python3
"""Live Webcam Face Detection for Raspberry Pi 5.

Runs INT8 ONNX face detection on live camera feed with real-time FPS overlay
and bounding boxes. Designed to run on Raspberry Pi OS without PyTorch or Ultralytics.

Usage:
    # Run from repository root:
    python pi/detect_webcam.py

    # Or run inside the pi/ directory:
    cd pi
    python detect_webcam.py --camera 0

    # Headless mode (e.g., SSH without X11 forwarding):
    python pi/detect_webcam.py --save-dir ./output_frames --max-frames 100
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import cv2

# Allow running both from repo root and directly inside pi/
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raspberry Pi 5 Real-Time Face Detection via Webcam",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    default_model = resolve_default_model_path()

    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="Path to YOLOv8 INT8 ONNX model file",
    )
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera device index (e.g. 0, 1, 2) or video file / stream URL",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Requested capture width in pixels (0 to use camera default)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Requested capture height in pixels (0 to use camera default)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for face detection [0.0 - 1.0]",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for Non-Maximum Suppression (NMS)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads for ONNX Runtime inference",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless mode (no GUI window opened)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save annotated frames as JPEGs (auto-created)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process before exiting (None = run forever)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"[ERROR] Model file not found: {model_path}", file=sys.stderr)
        print("Please check the path or ensure 'best_int8.onnx' is present.", file=sys.stderr)
        return 1

    print("==================================================")
    print(" Raspberry Pi 5 Face Detection")
    print("==================================================")
    print(f"Model       : {model_path}")
    print(f"Camera      : {args.camera}")
    print(f"Conf / IoU  : {args.conf:.2f} / {args.iou:.2f}")
    print(f"CPU Threads : {args.threads}")

    save_path: Path | None = None
    if args.save_dir:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving to   : {save_path.resolve()}")

    # Determine display capability
    has_display_env = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    gui_enabled = has_display_env and not args.headless
    if not gui_enabled:
        print("[INFO] Running in headless mode (no display window).")

    # Load model
    print("[INFO] Initializing ONNX Runtime engine...")
    try:
        detector = FaceDetector(
            model_path=str(model_path),
            conf_thresh=args.conf,
            iou_thresh=args.iou,
            num_threads=args.threads,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load ONNX model: {e}", file=sys.stderr)
        return 1

    # Open video capture
    cam_src = resolve_camera_source(args.camera)
    print(f"[INFO] Opening camera source: {cam_src}...")
    cap = cv2.VideoCapture(cam_src)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {cam_src}.", file=sys.stderr)
        print("Tip: If using USB webcam, try --camera 1 or --camera 2.", file=sys.stderr)
        return 1

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera stream opened: {actual_w}x{actual_h}")
    print("[INFO] Press 'q' or 'ESC' in the window, or Ctrl+C in terminal to stop.")

    # Graceful Ctrl+C handling
    running = True

    def sigint_handler(signum, frame):
        nonlocal running
        print("\n[INFO] Termination requested. Shutting down cleanly...")
        running = False

    signal.signal(signal.SIGINT, sigint_handler)

    frame_idx = 0
    fps_smooth = 0.0
    t_prev = time.perf_counter()

    window_name = "Raspberry Pi 5 - Face Detection"

    try:
        while running:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to read frame from camera. Exiting loop.")
                break

            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            fps_smooth = update_ema_fps(fps_smooth, dt)

            # Inference
            detections = detector.detect(frame)

            # Draw boxes and status banner
            draw_detections(frame, detections)
            num_faces = len(detections)
            draw_status_banner(frame, f"FPS: {fps_smooth:.1f} | Faces: {num_faces}", org=(12, 30), font_scale=0.75)

            # Optional save
            if save_path is not None:
                frame_filename = save_path / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_filename), frame)

            # GUI display
            if gui_enabled:
                try:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):  # 'q' or Esc
                        print("[INFO] Exit key pressed.")
                        break
                except cv2.error as cv_err:
                    print(f"[WARN] cv2.imshow unavailable ({cv_err}). Falling back to headless.")
                    gui_enabled = False

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                print(f"[INFO] Reached max-frames limit ({args.max_frames}). Stopping.")
                break

    finally:
        cap.release()
        if gui_enabled:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print(f"[INFO] Finished. Processed {frame_idx} frames. Clean exit.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
