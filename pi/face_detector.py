"""Face detection inference module using ONNX Runtime (CPU) with pure NumPy NMS.

Designed for lightweight edge deployment on Raspberry Pi 5 without PyTorch
or Ultralytics dependencies.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def resolve_default_model_path() -> str:
    """Return path to bundled `pi/best_int8.onnx` or fallback to `models/stage2/best_int8.onnx`."""
    script_dir = Path(__file__).resolve().parent
    local_model = script_dir / "best_int8.onnx"
    repo_model = script_dir.parent / "models" / "stage2" / "best_int8.onnx"
    return str(local_model if local_model.exists() else repo_model)


def resolve_camera_source(source_str: str | int) -> int | str:
    """Parse camera argument: converts digit strings to integer indices."""
    if isinstance(source_str, str) and source_str.isdigit():
        return int(source_str)
    return source_str


def update_ema_fps(prev_fps: float, dt: float, alpha: float = 0.9) -> float:
    """Compute exponentially smoothed FPS from frame delta `dt` seconds."""
    inst_fps = 1.0 / dt if dt > 0 else 0.0
    return (alpha * prev_fps + (1.0 - alpha) * inst_fps) if prev_fps > 0 else inst_fps


def draw_status_banner(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int] = (12, 28),
    font_scale: float = 0.7,
) -> np.ndarray:
    """Draw outlined high-contrast yellow HUD telemetry banner on `frame` in-place."""
    cv2.putText(
        frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2, cv2.LINE_AA
    )
    return frame


def letterbox(
    img: np.ndarray,
    new_shape: int = 416,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize and pad image while preserving aspect ratio.

    Args:
        img: Input image as BGR uint8 numpy array (H, W, C).
        new_shape: Target square dimension (default: 416).
        color: Padding fill color in BGR (default: gray 114, 114, 114).

    Returns:
        (padded_img, scale_ratio, (pad_w, pad_h))
    """
    h, w = img.shape[:2]
    ratio = min(new_shape / h, new_shape / w)
    new_unpad_w = int(round(w * ratio))
    new_unpad_h = int(round(h * ratio))

    dw = (new_shape - new_unpad_w) / 2
    dh = (new_shape - new_unpad_h) / 2

    if (w, h) != (new_unpad_w, new_unpad_h):
        resized = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = img

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, ratio, (left, top)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.45,
) -> list[int]:
    """Pure NumPy Non-Maximum Suppression (NMS).

    Args:
        boxes: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        scores: Array of shape (N,) with detection confidence scores.
        iou_thresh: IoU overlap threshold for suppression.

    Returns:
        List of integer indices to keep, ordered by descending score.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        remaining_indices = np.where(iou <= iou_thresh)[0]
        order = order[remaining_indices + 1]

    return keep


class FaceDetector:
    """YOLOv8 face detector powered by ONNX Runtime CPU."""

    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        num_threads: int = 4,
    ) -> None:
        """Initialize the ONNX Runtime inference session.

        Args:
            model_path: Path to the .onnx model file.
            conf_thresh: Minimum confidence threshold.
            iou_thresh: IoU threshold for NMS suppression.
            num_threads: Number of CPU threads (default 4 for Pi 5 cores).
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        # Shape: [batch, channels, height, width]
        self.img_size = int(input_meta.shape[2]) if isinstance(input_meta.shape[2], int) else 416

    def detect_timed(self, frame: np.ndarray) -> tuple[list[dict], dict[str, float]]:
        """Detect faces and return per-stage timing breakdown in milliseconds.

        Args:
            frame: OpenCV BGR frame (H, W, 3) uint8.

        Returns:
            Tuple of (detections, timings_ms) where timings_ms has keys:
            "preprocess_ms", "inference_ms", "postprocess_ms", "total_ms".
        """
        t_start = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]

        # 1. Letterbox resize with padding + BGR -> RGB, /255.0, CHW, batch dim
        padded, ratio, (pad_w, pad_h) = letterbox(frame, self.img_size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        tensor = np.ascontiguousarray(tensor)
        t_pre = time.perf_counter()

        # 2. Model inference
        raw_outputs = self.session.run(None, {self.input_name: tensor})
        raw = raw_outputs[0]
        t_inf = time.perf_counter()

        # 3. Post-processing: YOLOv8 output parsing + confidence filter + NMS + rescaling
        if raw.ndim == 3:
            if raw.shape[1] == 5:
                preds = raw[0].T  # -> (num_anchors, 5)
            else:
                preds = raw[0]  # -> (num_anchors, 5)
        else:
            preds = raw.reshape(-1, 5)

        scores = preds[:, 4]
        mask = scores > self.conf_thresh
        results: list[dict] = []
        if np.any(mask):
            preds = preds[mask]
            scores = scores[mask]

            cx = preds[:, 0]
            cy = preds[:, 1]
            w = preds[:, 2]
            h = preds[:, 3]

            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            boxes = np.stack([x1, y1, x2, y2], axis=1)

            keep = nms(boxes, scores, self.iou_thresh)
            for idx in keep:
                b = boxes[idx]
                s = float(scores[idx])

                orig_x1 = np.clip((b[0] - pad_w) / ratio, 0, orig_w)
                orig_y1 = np.clip((b[1] - pad_h) / ratio, 0, orig_h)
                orig_x2 = np.clip((b[2] - pad_w) / ratio, 0, orig_w)
                orig_y2 = np.clip((b[3] - pad_h) / ratio, 0, orig_h)

                results.append({
                    "box": [
                        round(float(orig_x1), 1),
                        round(float(orig_y1), 1),
                        round(float(orig_x2), 1),
                        round(float(orig_y2), 1),
                    ],
                    "conf": round(s, 4),
                })

        t_post = time.perf_counter()
        timings = {
            "preprocess_ms": (t_pre - t_start) * 1000.0,
            "inference_ms": (t_inf - t_pre) * 1000.0,
            "postprocess_ms": (t_post - t_inf) * 1000.0,
            "total_ms": (t_post - t_start) * 1000.0,
        }
        return results, timings

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Detect faces in a single BGR OpenCV frame.

        Args:
            frame: OpenCV BGR frame (H, W, 3) uint8.

        Returns:
            List of dicts: [{"box": [x1, y1, x2, y2], "conf": float}, ...]
            where box coordinates are in original frame pixel coordinates.
        """
        detections, _ = self.detect_timed(frame)
        return detections


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and confidence badges on frame in-place.

    Args:
        frame: OpenCV BGR frame.
        detections: List of detection dicts with "box" and "conf".
        color: Box color in BGR (default: bright green).
        thickness: Box line thickness.

    Returns:
        Annotated frame.
    """
    for det in detections:
        box = det["box"]
        conf = det["conf"]
        x1, y1, x2, y2 = map(int, box)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw label background and text
        label = f"face {conf * 100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thick = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

        badge_y1 = max(0, y1 - txt_h - baseline - 4)
        badge_y2 = y1
        badge_x2 = min(frame.shape[1], x1 + txt_w + 4)

        cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, badge_y2 - baseline),
            font,
            font_scale,
            (0, 0, 0),
            font_thick,
            cv2.LINE_AA,
        )

    return frame
