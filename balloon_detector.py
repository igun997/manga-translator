"""Local ONNX instance segmentation of manga speech balloons.

The bundled model is an ONNX export of huyvux3005/manga109-segmentation-bubble
(Apache-2.0). Inference never downloads weights or sends pages to a service.
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MODEL_SIZE = 1600
MODEL_PATH = Path(__file__).resolve().parent / "manga_models" / "bubble.onnx"


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort

    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Bundled balloon detector is missing: {MODEL_PATH}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(MODEL_PATH), options, providers=["CPUExecutionProvider"])


def detect_model_bubbles(image: Image.Image, min_area: int | None = None):
    """Return instance masks for balloons, including open/borderless balloons."""
    from manga_bubbles import Bubble

    rgb = np.asarray(image if image.mode == "RGB" else image.convert("RGB"))
    height, width = rgb.shape[:2]
    scale = min(MODEL_SIZE / height, MODEL_SIZE / width)
    scaled_width, scaled_height = round(width * scale), round(height * scale)
    resized = cv2.resize(rgb, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (MODEL_SIZE - scaled_width) / 2, (MODEL_SIZE - scaled_height) / 2
    left, top = round(pad_x - 0.1), round(pad_y - 0.1)
    right, bottom = round(pad_x + 0.1), round(pad_y + 0.1)
    canvas = np.full((MODEL_SIZE, MODEL_SIZE, 3), 114, dtype=np.uint8)
    canvas[top:MODEL_SIZE - bottom, left:MODEL_SIZE - right] = resized
    tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32)
    tensor *= 1 / 255

    session = _session()
    detections, prototypes = session.run(None, {session.get_inputs()[0].name: tensor})
    candidates = detections[0].T
    candidates = candidates[candidates[:, 4] >= 0.25]
    if not len(candidates):
        return []
    boxes = candidates[:, :4].copy()
    boxes[:, :2] -= boxes[:, 2:] / 2
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), candidates[:, 4].tolist(), 0.25, 0.6)
    area_limit = min_area if min_area is not None else max(400, width * height // 400)
    proto = prototypes[0].reshape(prototypes.shape[1], -1)
    bubbles = []
    for index in np.asarray(indices).reshape(-1):
        x, y, box_width, box_height = boxes[index]
        x1 = max(0, round((x - left) / scale))
        y1 = max(0, round((y - top) / scale))
        x2 = min(width, round((x + box_width - left) / scale))
        y2 = min(height, round((y + box_height - top) / scale))
        if x2 <= x1 or y2 <= y1:
            continue
        logits = (candidates[index, 5:] @ proto).reshape(prototypes.shape[2:])
        np.clip(logits, -30, 30, out=logits)
        scores = 1 / (1 + np.exp(-logits))
        scores = cv2.resize(scores, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
        scores = cv2.resize(scores[top:MODEL_SIZE - bottom, left:MODEL_SIZE - right],
                            (width, height), interpolation=cv2.INTER_LINEAR)
        mask = (scores > 0.5).astype(np.uint8) * 255
        mask[:y1] = 0
        mask[y2:] = 0
        mask[:, :x1] = 0
        mask[:, x2:] = 0
        # The coarse model edge can include neighboring lettering; keep OCR
        # away from it before applying the separate relettering clearance.
        mask = cv2.erode(mask, np.ones((5, 5), dtype=np.uint8))
        if np.count_nonzero(mask) < area_limit:
            continue
        safe = cv2.erode(mask, np.ones((9, 9), dtype=np.uint8))
        bubbles.append(Bubble((x1, y1, x2, y2), mask, safe))
    return bubbles
