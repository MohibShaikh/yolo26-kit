"""Decoder for non-e2e raw YOLO26 ONNX exports (1, 4+nc, N).

See spec/decode.md Algorithm D.
"""
from __future__ import annotations

from typing import Literal, Union

import numpy as np

from .types import COCO_CLASSES, Detection

_FormatT = Literal["dict", "arrays"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_detect(
    output: np.ndarray,
    conf: float = 0.25,
    format: _FormatT = "dict",
    *,
    assume_sigmoid: bool = True,
    strict: bool = False,
    strict_dtype: bool = False,
) -> Union[list[Detection], dict[str, np.ndarray]]:
    if not 0.0 <= conf <= 1.0:
        raise ValueError(f"conf must be in [0, 1]; got {conf}")
    arr = np.asarray(output)
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(f"expected (1, 4+nc, N) or (1, N, 4+nc); got {arr.shape}")

    a, b = arr.shape[1], arr.shape[2]
    # Channel axis is the smaller dim (4+nc << N typically). Require it >= 5.
    if a <= b:
        canonical = arr
        ch = a
    else:
        canonical = np.transpose(arr, (0, 2, 1))
        ch = b
    if ch < 5:
        raise ValueError(
            f"class axis must be ≥5 (4 + ≥1 classes); got {arr.shape}"
        )

    nc = canonical.shape[1] - 4

    if arr.dtype != np.float32:
        if strict_dtype:
            raise ValueError(f"expected float32; got {arr.dtype}")
        canonical = canonical.astype(np.float32, copy=False)

    boxes_cxcywh = canonical[0, 0:4, :]
    cls = canonical[0, 4:, :]
    if not assume_sigmoid:
        cls = _sigmoid(cls)

    scores = cls.max(axis=0)
    classes = cls.argmax(axis=0).astype(np.int32)

    cx, cy, w, h = boxes_cxcywh[0], boxes_cxcywh[1], boxes_cxcywh[2], boxes_cxcywh[3]
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    finite = np.isfinite(scores) & np.isfinite(boxes).all(axis=1)
    if not finite.all() and strict:
        raise ValueError("output contains NaN/Inf")
    valid_box = (w > 0) & (h > 0)
    mask = finite & valid_box & (scores >= conf)
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]

    src_idx = np.arange(scores.shape[0], dtype=np.int64)
    order = np.lexsort((src_idx, classes, -scores))
    boxes = boxes[order]
    scores = scores[order]
    classes = classes[order]

    if format == "arrays":
        return {
            "boxes": np.ascontiguousarray(boxes, dtype=np.float32),
            "scores": np.ascontiguousarray(scores, dtype=np.float32),
            "classes": np.ascontiguousarray(classes, dtype=np.int32),
        }
    if format == "dict":
        return [
            {
                "box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "score": float(s),
                "class": int(c),
                "label": COCO_CLASSES[int(c)],
            }
            for b, s, c in zip(boxes, scores, classes)
        ]
    raise ValueError(f"unknown format: {format!r}")
