"""Decoder for non-e2e raw YOLO26 ONNX exports (1, 4+nc, N).

See spec/decode.md Algorithm D.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, cast

import numpy as np

from ._axes import _split_channel_anchor_axes
from .types import COCO_CLASSES, Detection

_FormatT = Literal["dict", "arrays"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, 1.0 / (1.0 + np.exp(-x)))


def decode_detect(
    output: np.ndarray,
    conf: float = 0.25,
    classes: Iterable[int] | None = None,
    min_area: float | None = None,
    format: _FormatT = "dict",
    *,
    num_classes: int | None = None,
    assume_sigmoid: bool = True,
    strict: bool = False,
    strict_dtype: bool = False,
) -> list[Detection] | dict[str, np.ndarray]:
    if not 0.0 <= conf <= 1.0:
        raise ValueError(f"conf must be in [0, 1]; got {conf}")
    arr = np.asarray(output)
    canonical, _ch = _split_channel_anchor_axes(arr, num_classes=num_classes)

    if arr.dtype != np.float32:
        if strict_dtype:
            raise ValueError(f"expected float32; got {arr.dtype}")
        canonical = canonical.astype(np.float32, copy=False)

    boxes_cxcywh = canonical[0, 0:4, :]
    cls = canonical[0, 4:, :]
    if not assume_sigmoid:
        cls = _sigmoid(cls)

    scores = cls.max(axis=0)
    classes_arr = cls.argmax(axis=0).astype(np.int32)

    cx, cy, w, h = boxes_cxcywh[0], boxes_cxcywh[1], boxes_cxcywh[2], boxes_cxcywh[3]
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    finite = np.isfinite(scores) & np.isfinite(boxes).all(axis=1)
    if not finite.all() and strict:
        raise ValueError("output contains NaN/Inf")
    mask = finite & (scores >= conf)

    # Class-id range validation (argmax may pick a class index whose
    # underlying number of classes exceeds the COCO label table).
    valid_cls = (classes_arr >= 0) & (classes_arr < len(COCO_CLASSES))
    if not valid_cls.all():
        if strict:
            raise ValueError("class_id out of range")
        mask &= valid_cls

    if classes is not None:
        allowlist = np.fromiter(classes, dtype=np.int32)
        if allowlist.size:
            if (allowlist < 0).any() or (allowlist >= len(COCO_CLASSES)).any():
                raise ValueError(f"classes allowlist out of range: {allowlist.tolist()}")
            mask &= np.isin(classes_arr, allowlist)

    if min_area is not None:
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        mask &= (widths * heights) >= min_area

    boxes = boxes[mask]
    scores = scores[mask]
    classes_arr = classes_arr[mask]

    src_idx = np.arange(scores.shape[0], dtype=np.int64)
    order = np.lexsort((src_idx, classes_arr, -scores))
    boxes = boxes[order]
    scores = scores[order]
    classes_arr = classes_arr[order]

    if format == "arrays":
        return {
            "boxes": np.ascontiguousarray(boxes, dtype=np.float32),
            "scores": np.ascontiguousarray(scores, dtype=np.float32),
            "classes": np.ascontiguousarray(classes_arr, dtype=np.int32),
        }
    if format == "dict":
        return [
            {
                "box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "score": float(s),
                "class": int(c),
                "label": COCO_CLASSES[int(c)],
            }
            for b, s, c in zip(boxes, scores, classes_arr, strict=True)
        ]
    raise ValueError(f"unknown format: {format!r}")
