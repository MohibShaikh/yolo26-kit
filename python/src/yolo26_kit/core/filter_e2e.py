"""Filter helper for YOLO26 end-to-end (default) ONNX export output."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np

from .types import COCO_CLASSES, Detection

_FormatT = Literal["dict", "arrays"]


def filter_e2e(
    output: np.ndarray,
    conf: float = 0.25,
    classes: Iterable[int] | None = None,
    min_area: float | None = None,
    format: _FormatT = "dict",
    *,
    strict: bool = False,
) -> list[Detection] | dict[str, np.ndarray]:
    """Filter and format the (N, K, 6) e2e YOLO26 output.

    See spec/decode.md Algorithm A.
    """
    if not 0.0 <= conf <= 1.0:
        raise ValueError(f"conf must be in [0, 1]; got {conf}")

    arr = np.asarray(output)
    if arr.ndim == 2:
        pass
    elif arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(
                f"batched decode not supported in v1 — got batch={arr.shape[0]}; "
                "call per-batch-item",
            )
        arr = arr[0]
    else:
        raise ValueError(f"expected (K, 6) or (1, K, 6); got shape {arr.shape}")

    if arr.shape[-1] != 6:
        raise ValueError(
            f"filter_e2e expects last dim = 6 [x1,y1,x2,y2,conf,cls]; got {arr.shape[-1]}",
        )

    arr = arr.astype(np.float32, copy=False)
    boxes = arr[:, 0:4]
    scores = arr[:, 4]
    classes_arr = arr[:, 5].astype(np.int32)

    if not np.isfinite(arr).all():
        if strict:
            raise ValueError("output contains NaN/Inf")
        finite_mask = np.isfinite(arr).all(axis=1)
    else:
        finite_mask = np.ones(arr.shape[0], dtype=bool)

    mask = finite_mask & (scores >= conf)

    # Class-id range validation: drop or error on out-of-range class ids,
    # so downstream COCO_CLASSES lookup never goes out-of-bounds.
    if classes_arr.size:
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

    # Sort: descending score, stable tiebreak by (class asc, source idx asc).
    # np.lexsort uses ASCENDING; trailing key is primary. Negate score for desc.
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
        out: list[Detection] = [
            {
                "box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "score": float(s),
                "class": int(c),
                "label": COCO_CLASSES[int(c)],
            }
            for b, s, c in zip(boxes, scores, classes_arr, strict=True)
        ]
        return out
    raise ValueError(f"unknown format: {format!r}; expected 'dict' or 'arrays'")
