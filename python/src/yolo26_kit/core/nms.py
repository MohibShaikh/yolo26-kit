"""Class-aware NMS for non-e2e YOLO output. Pure numpy, no deps."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def class_aware_nms(
    boxes: NDArray[Any],     # (N, 4) xyxy
    scores: NDArray[Any],    # (N,)
    classes: NDArray[Any],   # (N,)
    iou_threshold: float = 0.45,
) -> NDArray[np.int64]:
    """Class-aware NMS. Returns indices to keep, sorted by descending score."""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    keep_all: list[int] = []
    for cls in np.unique(classes):
        mask = classes == cls
        idxs = np.nonzero(mask)[0]
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]
        keep_in_cls = _nms_single_class(cls_boxes, cls_scores, iou_threshold)
        keep_all.extend(idxs[keep_in_cls].tolist())
    keep = np.asarray(keep_all, dtype=np.int64)
    # Re-sort by descending global score
    order = np.argsort(-scores[keep])
    return keep[order]


def _nms_single_class(
    boxes: NDArray[Any], scores: NDArray[Any], iou_threshold: float
) -> NDArray[np.int64]:
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(-scores)
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
        ious = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        order = order[1:][ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)
