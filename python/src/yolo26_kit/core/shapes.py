"""Shape adapters between e2e (N,K,6) and v8-style (1, 4+nc, N) layouts.

See spec/decode.md Algorithm B and C.
"""
from __future__ import annotations

from typing import cast

import numpy as np

from ._axes import _split_channel_anchor_axes


def e2e_to_v8_shape(output: np.ndarray, *, num_classes: int = 80) -> np.ndarray:
    arr = np.asarray(output)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(f"expected (K,6) or (1,K,6); got {arr.shape}")
    if arr.shape[-1] != 6:
        raise ValueError(f"e2e last dim must be 6; got {arr.shape[-1]}")

    K = arr.shape[1]  # noqa: N806 (math convention)
    out = np.zeros((1, 4 + num_classes, K), dtype=arr.dtype)
    boxes = arr[0, :, 0:4]  # xyxy
    confs = arr[0, :, 4]
    cids = arr[0, :, 5].astype(np.int64)

    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    keep = confs > 0.0
    valid_idx = np.where(keep)[0]
    valid_cids = cids[valid_idx]
    if (valid_cids < 0).any() or (valid_cids >= num_classes).any():
        raise ValueError("class id out of range for given num_classes")

    # For padded rows (conf == 0), do not write box channels — leave them
    # at the zero initialization. Only write box channels for kept rows.
    out[0, 0, valid_idx] = cx[valid_idx]
    out[0, 1, valid_idx] = cy[valid_idx]
    out[0, 2, valid_idx] = w[valid_idx]
    out[0, 3, valid_idx] = h[valid_idx]
    out[0, 4 + valid_cids, valid_idx] = confs[valid_idx]
    return cast(np.ndarray, out)


def v8_shape_to_e2e(
    output: np.ndarray,
    *,
    num_classes: int | None = None,
) -> np.ndarray:
    arr = np.asarray(output)
    canonical, _ch = _split_channel_anchor_axes(arr, num_classes=num_classes)

    boxes_cxcywh = canonical[0, 0:4, :]
    cls = canonical[0, 4:, :]

    cx, cy, w, h = boxes_cxcywh[0], boxes_cxcywh[1], boxes_cxcywh[2], boxes_cxcywh[3]
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5

    scores = cls.max(axis=0)
    classes = cls.argmax(axis=0).astype(np.float32)
    rows = np.stack([x1, y1, x2, y2, scores, classes], axis=1).astype(arr.dtype, copy=False)
    return rows[None, ...]  # (1, N, 6)
