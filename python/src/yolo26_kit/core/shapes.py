"""Shape adapters between e2e (N,K,6) and v8-style (1, 4+nc, N) layouts.

See spec/decode.md Algorithm B and C.
"""
from __future__ import annotations

import numpy as np


def e2e_to_v8_shape(output: np.ndarray, *, num_classes: int = 80) -> np.ndarray:
    arr = np.asarray(output)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(f"expected (K,6) or (1,K,6); got {arr.shape}")
    if arr.shape[-1] != 6:
        raise ValueError(f"e2e last dim must be 6; got {arr.shape[-1]}")

    K = arr.shape[1]
    out = np.zeros((1, 4 + num_classes, K), dtype=arr.dtype)
    boxes = arr[0, :, 0:4]  # xyxy
    confs = arr[0, :, 4]
    cids = arr[0, :, 5].astype(np.int64)

    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    out[0, 0, :] = cx
    out[0, 1, :] = cy
    out[0, 2, :] = w
    out[0, 3, :] = h

    keep = confs > 0.0
    valid_idx = np.where(keep)[0]
    valid_cids = cids[valid_idx]
    if (valid_cids < 0).any() or (valid_cids >= num_classes).any():
        raise ValueError("class id out of range for given num_classes")
    out[0, 4 + valid_cids, valid_idx] = confs[valid_idx]

    # Zero out box channels for padded rows (conf == 0)
    pad = ~keep
    out[0, 0:4, pad] = 0
    return out


def v8_shape_to_e2e(output: np.ndarray) -> np.ndarray:
    arr = np.asarray(output)
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(f"expected (1, 4+nc, N) or (1, N, 4+nc); got {arr.shape}")

    a, b = arr.shape[1], arr.shape[2]
    # Heuristic: channel axis must be >=5 (4 box + >=1 class). When both dims
    # qualify, prefer canonical (1, 4+nc, N) where N > channels; when only one
    # qualifies (e.g. N=1 anchor), that dim is channels.
    a_ok = a >= 5
    b_ok = b >= 5
    if a_ok and b_ok:
        if a <= b:
            canonical = arr  # (1, 4+nc, N)
        else:
            canonical = np.transpose(arr, (0, 2, 1))
    elif a_ok and not b_ok:
        canonical = arr
    elif b_ok and not a_ok:
        canonical = np.transpose(arr, (0, 2, 1))
    else:
        raise ValueError(
            f"cannot infer channel axis from shape {arr.shape}; "
            "expected (1, 4+nc, N) with 4+nc >= 5"
        )

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
