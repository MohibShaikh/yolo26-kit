"""Letterbox coordinate utilities. See spec/decode.md Algorithm E."""
from __future__ import annotations

import numpy as np


def letterbox_unmap(
    boxes: np.ndarray,
    orig_size: tuple[int, int],   # (W_orig, H_orig)
    lb_size: tuple[int, int],     # (W_lb, H_lb)
    scale: float,
    pad: tuple[float, float],     # (pad_x, pad_y)
) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    pad_x, pad_y = pad
    out = np.empty_like(arr)
    out[:, 0] = (arr[:, 0] - pad_x) / scale
    out[:, 1] = (arr[:, 1] - pad_y) / scale
    out[:, 2] = (arr[:, 2] - pad_x) / scale
    out[:, 3] = (arr[:, 3] - pad_y) / scale
    W, H = orig_size  # noqa: N806 (math convention)
    np.clip(out[:, 0], 0.0, W, out=out[:, 0])
    np.clip(out[:, 1], 0.0, H, out=out[:, 1])
    np.clip(out[:, 2], 0.0, W, out=out[:, 2])
    np.clip(out[:, 3], 0.0, H, out=out[:, 3])
    return out
