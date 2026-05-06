"""Letterbox forward preprocess (HWC uint8 -> NCHW float32 in [0,1]).

Matches ultralytics LetterBox semantics: aspect-preserving resize,
center pad with gray (114, 114, 114).
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np


class LetterboxMeta(TypedDict):
    scale: float
    pad: tuple[int, int]              # (pad_x, pad_y) in target pixel units
    orig_size: tuple[int, int]        # (W_orig, H_orig)
    lb_size: tuple[int, int]          # (W_lb, H_lb) — both = target


def _resize_bilinear(img_hwc_u8: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("install yolo26-kit[pil] for preprocess") from e
    img = Image.fromarray(img_hwc_u8, mode="RGB")
    img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def letterbox_forward(
    img_hwc_u8: np.ndarray,
    target: int = 640,
    pad_value: int = 114,
) -> tuple[np.ndarray, LetterboxMeta]:
    H, W = img_hwc_u8.shape[:2]  # noqa: N806
    scale = min(target / W, target / H)
    new_w, new_h = round(W * scale), round(H * scale)
    resized = _resize_bilinear(img_hwc_u8, new_w, new_h)
    canvas = np.full((target, target, 3), pad_value, dtype=np.uint8)
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    arr = canvas.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    meta: LetterboxMeta = {
        "scale": scale,
        "pad": (pad_x, pad_y),
        "orig_size": (W, H),
        "lb_size": (target, target),
    }
    return np.ascontiguousarray(arr), meta
