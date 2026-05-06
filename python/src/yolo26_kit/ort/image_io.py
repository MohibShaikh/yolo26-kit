"""Image input adapters: bytes, file path, PIL.Image, ndarray HWC -> ndarray HWC uint8."""
from __future__ import annotations

import io
import os
from typing import Any

import numpy as np


def _from_bytes(b: bytes) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("install yolo26-kit[pil] to load image bytes") from e
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def to_hwc_uint8(image: Any) -> np.ndarray:
    """Coerce input to HWC uint8 ndarray. Accepts: file path, bytes, PIL.Image, ndarray."""
    if isinstance(image, np.ndarray):
        if image.dtype == np.uint8 and image.ndim == 3 and image.shape[-1] in (1, 3, 4):
            return image
        if image.dtype != np.uint8 and image.ndim == 3:
            return np.clip(image, 0, 255).astype(np.uint8)
        raise ValueError(f"unsupported ndarray shape/dtype: {image.shape} {image.dtype}")
    if isinstance(image, (bytes, bytearray)):
        return _from_bytes(bytes(image))
    if isinstance(image, (str, os.PathLike)):
        with open(image, "rb") as f:
            return _from_bytes(f.read())
    try:
        from PIL import Image as _PILImage

        if isinstance(image, _PILImage.Image):
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except ImportError:
        pass
    raise TypeError(f"unsupported image input type: {type(image).__name__}")
