"""Output dtype normalizer. See spec/decode.md Algorithm F."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def normalize_output(output: NDArray[Any]) -> NDArray[np.float32]:
    arr = np.asarray(output)
    if np.issubdtype(arr.dtype, np.integer):
        raise ValueError(
            f"got integer dtype {arr.dtype}; dequantize the model output before passing to "
            "yolo26-kit (see TFLite/RKNN runtime docs for dequant API)",
        )
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return np.ascontiguousarray(arr)
