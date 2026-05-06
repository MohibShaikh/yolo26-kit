"""Internal helper: detect channel/anchor axis split in v8-style outputs.

Single source of truth shared by ``decode_detect`` and ``v8_shape_to_e2e``.
See spec/decode.md Algorithms C and D.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _split_channel_anchor_axes(
    arr: NDArray[Any],
    num_classes: int | None = None,
) -> tuple[NDArray[Any], int]:
    """Return ``(canonical, channel_count)`` where canonical has shape ``(1, 4+nc, N)``.

    Selection rule:
      * If ``num_classes`` is provided, the channel axis is the trailing dim
        whose size equals ``4 + num_classes``. Error if neither matches.
      * Otherwise, the channel axis is the trailing dim with the *smaller* size,
        provided it is >= 5 (i.e. at least ``4 + 1`` classes). Error if both
        trailing dims are equal (ambiguous) or if the smaller dim is < 5.
    """
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(f"expected (1, 4+nc, N) or (1, N, 4+nc); got {arr.shape}")

    a, b = arr.shape[1], arr.shape[2]

    if num_classes is not None:
        target = 4 + num_classes
        a_match = a == target
        b_match = b == target
        if a_match and not b_match:
            return arr, a
        if b_match and not a_match:
            return np.transpose(arr, (0, 2, 1)), b
        if a_match and b_match:
            # Ambiguous: both axes equal target. Treat as error.
            raise ValueError(
                f"ambiguous channel/anchor axes: both trailing dims = {target} "
                f"with shape {arr.shape}",
            )
        raise ValueError(
            f"neither trailing dim matches num_classes+4={target}; got shape {arr.shape}",
        )

    # Heuristic path
    if a == b:
        raise ValueError(
            f"ambiguous channel/anchor axes: trailing dims equal in shape {arr.shape}; "
            "pass num_classes to disambiguate",
        )
    smaller = min(a, b)
    if smaller < 5:
        raise ValueError(
            f"class axis must be >=5 (4 + >=1 classes); got shape {arr.shape}",
        )
    if a < b:
        return arr, a
    return np.transpose(arr, (0, 2, 1)), b
