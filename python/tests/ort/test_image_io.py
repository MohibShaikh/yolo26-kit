import io

import numpy as np
import pytest
from PIL import Image

from yolo26_kit.ort.image_io import to_hwc_uint8


def _png_bytes(w=10, h=8, color=(200, 100, 50)):
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_load_from_bytes():
    arr = to_hwc_uint8(_png_bytes())
    assert arr.shape == (8, 10, 3)
    assert arr.dtype == np.uint8


def test_passthrough_ndarray_uint8():
    a = np.zeros((4, 5, 3), dtype=np.uint8)
    out = to_hwc_uint8(a)
    assert out is a


def test_pil_image_input():
    img = Image.new("RGB", (10, 8), (10, 20, 30))
    arr = to_hwc_uint8(img)
    assert arr.shape == (8, 10, 3)


def test_unsupported_input_errors():
    with pytest.raises(TypeError):
        to_hwc_uint8(12345)
