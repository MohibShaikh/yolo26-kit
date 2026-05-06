import numpy as np
import pytest

from yolo26_kit.core.normalize import normalize_output


def test_passthrough_float32():
    a = np.zeros((1, 3, 6), dtype=np.float32)
    out = normalize_output(a)
    assert out.dtype == np.float32
    assert out.flags.c_contiguous


def test_promote_float16():
    a = np.zeros((1, 3, 6), dtype=np.float16)
    out = normalize_output(a)
    assert out.dtype == np.float32


def test_rejects_int_dtype():
    a = np.zeros((1, 3, 6), dtype=np.int8)
    with pytest.raises(ValueError, match="dequantize"):
        normalize_output(a)
