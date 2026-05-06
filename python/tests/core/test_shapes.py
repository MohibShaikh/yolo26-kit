import numpy as np
import pytest

from yolo26_kit.core.shapes import e2e_to_v8_shape, v8_shape_to_e2e


def test_e2e_to_v8_shape_basic_layout():
    e2e = np.zeros((1, 2, 6), dtype=np.float32)
    e2e[0, 0] = (10, 20, 30, 40, 0.9, 5)
    e2e[0, 1] = (0, 0, 0, 0, 0.0, 0)  # padded
    v8 = e2e_to_v8_shape(e2e, num_classes=80)
    assert v8.shape == (1, 84, 2)
    # cxcywh of (10,20,30,40) = (20, 30, 20, 20)
    np.testing.assert_array_equal(v8[0, 0:4, 0], [20, 30, 20, 20])
    assert v8[0, 4 + 5, 0] == pytest.approx(0.9)
    # other class channels = 0
    assert v8[0, 4 + 0, 0] == 0
    assert v8[0, 4 + 79, 0] == 0
    # padded slot is fully zero
    assert (v8[0, :, 1] == 0).all()


def test_e2e_to_v8_shape_dtype_preserved():
    e2e = np.zeros((1, 1, 6), dtype=np.float32)
    e2e[0, 0] = (0, 0, 1, 1, 0.5, 0)
    v8 = e2e_to_v8_shape(e2e, num_classes=80)
    assert v8.dtype == np.float32


def test_e2e_to_v8_shape_squeeze_2d_input():
    e2e = np.asarray([(0, 0, 2, 2, 0.7, 1)], dtype=np.float32)
    v8 = e2e_to_v8_shape(e2e, num_classes=80)
    assert v8.shape == (1, 84, 1)


def test_e2e_to_v8_shape_rejects_bad_last_dim():
    bad = np.zeros((1, 1, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="last dim"):
        e2e_to_v8_shape(bad)


def test_v8_shape_to_e2e_round_trip():
    # Create canonical v8 shape with known detection
    v8 = np.zeros((1, 84, 1), dtype=np.float32)
    v8[0, 0:4, 0] = (20, 30, 20, 20)  # cxcywh
    v8[0, 4 + 5, 0] = 0.9
    e2e = v8_shape_to_e2e(v8)
    assert e2e.shape == (1, 1, 6)
    np.testing.assert_array_equal(e2e[0, 0, 0:4], [10, 20, 30, 40])  # xyxy
    assert e2e[0, 0, 4] == pytest.approx(0.9)
    assert int(e2e[0, 0, 5]) == 5


def test_v8_shape_to_e2e_accepts_transposed_input():
    # (1, N, 4+nc) instead of (1, 4+nc, N)
    v8_transposed = np.zeros((1, 1, 84), dtype=np.float32)
    v8_transposed[0, 0, 0:4] = (20, 30, 20, 20)
    v8_transposed[0, 0, 4 + 5] = 0.9
    e2e = v8_shape_to_e2e(v8_transposed)
    assert e2e.shape == (1, 1, 6)
    assert int(e2e[0, 0, 5]) == 5
