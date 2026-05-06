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
    # Create canonical v8 shape with known detection. n=1 is ambiguous via
    # the heuristic (smaller dim < 5), so we pin num_classes=80.
    v8 = np.zeros((1, 84, 1), dtype=np.float32)
    v8[0, 0:4, 0] = (20, 30, 20, 20)  # cxcywh
    v8[0, 4 + 5, 0] = 0.9
    e2e = v8_shape_to_e2e(v8, num_classes=80)
    assert e2e.shape == (1, 1, 6)
    np.testing.assert_array_equal(e2e[0, 0, 0:4], [10, 20, 30, 40])  # xyxy
    assert e2e[0, 0, 4] == pytest.approx(0.9)
    assert int(e2e[0, 0, 5]) == 5


def test_v8_shape_to_e2e_accepts_transposed_input():
    # (1, N, 4+nc) instead of (1, 4+nc, N)
    v8_transposed = np.zeros((1, 1, 84), dtype=np.float32)
    v8_transposed[0, 0, 0:4] = (20, 30, 20, 20)
    v8_transposed[0, 0, 4 + 5] = 0.9
    e2e = v8_shape_to_e2e(v8_transposed, num_classes=80)
    assert e2e.shape == (1, 1, 6)
    assert int(e2e[0, 0, 5]) == 5


def test_v8_shape_to_e2e_rejects_equal_trailing_dims():
    # Both trailing dims equal 84 is ambiguous without num_classes.
    v8 = np.zeros((1, 84, 84), dtype=np.float32)
    with pytest.raises(ValueError, match="ambiguous"):
        v8_shape_to_e2e(v8)


def test_v8_shape_to_e2e_explicit_num_classes_small_anchors():
    # K=10 anchors with 84 channels: heuristic picks the smaller dim (10)
    # incorrectly. Explicit num_classes=80 disambiguates and rotates the
    # tensor into canonical orientation.
    v8 = np.zeros((1, 84, 10), dtype=np.float32)
    v8[0, 0:4, 0] = (20, 30, 20, 20)
    v8[0, 4 + 7, 0] = 0.7
    e2e = v8_shape_to_e2e(v8, num_classes=80)
    assert e2e.shape == (1, 10, 6)
    np.testing.assert_array_equal(e2e[0, 0, 0:4], [10, 20, 30, 40])
    assert int(e2e[0, 0, 5]) == 7


def test_e2e_to_v8_then_decode_round_trip_small_k():
    # Round-trip invariant: B then D produces same elements as A.
    from yolo26_kit.core.decode_raw import decode_detect
    from yolo26_kit.core.filter_e2e import filter_e2e

    rows = np.asarray(
        [[10, 20, 30, 40, 0.9, 5], [50, 60, 70, 80, 0.7, 3]], dtype=np.float32,
    )
    e2e = rows[None, ...]  # (1, 2, 6) — K=2
    v8 = e2e_to_v8_shape(e2e, num_classes=80)
    direct = filter_e2e(e2e, conf=0.25, format="arrays")
    via_decode = decode_detect(v8, conf=0.25, num_classes=80, format="arrays")
    np.testing.assert_allclose(direct["boxes"], via_decode["boxes"], atol=1e-5)
    np.testing.assert_allclose(direct["scores"], via_decode["scores"], atol=1e-6)
    np.testing.assert_array_equal(direct["classes"], via_decode["classes"])
