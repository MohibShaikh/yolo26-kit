import numpy as np
import pytest

from yolo26_kit.core.filter_e2e import filter_e2e


def _make(rows):
    """rows = list of (x1,y1,x2,y2,conf,cls). Returns (1, len, 6) float32."""
    return np.asarray([rows], dtype=np.float32)


def test_filter_returns_dict_format_by_default():
    out = _make([(10, 10, 20, 20, 0.9, 0), (0, 0, 5, 5, 0.1, 1)])
    dets = filter_e2e(out, conf=0.25)
    assert isinstance(dets, list)
    assert len(dets) == 1
    assert dets[0]["box"] == [10.0, 10.0, 20.0, 20.0]
    assert pytest.approx(dets[0]["score"], abs=1e-6) == 0.9
    assert dets[0]["class"] == 0
    assert dets[0]["label"] == "person"


def test_filter_returns_arrays_format():
    out = _make([(10, 10, 20, 20, 0.9, 0), (5, 5, 15, 15, 0.5, 2)])
    dets = filter_e2e(out, conf=0.25, format="arrays")
    assert dets["boxes"].shape == (2, 4)
    assert dets["boxes"].dtype == np.float32
    assert dets["scores"].shape == (2,)
    assert dets["classes"].dtype == np.int32


def test_filter_sorted_descending_by_score():
    out = _make([(0, 0, 1, 1, 0.3, 0), (0, 0, 1, 1, 0.9, 0), (0, 0, 1, 1, 0.5, 0)])
    dets = filter_e2e(out, conf=0.0, format="arrays")
    np.testing.assert_array_equal(dets["scores"], np.array([0.9, 0.5, 0.3], dtype=np.float32))


def test_filter_classes_allowlist():
    out = _make([(0, 0, 1, 1, 0.9, 0), (0, 0, 1, 1, 0.9, 5)])
    dets = filter_e2e(out, conf=0.25, classes=[5], format="arrays")
    assert dets["classes"].tolist() == [5]


def test_filter_min_area():
    out = _make([(0, 0, 10, 10, 0.9, 0), (0, 0, 2, 2, 0.9, 0)])
    dets = filter_e2e(out, conf=0.25, min_area=50.0, format="arrays")
    assert dets["boxes"].shape == (1, 4)


def test_filter_empty_when_all_below_conf():
    out = _make([(0, 0, 1, 1, 0.1, 0)])
    dets = filter_e2e(out, conf=0.5, format="arrays")
    assert dets["boxes"].shape == (0, 4)
    assert dets["scores"].shape == (0,)


def test_filter_squeeze_or_2d_input():
    rows = np.asarray([(10, 10, 20, 20, 0.9, 0)], dtype=np.float32)
    dets = filter_e2e(rows, conf=0.25, format="arrays")
    assert dets["boxes"].shape == (1, 4)


def test_filter_rejects_batch_gt_1():
    out = np.zeros((2, 3, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="batched"):
        filter_e2e(out, conf=0.25)


def test_filter_rejects_wrong_last_dim():
    out = np.zeros((1, 3, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="last dim"):
        filter_e2e(out, conf=0.25)


def test_filter_rejects_conf_out_of_range():
    out = _make([(0, 0, 1, 1, 0.9, 0)])
    with pytest.raises(ValueError, match="conf"):
        filter_e2e(out, conf=1.5)


def test_filter_stable_tiebreak_by_class_then_index():
    out = _make([
        (0, 0, 1, 1, 0.5, 3),  # idx 0
        (0, 0, 1, 1, 0.5, 1),  # idx 1
        (0, 0, 1, 1, 0.5, 1),  # idx 2 (same class as idx 1, later index)
    ])
    dets = filter_e2e(out, conf=0.0, format="arrays")
    # All same score: sort by class asc, then source index asc.
    assert dets["classes"].tolist() == [1, 1, 3]


def test_filter_strict_false_drops_nan_rows():
    out = _make([(0, 0, 1, 1, np.nan, 0), (0, 0, 2, 2, 0.9, 1)])
    dets = filter_e2e(out, conf=0.25, format="arrays", strict=False)
    assert dets["scores"].shape == (1,)


def test_filter_strict_true_errors_on_nan():
    out = _make([(0, 0, 1, 1, np.nan, 0)])
    with pytest.raises(ValueError, match="NaN"):
        filter_e2e(out, conf=0.25, strict=True)


def test_filter_drops_out_of_range_class_id():
    # Without strict, a row with cls=200 (out of COCO range) is dropped, not crashed.
    out = _make([(0, 0, 1, 1, 0.9, 200), (0, 0, 1, 1, 0.9, 5)])
    dets = filter_e2e(out, conf=0.25, format="dict")
    assert isinstance(dets, list)
    assert len(dets) == 1
    assert dets[0]["class"] == 5


def test_filter_strict_true_errors_on_out_of_range_class_id():
    out = _make([(0, 0, 1, 1, 0.9, 200)])
    with pytest.raises(ValueError, match="class_id out of range"):
        filter_e2e(out, conf=0.25, strict=True)
