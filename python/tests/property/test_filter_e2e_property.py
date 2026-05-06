import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from yolo26_kit.core.filter_e2e import filter_e2e


@st.composite
def e2e_outputs(draw):
    n = draw(st.integers(min_value=0, max_value=300))
    rows = []
    for _ in range(n):
        x1 = draw(st.floats(0, 1000, allow_nan=False))
        y1 = draw(st.floats(0, 1000, allow_nan=False))
        x2 = x1 + draw(st.floats(1, 200, allow_nan=False))
        y2 = y1 + draw(st.floats(1, 200, allow_nan=False))
        conf = draw(st.floats(0, 1, allow_nan=False))
        cls = draw(st.integers(0, 79))
        rows.append((x1, y1, x2, y2, conf, cls))
    if not rows:
        return np.zeros((1, 0, 6), dtype=np.float32)
    return np.asarray([rows], dtype=np.float32)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(e2e_outputs(), st.floats(0, 1, allow_nan=False))
def test_filter_count_equals_threshold_count(out, conf):
    dets = filter_e2e(out, conf=conf, format="arrays")
    expected = int((out[0, :, 4] >= conf).sum())
    assert dets["scores"].shape[0] == expected


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(e2e_outputs())
def test_filter_output_sorted_descending(out):
    dets = filter_e2e(out, conf=0.0, format="arrays")
    s = dets["scores"]
    assert np.all(s[:-1] >= s[1:]) if s.size > 1 else True


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(e2e_outputs())
def test_filter_boxes_well_formed(out):
    dets = filter_e2e(out, conf=0.0, format="arrays")
    b = dets["boxes"]
    if b.size:
        assert (b[:, 2] >= b[:, 0]).all()
        assert (b[:, 3] >= b[:, 1]).all()


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(e2e_outputs())
def test_filter_classes_in_range(out):
    dets = filter_e2e(out, conf=0.0, format="arrays")
    c = dets["classes"]
    if c.size:
        assert (c >= 0).all()
        assert (c < 80).all()
