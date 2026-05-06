import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from yolo26_kit.core.decode_raw import decode_detect


@settings(max_examples=50)
@given(
    n=st.integers(1, 200),
    nc=st.integers(1, 80),
    conf=st.floats(0, 1, allow_nan=False),
)
def test_decode_count_equals_max_above_conf(n, nc, conf):
    assume(n != 4 + nc)  # equal trailing dims rejected by _split_channel_anchor_axes
    rng = np.random.default_rng(42)
    out = rng.uniform(0, 1, size=(1, 4 + nc, n)).astype(np.float32)
    dets = decode_detect(out, conf=conf, num_classes=nc, format="arrays", nms=False)
    max_per_anchor = out[0, 4:, :].max(axis=0)
    expected = int((max_per_anchor >= conf).sum())
    assert dets["scores"].shape[0] == expected
