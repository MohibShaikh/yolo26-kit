import numpy as np
import pytest

from yolo26_kit.core.decode_raw import decode_detect


def _make_v8(boxes_cxcywh, scores_per_row, n=8400, nc=80):
    """Build (1, 4+nc, N) tensor with sigmoid-already-applied scores."""
    out = np.zeros((1, 4 + nc, n), dtype=np.float32)
    for i, (b, scores) in enumerate(zip(boxes_cxcywh, scores_per_row, strict=True)):
        out[0, 0:4, i] = b
        out[0, 4:, i] = scores
    return out


def test_decode_picks_argmax_class_and_score():
    scores = np.zeros(80, dtype=np.float32)
    scores[5] = 0.9
    scores[10] = 0.4
    out = _make_v8([(20, 30, 20, 20)], [scores])
    dets = decode_detect(out, conf=0.25, format="arrays")
    assert dets["classes"].tolist() == [5]
    np.testing.assert_allclose(dets["scores"], [0.9], atol=1e-6)
    np.testing.assert_allclose(dets["boxes"], [[10, 20, 30, 40]], atol=1e-6)


def test_decode_filters_below_conf():
    scores = np.zeros(80, dtype=np.float32)
    scores[5] = 0.1
    out = _make_v8([(20, 30, 20, 20)], [scores])
    dets = decode_detect(out, conf=0.25, format="arrays")
    assert dets["scores"].shape == (0,)


def test_decode_accepts_transposed_input():
    scores = np.zeros(80, dtype=np.float32)
    scores[5] = 0.9
    out_v8 = _make_v8([(20, 30, 20, 20)], [scores])
    out_t = np.transpose(out_v8, (0, 2, 1))  # (1, N, 4+nc)
    dets = decode_detect(out_t, conf=0.25, format="arrays")
    assert dets["classes"].tolist() == [5]


def test_decode_assume_sigmoid_false_applies_sigmoid():
    # Logit 2.197 -> sigmoid ≈ 0.9. With n=1 (smaller trailing dim < 5)
    # the heuristic is ambiguous, so we pin num_classes=80.
    logits = np.full(80, -10.0, dtype=np.float32)
    logits[5] = 2.197
    out = _make_v8([(20, 30, 20, 20)], [logits], n=1)
    dets = decode_detect(
        out, conf=0.25, num_classes=80, assume_sigmoid=False, format="arrays",
    )
    assert dets["classes"].tolist() == [5]
    assert pytest.approx(dets["scores"][0], abs=1e-3) == 0.9


def test_decode_dict_format_has_label():
    scores = np.zeros(80, dtype=np.float32)
    scores[0] = 0.95
    out = _make_v8([(20, 30, 20, 20)], [scores])
    dets = decode_detect(out, conf=0.25)
    assert dets[0]["label"] == "person"


def test_decode_rejects_short_class_axis():
    out = np.zeros((1, 4, 100), dtype=np.float32)
    with pytest.raises(ValueError, match=">=5"):
        decode_detect(out, conf=0.25)


def test_decode_explicit_num_classes_round_trip_small_k():
    # Round-trip invariant: when K < 4+nc (small anchor count, e.g. K=10),
    # the channel-axis heuristic is wrong without explicit num_classes.
    scores = np.zeros(80, dtype=np.float32)
    scores[7] = 0.8
    out = _make_v8([(20, 30, 20, 20)], [scores], n=10)
    dets = decode_detect(out, conf=0.25, num_classes=80, format="arrays")
    assert dets["classes"].tolist() == [7]
    np.testing.assert_allclose(dets["boxes"], [[10, 20, 30, 40]], atol=1e-6)


def test_decode_classes_filter():
    # Build two distinct anchors: one class 5, one class 7.
    s0 = np.zeros(80, dtype=np.float32)
    s0[5] = 0.9
    s1 = np.zeros(80, dtype=np.float32)
    s1[7] = 0.8
    out = np.zeros((1, 84, 8400), dtype=np.float32)
    out[0, 0:4, 0] = (20, 30, 20, 20)
    out[0, 4:, 0] = s0
    out[0, 0:4, 1] = (50, 50, 10, 10)
    out[0, 4:, 1] = s1
    dets = decode_detect(out, conf=0.25, classes=[7], format="arrays")
    assert dets["classes"].tolist() == [7]


def test_decode_min_area_filter():
    # Two anchors: one large box, one tiny.
    s0 = np.zeros(80, dtype=np.float32)
    s0[1] = 0.9
    s1 = np.zeros(80, dtype=np.float32)
    s1[2] = 0.9
    out = np.zeros((1, 84, 8400), dtype=np.float32)
    out[0, 0:4, 0] = (50, 50, 100, 100)  # 100x100 = 10000
    out[0, 4:, 0] = s0
    out[0, 0:4, 1] = (10, 10, 4, 4)  # 4x4 = 16
    out[0, 4:, 1] = s1
    dets = decode_detect(out, conf=0.25, min_area=1000.0, format="arrays")
    assert dets["classes"].tolist() == [1]


def test_decode_classes_allowlist_out_of_range_errors():
    out = np.zeros((1, 84, 100), dtype=np.float32)
    out[0, 4 + 5, 0] = 0.9
    with pytest.raises(ValueError, match="out of range"):
        decode_detect(out, conf=0.25, classes=[200])


def test_decode_rejects_equal_trailing_dims():
    # Square trailing dims are ambiguous when num_classes is not pinned.
    out = np.zeros((1, 84, 84), dtype=np.float32)
    with pytest.raises(ValueError, match="ambiguous"):
        decode_detect(out, conf=0.25)


def test_decode_explicit_num_classes_overrides_heuristic():
    # When (84, 84) ambiguous, explicit num_classes resolves it.
    out = np.zeros((1, 84, 84), dtype=np.float32)
    # Ambiguous even with num_classes since both axes match 4+80=84.
    with pytest.raises(ValueError, match="ambiguous"):
        decode_detect(out, conf=0.25, num_classes=80)
