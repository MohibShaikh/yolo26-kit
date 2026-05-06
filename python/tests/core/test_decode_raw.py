import numpy as np
import pytest

from yolo26_kit.core.decode_raw import decode_detect


def _make_v8(boxes_cxcywh, scores_per_row, n=8400, nc=80):
    """Build (1, 4+nc, N) tensor with sigmoid-already-applied scores."""
    out = np.zeros((1, 4 + nc, n), dtype=np.float32)
    for i, (b, scores) in enumerate(zip(boxes_cxcywh, scores_per_row)):
        out[0, 0:4, i] = b
        out[0, 4:, i] = scores
    return out


def test_decode_picks_argmax_class_and_score():
    scores = np.zeros(80, dtype=np.float32); scores[5] = 0.9; scores[10] = 0.4
    out = _make_v8([(20, 30, 20, 20)], [scores])
    dets = decode_detect(out, conf=0.25, format="arrays")
    assert dets["classes"].tolist() == [5]
    np.testing.assert_allclose(dets["scores"], [0.9], atol=1e-6)
    np.testing.assert_allclose(dets["boxes"], [[10, 20, 30, 40]], atol=1e-6)


def test_decode_filters_below_conf():
    scores = np.zeros(80, dtype=np.float32); scores[5] = 0.1
    out = _make_v8([(20, 30, 20, 20)], [scores])
    dets = decode_detect(out, conf=0.25, format="arrays")
    assert dets["scores"].shape == (0,)


def test_decode_accepts_transposed_input():
    scores = np.zeros(80, dtype=np.float32); scores[5] = 0.9
    out_v8 = _make_v8([(20, 30, 20, 20)], [scores])
    out_t = np.transpose(out_v8, (0, 2, 1))  # (1, N, 4+nc)
    dets = decode_detect(out_t, conf=0.25, format="arrays")
    assert dets["classes"].tolist() == [5]


def test_decode_assume_sigmoid_false_applies_sigmoid():
    # Logit 2.197 -> sigmoid ≈ 0.9
    logits = np.full(80, -10.0, dtype=np.float32); logits[5] = 2.197
    out = _make_v8([(20, 30, 20, 20)], [logits])
    dets = decode_detect(out, conf=0.25, assume_sigmoid=False, format="arrays")
    assert dets["classes"].tolist() == [5]
    assert pytest.approx(dets["scores"][0], abs=1e-3) == 0.9


def test_decode_dict_format_has_label():
    scores = np.zeros(80, dtype=np.float32); scores[0] = 0.95
    out = _make_v8([(20, 30, 20, 20)], [scores])
    dets = decode_detect(out, conf=0.25)
    assert dets[0]["label"] == "person"


def test_decode_rejects_short_class_axis():
    out = np.zeros((1, 4, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="≥5"):
        decode_detect(out, conf=0.25)
