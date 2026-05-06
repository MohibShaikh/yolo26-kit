import numpy as np

from yolo26_kit.core.letterbox import letterbox_unmap


def test_unmap_round_trip_no_pad():
    boxes = np.asarray([[100, 100, 200, 200]], dtype=np.float32)  # in 640 space, scale 1, no pad
    out = letterbox_unmap(boxes, orig_size=(640, 640), lb_size=(640, 640), scale=1.0, pad=(0, 0))
    np.testing.assert_allclose(out, boxes)


def test_unmap_with_pad_and_scale():
    # original 1280x720 → letterbox 640 means scale=0.5, pad_y=(640-360)/2=140
    boxes = np.asarray([[100, 240, 200, 380]], dtype=np.float32)
    out = letterbox_unmap(boxes, orig_size=(1280, 720), lb_size=(640, 640),
                          scale=0.5, pad=(0, 140))
    expected = np.asarray([[200, 200, 400, 480]], dtype=np.float32)
    np.testing.assert_allclose(out, expected)


def test_unmap_clips_to_image():
    boxes = np.asarray([[-10, -5, 1290, 730]], dtype=np.float32)
    out = letterbox_unmap(boxes, orig_size=(1280, 720), lb_size=(1280, 720),
                          scale=1.0, pad=(0, 0))
    np.testing.assert_allclose(out, [[0, 0, 1280, 720]])
