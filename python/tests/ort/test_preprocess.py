import numpy as np

from yolo26_kit.ort.preprocess import letterbox_forward


def test_letterbox_square_input_no_pad():
    img = np.full((640, 640, 3), 100, dtype=np.uint8)
    arr, meta = letterbox_forward(img, target=640)
    assert arr.shape == (1, 3, 640, 640)
    assert arr.dtype == np.float32
    assert meta["scale"] == 1.0
    assert meta["pad"] == (0, 0)
    assert meta["orig_size"] == (640, 640)
    np.testing.assert_allclose(arr.max(), 100 / 255.0, atol=1e-6)


def test_letterbox_landscape_pad_y():
    img = np.full((360, 640, 3), 50, dtype=np.uint8)  # 16:9
    arr, meta = letterbox_forward(img, target=640)
    assert arr.shape == (1, 3, 640, 640)
    assert meta["scale"] == 1.0
    assert meta["pad"] == (0, 140)
    np.testing.assert_allclose(arr[0, 0, 0, 0], 114 / 255.0, atol=1e-6)


def test_letterbox_portrait_pad_x():
    img = np.full((640, 360, 3), 50, dtype=np.uint8)
    _arr, meta = letterbox_forward(img, target=640)
    assert meta["pad"] == (140, 0)
