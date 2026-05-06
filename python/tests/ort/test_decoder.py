from __future__ import annotations

import numpy as np

from yolo26_kit.ort.decoder import Decoder


class _MockSession:
    """Stub matching the bits of ort.InferenceSession we touch."""

    def __init__(self, output_shape: tuple[int, ...], producer):
        self._shape = output_shape
        self._producer = producer

    def get_inputs(self):
        class _Input:
            name = "images"
            shape = (1, 3, 640, 640)
            type = "tensor(float)"
        return [_Input()]

    def get_outputs(self):
        s = self._shape

        class _Output:
            name = "output0"
            shape = s
        return [_Output()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _names, _inputs):
        return [self._producer(_inputs)]


def _e2e_producer(_inputs):
    out = np.zeros((1, 300, 6), dtype=np.float32)
    out[0, 0] = (10, 10, 30, 30, 0.9, 5)
    return out


def _raw_producer(_inputs):
    out = np.zeros((1, 84, 8400), dtype=np.float32)
    out[0, 0:4, 0] = (20, 20, 20, 20)
    out[0, 4 + 5, 0] = 0.9
    return out


def test_decoder_routes_e2e_path():
    sess = _MockSession((1, 300, 6), _e2e_producer)
    dec = Decoder(sess)
    assert dec.is_e2e is True


def test_decoder_routes_non_e2e_path():
    sess = _MockSession((1, 84, 8400), _raw_producer)
    dec = Decoder(sess)
    assert dec.is_e2e is False


def test_decoder_predict_with_synthetic_image_e2e():
    sess = _MockSession((1, 300, 6), _e2e_producer)
    dec = Decoder(sess)
    img = np.full((640, 640, 3), 200, dtype=np.uint8)
    dets = dec.predict(img, conf=0.25, format="arrays")
    np.testing.assert_allclose(dets["boxes"][0], [10, 10, 30, 30], atol=1e-4)
    assert dets["classes"][0] == 5


def test_decoder_predict_unmaps_to_original_coords():
    sess = _MockSession((1, 300, 6), _e2e_producer)
    dec = Decoder(sess)
    img = np.full((1280, 1280, 3), 200, dtype=np.uint8)
    dets = dec.predict(img, conf=0.25, format="arrays")
    np.testing.assert_allclose(dets["boxes"][0], [20, 20, 60, 60], atol=1e-3)
