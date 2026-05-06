import numpy as np
import pytest

from yolo26_kit.core.decode_raw import decode_detect
from yolo26_kit.core.filter_e2e import filter_e2e


@pytest.mark.benchmark(group="filter_e2e")
def test_filter_e2e_300(benchmark):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    out = rng.uniform(0, 1, size=(1, 300, 6)).astype(np.float32)
    out[0, :, 5] = rng.integers(0, 80, size=300)
    benchmark(filter_e2e, out, 0.25, None, None, "arrays")


@pytest.mark.benchmark(group="decode_raw")
def test_decode_raw_8400(benchmark):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    out = rng.uniform(0, 1, size=(1, 84, 8400)).astype(np.float32)
    benchmark(decode_detect, out, 0.25, None, None, "arrays")
