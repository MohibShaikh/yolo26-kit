"""Run every fixture in fixtures/v1/ and assert decoder matches expected.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from yolo26_kit.core.decode_raw import decode_detect
from yolo26_kit.core.filter_e2e import filter_e2e

# Repo root is 3 levels up from this test file: python/tests/fixtures/test_fixtures.py
ROOT = Path(__file__).resolve().parents[3]
FIXTURES_BASE = ROOT / "fixtures" / "v1"
FIXTURES = sorted(FIXTURES_BASE.iterdir()) if FIXTURES_BASE.exists() else []


@pytest.mark.parametrize("fixture_dir", FIXTURES, ids=lambda p: p.name)
def test_fixture(fixture_dir: Path) -> None:
    raw = np.load(fixture_dir / "raw_output.npy")
    expected = json.loads((fixture_dir / "expected.json").read_text())
    meta = json.loads((fixture_dir / "meta.json").read_text())

    if meta["api"] == "filter_e2e":
        actual = filter_e2e(raw, conf=0.25)
    elif meta["api"] == "decode_detect":
        actual = decode_detect(raw, conf=0.25, num_classes=80)
    else:
        pytest.fail(f"unknown api: {meta['api']}")

    assert isinstance(actual, list)
    assert len(actual) == len(expected), (
        f"{fixture_dir.name}: detection count mismatch"
        f" — actual {len(actual)} vs expected {len(expected)}"
    )

    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        assert a["class"] == e["class"], (
            f"{fixture_dir.name}[{i}]: class mismatch — actual={a} expected={e}"
        )
        assert abs(a["score"] - e["score"]) <= 1e-4, (
            f"{fixture_dir.name}[{i}]: score — actual={a['score']} expected={e['score']}"
        )
        for ci in range(4):
            assert abs(a["box"][ci] - e["box"][ci]) <= 1.0, (
                f"{fixture_dir.name}[{i}].box[{ci}] — actual={a['box'][ci]} expected={e['box'][ci]}"
            )
