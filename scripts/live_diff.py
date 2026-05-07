"""Nightly: regenerate fixtures, diff vs checked-in DECODED detections (not raw tensors)."""
from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DIFF_FILE = ROOT / "live_diff_report.txt"


def _load_dets(d: Path) -> list[dict]:
    return json.loads((d / "expected.json").read_text())


def _diff_dets(old: list[dict], new: list[dict]) -> str | None:
    if len(old) != len(new):
        return f"detection count: {len(old)} -> {len(new)}"
    for i, (a, b) in enumerate(zip(old, new, strict=True)):
        if a["class"] != b["class"]:
            return f"det[{i}] class: {a['class']} -> {b['class']}"
        if abs(a["score"] - b["score"]) > 1e-3:
            return f"det[{i}] score: {a['score']:.4f} -> {b['score']:.4f}"
        for ci in range(4):
            if abs(a["box"][ci] - b["box"][ci]) > 1.0:
                return f"det[{i}] box[{ci}]: {a['box'][ci]:.2f} -> {b['box'][ci]:.2f}"
    return None


def main() -> int:
    sys.path.insert(0, str(ROOT))
    gen = importlib.import_module("scripts.gen_fixtures")
    with tempfile.TemporaryDirectory() as tmp:
        gen.FIXTURES_DIR = Path(tmp) / "v1"  # type: ignore[attr-defined]
        gen.main()
        fresh = Path(tmp) / "v1"
        baseline = ROOT / "fixtures" / "v1"
        report: list[str] = []
        for sub in sorted(fresh.iterdir()):
            base = baseline / sub.name
            if not base.exists():
                report.append(f"NEW   {sub.name}")
                continue
            # Shape sanity check on raw_output.npy first
            old_raw = np.load(base / "raw_output.npy")
            new_raw = np.load(sub / "raw_output.npy")
            if old_raw.shape != new_raw.shape:
                report.append(f"DRIFT {sub.name}: shape {old_raw.shape} -> {new_raw.shape}")
                continue
            # Compare DECODED detections (set-stable across ORT/numpy versions)
            old_dets = _load_dets(base)
            new_dets = _load_dets(sub)
            mismatch = _diff_dets(old_dets, new_dets)
            if mismatch:
                report.append(f"DRIFT {sub.name}: {mismatch}")
        DIFF_FILE.write_text("\n".join(report) + "\n" if report else "OK\n")
        return 1 if report else 0


if __name__ == "__main__":
    sys.exit(main())
