"""Nightly: install latest ultralytics, regenerate fixtures, diff vs checked-in expected.json."""
from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DIFF_FILE = ROOT / "live_diff_report.txt"


def main() -> int:
    sys.path.insert(0, str(ROOT))
    gen = importlib.import_module("scripts.gen_fixtures")
    with tempfile.TemporaryDirectory() as tmp:
        gen.FIXTURES_DIR = Path(tmp) / "v1"
        gen.main()
        fresh = Path(tmp) / "v1"
        baseline = ROOT / "fixtures" / "v1"
        report: list[str] = []
        for sub in sorted(fresh.iterdir()):
            base = baseline / sub.name
            if not base.exists():
                report.append(f"NEW   {sub.name}")
                continue
            old_raw = np.load(base / "raw_output.npy")
            new_raw = np.load(sub / "raw_output.npy")
            if old_raw.shape != new_raw.shape:
                report.append(f"DRIFT {sub.name}: shape {old_raw.shape} -> {new_raw.shape}")
                continue
            diff = float(np.max(np.abs(old_raw - new_raw)))
            if diff > 1e-3:
                report.append(f"DRIFT {sub.name}: max-abs-diff = {diff:.6f}")
        DIFF_FILE.write_text("\n".join(report) + "\n" if report else "OK\n")
        return 1 if report else 0


if __name__ == "__main__":
    sys.exit(main())
