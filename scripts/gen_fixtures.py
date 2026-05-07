"""Generate golden fixtures using ultralytics. Run once, check results in.

Usage:
    cd /home/tsd/open-source-contrib/yolo26
    python/.venv/bin/python scripts/gen_fixtures.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "fixtures" / "v1"

CASES = [
    ("coco_bus", "https://ultralytics.com/images/bus.jpg"),
    ("coco_zidane", "https://ultralytics.com/images/zidane.jpg"),
]


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        urllib.request.urlretrieve(url, dst)  # noqa: S310


def _ultralytics_version() -> str:
    import ultralytics
    return ultralytics.__version__


def _decode_v8_raw(out: np.ndarray, conf: float = 0.25) -> list[dict[str, Any]]:
    """Decode (1, 4+nc, N) raw tensor using the library's own decoder for parity."""
    sys.path.insert(0, str(ROOT / "python" / "src"))
    from yolo26_kit.core.decode_raw import decode_detect
    result = decode_detect(out, conf=conf, num_classes=80)
    assert isinstance(result, list)
    return result


def _decode_e2e(out: np.ndarray, conf: float = 0.25) -> list[dict[str, Any]]:
    """Decode (1, K, 6) e2e tensor using the library's filter_e2e for parity."""
    sys.path.insert(0, str(ROOT / "python" / "src"))
    from yolo26_kit.core.filter_e2e import filter_e2e
    result = filter_e2e(out, conf=conf)
    assert isinstance(result, list)
    return result


def _generate(name: str, img_path: Path, mode: str) -> None:
    from PIL import Image
    import onnxruntime as ort
    from ultralytics import YOLO

    out_dir = FIXTURES_DIR / f"{name}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "input.jpg").write_bytes(img_path.read_bytes())

    # Try to load YOLO26n. If not found, fall back to a version that exists.
    weight_candidates = ["yolo26n.pt", "yolov26n.pt", "yolo11n.pt", "yolov8n.pt"]
    model = None
    weights_used = None
    for w in weight_candidates:
        try:
            model = YOLO(w)
            weights_used = w
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError(f"could not load any YOLO weights from {weight_candidates}")

    print(f"  using weights: {weights_used}")

    onnx_path = Path(model.export(format="onnx", end2end=(mode == "e2e"), imgsz=640, opset=12))

    # Letterbox preprocess identical to ours
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    scale = min(640 / W, 640 / H)
    new_w, new_h = int(round(W * scale)), int(round(H * scale))
    img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (640, 640), (114, 114, 114))
    pad_x = (640 - new_w) // 2
    pad_y = (640 - new_h) // 2
    canvas.paste(img_resized, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output = sess.run(None, {sess.get_inputs()[0].name: arr})[0]

    np.save(out_dir / "raw_output.npy", output)
    (out_dir / "raw_output_shape.json").write_text(
        json.dumps({"shape": list(output.shape), "dtype": str(output.dtype)})
    )

    # Decode expected detections. Coordinates remain in 640 letterbox space.
    if mode == "e2e":
        expected = _decode_e2e(output, conf=0.25)
    else:
        expected = _decode_v8_raw(output, conf=0.25)

    (out_dir / "expected.json").write_text(json.dumps(expected, indent=2))

    meta = {
        "orig_size": [W, H],
        "lb_size": [640, 640],
        "scale": scale,
        "pad": [pad_x, pad_y],
        "model_name": weights_used,
        "export_mode": mode,
        "ultralytics_version": _ultralytics_version(),
        "api": "filter_e2e" if mode == "e2e" else "decode_detect",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {out_dir} ({len(expected)} detections)")


def main() -> int:
    img_dir = ROOT / ".cache" / "fixtures-src"
    img_dir.mkdir(parents=True, exist_ok=True)
    for name, url in CASES:
        img_path = img_dir / f"{name}.jpg"
        _download(url, img_path)
        for mode in ("e2e", "raw"):
            print(f"generating {name}_{mode}...")
            try:
                _generate(name, img_path, mode)
            except Exception as e:
                print(f"  FAILED: {e}")
                raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
