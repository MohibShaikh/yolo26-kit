# yolo26-kit

> Drop YOLO26 into your existing YOLOv8 pipeline. Bridge library for the NMS-free, end-to-end era.

[![CI](https://github.com/MohibShaikh/yolo26-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/MohibShaikh/yolo26-kit/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## Why

YOLO26 default ONNX export is `end2end=True`, producing `(N, 300, 6)` already-decoded detections. That's great — but it breaks every Triton / DeepStream / OpenCV DNN config that expects YOLOv8-style `(1, 4+nc, N)`. It also leaves you to write letterbox-coord undo math by hand. And some targets (RKNN, certain TFLite quant paths) can't fuse the e2e head, so they ship raw `(1, 4+nc, N)` you have to decode yourself.

`yolo26-kit` is a small, pure-functional library that bridges that gap — and nothing else.

## Install

```bash
# Python — https://pypi.org/project/yolo26-kit/
pip install yolo26-kit            # core
pip install "yolo26-kit[ort]"     # + ORT wrapper

# TypeScript / JavaScript — https://www.npmjs.com/package/yolo26-kit
npm i yolo26-kit                 # core
npm i yolo26-kit onnxruntime-web # + ORT wrapper (peer dep)
```

## Export YOLO26 to ONNX

The "happy path" is to export with `end2end=True` so the ONNX graph emits already-deduplicated `(N, 300, 6)` detections:

```python
from ultralytics import YOLO

YOLO("yolo26n.pt").export(format="onnx", end2end=True)
# produces yolo26n.onnx with output shape (1, 300, 6)
```

If you cannot use `end2end=True` (e.g., quantized TFLite or some NPU runtimes that don't support the e2e head), `yolo26-kit` auto-detects and runs class-aware NMS for you when decoding raw `(1, 4+nc, N)` outputs.

## Use

### Python — e2e default (most users)

```python
import onnxruntime as ort
import yolo26_kit

sess = ort.InferenceSession("yolo26n.onnx")
out = sess.run(None, {"images": pre})[0]   # (1, 300, 6)
out = yolo26_kit.normalize_output(out)
dets = yolo26_kit.filter_e2e(out, conf=0.25)
# [{'box': [...], 'score': ..., 'class': ..., 'label': '...'}]
```

### Python — drop into existing YOLOv8 pipeline

```python
out_v8 = yolo26_kit.e2e_to_v8_shape(out)   # (1, 84, 300)
# Feed out_v8 to your existing v8 Triton / DeepStream / OpenCV DNN parser.
```

### Python — convenience wrapper (image in, dets in original coords out)

```python
from yolo26_kit import from_ort

decoder = from_ort("yolo26n.onnx")
dets = decoder.predict("bus.jpg", conf=0.25)
```

### TypeScript

```ts
import { filterE2E } from "yolo26-kit";
import { fromOrt } from "yolo26-kit/ort";
import * as ort from "onnxruntime-web";

const session = await ort.InferenceSession.create("yolo26n.onnx");
const decoder = fromOrt(session);
const dets = await decoder.predict(canvas, { conf: 0.25 });
```

> Single image at a time (batch=1). Loop in your code if you have multiple frames.

### TypeScript — non-e2e raw export

```ts
import { decodeDetect, normalizeOutput } from "yolo26-kit";

const out = normalizeOutput(rawTensor);
const dets = decodeDetect(out, [1, 84, 8400], { conf: 0.25, numClasses: 80 });
```

### Decoder auto-routing

The `Decoder` auto-detects whether the ONNX is e2e (output shape ends in 6) or raw (output shape `(1, 4+nc, N)`):

- **e2e:** runs `filter_e2e` — already deduped by the model, no NMS.
- **raw:** runs `decode_detect` with class-aware NMS (default `iou_threshold=0.45`).

Override the NMS default if you want raw outputs (e.g., for further post-processing):

```python
decoder.predict("bus.jpg", conf=0.25, nms=False)
```

```ts
await decoder.predict(canvas, { conf: 0.25, nms: false });
```

## What's in the box (v0.1.x)

| Function | Purpose |
|---|---|
| `filter_e2e` / `filterE2E` | Filter the default `(N, 300, 6)` e2e output by confidence + classes + min-area |
| `decode_detect` / `decodeDetect` | Decode raw `(1, 4+nc, N)` non-e2e exports (NPU / quantized targets) |
| `e2e_to_v8_shape` / `e2eToV8Shape` | Adapter to drop YOLO26 into legacy YOLOv8 pipelines |
| `v8_shape_to_e2e` / `v8ShapeToE2E` | Reverse adapter |
| `letterbox_unmap` / `letterboxUnmap` | Undo letterbox padding back to original image coords |
| `normalize_output` / `normalizeOutput` | Dtype workaround for upstream YOLO26 fp16 export issue (#23645) |
| `class_aware_nms` / `classAwareNMS` | Class-aware non-maximum suppression for raw decoder output |
| `Decoder.predict()` | Image in → detections in original-image coords out (auto e2e/non-e2e routing) |

## Compatibility

| `yolo26-kit` | `ultralytics` (verified) | `onnxruntime` |
|---|---|---|
| 0.1.0 | 8.4.x (yolo26n.pt) | 1.17+ |

A nightly live-diff workflow installs the latest `ultralytics` and re-validates fixtures. Drift opens an auto-tagged `drift` issue.

## Status

**v1 (this release):** Detect task only. e2e + non-e2e paths. Both Python and TypeScript, cross-language equivalent (verified via golden fixtures).

**Roadmap:** Seg, pose, cls, OBB tasks. Reproduction kit (Docker + COCO train/eval). ONNX schema standardizer. NPU ports (RKNN, Hailo, RDK, Qualcomm SNPE/QNN).

## Development

```bash
# Python
cd python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
mypy src/yolo26_kit
ruff check src tests

# TypeScript
cd js
npm install
npm test
npm run typecheck
npm run lint
```

## Limitations (v0.1.x)

- **Single image only:** decoders reject `batch > 1`. Loop in user code for multi-image inference. Batched decode tracked for v0.2.
- **Detect task only:** seg/pose/cls/OBB queued.
- **Browser preprocess uses nearest-neighbor resize** for portability. For pixel-perfect parity with PIL bilinear, pre-resize via canvas.

## License

Apache-2.0 for the code in this repository. Model weights are not redistributed; users fetch them from the official Ultralytics distribution under its own terms (AGPL-3.0).

See [`fixtures/LICENSES.md`](fixtures/LICENSES.md) for fixture asset attribution.
