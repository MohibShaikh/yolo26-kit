# YOLO26 Pipeline Bridge (`yolo26-kit`) — Design

**Date:** 2026-05-05 (revised same-day after upstream-default discovery)
**Status:** Design approved (brainstorm complete, awaiting user spec review)
**License:** Apache-2.0

## Problem

Ultralytics YOLO26 (Jan 2026) ships an end-to-end NMS-free architecture. By **default**, ONNX export sets `end2end=True`, producing already-decoded output of shape `(N, 300, 6)` with columns `[x1, y1, x2, y2, conf, class_id]`. Filtering the raw tensor by confidence is a one-liner — there is no NMS to run.

That sounds easy. In practice it breaks every existing deployment:

- **Triton / DeepStream / OpenCV DNN** post-processing graphs and parser configs were written for YOLOv8-shape `(1, 4+nc, N)` raw outputs. They cannot consume `(N, 300, 6)` decoded boxes.
- **Letterbox coordinates remain in 640-padded space** regardless of `end2end`. Every pipeline still needs `unmap → original image coords`. Most users hand-roll this and get the math subtly wrong.
- **Some targets cannot fuse the e2e head.** Quantized TFLite, certain NPU runtimes (RKNN, some SNPE paths), and stripped ONNX exports produce non-e2e raw output `(1, 4+nc, N)` that must be decoded explicitly.
- **Known upstream bugs** force users to add wrapper logic anyway — e.g. `#23645` (e2e+fp16 export emits fp32 output0), `#23339` (dynamic batch + NMS export gaps), `#23685` (e2e seg duplicate detections — closed but recurring).
- **Browser / Node TS deployments** have nothing equivalent to the Python ergonomics ultralytics ships internally.

Confirmed via Ultralytics issues `#23978`, `#23850`, `#23685`, `#24265`, `#23645`, `#23339`, `#23756`, `#24267`.

## Solution — `yolo26-kit`: a YOLO26 ↔ YOLOv8-pipeline bridge

A small, focused, cross-language toolbox. Pitch: *"Drop YOLO26 into your existing YOLOv8 pipeline. Or onto a target that can't fuse the e2e head. Either way, this 200-LOC library handles the glue."*

Initial scope: **detect task only**. Seg / pose / cls / OBB queued as follow-up modules in the same repo.

## What it ships

| Module | Purpose | Why |
|---|---|---|
| **Shape adapter** | `e2e_to_v8_shape()` and `v8_shape_to_e2e()` | Bidirectional layout translation between `(N,300,6)` and `(1, 4+nc, N)` so YOLO26 ONNX drops into pre-existing v8 pipelines (Triton, DeepStream, OpenCV DNN) without rewriting parser configs. |
| **e2e filter helper** | `filter_e2e(output, conf, classes=None, min_area=None)` | One-call confidence + class-allowlist + min-area filter on the default e2e output. Avoids users re-deriving the `(N,300,6)` slice math. |
| **Letterbox unmap** | `letterbox_unmap(boxes, orig_size, lb_size, scale, pad)` | Converts 640-padded box coords back to original image coords. Always needed. Math nobody enjoys writing. |
| **Decoder for non-e2e exports** | `decode_detect(output, conf)` | Handles `(1, 4+nc, N)` raw output. Used when e2e is unsupported on target (NPU, quantized TFLite, stripped exports). |
| **Dtype normalizer** | `normalize_output(output)` | Works around upstream issues like `#23645` (fp16 export emits fp32). Casts and validates. |
| **ORT wrappers** | `from_ort()` / `fromOrt()` | Image-in, detections-in-original-coords-out. Auto-detects e2e vs non-e2e, picks correct path. |

## Goals

1. Drop-in unblocking of YOLO26 deployments on existing YOLOv8-shape post-processing pipelines.
2. Cross-language equivalence (Python + TypeScript) verified by golden fixtures.
3. Pure functional core, zero side effects, zero hard deps beyond numpy / typed arrays.
4. Single source of algorithmic truth in `spec/decode.md` so both impls stay in lockstep.
5. Ship-fast: weekend-scale v1.

## Non-Goals (v1)

- Seg, pose, cls, OBB tasks.
- Training, fine-tuning, model conversion.
- Re-implementing or wrapping ultralytics's `Results` class.
- C++ / Rust / Go bindings.
- Batched (batch>1) decoding (planned for v1.1).

## Decisions Captured

| Q | Decision |
|---|---|
| Languages | Python + TypeScript, parallel hand-written impls. |
| Tasks v1 | Detect only. |
| Input contract | Raw tensor in **and** convenience ORT wrapper. |
| Output contract | Both `list[dict]` and `dict[str, ndarray]` via `format=` flag (Py); `Detection[]` and `{boxes,scores,classes}` via flag (TS). |
| Legacy compat | Bidirectional shape adapter `(N,300,6) ↔ (1,4+nc,N)`. |
| Repo | Single monorepo `yolo26-kit`, bridge is first module. |
| Verification | Golden fixtures (CI) + nightly live-diff against latest ultralytics. |
| Deps posture | Zero-dep core; pluggable extras for ORT / Pillow / cv2. |
| Implementation | Approach 1 — parallel Py + TS, shared spec doc. |
| Framing (revised) | "Pipeline bridge" — adapter primary; decoder is optional path for non-e2e exports. |

## Architecture

```
yolo26-kit/                          (single GitHub repo, Apache-2.0)
├── spec/decode.md                   ← canonical algorithm spec — single source of truth
├── fixtures/v1/                     ← (image, raw_output.npy, expected.json) triples
│                                       — captured for both e2e and non-e2e exports
├── python/                          ← package: yolo26-kit (PyPI)
├── js/                              ← package: @yolo26/kit (npm)
├── scripts/
│   ├── gen_fixtures.py              ← uses ultralytics; rerun on YOLO26 release bumps
│   └── live_diff.py                 ← nightly CI: install ultralytics, compare
└── .github/workflows/
    ├── ci.yml                       ← unit + fixture tests; Py 3.9-3.13, Node 18-22
    └── live_diff.yml                ← scheduled nightly, opens issue on drift
```

### Public API — Python (`yolo26_kit`)

```python
# --- bridge primary path (e2e default exports) ---
def filter_e2e(
    output: np.ndarray,                      # (N, 300, 6) or (300, 6)
    conf: float = 0.25,
    classes: Iterable[int] | None = None,
    min_area: float | None = None,
    format: Literal["dict", "arrays"] = "dict",
) -> list[dict] | dict[str, np.ndarray]: ...

def e2e_to_v8_shape(output: np.ndarray, *, num_classes: int = 80) -> np.ndarray:
    """(N, 300, 6) → (1, 4+nc, N) v8-style for Triton/DeepStream/OpenCV DNN."""

def v8_shape_to_e2e(output: np.ndarray) -> np.ndarray:
    """(1, 4+nc, N) → (N, K, 6) — runs argmax + confidence packing."""

# --- non-e2e path (raw exports / NPU / quantized) ---
def decode_detect(
    output: np.ndarray,                      # (1, 4+nc, N) or (1, N, 4+nc)
    conf: float = 0.25,
    format: Literal["dict", "arrays"] = "dict",
    *,
    assume_sigmoid: bool = True,
    strict: bool = False,
    strict_dtype: bool = False,
) -> list[dict] | dict[str, np.ndarray]: ...

# --- shared utilities ---
def letterbox_unmap(boxes, orig_size, lb_size, scale, pad) -> np.ndarray: ...
def normalize_output(output) -> np.ndarray: ...

# --- wrapper (extras=[ort]) ---
def from_ort(session_or_path, *, providers=None) -> Decoder: ...
class Decoder:
    backend: str
    is_e2e: bool                              # auto-detected from output rank/shape
    def predict(self, image, conf=0.25, classes=None, format="dict") -> ...: ...
```

### Public API — TypeScript (`@yolo26/kit`)

```ts
// e2e primary
export function filterE2E(
  output: Float32Array, shape: readonly number[],
  opts?: { conf?: number; classes?: number[]; minArea?: number; format?: "dict" | "arrays" }
): Detection[] | { boxes: Float32Array; scores: Float32Array; classes: Int32Array };

export function e2eToV8Shape(output: Float32Array, shape: number[], opts?: { numClasses?: number }):
  { data: Float32Array; shape: number[] };

export function v8ShapeToE2E(output: Float32Array, shape: number[]):
  { data: Float32Array; shape: number[] };

// non-e2e fallback
export function decodeDetect(
  output: Float32Array, shape: readonly number[],
  opts?: { conf?: number; format?: "dict" | "arrays"; assumeSigmoid?: boolean; strict?: boolean }
): Detection[] | { boxes: Float32Array; scores: Float32Array; classes: Int32Array };

export function letterboxUnmap(...): Box[];
export function normalizeOutput(output: Float32Array | Float16Array): Float32Array;

// peerDep: onnxruntime-web
export function fromOrt(session: ort.InferenceSession): Decoder;
export class Decoder {
  readonly backend: "webgpu" | "wasm" | "webgl";
  readonly isE2E: boolean;
  predict(
    input: HTMLCanvasElement | ImageBitmap | ImageData,
    opts?: { conf?: number; classes?: number[]; format?: "dict" | "arrays" }
  ): Promise<Detection[] | { boxes: Float32Array; scores: Float32Array; classes: Int32Array }>;
}
```

### Cross-language equivalence contract

- `spec/decode.md` defines all algorithms in numbered steps with explicit shapes, dtypes, formulas.
- Golden fixtures are the binding contract: every fixture must produce identical output (tolerance 1e-4 on score, 1px on boxes, exact on class) in both languages.
- Fixtures cover **both** export modes: `_e2e` and `_raw` variants for the same image.
- CI fails on any drift in either direction.

## Components

### Python `python/src/yolo26_kit/`
```
__init__.py              # public API re-exports
core/
  filter_e2e.py          # filter_e2e()
  shapes.py              # e2e_to_v8_shape, v8_shape_to_e2e
  decode_raw.py          # decode_detect() for non-e2e
  letterbox.py           # letterbox_unmap()
  normalize.py           # normalize_output()
  types.py               # Detection TypedDict, COCO_CLASSES
ort/                     # extras: [ort]
  decoder.py             # Decoder (auto e2e/non-e2e routing)
  preprocess.py          # numpy / Pillow / cv2 letterbox forward
  image_io.py            # autoload from path / PIL / ndarray / bytes
py.typed
```

### TypeScript `js/src/`
```
index.ts
core/
  filterE2E.ts
  shapes.ts
  decodeRaw.ts
  letterbox.ts
  normalize.ts
  types.ts
  classes.ts
ort/                     # peerDep: onnxruntime-web
  decoder.ts
  preprocess.ts
```

### Boundary rules

- `core/` = pure functions, no I/O, no DOM, no FS. Numpy / typed arrays only.
- `ort/` = optional. Imports onnxruntime / onnxruntime-web lazily. Never imported by core.
- `core/types.ts` mirrors `core/types.py` field-for-field.
- Future task modules (seg/pose/cls/obb) drop in as siblings under same boundary contract.

### Dependencies

| Package | Hard | Optional |
|---|---|---|
| Py core | numpy>=1.23 | — |
| Py [ort] | + onnxruntime>=1.17 | — |
| Py [pil] | + Pillow>=10 | — |
| Py [cv2] | + opencv-python-headless | — |
| Py [dev] | + pytest, pytest-cov, ruff, mypy, hypothesis, ultralytics (live-diff only) | — |
| TS core | none | — |
| TS ort/ | peer: onnxruntime-web>=1.20 | — |
| TS dev | vitest, fast-check, tsup, typescript, @biomejs/biome | — |

### Size budget

- Py core ≤350 LOC, TS core ≤400 LOC, each wrapper ≤200 LOC.
- Total repo (excl. fixtures + tests) ≤1700 LOC.

## Data Flow

### Path A — e2e default (most users)

```python
out = ort_session.run(None, {"images": pre})[0]    # (1, 300, 6)
out = yolo26_kit.normalize_output(out)             # workaround #23645
dets = yolo26_kit.filter_e2e(out, conf=0.25)
```

Inside `filter_e2e`:

1. Validate shape: accept `(N, 300, 6)`, `(300, 6)`, or any `(*, K, 6)`. Squeeze leading 1.
2. Slice columns: `boxes = out[..., :4]`, `scores = out[..., 4]`, `classes = out[..., 5].astype(int32)`.
3. Apply filters: `conf >= threshold`, optional `classes ∈ allowlist`, optional `min_area`.
4. Sort descending by score; stable tiebreak: class id, then index.
5. Return per `format=`.

Boxes still in 640 letterbox space — wrapper handles unmap. Core stays pure.

### Path B — pipeline-bridge mode (drop-in YOLOv8 replacement)

```python
out_e2e = ort_session.run(None, {"images": pre})[0]                # (1, 300, 6)
out_v8 = yolo26_kit.e2e_to_v8_shape(out_e2e, num_classes=80)        # (1, 84, 8400)
# feed out_v8 directly to existing v8 Triton/DeepStream/OpenCV DNN parser
```

Inside `e2e_to_v8_shape`:

1. From `(N, 300, 6)`: synthesize a `(1, 4+num_classes, max_det)` tensor (typically `(1, 84, 300)` for COCO).
2. For each detection slot: convert box xyxy → cxcywh (`cx=(x1+x2)/2, cy=(y1+y2)/2, w=x2-x1, h=y2-y1`), write to channels `0..3`. Set `cls_score[class_id] = conf` at channel `4+class_id`; all other class channels = 0.
3. Pad unused slots (where original `conf == 0`) with zeros.
4. Return contiguous tensor + shape.

**Note on parser compatibility:** The output anchor count is `max_det` (300), not the full 8400 anchors a stock v8 export emits. Downstream Triton/DeepStream/OpenCV DNN parsers typically iterate the anchor axis without assuming a fixed count, so this works — but configs that hard-code `8400` must be updated. Document this in README compat table.

Reverse `v8_shape_to_e2e`: argmax across class columns, pack into `(N, K, 6)`. Drops the explicit objectness column (YOLO26 has none).

### Path C — non-e2e exports (NPU, quantized, manual export)

```python
out_raw = ort_session.run(None, {"images": pre})[0]                # (1, 4+nc, N)
dets = yolo26_kit.decode_detect(out_raw, conf=0.25)
```

Inside `decode_detect`:

1. Validate shape: accept `(1, 4+nc, N)` or `(1, N, 4+nc)` (transpose if needed).
2. Split: `boxes = ...[:4]` (cxcywh in 640 space), `cls_scores = ...[4:]`.
3. Apply sigmoid based on explicit `assume_sigmoid: bool = True` parameter (default: True — matches ultralytics export). Caller overrides for raw-logit exports. No magic heuristics.
4. `scores = cls_scores.max(-1)`, `classes = cls_scores.argmax(-1)`.
5. cxcywh → xyxy.
6. Filter `scores >= conf`. Sort descending. Stable tiebreak.

### Path D — wrapper (image in, dets in original coords out)

```python
decoder = yolo26_kit.from_ort("yolo26n.onnx")
dets = decoder.predict("bus.jpg", conf=0.25)
```

Inside `Decoder.predict`:

1. Image → ndarray (PIL / cv2 / raw bytes paths).
2. Letterbox forward: scale-fit to 640×640, pad gray (114,114,114), record `(scale, pad_x, pad_y, orig_w, orig_h)`.
3. Normalize `/255`, NHWC → NCHW, cast to model input dtype.
4. `out = session.run(...)[0]; out = normalize_output(out)`.
5. Branch on `is_e2e` (set at session-load time from output shape):
   - e2e → `dets = filter_e2e(out, conf, classes, format="arrays")`.
   - non-e2e → `dets = decode_detect(out, conf, format="arrays")`.
6. `boxes = letterbox_unmap(...)` → original image coords.
7. Repack to user `format=` and return.

### Determinism

- Same input → bit-identical output across Py/TS for fp32 (tol 1e-5); fp16 paths tol 1e-3.
- Output sorted descending by score with stable tiebreak.

### Hot path memory

- Py: minimal allocations beyond numpy intermediates; no copies on `format="arrays"`.
- TS: typed-array views reused where possible; output is fresh `Detection[]`.

## Error Handling

### Core input validation

| Bad input | Behavior |
|---|---|
| Wrong rank for the chosen function | `ValueError` w/ expected vs actual shape |
| Last dim of e2e ≠ 6 | `ValueError("filter_e2e expects last dim = 6 [x1,y1,x2,y2,conf,cls]; got {n}")` |
| Last dim of non-e2e < 5 | `ValueError("decode_detect expects 4+nc on the class axis (≥5)")` |
| Batch > 1 | `ValueError("batched decode not supported in v1 — call per-batch-item")` |
| NaN/Inf | `ValueError("output contains NaN/Inf")`; `strict=False` drops affected rows |
| `conf` ∉ [0,1] | `ValueError` |
| Wrong dtype | Auto-cast to fp32 with `RuntimeWarning`; `strict_dtype=True` errors instead |
| Class allowlist out of range | `ValueError` listing offending ids |

### Wrapper validation

| Bad input | Behavior |
|---|---|
| ORT not installed | Import-time `ImportError("install yolo26-kit[ort] to use Decoder")` |
| Model file missing | `FileNotFoundError` |
| Output count ≠ 1 | `ValueError("expected single output tensor; got {n}. Is this YOLO26 detect ONNX?")` |
| Unexpected output shape | `ValueError` w/ inspected shape + export-args hint |
| Image decode fails | Propagate underlying error w/ added context |
| Tainted canvas (TS) | `Error("canvas tainted; load image with crossOrigin='anonymous'")` |

### TS wrapper EP fallback

1. Try WebGPU EP. On unsupported-op or backend-init failure → fall through.
2. Try WASM SIMD. On failure → throw `UnsupportedRuntimeError` listing attempted EPs.
3. `decoder.backend` is read-only and reflects the active EP for UI display.

### Drift detection (live-diff CI)

- Nightly: install latest ultralytics, run on fixtures (both `_e2e` and `_raw`), decode w/ both ours and theirs.
- If max-abs-diff > tolerance OR detection-count differs → auto-open GitHub issue tagged `drift`.
- README maintains `yolo26-kit X.Y.Z` ↔ `ultralytics A.B.C` compatibility table.

### Logging

- Core: zero logging. Errors via exceptions only.
- Wrappers: structured logger `yolo26_kit.*`. Off by default.

### Versioning

- SemVer.
- Output schema change in YOLO26 → MAJOR bump.
- Fixtures versioned `fixtures/v1/`, `fixtures/v2/`, etc.

## Testing

### 1. Spec conformance (golden fixtures) — primary contract

```
fixtures/v1/coco_bus_e2e/
  ├── input.jpg
  ├── raw_output.npy        (1, 300, 6) — e2e default
  ├── raw_output_shape.json
  ├── expected.json
  └── meta.json             (orig_size, lb_pad, model_name, ultralytics_version, export_mode)
fixtures/v1/coco_bus_raw/
  └── ...                   (1, 84, 8400) — end2end=False export
```

- Py: `pytest fixtures/v1/*` runs the appropriate function (`filter_e2e` or `decode_detect`); tol 1e-4 score, 1px box, exact class.
- TS: vitest, same fixtures via small `.npy` parser, same assertions.
- Cross-language parity test: run both, compare output symbol-by-symbol.
- Adapter round-trip: `e2e_to_v8_shape(filter_e2e(x))` and `v8_shape_to_e2e(decode_detect(y))` produce equivalent detection sets.

### 2. Unit tests

- `e2e_to_v8_shape`: synthetic `(N,300,6)` → check shape, dtype, that downstream `decode_detect` recovers same dets.
- `v8_shape_to_e2e`: synthetic `(1,84,8400)` → check shape, conf packing.
- `letterbox_unmap`: synthetic `(scale=0.5, pad=(80,0))` round-trip.
- `normalize_output`: fp16-input/fp32-output (issue #23645), int8 dequant edge cases.
- `filter_e2e`: empty (all below conf), single det, all-NaN row w/ `strict=False`, classes allowlist, min_area filter.
- `decode_detect`: tied class logits, batch=1 vs squeezed, transposed input.

### 3. Wrapper tests (`[ort]`)

- Tiny synthetic ONNX (~5KB, hand-crafted) for both e2e and raw modes → end-to-end pipe yields expected dets.
- Image-format paths: PIL, ndarray HWC uint8/fp32, raw bytes, file path. All identical dets.
- TS: mock InferenceSession returning Float32Array → wrapper produces same dets.
- Auto e2e/non-e2e routing: feed both shapes, verify correct branch.

### 4. Property tests

- Py `hypothesis`, TS `fast-check`. Invariants:
  - `len(dets) == (scores >= conf).sum()` for any conf, any output mode.
  - All boxes have `x1<x2 ∧ y1<y2`.
  - All `class ∈ [0, nc)`.
  - Output sorted descending by score.
  - `e2e_to_v8_shape ∘ v8_shape_to_e2e` is detection-equivalent (modulo zero-pad rows).

### 5. Live-diff (nightly only)

- Separate workflow. Installs latest ultralytics. Generates a fixture on-the-fly. Compares. Files issue on drift. Updates README compat table.

### 6. Performance smoke

- `pytest-benchmark` and `vitest bench`. Fail PR if median regresses >25% vs main.

### CI matrix

```yaml
ci.yml (every PR):
  py:   {3.9, 3.10, 3.11, 3.12, 3.13} × {ubuntu, macos, windows}
  node: {18, 20, 22}                  × {ubuntu, macos, windows}
  steps: install → unit + fixture + property → bench (regression gate)

live_diff.yml (nightly, ubuntu only):
  install latest ultralytics → run live-diff → file issue if drift
```

### Coverage target

- Core ≥95% line coverage.
- Wrapper ≥80%.
- Mutation testing optional (`mutmut` / `stryker`); set up but not gating.

## Open Questions (revisit during planning)

- Final repo URL / GitHub org choice.
- Whether to pre-publish placeholder PyPI / npm names to reserve them.
- Whether to seed fixtures from a single `yolo26n.onnx` or all 5 sizes (`n/s/m/l/x`).
- Whether `e2e_to_v8_shape` should default `num_classes=80` or require explicit pass (less footgun risk).

## Out of Scope (future modules in same repo)

- Seg / pose / cls / OBB decoders + adapters.
- Reproduction kit (Docker + pinned hyperparams + COCO train/eval).
- ONNX output schema standardizer (issue #24265).
- NPU ports (RKNN, Hailo, RDK, SNPE/QNN) — will reuse this bridge.
- Pretrained P2/P6 weights.
- Quantization recipe library.

## References

- Ultralytics YOLO26 docs: <https://docs.ultralytics.com/models/yolo26/>
- End-to-End Detection guide: <https://docs.ultralytics.com/guides/end2end-detection/>
- Issue #23645 (e2e fp16 export emits fp32 output): <https://github.com/ultralytics/ultralytics/issues/23645>
- Issue #23339 (dynamic batch + NMS): <https://github.com/ultralytics/ultralytics/issues/23339>
- Issue #23978 (abnormal box): <https://github.com/ultralytics/ultralytics/issues/23978>
- Issue #23850 (overlapping boxes): <https://github.com/ultralytics/ultralytics/issues/23850>
- Issue #23685 (seg duplicate dets): <https://github.com/ultralytics/ultralytics/issues/23685>
- Issue #23756 (TRT export errors): <https://github.com/ultralytics/ultralytics/issues/23756>
- Issue #24265 (ONNX schema RFC): <https://github.com/ultralytics/ultralytics/issues/24265>
- arXiv 2510.09653 (YOLO Evolution overview)
