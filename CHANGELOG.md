# Changelog

## [0.1.0] - 2026-05-06

Initial release.

### Added
- `filter_e2e` / `filterE2E` — confidence + class + min-area filter for `(N, 300, 6)` e2e output (Algorithm A).
- `e2e_to_v8_shape` / `e2eToV8Shape` — adapter to drop YOLO26 into legacy YOLOv8 pipelines (Algorithm B).
- `v8_shape_to_e2e` / `v8ShapeToE2E` — reverse adapter (Algorithm C).
- `decode_detect` / `decodeDetect` — decoder for non-e2e raw `(1, 4+nc, N)` exports for NPU / quantized targets (Algorithm D).
- `letterbox_unmap` / `letterboxUnmap` — coord undo to original image space (Algorithm E).
- `normalize_output` / `normalizeOutput` — dtype workaround for upstream issue ultralytics#23645 (Algorithm F).
- `Decoder` class with auto e2e/non-e2e routing in both Python and TypeScript.
- Shared axis-detection helper (`_axes._split_channel_anchor_axes` Py / `_axes.splitChannelAnchorAxes` TS) used by both `decode_detect` and `v8_shape_to_e2e` for consistent channel-axis detection.
- Golden fixtures (`fixtures/v1/`): `coco_bus` and `coco_zidane` in both export modes (e2e + raw), generated from real `yolo26n.pt`.
- Cross-language parity: Python and TypeScript decoders verified bit-for-bit equivalent (within 1e-4 score / 1px box / exact class) via shared fixtures.
- Hypothesis (Py) and fast-check (TS) property tests covering count-matches-threshold, sort-order, box-validity, class-range invariants.
- pytest-benchmark smoke tests for hot decode paths.
- GitHub Actions CI: Python 3.10–3.13 × {ubuntu, macos, windows}, Node 18–22 × {ubuntu, macos, windows}.
- Nightly live-diff workflow that opens auto-tagged drift issues.

### Known issues
- Batched (batch>1) decoding not supported. Use per-batch loop. Tracked for v1.1.
- TS `letterboxImageData` uses nearest-neighbor resize for portability; for pixel-perfect parity with PIL bilinear, pre-resize via canvas.
- Spec/decode.md and the implementation diverge slightly on some edge cases (e.g., heuristic for ambiguous channel-vs-anchor when both trailing dims are ≥5); fixtures bind the implementation behavior.
