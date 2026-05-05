# yolo26-kit decode spec (canonical)

Both Python and TypeScript implementations MUST follow these algorithms exactly. Golden fixtures bind correctness — see `fixtures/v1/`.

## Conventions

- Box format on output: `xyxy` (`[x1, y1, x2, y2]`).
- Internal box format from raw decode: `cxcywh`.
- Coordinate space: 640×640 letterbox unless `letterbox_unmap` applied.
- Class scores: in `[0, 1]` (sigmoid'd). YOLO26 head emits sigmoid'd values; raw-logit exports require `assume_sigmoid=False`.
- Output sort order: descending by score; stable tiebreak by `(class_id, source_index)`.

## Algorithm A — `filter_e2e`

Input: tensor of shape `(N, K, 6)` or `(K, 6)`. Last-dim columns: `[x1, y1, x2, y2, conf, class_id]`.

1. If rank 3 with leading dim 1, squeeze. If rank 3 with leading dim > 1, error.
2. Validate last dim == 6.
3. Mask = `conf >= conf_threshold`.
4. If `classes` allowlist provided: mask &= `class_id ∈ classes`.
5. If `min_area` provided: mask &= `(x2-x1) * (y2-y1) >= min_area`.
6. Apply mask.
7. Sort descending by `conf`. Stable tiebreak: lower `class_id` first, then lower source index.
8. Return per `format`:
   - `"dict"`: list of `{box: [x1,y1,x2,y2], score, class, label}` where `label = COCO_CLASSES[class]`.
   - `"arrays"`: `{boxes: float32[N,4], scores: float32[N], classes: int32[N]}`.

## Algorithm B — `e2e_to_v8_shape`

Input: tensor `(N, K, 6)`, `num_classes` (default 80).

1. Squeeze leading 1 if present. Let `K_eff = K`.
2. Allocate output `(1, 4 + num_classes, K_eff)` filled with zeros, dtype = input dtype.
3. For each row `i ∈ [0, K_eff)`:
   - Read `x1, y1, x2, y2, conf, cid = input[i]`.
   - If `conf == 0`: skip (leave zero pad).
   - Compute `cx = (x1+x2)/2`, `cy = (y1+y2)/2`, `w = x2-x1`, `h = y2-y1`.
   - Write `output[0, 0, i] = cx`, `output[0, 1, i] = cy`, `output[0, 2, i] = w`, `output[0, 3, i] = h`.
   - Write `output[0, 4 + int(cid), i] = conf`.
4. Return.

## Algorithm C — `v8_shape_to_e2e`

Input: tensor `(1, 4 + nc, N)` or `(1, N, 4 + nc)`.

1. Detect orientation. In YOLO outputs the anchor count N is always greater than the channel count `4+num_classes` (e.g. 8400 anchors with 84 channels). Therefore: identify the channel axis as the trailing dim with the *smaller* size, provided that size is at least 5 (i.e. `4 + num_classes ≥ 5`). The other trailing dim is the anchor axis. If both trailing dims are equal, or if the smaller dim is < 5, raise an error. Canonical layout `(1, 4+nc, N)` is preferred when emitting; if input is already canonical no transpose occurs.
2. Transpose to canonical `(1, 4+nc, N)`.
3. `boxes_cxcywh = output[0, 0:4, :]`. Convert to `xyxy`.
4. `cls = output[0, 4:, :]`. `scores = cls.max(axis=0)`. `classes = cls.argmax(axis=0)`.
5. Pack `(N, 6)` rows: `[x1, y1, x2, y2, score, class]`. Add leading axis to produce `(1, N, 6)`.

## Algorithm D — `decode_detect` (non-e2e raw)

Input: tensor `(1, 4+nc, N)` or `(1, N, 4+nc)`. Param `assume_sigmoid: bool` (default True).

1. Detect orientation: identify the channel axis as the trailing dim with the smaller size (must be ≥ 5); the other trailing dim is the anchor axis. Error if the trailing dims are equal or the smaller dim is < 5. Canonical layout is `(1, 4+nc, N)`.
2. Transpose to canonical `(1, 4+nc, N)`.
3. `boxes_cxcywh = output[0, 0:4, :]`. Convert to `xyxy`.
4. `cls = output[0, 4:, :]`. If `not assume_sigmoid`: `cls = sigmoid(cls)`.
5. `scores = cls.max(axis=0)`, `classes = cls.argmax(axis=0)`.
6. Mask `scores >= conf_threshold`.
7. Apply, sort, return per `format` (same as Algorithm A step 8).

## Algorithm E — `letterbox_unmap`

Input: `boxes (N, 4) xyxy`, `orig_size = (W_orig, H_orig)`, `lb_size = (W_lb, H_lb)`, `scale: float`, `pad: (pad_x, pad_y)`.

1. `x1' = (x1 - pad_x) / scale`; same for `y1, x2, y2`.
2. Clip to `[0, W_orig]` for x, `[0, H_orig]` for y.
3. Return.

## Algorithm F — `normalize_output`

Input: any tensor.

1. If dtype is `float16`/`uint16` (raw fp16 reinterpreted as fp32 in some exports): promote to float32.
2. If dtype is integer (e.g., int8 quantized output without dequant): error with hint to dequantize first.
3. Return contiguous float32 tensor.

## Conformance

For each fixture in `fixtures/v1/<name>/`:
- `expected.json` is the binding output for the function described in `meta.json["api"]` (e.g., `"filter_e2e"`).
- Tolerances: score ≤ 1e-4 abs, box ≤ 1.0 px abs, class exact.
- Equivalent in Python and TypeScript.

## Round-trip invariants

- Composition `e2e_to_v8_shape` followed by `decode_detect` (Algorithm B then D) returns a detection set equivalent to applying `filter_e2e` (Algorithm A) directly to the original input, modulo ordering — both produce the same elements when sorted by score descending.
- Composition `filter_e2e` (Algorithm A) on a synthetic e2e input followed by `e2e_to_v8_shape` then `v8_shape_to_e2e` (Algorithms B then C) returns a detection set equivalent to the input.
