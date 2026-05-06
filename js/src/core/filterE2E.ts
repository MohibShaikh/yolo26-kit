import { COCO_CLASSES } from "./classes";
import type { ArrayDetections, Detection, FilterE2EOptions } from "./types";

function at(arr: number[], i: number): number {
  // biome-ignore lint/style/noNonNullAssertion: index is always in-bounds at call sites
  return arr[i]!;
}

function atF32(arr: Float32Array, i: number): number {
  // biome-ignore lint/style/noNonNullAssertion: index is always in-bounds at call sites
  return arr[i]!;
}

export function filterE2E(
  output: Float32Array,
  shape: readonly number[],
  opts: FilterE2EOptions = {},
): Detection[] | ArrayDetections {
  const { conf = 0.25, classes, minArea, format = "dict", strict = false } = opts;
  if (conf < 0 || conf > 1) throw new Error(`conf must be in [0, 1]; got ${conf}`);

  let K: number;
  let lastDim: number;
  if (shape.length === 3) {
    if (shape[0] !== 1) throw new Error(`batched decode not supported; got batch=${shape[0]}`);
    K = shape[1] as number;
    lastDim = shape[2] as number;
  } else if (shape.length === 2) {
    K = shape[0] as number;
    lastDim = shape[1] as number;
  } else {
    throw new Error(`expected (K,6) or (1,K,6); got rank ${shape.length}`);
  }
  if (lastDim !== 6) throw new Error(`filter_e2e expects last dim = 6; got ${lastDim}`);

  const boxes: number[] = [];
  const scores: number[] = [];
  const cls: number[] = [];
  const idxs: number[] = [];

  const allowlist = classes ? new Set(classes) : null;
  if (allowlist) {
    for (const c of allowlist) {
      if (c < 0 || c >= COCO_CLASSES.length) {
        throw new Error(`classes allowlist out of range: ${c}`);
      }
    }
  }

  for (let i = 0; i < K; i++) {
    const off = i * 6;
    const x1 = atF32(output, off);
    const y1 = atF32(output, off + 1);
    const x2 = atF32(output, off + 2);
    const y2 = atF32(output, off + 3);
    const confI = atF32(output, off + 4);
    const cid = atF32(output, off + 5) | 0;

    const finite =
      Number.isFinite(x1) &&
      Number.isFinite(y1) &&
      Number.isFinite(x2) &&
      Number.isFinite(y2) &&
      Number.isFinite(confI);
    if (!finite) {
      if (strict) throw new Error("output contains NaN/Inf");
      continue;
    }
    if (confI < conf) continue;
    if (allowlist && !allowlist.has(cid)) continue;
    if (cid < 0 || cid >= COCO_CLASSES.length) {
      if (strict) throw new Error(`class_id out of range: ${cid}`);
      continue;
    }
    if (minArea !== undefined) {
      const area = (x2 - x1) * (y2 - y1);
      if (area < minArea) continue;
    }
    boxes.push(x1, y1, x2, y2);
    scores.push(confI);
    cls.push(cid);
    idxs.push(i);
  }

  const order = scores
    .map((_, i) => i)
    .sort((a, b) => {
      const sd = at(scores, b) - at(scores, a);
      if (sd !== 0) return sd;
      const cd = at(cls, a) - at(cls, b);
      if (cd !== 0) return cd;
      return at(idxs, a) - at(idxs, b);
    });

  const N = order.length;
  const outBoxes = new Float32Array(N * 4);
  const outScores = new Float32Array(N);
  const outClasses = new Int32Array(N);
  for (let k = 0; k < N; k++) {
    const j = at(order, k);
    outBoxes[k * 4] = at(boxes, j * 4);
    outBoxes[k * 4 + 1] = at(boxes, j * 4 + 1);
    outBoxes[k * 4 + 2] = at(boxes, j * 4 + 2);
    outBoxes[k * 4 + 3] = at(boxes, j * 4 + 3);
    outScores[k] = at(scores, j);
    outClasses[k] = at(cls, j);
  }

  if (format === "arrays") {
    return { boxes: outBoxes, scores: outScores, classes: outClasses };
  }
  if (format === "dict") {
    const dets: Detection[] = [];
    for (let k = 0; k < N; k++) {
      const c = atF32(outClasses as unknown as Float32Array, k) | 0;
      dets.push({
        box: [
          atF32(outBoxes, k * 4),
          atF32(outBoxes, k * 4 + 1),
          atF32(outBoxes, k * 4 + 2),
          atF32(outBoxes, k * 4 + 3),
        ],
        score: atF32(outScores, k),
        class: c,
        label: COCO_CLASSES[c] as string,
      });
    }
    return dets;
  }
  throw new Error(`unknown format: ${format}`);
}
