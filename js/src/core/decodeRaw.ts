import { splitChannelAnchorAxes } from "./_axes";
import { COCO_CLASSES } from "./classes";
import { classAwareNMS } from "./nms";
import type { ArrayDetections, DecodeRawOptions, Detection } from "./types";

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

function readF32(arr: Float32Array, i: number): number {
  // biome-ignore lint/style/noNonNullAssertion: index bounds guaranteed by caller
  return arr[i]!;
}

export function decodeDetect(
  data: Float32Array,
  shape: readonly number[],
  opts: DecodeRawOptions = {},
): Detection[] | ArrayDetections {
  const {
    conf = 0.25,
    format = "dict",
    assumeSigmoid = true,
    strict = false,
    numClasses,
    classes,
    minArea,
    nms = true,
    iouThreshold = 0.45,
  } = opts;
  if (conf < 0 || conf > 1) throw new Error(`conf must be in [0, 1]; got ${conf}`);

  const { canonicalAxes } = splitChannelAnchorAxes(shape, numClasses);
  const { channelStride, anchorStride, channels, anchors } = canonicalAxes;
  const nc = channels - 4;

  const at = (ch: number, n: number) => readF32(data, ch * channelStride + n * anchorStride);

  const allowlist = classes ? new Set(classes) : null;
  if (allowlist) {
    for (const c of allowlist) {
      if (c < 0 || c >= COCO_CLASSES.length) {
        throw new Error(`classes allowlist out of range: ${c}`);
      }
    }
  }

  const boxesArr: number[] = [];
  const scoresArr: number[] = [];
  const clsArr: number[] = [];
  const idxs: number[] = [];

  for (let i = 0; i < anchors; i++) {
    const cx = at(0, i);
    const cy = at(1, i);
    const w = at(2, i);
    const h = at(3, i);
    let bestRaw = Number.NEGATIVE_INFINITY;
    let bestC = 0;
    for (let c = 0; c < nc; c++) {
      const v = at(4 + c, i);
      if (v > bestRaw) {
        bestRaw = v;
        bestC = c;
      }
    }
    const score = assumeSigmoid ? bestRaw : sigmoid(bestRaw);
    const finite =
      Number.isFinite(cx) &&
      Number.isFinite(cy) &&
      Number.isFinite(w) &&
      Number.isFinite(h) &&
      Number.isFinite(score);
    if (!finite) {
      if (strict) throw new Error("output contains NaN/Inf");
      continue;
    }
    if (score < conf) continue;
    if (allowlist && !allowlist.has(bestC)) continue;
    if (bestC < 0 || bestC >= COCO_CLASSES.length) {
      if (strict) throw new Error(`class_id out of range: ${bestC}`);
      continue;
    }
    if (minArea !== undefined) {
      const area = w * h;
      if (area < minArea) continue;
    }
    boxesArr.push(cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5);
    scoresArr.push(score);
    clsArr.push(bestC);
    idxs.push(i);
  }

  let keepIndices: number[] | null = null;
  if (nms && scoresArr.length > 0) {
    const M0 = scoresArr.length;
    const flatBoxes = new Float32Array(M0 * 4);
    for (let k = 0; k < M0 * 4; k++) flatBoxes[k] = boxesArr[k] as number;
    const flatScores = new Float32Array(scoresArr);
    const flatClasses = new Int32Array(clsArr);
    const kept = classAwareNMS(flatBoxes, flatScores, flatClasses, iouThreshold);
    keepIndices = Array.from(kept);
  }

  const candidateIdxs = keepIndices !== null ? keepIndices : scoresArr.map((_, i) => i);

  const order = candidateIdxs.slice().sort((p, q) => {
    const sd = (scoresArr[q] as number) - (scoresArr[p] as number);
    if (sd !== 0) return sd;
    const cd = (clsArr[p] as number) - (clsArr[q] as number);
    if (cd !== 0) return cd;
    return (idxs[p] as number) - (idxs[q] as number);
  });

  const M = order.length;
  const outBoxes = new Float32Array(M * 4);
  const outScores = new Float32Array(M);
  const outClasses = new Int32Array(M);
  for (let k = 0; k < M; k++) {
    const j = order[k] as number;
    outBoxes[k * 4] = boxesArr[j * 4] as number;
    outBoxes[k * 4 + 1] = boxesArr[j * 4 + 1] as number;
    outBoxes[k * 4 + 2] = boxesArr[j * 4 + 2] as number;
    outBoxes[k * 4 + 3] = boxesArr[j * 4 + 3] as number;
    outScores[k] = scoresArr[j] as number;
    outClasses[k] = clsArr[j] as number;
  }

  if (format === "arrays") {
    return { boxes: outBoxes, scores: outScores, classes: outClasses };
  }
  if (format === "dict") {
    const dets: Detection[] = [];
    for (let k = 0; k < M; k++) {
      const c = outClasses[k] as number;
      dets.push({
        box: [
          outBoxes[k * 4] as number,
          outBoxes[k * 4 + 1] as number,
          outBoxes[k * 4 + 2] as number,
          outBoxes[k * 4 + 3] as number,
        ],
        score: outScores[k] as number,
        class: c,
        label: COCO_CLASSES[c] as string,
      });
    }
    return dets;
  }
  throw new Error(`unknown format: ${format}`);
}
