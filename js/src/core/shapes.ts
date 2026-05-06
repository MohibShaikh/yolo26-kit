export interface ShapedTensor {
  data: Float32Array;
  shape: number[];
}

function readF32(arr: Float32Array, i: number): number {
  // biome-ignore lint/style/noNonNullAssertion: index bounds guaranteed by caller
  return arr[i]!;
}

interface AxesSplit {
  channelStride: number; // stride in the flat array to advance one channel
  anchorStride: number; // stride to advance one anchor
  channels: number;
  anchors: number;
}

function splitChannelAnchorAxes(
  shape: readonly number[],
  numClasses?: number,
): { canonicalAxes: AxesSplit } {
  if (shape.length !== 3 || shape[0] !== 1) {
    throw new Error(`expected (1, ?, ?); got ${JSON.stringify(shape)}`);
  }
  const a = shape[1] as number;
  const b = shape[2] as number;
  const expected = numClasses === undefined ? undefined : 4 + numClasses;

  // If numClasses provided, prefer the dim that exactly equals 4+numClasses.
  if (expected !== undefined) {
    if (a === expected && b === expected) {
      throw new Error(`ambiguous: both trailing dims equal ${expected}`);
    }
    if (a === expected) return { canonicalAxes: channelsFirst(a, b) };
    if (b === expected) return { canonicalAxes: anchorsFirst(a, b) };
    throw new Error(
      `neither trailing dim equals num_classes+4=${expected}; shape=${JSON.stringify(shape)}`,
    );
  }

  // Heuristic: smaller trailing dim is channels, must be >=5.
  if (a === b) throw new Error("ambiguous channel/anchor: trailing dims equal");
  const aOk = a >= 5;
  const bOk = b >= 5;
  if (aOk && bOk) {
    return a < b ? { canonicalAxes: channelsFirst(a, b) } : { canonicalAxes: anchorsFirst(a, b) };
  }
  if (aOk) return { canonicalAxes: channelsFirst(a, b) };
  if (bOk) return { canonicalAxes: anchorsFirst(a, b) };
  throw new Error(`class axis must be ≥5; got shape ${JSON.stringify(shape)}`);
}

function channelsFirst(channels: number, anchors: number): AxesSplit {
  // shape (1, channels, anchors): stride for channel = anchors, anchor = 1
  return { channelStride: anchors, anchorStride: 1, channels, anchors };
}

function anchorsFirst(anchors: number, channels: number): AxesSplit {
  // shape (1, anchors, channels): stride for channel = 1, anchor = channels
  return { channelStride: 1, anchorStride: channels, channels, anchors };
}

export function e2eToV8Shape(
  data: Float32Array,
  shape: readonly number[],
  opts: { numClasses?: number } = {},
): ShapedTensor {
  const numClasses = opts.numClasses ?? 80;

  let K: number;
  if (shape.length === 3 && shape[0] === 1) K = shape[1] as number;
  else if (shape.length === 2) K = shape[0] as number;
  else throw new Error(`expected (K,6) or (1,K,6); got ${JSON.stringify(shape)}`);
  if ((shape[shape.length - 1] as number) !== 6) {
    throw new Error(`e2e last dim must be 6; got ${shape[shape.length - 1]}`);
  }

  const C = 4 + numClasses;
  const out = new Float32Array(C * K);

  for (let i = 0; i < K; i++) {
    const off = i * 6;
    const x1 = readF32(data, off);
    const y1 = readF32(data, off + 1);
    const x2 = readF32(data, off + 2);
    const y2 = readF32(data, off + 3);
    const conf = readF32(data, off + 4);
    const cid = readF32(data, off + 5) | 0;
    if (conf <= 0) continue;
    if (cid < 0 || cid >= numClasses) {
      throw new Error(`class id ${cid} out of range for numClasses=${numClasses}`);
    }
    out[0 * K + i] = (x1 + x2) * 0.5;
    out[1 * K + i] = (y1 + y2) * 0.5;
    out[2 * K + i] = x2 - x1;
    out[3 * K + i] = y2 - y1;
    out[(4 + cid) * K + i] = conf;
  }
  return { data: out, shape: [1, C, K] };
}

export function v8ShapeToE2E(
  data: Float32Array,
  shape: readonly number[],
  opts: { numClasses?: number } = {},
): ShapedTensor {
  const { canonicalAxes } = splitChannelAnchorAxes(shape, opts.numClasses);
  const { channelStride, anchorStride, channels, anchors } = canonicalAxes;

  const at = (ch: number, n: number) => readF32(data, ch * channelStride + n * anchorStride);

  const out = new Float32Array(anchors * 6);
  for (let i = 0; i < anchors; i++) {
    const cx = at(0, i);
    const cy = at(1, i);
    const w = at(2, i);
    const h = at(3, i);
    let bestScore = Number.NEGATIVE_INFINITY;
    let bestClass = 0;
    for (let c = 4; c < channels; c++) {
      const s = at(c, i);
      if (s > bestScore) {
        bestScore = s;
        bestClass = c - 4;
      }
    }
    out[i * 6] = cx - w * 0.5;
    out[i * 6 + 1] = cy - h * 0.5;
    out[i * 6 + 2] = cx + w * 0.5;
    out[i * 6 + 3] = cy + h * 0.5;
    out[i * 6 + 4] = bestScore;
    out[i * 6 + 5] = bestClass;
  }
  return { data: out, shape: [1, anchors, 6] };
}
