export function classAwareNMS(
  boxes: Float32Array, // flat (N*4) xyxy
  scores: Float32Array, // (N)
  classes: Int32Array, // (N)
  iouThreshold = 0.45,
): Int32Array {
  const N = scores.length;
  if (N === 0) return new Int32Array(0);

  // Group by class
  const byClass = new Map<number, number[]>();
  for (let i = 0; i < N; i++) {
    const c = classes[i] as number;
    let arr = byClass.get(c);
    if (!arr) {
      arr = [];
      byClass.set(c, arr);
    }
    arr.push(i);
  }

  const keepGlobal: number[] = [];
  for (const [, idxs] of byClass) {
    const kept = nmsSingleClass(boxes, scores, idxs, iouThreshold);
    keepGlobal.push(...kept);
  }
  // Sort by descending global score
  keepGlobal.sort((a, b) => (scores[b] as number) - (scores[a] as number));
  return new Int32Array(keepGlobal);
}

function nmsSingleClass(
  boxes: Float32Array,
  scores: Float32Array,
  idxs: number[],
  iou: number,
): number[] {
  // Sort idxs by score desc
  idxs.sort((a, b) => (scores[b] as number) - (scores[a] as number));
  const keep: number[] = [];
  const suppressed = new Set<number>();
  for (const i of idxs) {
    if (suppressed.has(i)) continue;
    keep.push(i);
    const ix1 = boxes[i * 4] as number;
    const iy1 = boxes[i * 4 + 1] as number;
    const ix2 = boxes[i * 4 + 2] as number;
    const iy2 = boxes[i * 4 + 3] as number;
    const iArea = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
    for (const j of idxs) {
      if (j === i || suppressed.has(j)) continue;
      const jx1 = boxes[j * 4] as number;
      const jy1 = boxes[j * 4 + 1] as number;
      const jx2 = boxes[j * 4 + 2] as number;
      const jy2 = boxes[j * 4 + 3] as number;
      const jArea = Math.max(0, jx2 - jx1) * Math.max(0, jy2 - jy1);
      const xx1 = Math.max(ix1, jx1);
      const yy1 = Math.max(iy1, jy1);
      const xx2 = Math.min(ix2, jx2);
      const yy2 = Math.min(iy2, jy2);
      const w = Math.max(0, xx2 - xx1);
      const h = Math.max(0, yy2 - yy1);
      const inter = w * h;
      const u = iArea + jArea - inter;
      if (u > 0 && inter / u > iou) suppressed.add(j);
    }
  }
  return keep;
}
