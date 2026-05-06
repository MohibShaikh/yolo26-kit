export function letterboxUnmap(
  boxes: Float32Array,
  origSize: readonly [number, number],
  _lbSize: readonly [number, number],
  scale: number,
  pad: readonly [number, number],
): Float32Array {
  const W = origSize[0];
  const H = origSize[1];
  const padX = pad[0];
  const padY = pad[1];
  const N = boxes.length / 4;
  const out = new Float32Array(boxes.length);
  for (let i = 0; i < N; i++) {
    const o = i * 4;
    const x1 = ((boxes[o] as number) - padX) / scale;
    const y1 = ((boxes[o + 1] as number) - padY) / scale;
    const x2 = ((boxes[o + 2] as number) - padX) / scale;
    const y2 = ((boxes[o + 3] as number) - padY) / scale;
    out[o] = Math.max(0, Math.min(W, x1));
    out[o + 1] = Math.max(0, Math.min(H, y1));
    out[o + 2] = Math.max(0, Math.min(W, x2));
    out[o + 3] = Math.max(0, Math.min(H, y2));
  }
  return out;
}
