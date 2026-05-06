export interface AxesSplit {
  channelStride: number;
  anchorStride: number;
  channels: number;
  anchors: number;
}

export function splitChannelAnchorAxes(
  shape: readonly number[],
  numClasses?: number,
): { canonicalAxes: AxesSplit } {
  if (shape.length !== 3 || shape[0] !== 1) {
    throw new Error(`expected (1, ?, ?); got ${JSON.stringify(shape)}`);
  }
  const a = shape[1] as number;
  const b = shape[2] as number;
  const expected = numClasses === undefined ? undefined : 4 + numClasses;

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
  return { channelStride: anchors, anchorStride: 1, channels, anchors };
}

function anchorsFirst(anchors: number, channels: number): AxesSplit {
  return { channelStride: 1, anchorStride: channels, channels, anchors };
}
