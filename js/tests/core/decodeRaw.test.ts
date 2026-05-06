import { describe, expect, it } from "vitest";
import { decodeDetect } from "../../src/core/decodeRaw";

const makeV8 = (boxesCxcywh: number[][], scoresPerRow: number[][], n: number, nc = 80) => {
  const data = new Float32Array((4 + nc) * n);
  for (let i = 0; i < boxesCxcywh.length; i++) {
    const b = boxesCxcywh[i] as number[];
    data[0 * n + i] = b[0] as number;
    data[1 * n + i] = b[1] as number;
    data[2 * n + i] = b[2] as number;
    data[3 * n + i] = b[3] as number;
    const sc = scoresPerRow[i] as number[];
    for (let c = 0; c < nc; c++) data[(4 + c) * n + i] = sc[c] ?? 0;
  }
  return { data, shape: [1, 4 + nc, n] as const };
};

describe("decodeDetect", () => {
  it("picks argmax class and score", () => {
    const sc = new Array(80).fill(0);
    sc[5] = 0.9;
    sc[10] = 0.4;
    const { data, shape } = makeV8([[20, 30, 20, 20]], [sc], 1);
    const out = decodeDetect(data, shape, { conf: 0.25, format: "arrays", numClasses: 80 }) as {
      boxes: Float32Array;
      scores: Float32Array;
      classes: Int32Array;
    };
    expect([...out.classes]).toEqual([5]);
    expect(out.scores[0]).toBeCloseTo(0.9, 5);
    expect([...out.boxes]).toEqual([10, 20, 30, 40]);
  });

  it("filters below conf", () => {
    const sc = new Array(80).fill(0);
    sc[5] = 0.1;
    const { data, shape } = makeV8([[20, 30, 20, 20]], [sc], 1);
    const out = decodeDetect(data, shape, { conf: 0.25, format: "arrays", numClasses: 80 }) as {
      scores: Float32Array;
    };
    expect(out.scores.length).toBe(0);
  });

  it("applies sigmoid when assumeSigmoid=false", () => {
    const sc = new Array(80).fill(-10);
    sc[5] = 2.197;
    const { data, shape } = makeV8([[20, 30, 20, 20]], [sc], 1);
    const out = decodeDetect(data, shape, {
      conf: 0.25,
      assumeSigmoid: false,
      format: "arrays",
      numClasses: 80,
    }) as { scores: Float32Array; classes: Int32Array };
    expect([...out.classes]).toEqual([5]);
    expect(out.scores[0] as number).toBeCloseTo(0.9, 3);
  });
});
