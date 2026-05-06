import { describe, expect, it } from "vitest";
import { e2eToV8Shape, v8ShapeToE2E } from "../../src/core/shapes";

describe("e2eToV8Shape", () => {
  it("places cxcywh in channels 0..3 and conf at 4+cls", () => {
    const e2e = new Float32Array(2 * 6);
    e2e.set([10, 20, 30, 40, 0.9, 5], 0);
    e2e.set([0, 0, 0, 0, 0, 0], 6);
    const out = e2eToV8Shape(e2e, [1, 2, 6], { numClasses: 80 });
    expect(out.shape).toEqual([1, 84, 2]);
    const at = (ch: number, slot: number) => out.data[ch * 2 + slot] as number;
    expect(at(0, 0)).toBe(20);
    expect(at(1, 0)).toBe(30);
    expect(at(2, 0)).toBe(20);
    expect(at(3, 0)).toBe(20);
    expect(at(4 + 5, 0)).toBeCloseTo(0.9, 5);
    expect(at(4 + 0, 0)).toBe(0);
    for (let ch = 0; ch < 84; ch++) expect(at(ch, 1)).toBe(0);
  });
});

describe("v8ShapeToE2E", () => {
  it("round-trips a single detection", () => {
    const v8 = new Float32Array(1 * 84 * 1);
    v8[0] = 20;
    v8[1] = 30;
    v8[2] = 20;
    v8[3] = 20;
    v8[4 + 5] = 0.9;
    const out = v8ShapeToE2E(v8, [1, 84, 1], { numClasses: 80 });
    expect(out.shape).toEqual([1, 1, 6]);
    const get = (i: number) => out.data[i] as number;
    expect(get(0)).toBe(10);
    expect(get(1)).toBe(20);
    expect(get(2)).toBe(30);
    expect(get(3)).toBe(40);
    expect(get(4)).toBeCloseTo(0.9, 5);
    expect(get(5) | 0).toBe(5);
  });

  it("accepts transposed (1, N, 4+nc)", () => {
    const v8t = new Float32Array(1 * 1 * 84);
    v8t[0] = 20;
    v8t[1] = 30;
    v8t[2] = 20;
    v8t[3] = 20;
    v8t[4 + 5] = 0.9;
    const out = v8ShapeToE2E(v8t, [1, 1, 84], { numClasses: 80 });
    const get = (i: number) => out.data[i] as number;
    expect(get(5) | 0).toBe(5);
  });
});
