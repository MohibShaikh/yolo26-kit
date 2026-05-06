import { describe, expect, it } from "vitest";
import { filterE2E } from "../../src/core/filterE2E";

const make = (rows: number[][]): { data: Float32Array; shape: number[] } => {
  const flat = new Float32Array(rows.flat());
  return { data: flat, shape: [1, rows.length, 6] };
};

describe("filterE2E", () => {
  it("returns dict format by default", () => {
    const { data, shape } = make([
      [10, 10, 20, 20, 0.9, 0],
      [0, 0, 5, 5, 0.1, 1],
    ]);
    const dets = filterE2E(data, shape) as {
      box: number[];
      score: number;
      class: number;
      label: string;
    }[];
    expect(dets.length).toBe(1);
    expect(dets[0]?.label).toBe("person");
  });

  it("returns arrays format", () => {
    const { data, shape } = make([
      [10, 10, 20, 20, 0.9, 0],
      [5, 5, 15, 15, 0.5, 2],
    ]);
    const out = filterE2E(data, shape, { format: "arrays" }) as {
      boxes: Float32Array;
      scores: Float32Array;
      classes: Int32Array;
    };
    expect(out.boxes.length).toBe(2 * 4);
    expect(out.scores.length).toBe(2);
  });

  it("sorts descending by score", () => {
    const { data, shape } = make([
      [0, 0, 1, 1, 0.3, 0],
      [0, 0, 1, 1, 0.9, 0],
      [0, 0, 1, 1, 0.5, 0],
    ]);
    const out = filterE2E(data, shape, { conf: 0, format: "arrays" }) as {
      scores: Float32Array;
    };
    expect(out.scores[0]).toBeCloseTo(0.9, 5);
    expect(out.scores[1]).toBeCloseTo(0.5, 5);
    expect(out.scores[2]).toBeCloseTo(0.3, 5);
  });

  it("classes allowlist filter", () => {
    const { data, shape } = make([
      [0, 0, 1, 1, 0.9, 0],
      [0, 0, 1, 1, 0.9, 5],
    ]);
    const out = filterE2E(data, shape, { classes: [5], format: "arrays" }) as {
      classes: Int32Array;
    };
    expect([...out.classes]).toEqual([5]);
  });

  it("min area filter", () => {
    const { data, shape } = make([
      [0, 0, 10, 10, 0.9, 0],
      [0, 0, 2, 2, 0.9, 0],
    ]);
    const out = filterE2E(data, shape, { minArea: 50, format: "arrays" }) as {
      boxes: Float32Array;
    };
    expect(out.boxes.length).toBe(4);
  });

  it("rejects batch > 1", () => {
    const data = new Float32Array(2 * 1 * 6);
    expect(() => filterE2E(data, [2, 1, 6])).toThrow(/batched/);
  });

  it("rejects last dim != 6", () => {
    const data = new Float32Array(1 * 2 * 5);
    expect(() => filterE2E(data, [1, 2, 5])).toThrow(/last dim/);
  });
});
