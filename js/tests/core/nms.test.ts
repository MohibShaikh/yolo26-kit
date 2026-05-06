import { describe, expect, it } from "vitest";

import { classAwareNMS } from "../../src/core/nms";

describe("classAwareNMS", () => {
  it("drops overlapping boxes of same class", () => {
    const boxes = new Float32Array([10, 10, 50, 50, 12, 12, 52, 52, 100, 100, 150, 150]);
    const scores = new Float32Array([0.9, 0.8, 0.7]);
    const classes = new Int32Array([0, 0, 0]);
    const keep = classAwareNMS(boxes, scores, classes, 0.45);
    expect(Array.from(keep)).toEqual([0, 2]);
  });

  it("keeps overlapping boxes of different classes", () => {
    const boxes = new Float32Array([10, 10, 50, 50, 12, 12, 52, 52]);
    const scores = new Float32Array([0.9, 0.8]);
    const classes = new Int32Array([0, 1]);
    const keep = classAwareNMS(boxes, scores, classes, 0.45);
    expect(Array.from(keep).sort()).toEqual([0, 1]);
  });

  it("handles empty input", () => {
    const keep = classAwareNMS(new Float32Array(0), new Float32Array(0), new Int32Array(0));
    expect(keep.length).toBe(0);
  });
});
