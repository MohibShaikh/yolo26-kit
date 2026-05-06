import { describe, expect, it } from "vitest";
import { letterboxUnmap } from "../../src/core/letterbox";

describe("letterboxUnmap", () => {
  it("round trips no pad", () => {
    const boxes = new Float32Array([100, 100, 200, 200]);
    const out = letterboxUnmap(boxes, [640, 640], [640, 640], 1.0, [0, 0]);
    expect([...out]).toEqual([100, 100, 200, 200]);
  });

  it("applies scale and pad", () => {
    const boxes = new Float32Array([100, 240, 200, 380]);
    const out = letterboxUnmap(boxes, [1280, 720], [640, 640], 0.5, [0, 140]);
    expect([...out]).toEqual([200, 200, 400, 480]);
  });

  it("clips to image", () => {
    const boxes = new Float32Array([-10, -5, 1290, 730]);
    const out = letterboxUnmap(boxes, [1280, 720], [1280, 720], 1.0, [0, 0]);
    expect([...out]).toEqual([0, 0, 1280, 720]);
  });
});
