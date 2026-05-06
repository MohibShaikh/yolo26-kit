import { describe, expect, it } from "vitest";
import { normalizeOutput } from "../../src/core/normalize";

describe("normalizeOutput", () => {
  it("passthrough Float32Array", () => {
    const a = new Float32Array(6);
    expect(normalizeOutput(a)).toBe(a);
  });

  it("converts Float64Array to Float32Array", () => {
    const a = new Float64Array([1.5, 2.5]);
    const out = normalizeOutput(a);
    expect(out).toBeInstanceOf(Float32Array);
    expect(out[0]).toBeCloseTo(1.5, 5);
  });
});
