import { describe, expect, it } from "vitest";
import { letterboxImageData } from "../../src/ort/preprocess";

describe("letterboxImageData", () => {
  it("square input has scale=1 and pad=(0,0)", () => {
    const W = 640;
    const H = 640;
    const data = new Uint8ClampedArray(W * H * 4);
    for (let i = 0; i < data.length; i += 4) {
      data[i] = 200;
      data[i + 1] = 200;
      data[i + 2] = 200;
      data[i + 3] = 255;
    }
    const imageData = { data, width: W, height: H, colorSpace: "srgb" } as ImageData;
    const { tensor, meta } = letterboxImageData(imageData, 640);
    expect(tensor.length).toBe(1 * 3 * 640 * 640);
    expect(meta.scale).toBe(1);
    expect(meta.pad).toEqual([0, 0]);
    const c = tensor[0 * 640 * 640 + 320 * 640 + 320] as number;
    expect(c).toBeCloseTo(200 / 255, 4);
  });

  it("landscape input pads y", () => {
    const W = 640;
    const H = 360;
    const data = new Uint8ClampedArray(W * H * 4);
    for (let i = 0; i < data.length; i += 4) {
      data[i] = 50;
      data[i + 1] = 50;
      data[i + 2] = 50;
      data[i + 3] = 255;
    }
    const imageData = { data, width: W, height: H, colorSpace: "srgb" } as ImageData;
    const { meta } = letterboxImageData(imageData, 640);
    expect(meta.pad).toEqual([0, 140]);
  });
});
