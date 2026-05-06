import { describe, expect, it } from "vitest";
import { Decoder } from "../../src/ort/decoder";
import type { MinimalSession } from "../../src/ort/decoder";

const mockSession = (outputShape: number[], producer: () => Float32Array): MinimalSession => ({
  inputNames: ["images"],
  outputNames: ["output0"],
  outputMetadata: { output0: { dims: outputShape } },
  async run() {
    return {
      output0: { data: producer(), dims: outputShape, type: "float32" },
    };
  },
});

const e2eProducer = () => {
  const out = new Float32Array(1 * 300 * 6);
  out[0] = 10;
  out[1] = 10;
  out[2] = 30;
  out[3] = 30;
  out[4] = 0.9;
  out[5] = 5;
  return out;
};

const buildImageData = (W: number, H: number, color: number): ImageData => {
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = color;
    data[i + 1] = color;
    data[i + 2] = color;
    data[i + 3] = 255;
  }
  return { data, width: W, height: H, colorSpace: "srgb" } as ImageData;
};

describe("Decoder", () => {
  it("detects e2e from output shape", () => {
    const sess = mockSession([1, 300, 6], e2eProducer);
    const d = new Decoder(sess);
    expect(d.isE2E).toBe(true);
  });

  it("predict on 640x640 image returns dets", async () => {
    const sess = mockSession([1, 300, 6], e2eProducer);
    const d = new Decoder(sess);
    const img = buildImageData(640, 640, 200);
    const dets = (await d.predict(img, { conf: 0.25, format: "arrays" })) as {
      boxes: Float32Array;
      classes: Int32Array;
    };
    expect(dets.classes[0]).toBe(5);
    expect([...dets.boxes].slice(0, 4)).toEqual([10, 10, 30, 30]);
  });
});
