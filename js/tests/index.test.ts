import { describe, expect, it } from "vitest";
import { VERSION } from "../src/index.js";

describe("@yolo26/kit", () => {
  it("exports VERSION", () => {
    expect(VERSION).toBe("0.1.0");
  });
});
