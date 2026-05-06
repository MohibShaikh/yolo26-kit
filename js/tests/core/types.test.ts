import { describe, expect, it } from "vitest";
import { COCO_CLASSES } from "../../src/core/classes";

describe("COCO_CLASSES", () => {
  it("has 80 entries", () => expect(COCO_CLASSES.length).toBe(80));
  it("entry 0 is person", () => expect(COCO_CLASSES[0]).toBe("person"));
  it("entry 79 is toothbrush", () => expect(COCO_CLASSES[79]).toBe("toothbrush"));
});
