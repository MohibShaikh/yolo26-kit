export { decodeDetect } from "./core/decodeRaw";
export { filterE2E } from "./core/filterE2E";
export { letterboxUnmap } from "./core/letterbox";
export { normalizeOutput } from "./core/normalize";
export { e2eToV8Shape, v8ShapeToE2E } from "./core/shapes";
export { COCO_CLASSES } from "./core/classes";
export type {
  ArrayDetections,
  DecodeRawOptions,
  DetectFormat,
  Detection,
  FilterE2EOptions,
} from "./core/types";

export const VERSION = "0.1.0";
