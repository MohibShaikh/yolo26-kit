export interface Detection {
  box: [number, number, number, number];
  score: number;
  class: number;
  label: string;
}

export interface ArrayDetections {
  boxes: Float32Array;
  scores: Float32Array;
  classes: Int32Array;
}

export type DetectFormat = "dict" | "arrays";

export interface FilterE2EOptions {
  conf?: number;
  classes?: number[];
  minArea?: number;
  format?: DetectFormat;
  strict?: boolean;
}

export interface DecodeRawOptions {
  conf?: number;
  format?: DetectFormat;
  assumeSigmoid?: boolean;
  strict?: boolean;
  numClasses?: number;
  classes?: number[];
  minArea?: number;
  nms?: boolean;
  iouThreshold?: number;
}
