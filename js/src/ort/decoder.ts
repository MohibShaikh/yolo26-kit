import { COCO_CLASSES } from "../core/classes";
import { decodeDetect } from "../core/decodeRaw";
import { filterE2E } from "../core/filterE2E";
import { letterboxUnmap } from "../core/letterbox";
import { normalizeOutput } from "../core/normalize";
import type { ArrayDetections, Detection } from "../core/types";
import { canvasToImageData, imageBitmapToImageData, letterboxImageData } from "./preprocess";

export interface MinimalSession {
  inputNames: readonly string[];
  outputNames: readonly string[];
  outputMetadata?: Record<string, { dims?: readonly number[] }>;
  run(
    feeds: Record<string, { data: Float32Array; dims: readonly number[]; type: "float32" }>,
  ): Promise<Record<string, { data: Float32Array; dims: readonly number[]; type: string }>>;
}

export interface PredictOptions {
  conf?: number;
  classes?: number[];
  format?: "dict" | "arrays";
}

export type PredictInput = HTMLCanvasElement | ImageBitmap | ImageData;

export class Decoder {
  readonly backend: string;
  readonly isE2E: boolean;
  private readonly _session: MinimalSession;
  private readonly _inputName: string;
  private readonly _outputName: string;

  constructor(session: MinimalSession) {
    this._session = session;
    if (session.outputNames.length !== 1) {
      throw new Error(
        `expected single output tensor; got ${session.outputNames.length}. Is this a YOLO26 detect ONNX?`,
      );
    }
    this._inputName = session.inputNames[0] as string;
    this._outputName = session.outputNames[0] as string;
    const dims = session.outputMetadata?.[this._outputName]?.dims ?? [];
    this.isE2E = dims.length > 0 && dims[dims.length - 1] === 6;
    this.backend = "unknown";
  }

  async predict(
    input: PredictInput,
    opts: PredictOptions = {},
  ): Promise<Detection[] | ArrayDetections> {
    const { conf = 0.25, classes, format = "dict" } = opts;
    let imageData: ImageData;
    if (typeof ImageBitmap !== "undefined" && input instanceof ImageBitmap) {
      imageData = await imageBitmapToImageData(input);
    } else if (typeof HTMLCanvasElement !== "undefined" && input instanceof HTMLCanvasElement) {
      imageData = canvasToImageData(input);
    } else {
      imageData = input as ImageData;
    }

    const { tensor, meta } = letterboxImageData(imageData, 640);
    const result = await this._session.run({
      [this._inputName]: { data: tensor, dims: [1, 3, 640, 640], type: "float32" },
    });
    const raw = result[this._outputName];
    if (!raw) throw new Error(`session output ${this._outputName} missing`);
    const out = normalizeOutput(raw.data);

    let arr: ArrayDetections;
    if (this.isE2E) {
      arr = filterE2E(out, raw.dims, { conf, classes, format: "arrays" }) as ArrayDetections;
    } else {
      arr = decodeDetect(out, raw.dims, { conf, format: "arrays" }) as ArrayDetections;
    }

    if (arr.boxes.length) {
      const unmapped = letterboxUnmap(arr.boxes, meta.origSize, meta.lbSize, meta.scale, meta.pad);
      arr = { boxes: unmapped, scores: arr.scores, classes: arr.classes };
    }

    if (format === "arrays") return arr;
    const dets: Detection[] = [];
    for (let i = 0; i < arr.classes.length; i++) {
      const c = arr.classes[i] as number;
      dets.push({
        box: [
          arr.boxes[i * 4] as number,
          arr.boxes[i * 4 + 1] as number,
          arr.boxes[i * 4 + 2] as number,
          arr.boxes[i * 4 + 3] as number,
        ],
        score: arr.scores[i] as number,
        class: c,
        label: COCO_CLASSES[c] as string,
      });
    }
    return dets;
  }
}

export function fromOrt(session: MinimalSession): Decoder {
  return new Decoder(session);
}
