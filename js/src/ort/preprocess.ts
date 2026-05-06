export interface LetterboxMeta {
  scale: number;
  pad: [number, number];
  origSize: [number, number];
  lbSize: [number, number];
}

interface PreprocessResult {
  tensor: Float32Array;
  meta: LetterboxMeta;
}

function resizeNN(
  src: Uint8ClampedArray,
  sw: number,
  sh: number,
  dw: number,
  dh: number,
): Uint8ClampedArray {
  const dst = new Uint8ClampedArray(dw * dh * 4);
  const xRatio = sw / dw;
  const yRatio = sh / dh;
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(sh - 1, Math.floor(y * yRatio));
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(sw - 1, Math.floor(x * xRatio));
      const so = (sy * sw + sx) * 4;
      const dOff = (y * dw + x) * 4;
      dst[dOff] = src[so] as number;
      dst[dOff + 1] = src[so + 1] as number;
      dst[dOff + 2] = src[so + 2] as number;
      dst[dOff + 3] = 255;
    }
  }
  return dst;
}

export function letterboxImageData(img: ImageData, target = 640, padValue = 114): PreprocessResult {
  const W = img.width;
  const H = img.height;
  const data = img.data;
  const scale = Math.min(target / W, target / H);
  const newW = Math.round(W * scale);
  const newH = Math.round(H * scale);
  const resized = resizeNN(data, W, H, newW, newH);
  const padX = Math.floor((target - newW) / 2);
  const padY = Math.floor((target - newH) / 2);

  const tensor = new Float32Array(1 * 3 * target * target);
  const planeSize = target * target;
  const padNorm = padValue / 255;
  tensor.fill(padNorm);

  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const so = (y * newW + x) * 4;
      const px = padX + x;
      const py = padY + y;
      const dst = py * target + px;
      tensor[0 * planeSize + dst] = (resized[so] as number) / 255;
      tensor[1 * planeSize + dst] = (resized[so + 1] as number) / 255;
      tensor[2 * planeSize + dst] = (resized[so + 2] as number) / 255;
    }
  }
  return {
    tensor,
    meta: {
      scale,
      pad: [padX, padY],
      origSize: [W, H],
      lbSize: [target, target],
    },
  };
}

export async function imageBitmapToImageData(bitmap: ImageBitmap): Promise<ImageData> {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("could not get 2d context for OffscreenCanvas");
  ctx.drawImage(bitmap, 0, 0);
  return ctx.getImageData(0, 0, bitmap.width, bitmap.height);
}

export function canvasToImageData(canvas: HTMLCanvasElement): ImageData {
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("could not get 2d context (canvas tainted by cross-origin?)");
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}
