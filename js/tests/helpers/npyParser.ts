import { readFileSync } from "node:fs";

interface Npy {
  data: Float32Array;
  shape: number[];
  dtype: string;
}

const MAGIC = new Uint8Array([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]);

export function readNpy(path: string): Npy {
  const buf = readFileSync(path);
  for (let i = 0; i < MAGIC.length; i++) {
    if (buf[i] !== MAGIC[i]) throw new Error(`bad npy magic at ${path}`);
  }
  const major = buf[6];
  if (major !== 1) throw new Error(`unsupported npy version ${major}`);
  const headerLen = buf.readUInt16LE(8);
  const header = buf.subarray(10, 10 + headerLen).toString("ascii");
  const m = header.match(
    /'descr': '([^']+)'.+?'fortran_order': (True|False).+?'shape': \(([^)]*)\)/,
  );
  if (!m) throw new Error(`npy header parse failed: ${header}`);
  const descr = m[1] as string;
  const fortran = m[2] === "True";
  const shape = (m[3] as string)
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length)
    .map(Number);
  if (descr !== "<f4" && descr !== "|f4") {
    throw new Error(`unsupported dtype ${descr} (only float32)`);
  }
  if (fortran) throw new Error("fortran order npy not supported");
  const dataOffset = 10 + headerLen;
  const total = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(buf.buffer, buf.byteOffset + dataOffset, total).slice();
  return { data, shape, dtype: descr };
}
