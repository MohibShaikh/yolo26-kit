export function normalizeOutput(output: Float32Array | Float64Array): Float32Array {
  if (output instanceof Float32Array) return output;
  const out = new Float32Array(output.length);
  for (let i = 0; i < output.length; i++) {
    out[i] = output[i] as number;
  }
  return out;
}
