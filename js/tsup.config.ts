import { defineConfig } from "tsup";

export default defineConfig([
  { entry: ["src/index.ts"], format: ["esm", "cjs"], dts: true, clean: true, sourcemap: true },
  { entry: { "ort/index": "src/ort/index.ts" }, format: ["esm", "cjs"], dts: true, sourcemap: true },
]);
