import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { decodeDetect } from "../../src/core/decodeRaw";
import { filterE2E } from "../../src/core/filterE2E";
import { readNpy } from "../helpers/npyParser";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const ROOT = path.resolve(__dirname, "../../..");
const FIXTURES_DIR = path.join(ROOT, "fixtures", "v1");

const fixtures = existsSync(FIXTURES_DIR) ? readdirSync(FIXTURES_DIR) : [];

interface Det {
  box: [number, number, number, number];
  score: number;
  class: number;
  label: string;
}

describe.each(fixtures)("fixture %s", (name) => {
  it("matches expected.json", () => {
    const dir = path.join(FIXTURES_DIR, name);
    const raw = readNpy(path.join(dir, "raw_output.npy"));
    const expected = JSON.parse(readFileSync(path.join(dir, "expected.json"), "utf-8")) as Det[];
    const meta = JSON.parse(readFileSync(path.join(dir, "meta.json"), "utf-8")) as { api: string };

    const actual =
      meta.api === "filter_e2e"
        ? (filterE2E(raw.data, raw.shape, { conf: 0.25 }) as Det[])
        : (decodeDetect(raw.data, raw.shape, { conf: 0.25, numClasses: 80 }) as Det[]);

    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < actual.length; i++) {
      const a = actual[i] as Det;
      const e = expected[i] as Det;
      expect(a.class).toBe(e.class);
      expect(Math.abs(a.score - e.score)).toBeLessThanOrEqual(1e-4);
      for (let ci = 0; ci < 4; ci++) {
        expect(Math.abs((a.box[ci] as number) - (e.box[ci] as number))).toBeLessThanOrEqual(1.0);
      }
    }
  });
});
