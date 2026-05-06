import * as fc from "fast-check";
import { describe, expect, it } from "vitest";
import { filterE2E } from "../../src/core/filterE2E";

const rowArb = fc.tuple(
  fc.float({ min: 0, max: 1000, noNaN: true }),
  fc.float({ min: 0, max: 1000, noNaN: true }),
  fc.float({ min: 1, max: 200, noNaN: true }),
  fc.float({ min: 1, max: 200, noNaN: true }),
  fc.float({ min: 0, max: 1, noNaN: true }),
  fc.integer({ min: 0, max: 79 }),
);

describe("filterE2E property tests", () => {
  it("filtered count == sum(scores >= conf) when no class/area filter", () => {
    fc.assert(
      fc.property(
        fc.array(rowArb, { minLength: 0, maxLength: 200 }),
        fc.float({ min: 0, max: 1, noNaN: true }),
        (rows, conf) => {
          const flat = new Float32Array(rows.length * 6);
          let above = 0;
          rows.forEach(([x1, y1, dw, dh, c, cls], i) => {
            flat[i * 6] = x1;
            flat[i * 6 + 1] = y1;
            flat[i * 6 + 2] = x1 + dw;
            flat[i * 6 + 3] = y1 + dh;
            flat[i * 6 + 4] = c;
            flat[i * 6 + 5] = cls;
            if (c >= conf) above++;
          });
          const out = filterE2E(flat, [1, rows.length, 6], { conf, format: "arrays" }) as {
            scores: Float32Array;
          };
          expect(out.scores.length).toBe(above);
        },
      ),
      { numRuns: 200 },
    );
  });

  it("output sorted descending by score", () => {
    fc.assert(
      fc.property(fc.array(rowArb, { minLength: 0, maxLength: 200 }), (rows) => {
        const flat = new Float32Array(rows.length * 6);
        rows.forEach(([x1, y1, dw, dh, c, cls], i) => {
          flat[i * 6] = x1;
          flat[i * 6 + 1] = y1;
          flat[i * 6 + 2] = x1 + dw;
          flat[i * 6 + 3] = y1 + dh;
          flat[i * 6 + 4] = c;
          flat[i * 6 + 5] = cls;
        });
        const out = filterE2E(flat, [1, rows.length, 6], { conf: 0, format: "arrays" }) as {
          scores: Float32Array;
        };
        for (let i = 1; i < out.scores.length; i++) {
          expect((out.scores[i - 1] as number) >= (out.scores[i] as number)).toBe(true);
        }
      }),
      { numRuns: 100 },
    );
  });
});
