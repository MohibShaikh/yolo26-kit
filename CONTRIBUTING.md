# Contributing to yolo26-kit

Thanks for your interest. This is a small, focused library — the project values predictable behavior over feature breadth.

## Project shape

- **Python package** in `python/` — published to PyPI as `yolo26-kit`
- **TypeScript package** in `js/` — published to npm as `yolo26-kit`
- **Spec** in `spec/decode.md` — canonical algorithm definitions, both implementations must match
- **Golden fixtures** in `fixtures/v1/` — bind cross-language equivalence

The Python and TypeScript implementations are intentional siblings. They share `spec/decode.md` as the source of truth and are validated against the same fixtures. Changes to one usually mean changes to the other.

## Setup

### Python

```bash
cd python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### TypeScript

```bash
cd js
npm install
```

## Verification trio (required before every commit)

Both languages have a "trio" of checks that must pass with **0 errors** before a commit lands:

### Python

```bash
.venv/bin/pytest tests/ -v
.venv/bin/mypy src/yolo26_kit
.venv/bin/ruff check src tests
```

### TypeScript

```bash
npm run typecheck
npm run lint
npm test -- --run
```

CI runs the same trio across Python 3.10–3.13 × {ubuntu, macos, windows} and Node 18–22 × {ubuntu, macos, windows}.

## Workflow

1. **Open an issue first** for non-trivial changes so we can align on scope before you write code.
2. **TDD** — write the failing test before the implementation.
3. **Match the spec** — if you're touching algorithm code, update `spec/decode.md` if behavior changes.
4. **Cross-language parity** — if you change Python core behavior, mirror in TypeScript (and vice versa). Fixture tests will catch divergence.
5. **Conventional Commits** — prefix: `feat(py-core)`, `feat(js-core)`, `feat(py-ort)`, `feat(js-ort)`, `fix(py-core)`, `fix(js-core)`, `test(py)`, `test(js)`, `docs`, `chore`, `ci`, `build(python)`, `build(js)`.
6. **No `Co-Authored-By:` trailers** in commits — keep history clean.
7. **One PR = one logical change** — easier to review and revert.

## Adding a new task (seg, pose, cls, OBB)

These are queued for v0.2+. Before starting:

1. Open an issue to discuss the spec section (Algorithm H, I, J, K).
2. Add the algorithm to `spec/decode.md` with explicit shapes, formulas, edge cases.
3. Generate fixtures using `scripts/gen_fixtures.py` for the new task.
4. Implement in `python/src/yolo26_kit/core/` and `js/src/core/` symmetrically.
5. Wire into `Decoder.predict` routing if it touches the wrapper.

## Adding a new export target / NPU port

Keep the core decoder library untouched. New target ports live as separate modules under `npu/` (or similar) that consume `yolo26_kit.decode_detect` / `yolo26_kit.filter_e2e` to handle vendor specifics.

## Reporting bugs

- Include `yolo26-kit` version (Python `yolo26_kit.__version__` or `npm view yolo26-kit version`).
- Include `ultralytics` version if relevant (`pip show ultralytics`).
- Include input ONNX output shape (`session.get_outputs()[0].shape`).
- Minimal reproducible example: a small numpy array or `Float32Array` plus expected vs actual output.

## Licensing

By submitting a contribution you agree it will be licensed under Apache-2.0, the same as the project.
