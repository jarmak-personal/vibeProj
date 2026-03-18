---
description: "Add a new map projection to vibeProj. Guides through all 7 phases: xp implementation, registration, CRS mapping, fused CUDA kernels, arg packing, tests, and best-practice integration (stream support, warm-up). Use when: user says 'add projection', 'new projection', 'implement <projection-name>', or references a projection method not yet in _METHOD_MAP."
---

# Write New Projection

You are adding a new map projection to vibeProj. Follow these phases **in order**. Each phase has acceptance criteria — do not advance until they are met. The pre-commit hook (`scripts/check_projections.py`) enforces PROJ001-PROJ004 consistency.

## Before you start

1. Read the reference projection math (Wikipedia, PROJ docs, Snyder's "Map Projections: A Working Manual")
2. Identify the pyproj method name: `python -c "from pyproj import CRS; print(CRS.from_epsg(XXXX).coordinate_operation.method_name)"`
3. Identify ellipsoid vs sphere (does it need `e`, `es` parameters?)
4. Identify the computed parameters `setup()` will need

---

## Phase 1: xp fallback implementation

Create `src/vibeproj/projections/<name>.py`:

```python
"""<Full Projection Name> projection.

<One-line description and common use cases.>
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams


class <ClassName>(Projection):
    name = "<name>"

    def setup(self, params: ProjectionParams) -> dict:
        # Pre-compute constants from CRS parameters.
        # These are computed ONCE and passed to every forward/inverse call.
        return {
            "a": params.ellipsoid.a,
            "e": params.ellipsoid.e,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
            # ... projection-specific constants
        }

    def forward(self, lam, phi, params, computed, xp):
        # lam = longitude (radians, central meridian already subtracted)
        # phi = latitude (radians)
        # Returns (easting, northing) in NORMALIZED units (divide by a to get meters)
        ...
        return easting, northing

    def inverse(self, x, y, params, computed, xp):
        # x = easting (normalized, already divided by a, false easting removed)
        # y = northing (normalized)
        # Returns (lam, phi) in radians
        ...
        return lam, phi


register("<name>", <ClassName>())
```

**Key conventions:**
- `forward()` input: `lam` has central meridian subtracted, `phi` is raw latitude in radians
- `forward()` output: normalized (pipeline multiplies by `a` and adds `x0`/`y0`)
- `inverse()` input: normalized (pipeline already subtracted `x0`/`y0` and divided by `a`)
- `xp` is the array module (numpy or cupy) — use `xp.sin()`, not `math.sin()`

**Acceptance:** `register()` call at module bottom. Class has `name`, `setup`, `forward`, `inverse`.

---

## Phase 2: Registration + import

1. Import the module in `src/vibeproj/projections/__init__.py`:

```python
from vibeproj.projections import (  # noqa: E402, F401
    ...
    <new_module>,  # add alphabetically
)
```

**Acceptance:** `uv run python -c "from vibeproj.projections import get_projection; print(get_projection('<name>').name)"` prints `<name>`.

---

## Phase 3: CRS mapping (if applicable)

If this projection has standard EPSG codes, add the pyproj method name to `_METHOD_MAP` in `src/vibeproj/crs.py`:

```python
_METHOD_MAP = {
    ...
    "<Pyproj Method Name>": "<name>",
}
```

Find the method name: `CRS.from_epsg(XXXX).coordinate_operation.method_name`

Some projections (Mollweide, Robinson, etc.) have no standard EPSG code — skip this phase.

**Acceptance:** `Transformer.from_crs("EPSG:4326", "EPSG:XXXX")` resolves without error.

---

## Phase 4: Fused CUDA kernel source

Add forward and inverse kernel source strings in `src/vibeproj/fused_kernels.py`.

**Pattern — use the preamble/postamble helpers:**

```python
_<NAME>_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="<name>_forward", real_t="{real_t}")
    + """
    {real_t} <param1>, {real_t} <param2>, ...,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{
"""
    + _FWD_PREAMBLE
    + """
    // === Projection math (compute precision: {real_t}) ===
    {real_t} easting = ...;
    {real_t} northing = ...;

    // Scale + offset (ALWAYS fp64 for sub-meter precision)
    easting  = (double)easting * (double)a + (double)x0;
    northing = (double)northing * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + """
}}
"""
)
```

**Critical rules:**
- Kernel signature: `const double* in_x, const double* in_y, double* out_x, double* out_y` — I/O is ALWAYS fp64 (ADR-0002)
- Use `{real_t}` for compute-precision variables (parameterized by precision mode)
- Scale/offset in the forward kernel MUST be in fp64: `(double)easting * (double)a + (double)x0`
- Use `{{` and `}}` for literal braces (Python `.format()` escaping)
- Inverse kernel uses `_INV_PREAMBLE` / `_INV_POSTAMBLE` which handle offset removal

**Acceptance:** Both source strings compile without syntax errors in the kernel.

---

## Phase 5: Kernel registration + arg packing

Three additions in `fused_kernels.py`:

### 5a. Add to `_SOURCE_MAP`:

```python
_SOURCE_MAP = {
    ...
    ("<name>", "forward"): (_<NAME>_FORWARD_SOURCE, "<name>_forward"),
    ("<name>", "inverse"): (_<NAME>_INVERSE_SOURCE, "<name>_inverse"),
}
```

### 5b. Add to `_SUPPORTED`:

```python
_SUPPORTED = {
    ...
    ("<name>", "forward"),
    ("<name>", "inverse"),
}
```

### 5c. Add arg packing in `fused_transform()`:

```python
    elif projection_name == "<name>":
        args = base + (
            real_t(computed["<param1>"]),
            real_t(computed["<param2>"]),
            # ... projection-specific params
            real_t(computed["lam0"]),
            real_t(computed["a"]),
            real_t(computed["x0"]),
            real_t(computed["y0"]),
            snf,
            dnf,
            nn,
        )
```

**Arg order must match the kernel signature exactly.** Common params (lam0, a, x0, y0, snf, dnf, nn) always come last.

**Acceptance:** `can_fuse("<name>", "forward")` returns `True`.

---

## Phase 6: Tests

Add to `tests/test_fused_kernels.py` (GPU tests) and/or `tests/test_transformer.py` (CPU tests).

### CPU roundtrip test (always add):

```python
def test_<name>_roundtrip():
    from vibeproj.crs import ProjectionParams
    from vibeproj.ellipsoid import WGS84
    from vibeproj.pipeline import TransformPipeline

    params = ProjectionParams(
        projection_name="<name>",
        ellipsoid=WGS84,
        lon_0=0.0,
        lat_0=<appropriate_value>,
        north_first=False,
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    pipe = TransformPipeline(src, params)
    lat = np.array([<test points>])
    lon = np.array([<test points>])
    x, y = pipe.transform(lat, lon, np)
    inv_pipe = TransformPipeline(params, src)
    lat2, lon2 = inv_pipe.transform(x, y, np)
    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)
```

### If EPSG code exists, add pyproj validation test:

```python
def test_<name>_forward():
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:XXXX")
    t = Transformer.from_crs("EPSG:4326", "EPSG:XXXX", always_xy=False)
    lat, lon = np.array([...]), np.array([...])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)
    assert_allclose(vp_x, exp_x, atol=0.01)
    assert_allclose(vp_y, exp_y, atol=0.01)
```

### GPU fused kernel test (add to test_fused_kernels.py):

Tests are parametrized by grid corners — add the projection to the existing parametrize lists if it uses an EPSG code.

**Acceptance:** `uv run pytest tests/ -k <name>` passes.

---

## Phase 7: Best-practice integration

### 7a. Add to `warm_up()` / `compile()` coverage

No code needed — `warm_up()` and `Transformer.compile()` already discover all projections in `_SUPPORTED` automatically.

Verify: `uv run python -c "import vibeproj; vibeproj.warm_up(['<name>'])"` works (requires CuPy).

### 7b. Stream support

The `stream=` parameter already flows through `transform_buffers()` → `pipeline.transform()` → `_try_fused()` → `fused_transform()` → kernel launch. No per-projection work needed.

### 7c. Run the pre-commit checks

```bash
uv run python scripts/check_projections.py --all
```

This verifies PROJ001-PROJ004 all pass. If any fail, you missed a step above.

### 7d. Run the full test suite

```bash
uv run pytest tests/ -v
```

---

## Checklist summary

- [ ] Phase 1: `projections/<name>.py` with `setup()`, `forward()`, `inverse()`, `register()`
- [ ] Phase 2: Import in `projections/__init__.py`
- [ ] Phase 3: `_METHOD_MAP` entry in `crs.py` (if EPSG code exists)
- [ ] Phase 4: Forward + inverse CUDA kernel source strings
- [ ] Phase 5a: `_SOURCE_MAP` entries
- [ ] Phase 5b: `_SUPPORTED` entries
- [ ] Phase 5c: `fused_transform()` arg packing
- [ ] Phase 6: Tests (CPU roundtrip + pyproj validation + GPU fused)
- [ ] Phase 7: `check_projections.py --all` passes, `pytest` passes
