# Adding a New Projection

This guide walks through adding a new projection to vibeProj. Every
projection needs both an xp (NumPy/CuPy element-wise) implementation
and a fused CUDA kernel.

## Step 1: Create the projection class

Create `src/vibeproj/projections/<name>.py`:

```python
"""<Name> projection."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams


class MyProjection(Projection):
    name = "myprojection"

    def setup(self, params: ProjectionParams) -> dict:
        """Compute derived constants from projection parameters.

        Called once at construction time. The returned dict is passed
        to forward() and inverse() as the `computed` argument.
        """
        return {
            "a": params.ellipsoid.a,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
            # Add projection-specific derived constants here
        }

    def forward(self, lam, phi, params, computed, xp):
        """Geographic -> Projected.

        lam: longitude relative to central meridian (radians), array
        phi: latitude (radians), array
        xp: numpy or cupy module

        Returns (x, y) in normalised units (pipeline multiplies by `a`
        and adds false easting/northing).
        """
        x = lam  # replace with actual math
        y = phi
        return x, y

    def inverse(self, x, y, params, computed, xp):
        """Projected -> Geographic.

        x, y: normalised projected coordinates (pipeline has already
              removed false easting/northing and divided by `a`).

        Returns (lam, phi) in radians, lam relative to central meridian.
        """
        lam = x  # replace with actual math
        phi = y
        return lam, phi


register("myprojection", MyProjection())
```

Key points:

- `forward()` and `inverse()` receive `xp` (numpy or cupy) and must use
  it for all array operations -- never import numpy directly.
- The pipeline handles deg/rad conversion, central meridian, scale, and
  false easting/northing. Your projection math should not do these.
- `setup()` receives `ProjectionParams` and returns a dict. Put all
  derived constants here (trig of standard parallels, series coefficients,
  etc.) to avoid recomputing them per-transform.

## Step 2: Register the import

Add the import to `src/vibeproj/projections/__init__.py`:

```python
from vibeproj.projections import (
    # ... existing imports ...
    my_projection,
)
```

## Step 3: Add pyproj method mapping

In `src/vibeproj/crs.py`, add the pyproj method name to `_METHOD_MAP`:

```python
_METHOD_MAP = {
    # ... existing entries ...
    "My Projection Method Name": "myprojection",
}
```

Find the exact method name by checking pyproj:

```python
from pyproj import CRS
crs = CRS.from_epsg(XXXX)
print(crs.coordinate_operation.method_name)
```

If your projection has no EPSG code, skip this step.

## Step 4: Add the fused CUDA kernel

In `src/vibeproj/fused_kernels.py`:

1. Write the kernel source strings using the preamble/postamble macros:

```python
_MY_FORWARD_SOURCE = _FWD_SIGNATURE.format(
    func="my_forward", real_t="{real_t}"
) + """
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{""" + _FWD_PREAMBLE + """
    // Your projection math here using phi and lam (radians)
    {real_t} easting  = lam * a + x0;   // replace with actual math
    {real_t} northing = phi * a + y0;
""" + _FWD_POSTAMBLE + "}}"
```

2. Add to `_SOURCE_MAP`:

```python
_SOURCE_MAP = {
    # ... existing entries ...
    ("myprojection", "forward"): (_MY_FORWARD_SOURCE, "my_forward"),
    ("myprojection", "inverse"): (_MY_INVERSE_SOURCE, "my_inverse"),
}
```

3. Add to `_SUPPORTED`:

```python
_SUPPORTED = {
    # ... existing entries ...
    ("myprojection", "forward"), ("myprojection", "inverse"),
}
```

4. Add argument packing in `fused_transform()`:

```python
elif projection_name == "myprojection":
    args = base + (
        real_t(computed["lam0"]),
        real_t(computed["a"]),
        real_t(computed["x0"]),
        real_t(computed["y0"]),
        snf, dnf, nn,
    )
```

### Kernel conventions

- `{real_t}` is substituted with `double` or `float` at compile time.
- `{pi}` is substituted with the appropriate pi literal.
- `_FWD_PREAMBLE` handles: thread index, bounds check, input read,
  source axis swap, deg-to-rad, central meridian subtraction. After the
  preamble, `phi` and `lam` are available in radians.
- `_FWD_POSTAMBLE` handles: destination axis swap and output write.
  Before the postamble, set `easting` and `northing`.
- I/O is always `double*` regardless of compute precision.
- Use `fmin`/`fmax`/`fabs` (not `min`/`max`/`abs`) for CUDA compatibility.

## Step 5: Add tests

### In `tests/test_transformer.py` (CPU path)

If the projection has an EPSG code, add a forward test against pyproj
and a roundtrip test:

```python
def test_myprojection_forward():
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:XXXX")
    t = Transformer.from_crs("EPSG:4326", "EPSG:XXXX")
    lat, lon = np.array([40.0]), np.array([-74.0])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)
    assert_allclose(vp_x, exp_x, atol=0.01)
    assert_allclose(vp_y, exp_y, atol=0.01)

def test_myprojection_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:XXXX")
    lat, lon = 40.0, -74.0
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")
    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)
```

If there's no EPSG code, use the `TransformPipeline` pattern for roundtrip.

### In `tests/test_fused_kernels.py` (GPU path)

Add forward GPU-vs-CPU comparison and GPU roundtrip tests.

## Checklist

- [ ] Projection class in `src/vibeproj/projections/<name>.py`
- [ ] `register()` call at module bottom
- [ ] Import in `src/vibeproj/projections/__init__.py`
- [ ] `_METHOD_MAP` entry in `src/vibeproj/crs.py` (if EPSG exists)
- [ ] Forward + inverse kernel sources in `fused_kernels.py`
- [ ] Added to `_SOURCE_MAP`, `_SUPPORTED`, and `fused_transform()` arg packing
- [ ] CPU tests in `test_transformer.py`
- [ ] GPU tests in `test_fused_kernels.py`
