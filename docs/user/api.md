# API Reference

## Transformer

The primary user-facing class.

### `Transformer.from_crs(crs_from, crs_to)`

Create a transformer between two coordinate reference systems.

**Parameters:**

- `crs_from` -- Source CRS. Accepts:
  - EPSG integer: `4326`
  - Authority string: `"EPSG:4326"`
  - Tuple: `("EPSG", 4326)`
  - pyproj `CRS` object
- `crs_to` -- Target CRS. Same formats as `crs_from`.

**Returns:** `Transformer` instance.

```python
t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
```

### `Transformer.transform(x, y, direction="FORWARD")`

Transform coordinates.

**Parameters:**

- `x`, `y` -- Input coordinates. Accepts scalars, lists, NumPy arrays, or CuPy arrays. For geographic CRS, `x` = latitude, `y` = longitude (pyproj convention).
- `direction` -- `"FORWARD"` or `"INVERSE"`.

**Returns:** Tuple `(x_out, y_out)`. Scalars if input was scalar, arrays otherwise.

```python
# Forward: geographic -> projected
easting, northing = t.transform(49.0, 2.0)

# Inverse: projected -> geographic
lat, lon = t.transform(easting, northing, direction="INVERSE")

# Array input
x, y = t.transform(lat_array, lon_array)
```

### `Transformer.transform_buffers(x, y, *, direction="FORWARD", out_x=None, out_y=None, precision="auto")`

Zero-copy transform for device-resident arrays. Skips scalar detection
and dtype conversion for maximum throughput.

**Parameters:**

- `x`, `y` -- Input coordinate arrays (fp64).
- `direction` -- `"FORWARD"` or `"INVERSE"`.
- `out_x`, `out_y` -- Optional pre-allocated output arrays. When provided, results are written directly into these arrays and the same objects are returned.
- `precision` -- Compute precision: `"auto"`, `"fp64"`, `"fp32"`, or `"ds"`.

**Returns:** Tuple `(out_x, out_y)`.

```python
out_x = cp.empty(n, dtype=cp.float64)
out_y = cp.empty(n, dtype=cp.float64)
rx, ry = t.transform_buffers(lat, lon, out_x=out_x, out_y=out_y)
assert rx is out_x  # same object, no allocation
```

## Pipeline API

For projections without EPSG codes, use `TransformPipeline` directly.

### `ProjectionParams`

Dataclass holding projection parameters:

```python
from vibeproj.crs import ProjectionParams
from vibeproj.ellipsoid import WGS84

params = ProjectionParams(
    projection_name="ortho",   # internal projection name
    ellipsoid=WGS84,           # reference ellipsoid
    lon_0=0.0,                 # central meridian (degrees)
    lat_0=45.0,                # latitude of origin (degrees)
    lat_1=0.0,                 # first standard parallel
    lat_2=0.0,                 # second standard parallel
    k_0=1.0,                   # scale factor
    x_0=0.0,                   # false easting (meters)
    y_0=0.0,                   # false northing (meters)
    north_first=False,         # axis order flag
    extra={},                  # projection-specific params
)
```

### `TransformPipeline`

```python
from vibeproj.pipeline import TransformPipeline
import numpy as np

src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
dst = ProjectionParams(projection_name="ortho", ellipsoid=WGS84, lon_0=0.0, lat_0=45.0)

pipe = TransformPipeline(src, dst)
x, y = pipe.transform(lat_array, lon_array, np)
```

## Utility functions

### `vibeproj.runtime.gpu_available()`

Returns `True` if CuPy is installed and a CUDA GPU is accessible.

### `vibeproj.runtime.get_array_module(x)`

Returns `cupy` if `x` is a CuPy array, `numpy` otherwise.
