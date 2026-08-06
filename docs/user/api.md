# API Reference

:::{tip}
Full auto-generated API docs for every module, class, and function are
available in the [API Reference](/api/index) section (powered by sphinx-autoapi).
:::

## Transformer

The primary user-facing class.

### `Transformer.from_crs(crs_from, crs_to, *, always_xy=True, datum_shift="accurate", epoch=None)`

Create a transformer between two coordinate reference systems.

**Parameters:**

- `crs_from` -- Source CRS. Accepts:
  - EPSG integer: `4326`
  - Authority string: `"EPSG:4326"`
  - Tuple: `("EPSG", 4326)`
  - pyproj `CRS` object
- `crs_to` -- Target CRS. Same formats as `crs_from`.
- `always_xy` -- If `True` (default), geographic CRS coordinates use longitude, latitude order. If `False`, use native CRS axis order.
- `datum_shift` -- `"accurate"` (default) uses time-dependent Helmert parameters when available; `"fast"` uses base 7-parameter Helmert parameters.
- `epoch` -- Optional decimal year for time-dependent Helmert transforms.

**Returns:** `Transformer` instance.

```python
t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
```

### `Transformer.transform(x, y, z=None, direction="FORWARD", *, precision="auto", transcendentals="auto")`

Transform coordinates.

**Parameters:**

- `x`, `y` -- Input coordinates. Accepts scalars, lists, NumPy arrays, or CuPy arrays. By default, `always_xy=True`, so geographic CRS input/output is `x` = longitude, `y` = latitude. Pass `always_xy=False` to use native CRS axis order.
- `z` -- Optional ellipsoidal height in meters. When a Helmert datum shift is active (cross-datum transform), z is transformed through the ECEF intermediate. When no datum shift is needed, z is passed through unchanged.
- `direction` -- `"FORWARD"` or `"INVERSE"`.
- `precision` -- Compute precision: `"auto"`, `"fp64"`, `"fp32"`, or `"ds"`.
- `transcendentals` -- Special-function policy: `"auto"`, `"native"`, or
  `"accelerated"`. This is a per-call choice and is not stored on the
  transformer. `"auto"` uses the concrete input size and device capability
  when applying qualified acceleration crossover thresholds.

**Returns:** Tuple `(x_out, y_out)` or `(x_out, y_out, z_out)` if z was provided. Scalars if input was scalar, arrays otherwise.

```python
# Forward: geographic -> projected
easting, northing = t.transform(2.0, 49.0)

# Inverse: projected -> geographic
lon, lat = t.transform(easting, northing, direction="INVERSE")

# Array input
x, y = t.transform(lon_array, lat_array)

# With ellipsoidal height (cross-datum: z is transformed; same-datum: z passthrough)
x, y, z_out = t.transform(lon, lat, z=45.0)
```

### `Transformer.transform_buffers(x, y, z=None, *, direction="FORWARD", out_x=None, out_y=None, out_z=None, precision="auto", transcendentals="auto", stream=None)`

Transform device-resident arrays with optional pre-allocated outputs. This path
skips scalar detection and dtype conversion. With pre-allocated outputs,
repeated same-size GPU calls on a warmed, cached stream avoid output and
correction-scratch allocation; first use, cache growth, and kernel compilation
can allocate.

**Parameters:**

- `x`, `y` -- Input coordinate arrays (fp64).
- `z` -- Optional ellipsoidal height array. Transformed through Helmert when a datum shift is active; passed through unchanged otherwise.
- `direction` -- `"FORWARD"` or `"INVERSE"`.
- `out_x`, `out_y` -- Optional pre-allocated output arrays. When provided, results are written directly into these arrays and the same objects are returned.
- `out_z` -- Optional pre-allocated output height array. When `z` is provided,
  this buffer is written and returned identically whether height is transformed
  by Helmert or passed through another pipeline.
- `precision` -- Compute precision: `"auto"`, `"fp64"`, `"fp32"`, or `"ds"`.
- `transcendentals` -- Special-function policy: `"auto"`, `"native"`, or
  `"accelerated"`, independent of `precision`. Unsupported accelerated
  combinations explicitly use native fallback. `"auto"` applies workload-size
  crossover thresholds.
- `stream` -- Optional CuPy CUDA stream. `None` uses the current stream on the
  input array's device; the null stream is supported, cross-device streams are
  rejected, and the caller owns synchronization.

**Returns:** Tuple `(out_x, out_y)` or `(out_x, out_y, z_out)` if z was provided.

```python
out_x = cp.empty(n, dtype=cp.float64)
out_y = cp.empty(n, dtype=cp.float64)
rx, ry = t.transform_buffers(lon, lat, out_x=out_x, out_y=out_y)
assert rx is out_x  # supplied output object is returned identically

# With height (cross-datum transforms)
out_z = cp.empty(n, dtype=cp.float64)
rx, ry, rz = t.transform_buffers(lon, lat, h, out_x=out_x, out_y=out_y, out_z=out_z)
```

### `Transformer.transform_chunked(x, y, z=None, *, direction="FORWARD", chunk_size=1000000, precision="auto", transcendentals="auto")`

Transform host arrays through a double-buffered GPU pipeline. Each Transformer
keeps a grow-only workspace per CUDA device: two non-blocking streams plus
pinned-host and device staging slots persist across calls. Calls that share the
same Transformer and CUDA device serialize around that workspace, while the two
streams still overlap transfers and compute within each call. Different
Transformer instances or CUDA devices do not share the lock.

The workspace and any correction scratch allocate lazily on first use or growth
and are reused for subsequent same-size calls. `precision` accepts `"auto"`,
`"fp64"`, `"fp32"`, or `"ds"`; `transcendentals` accepts `"auto"`, `"native"`,
or `"accelerated"`. Size-aware automatic dispatch uses the planned chunk-buffer
size once and reuses that decision for every chunk, including a smaller final
chunk.

For `transform_buffers()`, `stream=None` means the current stream on the input
array's device, and CuPy's legacy null stream is supported. An explicit stream
from a different device is rejected.

### `Transformer.compile(*, precision="auto", transcendentals="auto")`

Pre-compile this transformer's fused kernels. The two policy keywords use the
same resolver as transform calls; compilation does not persist either choice
on the transformer. `precision` accepts `"auto"`, `"fp64"`, `"fp32"`, or `"ds"`.
Because compilation has no concrete workload, `transcendentals="auto"` selects
an otherwise-qualified implementation without a runtime crossover threshold.
Compilation does not allocate per-Transformer stream scratch or chunk staging
workspaces; those remain lazy.

### `Transformer.transform_bounds(left, bottom, right, top, *, densify_pts=21, direction="FORWARD", precision="auto", transcendentals="auto")`

Transform a densified bounding-box perimeter and return its output envelope.
`precision` accepts `"auto"`, `"fp64"`, `"fp32"`, or `"ds"`;
`transcendentals` accepts `"auto"`, `"native"`, or `"accelerated"` independently.
Size-aware automatic dispatch uses the total densified point count.

### `vibeproj.warm_up(projections=None, *, precision="auto", transcendentals="auto")`

Pre-compile selected fused projection kernels. It accepts the same four
precision values and three transcendental policies as `Transformer.compile()`.
Warm-up has no concrete workload size, so automatic policy selects an otherwise-
qualified implementation without applying runtime crossover thresholds. It
does not allocate Transformer-owned correction scratch or chunk workspaces.

### `Transformer.explain_strategy(*, transcendentals="auto", precision="auto", direction="FORWARD", device=None, workload_size=None)`

Return an immutable explanation containing one decision for every projection
stage and any Helmert stage. Each decision reports its stable implementation
ID, operation/family, device and domain, workload size, accuracy contract,
reason, and whether an explicit accelerated request fell back to native. Pass
`workload_size` to preview size-aware automatic selection. Passing a
`DeviceCapability` makes hardware-policy checks deterministic without querying
a GPU.

```python
explanation = t.explain_strategy(transcendentals="accelerated")
for decision in explanation.decisions:
    print(decision.implementation_id, decision.fallback, decision.reason)
```

### `vibeproj.list_transcendental_strategies()`

Return the immutable central registry. Registry entries describe supported
policies, backends, compute capabilities, compute precisions, domains, accuracy
contracts, exact `min_elements` crossover thresholds, and native fallback
behavior.

See [Transcendental policy](transcendentals.md) for the qualified coverage
matrix and hardware behavior.

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
    lat_1=None,                # first standard parallel; None means omitted
    lat_2=None,                # second standard parallel; None means omitted
    k_0=1.0,                   # scale factor
    x_0=0.0,                   # false easting (meters)
    y_0=0.0,                   # false northing (meters)
    north_first=False,         # axis order flag
    easting_axis_sign=1.0,     # -1 for a Westing component
    northing_axis_sign=1.0,    # -1 for a Southing component
    visualization_north_first=False,  # projected PROJ X/Y display order
    extra={},                  # projection-specific params
)
```

`lat_1` and `lat_2` distinguish omission from an explicit zero. Conic
projections apply their method-specific fallback only when a value is `None`;
zero remains a valid supplied standard parallel. Winkel Tripel uses
`acos(2/pi)` only when `lat_1` is `None`, while CEA and Equidistant
Cylindrical treat omission as zero. Mercator variant B requires `lat_1`, and
therefore accepts an explicit `lat_1=0.0` but not `None`.

Projected axis signs are independent from component order and unit magnitude.
Resolved unit factors remain positive; Westing and Southing are represented by
the two sign fields. `always_xy=True` follows PROJ visualization order. In
particular, EPSG:5513 retains X=Southing, Y=Westing under both axis settings.

### `TransformPipeline`

```python
from vibeproj.pipeline import TransformPipeline
import numpy as np

src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
dst = ProjectionParams(projection_name="ortho", ellipsoid=WGS84, lon_0=0.0, lat_0=45.0)

pipe = TransformPipeline(src, dst)
x, y = pipe.transform(lat_array, lon_array, np)
```

### `Transformer.accuracy`

Read-only property indicating the accuracy classification of this transform.

- `"sub-millimeter"` -- same datum, projection math only.
- `"sub-5cm"` -- cross-datum with SVD-compressed grid correction (e.g. NAD27 to NAD83).
- `"sub-decimeter"` -- cross-datum with 15-param time-dependent Helmert at a known epoch.
- `"sub-meter"` -- cross-datum with Helmert 7-parameter shift applied (~1--5m).
- `"datum no-op (... m PROJ accuracy)"` -- PROJ's best available datum operation is an explicit no-op with meter-level expected accuracy.
- `"degraded — no datum shift applied"` -- cross-datum, no Helmert or SVD correction available.

```python
t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
print(t.accuracy)  # "sub-millimeter" (same datum)

t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
print(t.accuracy)  # "sub-meter" (Helmert applied)
```

## Utility functions

### `vibeproj.runtime.gpu_available()`

Returns `True` if CuPy is installed and a CUDA GPU is accessible.

### `vibeproj.runtime.get_array_module(x)`

Returns `cupy` if `x` is a CuPy array, `numpy` otherwise.
