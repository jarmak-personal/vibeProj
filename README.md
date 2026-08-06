# vibeProj

GPU-accelerated coordinate projection library. Extracted from [RAPIDS cuProj](https://github.com/rapidsai/cuspatial), re-engineered as a pure Python + CuPy package, and expanded from 1 to 24 projections. Each mathematical projection, Helmert, or SVD stage uses one fused NVRTC kernel launch; complete multi-stage transforms use one launch per stage.

## Performance

On an RTX 4090 vs i9-13900k, 1M coordinates:
(Note: datacenter GPUs will see far higher speedups due to better double precision performance)

| Projection | GPU | vs CPU |
|---|---|---|
| Transverse Mercator / UTM | 0.49 ms | 281x |
| Lambert Conformal Conic | 0.54 ms | 96x |
| Albers Equal Area | 0.27 ms | 143x |
| Web Mercator | 0.15 ms | 126x |
| Equal Earth | 0.43 ms | 145x |
| Plate Carrée | 0.04 ms | 311x |
| Oblique Mercator (Hotine) | 0.76 ms | 115x |
| Krovak | 2.08 ms | 173x |

All 24 projections run in under 3 ms at 1M coordinates. See full benchmark in the repo.

## Supported Projections

| Projection | Internal Name | EPSG Examples |
|---|---|---|
| Transverse Mercator / UTM | `tmerc` | 32601–32760, 27700 |
| Web Mercator | `webmerc` | 3857 |
| Mercator (ellipsoidal) | `merc` | 3395 |
| Lambert Conformal Conic | `lcc` | 2154 |
| Albers Equal Area | `aea` | 5070 |
| Polar Stereographic | `stere` | 3031, 3413 |
| Lambert Azimuthal Equal Area | `laea` | 3035 |
| Oblique Stereographic | `sterea` | 28992 |
| Plate Carrée | `eqc` | 4087 |
| Sinusoidal | `sinu` | — |
| Equal Earth | `eqearth` | 8857 |
| Cylindrical Equal Area | `cea` | 6933 |
| Orthographic | `ortho` | — |
| Gnomonic | `gnom` | — |
| Mollweide | `moll` | — |
| Robinson | `robin` | — |
| Winkel Tripel | `wintri` | — |
| Natural Earth | `natearth` | — |
| Azimuthal Equidistant (spherical `+R`) | `aeqd` | — |
| Geostationary Satellite | `geos` | — |
| Oblique Mercator (Hotine) | `omerc` | 3375 |
| Krovak | `krovak` | 5513, 5514 |
| Eckert IV | `eck4` | — |
| Eckert VI | `eck6` | — |

Azimuthal Equidistant support is spherical only. Ellipsoidal, Modified, and
Guam methods raise `UnsupportedProjectionError` instead of silently using the
spherical equations. Geostationary CRS definitions preserve Sweep X/Y and
satellite height; points beyond the visible limb or inverse Earth-intersection
disk return non-finite sentinels.

## Install

```bash
pip install vibeproj            # CPU-only (NumPy fallback)
pip install vibeproj[cu12]      # CUDA 12
pip install vibeproj[cu13]      # CUDA 13
```

For development:

```bash
uv sync                         # CPU-only
uv sync --extra cu12            # CUDA 12
uv sync --extra cu13            # CUDA 13
```

## Usage

```python
from vibeproj import Transformer

# Default: always_xy=True — (lon, lat) order, matches shapely/geopandas
t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
x, y = t.transform(2.0, 49.0)           # (lon, lat) in, (easting, northing) out

# always_xy=False: native CRS axis order (matches pyproj default)
t = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=False)
x, y = t.transform(49.0, 2.0)           # (lat, lon) in, (easting, northing) out
```

Compute precision and transcendental implementation are independent per-call
choices:

```python
t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
x, y = t.transform(
    2.0,
    49.0,
    precision="fp64",
    transcendentals="accelerated",
)
print(t.explain_strategy(transcendentals="accelerated"))
```

`transcendentals="accelerated"` uses an accuracy-qualified implementation when
the transform and GPU are supported, and explicitly falls back to native math
otherwise. Automatic acceleration currently covers bounded Helmert, forward
UTM, spherical-sinusoidal-forward, orthographic-forward, and spherical-equatorial
orthographic-inverse, GEOS-forward (sphere/ellipsoid and sweep x/y), and
spherical-polar LAEA-forward, ellipsoidal Polar Stereographic inverse,
spherical regular-Mercator forward, and regular-Mercator inverse
operations on validated Ada `sm_89` consumer GPUs
above their measured workload-size crossovers. GEOS forward starts at 2,097,152
coordinates, spherical-polar LAEA forward at 1,048,576, Polar Stereographic
inverse at 1,000,000, spherical Mercator forward at 262,144, and Mercator inverse at
65,536; the canonical policy matrix lists every threshold. Hopper and
unmeasured devices remain native. A selected
accelerated kernel may still take its warp-atomic native branch when any lane
is out of domain or the physical scale is unsupported. See the
[transcendental policy](https://jarmak-personal.github.io/vibeProj/user/transcendentals.html).

Spherical Gnomonic inverse also has an Ada-qualified bounded reframe for expert
`transcendentals="accelerated"` opt-in. It is intentionally excluded from `"auto"`:
mixed in/out-of-guard radius distributions can regress, and the host planner cannot
inspect coordinate values before dispatch.

Ellipsoidal regular-Mercator forward is likewise an explicit accelerated
option. Its polar cap falls back exactly to native math, but mixed polar-cap
workloads can regress, so automatic policy remains native for that geometry.

### Cross-datum transforms (Helmert)

```python
# Cross-datum: Helmert 7/15-parameter shift applied automatically
t = Transformer.from_crs("EPSG:4326", "EPSG:27700")  # WGS84 → OSGB36
x, y = t.transform(-0.1278, 51.5074)

# With ellipsoidal height — z is transformed through the ECEF intermediate
x, y, z = t.transform(-0.1278, 51.5074, z=45.0)

# Same-datum: z passes through unchanged
t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
x, y, z = t.transform(2.0, 49.0, z=45.0)  # z == 45.0
```

For datum pairs with baked SVD corrections (e.g. NAD27 to NAD83), vibeProj achieves sub-5cm accuracy without external grid files. You can also [generate your own SVD corrections](https://jarmak-personal.github.io/vibeProj/user/datum-corrections.html) for any datum pair that pyproj supports using the included fitting tool (`tools/fit_datum_corrections.py`). For other grid-only datum pairs without a baked or custom correction, vibeProj warns and proceeds without a datum shift.

### Integration with CPU libraries

vibeProj works with popular geospatial Python libraries. GPU acceleration is automatic when CuPy is installed; otherwise it falls back to NumPy transparently.

- [GeoPandas](https://jarmak-personal.github.io/vibeProj/recipes/geopandas.html) — bulk GeoDataFrame reprojection
- [Rasterio](https://jarmak-personal.github.io/vibeProj/recipes/rasterio.html) — GPU-accelerated raster coordinate grids
- [Shapely](https://jarmak-personal.github.io/vibeProj/recipes/shapely.html) — geometry transforms via `shapely.transform()`

### vibeSpatial Integration (zero-copy GPU)

```python
# Pre-allocated final output; all stages stay on the GPU
t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
new_x = cp.empty_like(buf.x)
new_y = cp.empty_like(buf.y)
t.transform_buffers(buf.x, buf.y, out_x=new_x, out_y=new_y)

# 3D: z is transformed through Helmert when crossing datums
new_z = cp.empty_like(buf.z)
t.transform_buffers(buf.x, buf.y, buf.z, out_x=new_x, out_y=new_y, out_z=new_z)
```

`transform_buffers()` accepts pre-allocated CuPy output arrays, writes results directly into them, and returns the same objects. No stage performs a host round-trip. A projection-only call needs no intermediate buffers; multi-stage Helmert/SVD pipelines lazily allocate bounded scratch and reuse it on warmed calls. Designed for vibeSpatial's `OwnedGeometryArray` coordinate buffers.

## Architecture

- **Pure Python + CuPy** — no compiled extensions, no CMake
- **Fused NVRTC kernels** — each mathematical stage runs in one CUDA launch via CuPy `RawKernel`; Helmert/SVD plus projection paths use the corresponding two or three stage launches
- **NumPy fallback** — all projections work on CPU when CuPy is unavailable
- **Helmert datum shifts** — 7/15-parameter (time-dependent) datum transformation with 3D ellipsoidal height support, runs on its own GPU kernel
- **SVD datum corrections** — baked SVD-compressed grid corrections for sub-5cm accuracy on supported datum pairs (e.g. NAD27 to NAD83), no external grid files needed
- **pyproj for CRS metadata** — EPSG codes resolved via pyproj, transform math is ours
- **fp64 I/O** — input/output arrays always double precision (ADR-0002 compliant)
- **Inspectable transcendental policy** — a central registry resolves
  `auto`/`native`/`accelerated` from hardware, projection domain, direction,
  precision, workload size, and accuracy qualification, with observable host
  fallback and guarded native execution inside qualified kernels

## Test

```bash
uv run pytest                    # all tests
uv run pytest tests/test_fused_kernels.py  # GPU kernel tests (requires CuPy)
```
