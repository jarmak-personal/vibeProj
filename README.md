# vibeProj

GPU-accelerated coordinate projection library. Extracted from [RAPIDS cuProj](https://github.com/rapidsai/cuspatial), re-engineered as a pure Python + CuPy package, and expanded from 1 to 24 projections — each with a fused NVRTC kernel that runs the full transform pipeline in a single GPU kernel launch.

> [!WARNING]
> vibeProj is very early in development. Operations may be unoptimized or have multiple Host/Device transfers causing reduced performance. [File an issue](https://github.com/jarmak-personal/vibeProj/issues) if you hit a problem!

## Performance

On an RTX 4090 vs i9-13900k, 1M coordinates:
(Note: datacenter GPUs will see far higher speedups due to better double precision performance)

| Projection | GPU | vs CPU |
|---|---|---|
| Transverse Mercator / UTM | 0.49 ms | 284x |
| Lambert Conformal Conic | 0.53 ms | 96x |
| Albers Equal Area | 0.27 ms | 137x |
| Web Mercator | 0.15 ms | 124x |
| Equal Earth | 0.43 ms | 148x |
| Plate Carrée | 0.04 ms | 313x |
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
| Azimuthal Equidistant | `aeqd` | — |
| Geostationary Satellite | `geos` | — |
| Oblique Mercator (Hotine) | `omerc` | 3375 |
| Krovak | `krovak` | 5514 |
| Eckert IV | `eck4` | — |
| Eckert VI | `eck6` | — |

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

### Cross-datum transforms (Helmert)

```python
# Cross-datum: Helmert 7/15-parameter shift applied automatically
t = Transformer.from_crs("EPSG:4326", "EPSG:27700")  # WGS84 → OSGB36
x, y = t.transform(-0.1278, 51.5074)

# With ellipsoidal height — z is transformed through the ECEF intermediate
x, y, z = t.transform(-0.1278, 51.5074, z=45.0)

# Same-datum: z passes through unchanged, zero overhead
t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
x, y, z = t.transform(2.0, 49.0, z=45.0)  # z == 45.0
```

### vibeSpatial Integration (zero-copy GPU)

```python
# Pre-allocated output, no intermediate allocations, stays on GPU
t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
new_x = cp.empty_like(buf.x)
new_y = cp.empty_like(buf.y)
t.transform_buffers(buf.x, buf.y, out_x=new_x, out_y=new_y)

# 3D: z is transformed through Helmert when crossing datums
new_z = cp.empty_like(buf.z)
t.transform_buffers(buf.x, buf.y, buf.z, out_x=new_x, out_y=new_y, out_z=new_z)
```

`transform_buffers()` accepts pre-allocated CuPy output arrays, writes results directly into them, and returns the same objects. No host round-trip, no intermediate allocation. Designed for vibeSpatial's `OwnedGeometryArray` coordinate buffers.

## Architecture

- **Pure Python + CuPy** — no compiled extensions, no CMake
- **Fused NVRTC kernels** — each projection's full pipeline (axis swap, deg/rad, central meridian, projection math, scale/offset) runs in a single CUDA kernel launch via CuPy `RawKernel`
- **NumPy fallback** — all projections work on CPU when CuPy is unavailable
- **Helmert datum shifts** — 7/15-parameter (time-dependent) datum transformation with 3D ellipsoidal height support, runs on its own GPU kernel
- **pyproj for CRS metadata** — EPSG codes resolved via pyproj, transform math is ours
- **fp64 I/O** — input/output arrays always double precision (ADR-0002 compliant)
- **Auto GPU detection** — queries `SingleToDoublePrecisionPerfRatio` to classify consumer vs datacenter GPU

## Test

```bash
uv run pytest                    # all tests (251 total)
uv run pytest tests/test_fused_kernels.py  # GPU kernel tests (requires CuPy)
```
