# Shapely Reprojection

Reproject Shapely 2.x geometries using vibeProj. Works on both GPU (CuPy
available) and CPU (automatic fallback).

**Requirements:** `shapely >= 2.0`, `vibeproj`, `pyproj`

---

## Single geometry: `shapely.transform()`

Use `make_shapely_transform` to get a function compatible with Shapely's
`shapely.transform()` API.

```python
import shapely
from shapely.geometry import Point, Polygon
from vibeproj.compat import make_shapely_transform

# Build the transform function (CRS resolution happens once here)
func = make_shapely_transform("EPSG:4326", "EPSG:32631")

# Reproject a point (WGS 84 -> UTM zone 31N)
pt = Point(2.3522, 48.8566)  # Paris (lon, lat)
new_pt = shapely.transform(pt, func)
print(new_pt)  # POINT (448251.8... 5411932.6...)

# Works with any geometry type
poly = Polygon([(2.0, 48.5), (2.5, 48.5), (2.5, 49.0), (2.0, 49.0)])
new_poly = shapely.transform(poly, func)
```

The returned function accepts (N, 2) or (N, 3) coordinate arrays, so 3D
geometries with Z values are handled automatically. Pass ``include_z=True``
to ``shapely.transform()`` to preserve the Z dimension:

```python
# 3D geometry — Z is transformed through Helmert when crossing datums,
# passed through unchanged for same-datum transforms
pt3d = Point(2.3522, 48.8566, 100.0)
new_pt3d = shapely.transform(pt3d, func, include_z=True)
```

Without ``include_z=True``, Shapely strips Z coordinates before calling
the transform function.

### Reusing across multiple geometries

The `Transformer` is built once inside `make_shapely_transform`. Calling the
returned function repeatedly does not re-resolve the CRS.

```python
func = make_shapely_transform("EPSG:4326", "EPSG:32631")

results = [shapely.transform(g, func) for g in geometry_list]
```

However, each call extracts and transforms coordinates independently. For
large batches, use the bulk API below.

---

## Batch path: `reproject_geometries()`

For lists or arrays of geometries, `reproject_geometries()` extracts all
coordinates in one call, transforms them in bulk, and reconstructs the
geometries. This avoids per-geometry overhead.

```python
from shapely.geometry import Point
from vibeproj.compat import reproject_geometries

# List of geometries
points = [Point(lon, lat) for lon, lat in zip(lons, lats)]

# Bulk reproject — coordinates are extracted once, transformed in a single
# call to transform_chunked(), then put back
reprojected = reproject_geometries(points, "EPSG:4326", "EPSG:32631")
# Returns a list (same type as input)
```

Accepts a single geometry, a list, or a NumPy array of geometries. Returns
the same type as the input.

```python
# Single geometry
new_geom = reproject_geometries(polygon, "EPSG:4326", "EPSG:32631")

# NumPy array of geometries
import numpy as np
geom_arr = np.array([Point(2, 48), Point(3, 49)])
result = reproject_geometries(geom_arr, "EPSG:4326", "EPSG:32631")
# Returns a NumPy array
```

### Reusing a Transformer

Build the `Transformer` once and pass it to avoid repeated CRS resolution:

```python
from vibeproj import Transformer
from vibeproj.compat import reproject_geometries

t = Transformer.from_crs("EPSG:4326", "EPSG:27700")

batch_1 = reproject_geometries(geoms_a, "EPSG:4326", "EPSG:27700", transformer=t)
batch_2 = reproject_geometries(geoms_b, "EPSG:4326", "EPSG:27700", transformer=t)
```

---

## Cross-datum transforms

When the source and destination CRS use different datums, vibeProj applies a
Helmert datum shift automatically, plus an SVD correction when a baked pair is
available (e.g. NAD27 to NAD83 for sub-5cm accuracy). Control accuracy with
`datum_shift` and `epoch`:

```python
# Default: accurate (15-param time-dependent Helmert when available)
func = make_shapely_transform("EPSG:4326", "EPSG:27700")

# Explicit epoch for time-dependent transforms
func = make_shapely_transform("EPSG:4326", "EPSG:27700", epoch=2024.0)

# Fast mode: 7-param Helmert (sub-meter accuracy, lower overhead)
func = make_shapely_transform("EPSG:4326", "EPSG:27700", datum_shift="fast")
```

These keyword arguments are forwarded to `Transformer.from_crs()`.

---

## Controlling chunk size

For large geometries (millions of vertices), `chunk_size` controls how many
coordinate pairs are transferred to the GPU per batch. The default (1M) is
suitable for most workloads.

```python
# Smaller chunks for memory-constrained GPUs
func = make_shapely_transform("EPSG:4326", "EPSG:32631", chunk_size=500_000)

# Larger chunks if GPU memory allows
reprojected = reproject_geometries(
    big_polygons, "EPSG:4326", "EPSG:32631", chunk_size=5_000_000
)
```

On CPU (no CuPy), `chunk_size` is a harmless no-op.

---

## When to use GPU vs CPU

The GPU path is faster than pyproj once the number of coordinate pairs exceeds
a breakeven point. Below that threshold, the host-to-device transfer cost
dominates.

| Projection | Breakeven N (PCIe 4.0) | Breakeven N (PCIe 3.0) |
|------------|------------------------|------------------------|
| tmerc      | ~100                   | ~150                   |
| krovak     | ~30                    | ~50                    |
| lcc        | ~200                   | ~400                   |
| webmerc    | ~700                   | ~1,500                 |
| eqc        | ~2,000                 | ~5,000                 |

**Rules of thumb:**

- **Single Point/LineString with < 100 vertices:** CPU is fine. The GPU
  transfer overhead exceeds the compute savings.
- **Polygon with 1K+ vertices:** GPU wins for all projections.
- **Batch of 100+ geometries:** Always use `reproject_geometries()` to
  extract all coordinates at once. Even if individual geometries are small,
  the bulk extraction amortizes transfer cost.
- **GeoDataFrame with 10K+ rows:** Use `reproject_geodataframe()` from
  `vibeproj.compat` instead -- see the [GeoPandas recipe](geopandas.md).

When CuPy is not installed, vibeProj falls back to NumPy automatically. No
code changes needed.
