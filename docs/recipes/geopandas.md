# GeoPandas Reprojection

Reproject a GeoDataFrame on the GPU using vibeProj. Works on CPU too --
`transform_chunked()` falls back to NumPy transparently when CuPy is not
installed.

## Quick start: `reproject_geodataframe()`

```python
from vibeproj.compat import reproject_geodataframe

gdf_utm = reproject_geodataframe(gdf, "EPSG:32631")
```

This is the one-liner path. It handles CRS extraction from the source
GeoDataFrame, 2D/3D geometries, and bulk coordinate transform in a single
call. The result is a new GeoDataFrame with the target CRS set.

You can pass any `Transformer.from_crs()` keyword through `**kw`:

```python
# Cross-datum with explicit epoch for sub-decimeter accuracy
gdf_etrs = reproject_geodataframe(
    gdf,
    "EPSG:4258",
    datum_shift="accurate",
    epoch=2024.0,
    chunk_size=2_000_000,
)
```

To reuse the transformer across multiple calls (avoids repeated CRS
resolution and kernel compilation):

```python
from vibeproj import Transformer

t = Transformer.from_crs("EPSG:4326", "EPSG:32631")

gdf_a_utm = reproject_geodataframe(gdf_a, "EPSG:32631", transformer=t)
gdf_b_utm = reproject_geodataframe(gdf_b, "EPSG:32631", transformer=t)
```

## Manual recipe

For full control over the coordinate pipeline, use `shapely.get_coordinates()`
and `shapely.set_coordinates()` directly. This is roughly what
`reproject_geodataframe()` does internally.

```python
import numpy as np
import shapely
from vibeproj import Transformer

# 1. Build transformer once
t = Transformer.from_crs(str(gdf.crs), "EPSG:32631")

# 2. Extract all coordinates as a single (N, 2) or (N, 3) array
geom_arr = gdf.geometry.values
has_z = bool(shapely.has_z(geom_arr).any())
coords = shapely.get_coordinates(geom_arr, include_z=has_z)

# 3. Bulk transform
x, y = coords[:, 0], coords[:, 1]
if has_z:
    rx, ry, rz = t.transform_chunked(x, y, z=coords[:, 2])
    new_coords = np.column_stack([rx, ry, rz])
else:
    rx, ry = t.transform_chunked(x, y)
    new_coords = np.column_stack([rx, ry])

# 4. Reconstruct geometries and build result
new_geoms = shapely.set_coordinates(geom_arr.copy(), new_coords)
result = gdf.copy()
result[gdf.geometry.name] = new_geoms
result = result.set_geometry(gdf.geometry.name).set_crs("EPSG:32631")
```

The key insight is that `shapely.get_coordinates()` flattens every vertex
from every geometry in the GeoDataFrame into one contiguous array. This
turns a GeoDataFrame with 1M polygons (50 vertices each) into a single
50M-row coordinate array -- exactly the kind of large, embarrassingly
parallel workload that saturates the GPU.

## Performance comparison

All numbers use Transverse Mercator (EPSG:4326 to EPSG:32631). GPU timings
include host-to-device and device-to-host transfer via `transform_chunked()`.

| Scenario | GPU | CPU (`gdf.to_crs()`) | Speedup |
|---|---|---|---|
| 1M-geom GeoDataFrame (50 verts/geom) | ~40 ms | ~7 s | **175x** |
| 1M tmerc (optimized chunked pipeline) | ~1.3 ms | ~139 ms | **107x** |
| 1M tmerc (kernel only, no transfer) | ~0.49 ms | ~139 ms | **284x** |

The `gdf.to_crs()` path goes through pyproj, which processes coordinates
serially on the CPU. vibeProj extracts all coordinates in one shot, transfers
them to the GPU, runs a single fused kernel, and writes the results back.
Transfer overhead dominates at small scales -- the GPU breakeven for
Transverse Mercator is roughly 100 coordinate pairs on PCIe 4.0.

On CPU (no CuPy), vibeProj's `transform_chunked()` falls back to vectorized
NumPy operations. This is faster than pyproj for large arrays but slower
than the GPU path. The `chunk_size` parameter becomes a no-op.

## GeoArrow future

GeoPandas is moving toward native GeoArrow storage for geometry columns
(`geopandas >= 1.0` with `use_pyarrow=True`). When GeoArrow becomes the
default storage backend, the coordinate extraction step
(`shapely.get_coordinates()` / `set_coordinates()`) can be replaced with
direct access to the underlying Arrow coordinate buffers -- eliminating the
host-side copy entirely.

Combined with `transform_buffers()` for zero-copy GPU dispatch, this will
remove the last remaining memory allocation from the reprojection path:
Arrow buffer on host, pinned transfer to device, in-place kernel, pinned
transfer back, Arrow buffer on host. No intermediate arrays.
