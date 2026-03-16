# vibeSpatial Integration

vibeProj is designed as the GPU projection engine for
{external+vibespatial:doc}`vibeSpatial <index>`. This guide
covers the zero-copy integration pattern for projecting coordinate
buffers that already live on the GPU.

## The problem

A typical GIS reprojection workflow copies data off the GPU, reprojects
on the CPU via pyproj, then copies back:

```
GPU buffer → host copy → pyproj (CPU) → host copy → GPU buffer
```

For a 10M-point GeoDataFrame this adds ~40ms of PCIe transfer on top of
~500ms of CPU projection time. vibeProj eliminates both:

```
GPU buffer → fused kernel (GPU) → GPU buffer
```

One kernel launch, no host round-trip, no intermediate allocation.

## `transform_buffers()` API

The key integration point is `transform_buffers()`. Unlike `transform()`,
it:

- Skips scalar detection and dtype inference
- Accepts pre-allocated output arrays (zero allocation)
- Returns the same output array objects (verifiable with `is`)

```python
import cupy as cp
from vibeproj import Transformer

t = Transformer.from_crs("EPSG:4326", "EPSG:32631")

# Input: device-resident fp64 arrays
x_in = cp.asarray(lat_data, dtype=cp.float64)
y_in = cp.asarray(lon_data, dtype=cp.float64)

# Output: pre-allocated on device
x_out = cp.empty_like(x_in)
y_out = cp.empty_like(y_in)

# Single fused kernel launch, no allocation, no host transfer
rx, ry = t.transform_buffers(x_in, y_in, out_x=x_out, out_y=y_out)

assert rx is x_out  # same object — no copy
```

## Integration with vibeSpatial's coordinate buffers

vibeSpatial stores geometry coordinates as separated CuPy float64 arrays
inside `OwnedGeometryArray`
(see {external+vibespatial:doc}`GPU acceleration <user/gpu>` and
{external+vibespatial:doc}`API reference <user/api>`).
A typical `to_crs()` implementation looks like:

```python
def to_crs(gdf, target_crs):
    """Reproject a GPU GeoDataFrame using vibeProj."""
    from vibeproj import Transformer

    src_crs = gdf.crs
    t = Transformer.from_crs(src_crs, target_crs)

    # Access the raw coordinate buffers (already on GPU)
    coords = gdf.geometry._data  # OwnedGeometryArray
    x_in = coords.x   # cupy.ndarray, fp64
    y_in = coords.y   # cupy.ndarray, fp64

    # Pre-allocate output buffers for the new geometry
    x_out = cp.empty_like(x_in)
    y_out = cp.empty_like(y_in)

    # Project — single fused kernel launch
    t.transform_buffers(x_in, y_in, out_x=x_out, out_y=y_out)

    # Build new GeoDataFrame with projected coordinates
    return gdf._with_coordinates(x_out, y_out, crs=target_crs)
```

The critical property: **no data leaves the GPU**. The input coordinate
arrays are read by the fused kernel, and the output arrays are written
by the same kernel. There are no intermediate Python objects, no
temporary arrays, and no host-device transfers.

## Caching the Transformer

`Transformer.from_crs()` resolves CRS metadata via pyproj and
precomputes projection constants. For repeated transforms (e.g. in a
processing loop), create the transformer once and reuse it:

```python
# Create once
t = Transformer.from_crs("EPSG:4326", "EPSG:32631")

# Reuse across many batches
for batch in data_stream:
    t.transform_buffers(batch.x, batch.y, out_x=out_x, out_y=out_y)
```

The first call to `transform_buffers()` compiles the NVRTC kernel
(~100ms). Subsequent calls reuse the cached compiled kernel. CuPy also
persists the compiled PTX to disk, so the next Python session starts
fast.

## Inverse transforms

`transform_buffers()` supports inverse transforms with the same
zero-copy semantics:

```python
# Project: geographic → UTM
t.transform_buffers(lat, lon, out_x=easting, out_y=northing)

# Unproject: UTM → geographic
t.transform_buffers(
    easting, northing,
    direction="INVERSE",
    out_x=lat_out, out_y=lon_out,
)
```

## Chained projection (projected → projected)

When both source and destination are projected CRS (e.g. UTM → Web
Mercator), vibeProj chains two fused kernel calls through a geographic
intermediate. The intermediate coordinates stay on the GPU in registers
— no extra device memory is allocated:

```python
t = Transformer.from_crs("EPSG:32631", "EPSG:3857")
t.transform_buffers(utm_x, utm_y, out_x=webmerc_x, out_y=webmerc_y)
```

## Performance expectations

On an RTX 4090 with 1M fp64 coordinates:

| Operation | Time |
|---|---|
| Fused TM forward kernel | 0.50 ms |
| `transform_buffers()` overhead | < 0.01 ms |
| Equivalent pyproj CPU (i9-13900k) | ~130 ms |
| Host→Device copy (16 MB) | ~2 ms |

The GPU kernel is 260x faster than CPU pyproj, and `transform_buffers()`
adds negligible Python overhead on top of the raw kernel time.
