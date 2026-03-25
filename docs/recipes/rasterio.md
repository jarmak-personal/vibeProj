# Rasterio Raster Reprojection

vibeProj transforms **coordinate grids**, not pixels. To reproject a raster you
build a destination meshgrid, inverse-project it to source CRS coordinates, then
resample pixels from the source image at those locations.

This is the same algorithm GDAL/rasterio uses internally. The difference is that
vibeProj runs the coordinate transform on the GPU (100--350x faster than pyproj),
and if you generate the meshgrid on the GPU too, the entire coordinate pipeline
stays device-resident with zero host-device transfers.

## CPU path: `transform_chunked()`

Use this when your raster data lives in host memory (the normal rasterio case).
`transform_chunked()` handles H<->D transfers internally with a double-buffered
pinned-memory pipeline.

```python
import numpy as np
import rasterio
from scipy.ndimage import map_coordinates

from vibeproj import Transformer

# --- 1. Open source raster ---
with rasterio.open("input.tif") as src:
    data = src.read(1)  # shape: (src_height, src_width)
    src_crs = src.crs
    src_transform = src.transform

# --- 2. Define destination grid ---
dst_crs = "EPSG:32631"  # UTM zone 31N
dst_transform = rasterio.transform.from_bounds(
    *target_bounds, width=dst_width, height=dst_height
)

# Build meshgrid of destination pixel centers
cols = np.arange(dst_width, dtype=np.float64) * dst_transform.a + dst_transform.c
rows = np.arange(dst_height, dtype=np.float64) * dst_transform.e + dst_transform.f
grid_x, grid_y = np.meshgrid(cols, rows)

# --- 3. Inverse-project: destination CRS -> source CRS ---
t = Transformer.from_crs(dst_crs, src_crs)
src_lon, src_lat = t.transform_chunked(grid_x.ravel(), grid_y.ravel())
src_lon = src_lon.reshape(dst_height, dst_width)
src_lat = src_lat.reshape(dst_height, dst_width)

# --- 4. Convert source CRS coords to source pixel coordinates ---
inv_src_transform = ~src_transform
src_col, src_row = rasterio.transform.rowcol(src_transform, src_lon, src_lat)
# rowcol returns (row, col) arrays; map_coordinates wants (row, col)
src_row = np.asarray(src_row, dtype=np.float64)
src_col = np.asarray(src_col, dtype=np.float64)

# --- 5. Resample pixels ---
reprojected = map_coordinates(data, [src_row, src_col], order=1)
reprojected = reprojected.reshape(dst_height, dst_width)
```

### Notes

- **`order=1`** is bilinear interpolation. Use `order=3` for bicubic (slower,
  smoother) or `order=0` for nearest-neighbor (categorical data, e.g. land cover).
- **`rasterio.transform.rowcol`** converts geographic/projected coordinates to
  fractional pixel coordinates in the source image. This is a simple affine
  inverse -- it runs instantly on the CPU.
- **Multi-band**: loop over bands or stack into a 3D array and adjust the
  `map_coordinates` call accordingly.

## GPU-native path: `transform_buffers()` (zero H->D transfer)

When the meshgrid is generated on the GPU, the entire coordinate transform
stays device-resident. No host-device transfer occurs for the coordinate grid.

```python
import cupy as cp

from vibeproj import Transformer

# Build destination meshgrid directly on GPU
cols = cp.arange(dst_width, dtype=cp.float64) * dst_transform.a + dst_transform.c
rows = cp.arange(dst_height, dtype=cp.float64) * dst_transform.e + dst_transform.f
grid_x, grid_y = cp.meshgrid(cols, rows)

# Inverse-project entirely on GPU (zero-copy)
t = Transformer.from_crs(dst_crs, src_crs)
src_x, src_y = t.transform_buffers(grid_x.ravel(), grid_y.ravel())

# Reshape on device
src_x = src_x.reshape(dst_height, dst_width)
src_y = src_y.reshape(dst_height, dst_width)
```

From here, pixel resampling can be done on the GPU (e.g. with a custom CuPy
kernel or `cupyx.scipy.ndimage.map_coordinates`) or the coordinate arrays can
be transferred to the host for `scipy.ndimage.map_coordinates`.

## Tile-based processing for large rasters

For rasters that exceed GPU memory, process tiles independently. Each tile
generates its own sub-meshgrid and reprojects it.

```python
TILE_SIZE = 4096  # pixels per tile edge

for tile_row in range(0, dst_height, TILE_SIZE):
    for tile_col in range(0, dst_width, TILE_SIZE):
        th = min(TILE_SIZE, dst_height - tile_row)
        tw = min(TILE_SIZE, dst_width - tile_col)

        # Tile meshgrid
        c = cp.arange(tile_col, tile_col + tw, dtype=cp.float64) * dst_transform.a + dst_transform.c
        r = cp.arange(tile_row, tile_row + th, dtype=cp.float64) * dst_transform.e + dst_transform.f
        gx, gy = cp.meshgrid(c, r)

        # Inverse-project tile coords (GPU-resident, zero-copy)
        sx, sy = t.transform_buffers(gx.ravel(), gy.ravel())

        # ... resample source pixels for this tile ...
```

Memory usage per tile: `4 * tile_h * tile_w * 8` bytes for the coordinate
arrays (grid_x, grid_y, src_x, src_y). A 4096x4096 tile uses ~512 MB. Reduce
tile size if needed.

## Performance

Coordinate grid reprojection only (excludes pixel resampling):

| Tile size | Coord pairs | GPU (ms) | CPU (ms) | Speedup |
|-----------|-------------|----------|----------|---------|
| 256x256   | 65K         | ~0.07    | ~9       | 129x    |
| 4096x4096 | 16.8M       | ~18      | ~2,200   | 122x    |

GPU timings are for the coordinate transform kernel only (`transform_buffers`
with grid already on device). CPU timings are pyproj on a single core. Actual
end-to-end raster reprojection time depends on pixel resampling and I/O.

## When to use this vs `rasterio.warp.reproject`

- **Use rasterio** when you need full GDAL reprojection (NTv2 grid shifts,
  geoid models, output-format writing, GDAL VRT integration).
- **Use vibeProj + manual resampling** when coordinate reprojection is the
  bottleneck and you want GPU acceleration. This is most impactful for large
  rasters (>10M pixels) or batch reprojection of many tiles where the
  coordinate grid transform dominates wall time.
