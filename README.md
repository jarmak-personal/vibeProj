# vibeProj

Fast coordinate transforms on NVIDIA GPUs, with a NumPy fallback for CPU-only
environments. vibeProj supports 24 projection families, datum shifts, and
preallocated device buffers through a small pyproj-style API.

- Fused CuPy CUDA kernels keep coordinates on the GPU.
- Helmert 7/15-parameter and SVD-compressed datum corrections are built in.
- Hardware-aware transcendental strategies preserve native fallbacks.
- GeoPandas, Rasterio, Shapely, and vibeSpatial workflows are supported.

## Performance

One million fp64 coordinates on an RTX 4090 and i9-13900K, measured August
2026. GPU measurements use preallocated device inputs and outputs; results are
the median of three runs with ten timed calls per run.

| Projection | GPU | NumPy CPU | Speedup |
|---|---:|---:|---:|
| Transverse Mercator / UTM | 0.47 ms | 125.0 ms | 269x |
| Lambert Conformal Conic | 0.48 ms | 48.7 ms | 101x |
| Albers Equal Area | 0.28 ms | 34.4 ms | 124x |
| Web Mercator | 0.19 ms | 16.9 ms | 90x |
| Equal Earth | 0.45 ms | 56.6 ms | 126x |
| Plate Carrée | 0.07 ms | 9.5 ms | 139x |
| Oblique Mercator | 0.77 ms | 79.2 ms | 103x |
| Krovak | 1.95 ms | 360.5 ms | 185x |

Run the same suite with:

```bash
uv run python benchmarks/bench_projections.py run --n 1000000
```

## Install

```bash
pip install vibeproj            # NumPy
pip install vibeproj[cu12]      # CUDA 12
pip install vibeproj[cu13]      # CUDA 13
```

## Quick start

```python
from vibeproj import Transformer

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
easting, northing = transformer.transform(2.0, 49.0)
```

Inputs may be scalars, lists, NumPy arrays, or CuPy arrays. Geographic
coordinates default to longitude/latitude order; pass `always_xy=False` for
native CRS axis order.

For caller-owned GPU buffers:

```python
transformer.transform_buffers(
    x_gpu,
    y_gpu,
    out_x=easting_gpu,
    out_y=northing_gpu,
)
```

Defaults choose a safe compute precision and transcendental implementation.
Advanced callers can select `precision=` and `transcendentals=` explicitly,
then inspect the decision with `transformer.explain_strategy()`.

## Documentation

- [Quickstart](https://jarmak-personal.github.io/vibeProj/user/quickstart.html)
  and [API](https://jarmak-personal.github.io/vibeProj/user/api.html)
- [Supported projections](https://jarmak-personal.github.io/vibeProj/user/projections.html)
- [Precision](https://jarmak-personal.github.io/vibeProj/user/precision.html)
  and [transcendental strategies](https://jarmak-personal.github.io/vibeProj/user/transcendentals.html)
- Integration recipes for
  [GeoPandas](https://jarmak-personal.github.io/vibeProj/recipes/geopandas.html),
  [Rasterio](https://jarmak-personal.github.io/vibeProj/recipes/rasterio.html),
  [Shapely](https://jarmak-personal.github.io/vibeProj/recipes/shapely.html),
  and [vibeSpatial](https://jarmak-personal.github.io/vibeProj/user/vibespatial.html)

## Development

```bash
uv sync
uv run pytest
```

CUDA development environments can use `uv sync --extra cu12` or
`uv sync --extra cu13`.
