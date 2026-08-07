# vibeProj

Fast coordinate transforms on NVIDIA GPUs, with a NumPy fallback for CPU-only
environments. vibeProj supports 24 projection families, datum shifts, and
preallocated device buffers through a small pyproj-style API.

- Fused CuPy CUDA kernels keep coordinates on the GPU.
- Helmert 7/15-parameter and SVD-compressed datum corrections are built in.
- Hardware-aware transcendental strategies preserve native fallbacks.
- GeoPandas, Rasterio, Shapely, and vibeSpatial workflows are supported.

## Performance

One million fp64 coordinates on an RTX 4090 and i9-13900K, measured with
pyproj 3.7.2 in August 2026. GPU measurements use preallocated device inputs
and outputs; results are the median of three runs with ten timed calls per run.

| Projection | vibeProj GPU | vibeProj NumPy | pyproj |
|---|---:|---:|---:|
| Transverse Mercator / UTM | 0.47 ms (**186x faster**) | 123.9 ms (**1.43x slower**) | 86.7 ms |
| Lambert Conformal Conic | 0.48 ms (**150x faster**) | 48.1 ms (**1.50x faster**) | 72.1 ms |
| Albers Equal Area | 0.28 ms (**173x faster**) | 34.5 ms (**1.39x faster**) | 48.0 ms |
| Web Mercator | 0.19 ms (**295x faster**) | 17.1 ms (**3.22x faster**) | 54.9 ms |
| Equal Earth | 0.45 ms (**148x faster**) | 56.4 ms (**1.18x faster**) | 66.4 ms |
| Plate Carrée | 0.07 ms (**391x faster**) | 9.2 ms (**2.86x faster**) | 26.2 ms |
| Oblique Mercator | 0.77 ms (**129x faster**) | 79.3 ms (**1.25x faster**) | 99.5 ms |
| Krovak | 1.95 ms (**206x faster**) | 360.4 ms (**1.11x faster**) | 401.3 ms |

Parenthetical comparisons use pyproj as the baseline.

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
