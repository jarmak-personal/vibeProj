# Architecture

## Overview

vibeProj is a pure Python + CuPy library -- no compiled C/C++ extensions,
no CMake. GPU kernels are compiled at runtime via CuPy's NVRTC interface.

```
Transformer.from_crs("EPSG:4326", "EPSG:32631")
        │
        ▼
┌─────────────┐
│  crs.py     │  pyproj resolves EPSG → projection params
└──────┬──────┘
       │
       ▼
┌──────────────┐
│  pipeline.py │  chains pre/post ops with projection core
└──────┬───────┘
       │
       ├──── GPU? ──► fused_kernels.py  (single kernel launch)
       │
       └──── CPU? ──► projections/<name>.py  (NumPy element-wise)
```

## Module map

| Module | Responsibility |
|---|---|
| `transformer.py` | Public API. `Transformer` class with `transform()` and `transform_buffers()`. |
| `crs.py` | CRS resolution via pyproj. Extracts projection type + parameters from EPSG codes. Maps pyproj method names to internal names via `_METHOD_MAP`. |
| `pipeline.py` | Transform pipeline. Chains axis swap, deg/rad, central meridian, projection core, scale, offset. Contains `_try_fused()` fast-path for GPU arrays. |
| `fused_kernels.py` | 40 CUDA kernel source strings (20 projections x fwd/inv). Compiled and cached via CuPy `RawKernel`. |
| `projections/` | NumPy/CuPy element-wise implementations. Each is a `Projection` subclass with `setup()`, `forward()`, `inverse()`. |
| `ellipsoid.py` | Reference ellipsoid definitions (WGS84, GRS80, sphere). |
| `helmert.py` | Helmert 7/15-parameter datum transformation. `HelmertParams` dataclass, geodetic/ECEF conversion (with optional ellipsoidal height), `apply_helmert()`. Supports 3D: when z is provided, height is transformed through the ECEF intermediate. |
| `runtime.py` | GPU/CPU detection and array module selection. |
| `gpu_detect.py` | Consumer vs datacenter GPU classification. |
| `_ds_device_fns.py` | Double-single fp32 arithmetic CUDA device functions. |

## Transform pipeline stages

A forward transform (geographic -> projected) executes these stages:

1. **Axis swap** -- CRS-dependent. EPSG:4326 is (lat, lon); some projected CRS are (E, N).
2. **Datum shift** -- Helmert 7/15-parameter (if cross-datum). Converts geodetic coords from source ellipsoid to destination ellipsoid via ECEF intermediate. When z (ellipsoidal height) is provided, it is included in the ECEF conversion and recovered on the destination ellipsoid. Skipped entirely when `helmert is None` (same-datum). Projection stages (3-8) are inherently 2D — z passes through unchanged.
3. **Degree to radian** -- `lat * pi/180`, `lon * pi/180`.
4. **Central meridian** -- `lon -= lon_0`, wrapped to [-pi, pi].
5. **Projection core** -- the actual math (Transverse Mercator, Lambert, etc.).
6. **Scale** -- multiply by semi-major axis `a`.
7. **False easting/northing** -- add `x_0`, `y_0`.
8. **Output axis swap** -- match destination CRS axis order.

Inverse transforms reverse these stages.

For proj-to-proj transforms (projected -> projected), the pipeline decomposes into:
inverse(src) -> datum shift (if cross-datum) -> forward(dst). Each sub-step
may use its own fused GPU kernel, with the Helmert shift running as a separate
kernel launch between them.

## Fused kernel fast-path

When `_try_fused()` detects a CuPy array input and a supported projection,
it dispatches to a single GPU kernel that performs all 7 pipeline stages
in one kernel launch. This eliminates:

- ~20 intermediate CuPy kernel launches
- ~20 temporary array allocations
- Multiple global memory round-trips

The xp (NumPy/CuPy element-wise) path runs the same stages as individual
operations. It serves as the CPU fallback and the reference for testing.

## Lazy CuPy imports

CuPy is always imported lazily to keep vibeProj usable without a GPU:

```python
# In pipeline.py
def _get_cupy():
    global _cupy_module
    if _cupy_module is None:
        try:
            import cupy
            _cupy_module = cupy
        except ImportError:
            _cupy_module = False
    return _cupy_module if _cupy_module is not False else None
```

This pattern is used throughout the codebase. Never add a top-level
`import cupy` to any module.
