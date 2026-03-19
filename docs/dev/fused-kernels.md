# Fused NVRTC Kernels

## Why fused kernels

The NumPy/CuPy element-wise path executes a transform as ~20 separate
operations: axis swap, deg-to-rad, central meridian subtraction, each
trig call, scale, offset. On the GPU, each operation is a separate kernel
launch with its own global memory read/write.

A fused kernel runs the entire pipeline in a single kernel launch:
one thread per coordinate pair, all stages in registers. This eliminates:

- ~20 kernel launches (each has ~5us overhead)
- ~20 temporary array allocations
- Multiple global memory round-trips

For 1M coordinates on an RTX 4090, the fused Transverse Mercator kernel
runs in 0.49ms vs ~2ms for the element-wise path.

## Kernel structure

Every fused kernel follows this template:

```c
extern "C" __global__ void my_forward(
    const double* __restrict__ in_x,
    const double* __restrict__ in_y,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    // projection-specific scalar parameters
    double lam0, double a, double x0, double y0,
    int src_north_first, int dst_north_first, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 1. Read input, handle axis order
    double d1 = in_x[idx], d2 = in_y[idx];
    double lat_deg, lon_deg;
    if (src_north_first) { lat_deg = d1; lon_deg = d2; }
    else                 { lon_deg = d1; lat_deg = d2; }

    // 2. Deg -> Rad, subtract central meridian
    double phi = lat_deg * DEG_TO_RAD;
    double lam = lon_deg * DEG_TO_RAD - lam0;
    // wrap to [-pi, pi]
    lam = lam - 2.0 * PI * round(lam / (2.0 * PI));

    // 3. Projection math (the interesting part)
    double easting  = /* ... */;
    double northing = /* ... */;

    // 4. Write output, handle axis order
    if (dst_north_first) { out_x[idx] = northing; out_y[idx] = easting; }
    else                 { out_x[idx] = easting;  out_y[idx] = northing; }
}
```

## Preamble/postamble macros

To avoid repeating the boilerplate, `fused_kernels.py` defines macros:

- `_FWD_SIGNATURE` -- the `extern "C" __global__ void` line with I/O params
- `_FWD_PREAMBLE` -- steps 1-2 above (read, axis swap, deg/rad, central meridian)
- `_FWD_POSTAMBLE` -- step 4 (axis swap, write)
- `_INV_SIGNATURE`, `_INV_PREAMBLE`, `_INV_POSTAMBLE` -- inverse equivalents

Your kernel source only needs to contain the projection-specific scalar
parameters and the actual projection math between preamble and postamble.

## Type parameterisation

Kernel sources use `{real_t}` and `{pi}` placeholders:

```python
_MY_FORWARD_SOURCE = _FWD_SIGNATURE.format(
    func="my_forward", real_t="{real_t}"
) + """
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    ...
"""
```

At compile time, `{real_t}` is substituted with `double` (fp64) or
`float` (fp32), and `{pi}` with the appropriate literal. I/O arrays
are always `double*` regardless of compute precision.

## Kernel cache

Compiled kernels are cached in `_kernel_cache`:

```python
_kernel_cache: dict[tuple[str, str, str], RawKernel] = {}
# key: (projection_name, direction, compute_dtype)
```

The first call to a kernel compiles it via NVRTC (~100ms). Subsequent
calls reuse the cached `RawKernel`. CuPy also caches the compiled PTX
on disk across Python sessions.

## Argument packing

The `fused_transform()` function packs projection-specific parameters
into the kernel call. Each projection has its own branch:

```python
if projection_name == "tmerc":
    c6 = [real_t(c) for c in computed["cbg"]]
    g6 = [real_t(c) for c in computed["gtu"]]
    args = base + (*c6, *g6, real_t(computed["Qn"]), ...)
elif projection_name == "webmerc":
    args = base + (real_t(computed["lam0"]), ...)
```

Parameters are cast to the compute dtype (`np.float64` or `np.float32`)
and passed as kernel arguments.

## Helmert datum shift kernel

The `helmert_shift` kernel in `_HELMERT_SHIFT_SOURCE` runs the full
geodetic→ECEF→Helmert→ECEF→geodetic pipeline on the GPU. It supports
3D transforms via `in_h`/`out_h` array pointers and a `has_z` int flag:

- `has_z=0`: height is assumed zero, no height recovery — one integer
  comparison per thread (negligible overhead vs 2D-only).
- `has_z=1`: reads ellipsoidal height from `in_h`, includes it in ECEF
  conversion, recovers height on the destination ellipsoid via
  `h = p / cos(lat) - N` (with a near-pole guard using the Z-based formula).

The Helmert kernel is separate from the 40 fused projection kernels —
projections are inherently 2D. For cross-datum transforms, the pipeline
runs the Helmert kernel first (or after inverse projection), then the
projection kernel. z passes through the projection kernel unchanged.

## Double-single kernels

The `_DS_SOURCE_MAP` contains ds-specific kernel sources that use
`ds_t` pair arithmetic instead of `{real_t}`. These kernels take
`double` parameters (the ds arithmetic is internal to the kernel).

Currently only Transverse Mercator has a ds variant. Other projections
fall back to fp64 when `precision="ds"` is requested.
