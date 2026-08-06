# Fused NVRTC Kernels

## Why fused kernels

The NumPy/CuPy element-wise path executes a projection stage as ~20 separate
operations: axis swap, deg-to-rad, central meridian subtraction, each
trig call, scale, offset. On the GPU, each operation is a separate kernel
launch with its own global memory read/write.

A fused kernel runs one complete mathematical stage in a single kernel launch:
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
    double x_unit_to_m, double y_unit_to_m,
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

Projected westing/southing signs are folded into the two existing unit-factor
arguments on the host. This preserves the kernel ABI and per-thread operation
count: forward kernels divide by a signed factor and inverse kernels multiply
by it. Public `ProjectionParams` retains positive unit magnitudes and stores
component signs separately.

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
_kernel_cache: dict[tuple[str, str, str, str], RawKernel] = {}
# key: (projection_name, direction, compute_dtype, implementation_id)
```

The first call to a kernel compiles it via NVRTC (~100ms). Subsequent
calls reuse the cached `RawKernel`. CuPy also caches the compiled PTX
on disk across Python sessions.

`compile_kernels()` calls `RawKernel.compile()` eagerly for the requested
projection directions and compute precision; constructing a `RawKernel` alone
is lazy. Module-level `warm_up()` uses this path for projection kernels.
`Transformer.compile()` is narrower: it compiles only that transformer's one
or two projection families plus its selected Helmert and SVD kernels, when
present. SVD coefficient arrays are data rather than kernel code and remain a
first-transform, per-device initialization.

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

## Guarded forward Transverse Mercator

The fp64 forward-TM kernel has two separately cached strategies. Native fp64
uses paired CUDA `sincos`; the validated Ada `sm_89` strategy keeps fp64 I/O,
series evaluation, scale, and offset while accelerating three bounded pieces:

- Table-free Q1.62 paired sine/cosine.
- `atan2(sin(B), cos(B)cos(L))` reframed as `B + atan(delta)`. Over the
  complete `|L| <= 0.06` guard, `|delta| <= 9.01e-4` and the degree-5 odd
  correction's maximum absolute error is `2.3e-16 rad`. Normal UTM's tighter
  `+/-3 degree` zone has `|delta| < 6.9e-4`.
- A degree-11 odd `asinh` series for `|x| <= 0.06`, with maximum absolute
  error `2e-17`.

Every guard is per-coordinate. Wider TM coordinates, non-finite values, and
unknown devices use paired native fp64 behavior. Public selection is explicit
for UTM forward transforms on validated Ada consumer GPUs because the host
cannot prove coordinates are in-zone; `auto`, generic TM CRSs, Hopper, and
other architectures use the native strategy pending independent benchmarks. An internal
`tmerc_mode="fp64"|"int64"` override supports accuracy and A/B testing.

On the validation RTX 4090, `benchmarks/bench_tmerc_int64.py` measured 5M
normal-zone forward points at 2.315 ms for the pre-branch fp64 kernel, 2.042 ms
for paired native fp64, and 1.305 ms for the guarded production strategy
(1.77x versus the pre-branch kernel). Maximum radial difference was 4.66 nm.

## Helmert datum shift kernel

The `helmert_shift` kernel in `_HELMERT_SHIFT_SOURCE` runs the full
geodetic→ECEF→Helmert→ECEF→geodetic pipeline on the GPU. It supports
3D transforms via `in_h`/`out_h` array pointers and a `has_z` int flag:

- `has_z=0`: height is assumed zero, no height recovery — one integer
  comparison per thread (negligible overhead vs 2D-only).
- `has_z=1`: reads ellipsoidal height from `in_h`, includes it in ECEF
  conversion, recovers height on the destination ellipsoid via
  `h = p / cos(lat) - N` (with a near-pole guard using the Z-based formula).

The Helmert kernel is separate from the fused projection kernels —
projections are inherently 2D. For cross-datum transforms, the pipeline
runs the Helmert kernel first (or after inverse projection), then the
projection kernel. z passes through the projection kernel unchanged.

Every direct same-argument sine/cosine site uses the shared native paired
`sincos` helper (with an equivalent helper for double-single TM). On
validated Ada consumer GPUs (`sm_89` with weak native-fp64 throughput),
automatic dispatch instead uses table-free signed Q1.62 trig for bounded
Helmert angles. It reduces to `[-pi/4, pi/4]` and evaluates degree-17 sine
and degree-18 cosine polynomials with full-width INT64 products. Kernel I/O,
ECEF math, the Helmert matrix, and Bowring iteration remain fp64. Unknown,
datacenter, out-of-domain, non-finite, and near-pole cases use native fp64.

The two Helmert variants are cached separately under the stable
`native.libdevice` and `helmert.fixed_q62` implementation IDs.
`fused_helmert_shift(..., trig_mode=...)` retains the private legacy
`"fp64"`/`"int64"` override for deterministic tests and research benchmarks;
public Transformer dispatch passes the resolved implementation ID.

## SVD datum correction kernel

The `svd_correction` kernel evaluates the SVD-compressed residual correction
after the Helmert shift. For each point it computes:

```
correction(lat, lon) = sum_k S[k] * lerp(U_k, lat_idx) * lerp(V_k, lon_idx)
```

The kernel receives the flattened U, S, V matrices, grid bounds, and grid
dimensions as arguments. Bilinear interpolation is used for sub-grid positions.
Applied only when a baked `DatumCorrectionData` exists for the datum pair.

The SVD `RawKernel` has its own process-local cache. Its six coefficient arrays
are cached separately under `(CUDA device ID, id(DatumCorrectionData))`, with a
strong reference that prevents object-ID reuse. First use uploads them on the
caller's stream and records a readiness event; a consumer on another stream
waits for an incomplete event without a device-wide or host synchronization.
Once complete, the event is retired atomically so a warmed call—including one
inside CUDA graph capture—adds no event operation. Warmed transforms reuse the
arrays and do not allocate or upload them again.

The SVD launch device is the input latitude array's CUDA device. The legacy
null stream (`device_id == -1`) is normalized inside that device context;
explicit non-null streams must belong to the same device. Longitude, output
buffers, cached coefficient arrays, and streams are validated for device
agreement before launch. First-use coefficient copies, event operations,
contiguity copies, and the kernel launch all execute under the same explicit
device-and-stream context.

## Double-single kernels

The `_DS_SOURCE_MAP` contains ds-specific kernel sources that use
`ds_t` pair arithmetic instead of `{real_t}`. These kernels take
`double` parameters (the ds arithmetic is internal to the kernel).

Currently only Transverse Mercator has a ds variant. Other projections
fall back to fp64 when `precision="ds"` is requested.
