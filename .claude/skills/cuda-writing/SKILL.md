---
name: cuda-writing
description: "PROACTIVELY USE THIS SKILL when writing, modifying, or reviewing GPU kernels, fused NVRTC kernel source, or any GPU code in src/vibeproj/. Covers fused kernel architecture, preamble/postamble system, precision parameterization ({real_t}), device helper functions, kernel compilation via CuPy RawKernel, the Helmert datum shift kernel, and double-single arithmetic."
user-invocable: true
argument-hint: <optional kernel-file-path or topic>
---

# GPU Kernel Development Guide — vibeProj

You are writing GPU code for vibeProj. Follow these rules strictly.
vibeProj uses fused NVRTC kernels compiled via CuPy `RawKernel`.

## 1. Architecture Overview

vibeProj has 40 fused kernels (20 projections x forward/inverse), each
running the full transform pipeline in a single kernel launch:

```
axis swap (CRS-dependent) -> deg/rad -> central meridian subtract ->
projection math -> scale/offset -> axis swap output
```

All kernels live in `src/vibeproj/fused_kernels.py`. The xp fallback path
(`src/vibeproj/projections/`) provides NumPy/CuPy element-wise reference
implementations.

### Key Files

| File | Purpose |
|------|---------|
| `fused_kernels.py` | All 40 fused NVRTC kernel source strings + `fused_transform()` |
| `pipeline.py` | `_try_fused()` fast-path intercepts CuPy arrays |
| `transformer.py` | Public API, `transform_buffers()` zero-copy entry |
| `helmert.py` | 7-param + 15-param datum shift GPU kernel |
| `_ds_device_fns.py` | Double-single fp32 pair arithmetic (experimental) |
| `runtime.py` | GPU detection, precision selection |
| `projections/*.py` | xp fallback implementations (NumPy/CuPy) |

## 2. Fused Kernel Source Pattern

Every projection kernel uses the preamble/postamble system to avoid
repeating axis swap, deg/rad, and scale/offset logic.

### Forward Kernel Template

```python
_<NAME>_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="<name>_forward", real_t="{real_t}")
    + """
    {real_t} <param1>, {real_t} <param2>, ...,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{
"""
    + _FWD_PREAMBLE
    + """
    // === Projection math (compute precision: {real_t}) ===
    {real_t} easting = ...;
    {real_t} northing = ...;

    // Scale + offset (ALWAYS fp64 for sub-meter precision)
    easting  = (double)easting * (double)a + (double)x0;
    northing = (double)northing * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + """
}}
"""
)
```

### Inverse Kernel Template

```python
_<NAME>_INVERSE_SOURCE = (
    _INV_SIGNATURE.format(func="<name>_inverse", real_t="{real_t}")
    + """
    {real_t} <param1>, {real_t} <param2>, ...,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{
"""
    + _INV_PREAMBLE
    + """
    // === Inverse projection math ===
    {real_t} lam = ...;
    {real_t} phi = ...;
"""
    + _INV_POSTAMBLE
    + """
}}
"""
)
```

### What the Preambles/Postambles Do

**`_FWD_PREAMBLE`:**
1. Grid-stride index: `int idx = blockIdx.x * blockDim.x + threadIdx.x;`
2. Bounds check: `if (idx >= n) return;`
3. Read input as fp64: `double d_arg1 = in_x[idx], d_arg2 = in_y[idx];`
4. Axis swap based on `src_north_first`
5. Convert to radians + subtract central meridian: `{real_t} phi`, `{real_t} lam`

**`_FWD_POSTAMBLE`:**
1. Axis swap output based on `dst_north_first`
2. Write to fp64 output arrays: `out_x[idx] = ...`, `out_y[idx] = ...`

**`_INV_PREAMBLE`:**
1. Grid-stride index + bounds check
2. Read input as fp64
3. Axis swap based on `src_north_first`
4. Remove offset and scale in fp64, then cast to `{real_t}`

**`_INV_POSTAMBLE`:**
1. Add central meridian back
2. Convert from radians to degrees
3. Axis swap output based on `dst_north_first`
4. Write to fp64 output arrays

## 3. Precision Model

vibeProj follows a mixed-precision design:

| Layer | Precision | Why |
|-------|-----------|-----|
| I/O arrays | Always `double*` (fp64) | Canonical storage, sub-meter accuracy |
| Compute variables | `{real_t}` (parameterized) | `float` on consumer GPUs = 32x throughput |
| Scale/offset (fwd) | Always `double` | `(double)easting * (double)a + (double)x0` |
| Offset removal (inv) | Always `double` | Remove before casting to `{real_t}` |

### Rules

- **NEVER** hardcode `double` in compute paths — use `{real_t}`.
- **ALWAYS** use `(double)` casts for scale/offset in forward postamble.
- Use `({real_t})` casts for literal constants in compute paths:
  `({real_t})0.5`, `({real_t})1.0`, `({real_t})2.0`.
- The double-single (`ds`) mode in `_ds_device_fns.py` uses `ds_t` (fp32
  pair) for experimental fp64-equivalent accuracy at fp32 throughput.

## 4. Shared Device Functions

Projections reuse shared CUDA device functions:

| Block | Contains | Used By |
|-------|----------|---------|
| `_TM_DEVICE_FNS` | `gatg()`, `clenshaw_complex()` | Transverse Mercator (`tmerc`) |
| `_CONIC_DEVICE_FNS` | `tsfn()`, `phi2()`, `HALF_PI` | LCC, Stereographic, Mercator |
| `_EA_DEVICE_FNS` | `qsfn()`, `phi_from_q()` | Albers Equal-Area, CEA, LAEA |
| `_DS_ARITH` | `ds_t`, `ds_add/mul/sub/from_float` | ds-mode kernels |
| `_DS_TM_DEVICE_FNS` | `ds_gatg()`, `ds_clenshaw_complex()` | ds-mode Transverse Mercator |

When adding a new projection that needs a shared helper, add it as a
named block and include it in the kernel source via string concatenation.

## 5. Kernel Compilation and Caching

```python
import cupy as cp

# Kernels are compiled once and cached by (name, direction, dtype_name)
key = (projection_name, direction, dtype_name)
with _kernel_cache_lock:
    if key not in _kernel_cache:
        source = template.format(real_t=real_t, pi=pi_str)
        _kernel_cache[key] = cp.RawKernel(source, func_name)
kernel = _kernel_cache[key]
```

### Cache Key Rules
- Key tuple: `(projection_name, direction, dtype_name)` where dtype_name
  is `"float64"` or `"float32"`.
- Thread-safe via `_kernel_cache_lock` (`threading.RLock`).
- `warm_up()` and `Transformer.compile()` pre-compile all projections
  in `_SUPPORTED` by calling `fused_transform()` with tiny arrays.

## 6. Kernel Registration

Three data structures must be updated for each new projection:

```python
# 1. Source map: (name, direction) -> (source_template, func_name)
_SOURCE_MAP = {
    ("<name>", "forward"): (_<NAME>_FORWARD_SOURCE, "<name>_forward"),
    ("<name>", "inverse"): (_<NAME>_INVERSE_SOURCE, "<name>_inverse"),
}

# 2. Supported set
_SUPPORTED = {
    ("<name>", "forward"),
    ("<name>", "inverse"),
}

# 3. Arg packing in fused_transform()
elif projection_name == "<name>":
    args = base + (
        real_t(computed["<param>"]),
        ...,
        real_t(computed["lam0"]),
        real_t(computed["a"]),
        real_t(computed["x0"]),
        real_t(computed["y0"]),
        snf, dnf, nn,
    )
```

**Arg order must match the kernel signature exactly.** Common params
(lam0, a, x0, y0, src_north_first, dst_north_first, n) always come last.

## 7. Helmert Datum Shift Kernel

`helmert.py` has a separate GPU kernel for 7-param and 15-param datum
transformations via ECEF intermediate. Key points:

- Supports 3D: `has_z` flag (one int comparison per thread, zero overhead
  when `z=None`).
- 15-param adds 7 rate-of-change params + reference epoch for sub-decimeter
  accuracy.
- Zero overhead for same-datum transforms (`helmert=None`).

## 8. Performance Rules

### Memory Access
- I/O is contiguous `double*` arrays — naturally coalesced.
- One thread per coordinate pair (grid-stride not needed for typical sizes,
  but the preamble uses simple `idx = blockIdx.x * blockDim.x + threadIdx.x`).

### Kernel Launch
- Block size: 256 threads (good occupancy across RTX 3090/4090, A100, H100).
- Grid size: `(n + 255) // 256` blocks.
- All 40+ params passed as kernel arguments (no shared memory needed for
  projection constants — they fit in registers).

### What NOT to Do
- **NEVER** break the fused pipeline into multiple kernel launches.
- **NEVER** allocate intermediate arrays between axis swap, projection, and
  scale/offset — this is why the kernels are fused.
- **NEVER** hardcode `double` in compute paths — use `{real_t}`.
- **NEVER** use `math.sin()`/`math.cos()` in kernel source — use bare
  `sin()`/`cos()` (CUDA intrinsics).
- **NEVER** forget `{{` and `}}` for literal braces — these are Python
  `.format()` template strings.

## 9. Testing Pattern

```python
# GPU tests compare fused kernel output against NumPy xp path
def test_projection_gpu():
    # 1. Run NumPy xp path (reference)
    expected_x, expected_y = pipeline.transform(lat, lon, np)

    # 2. Run CuPy fused kernel path
    lat_gpu, lon_gpu = cp.asarray(lat), cp.asarray(lon)
    actual_x, actual_y = pipeline.transform(lat_gpu, lon_gpu, cp)

    # 3. Compare
    assert_allclose(cp.asnumpy(actual_x), expected_x, atol=1e-7)

# CPU tests validate against pyproj
def test_projection_pyproj():
    pp_x, pp_y = pyproj_transformer.transform(lat, lon)
    vp_x, vp_y = vibeproj_transformer.transform(lat, lon)
    assert_allclose(vp_x, pp_x, atol=0.01)
```
