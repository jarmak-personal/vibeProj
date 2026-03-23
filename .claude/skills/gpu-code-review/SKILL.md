---
name: gpu-code-review
description: "PROACTIVELY USE THIS SKILL when reviewing GPU code changes in vibeProj. Provides the 6-pass review procedure with thresholds, anti-patterns, and hardware specs specific to fused NVRTC projection kernels."
user-invocable: true
argument-hint: <file-path or diff to review>
---

# GPU Code Review — vibeProj

You are reviewing GPU code for vibeProj, a coordinate projection library
with 40 fused NVRTC kernels. Use this 6-pass procedure for every review.

## Hardware Reference

| GPU | SMs | Max Threads/SM | fp64:fp32 Ratio | Notes |
|-----|-----|----------------|-----------------|-------|
| RTX 3090 (CC 8.6) | 82 | 1,536 | 1:64 | Consumer, fp32 is 64x faster |
| RTX 4090 (CC 8.9) | 128 | 1,536 | 1:64 | Consumer, fp32 is 64x faster |
| A100 (CC 8.0) | 108 | 2,048 | 1:2 | Datacenter, fp64 is viable |
| H100 (CC 9.0) | 132 | 2,560 | 1:2 | Datacenter, fp64 is viable |

vibeProj projection math is SFU-bound (sin/cos/atan/exp), so the fp64:fp32
throughput gap matters less than for ALU-bound kernels — but `{real_t}`
parameterization is still required for correctness and consistency.

## Pass 1: Host-Device Boundary (CRITICAL)

**What to check:**
- No D->H->D ping-pong (data goes to device, stays there until user requests)
- No `.get()` / `cp.asnumpy()` in the middle of `transform_buffers()` or
  `_try_fused()` pipeline
- No Python loops over device array elements
- `transform_buffers()` zero-copy contract maintained: reads device buffers,
  writes to pre-allocated device buffers, no host round-trips
- Cross-datum transforms (Helmert + projection) stay entirely on device

**Anti-patterns:**
```python
# BAD: host round-trip between Helmert and projection
x_host = cp.asnumpy(x_shifted)  # D2H
x_proj = np_projection(x_host)  # CPU
result = cp.asarray(x_proj)     # H2D

# GOOD: stays on device
fused_transform(x_shifted, y_shifted, ...)  # device -> device
```

**Severity:** BLOCKING for any transfer in `transform_buffers()`, `_try_fused()`,
or `fused_transform()` paths. Necessary transfers only at user-facing
`transform()` return.

## Pass 2: Synchronization (HIGH)

**What to check:**
- No unnecessary `cp.cuda.Stream.null.synchronize()` calls
- No implicit sync from `print(cupy_array)`, `int(cupy_scalar)`,
  `float(cupy_scalar)` in hot paths
- Stream parameter flows correctly through `transform_buffers()` ->
  `pipeline.transform()` -> `_try_fused()` -> `fused_transform()` ->
  kernel launch

**Severity:** BLOCKING for any sync in the fused kernel dispatch path that
is not at the pipeline boundary.

## Pass 3: Kernel Efficiency (HIGH)

**What to check:**
- Bounds check: `if (idx >= n) return;` (must be present)
- Grid size: `(n + block_size - 1) // block_size` (sufficient to cover all elements)
- No branch divergence in inner loops (projection math should be uniform)
- `const double* __restrict__` on read-only pointer parameters
- Fused pipeline maintained — single kernel launch per projection
- Device functions use `__device__ inline`
- No `__syncthreads()` (not needed for element-wise projection kernels)

**Thresholds:**
- Block size: 256 threads is standard for vibeProj
- Register target: <32 per thread for max occupancy
- Grid size should saturate at least 80% of SMs at 1M coordinates

**Severity:** BLOCKING for broken fused pipeline, missing bounds check,
or insufficient grid sizing.

## Pass 4: Precision Compliance (MEDIUM-HIGH)

**What to check:**
- Compute variables use `{real_t}`, not hardcoded `double`
- Kernel I/O is always `double*` (read/write as fp64)
- Scale/offset in forward stays fp64: `(double)easting * (double)a + (double)x0`
- Offset removal in inverse stays fp64 before cast to `{real_t}`
- Literal constants use `({real_t})` cast: `({real_t})0.5`, `({real_t})1.0`
- `{{` and `}}` for literal braces (Python `.format()` escaping)
- Cache key includes precision: `(name, direction, dtype_name)`

**Anti-patterns:**
```c
// BAD: hardcoded double in compute path
double sin_phi = sin(phi);

// GOOD: parameterized precision
{real_t} sin_phi = sin(phi);

// BAD: fp32 scale/offset (loses sub-meter accuracy)
float easting_m = easting * a + x0;

// GOOD: fp64 scale/offset
easting = (double)easting * (double)a + (double)x0;
```

**Severity:** BLOCKING for hardcoded `double` in compute paths or fp32
scale/offset.

## Pass 5: Memory Management (MEDIUM)

**What to check:**
- No per-call allocations in `fused_transform()` hot path
- Output buffers sized correctly (`out_x`, `out_y` same length as input)
- `_kernel_cache_lock` used for thread-safe compilation
- No memory leaks (CuPy manages GPU memory via pool, but check for
  retained references to large arrays)

**Severity:** BLOCKING for allocations in hot path or thread-safety violations.

## Pass 6: NVRTC/Compilation (LOW-MEDIUM)

**What to check:**
- Cache key `(projection_name, direction, dtype_name)` covers all
  parameterizations — fp32 and fp64 compile to different kernels
- Compilation happens once (on first call), not per-transform
- `warm_up()` and `compile()` discover new projections via `_SUPPORTED`
- Source string uses `.format(real_t=..., pi=...)` correctly — no
  unsubstituted `{real_t}` in compiled source

**Severity:** BLOCKING for compilation in hot path or missing cache key
dimension.

## Output Format

For each pass, report: **CLEAN** or list findings with severity and location.
End with overall verdict: **CLEAN** or **BLOCKING ISSUES** (list all).
