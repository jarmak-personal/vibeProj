---
name: cuda-optimizer
description: "Use this skill to optimize existing fused NVRTC kernel code, GPU dispatch logic, or the Helmert datum shift kernel in src/vibeproj/. Reads existing code and produces concrete rewrites with measured justification. Invoke on pre-existing kernel code to bring it up to NVIDIA best-practice performance standards."
user-invocable: true
argument-hint: <file-path or projection-name to optimize>
---

# CUDA Optimizer — vibeProj

You are optimizing GPU code in vibeProj. Your job is to read the target
file, identify every concrete optimization opportunity, and produce
ready-to-apply rewrites ranked by expected impact.

**Target:** `$ARGUMENTS`

---

## Procedure

### Step 0: Read the Code

Read the target file completely. Also read `fused_kernels.py` if not
already the target. Identify:

- All NVRTC kernel source strings (look for `_SOURCE` or multi-line strings
  containing `__global__`)
- All CuPy operations (`cp.` calls, `RawKernel`)
- All host-device transfers (`.get()`, `cp.asnumpy()`, `cp.asarray()`)
- All synchronization points
- The `fused_transform()` dispatch function
- The preamble/postamble system (`_FWD_PREAMBLE`, `_INV_PREAMBLE`, etc.)

### Step 1: Host-Device Boundary (CRITICAL — highest impact)

Scan for these patterns and produce rewrites:

**1a. D->H->D ping-pong in transform paths**
```python
# BEFORE: host round-trip between Helmert and projection
x_np = cp.asnumpy(x_shifted)
x_proj, y_proj = np_forward(x_np, y_np, ...)
result_x = cp.asarray(x_proj)

# AFTER: stay on device
fused_transform(x_shifted, y_shifted, out_x, out_y, ...)
```

**1b. Python loops over device arrays**
```python
# BEFORE: element-wise Python loop
for i in range(n):
    result[i] = transform_one(x[i].get(), y[i].get())

# AFTER: single fused kernel launch
fused_transform(x, y, out_x, out_y, ...)
```

**1c. Mid-pipeline `.get()` / `cp.asnumpy()`**

Any `.get()` or `cp.asnumpy()` that is NOT at the final return of
`transform()` is suspect. Check if the host value feeds back into GPU
work. If yes, rewrite to keep it on device.

### Step 2: Synchronization Elimination (HIGH impact)

**2a. Unnecessary syncs between fused kernel launches**

If Helmert + projection run on the same (null) stream, CUDA guarantees
order. No sync needed between them.

```python
# BEFORE
helmert_kernel(...)
cp.cuda.Stream.null.synchronize()  # REMOVE
projection_kernel(...)

# AFTER
helmert_kernel(...)
projection_kernel(...)  # same stream, execution order guaranteed
```

**2b. Implicit syncs from scalar reads**

Flag: `int(cupy_scalar)`, `float(cupy_scalar)`, `print(cupy_array)`,
`len()` that triggers `.get()`.

### Step 3: Kernel Source Optimization (HIGH impact)

Read every NVRTC kernel source string and check:

**3a. `const double* __restrict__` on read-only pointers**
```c
// BEFORE
extern "C" __global__ void kernel(double* in_x, double* in_y, double* out_x, ...)

// AFTER
extern "C" __global__ void kernel(
    const double* __restrict__ in_x,
    const double* __restrict__ in_y,
    double* __restrict__ out_x, ...)
```
Enables `__ldg` read-only cache path (CC 3.5+).

**3b. `__launch_bounds__` directive**
```c
// BEFORE
extern "C" __global__ void kernel(...)

// AFTER
extern "C" __global__ void __launch_bounds__(256, 4) kernel(...)
```
Without it, the compiler may over-allocate registers.

**3c. Float constant precision**

In fp32 kernels (`{real_t}` = `float`), check for unqualified float
constants (`0.0`, `1.0`, `1e-7`) which compile as doubles and force
conversion. Use explicit casts: `({real_t})0.0`.

**3d. Integer division/modulo optimization**

`threadIdx.x / 32` -> `threadIdx.x >> 5`
`threadIdx.x % 32` -> `threadIdx.x & 31`

Integer div/mod compiles to up to 20 instructions when the divisor is
not a compile-time power of 2.

**3e. Iterative convergence loops**

Many projections (LCC, AEA, Stereographic) use Newton-Raphson or similar
iteration. Check:
- Convergence tolerance matches precision: `({real_t})1e-14` for fp64,
  `({real_t})1e-7` for fp32
- Max iteration count is bounded (15 is standard)
- Early exit on convergence (`if (fabs(dphi) < tol) break;`)

**3f. Device function inlining**

All shared device functions (`gatg`, `clenshaw_complex`, `tsfn`, `phi2`,
`qsfn`, `phi_from_q`) should be `__device__ inline`. Without `inline`,
the compiler may not inline small functions, adding call overhead.

### Step 4: Launch Configuration (MEDIUM impact)

**4a. Block size**

vibeProj uses 256 threads/block. Verify this is correct for the register
pressure of each kernel. Kernels with many local variables (e.g., Transverse
Mercator with 20+ `{real_t}` locals) may benefit from lower block size to
reduce register spilling.

**4b. Grid size**

Verify: `grid = (n + 255) // 256`. For small n (< 1024), the GPU is
underutilized — consider batching multiple small transforms into one
launch (vibeProj currently doesn't, but it could for vibeSpatial
integration where many small geometries transform independently).

### Step 5: Memory Access Patterns (MEDIUM impact)

**5a. Coalesced access**

vibeProj uses separate `in_x[]`, `in_y[]` arrays (SoA layout) — this is
already optimal. Flag any change that interleaves x/y into a single array
(AoS layout is 5.9x slower due to strided access).

**5b. L2 cache for Helmert + projection pipeline**

If Helmert output feeds directly into the projection kernel on the same
data, the second kernel benefits from L2 cache residency of the first
kernel's output. Ensure no unnecessary sync or allocation between them
that would evict the cache.

### Step 6: Precision Dispatch (MEDIUM impact)

Check that the `{real_t}` parameterization is correct:

- Does the kernel source use `{real_t}` consistently for compute variables?
- Are scale/offset operations in fp64?
- Does the cache key include the dtype dimension?
- For projections with iterative solvers: does the convergence tolerance
  adapt to precision?

---

## Output Format

For each finding, produce:

```
### [PRIORITY] Finding Title

**File:** `path/to/file.py:LINE`
**Impact:** Brief explanation of why this matters
**Category:** (host-device | sync | kernel-source | launch-config | memory-access | precision)

**Before:**
```python (or c)
<exact current code>
```

**After:**
```python (or c)
<concrete rewrite>
```
```

Sort findings by priority: CRITICAL > HIGH > MEDIUM > LOW.

Summary table at the end:

```
| # | Priority | Category | File:Line | Finding | Est. Impact |
|---|----------|----------|-----------|---------|-------------|
```

---

## Rules

- **Read before recommending.** Never suggest changes to code you
  haven't read. Always quote the exact current code in "Before."
- **Concrete rewrites only.** Every finding must have a "Before" and
  "After" block. No vague advice like "consider optimizing."
- **Don't break correctness.** Precision changes must go through
  `{real_t}`. Don't change `double` to `float` directly.
- **One file at a time.** Note cross-file findings but don't rewrite
  imported modules without being asked.
- **Respect existing patterns.** Use the preamble/postamble system,
  `_SOURCE_MAP`, `_SUPPORTED`, and `fused_transform()` conventions.
