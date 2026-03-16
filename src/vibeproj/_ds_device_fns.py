"""Double-single (ds) arithmetic CUDA device functions.

A ds number is a pair of floats (hi, lo) where value = hi + lo.
This gives ~48 bits of mantissa (~14 decimal digits) using only fp32 operations.
FMA (fused multiply-add) is the key primitive — 1 cycle on NVIDIA GPUs.

The cost of ds operations vs plain fp32:
  ds_add: ~6 fp32 ops    (vs 1)
  ds_mul: ~4 fp32 ops    (vs 1)  — uses FMA for exact error
  ds_div: ~10 fp32 ops   (vs 1)

With a 1:64 fp64:fp32 ratio on consumer GPUs, ds operations give:
  ds_add: 64/6 ≈ 10x faster than fp64 add
  ds_mul: 64/4 ≈ 16x faster than fp64 mul

References:
  - Dekker (1971): "A floating-point technique for extending the available precision"
  - Shewchuk (1997): "Adaptive Precision Floating-Point Arithmetic"
  - NVIDIA GPU Computing Gems: "Double precision on the GPU using double-single"
"""

# fmt: off

DS_DEVICE_FNS = """
// ---- Double-single (ds) arithmetic ----
// A ds number: value = hi + lo, where |lo| <= 0.5 ULP of |hi|.
// Provides ~48 bits of mantissa using fp32 ops + FMA.

typedef struct {{ float hi; float lo; }} ds_t;

__device__ inline ds_t ds_from_float(float a) {{
    ds_t r; r.hi = a; r.lo = 0.0f; return r;
}}

__device__ inline ds_t ds_from_double(double a) {{
    ds_t r;
    r.hi = (float)a;
    r.lo = (float)(a - (double)r.hi);
    return r;
}}

__device__ inline double ds_to_double(ds_t a) {{
    return (double)a.hi + (double)a.lo;
}}

// Two-Sum: exact sum of two floats → (s, err) where a + b = s + err exactly.
__device__ inline ds_t ds_two_sum(float a, float b) {{
    float s = a + b;
    float v = s - a;
    ds_t r;
    r.hi = s;
    r.lo = (a - (s - v)) + (b - v);
    return r;
}}

// Quick-Two-Sum: requires |a| >= |b|.
__device__ inline ds_t ds_quick_two_sum(float a, float b) {{
    float s = a + b;
    ds_t r;
    r.hi = s;
    r.lo = b - (s - a);
    return r;
}}

// Two-Product: exact product using FMA → (p, err) where a * b = p + err exactly.
__device__ inline ds_t ds_two_product(float a, float b) {{
    float p = a * b;
    ds_t r;
    r.hi = p;
    r.lo = fmaf(a, b, -p);
    return r;
}}

// DS addition: (a.hi + a.lo) + (b.hi + b.lo)
__device__ inline ds_t ds_add(ds_t a, ds_t b) {{
    ds_t s = ds_two_sum(a.hi, b.hi);
    s.lo += a.lo + b.lo;
    // Renormalize
    ds_t r = ds_quick_two_sum(s.hi, s.lo);
    return r;
}}

// DS subtraction
__device__ inline ds_t ds_sub(ds_t a, ds_t b) {{
    ds_t nb; nb.hi = -b.hi; nb.lo = -b.lo;
    return ds_add(a, nb);
}}

// DS multiplication: (a.hi + a.lo) * (b.hi + b.lo)
__device__ inline ds_t ds_mul(ds_t a, ds_t b) {{
    ds_t p = ds_two_product(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    ds_t r = ds_quick_two_sum(p.hi, p.lo);
    return r;
}}

// DS division: a / b (approximate — uses Newton refinement)
__device__ inline ds_t ds_div(ds_t a, ds_t b) {{
    float q1 = a.hi / b.hi;
    // Compute residual: a - q1 * b
    ds_t qb = ds_mul(ds_from_float(q1), b);
    ds_t rem = ds_sub(a, qb);
    float q2 = rem.hi / b.hi;
    ds_t r = ds_quick_two_sum(q1, q2);
    return r;
}}

// DS absolute value
__device__ inline ds_t ds_abs(ds_t a) {{
    if (a.hi < 0.0f) {{ a.hi = -a.hi; a.lo = -a.lo; }}
    return a;
}}

// DS comparison
__device__ inline int ds_gt(ds_t a, ds_t b) {{
    return (a.hi > b.hi) || (a.hi == b.hi && a.lo > b.lo);
}}

// ---- DS transcendental functions ----
// Strategy: use fp64 hardware for transcendentals (sin, cos, atan2, etc.)
// The SFU-based fp64 trig is only ~4x slower than fp32 (not 64x).
// The big win is ds arithmetic for the Clenshaw recurrence (64x fp32 throughput).

#define DS_TWOPI_HI 6.2831855f
#define DS_TWOPI_LO -1.7484556e-07f

__device__ inline ds_t ds_wrap_to_pi(ds_t a) {{
    float n = rintf(a.hi * 0.15915494f);
    ds_t twopi; twopi.hi = DS_TWOPI_HI; twopi.lo = DS_TWOPI_LO;
    return ds_sub(a, ds_mul(ds_from_float(n), twopi));
}}

// All transcendentals: convert to fp64, compute, convert back.
// This gives exact results (fp64 precision) at ~4x fp32 cost (SFU).
__device__ inline ds_t ds_sin(ds_t a) {{ return ds_from_double(sin(ds_to_double(a))); }}
__device__ inline ds_t ds_cos(ds_t a) {{ return ds_from_double(cos(ds_to_double(a))); }}
__device__ inline ds_t ds_atan2(ds_t y, ds_t x) {{ return ds_from_double(atan2(ds_to_double(y), ds_to_double(x))); }}
__device__ inline ds_t ds_asinh(ds_t x) {{ return ds_from_double(asinh(ds_to_double(x))); }}
__device__ inline ds_t ds_exp(ds_t x) {{ return ds_from_double(exp(ds_to_double(x))); }}
__device__ inline ds_t ds_log(ds_t x) {{ return ds_from_double(log(ds_to_double(x))); }}
__device__ inline ds_t ds_sinh(ds_t x) {{ return ds_from_double(sinh(ds_to_double(x))); }}
__device__ inline ds_t ds_cosh(ds_t x) {{ return ds_from_double(cosh(ds_to_double(x))); }}

__device__ inline ds_t ds_sqrt(ds_t a) {{
    float s = sqrtf(a.hi);
    ds_t sd = ds_from_float(s);
    ds_t q = ds_div(a, sd);
    ds_t sum = ds_add(sd, q);
    ds_t r; r.hi = sum.hi * 0.5f; r.lo = sum.lo * 0.5f;
    return r;
}}

__device__ inline ds_t ds_hypot(ds_t x, ds_t y) {{
    return ds_sqrt(ds_add(ds_mul(x, x), ds_mul(y, y)));
}}
"""

# fmt: on
