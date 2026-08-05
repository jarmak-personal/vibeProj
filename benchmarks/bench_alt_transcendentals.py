#!/usr/bin/env python3
"""Explore alternatives to fp64 ``atan2`` and ``asinh`` on NVIDIA GPUs.

This is an isolated experiment, not a production precision mode.  It compares
CUDA's fp64 implementations with bounded-domain polynomials, double-single
evaluation, fp32-reciprocal refinement, a deliberately low-accuracy "magic"
formula, and identities tailored to the forward Transverse Mercator (TM)
calculation.

The TM data models a normal UTM zone: latitude is in [-84, 84] degrees and the
longitude offset is in [-3, 3] degrees.  For those inputs the projection uses

    atan2(sin(B), cos(B) * cos(L))
    asinh(sin(L) * cos(B) / hypot(sin(B), cos(B) * cos(L))).

The generic atan2 domain covers the full circle, while the wider asinh domain
is [-1, 1].  Approximation errors are measured against the corresponding CUDA
fp64 kernel.  Multiplication by the WGS84 semi-major axis gives a useful upper
bound on the projected linear error, though it is not a complete TM error
propagation calculation.

Usage:
    uv run python benchmarks/bench_alt_transcendentals.py
    uv run python benchmarks/bench_alt_transcendentals.py --n 1000000 --iterations 50
    uv run python benchmarks/bench_alt_transcendentals.py --json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
from collections.abc import Callable
from typing import Any

import numpy as np


EARTH_RADIUS_M = 6_378_137.0
CUDA_OPTIONS = ("--std=c++11",)
BLOCK_SIZE = 256


CUDA_SOURCE = r"""
typedef struct { float hi; float lo; } ds_t;

__device__ __forceinline__ ds_t ds_from_float(float value) {
    ds_t result = {value, 0.0f};
    return result;
}

__device__ __forceinline__ ds_t ds_from_double(double value) {
    const float hi = (float)value;
    ds_t result = {hi, (float)(value - (double)hi)};
    return result;
}

__device__ __forceinline__ double ds_to_double(ds_t value) {
    return (double)value.hi + (double)value.lo;
}

__device__ __forceinline__ ds_t ds_quick_two_sum(float a, float b) {
    const float sum = a + b;
    ds_t result = {sum, b - (sum - a)};
    return result;
}

__device__ __forceinline__ ds_t ds_two_sum(float a, float b) {
    const float sum = a + b;
    const float virtual_b = sum - a;
    ds_t result = {sum, (a - (sum - virtual_b)) + (b - virtual_b)};
    return result;
}

__device__ __forceinline__ ds_t ds_add(ds_t a, ds_t b) {
    ds_t sum = ds_two_sum(a.hi, b.hi);
    sum.lo += a.lo + b.lo;
    return ds_quick_two_sum(sum.hi, sum.lo);
}

__device__ __forceinline__ ds_t ds_sub(ds_t a, ds_t b) {
    b.hi = -b.hi;
    b.lo = -b.lo;
    return ds_add(a, b);
}

__device__ __forceinline__ ds_t ds_mul(ds_t a, ds_t b) {
    const float product = a.hi * b.hi;
    float error = fmaf(a.hi, b.hi, -product);
    error += a.hi * b.lo + a.lo * b.hi;
    return ds_quick_two_sum(product, error);
}

__device__ __forceinline__ ds_t ds_div(ds_t a, ds_t b) {
    const float quotient_hi = a.hi / b.hi;
    const ds_t remainder = ds_sub(a, ds_mul(ds_from_float(quotient_hi), b));
    const float quotient_lo = remainder.hi / b.hi;
    return ds_quick_two_sum(quotient_hi, quotient_lo);
}

// A float reciprocal followed by one fp64 Newton step.  Its starting relative
// error is about 2^-24; the refinement normally leaves about 48 useful bits.
__device__ __forceinline__ double reciprocal_fp32_refined(double denominator) {
    double reciprocal = (double)__frcp_rn((float)denominator);
    reciprocal *= fma(-denominator, reciprocal, 2.0);
    return reciprocal;
}

// One fp32 reciprocal-square-root and one fp64 Newton step.  This is used only
// by an intentionally experimental asinh half-angle decomposition.
__device__ __forceinline__ double sqrt_fp32_refined(double value) {
    double reciprocal_sqrt = (double)rsqrtf((float)value);
    reciprocal_sqrt *= fma(-0.5 * value, reciprocal_sqrt * reciprocal_sqrt, 1.5);
    return value * reciprocal_sqrt;
}

// Degree 19 (odd) fit after reducing atan to [-tan(pi/8), tan(pi/8)].
// The coefficients are a power-basis Chebyshev least-squares fit to atan(r)/r.
__device__ __forceinline__ double atan_reduced_poly9(double r) {
    const double t = r * r;
    double polynomial = -0.0253572146897972396;
    polynomial = fma(polynomial, t, 0.0502732951054542521);
    polynomial = fma(polynomial, t, -0.0650660448350237341);
    polynomial = fma(polynomial, t, 0.0767359562075200369);
    polynomial = fma(polynomial, t, -0.0908952816824303012);
    polynomial = fma(polynomial, t, 0.111110478641898935);
    polynomial = fma(polynomial, t, -0.142857125666355722);
    polynomial = fma(polynomial, t, 0.199999999743484064);
    polynomial = fma(polynomial, t, -0.333333333331503223);
    polynomial = fma(polynomial, t, 0.999999999999996225);
    return r * polynomial;
}

__device__ __forceinline__ ds_t atan_reduced_ds_poly9(ds_t r) {
    const ds_t t = ds_mul(r, r);
    ds_t polynomial = ds_from_double(-0.0253572146897972396);
    polynomial = ds_add(ds_mul(polynomial, t), ds_from_double(0.0502732951054542521));
    polynomial = ds_add(ds_mul(polynomial, t), ds_from_double(-0.0650660448350237341));
    polynomial = ds_add(ds_mul(polynomial, t), ds_from_double(0.0767359562075200369));
    polynomial = ds_add(ds_mul(polynomial, t), ds_from_double(-0.0908952816824303012));
    polynomial = ds_add(ds_mul(polynomial, t), ds_from_double(0.111110478641898935));
    polynomial = ds_add(ds_mul(polynomial, t), ds_from_double(-0.142857125666355722));
    polynomial = ds_add(ds_mul(polynomial, t), ds_from_double(0.199999999743484064));
    polynomial = ds_add(ds_mul(polynomial, t), ds_from_double(-0.333333333331503223));
    polynomial = ds_add(ds_mul(polynomial, t), ds_from_double(0.999999999999996225));
    return ds_mul(r, polynomial);
}

__device__ __forceinline__ double atan2_poly_impl(double y, double x, bool refined_rcp) {
    const double abs_x = fabs(x);
    const double abs_y = fabs(y);
    const bool swap = abs_y > abs_x;
    const double numerator = swap ? abs_x : abs_y;
    const double denominator = swap ? abs_y : abs_x;
    if (denominator == 0.0) return copysign(0.0, y);

    double ratio = refined_rcp
        ? numerator * reciprocal_fp32_refined(denominator)
        : numerator / denominator;
    double angle;
    if (ratio > 0.4142135623730950488016887242096981) {
        const double reduced_numerator = ratio - 1.0;
        const double reduced_denominator = ratio + 1.0;
        const double reduced = refined_rcp
            ? reduced_numerator * reciprocal_fp32_refined(reduced_denominator)
            : reduced_numerator / reduced_denominator;
        angle = 0.7853981633974483096156608458198757 + atan_reduced_poly9(reduced);
    } else {
        angle = atan_reduced_poly9(ratio);
    }
    if (swap) angle = 1.5707963267948966192313216916397514 - angle;
    if (x < 0.0) angle = 3.1415926535897932384626433832795029 - angle;
    return copysign(angle, y);
}

__device__ __forceinline__ double atan2_ds_poly_impl(double y, double x) {
    ds_t abs_x = ds_from_double(fabs(x));
    ds_t abs_y = ds_from_double(fabs(y));
    const bool swap = abs_y.hi > abs_x.hi;
    ds_t numerator = swap ? abs_x : abs_y;
    ds_t denominator = swap ? abs_y : abs_x;
    if (denominator.hi == 0.0f) return copysign(0.0, y);

    ds_t ratio = ds_div(numerator, denominator);
    ds_t angle;
    if (ratio.hi > 0.4142135623730950488016887242096981f) {
        const ds_t one = ds_from_float(1.0f);
        const ds_t reduced = ds_div(ds_sub(ratio, one), ds_add(ratio, one));
        angle = ds_add(
            ds_from_double(0.7853981633974483096156608458198757),
            atan_reduced_ds_poly9(reduced)
        );
    } else {
        angle = atan_reduced_ds_poly9(ratio);
    }
    if (swap) {
        angle = ds_sub(ds_from_double(1.5707963267948966192313216916397514), angle);
    }
    if (x < 0.0) {
        angle = ds_sub(ds_from_double(3.1415926535897932384626433832795029), angle);
    }
    return copysign(ds_to_double(angle), y);
}

__device__ __forceinline__ double atan2_magic_impl(double y, double x) {
    const double abs_x = fabs(x);
    const double abs_y = fabs(y);
    const bool swap = abs_y > abs_x;
    const double numerator = swap ? abs_x : abs_y;
    const double denominator = swap ? abs_y : abs_x;
    if (denominator == 0.0) return copysign(0.0, y);
    const double ratio = numerator * reciprocal_fp32_refined(denominator);
    // A popular low-order approximation: fast, but included partly to quantify
    // how badly "magic number" accuracy misses geospatial requirements.
    double angle = 0.7853981633974483 * ratio
        - ratio * (ratio - 1.0) * (0.2447 + 0.0663 * ratio);
    if (swap) angle = 1.5707963267948966 - angle;
    if (x < 0.0) angle = 3.141592653589793 - angle;
    return copysign(angle, y);
}

__device__ __forceinline__ double asinh_taylor7_impl(double x) {
    const double t = x * x;
    double polynomial = -0.04464285714285714285714285714285714; // -5/112
    polynomial = fma(polynomial, t, 0.075);                       // 3/40
    polynomial = fma(polynomial, t, -0.1666666666666666666667);  // -1/6
    return fma(x * t, polynomial, x);
}

__device__ __forceinline__ double asinh_taylor11_impl(double x) {
    const double t = x * x;
    double polynomial = -0.02237215909090909090909090909090909; // -63/2816
    polynomial = fma(polynomial, t, 0.03038194444444444444444444444444444); // 35/1152
    polynomial = fma(polynomial, t, -0.04464285714285714285714285714285714);
    polynomial = fma(polynomial, t, 0.075);
    polynomial = fma(polynomial, t, -0.1666666666666666666667);
    return fma(x * t, polynomial, x);
}

__device__ __forceinline__ double asinh_taylor17_impl(double x) {
    const double t = x * x;
    double polynomial = 0.011551800896139705; // 6435/557056, coefficient of x^17
    polynomial = fma(polynomial, t, -0.01396484375); // -143/10240, x^15
    polynomial = fma(polynomial, t, 0.017352764423076924); // 231/13312, x^13
    polynomial = fma(polynomial, t, -0.02237215909090909090909090909090909);
    polynomial = fma(polynomial, t, 0.03038194444444444444444444444444444);
    polynomial = fma(polynomial, t, -0.04464285714285714285714285714285714);
    polynomial = fma(polynomial, t, 0.075);
    polynomial = fma(polynomial, t, -0.1666666666666666666667);
    return fma(x * t, polynomial, x);
}

__device__ __forceinline__ double asinh_rational44_impl(double x) {
    const double t = x * x;
    double numerator = 0.00211761701575277628;
    numerator = fma(numerator, t, 0.101178196340026338);
    numerator = fma(numerator, t, 0.73080839611118098);
    numerator = fma(numerator, t, 1.57158374843375892);
    numerator = fma(numerator, t, 0.999999999995441757);
    double denominator = 0.00726306839677309801;
    denominator = fma(denominator, t, 0.173038254778586392);
    denominator = fma(denominator, t, 0.945516811635670251);
    denominator = fma(denominator, t, 1.73825041461003504);
    denominator = fma(denominator, t, 1.0);
    return x * numerator / denominator;
}

__device__ __forceinline__ double asinh_half_reduce_mixed_impl(double x) {
    double reduced = x;
#pragma unroll
    for (int iteration = 0; iteration < 2; ++iteration) {
        const double cosh_value = sqrt_fp32_refined(fma(reduced, reduced, 1.0));
        const double denominator = sqrt_fp32_refined(2.0 * (cosh_value + 1.0));
        reduced *= reciprocal_fp32_refined(denominator);
    }
    return 4.0 * asinh_taylor17_impl(reduced);
}

__device__ __forceinline__ double asinh_ds_taylor11_impl(double x) {
    const ds_t value = ds_from_double(x);
    const ds_t t = ds_mul(value, value);
    ds_t polynomial = ds_from_double(-0.02237215909090909090909090909090909);
    polynomial = ds_add(ds_mul(polynomial, t), ds_from_double(0.03038194444444444444444444444444444));
    polynomial = ds_add(ds_mul(polynomial, t), ds_from_double(-0.04464285714285714285714285714285714));
    polynomial = ds_add(ds_mul(polynomial, t), ds_from_double(0.075));
    polynomial = ds_add(ds_mul(polynomial, t), ds_from_double(-0.1666666666666666666667));
    return ds_to_double(ds_add(value, ds_mul(ds_mul(value, t), polynomial)));
}

#define KERNEL_ARGS \
    const double* __restrict__ a, \
    const double* __restrict__ b, \
    const double* __restrict__ c, \
    const double* __restrict__ d, \
    const double* __restrict__ e, \
    double* __restrict__ output, int n

#define ELEMENT_INDEX \
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= n) return

extern "C" __global__ void __launch_bounds__(256) atan2_fp64(KERNEL_ARGS) {
    ELEMENT_INDEX;
    output[idx] = atan2(a[idx], b[idx]);
}

extern "C" __global__ void __launch_bounds__(256) atan2_fp32(KERNEL_ARGS) {
    ELEMENT_INDEX;
    output[idx] = (double)atan2f((float)a[idx], (float)b[idx]);
}

extern "C" __global__ void __launch_bounds__(256) atan2_native_atan_ratio(KERNEL_ARGS) {
    ELEMENT_INDEX;
    const double y = a[idx];
    const double x = b[idx];
    double angle = atan(fabs(y) / fabs(x));
    if (x < 0.0) angle = 3.1415926535897932384626433832795029 - angle;
    output[idx] = copysign(angle, y);
}

extern "C" __global__ void __launch_bounds__(256) atan2_poly_div(KERNEL_ARGS) {
    ELEMENT_INDEX;
    output[idx] = atan2_poly_impl(a[idx], b[idx], false);
}

extern "C" __global__ void __launch_bounds__(256) atan2_poly_refined_rcp(KERNEL_ARGS) {
    ELEMENT_INDEX;
    output[idx] = atan2_poly_impl(a[idx], b[idx], true);
}

extern "C" __global__ void __launch_bounds__(256) atan2_ds_poly(KERNEL_ARGS) {
    ELEMENT_INDEX;
    output[idx] = atan2_ds_poly_impl(a[idx], b[idx]);
}

extern "C" __global__ void __launch_bounds__(256) atan2_magic(KERNEL_ARGS) {
    ELEMENT_INDEX;
    output[idx] = atan2_magic_impl(a[idx], b[idx]);
}

// For |B| < pi/2 and small L:
// atan2(sin(B), cos(B)cos(L)) = B + atan(delta), where
// delta = sin(B)cos(B)(1-cos(L)) / (cos(L)cos(B)^2 + sin(B)^2).
// Here a=sin(B), b=cos(B)cos(L), c=cos(B), d=cos(L), and e=B.
extern "C" __global__ void __launch_bounds__(256) atan2_tm_delta_div(KERNEL_ARGS) {
    ELEMENT_INDEX;
    const double sin_b = a[idx];
    const double cos_b = c[idx];
    const double cos_l = d[idx];
    const double numerator = sin_b * cos_b * (1.0 - cos_l);
    const double denominator = fma(cos_l * cos_b, cos_b, sin_b * sin_b);
    const double delta = numerator / denominator;
    const double delta2 = delta * delta;
    output[idx] = e[idx] + delta * fma(delta2, fma(delta2, 0.2, -1.0 / 3.0), 1.0);
}

extern "C" __global__ void __launch_bounds__(256) atan2_tm_delta_refined_rcp(KERNEL_ARGS) {
    ELEMENT_INDEX;
    const double sin_b = a[idx];
    const double cos_b = c[idx];
    const double cos_l = d[idx];
    const double numerator = sin_b * cos_b * (1.0 - cos_l);
    const double denominator = fma(cos_l * cos_b, cos_b, sin_b * sin_b);
    const double delta = numerator * reciprocal_fp32_refined(denominator);
    const double delta2 = delta * delta;
    output[idx] = e[idx] + delta * fma(delta2, fma(delta2, 0.2, -1.0 / 3.0), 1.0);
}

// The cubic correction tests whether even degree five is unnecessary here.
extern "C" __global__ void __launch_bounds__(256) atan2_tm_delta_bounded(KERNEL_ARGS) {
    ELEMENT_INDEX;
    const double sin_b = a[idx];
    const double cos_b = c[idx];
    const double cos_l = d[idx];
    const double numerator = sin_b * cos_b * (1.0 - cos_l);
    const double denominator = fma(cos_l * cos_b, cos_b, sin_b * sin_b);
    const double delta = numerator * reciprocal_fp32_refined(denominator);
    output[idx] = e[idx] + delta - delta * delta * delta / 3.0;
}

extern "C" __global__ void __launch_bounds__(256) asinh_fp64(KERNEL_ARGS) {
    ELEMENT_INDEX;
    output[idx] = asinh(a[idx]);
}

extern "C" __global__ void __launch_bounds__(256) asinh_fp32(KERNEL_ARGS) {
    ELEMENT_INDEX;
    output[idx] = (double)asinhf((float)a[idx]);
}

extern "C" __global__ void __launch_bounds__(256) asinh_log1p_identity(KERNEL_ARGS) {
    ELEMENT_INDEX;
    const double x = a[idx];
    const double abs_x = fabs(x);
    const double root = sqrt(fma(abs_x, abs_x, 1.0));
    output[idx] = copysign(log1p(abs_x + abs_x * abs_x / (1.0 + root)), x);
}

extern "C" __global__ void __launch_bounds__(256) asinh_taylor7(KERNEL_ARGS) {
    ELEMENT_INDEX;
    output[idx] = asinh_taylor7_impl(a[idx]);
}

extern "C" __global__ void __launch_bounds__(256) asinh_taylor11(KERNEL_ARGS) {
    ELEMENT_INDEX;
    output[idx] = asinh_taylor11_impl(a[idx]);
}

extern "C" __global__ void __launch_bounds__(256) asinh_ds_taylor11(KERNEL_ARGS) {
    ELEMENT_INDEX;
    output[idx] = asinh_ds_taylor11_impl(a[idx]);
}

extern "C" __global__ void __launch_bounds__(256) asinh_rational44(KERNEL_ARGS) {
    ELEMENT_INDEX;
    output[idx] = asinh_rational44_impl(a[idx]);
}

extern "C" __global__ void __launch_bounds__(256) asinh_half_reduce_mixed(KERNEL_ARGS) {
    ELEMENT_INDEX;
    output[idx] = asinh_half_reduce_mixed_impl(a[idx]);
}
"""


ATAN_KERNELS = {
    "fp64": "atan2_fp64",
    "fp32": "atan2_fp32",
    "native_atan_ratio": "atan2_native_atan_ratio",
    "poly9_div": "atan2_poly_div",
    "poly9_refined_rcp": "atan2_poly_refined_rcp",
    "ds_poly9": "atan2_ds_poly",
    "magic_cubic": "atan2_magic",
}

TM_IDENTITY_KERNELS = {
    "tm_delta_div": "atan2_tm_delta_div",
    "tm_delta_refined_rcp": "atan2_tm_delta_refined_rcp",
    "tm_delta_rcp_cubic": "atan2_tm_delta_bounded",
}

ASINH_KERNELS = {
    "fp64": "asinh_fp64",
    "fp32": "asinh_fp32",
    "log1p_identity": "asinh_log1p_identity",
    "taylor7": "asinh_taylor7",
    "taylor11": "asinh_taylor11",
    "ds_taylor11": "asinh_ds_taylor11",
    "rational44": "asinh_rational44",
    "half_reduce_mixed": "asinh_half_reduce_mixed",
}


def _percentile(sorted_values: list[float], fraction: float) -> float:
    index = min(len(sorted_values) - 1, max(0, round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def _timing_stats(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    return {
        "median_ms": statistics.median(ordered),
        "p05_ms": _percentile(ordered, 0.05),
        "p95_ms": _percentile(ordered, 0.95),
        "min_ms": ordered[0],
    }


def _time_interleaved(
    cp: Any,
    launches: dict[str, Callable[[], None]],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, dict[str, float]]:
    for launch in launches.values():
        for _ in range(warmup):
            launch()
    cp.cuda.get_current_stream().synchronize()

    samples = {name: [] for name in launches}
    names = list(launches)
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    for iteration in range(iterations):
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            start.record()
            launches[name]()
            end.record()
            end.synchronize()
            samples[name].append(float(cp.cuda.get_elapsed_time(start, end)))
    return {name: _timing_stats(values) for name, values in samples.items()}


def _error_stats(cp: Any, values: Any, reference: Any) -> dict[str, float]:
    absolute = cp.abs(values - reference)
    return {
        "max_abs": float(cp.max(absolute).item()),
        "p99_abs": float(cp.percentile(absolute, 99).item()),
        "rms": float(cp.sqrt(cp.mean(absolute * absolute)).item()),
        "max_earth_radius_m": float(cp.max(absolute).item() * EARTH_RADIUS_M),
        "p99_earth_radius_m": float(cp.percentile(absolute, 99).item() * EARTH_RADIUS_M),
    }


def _make_launches(
    cp: Any,
    kernels: dict[str, Any],
    inputs: tuple[Any, Any, Any, Any, Any],
    outputs: dict[str, Any],
    n: int,
) -> dict[str, Callable[[], None]]:
    block = BLOCK_SIZE
    grid = max(1, (n + block - 1) // block)
    launches: dict[str, Callable[[], None]] = {}
    for name, kernel in kernels.items():
        kernel_args = (*inputs, outputs[name], np.int32(n))

        def launch(kernel: Any = kernel, kernel_args: tuple[Any, ...] = kernel_args) -> None:
            kernel((grid,), (block,), kernel_args)

        launches[name] = launch
    return launches


def _run_group(
    cp: Any,
    kernel_functions: dict[str, str],
    inputs: tuple[Any, Any, Any, Any, Any],
    *,
    reference_name: str,
    n: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    kernels = {
        name: cp.RawKernel(CUDA_SOURCE, function_name, options=CUDA_OPTIONS)
        for name, function_name in kernel_functions.items()
    }
    outputs = {name: cp.empty(n, dtype=cp.float64) for name in kernels}
    launches = _make_launches(cp, kernels, inputs, outputs, n)
    timing = _time_interleaved(cp, launches, warmup=warmup, iterations=iterations)
    for launch in launches.values():
        launch()
    cp.cuda.get_current_stream().synchronize()

    reference = outputs[reference_name]
    errors = {
        name: _error_stats(cp, output, reference)
        for name, output in outputs.items()
        if name != reference_name
    }
    reference_median = timing[reference_name]["median_ms"]
    for stats in timing.values():
        stats["mvalues_per_s"] = n / stats["median_ms"] / 1000.0
        stats["speedup_vs_fp64"] = reference_median / stats["median_ms"]
    return {"timing": timing, "error_vs_fp64": errors}


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - benchmark environment guard
        raise SystemExit("CuPy is required for this GPU benchmark") from exc

    device = cp.cuda.Device(args.device)
    device.use()
    properties = cp.cuda.runtime.getDeviceProperties(args.device)
    gpu_name = properties["name"].decode()

    rng = np.random.default_rng(args.seed)
    radius = rng.uniform(0.5, 1.5, args.n)
    angle = rng.uniform(-math.pi, math.pi, args.n)
    full_y_host = (radius * np.sin(angle)).astype(np.float64)
    full_x_host = (radius * np.cos(angle)).astype(np.float64)
    full_edges = np.array(
        [
            (-1.0, 0.0),
            (1.0, 0.0),
            (0.0, -1.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (-1.0, -1.0),
            (np.nextafter(0.0, 1.0), -1.0),
        ],
        dtype=np.float64,
    )
    edge_count = min(args.n, len(full_edges))
    full_y_host[:edge_count] = full_edges[:edge_count, 0]
    full_x_host[:edge_count] = full_edges[:edge_count, 1]

    b_host = np.deg2rad(rng.uniform(-84.0, 84.0, args.n)).astype(np.float64)
    l_host = np.deg2rad(rng.uniform(-3.0, 3.0, args.n)).astype(np.float64)
    sin_b_host = np.sin(b_host)
    cos_b_host = np.cos(b_host)
    cos_l_host = np.cos(l_host)
    tm_x_host = cos_b_host * cos_l_host
    tm_asinh_host = np.sin(l_host) * cos_b_host / np.hypot(sin_b_host, tm_x_host)
    tm_edges = np.array(
        [
            (-math.radians(84.0), -math.radians(3.0)),
            (-math.radians(84.0), math.radians(3.0)),
            (0.0, -math.radians(3.0)),
            (0.0, math.radians(3.0)),
            (math.radians(84.0), -math.radians(3.0)),
            (math.radians(84.0), math.radians(3.0)),
            (0.0, 0.0),
        ]
    )
    edge_count = min(args.n, len(tm_edges))
    if edge_count:
        edge_b = tm_edges[:edge_count, 0]
        edge_l = tm_edges[:edge_count, 1]
        b_host[:edge_count] = edge_b
        l_host[:edge_count] = edge_l
        sin_b_host[:edge_count] = np.sin(edge_b)
        cos_b_host[:edge_count] = np.cos(edge_b)
        cos_l_host[:edge_count] = np.cos(edge_l)
        tm_x_host[:edge_count] = cos_b_host[:edge_count] * cos_l_host[:edge_count]
        tm_asinh_host[:edge_count] = (
            np.sin(edge_l)
            * cos_b_host[:edge_count]
            / np.hypot(sin_b_host[:edge_count], tm_x_host[:edge_count])
        )

    zero_host = np.zeros(args.n, dtype=np.float64)
    wide_asinh_host = rng.uniform(-1.0, 1.0, args.n).astype(np.float64)
    wide_edges = np.array([-1.0, 1.0, -0.0, 0.0, np.nextafter(0.0, 1.0)])
    wide_asinh_host[: min(args.n, len(wide_edges))] = wide_edges[: args.n]

    full_inputs = (
        cp.asarray(full_y_host),
        cp.asarray(full_x_host),
        cp.asarray(zero_host),
        cp.asarray(zero_host),
        cp.asarray(zero_host),
    )
    tm_inputs = (
        cp.asarray(sin_b_host),
        cp.asarray(tm_x_host),
        cp.asarray(cos_b_host),
        cp.asarray(cos_l_host),
        cp.asarray(b_host),
    )
    tm_asinh_values = cp.asarray(tm_asinh_host)
    wide_asinh_values = cp.asarray(wide_asinh_host)
    tm_asinh_inputs = (tm_asinh_values, tm_inputs[1], tm_inputs[2], tm_inputs[3], tm_inputs[4])
    wide_asinh_inputs = (
        wide_asinh_values,
        full_inputs[1],
        full_inputs[2],
        full_inputs[3],
        full_inputs[4],
    )

    atan_full = _run_group(
        cp,
        ATAN_KERNELS,
        full_inputs,
        reference_name="fp64",
        n=args.n,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    atan_tm = _run_group(
        cp,
        ATAN_KERNELS | TM_IDENTITY_KERNELS,
        tm_inputs,
        reference_name="fp64",
        n=args.n,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    asinh_tm = _run_group(
        cp,
        ASINH_KERNELS,
        tm_asinh_inputs,
        reference_name="fp64",
        n=args.n,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    asinh_wide = _run_group(
        cp,
        ASINH_KERNELS,
        wide_asinh_inputs,
        reference_name="fp64",
        n=args.n,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    return {
        "meta": {
            "gpu": gpu_name,
            "compute_capability": f"{properties['major']}.{properties['minor']}",
            "python": platform.python_version(),
            "cupy": cp.__version__,
            "n_values": args.n,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "seed": args.seed,
            "cuda_options": list(CUDA_OPTIONS),
            "compiler": "CuPy RawKernel with NVRTC",
            "block_size": BLOCK_SIZE,
            "atan_full_domain": "full circle; radius in [0.5, 1.5]",
            "tm_domain": "B in [-84, 84] deg; L in [-3, 3] deg",
            "tm_max_abs_asinh_input": float(np.max(np.abs(tm_asinh_host))),
            "asinh_wide_domain": [-1.0, 1.0],
            "earth_radius_m": EARTH_RADIUS_M,
        },
        "atan2_full_circle": atan_full,
        "atan2_tm": atan_tm,
        "asinh_tm": asinh_tm,
        "asinh_wide": asinh_wide,
    }


def _print_group(title: str, group: dict[str, Any]) -> None:
    print(title)
    print(
        f"{'mode':<24} {'median ms':>10} {'Mvalue/s':>10} {'vs fp64':>9} {'max abs':>13} {'p99 abs':>13} {'max Earth m':>13}"
    )
    errors = group["error_vs_fp64"]
    for name, timing in group["timing"].items():
        error = errors.get(name)
        if error is None:
            max_abs = p99_abs = earth = "reference"
        else:
            max_abs = f"{error['max_abs']:.4g}"
            p99_abs = f"{error['p99_abs']:.4g}"
            earth = f"{error['max_earth_radius_m']:.4g}"
        print(
            f"{name:<24} {timing['median_ms']:>10.4f} {timing['mvalues_per_s']:>10.1f} "
            f"{timing['speedup_vs_fp64']:>8.2f}x {max_abs:>13} {p99_abs:>13} {earth:>13}"
        )
    print()


def _print_human(results: dict[str, Any]) -> None:
    meta = results["meta"]
    sm = meta["compute_capability"].replace(".", "")
    print(f"GPU: {meta['gpu']} (sm_{sm}); CuPy {meta['cupy']}; options {meta['cuda_options']}")
    print(
        f"{meta['n_values']:,} values; {meta['iterations']} iterations; "
        f"TM max |asinh input|={meta['tm_max_abs_asinh_input']:.8f}"
    )
    print("Earth error is angular error multiplied by 6,378,137 m.")
    print()
    _print_group("atan2: full-circle synthetic domain", results["atan2_full_circle"])
    _print_group("atan2: TM/UTM domain", results["atan2_tm"])
    _print_group("asinh: TM/UTM domain", results["asinh_tm"])
    _print_group("asinh: wider [-1, 1] domain", results["asinh_wide"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5_000_000, help="number of values per domain")
    parser.add_argument("--iterations", type=int, default=30, help="timed iterations per mode")
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations per mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    if args.n <= 0 or args.iterations <= 0 or args.warmup < 0:
        parser.error("n and iterations must be positive; warmup must be non-negative")

    results = run(args)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_human(results)


if __name__ == "__main__":
    main()
