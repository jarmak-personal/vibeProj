#!/usr/bin/env python3
"""Benchmark UTM forward precision strategies on an NVIDIA GPU.

This is an intentionally isolated experiment rather than a public vibeproj
precision mode.  It compares the production fp64, fp32, and double-single
kernels with two hybrid experiments: double-single arithmetic with table-free
Q1.62 sine/cosine, and a Q11.52 signed-INT64 arithmetic kernel.  Atan2 and
asinh remain fp64 in both hybrids.

Q11.52 has a step of 2^-52 in the dimensionless projection domain (about
1.4 nanometres at the WGS84 semi-major axis) and a range of [-2048, 2048).
That range is ample inside a UTM zone, but it is not a general replacement for
floating point outside the projection's normal area of use.

Usage:
    uv run python benchmarks/bench_tmerc_int64.py
    uv run python benchmarks/bench_tmerc_int64.py --n 1000000 --iterations 50
    uv run python benchmarks/bench_tmerc_int64.py --wide-domain
    uv run python benchmarks/bench_tmerc_int64.py --json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from collections.abc import Callable
from typing import Any

import numpy as np


Q52_FRAC_BITS = 52
Q52_SCALE = 1 << Q52_FRAC_BITS


# The FP64 transcendental calls are deliberate: UTM cannot be expressed as a
# fixed-point-only kernel without replacing sin/cos/atan2/asinh too.  This test
# targets the ordinary arithmetic between those calls, where consumer NVIDIA
# GPUs have very low native FP64 throughput.
Q52_TM_FORWARD_SOURCE = r"""
typedef long long q52_t;
typedef unsigned long long uq64_t;

#define Q52_FRAC_BITS 52
#define Q52_SCALE_D 4503599627370496.0
#define Q52_INV_SCALE_D 2.220446049250313080847263336181640625e-16
#define Q52_ONE ((q52_t)4503599627370496LL)

__device__ __forceinline__ q52_t q52_from_double(double value) {
    return __double2ll_rn(value * Q52_SCALE_D);
}

__device__ __forceinline__ double q52_to_double(q52_t value) {
    return (double)value * Q52_INV_SCALE_D;
}

__device__ __forceinline__ q52_t q52_add(q52_t a, q52_t b) {
    return a + b;
}

__device__ __forceinline__ q52_t q52_sub(q52_t a, q52_t b) {
    return a - b;
}

// Rounded (a * b) >> 52 using the full 128-bit unsigned product.  CUDA exposes
// the high half as __umul64hi; no native or compiler __int128 is required.
__device__ __forceinline__ q52_t q52_mul(q52_t a, q52_t b) {
    const bool negative = (a < 0) != (b < 0);
    const uq64_t ua = a < 0 ? 0ULL - (uq64_t)a : (uq64_t)a;
    const uq64_t ub = b < 0 ? 0ULL - (uq64_t)b : (uq64_t)b;

    uq64_t lo = ua * ub;
    uq64_t hi = __umul64hi(ua, ub);

    // Round the magnitude to nearest before the fixed-point shift.
    const uq64_t rounded_lo = lo + (1ULL << (Q52_FRAC_BITS - 1));
    hi += rounded_lo < lo;
    const uq64_t magnitude = (hi << (64 - Q52_FRAC_BITS)) | (rounded_lo >> Q52_FRAC_BITS);
    const q52_t result = (q52_t)magnitude;
    return negative ? -result : result;
}

__device__ __forceinline__ q52_t q52_sin(q52_t a) {
    return q52_from_double(sin(q52_to_double(a)));
}

__device__ __forceinline__ q52_t q52_cos(q52_t a) {
    return q52_from_double(cos(q52_to_double(a)));
}

__device__ __forceinline__ q52_t q52_atan2(q52_t y, q52_t x) {
    return q52_from_double(atan2(q52_to_double(y), q52_to_double(x)));
}

__device__ __forceinline__ q52_t q52_asinh(q52_t a) {
    return q52_from_double(asinh(q52_to_double(a)));
}

__device__ __forceinline__ q52_t q52_inv_hypot(q52_t x, q52_t y) {
    const double xd = q52_to_double(x);
    const double yd = q52_to_double(y);
    return q52_from_double(1.0 / hypot(xd, yd));
}

__device__ __forceinline__ q52_t q52_gatg(
    q52_t p0, q52_t p1, q52_t p2, q52_t p3, q52_t p4, q52_t p5,
    q52_t B, q52_t cos_2B, q52_t sin_2B
) {
    const q52_t two_cos_2B = q52_add(cos_2B, cos_2B);
    q52_t h2 = 0;
    q52_t h1 = p5;
    q52_t h;
    h = q52_add(q52_sub(q52_mul(two_cos_2B, h1), h2), p4); h2 = h1; h1 = h;
    h = q52_add(q52_sub(q52_mul(two_cos_2B, h1), h2), p3); h2 = h1; h1 = h;
    h = q52_add(q52_sub(q52_mul(two_cos_2B, h1), h2), p2); h2 = h1; h1 = h;
    h = q52_add(q52_sub(q52_mul(two_cos_2B, h1), h2), p1); h2 = h1; h1 = h;
    h = q52_add(q52_sub(q52_mul(two_cos_2B, h1), h2), p0); h2 = h1; h1 = h;
    return q52_add(B, q52_mul(h, sin_2B));
}

__device__ __forceinline__ void q52_clenshaw_complex(
    q52_t a0, q52_t a1, q52_t a2, q52_t a3, q52_t a4, q52_t a5,
    q52_t sin_r, q52_t cos_r, q52_t sinh_i, q52_t cosh_i,
    q52_t* out_R, q52_t* out_I
) {
    const q52_t two_cos_r = q52_add(cos_r, cos_r);
    q52_t r = q52_mul(two_cos_r, cosh_i);
    q52_t im = -q52_mul(q52_add(sin_r, sin_r), sinh_i);
    q52_t hr = a5;
    q52_t hi = 0;
    q52_t hr1 = 0;
    q52_t hi1 = 0;
    q52_t hr2;
    q52_t hi2;

#define Q52_CLEN_STEP(coeff) \
    hr2 = hr1; hi2 = hi1; hr1 = hr; hi1 = hi; \
    hr = q52_add(q52_sub(q52_sub(q52_mul(r, hr1), q52_mul(im, hi1)), hr2), coeff); \
    hi = q52_sub(q52_add(q52_mul(im, hr1), q52_mul(r, hi1)), hi2);
    Q52_CLEN_STEP(a4)
    Q52_CLEN_STEP(a3)
    Q52_CLEN_STEP(a2)
    Q52_CLEN_STEP(a1)
    Q52_CLEN_STEP(a0)
#undef Q52_CLEN_STEP

    r = q52_mul(sin_r, cosh_i);
    im = q52_mul(cos_r, sinh_i);
    *out_R = q52_sub(q52_mul(r, hr), q52_mul(im, hi));
    *out_I = q52_add(q52_mul(r, hi), q52_mul(im, hr));
}

extern "C" __global__ void __launch_bounds__(256) tm_forward_q52(
    const double* __restrict__ in_x,
    const double* __restrict__ in_y,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    q52_t cbg0, q52_t cbg1, q52_t cbg2, q52_t cbg3, q52_t cbg4, q52_t cbg5,
    q52_t gtu0, q52_t gtu1, q52_t gtu2, q52_t gtu3, q52_t gtu4, q52_t gtu5,
    q52_t Qn, q52_t Zb,
    double lam0, double a, double x0, double y0,
    double x_unit_to_m, double y_unit_to_m,
    int src_north_first, int dst_north_first, int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const double d1 = in_x[idx];
    const double d2 = in_y[idx];
    const double d_lat = src_north_first ? d1 : d2;
    const double d_lon = src_north_first ? d2 : d1;

    q52_t phi = q52_from_double(d_lat * 0.017453292519943295);
    double lam_d = d_lon * 0.017453292519943295 - lam0;
    lam_d -= 6.283185307179586476925286766559 * nearbyint(lam_d * 0.159154943091895335768883763373);
    q52_t lam = q52_from_double(lam_d);

    const q52_t two_phi = q52_add(phi, phi);
    q52_t Cn = q52_gatg(
        cbg0, cbg1, cbg2, cbg3, cbg4, cbg5,
        phi, q52_cos(two_phi), q52_sin(two_phi)
    );

    const q52_t sin_Cn = q52_sin(Cn);
    const q52_t cos_Cn = q52_cos(Cn);
    const q52_t sin_Ce = q52_sin(lam);
    const q52_t cos_Ce = q52_cos(lam);
    const q52_t cos_Cn_cos_Ce = q52_mul(cos_Cn, cos_Ce);
    Cn = q52_atan2(sin_Cn, cos_Cn_cos_Ce);
    const q52_t inv_denom = q52_inv_hypot(sin_Cn, cos_Cn_cos_Ce);
    const q52_t tan_Ce = q52_mul(q52_mul(sin_Ce, cos_Cn), inv_denom);
    q52_t Ce = q52_asinh(tan_Ce);

    const q52_t two_inv = q52_add(inv_denom, inv_denom);
    const q52_t two_inv_sq = q52_mul(two_inv, inv_denom);
    const q52_t tmp_r = q52_mul(cos_Cn_cos_Ce, two_inv_sq);
    const q52_t sin_arg_r = q52_mul(sin_Cn, tmp_r);
    const q52_t cos_arg_r = q52_sub(q52_mul(cos_Cn_cos_Ce, tmp_r), Q52_ONE);
    const q52_t sinh_arg_i = q52_mul(tan_Ce, two_inv);
    const q52_t cosh_arg_i = q52_sub(two_inv_sq, Q52_ONE);

    q52_t dCn;
    q52_t dCe;
    q52_clenshaw_complex(
        gtu0, gtu1, gtu2, gtu3, gtu4, gtu5,
        sin_arg_r, cos_arg_r, sinh_arg_i, cosh_arg_i, &dCn, &dCe
    );
    Cn = q52_add(Cn, dCn);
    Ce = q52_add(Ce, dCe);

    const double easting = q52_to_double(q52_mul(Qn, Ce)) * a + x0;
    const double northing = q52_to_double(q52_add(q52_mul(Qn, Cn), Zb)) * a + y0;
    const double easting_out = easting / x_unit_to_m;
    const double northing_out = northing / y_unit_to_m;

    if (dst_north_first) {
        out_x[idx] = northing_out;
        out_y[idx] = easting_out;
    } else {
        out_x[idx] = easting_out;
        out_y[idx] = northing_out;
    }
}
"""


def _q52(value: float) -> np.int64:
    """Convert a bounded host scalar to Q11.52 with round-to-nearest."""
    scaled = round(float(value) * Q52_SCALE)
    if not -(1 << 63) <= scaled < (1 << 63):
        raise OverflowError(f"{value!r} is outside Q11.52 range")
    return np.int64(scaled)


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


def _error_stats(
    x: np.ndarray, y: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray
) -> dict[str, float]:
    dx = x - ref_x
    dy = y - ref_y
    radial = np.hypot(dx, dy)
    return {
        "max_component": float(max(np.max(np.abs(dx)), np.max(np.abs(dy)))),
        "max_radial": float(np.max(radial)),
        "p99_radial": float(np.percentile(radial, 99)),
        "rms_radial": float(np.sqrt(np.mean(radial * radial))),
    }


def _make_q52_launch(
    cp: Any,
    kernel: Any,
    computed: dict[str, Any],
    lat: Any,
    lon: Any,
    out_x: Any,
    out_y: Any,
) -> Callable[[], None]:
    n = lat.size
    block_size = 256
    grid_size = max(1, (n + block_size - 1) // block_size)
    args = (
        lat,
        lon,
        out_x,
        out_y,
        *(_q52(value) for value in computed["cbg"]),
        *(_q52(value) for value in computed["gtu"]),
        _q52(computed["Qn"]),
        _q52(computed["Zb"]),
        np.float64(computed["lam0"]),
        np.float64(computed["a"]),
        np.float64(computed["x0"]),
        np.float64(computed["y0"]),
        np.float64(computed.get("x_unit_to_m", 1.0)),
        np.float64(computed.get("y_unit_to_m", 1.0)),
        np.int32(True),
        np.int32(False),
        np.int32(n),
    )

    def launch() -> None:
        kernel((grid_size,), (block_size,), args)

    return launch


def _make_ds_trig_source(strategy: str) -> tuple[str, str]:
    """Build an isolated DS TM kernel with paired native or Q1.62 trig."""
    from vibeproj._fixed_trig_device_fns import FIXED_TRIG_DEVICE_FNS
    from vibeproj.fused_kernels import _TM_FORWARD_DS_SOURCE, _inject_linear_unit_args

    if strategy not in ("native", "q62"):
        raise ValueError(f"unknown DS trig strategy: {strategy!r}")

    source = _inject_linear_unit_args(_TM_FORWARD_DS_SOURCE.format())
    kernel_name = f"tm_forward_ds_{strategy}_trig"
    kernel_needle = 'extern "C" __global__ void __launch_bounds__(256) tm_forward_ds('

    if strategy == "q62":
        prefix = FIXED_TRIG_DEVICE_FNS
        trig_call = "vp_fixed_sincos_bounded(angle, &sin_value, &cos_value);"
    else:
        prefix = ""
        trig_call = "sincos(angle, &sin_value, &cos_value);"

    wrapper = f"""
__device__ __forceinline__ void ds_bench_sincos(
    ds_t input, ds_t* sin_out, ds_t* cos_out
) {{
    const double angle = ds_to_double(input);
    double sin_value;
    double cos_value;
    {trig_call}
    *sin_out = ds_from_double(sin_value);
    *cos_out = ds_from_double(cos_value);
}}

"""
    if kernel_needle not in source:
        raise RuntimeError("DS TM kernel signature changed")
    source = source.replace(
        kernel_needle,
        wrapper + f'extern "C" __global__ void __launch_bounds__(256) {kernel_name}(',
        1,
    )

    replacements = {
        "ds_t Cn = ds_gatg(c0,c1,c2,c3,c4,c5, phi, ds_cos(two_phi), ds_sin(two_phi));": (
            "ds_t sin_two_phi, cos_two_phi;\n"
            "    ds_bench_sincos(two_phi, &sin_two_phi, &cos_two_phi);\n"
            "    ds_t Cn = ds_gatg(c0,c1,c2,c3,c4,c5, phi, cos_two_phi, sin_two_phi);"
        ),
        "ds_t sin_Cn = ds_sin(Cn), cos_Cn = ds_cos(Cn);": (
            "ds_t sin_Cn, cos_Cn;\n    ds_bench_sincos(Cn, &sin_Cn, &cos_Cn);"
        ),
        "ds_t sin_Ce = ds_sin(lam), cos_Ce = ds_cos(lam);": (
            "ds_t sin_Ce, cos_Ce;\n    ds_bench_sincos(lam, &sin_Ce, &cos_Ce);"
        ),
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"DS TM trig call site changed: {old}")
        source = source.replace(old, new, 1)
    return prefix + source, kernel_name


def _make_fp64_experiment_source(strategy: str) -> tuple[str, str]:
    """Build paired and TM-domain-reframed fp64 comparison kernels."""
    from vibeproj._fixed_trig_device_fns import FIXED_TRIG_DEVICE_FNS
    from vibeproj.fused_kernels import _TM_FORWARD_SOURCE, _inject_linear_unit_args

    if strategy not in ("paired", "reframed", "q62_reframed"):
        raise ValueError(f"unknown fp64 experiment strategy: {strategy!r}")

    source = _inject_linear_unit_args(
        _TM_FORWARD_SOURCE.format(
            real_t="double",
            pi="3.14159265358979323846",
            tol="1e-14",
        )
    )
    kernel_name = f"tm_forward_fp64_{strategy}"
    source = source.replace("tm_forward(", f"{kernel_name}(", 1)

    old = """double Cn = gatg(cbg0, cbg1, cbg2, cbg3, cbg4, cbg5,
                       phi, cos((double)2.0 * phi), sin((double)2.0 * phi));
    double sin_Cn = sin(Cn), cos_Cn = cos(Cn);
    double sin_Ce = sin(lam), cos_Ce = cos(lam);
    double cos_Cn_cos_Ce = cos_Cn * cos_Ce;
    Cn = atan2(sin_Cn, cos_Cn_cos_Ce);
    double inv_denom = rsqrt(sin_Cn * sin_Cn + cos_Cn_cos_Ce * cos_Cn_cos_Ce);
    double tan_Ce = sin_Ce * cos_Cn * inv_denom;
    double Ce = asinh(tan_Ce);"""

    if strategy == "q62_reframed":
        trig_two_phi = (
            "vp_tm_experiment_sincos((double)2.0 * phi, fabs(lam) <= 0.06, "
            "&sin_two_phi, &cos_two_phi);"
        )
        trig_cn = "vp_tm_experiment_sincos(Cn, fabs(lam) <= 0.06, &sin_Cn, &cos_Cn);"
        trig_lam = "vp_tm_experiment_sincos(lam, fabs(lam) <= 0.06, &sin_Ce, &cos_Ce);"
    else:
        trig_two_phi = "sincos((double)2.0 * phi, &sin_two_phi, &cos_two_phi);"
        trig_cn = "sincos(Cn, &sin_Cn, &cos_Cn);"
        trig_lam = "sincos(lam, &sin_Ce, &cos_Ce);"
    new = f"""double sin_two_phi, cos_two_phi;
    {trig_two_phi}
    double Cn = gatg(cbg0, cbg1, cbg2, cbg3, cbg4, cbg5,
                     phi, cos_two_phi, sin_two_phi);
    const double gaussian_lat = Cn;
    double sin_Cn, cos_Cn;
    {trig_cn}
    double sin_Ce, cos_Ce;
    {trig_lam}
    double cos_Cn_cos_Ce = cos_Cn * cos_Ce;
    double inv_denom = rsqrt(sin_Cn * sin_Cn + cos_Cn_cos_Ce * cos_Cn_cos_Ce);
    double tan_Ce = sin_Ce * cos_Cn * inv_denom;"""

    if strategy == "paired":
        new += """
    Cn = atan2(sin_Cn, cos_Cn_cos_Ce);
    double Ce = asinh(tan_Ce);"""
    else:
        new += """
    if (fabs(lam) <= 0.06) {
        // Reframe the general atan2 as the original Gaussian latitude plus a
        // tiny correction. For a UTM zone, |correction| < 6.9e-4 radians.
        const double denominator = fma(
            cos_Ce * cos_Cn, cos_Cn, sin_Cn * sin_Cn
        );
        double reciprocal = (double)__frcp_rn((float)denominator);
        reciprocal *= fma(-denominator, reciprocal, 2.0);
        const double correction =
            sin_Cn * cos_Cn * (1.0 - cos_Ce) * reciprocal;
        const double correction_sq = correction * correction;
        Cn = gaussian_lat + correction *
            (1.0 + correction_sq * (-1.0 / 3.0 + correction_sq / 5.0));
    } else {
        Cn = atan2(sin_Cn, cos_Cn_cos_Ce);
    }

    double Ce;
    if (fabs(tan_Ce) <= 0.06) {
        // asinh(x) odd series through x^11. The next term is below one ulp
        // throughout the normal UTM domain (actual |x| <= tan(3 degrees)).
        const double x2 = tan_Ce * tan_Ce;
        Ce = tan_Ce * (1.0 + x2 * (
            -1.0 / 6.0 + x2 * (
             3.0 / 40.0 + x2 * (
            -5.0 / 112.0 + x2 * (
            35.0 / 1152.0 + x2 * (-63.0 / 2816.0))))));
    } else {
        Ce = asinh(tan_Ce);
    }"""

    if old not in source:
        raise RuntimeError("fp64 TM transcendental block changed")
    source = source.replace(old, new, 1)
    prefix = ""
    if strategy == "q62_reframed":
        prefix = (
            FIXED_TRIG_DEVICE_FNS
            + r"""
__device__ __forceinline__ void vp_tm_experiment_sincos(
    double angle, bool use_fixed, double* sin_out, double* cos_out
) {
    if (use_fixed) {
        vp_fixed_sincos_bounded(angle, sin_out, cos_out);
    } else {
        sincos(angle, sin_out, cos_out);
    }
}
"""
        )
    return prefix + source, kernel_name


def _make_double_param_launch(
    cp: Any,
    kernel: Any,
    computed: dict[str, Any],
    lat: Any,
    lon: Any,
    out_x: Any,
    out_y: Any,
) -> Callable[[], None]:
    """Construct a production-compatible launch for a double-parameter TM kernel."""
    n = lat.size
    block_size = 256
    grid_size = max(1, (n + block_size - 1) // block_size)
    args = (
        lat,
        lon,
        out_x,
        out_y,
        *(np.float64(value) for value in computed["cbg"]),
        *(np.float64(value) for value in computed["gtu"]),
        np.float64(computed["Qn"]),
        np.float64(computed["Zb"]),
        np.float64(computed["lam0"]),
        np.float64(computed["a"]),
        np.float64(computed["x0"]),
        np.float64(computed["y0"]),
        np.float64(computed.get("x_unit_to_m", 1.0)),
        np.float64(computed.get("y_unit_to_m", 1.0)),
        np.int32(True),
        np.int32(False),
        np.int32(n),
    )

    def launch() -> None:
        kernel((grid_size,), (block_size,), args)

    return launch


def _time_interleaved(
    cp: Any,
    launches: dict[str, Callable[[], None]],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, dict[str, float]]:
    # Compile and warm every path before collecting measurements.
    for launch in launches.values():
        for _ in range(warmup):
            launch()
    cp.cuda.get_current_stream().synchronize()

    samples = {name: [] for name in launches}
    names = list(launches)
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    for iteration in range(iterations):
        # Rotate the order to reduce systematic boost/thermal bias.
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            start.record()
            launches[name]()
            end.record()
            end.synchronize()
            samples[name].append(float(cp.cuda.get_elapsed_time(start, end)))
    return {name: _timing_stats(values) for name, values in samples.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - benchmark environment guard
        raise SystemExit("CuPy is required for this GPU benchmark") from exc

    from pyproj import Transformer as PyProjTransformer

    from vibeproj import Transformer

    device = cp.cuda.Device(args.device)
    device.use()
    properties = cp.cuda.runtime.getDeviceProperties(args.device)
    gpu_name = properties["name"].decode()

    rng = np.random.default_rng(args.seed)
    if args.wide_domain:
        # Exercise the native fallback well outside the fast UTM-zone domain
        # without approaching TM's +/-90-degree longitude singularity.
        lat_range = (-80.0, 80.0)
        lon_range = (-57.0, 63.0)
    else:
        # Stay inside UTM zone 31N's normal area of use. Q11.52's fixed range
        # is safe here and accuracy comparisons represent real UTM work.
        lat_range = (0.0, 84.0)
        lon_range = (0.0, 6.0)
    lat_host = rng.uniform(*lat_range, args.n).astype(np.float64)
    lon_host = rng.uniform(*lon_range, args.n).astype(np.float64)
    lat = cp.asarray(lat_host)
    lon = cp.asarray(lon_host)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=False)
    computed = transformer._pipeline.computed

    outputs = {
        name: (cp.empty(args.n, dtype=cp.float64), cp.empty(args.n, dtype=cp.float64))
        for name in (
            "fp64_legacy",
            "fp64",
            "fp64_reframed",
            "production_fast",
            "fp32",
            "double_single",
            "ds_paired",
            "ds_q62_trig",
            "int64_q52",
        )
    }

    launches: dict[str, Callable[[], None]] = {}
    for name, precision in (("fp32", "fp32"), ("double_single", "ds")):
        out_x, out_y = outputs[name]

        def launch(
            precision: str = precision,
            out_x: Any = out_x,
            out_y: Any = out_y,
        ) -> None:
            transformer.transform_buffers(
                lat,
                lon,
                out_x=out_x,
                out_y=out_y,
                precision=precision,
            )

        launches[name] = launch

    from vibeproj.fused_kernels import (
        _TM_FORWARD_SOURCE,
        _get_kernel,
        _inject_linear_unit_args,
    )

    legacy_source = _inject_linear_unit_args(
        _TM_FORWARD_SOURCE.format(
            real_t="double",
            pi="3.14159265358979323846",
            tol="1e-14",
        )
    )
    legacy_kernel = cp.RawKernel(legacy_source, "tm_forward", options=("--std=c++11",))
    launches["fp64_legacy"] = _make_double_param_launch(
        cp, legacy_kernel, computed, lat, lon, *outputs["fp64_legacy"]
    )

    for name, mode in (("fp64", "fp64"), ("production_fast", "int64")):
        kernel = _get_kernel("tmerc", "forward", "float64", tmerc_mode=mode)
        launches[name] = _make_double_param_launch(cp, kernel, computed, lat, lon, *outputs[name])

    for name, strategy in (("ds_paired", "native"), ("ds_q62_trig", "q62")):
        source, function_name = _make_ds_trig_source(strategy)
        kernel = cp.RawKernel(source, function_name, options=("--std=c++11",))
        launches[name] = _make_double_param_launch(cp, kernel, computed, lat, lon, *outputs[name])

    for name, strategy in (("fp64_reframed", "reframed"),):
        source, function_name = _make_fp64_experiment_source(strategy)
        kernel = cp.RawKernel(source, function_name, options=("--std=c++11",))
        launches[name] = _make_double_param_launch(cp, kernel, computed, lat, lon, *outputs[name])

    q52_kernel = cp.RawKernel(Q52_TM_FORWARD_SOURCE, "tm_forward_q52", options=("--std=c++11",))
    launches["int64_q52"] = _make_q52_launch(
        cp, q52_kernel, computed, lat, lon, *outputs["int64_q52"]
    )

    timings = _time_interleaved(
        cp,
        launches,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    # Refresh outputs once after timing, then copy for deterministic accuracy metrics.
    for launch in launches.values():
        launch()
    device.synchronize()
    host_outputs = {
        name: (cp.asnumpy(out_x), cp.asnumpy(out_y)) for name, (out_x, out_y) in outputs.items()
    }

    fp64_x, fp64_y = host_outputs["fp64_legacy"]
    vs_fp64 = {
        name: _error_stats(x, y, fp64_x, fp64_y)
        for name, (x, y) in host_outputs.items()
        if name != "fp64_legacy"
    }

    oracle_n = min(args.oracle_n, args.n)
    oracle = PyProjTransformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    oracle_x, oracle_y = oracle.transform(lon_host[:oracle_n], lat_host[:oracle_n])
    vs_pyproj = {
        name: _error_stats(x[:oracle_n], y[:oracle_n], oracle_x, oracle_y)
        for name, (x, y) in host_outputs.items()
    }

    fp64_median = timings["fp64_legacy"]["median_ms"]
    for stats in timings.values():
        stats["mcoords_per_s"] = args.n / stats["median_ms"] / 1000.0
        stats["speedup_vs_legacy"] = fp64_median / stats["median_ms"]

    return {
        "meta": {
            "gpu": gpu_name,
            "compute_capability": f"{properties['major']}.{properties['minor']}",
            "python": platform.python_version(),
            "cupy": cp.__version__,
            "n_coords": args.n,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "oracle_coords": oracle_n,
            "utm": "EPSG:4326 -> EPSG:32631",
            "input_domain": {
                "latitude_degrees": list(lat_range),
                "longitude_degrees": list(lon_range),
            },
            "int64_format": "signed Q11.52",
            "hybrid_trig_format": "table-free signed Q1.62",
        },
        "timing": timings,
        "error_vs_fp64_m": vs_fp64,
        "error_vs_pyproj_m": vs_pyproj,
    }


def _print_human(results: dict[str, Any]) -> None:
    meta = results["meta"]
    print(f"GPU: {meta['gpu']} (sm_{meta['compute_capability'].replace('.', '')})")
    print(f"Transform: {meta['utm']}; {meta['n_coords']:,} points")
    print()
    print(f"{'mode':<16} {'median ms':>10} {'Mcoord/s':>10} {'vs legacy':>10} {'p05..p95 ms':>20}")
    for name, stats in results["timing"].items():
        spread = f"{stats['p05_ms']:.4f}..{stats['p95_ms']:.4f}"
        print(
            f"{name:<16} {stats['median_ms']:>10.4f} {stats['mcoords_per_s']:>10.1f} "
            f"{stats['speedup_vs_legacy']:>9.2f}x {spread:>20}"
        )

    print()
    print("Error versus pre-branch vibeproj fp64 (metres)")
    print(f"{'mode':<16} {'max radial':>14} {'p99 radial':>14} {'RMS radial':>14}")
    for name, stats in results["error_vs_fp64_m"].items():
        print(
            f"{name:<16} {stats['max_radial']:>14.6g} "
            f"{stats['p99_radial']:>14.6g} {stats['rms_radial']:>14.6g}"
        )

    print()
    print(f"Error versus pyproj ({meta['oracle_coords']:,}-point subset, metres)")
    print(f"{'mode':<16} {'max radial':>14} {'p99 radial':>14} {'RMS radial':>14}")
    for name, stats in results["error_vs_pyproj_m"].items():
        print(
            f"{name:<16} {stats['max_radial']:>14.6g} "
            f"{stats['p99_radial']:>14.6g} {stats['rms_radial']:>14.6g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1_000_000, help="number of coordinates")
    parser.add_argument("--iterations", type=int, default=50, help="timed iterations per mode")
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations per mode")
    parser.add_argument(
        "--oracle-n", type=int, default=100_000, help="points checked against pyproj"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--wide-domain",
        action="store_true",
        help="exercise guarded fallbacks outside the normal UTM zone",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    if args.n <= 0 or args.iterations <= 0 or args.warmup < 0 or args.oracle_n <= 0:
        parser.error("n, iterations, and oracle-n must be positive; warmup must be non-negative")

    results = run(args)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_human(results)


if __name__ == "__main__":
    main()
