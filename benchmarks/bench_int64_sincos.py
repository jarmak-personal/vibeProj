#!/usr/bin/env python3
"""Benchmark bounded-domain INT64 sin/cos strategies on an NVIDIA GPU.

The inputs are fp64 radians in [-pi, pi], matching longitude and reduced-angle
projection work.  Outputs remain fp64.  Two fixed-point implementations are
compared with CUDA's native fp64 paths:

* Q3.60 CORDIC: 60 shift/add iterations and no lookup table.
* Q1.62 table+polynomial: a 256-entry lookup table with a degree-6
  small-angle reconstruction polynomial.

The experiment is deliberately bounded-domain.  A general libm replacement
would also need high-precision range reduction for arbitrarily large inputs.

Usage:
    uv run python benchmarks/bench_int64_sincos.py
    uv run python benchmarks/bench_int64_sincos.py --n 5000000 --iterations 100
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


TABLE_SIZE = 256
ANGLE_FRAC_BITS = 60
VALUE_FRAC_BITS = 62
CORDIC_ITERATIONS = 60


def _fixed(value: float, fractional_bits: int) -> int:
    scaled = round(value * (1 << fractional_bits))
    if not -(1 << 63) <= scaled < (1 << 63):
        raise OverflowError(f"{value!r} is outside signed Q{fractional_bits} range")
    return scaled


def _cuda_i64(values: list[int]) -> str:
    return ",\n    ".join(f"{value}LL" for value in values)


def _make_source(table_size: int = TABLE_SIZE) -> str:
    if table_size <= 0 or table_size & (table_size - 1):
        raise ValueError("table_size must be a positive power of two")
    step = 2.0 * math.pi / table_size
    sin_table = [_fixed(math.sin(index * step), VALUE_FRAC_BITS) for index in range(table_size)]
    cos_table = [_fixed(math.cos(index * step), VALUE_FRAC_BITS) for index in range(table_size)]
    cordic_angles = [
        _fixed(math.atan(2.0**-index), ANGLE_FRAC_BITS) for index in range(CORDIC_ITERATIONS)
    ]
    cordic_gain = math.prod(math.sqrt(1.0 + 2.0 ** (-2 * index)) for index in range(80))
    sin_coefficients = [
        _fixed((-1.0) ** index / math.factorial(2 * index + 1), VALUE_FRAC_BITS)
        for index in range(9)
    ]
    cos_coefficients = [
        _fixed((-1.0) ** index / math.factorial(2 * index), VALUE_FRAC_BITS) for index in range(10)
    ]

    def horner_lines(name: str, coefficients: list[int]) -> str:
        lines = [f"    q62_t {name} = (q62_t){coefficients[-1]}LL;"]
        lines.extend(
            f"    {name} = (q62_t){coefficient}LL + q62_mul(r2, {name});"
            for coefficient in reversed(coefficients[:-1])
        )
        return "\n".join(lines)

    source = (
        r"""
typedef long long q60_t;
typedef long long q62_t;
typedef unsigned long long uq64_t;

#define Q60_SCALE_D 1152921504606846976.0
#define Q62_SCALE_D 4611686018427387904.0
#define Q62_INV_SCALE_D 2.1684043449710088680149056017398834228515625e-19
#define TABLE_SIZE 1024
#define TABLE_MASK 1023
#define TABLE_INV_STEP 162.9746617261008337443710254418462
#define TABLE_STEP 0.006135923151542564918872350357967779
#define HALF_PI_Q60 ((q60_t)1811004864519280640LL)
#define PI_Q60 ((q60_t)3622009729038561280LL)

__device__ __forceinline__ q62_t q62_from_double(double value) {
    return __double2ll_rn(value * Q62_SCALE_D);
}

__device__ __forceinline__ double q62_to_double(q62_t value) {
    return (double)value * Q62_INV_SCALE_D;
}

__device__ __forceinline__ q62_t q62_mul(q62_t a, q62_t b) {
    const bool negative = (a < 0) != (b < 0);
    const uq64_t ua = a < 0 ? 0ULL - (uq64_t)a : (uq64_t)a;
    const uq64_t ub = b < 0 ? 0ULL - (uq64_t)b : (uq64_t)b;
    const uq64_t lo = ua * ub;
    uq64_t hi = __umul64hi(ua, ub);
    const uq64_t rounded_lo = lo + (1ULL << 61);
    hi += rounded_lo < lo;
    const uq64_t magnitude = (hi << 2) | (rounded_lo >> 62);
    const q62_t result = (q62_t)magnitude;
    return negative ? -result : result;
}

__constant__ q62_t SIN_TABLE[TABLE_SIZE] = {
    """
        + _cuda_i64(sin_table)
        + r"""
};

__constant__ q62_t COS_TABLE[TABLE_SIZE] = {
    """
        + _cuda_i64(cos_table)
        + r"""
};

__constant__ q60_t CORDIC_ANGLES[60] = {
    """
        + _cuda_i64(cordic_angles)
        + r"""
};

__device__ __forceinline__ void fixed_table_sincos(double angle, q62_t* sin_out, q62_t* cos_out) {
    const int center = __double2int_rn(angle * TABLE_INV_STEP);
    const int table_index = center & TABLE_MASK;
    const q62_t residual = q62_from_double(angle - (double)center * TABLE_STEP);
    const q62_t r2 = q62_mul(residual, residual);

    // On |r| <= pi/1024, these truncation errors are far below one Q1.62 unit.
    const q62_t sin_coefficient_3 = (q62_t)-768614336404564608LL;  // -1/6
    const q62_t sin_coefficient_5 = (q62_t)38430716820228232LL;    // 1/120
    const q62_t cos_coefficient_2 = (q62_t)-2305843009213693952LL; // -1/2
    const q62_t cos_coefficient_4 = (q62_t)192153584101141152LL;   // 1/24
    const q62_t cos_coefficient_6 = (q62_t)-6405119470038039LL;    // -1/720
    const q62_t one = (q62_t)4611686018427387904LL;

    const q62_t sin_poly = residual + q62_mul(
        q62_mul(residual, r2),
        sin_coefficient_3 + q62_mul(r2, sin_coefficient_5)
    );
    const q62_t cos_poly = one + q62_mul(
        r2,
        cos_coefficient_2 + q62_mul(r2, cos_coefficient_4 + q62_mul(r2, cos_coefficient_6))
    );

    const q62_t table_sin = SIN_TABLE[table_index];
    const q62_t table_cos = COS_TABLE[table_index];
    *sin_out = q62_mul(table_sin, cos_poly) + q62_mul(table_cos, sin_poly);
    *cos_out = q62_mul(table_cos, cos_poly) - q62_mul(table_sin, sin_poly);
}

__device__ __forceinline__ void fixed_cordic_sincos(double angle, q62_t* sin_out, q62_t* cos_out) {
    q60_t z = __double2ll_rn(angle * Q60_SCALE_D);
    int sign = 1;
    if (z > HALF_PI_Q60) {
        z -= PI_Q60;
        sign = -1;
    } else if (z < -HALF_PI_Q60) {
        z += PI_Q60;
        sign = -1;
    }

    q62_t x = (q62_t)"""
        + str(_fixed(1.0 / cordic_gain, VALUE_FRAC_BITS))
        + r"""LL;
    q62_t y = 0;
#pragma unroll 1
    for (int iteration = 0; iteration < 60; iteration++) {
        const q62_t old_x = x;
        const q62_t old_y = y;
        if (z >= 0) {
            x = old_x - (old_y >> iteration);
            y = old_y + (old_x >> iteration);
            z -= CORDIC_ANGLES[iteration];
        } else {
            x = old_x + (old_y >> iteration);
            y = old_y - (old_x >> iteration);
            z += CORDIC_ANGLES[iteration];
        }
    }
    *sin_out = sign > 0 ? y : -y;
    *cos_out = sign > 0 ? x : -x;
}

__device__ __forceinline__ void fixed_reduced_poly_sincos(
    double angle, q62_t* sin_out, q62_t* cos_out
) {
    // Nearest-quadrant reduction gives a residual in [-pi/4, pi/4].
    const int quadrant = __double2int_rn(angle * 0.6366197723675813430755350534900574);
    const q62_t residual = q62_from_double(
        angle - (double)quadrant * 1.5707963267948966192313216916397514
    );
    const q62_t r2 = q62_mul(residual, residual);
"""
        + horner_lines("sin_horner", sin_coefficients)
        + "\n"
        + horner_lines("cos_horner", cos_coefficients)
        + r"""
    const q62_t sin_residual = q62_mul(residual, sin_horner);
    const q62_t cos_residual = cos_horner;

    switch (quadrant & 3) {
        case 0: *sin_out = sin_residual;  *cos_out = cos_residual;  break;
        case 1: *sin_out = cos_residual;  *cos_out = -sin_residual; break;
        case 2: *sin_out = -sin_residual; *cos_out = -cos_residual; break;
        default:*sin_out = -cos_residual; *cos_out = sin_residual;  break;
    }
}

extern "C" __global__ void __launch_bounds__(256) sincos_fp64_separate(
    const double* __restrict__ angles,
    double* __restrict__ out_sin,
    double* __restrict__ out_cos,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const double angle = angles[idx];
    out_sin[idx] = sin(angle);
    out_cos[idx] = cos(angle);
}

extern "C" __global__ void __launch_bounds__(256) sincos_fp64_paired(
    const double* __restrict__ angles,
    double* __restrict__ out_sin,
    double* __restrict__ out_cos,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    sincos(angles[idx], &out_sin[idx], &out_cos[idx]);
}

extern "C" __global__ void __launch_bounds__(256) sincospi_fp64_paired(
    const double* __restrict__ angles,
    double* __restrict__ out_sin,
    double* __restrict__ out_cos,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    sincospi(
        angles[idx] * 0.3183098861837906715377675267450287,
        &out_sin[idx],
        &out_cos[idx]
    );
}

extern "C" __global__ void __launch_bounds__(256) sincos_fixed_table(
    const double* __restrict__ angles,
    double* __restrict__ out_sin,
    double* __restrict__ out_cos,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    q62_t sin_value;
    q62_t cos_value;
    fixed_table_sincos(angles[idx], &sin_value, &cos_value);
    out_sin[idx] = q62_to_double(sin_value);
    out_cos[idx] = q62_to_double(cos_value);
}

extern "C" __global__ void __launch_bounds__(256) sincos_fixed_cordic(
    const double* __restrict__ angles,
    double* __restrict__ out_sin,
    double* __restrict__ out_cos,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    q62_t sin_value;
    q62_t cos_value;
    fixed_cordic_sincos(angles[idx], &sin_value, &cos_value);
    out_sin[idx] = q62_to_double(sin_value);
    out_cos[idx] = q62_to_double(cos_value);
}

extern "C" __global__ void __launch_bounds__(256) sincos_fixed_reduced_poly(
    const double* __restrict__ angles,
    double* __restrict__ out_sin,
    double* __restrict__ out_cos,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    q62_t sin_value;
    q62_t cos_value;
    fixed_reduced_poly_sincos(angles[idx], &sin_value, &cos_value);
    out_sin[idx] = q62_to_double(sin_value);
    out_cos[idx] = q62_to_double(cos_value);
}
"""
    )
    return (
        source.replace("#define TABLE_SIZE 1024", f"#define TABLE_SIZE {table_size}")
        .replace("#define TABLE_MASK 1023", f"#define TABLE_MASK {table_size - 1}")
        .replace(
            "#define TABLE_INV_STEP 162.9746617261008337443710254418462",
            f"#define TABLE_INV_STEP {table_size / (2.0 * math.pi):.35g}",
        )
        .replace(
            "#define TABLE_STEP 0.006135923151542564918872350357967779",
            f"#define TABLE_STEP {step:.35g}",
        )
    )


SINCOS_SOURCE = _make_source()
GLOBAL_TABLE_SOURCE = SINCOS_SOURCE.replace(
    "__constant__ q62_t SIN_TABLE", "__device__ q62_t SIN_TABLE"
).replace("__constant__ q62_t COS_TABLE", "__device__ q62_t COS_TABLE")
UNROLLED_CORDIC_SOURCE = SINCOS_SOURCE.replace("#pragma unroll 1", "#pragma unroll")


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


def _error_stats(cp: Any, values: tuple[Any, Any], reference: tuple[Any, Any]) -> dict[str, float]:
    sin_error = cp.abs(values[0] - reference[0])
    cos_error = cp.abs(values[1] - reference[1])
    combined = cp.maximum(sin_error, cos_error)
    return {
        "max_abs": float(cp.max(combined).item()),
        "p99_abs": float(cp.percentile(combined, 99).item()),
        "rms_pair": float(cp.sqrt(cp.mean(sin_error * sin_error + cos_error * cos_error)).item()),
        "max_earth_radius_m": float(cp.max(combined).item() * 6_378_137.0),
    }


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
    angle_host = rng.uniform(-math.pi, math.pi, args.n).astype(np.float64)
    # Include exact quadrant boundaries and near-boundary values in every run.
    edge_values = np.array(
        [
            -math.pi,
            -math.pi / 2,
            0.0,
            math.pi / 2,
            math.pi,
            np.nextafter(-math.pi, 0.0),
            np.nextafter(math.pi, 0.0),
        ],
        dtype=np.float64,
    )
    angle_host[: min(args.n, edge_values.size)] = edge_values[: args.n]
    angles = cp.asarray(angle_host)

    kernel_specs = {
        "fp64_separate": (SINCOS_SOURCE, "sincos_fp64_separate"),
        "fp64_paired": (SINCOS_SOURCE, "sincos_fp64_paired"),
        "fp64_sincospi": (SINCOS_SOURCE, "sincospi_fp64_paired"),
        "int64_table_constant": (SINCOS_SOURCE, "sincos_fixed_table"),
        "int64_table_global": (GLOBAL_TABLE_SOURCE, "sincos_fixed_table"),
        "int64_reduced_poly": (SINCOS_SOURCE, "sincos_fixed_reduced_poly"),
        "int64_cordic": (SINCOS_SOURCE, "sincos_fixed_cordic"),
        "int64_cordic_unrolled": (UNROLLED_CORDIC_SOURCE, "sincos_fixed_cordic"),
    }
    names = tuple(kernel_specs)
    kernels = {
        name: cp.RawKernel(source, function_name, options=("--std=c++11",))
        for name, (source, function_name) in kernel_specs.items()
    }
    outputs = {
        name: (cp.empty(args.n, dtype=cp.float64), cp.empty(args.n, dtype=cp.float64))
        for name in names
    }
    block = 256
    grid = max(1, (args.n + block - 1) // block)
    launches: dict[str, Callable[[], None]] = {}
    for name in names:
        kernel = kernels[name]
        out_sin, out_cos = outputs[name]
        kernel_args = (angles, out_sin, out_cos, np.int32(args.n))

        def launch(kernel: Any = kernel, kernel_args: tuple[Any, ...] = kernel_args) -> None:
            kernel((grid,), (block,), kernel_args)

        launches[name] = launch

    timing = _time_interleaved(cp, launches, warmup=args.warmup, iterations=args.iterations)
    for launch in launches.values():
        launch()
    device.synchronize()

    reference = outputs["fp64_paired"]
    errors = {
        name: _error_stats(cp, values, reference)
        for name, values in outputs.items()
        if name != "fp64_paired"
    }

    fp64_median = timing["fp64_paired"]["median_ms"]
    for stats in timing.values():
        stats["mangles_per_s"] = args.n / stats["median_ms"] / 1000.0
        stats["speedup_vs_fp64_paired"] = fp64_median / stats["median_ms"]

    return {
        "meta": {
            "gpu": gpu_name,
            "compute_capability": f"{properties['major']}.{properties['minor']}",
            "python": platform.python_version(),
            "cupy": cp.__version__,
            "n_angles": args.n,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "domain_radians": [-math.pi, math.pi],
            "table_entries": TABLE_SIZE,
            "cordic_iterations": CORDIC_ITERATIONS,
        },
        "timing": timing,
        "error_vs_fp64_paired": errors,
    }


def _print_human(results: dict[str, Any]) -> None:
    meta = results["meta"]
    print(f"GPU: {meta['gpu']} (sm_{meta['compute_capability'].replace('.', '')})")
    print(f"Domain: [-pi, pi]; {meta['n_angles']:,} angles; simultaneous sin+cos outputs")
    print()
    print(f"{'mode':<18} {'median ms':>10} {'Mangle/s':>10} {'vs paired':>10} {'p05..p95 ms':>20}")
    for name, stats in results["timing"].items():
        spread = f"{stats['p05_ms']:.4f}..{stats['p95_ms']:.4f}"
        print(
            f"{name:<18} {stats['median_ms']:>10.4f} {stats['mangles_per_s']:>10.1f} "
            f"{stats['speedup_vs_fp64_paired']:>9.2f}x {spread:>20}"
        )

    print()
    print("Error versus CUDA fp64 sincos")
    print(f"{'mode':<18} {'max abs':>14} {'p99 abs':>14} {'Earth-radius max':>18}")
    for name, stats in results["error_vs_fp64_paired"].items():
        print(
            f"{name:<18} {stats['max_abs']:>14.6g} {stats['p99_abs']:>14.6g} "
            f"{stats['max_earth_radius_m']:>17.6g} m"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1_000_000, help="number of input angles")
    parser.add_argument("--iterations", type=int, default=50, help="timed iterations per mode")
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
