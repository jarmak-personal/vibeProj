#!/usr/bin/env python3
"""Benchmark an INT64 fixed-point Helmert core on an NVIDIA GPU.

The production kernel performs:

    geodetic -> ECEF -> Helmert matrix -> geodetic

This benchmark compares the production paired-fp64 and table-free Q1.62 trig
variants, then measures an experimental fixed-point matrix core in isolation.
Square-root, atan2, ECEF math, and the Helmert matrix remain fp64 in the
production INT64-trig path.

Two fixed-point formats preserve both ECEF range and Helmert parameter detail:

* ECEF coordinates/translations: signed Q24.39, range +/-16,777,216 metres.
* rotations/scale: signed Q7.56, range +/-128.

This is an isolated experiment, not a public vibeproj precision mode.

Usage:
    uv run python benchmarks/bench_helmert_int64.py
    uv run python benchmarks/bench_helmert_int64.py --n 5000000 --iterations 100
    uv run python benchmarks/bench_helmert_int64.py --with-height --json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np


COORD_FRAC_BITS = 39
PARAM_FRAC_BITS = 56


FIXED_DEVICE_FUNCTIONS = r"""
typedef long long q39_t;
typedef long long q56_t;
typedef unsigned long long uq64_t;

#define Q39_SCALE_D 549755813888.0
#define Q39_INV_SCALE_D 1.818989403545856475830078125e-12

__device__ __forceinline__ q39_t q39_from_double(double value) {
    return __double2ll_rn(value * Q39_SCALE_D);
}

__device__ __forceinline__ double q39_to_double(q39_t value) {
    return (double)value * Q39_INV_SCALE_D;
}

// Multiply a Q24.39 coordinate by a Q7.56 parameter, returning Q24.39.
// The full 128-bit product is assembled from CUDA's low/high 64-bit products.
__device__ __forceinline__ q39_t q39_mul_q56(q39_t coordinate, q56_t parameter) {
    const bool negative = (coordinate < 0) != (parameter < 0);
    const uq64_t ua = coordinate < 0 ? 0ULL - (uq64_t)coordinate : (uq64_t)coordinate;
    const uq64_t ub = parameter < 0 ? 0ULL - (uq64_t)parameter : (uq64_t)parameter;

    const uq64_t lo = ua * ub;
    uq64_t hi = __umul64hi(ua, ub);
    const uq64_t rounded_lo = lo + (1ULL << 55);
    hi += rounded_lo < lo;
    const uq64_t magnitude = (hi << 8) | (rounded_lo >> 56);
    const q39_t result = (q39_t)magnitude;
    return negative ? -result : result;
}

__device__ __forceinline__ void helmert_core_fixed(
    q39_t X, q39_t Y, q39_t Z,
    q39_t tx, q39_t ty, q39_t tz,
    q56_t rx, q56_t ry, q56_t rz, q56_t ds,
    q39_t* X2, q39_t* Y2, q39_t* Z2
) {
    const q39_t rotated_x = X - q39_mul_q56(Y, rz) + q39_mul_q56(Z, ry);
    const q39_t rotated_y = q39_mul_q56(X, rz) + Y - q39_mul_q56(Z, rx);
    const q39_t rotated_z = -q39_mul_q56(X, ry) + q39_mul_q56(Y, rx) + Z;
    *X2 = q39_mul_q56(rotated_x, ds) + tx;
    *Y2 = q39_mul_q56(rotated_y, ds) + ty;
    *Z2 = q39_mul_q56(rotated_z, ds) + tz;
}
"""


HELMERT_FIXED_SOURCE = (
    FIXED_DEVICE_FUNCTIONS
    + r"""
extern "C" __global__ void __launch_bounds__(256) helmert_shift_fixed(
    const double* __restrict__ in_lat,
    const double* __restrict__ in_lon,
    double* __restrict__ out_lat,
    double* __restrict__ out_lon,
    const double* __restrict__ in_h,
    double* __restrict__ out_h,
    double src_a, double src_es,
    double dst_a, double dst_es,
    q39_t tx, q39_t ty, q39_t tz,
    q56_t rx, q56_t ry, q56_t rz,
    q56_t ds,
    int n,
    int has_z
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const double lat = in_lat[idx] * 0.017453292519943295;
    const double lon = in_lon[idx] * 0.017453292519943295;

    const double sin_lat = sin(lat);
    const double cos_lat = cos(lat);
    const double sin_lon = sin(lon);
    const double cos_lon = cos(lon);
    const double N = src_a / sqrt(1.0 - src_es * sin_lat * sin_lat);

    double X;
    double Y;
    double Z;
    if (has_z) {
        const double h_val = in_h[idx];
        X = (N + h_val) * cos_lat * cos_lon;
        Y = (N + h_val) * cos_lat * sin_lon;
        Z = (N * (1.0 - src_es) + h_val) * sin_lat;
    } else {
        X = N * cos_lat * cos_lon;
        Y = N * cos_lat * sin_lon;
        Z = N * (1.0 - src_es) * sin_lat;
    }

    q39_t X2_fixed;
    q39_t Y2_fixed;
    q39_t Z2_fixed;
    helmert_core_fixed(
        q39_from_double(X), q39_from_double(Y), q39_from_double(Z),
        tx, ty, tz, rx, ry, rz, ds,
        &X2_fixed, &Y2_fixed, &Z2_fixed
    );
    const double X2 = q39_to_double(X2_fixed);
    const double Y2 = q39_to_double(Y2_fixed);
    const double Z2 = q39_to_double(Z2_fixed);

    const double p = sqrt(X2 * X2 + Y2 * Y2);
    const double lon_out = atan2(Y2, X2);
    double lat_out = atan2(Z2, p * (1.0 - dst_es));

    for (int i = 0; i < 10; i++) {
        const double sin_lat_i = sin(lat_out);
        const double N_i = dst_a / sqrt(1.0 - dst_es * sin_lat_i * sin_lat_i);
        const double lat_new = atan2(Z2 + dst_es * N_i * sin_lat_i, p);
        if (fabs(lat_new - lat_out) < 1e-14) {
            lat_out = lat_new;
            break;
        }
        lat_out = lat_new;
    }

    out_lat[idx] = lat_out * 57.29577951308232;
    out_lon[idx] = lon_out * 57.29577951308232;

    if (has_z) {
        const double sin_lat_f = sin(lat_out);
        const double cos_lat_f = cos(lat_out);
        const double N_f = dst_a / sqrt(1.0 - dst_es * sin_lat_f * sin_lat_f);
        if (fabs(cos_lat_f) < 1e-10) {
            out_h[idx] = fabs(Z2) / fabs(sin_lat_f) - N_f * (1.0 - dst_es);
        } else {
            out_h[idx] = p / cos_lat_f - N_f;
        }
    }
}
"""
)


CORE_SOURCE = (
    FIXED_DEVICE_FUNCTIONS
    + r"""
extern "C" __global__ void __launch_bounds__(256) helmert_core_fp64(
    const double* __restrict__ in_x,
    const double* __restrict__ in_y,
    const double* __restrict__ in_z,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    double* __restrict__ out_z,
    double tx, double ty, double tz,
    double rx, double ry, double rz, double ds,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const double X = in_x[idx];
    const double Y = in_y[idx];
    const double Z = in_z[idx];
    out_x[idx] = ds * ( X - rz * Y + ry * Z) + tx;
    out_y[idx] = ds * ( rz * X + Y - rx * Z) + ty;
    out_z[idx] = ds * (-ry * X + rx * Y + Z) + tz;
}

extern "C" __global__ void __launch_bounds__(256) helmert_core_fixed_kernel(
    const double* __restrict__ in_x,
    const double* __restrict__ in_y,
    const double* __restrict__ in_z,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    double* __restrict__ out_z,
    q39_t tx, q39_t ty, q39_t tz,
    q56_t rx, q56_t ry, q56_t rz, q56_t ds,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    q39_t X2;
    q39_t Y2;
    q39_t Z2;
    helmert_core_fixed(
        q39_from_double(in_x[idx]), q39_from_double(in_y[idx]), q39_from_double(in_z[idx]),
        tx, ty, tz, rx, ry, rz, ds, &X2, &Y2, &Z2
    );
    out_x[idx] = q39_to_double(X2);
    out_y[idx] = q39_to_double(Y2);
    out_z[idx] = q39_to_double(Z2);
}
"""
)


def _fixed(value: float, fractional_bits: int) -> np.int64:
    scaled = round(float(value) * (1 << fractional_bits))
    if not -(1 << 63) <= scaled < (1 << 63):
        raise OverflowError(f"{value!r} is outside signed Q{fractional_bits} range")
    return np.int64(scaled)


def _fixed_params(params: Any) -> tuple[np.int64, ...]:
    return (
        _fixed(params.tx, COORD_FRAC_BITS),
        _fixed(params.ty, COORD_FRAC_BITS),
        _fixed(params.tz, COORD_FRAC_BITS),
        _fixed(params.rx, PARAM_FRAC_BITS),
        _fixed(params.ry, PARAM_FRAC_BITS),
        _fixed(params.rz, PARAM_FRAC_BITS),
        _fixed(params.ds, PARAM_FRAC_BITS),
    )


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


def _surface_error_stats(cp: Any, lat: Any, lon: Any, ref_lat: Any, ref_lon: Any, radius: float):
    dlat = (lat - ref_lat) * (np.pi / 180.0) * radius
    dlon = (lon - ref_lon) * (np.pi / 180.0) * radius * cp.cos(ref_lat * (np.pi / 180.0))
    radial = cp.hypot(dlat, dlon)
    return {
        "max_horizontal_m": float(cp.max(radial).item()),
        "p99_horizontal_m": float(cp.percentile(radial, 99).item()),
        "rms_horizontal_m": float(cp.sqrt(cp.mean(radial * radial)).item()),
    }


def _ecef_error_stats(cp: Any, xyz: tuple[Any, Any, Any], reference: tuple[Any, Any, Any]):
    error = cp.sqrt(sum((value - ref) ** 2 for value, ref in zip(xyz, reference, strict=True)))
    return {
        "max_m": float(cp.max(error).item()),
        "p99_m": float(cp.percentile(error, 99).item()),
        "rms_m": float(cp.sqrt(cp.mean(error * error)).item()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - benchmark environment guard
        raise SystemExit("CuPy is required for this GPU benchmark") from exc

    from vibeproj import Transformer
    from vibeproj.fused_kernels import fused_helmert_shift

    device = cp.cuda.Device(args.device)
    device.use()
    properties = cp.cuda.runtime.getDeviceProperties(args.device)
    gpu_name = properties["name"].decode()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:4277")
    params = transformer._helmert
    if params is None:
        raise RuntimeError("Expected WGS84 -> OSGB36 Helmert parameters")

    rng = np.random.default_rng(args.seed)
    if args.global_domain:
        lat_range = (-89.5, 89.5)
        lon_range = (-180.0, 180.0)
    else:
        lat_range = (49.0, 61.0)
        lon_range = (-9.0, 3.0)
    lat_host = rng.uniform(*lat_range, args.n).astype(np.float64)
    lon_host = rng.uniform(*lon_range, args.n).astype(np.float64)
    lat = cp.asarray(lat_host)
    lon = cp.asarray(lon_host)
    height = cp.asarray(rng.uniform(-100.0, 3000.0, args.n)) if args.with_height else None

    full_outputs = {
        name: (
            cp.empty(args.n, dtype=cp.float64),
            cp.empty(args.n, dtype=cp.float64),
            cp.empty(args.n, dtype=cp.float64) if args.with_height else None,
        )
        for name in (
            "fp64",
            "int64_core",
            "int64_trig",
        )
    }

    fixed_core_kernel = cp.RawKernel(
        HELMERT_FIXED_SOURCE, "helmert_shift_fixed", options=("--std=c++11",)
    )
    core_fp64_kernel = cp.RawKernel(CORE_SOURCE, "helmert_core_fp64", options=("--std=c++11",))
    core_fixed_kernel = cp.RawKernel(
        CORE_SOURCE, "helmert_core_fixed_kernel", options=("--std=c++11",)
    )

    block = 256
    grid = max(1, (args.n + block - 1) // block)
    fixed_params = _fixed_params(params)
    dummy_height = lat

    fp64_lat, fp64_lon, fp64_h = full_outputs["fp64"]

    def launch_full_fp64() -> None:
        fused_helmert_shift(
            lat,
            lon,
            params,
            cp,
            h=height,
            out_lat=fp64_lat,
            out_lon=fp64_lon,
            out_h=fp64_h,
            trig_mode="fp64",
        )

    fixed_trig_lat, fixed_trig_lon, fixed_trig_h = full_outputs["int64_trig"]

    def launch_full_fixed_trig() -> None:
        fused_helmert_shift(
            lat,
            lon,
            params,
            cp,
            h=height,
            out_lat=fixed_trig_lat,
            out_lon=fixed_trig_lon,
            out_h=fixed_trig_h,
            trig_mode="int64",
        )

    def full_args(name: str, *, fixed: bool) -> tuple[Any, ...]:
        out_lat, out_lon, out_h = full_outputs[name]
        transform_params = (
            fixed_params
            if fixed
            else (
                np.float64(params.tx),
                np.float64(params.ty),
                np.float64(params.tz),
                np.float64(params.rx),
                np.float64(params.ry),
                np.float64(params.rz),
                np.float64(params.ds),
            )
        )
        return (
            lat,
            lon,
            out_lat,
            out_lon,
            height if args.with_height else dummy_height,
            out_h if args.with_height else out_lat,
            np.float64(params.src_ellipsoid.a),
            np.float64(params.src_ellipsoid.es),
            np.float64(params.dst_ellipsoid.a),
            np.float64(params.dst_ellipsoid.es),
            *transform_params,
            np.int32(args.n),
            np.int32(args.with_height),
        )

    custom_full_specs = {
        "int64_core": (fixed_core_kernel, True),
    }
    custom_full_launches: dict[str, Callable[[], None]] = {}
    for name, (kernel, uses_fixed_core) in custom_full_specs.items():
        kernel_args = full_args(name, fixed=uses_fixed_core)

        def launch(kernel: Any = kernel, kernel_args: tuple[Any, ...] = kernel_args) -> None:
            kernel((grid,), (block,), kernel_args)

        custom_full_launches[name] = launch

    sin_lat = cp.sin(lat * (np.pi / 180.0))
    cos_lat = cp.cos(lat * (np.pi / 180.0))
    sin_lon = cp.sin(lon * (np.pi / 180.0))
    cos_lon = cp.cos(lon * (np.pi / 180.0))
    src = params.src_ellipsoid
    prime_vertical = src.a / cp.sqrt(1.0 - src.es * sin_lat * sin_lat)
    ecef_height = height if height is not None else 0.0
    ecef_x = (prime_vertical + ecef_height) * cos_lat * cos_lon
    ecef_y = (prime_vertical + ecef_height) * cos_lat * sin_lon
    ecef_z = (prime_vertical * (1.0 - src.es) + ecef_height) * sin_lat

    core_outputs = {
        name: tuple(cp.empty(args.n, dtype=cp.float64) for _ in range(3))
        for name in ("fp64", "int64_fixed")
    }
    core_fp64_args = (
        ecef_x,
        ecef_y,
        ecef_z,
        *core_outputs["fp64"],
        np.float64(params.tx),
        np.float64(params.ty),
        np.float64(params.tz),
        np.float64(params.rx),
        np.float64(params.ry),
        np.float64(params.rz),
        np.float64(params.ds),
        np.int32(args.n),
    )
    core_fixed_args = (
        ecef_x,
        ecef_y,
        ecef_z,
        *core_outputs["int64_fixed"],
        *fixed_params,
        np.int32(args.n),
    )

    def launch_core_fp64() -> None:
        core_fp64_kernel((grid,), (block,), core_fp64_args)

    def launch_core_fixed() -> None:
        core_fixed_kernel((grid,), (block,), core_fixed_args)

    launches = {
        "full_fp64": launch_full_fp64,
        "full_int64_trig": launch_full_fixed_trig,
        **{f"full_{name}": launch for name, launch in custom_full_launches.items()},
        "core_fp64": launch_core_fp64,
        "core_int64_fixed": launch_core_fixed,
    }
    timing = _time_interleaved(cp, launches, warmup=args.warmup, iterations=args.iterations)

    for launch in launches.values():
        launch()
    device.synchronize()

    full_errors = {}
    for name, (out_lat, out_lon, out_h) in full_outputs.items():
        if name == "fp64":
            continue
        error = _surface_error_stats(
            cp,
            out_lat,
            out_lon,
            fp64_lat,
            fp64_lon,
            params.dst_ellipsoid.a,
        )
        if args.with_height:
            height_error = cp.abs(out_h - fp64_h)
            error.update(
                max_height_m=float(cp.max(height_error).item()),
                p99_height_m=float(cp.percentile(height_error, 99).item()),
            )
        full_errors[name] = error

    core_error = _ecef_error_stats(cp, core_outputs["int64_fixed"], core_outputs["fp64"])

    full_timing_names = ["full_fp64", "full_int64_trig", "full_int64_core"]
    core_timing_names = ["core_fp64", "core_int64_fixed"]
    for fp64_name, names in (("full_fp64", full_timing_names), ("core_fp64", core_timing_names)):
        fp64_median = timing[fp64_name]["median_ms"]
        for name in names:
            timing[name]["mcoords_per_s"] = args.n / timing[name]["median_ms"] / 1000.0
            timing[name]["speedup_vs_fp64"] = fp64_median / timing[name]["median_ms"]

    return {
        "meta": {
            "gpu": gpu_name,
            "compute_capability": f"{properties['major']}.{properties['minor']}",
            "python": platform.python_version(),
            "cupy": cp.__version__,
            "n_coords": args.n,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "with_height": args.with_height,
            "input_domain_degrees": {"latitude": lat_range, "longitude": lon_range},
            "transform": "EPSG:4326 -> EPSG:4277 Helmert",
            "trig_format": "table-free signed Q1.62",
            "coordinate_format": "signed Q24.39",
            "parameter_format": "signed Q7.56",
            "coordinate_range_m": [-16_777_216.0, 16_777_216.0],
        },
        "timing": {
            "full": {
                "fp64": timing["full_fp64"],
                "int64_trig": timing["full_int64_trig"],
                "int64_core": timing["full_int64_core"],
            },
            "core": {"fp64": timing["core_fp64"], "int64_fixed": timing["core_int64_fixed"]},
        },
        "error_vs_fp64": {"full": full_errors, "core_ecef": core_error},
    }


def _print_timing_group(name: str, n: int, timing: dict[str, dict[str, float]]) -> None:
    print(name)
    print(f"{'mode':<20} {'median ms':>10} {'Mcoord/s':>10} {'vs fp64':>10} {'p05..p95 ms':>20}")
    for mode, stats in timing.items():
        spread = f"{stats['p05_ms']:.4f}..{stats['p95_ms']:.4f}"
        print(
            f"{mode:<20} {stats['median_ms']:>10.4f} {stats['mcoords_per_s']:>10.1f} "
            f"{stats['speedup_vs_fp64']:>9.2f}x {spread:>20}"
        )
    print()


def _print_human(results: dict[str, Any]) -> None:
    meta = results["meta"]
    print(f"GPU: {meta['gpu']} (sm_{meta['compute_capability'].replace('.', '')})")
    print(f"Transform: {meta['transform']}; {meta['n_coords']:,} points")
    print(
        f"Production trig: {meta['trig_format']}; matrix experiment: "
        f"{meta['coordinate_format']} coordinates, {meta['parameter_format']} parameters"
    )
    print()
    _print_timing_group(
        "Complete geodetic -> ECEF -> Helmert -> geodetic kernel",
        meta["n_coords"],
        results["timing"]["full"],
    )
    _print_timing_group(
        "Isolated ECEF Helmert matrix core", meta["n_coords"], results["timing"]["core"]
    )

    print("Error versus fp64")
    for name, full_error in results["error_vs_fp64"]["full"].items():
        print(
            f"  {name:<18} horizontal max={full_error['max_horizontal_m']:.6g} m, "
            f"p99={full_error['p99_horizontal_m']:.6g} m, "
            f"RMS={full_error['rms_horizontal_m']:.6g} m"
        )
        if "max_height_m" in full_error:
            print(
                f"  {'':<18} height max={full_error['max_height_m']:.6g} m, "
                f"p99={full_error['p99_height_m']:.6g} m"
            )
    core_error = results["error_vs_fp64"]["core_ecef"]
    print(
        f"  {'matrix ECEF':<18} max={core_error['max_m']:.6g} m, "
        f"p99={core_error['p99_m']:.6g} m, RMS={core_error['rms_m']:.6g} m"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1_000_000, help="number of coordinates")
    parser.add_argument("--iterations", type=int, default=50, help="timed iterations per mode")
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations per mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--with-height", action="store_true", help="benchmark 3D height path")
    parser.add_argument(
        "--global-domain", action="store_true", help="sample the full geographic angle domain"
    )
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
