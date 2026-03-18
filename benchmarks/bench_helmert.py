#!/usr/bin/env python3
"""Benchmark cross-datum (Helmert) transforms on CPU and GPU, output JSON.

Usage:
    uv run benchmarks/bench_helmert.py run [--n N] [--output FILE]
    uv run benchmarks/bench_helmert.py compare BASE.json CURRENT.json [--threshold PCT]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np


# Cross-datum benchmarks — all involve Helmert datum shifts
BENCHMARKS = [
    ("helmert_fwd", "EPSG:4326", "EPSG:27700"),  # WGS84 -> OSGB36 BNG
    ("helmert_p2p", "EPSG:32631", "EPSG:27700"),  # UTM -> BNG cross-datum
    ("helmert_ll", "EPSG:4326", "EPSG:4277"),  # longlat cross-datum
]

WARMUP_ITERS = 3
BENCH_ITERS = 10


def _detect_gpu():
    """Return (cupy_module, device_name) or (None, None)."""
    try:
        import cupy as cp

        dev = cp.cuda.Device(0)
        name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        dev.synchronize()
        return cp, name
    except Exception:
        return None, None


def _bench_cpu(n):
    """Benchmark cross-datum projections on CPU (NumPy)."""
    import warnings

    from vibeproj import Transformer

    rng = np.random.default_rng(42)
    lon = rng.uniform(-10, 2, n).astype(np.float64)
    lat = rng.uniform(49, 59, n).astype(np.float64)

    results = {}
    for label, src, dst in BENCHMARKS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = Transformer.from_crs(src, dst)

        # For proj_to_proj, input is projected coordinates
        if label == "helmert_p2p":
            t_fwd = Transformer.from_crs("EPSG:4326", src)
            x_in, y_in = t_fwd.transform(lon[:1000], lat[:1000])
            x_full, y_full = t_fwd.transform(lon, lat)
        else:
            x_in, y_in = lon[:1000], lat[:1000]
            x_full, y_full = lon, lat

        # Warmup
        for _ in range(WARMUP_ITERS):
            t.transform(x_in, y_in)

        # Timed
        times = []
        for _ in range(BENCH_ITERS):
            t0 = time.perf_counter()
            t.transform(x_full, y_full)
            times.append((time.perf_counter() - t0) * 1000)

        times.sort()
        results[label] = {
            "median_ms": round(times[len(times) // 2], 3),
            "min_ms": round(times[0], 3),
            "max_ms": round(times[-1], 3),
        }

    return results


def _bench_gpu(cp, n):
    """Benchmark cross-datum projections on GPU (CuPy)."""
    import warnings

    from vibeproj import Transformer

    rng = np.random.default_rng(42)
    lon_np = rng.uniform(-10, 2, n).astype(np.float64)
    lat_np = rng.uniform(49, 59, n).astype(np.float64)
    lon = cp.asarray(lon_np)
    lat = cp.asarray(lat_np)
    cp.cuda.Device(0).synchronize()

    out_x = cp.empty(n, dtype=cp.float64)
    out_y = cp.empty(n, dtype=cp.float64)

    results = {}
    for label, src, dst in BENCHMARKS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = Transformer.from_crs(src, dst)

        # For proj_to_proj, input is projected coordinates
        if label == "helmert_p2p":
            t_fwd = Transformer.from_crs("EPSG:4326", src)
            x_in, y_in = t_fwd.transform_buffers(lon, lat, out_x=out_x, out_y=out_y)
            x_full = x_in.copy()
            y_full = y_in.copy()
        else:
            x_full, y_full = lon, lat

        # Warmup
        for _ in range(WARMUP_ITERS):
            t.transform_buffers(x_full, y_full, out_x=out_x, out_y=out_y)
            cp.cuda.Device(0).synchronize()

        # Timed
        times = []
        for _ in range(BENCH_ITERS):
            cp.cuda.Device(0).synchronize()
            t0 = time.perf_counter()
            t.transform_buffers(x_full, y_full, out_x=out_x, out_y=out_y)
            cp.cuda.Device(0).synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        times.sort()
        results[label] = {
            "median_ms": round(times[len(times) // 2], 3),
            "min_ms": round(times[0], 3),
            "max_ms": round(times[-1], 3),
        }

    return results


def cmd_run(args):
    """Run benchmarks, emit JSON."""
    n = args.n

    meta = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "n_coords": n,
        "warmup_iters": WARMUP_ITERS,
        "bench_iters": BENCH_ITERS,
    }

    print(f"Benchmarking Helmert CPU ({n:,} coords)...", file=sys.stderr)
    cpu_results = _bench_cpu(n)

    cp, gpu_name = _detect_gpu()
    gpu_results = None
    if cp is not None:
        print(f"Benchmarking Helmert GPU: {gpu_name} ({n:,} coords)...", file=sys.stderr)
        gpu_results = _bench_gpu(cp, n)
        meta["gpu"] = gpu_name
    else:
        print("No GPU detected, skipping GPU benchmarks.", file=sys.stderr)

    output = {"meta": meta, "cpu": cpu_results}
    if gpu_results is not None:
        output["gpu"] = gpu_results

    text = json.dumps(output, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(text)


def cmd_compare(args):
    """Compare two benchmark result files."""
    with open(args.base) as f:
        base = json.load(f)
    with open(args.current) as f:
        current = json.load(f)

    threshold = args.threshold
    has_regression = False

    for device in ("cpu", "gpu"):
        if device not in base or device not in current:
            continue

        print(f"\n{'=' * 60}")
        print(f" {device.upper()} Helmert Performance Comparison")
        print(f" base: {args.base}  vs  current: {args.current}")
        print(f"{'=' * 60}")
        print(
            f"{'Benchmark':<14} {'Base (ms)':>10} {'Current (ms)':>12} {'Change':>10} {'Status':>8}"
        )
        print("-" * 58)

        base_d = base[device]
        curr_d = current[device]

        for proj in base_d:
            if proj not in curr_d:
                continue

            b = base_d[proj]["median_ms"]
            c = curr_d[proj]["median_ms"]

            if b > 0:
                pct = ((c - b) / b) * 100
            else:
                pct = 0.0

            if pct > threshold:
                status = "REGRESS"
                has_regression = True
            elif pct < -threshold:
                status = "FASTER"
            else:
                status = "OK"

            sign = "+" if pct >= 0 else ""
            print(f"{proj:<14} {b:>10.3f} {c:>12.3f} {sign}{pct:>8.1f}%  {status:>8}")

    print()
    if has_regression:
        print(f"FAIL: Regressions detected (>{threshold}% slower)")
        sys.exit(1)
    else:
        print(f"PASS: No regressions (threshold: {threshold}%)")


def main():
    parser = argparse.ArgumentParser(description="vibeProj Helmert benchmarks")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run benchmarks")
    run_p.add_argument("--n", type=int, default=1_000_000, help="Number of coordinates")
    run_p.add_argument("--output", "-o", help="Write JSON to file instead of stdout")

    cmp_p = sub.add_parser("compare", help="Compare two benchmark results")
    cmp_p.add_argument("base", help="Baseline results JSON")
    cmp_p.add_argument("current", help="Current results JSON")
    cmp_p.add_argument(
        "--threshold",
        type=float,
        default=15.0,
        help="Regression threshold percentage (default: 15%%)",
    )

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
