#!/usr/bin/env python3
"""Benchmark key projections on CPU and GPU, output JSON.

Usage:
    uv run benchmarks/bench_projections.py run [--n N] [--output FILE]
    uv run benchmarks/bench_projections.py compare BASE.json CURRENT.json [--threshold PCT]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np


# Projections to benchmark (matches README table)
BENCHMARKS = [
    ("tmerc", "EPSG:4326", "EPSG:32631"),
    ("lcc", "EPSG:4326", "EPSG:2154"),
    ("aea", "EPSG:4326", "EPSG:5070"),
    ("webmerc", "EPSG:4326", "EPSG:3857"),
    ("eqearth", "EPSG:4326", "EPSG:8857"),
    ("eqc", "EPSG:4326", "EPSG:4087"),
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
    """Benchmark all projections on CPU (NumPy)."""
    from vibeproj import Transformer

    rng = np.random.default_rng(42)
    lat = rng.uniform(35, 65, n).astype(np.float64)
    lon = rng.uniform(-10, 30, n).astype(np.float64)

    results = {}
    for label, src, dst in BENCHMARKS:
        t = Transformer.from_crs(src, dst, always_xy=False)

        # Warmup
        for _ in range(WARMUP_ITERS):
            t.transform(lat[:1000], lon[:1000])

        # Timed
        times = []
        for _ in range(BENCH_ITERS):
            t0 = time.perf_counter()
            t.transform(lat, lon)
            times.append((time.perf_counter() - t0) * 1000)

        times.sort()
        results[label] = {
            "median_ms": round(times[len(times) // 2], 3),
            "min_ms": round(times[0], 3),
            "max_ms": round(times[-1], 3),
        }

    return results


def _bench_gpu(cp, n):
    """Benchmark all projections on GPU (CuPy, fused kernels)."""
    from vibeproj import Transformer

    rng = np.random.default_rng(42)
    lat_np = rng.uniform(35, 65, n).astype(np.float64)
    lon_np = rng.uniform(-10, 30, n).astype(np.float64)
    lat = cp.asarray(lat_np)
    lon = cp.asarray(lon_np)
    cp.cuda.Device(0).synchronize()

    # Pre-allocate output buffers
    out_x = cp.empty(n, dtype=cp.float64)
    out_y = cp.empty(n, dtype=cp.float64)

    results = {}
    for label, src, dst in BENCHMARKS:
        t = Transformer.from_crs(src, dst, always_xy=False)

        # Warmup (uses transform_buffers for zero-overhead path)
        for _ in range(WARMUP_ITERS):
            t.transform_buffers(lat, lon, out_x=out_x, out_y=out_y)
            cp.cuda.Device(0).synchronize()

        # Timed
        times = []
        for _ in range(BENCH_ITERS):
            cp.cuda.Device(0).synchronize()
            t0 = time.perf_counter()
            t.transform_buffers(lat, lon, out_x=out_x, out_y=out_y)
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

    # CPU
    print(f"Benchmarking CPU ({n:,} coords)...", file=sys.stderr)
    cpu_results = _bench_cpu(n)

    # GPU
    cp, gpu_name = _detect_gpu()
    gpu_results = None
    if cp is not None:
        print(f"Benchmarking GPU: {gpu_name} ({n:,} coords)...", file=sys.stderr)
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
        print(f" {device.upper()} Performance Comparison")
        print(f" base: {args.base}  vs  current: {args.current}")
        print(f"{'=' * 60}")
        print(
            f"{'Projection':<12} {'Base (ms)':>10} {'Current (ms)':>12} {'Change':>10} {'Status':>8}"
        )
        print("-" * 56)

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
            print(f"{proj:<12} {b:>10.3f} {c:>12.3f} {sign}{pct:>8.1f}%  {status:>8}")

    print()
    if has_regression:
        print(f"FAIL: Regressions detected (>{threshold}% slower)")
        sys.exit(1)
    else:
        print(f"PASS: No regressions (threshold: {threshold}%)")


def main():
    parser = argparse.ArgumentParser(description="vibeProj projection benchmarks")
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
