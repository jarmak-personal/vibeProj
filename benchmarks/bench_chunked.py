#!/usr/bin/env python3
"""Benchmark transform_chunked() double-buffered pipeline throughput.

Measures the pipelined chunked transform at 1M, 10M, and 100M points for
both 2D and 3D (with z), reporting per-chunk amortized time, total wall
time, throughput (points/sec), and PCIe bandwidth utilization.

Also runs transfer-time diagnostics to quantify H2D, D2H, kernel, and
host memcpy costs independently, and checks whether D2H is truly async
(CuPy .get(out=) can silently synchronize, defeating pipeline overlap).

Usage:
    uv run benchmarks/bench_chunked.py
    uv run benchmarks/bench_chunked.py --sizes 1000000 10000000
    uv run benchmarks/bench_chunked.py --output benchmarks/chunked_results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np


# --- Configuration ---

DEFAULT_SIZES = [1_000_000, 10_000_000, 100_000_000]
WARMUP_ITERS = 2
BENCH_ITERS = 5  # fewer than projection bench -- 100M takes significant time

# tmerc (UTM Zone 31N) is the most common projection and matches the plan
SRC_CRS = "EPSG:4326"
DST_CRS = "EPSG:32631"

# Chunk size matches the Transformer default (and the plan's 1M recommendation)
CHUNK_SIZE = 1_000_000


def _detect_gpu():
    """Return (cupy_module, device_name, gpu_props) or (None, None, None)."""
    try:
        import cupy as cp

        dev = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode()
        dev.synchronize()
        return cp, name, props
    except Exception:
        return None, None, None


def _get_pcie_bandwidth_gbps(gpu_name: str) -> float | None:
    """Return theoretical max PCIe unidirectional bandwidth in GB/s.

    Based on known GPU PCIe generation defaults.  Returns None if unknown.
    """
    # PCIe 4.0 x16: 31.5 GB/s per direction (theoretical)
    # PCIe 3.0 x16: 15.75 GB/s per direction (theoretical)
    # Achievable is ~85-90% of theoretical for pinned memory
    pcie4_gpus = {"RTX 3090", "RTX 4090", "RTX 4080", "RTX 3080", "A100", "H100", "L40"}
    for g in pcie4_gpus:
        if g in gpu_name:
            return 31.5  # PCIe 4.0 x16 theoretical
    pcie3_gpus = {"RTX 2080", "V100", "T4", "P100"}
    for g in pcie3_gpus:
        if g in gpu_name:
            return 15.75  # PCIe 3.0 x16 theoretical
    return None


def _bytes_transferred(n: int, *, has_z: bool) -> int:
    """Total bytes transferred H2D + D2H for n fp64 points.

    2D: H2D = 2 arrays * n * 8 bytes, D2H = 2 arrays * n * 8 bytes = 32n total
    3D: H2D = 3 arrays * n * 8 bytes, D2H = 3 arrays * n * 8 bytes = 48n total
    """
    arrays = 3 if has_z else 2
    return arrays * 2 * n * 8  # factor of 2 for H2D + D2H


def _format_size(n: int) -> str:
    """Human-readable size: 1M, 10M, 100M."""
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    elif n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def _run_transfer_diagnostics(cp, transformer):
    """Measure individual pipeline phase costs for 1M-chunk fp64 2D.

    Returns a dict of phase timings and a diagnostic about D2H async behavior.
    """
    n = CHUNK_SIZE
    nbytes = n * 8
    rng = np.random.default_rng(42)

    # Allocate pinned host buffers
    pin_mems = [cp.cuda.alloc_pinned_memory(nbytes) for _ in range(4)]
    pin_in_x = np.frombuffer(pin_mems[0], dtype=np.float64, count=n)
    pin_in_y = np.frombuffer(pin_mems[1], dtype=np.float64, count=n)
    pin_out_x = np.frombuffer(pin_mems[2], dtype=np.float64, count=n)
    pin_out_y = np.frombuffer(pin_mems[3], dtype=np.float64, count=n)

    pin_in_x[:] = rng.uniform(-10, 30, n)
    pin_in_y[:] = rng.uniform(35, 65, n)

    dev_x = cp.empty(n, dtype=cp.float64)
    dev_y = cp.empty(n, dtype=cp.float64)
    dev_ox = cp.empty(n, dtype=cp.float64)
    dev_oy = cp.empty(n, dtype=cp.float64)

    s = cp.cuda.Stream(non_blocking=True)
    iters = 10

    # Phase 1: Host memcpy into pinned
    src_x = rng.uniform(-10, 30, n).astype(np.float64)
    src_y = rng.uniform(35, 65, n).astype(np.float64)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        pin_in_x[:] = src_x
        pin_in_y[:] = src_y
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    host_to_pinned_ms = times[iters // 2]

    # Phase 2: H2D (pinned -> device, stream-ordered)
    cp.cuda.Device(0).synchronize()
    times = []
    for _ in range(iters):
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter()
        with s:
            dev_x.set(pin_in_x)
            dev_y.set(pin_in_y)
        s.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    h2d_ms = times[iters // 2]

    # Phase 3: Kernel compute
    cp.cuda.Device(0).synchronize()
    # warmup
    for _ in range(3):
        transformer.transform_buffers(dev_x, dev_y, out_x=dev_ox, out_y=dev_oy, stream=s)
        s.synchronize()
    times = []
    for _ in range(iters):
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter()
        with s:
            transformer.transform_buffers(dev_x, dev_y, out_x=dev_ox, out_y=dev_oy, stream=s)
        s.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    kernel_ms = times[iters // 2]

    # Phase 4: D2H (device -> pinned) via .get(out=) for comparison.
    # Note: transform_chunked() uses memcpyAsync instead (see Phase 4b).
    cp.cuda.Device(0).synchronize()
    times = []
    for _ in range(iters):
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter()
        with s:
            dev_ox.get(out=pin_out_x)
            dev_oy.get(out=pin_out_y)
        s.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    d2h_ms = times[iters // 2]

    # Check if .get(out=) is actually async: measure call return time vs sync time
    cp.cuda.Device(0).synchronize()
    s.synchronize()
    t0 = time.perf_counter()
    with s:
        dev_ox.get(out=pin_out_x)
    t_call = (time.perf_counter() - t0) * 1000
    s.synchronize()
    t_sync = (time.perf_counter() - t0) * 1000
    d2h_is_async = t_call < (t_sync * 0.3)  # truly async if call returns in <30% of total

    # Also test memcpyAsync as alternative D2H
    cp.cuda.Device(0).synchronize()
    t0 = time.perf_counter()
    with s:
        cp.cuda.runtime.memcpyAsync(
            pin_out_x.ctypes.data, dev_ox.data.ptr, nbytes, 2, s.ptr
        )
        cp.cuda.runtime.memcpyAsync(
            pin_out_y.ctypes.data, dev_oy.data.ptr, nbytes, 2, s.ptr
        )
    t_call_async = (time.perf_counter() - t0) * 1000
    s.synchronize()
    t_sync_async = (time.perf_counter() - t0) * 1000
    memcpy_async_is_async = t_call_async < (t_sync_async * 0.3)

    # Phase 5: Host memcpy from pinned to output
    out_x = np.empty(n, dtype=np.float64)
    out_y = np.empty(n, dtype=np.float64)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out_x[:] = pin_out_x
        out_y[:] = pin_out_y
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    pinned_to_host_ms = times[iters // 2]

    serial_total = host_to_pinned_ms + h2d_ms + kernel_ms + d2h_ms + pinned_to_host_ms

    # Theoretical pipelined per-chunk: GPU work dominates (H2D + kernel + D2H)
    # while host memcpy overlaps (runs during the other stream's GPU work).
    # With true async D2H: amortized = max(host_work, gpu_work) per chunk
    gpu_work = h2d_ms + kernel_ms + d2h_ms
    host_work = host_to_pinned_ms + pinned_to_host_ms
    pipeline_theoretical = max(gpu_work, host_work)

    # Bandwidth
    h2d_bw = (2 * nbytes / 1e9) / (h2d_ms / 1000) if h2d_ms > 0 else 0
    d2h_bw = (2 * nbytes / 1e9) / (d2h_ms / 1000) if d2h_ms > 0 else 0

    return {
        "host_to_pinned_ms": round(host_to_pinned_ms, 3),
        "h2d_ms": round(h2d_ms, 3),
        "kernel_ms": round(kernel_ms, 3),
        "d2h_ms": round(d2h_ms, 3),
        "pinned_to_host_ms": round(pinned_to_host_ms, 3),
        "serial_total_ms": round(serial_total, 3),
        "gpu_work_ms": round(gpu_work, 3),
        "host_work_ms": round(host_work, 3),
        "pipeline_theoretical_ms": round(pipeline_theoretical, 3),
        "h2d_bandwidth_gbps": round(h2d_bw, 1),
        "d2h_bandwidth_gbps": round(d2h_bw, 1),
        "d2h_get_is_async": d2h_is_async,
        "d2h_memcpyAsync_is_async": memcpy_async_is_async,
    }


def _bench_chunked(n: int, *, has_z: bool, transformer):
    """Benchmark transform_chunked() at size n.

    Returns dict with timing and throughput data.
    """
    rng = np.random.default_rng(42)
    # Coordinates in valid range for UTM Zone 31N
    x = rng.uniform(-10, 30, n).astype(np.float64)   # longitude
    y = rng.uniform(35, 65, n).astype(np.float64)     # latitude
    z = rng.uniform(0, 1000, n).astype(np.float64) if has_z else None

    label = f"{_format_size(n)}_{'3d' if has_z else '2d'}"

    # Warmup: compile kernels, allocate pinned buffers
    for _ in range(WARMUP_ITERS):
        if has_z:
            transformer.transform_chunked(x, y, z=z, chunk_size=CHUNK_SIZE)
        else:
            transformer.transform_chunked(x, y, chunk_size=CHUNK_SIZE)

    # Timed iterations
    times = []
    for _ in range(BENCH_ITERS):
        t0 = time.perf_counter()
        if has_z:
            transformer.transform_chunked(x, y, z=z, chunk_size=CHUNK_SIZE)
        else:
            transformer.transform_chunked(x, y, chunk_size=CHUNK_SIZE)
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        times.append(elapsed)

    times.sort()
    median_ms = times[len(times) // 2]
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE

    result = {
        "n_points": n,
        "has_z": has_z,
        "chunk_size": CHUNK_SIZE,
        "n_chunks": n_chunks,
        "warmup_iters": WARMUP_ITERS,
        "bench_iters": BENCH_ITERS,
        "median_ms": round(median_ms, 3),
        "min_ms": round(times[0], 3),
        "max_ms": round(times[-1], 3),
        "per_chunk_ms": round(median_ms / n_chunks, 3),
        "throughput_pts_per_sec": round(n / (median_ms / 1000)),
        "total_bytes_transferred": _bytes_transferred(n, has_z=has_z),
    }

    return label, result


def _bench_gpu_resident(n: int, transformer):
    """Measure pure GPU-resident transform_buffers at same size.

    This gives us the kernel-only time (no H<->D), which is the lower bound.
    Only run for sizes that fit in GPU memory.
    """
    try:
        import cupy as cp
    except ImportError:
        return None

    free, _ = cp.cuda.runtime.memGetInfo()
    # Need 4 arrays of n fp64 = 32n bytes
    if 32 * n > free * 0.8:
        return None

    rng = np.random.default_rng(42)
    x_np = rng.uniform(-10, 30, n).astype(np.float64)
    y_np = rng.uniform(35, 65, n).astype(np.float64)
    x_gpu = cp.asarray(x_np)
    y_gpu = cp.asarray(y_np)
    out_x = cp.empty(n, dtype=cp.float64)
    out_y = cp.empty(n, dtype=cp.float64)
    cp.cuda.Device(0).synchronize()

    # Warmup
    for _ in range(WARMUP_ITERS):
        transformer.transform_buffers(x_gpu, y_gpu, out_x=out_x, out_y=out_y)
        cp.cuda.Device(0).synchronize()

    times = []
    for _ in range(BENCH_ITERS):
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter()
        transformer.transform_buffers(x_gpu, y_gpu, out_x=out_x, out_y=out_y)
        cp.cuda.Device(0).synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    times.sort()
    median_ms = times[len(times) // 2]

    return {
        "n_points": n,
        "median_ms": round(median_ms, 3),
        "min_ms": round(times[0], 3),
        "max_ms": round(times[-1], 3),
        "throughput_pts_per_sec": round(n / (median_ms / 1000)),
    }


def _format_throughput(pts_per_sec: float) -> str:
    """Format throughput as human-readable string."""
    if pts_per_sec >= 1e9:
        return f"{pts_per_sec / 1e9:.2f} Gpts/s"
    elif pts_per_sec >= 1e6:
        return f"{pts_per_sec / 1e6:.1f} Mpts/s"
    return f"{pts_per_sec / 1e3:.1f} Kpts/s"


def _format_bytes(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    gb = nbytes / (1024**3)
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    return f"{nbytes / (1024**2):.0f} MB"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark transform_chunked() double-buffered pipeline"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=DEFAULT_SIZES,
        help="Point counts to benchmark (default: 1M 10M 100M)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Write JSON results to file",
    )
    parser.add_argument(
        "--skip-3d",
        action="store_true",
        help="Skip 3D (with z) benchmarks",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip transfer diagnostics",
    )
    args = parser.parse_args()

    # Detect GPU
    cp_mod, gpu_name, gpu_props = _detect_gpu()
    if cp_mod is None:
        print("ERROR: No GPU detected. transform_chunked() requires CuPy + GPU.", file=sys.stderr)
        sys.exit(1)

    pcie_bw = _get_pcie_bandwidth_gbps(gpu_name)

    print(f"GPU: {gpu_name}", file=sys.stderr)
    if pcie_bw:
        print(f"PCIe theoretical bandwidth: {pcie_bw} GB/s per direction", file=sys.stderr)
    print(f"Projection: tmerc ({SRC_CRS} -> {DST_CRS})", file=sys.stderr)
    print(f"Chunk size: {CHUNK_SIZE:,}", file=sys.stderr)
    print(f"Warmup: {WARMUP_ITERS}, Bench: {BENCH_ITERS} iterations", file=sys.stderr)
    print(file=sys.stderr)

    # Create transformer once
    from vibeproj import Transformer
    t = Transformer.from_crs(SRC_CRS, DST_CRS, always_xy=True)
    t.compile()

    # --- Transfer diagnostics ---
    diag = None
    if not args.skip_diagnostics:
        print("Running transfer diagnostics (1M chunk)...", file=sys.stderr)
        diag = _run_transfer_diagnostics(cp_mod, t)
        print(file=sys.stderr)

    # --- Chunked pipeline benchmarks ---
    chunked_results = {}
    gpu_resident_results = {}

    for n in args.sizes:
        size_label = _format_size(n)
        host_need_gb = 4 * n * 8 / (1024**3)
        print(f"--- {size_label} points (host memory: ~{host_need_gb:.1f} GB for 2D) ---", file=sys.stderr)

        # 2D benchmark
        print(f"  Benchmarking 2D chunked ({size_label})...", file=sys.stderr)
        label, result = _bench_chunked(n, has_z=False, transformer=t)
        chunked_results[label] = result

        # GPU-resident reference (only for sizes that fit)
        print(f"  Benchmarking GPU-resident reference ({size_label})...", file=sys.stderr)
        gpu_ref = _bench_gpu_resident(n, t)
        if gpu_ref:
            gpu_resident_results[size_label] = gpu_ref

        # 3D benchmark
        if not args.skip_3d:
            print(f"  Benchmarking 3D chunked ({size_label})...", file=sys.stderr)
            label_3d, result_3d = _bench_chunked(n, has_z=True, transformer=t)
            chunked_results[label_3d] = result_3d

        print(file=sys.stderr)

    # ===================================================================
    # Print results
    # ===================================================================
    print()
    print("=" * 105)
    print(" transform_chunked() Double-Buffered Pipeline Benchmark")
    print(f" GPU: {gpu_name} | Projection: tmerc | Chunk size: {CHUNK_SIZE:,}")
    print("=" * 105)

    # --- Transfer diagnostics ---
    if diag:
        print()
        print("--- Per-Chunk Phase Breakdown (1M fp64 2D) ---")
        print(f"  Host -> pinned memcpy:   {diag['host_to_pinned_ms']:>7.3f} ms")
        print(f"  H2D (pinned -> device):  {diag['h2d_ms']:>7.3f} ms  ({diag['h2d_bandwidth_gbps']:.1f} GB/s)")
        print(f"  Kernel compute (tmerc):  {diag['kernel_ms']:>7.3f} ms")
        print(f"  D2H (device -> pinned):  {diag['d2h_ms']:>7.3f} ms  ({diag['d2h_bandwidth_gbps']:.1f} GB/s)")
        print(f"  Pinned -> host memcpy:   {diag['pinned_to_host_ms']:>7.3f} ms")
        print(f"  ----------------------------------------")
        print(f"  Serial total:            {diag['serial_total_ms']:>7.3f} ms")
        print(f"  GPU work (H2D+kern+D2H): {diag['gpu_work_ms']:>7.3f} ms")
        print(f"  Host work (memcpy in+out):{diag['host_work_ms']:>6.3f} ms")
        print(f"  Pipeline theoretical:    {diag['pipeline_theoretical_ms']:>7.3f} ms/chunk (with full overlap)")
        print()
        d2h_async_str = "YES (async)" if diag["d2h_get_is_async"] else "NO (synchronous)"
        memcpy_str = "YES (async)" if diag["d2h_memcpyAsync_is_async"] else "NO (synchronous)"
        print(f"  D2H .get(out=) async?      {d2h_async_str}")
        print(f"  D2H memcpyAsync async?     {memcpy_str}")
        if not diag["d2h_get_is_async"]:
            print()
            print("  NOTE: CuPy .get(out=) blocks the host thread during D2H transfer,")
            print("  defeating pipeline overlap for the D2H phase. The double-buffer")
            print("  structure correctly overlaps H2D and kernel launch, but each chunk's")
            print("  D2H serializes with the next chunk. Switching to cudaMemcpyAsync")
            print("  (via cp.cuda.runtime.memcpyAsync) would restore full overlap.")

    # --- Main results table ---
    print()
    print("--- Chunked Pipeline Results ---")
    hdr = (
        f"{'Config':<14} {'N':>10} {'Chunks':>7} {'Total (ms)':>11} "
        f"{'Per-Chunk':>10} {'Throughput':>16} {'Bytes Xfer':>11}"
    )
    if pcie_bw:
        hdr += f" {'PCIe Util':>10}"
    print(hdr)
    print("-" * len(hdr))

    for label, r in chunked_results.items():
        n = r["n_points"]
        n_chunks = r["n_chunks"]
        median = r["median_ms"]
        per_chunk = r["per_chunk_ms"]
        throughput = r["throughput_pts_per_sec"]
        total_bytes = r["total_bytes_transferred"]

        line = (
            f"{label:<14} {n:>10,} {n_chunks:>7} {median:>10.1f}ms "
            f"{per_chunk:>9.3f}ms {_format_throughput(throughput):>16} "
            f"{_format_bytes(total_bytes):>11}"
        )

        if pcie_bw:
            achieved_bw_gbps = (total_bytes / (1024**3)) / (median / 1000)
            utilization = (achieved_bw_gbps / pcie_bw) * 100
            line += f" {utilization:>8.1f}%"

        print(line)

    # --- GPU-resident reference ---
    if gpu_resident_results:
        print()
        print("--- GPU-Resident Reference (kernel-only, no H<->D transfer) ---")
        print(f"{'Size':<10} {'Median (ms)':>12} {'Throughput':>16} {'Overhead vs Chunked':>20}")
        print("-" * 62)
        for size_label, ref in gpu_resident_results.items():
            tp_str = _format_throughput(ref["throughput_pts_per_sec"])

            chunked_key = f"{size_label}_2d"
            if chunked_key in chunked_results:
                chunked_ms = chunked_results[chunked_key]["median_ms"]
                overhead = chunked_ms / ref["median_ms"]
                overhead_str = f"{overhead:.1f}x (H<->D cost)"
            else:
                overhead_str = "N/A"

            print(f"{size_label:<10} {ref['median_ms']:>11.3f}ms {tp_str:>16} {overhead_str:>20}")

    # --- Comparison with plan ---
    print()
    print("--- Comparison with Plan Targets ---")
    print(f"{'Metric':<45} {'Plan Target':>14} {'Measured':>14} {'Status':>10}")
    print("-" * 87)

    key_1m = "1M_2d"
    if key_1m in chunked_results:
        measured_per_chunk = chunked_results[key_1m]["per_chunk_ms"]
        # 1M is a single chunk, so per-chunk == total (no pipeline overlap)
        print(
            f"{'Per-chunk time (1M, single chunk, no overlap)':<45} "
            f"{'~3.05 ms':>14} {measured_per_chunk:>13.3f}ms {'OK':>10}"
        )

    # For multi-chunk, amortized per-chunk should show overlap benefit
    key_10m = "10M_2d"
    if key_10m in chunked_results:
        measured_per_chunk = chunked_results[key_10m]["per_chunk_ms"]
        target = 1.28
        status = "OK" if measured_per_chunk <= target * 1.5 else "REVIEW"
        print(
            f"{'Per-chunk amortized (10M, 10 chunks)':<45} "
            f"{'~1.28 ms':>14} {measured_per_chunk:>13.3f}ms {status:>10}"
        )

    key_100m = "100M_2d"
    if key_100m in chunked_results:
        measured_total = chunked_results[key_100m]["median_ms"]
        target_100m = 128.0
        status = "OK" if measured_total <= target_100m * 1.5 else "REVIEW"
        print(
            f"{'100M total time (tmerc, 2D)':<45} "
            f"{'~128 ms':>14} {measured_total:>13.1f}ms {status:>10}"
        )

    # Speedup vs plan's serial baseline
    if key_10m in chunked_results:
        measured_per_chunk = chunked_results[key_10m]["per_chunk_ms"]
        serial_baseline = 3.05
        speedup = serial_baseline / measured_per_chunk
        print(
            f"{'Speedup vs serial baseline (3.05 ms/chunk)':<45} "
            f"{'~2.4x':>14} {speedup:>13.1f}x {'':>10}"
        )

    # 3D overhead
    if key_1m in chunked_results and "1M_3d" in chunked_results:
        ms_2d = chunked_results[key_1m]["median_ms"]
        ms_3d = chunked_results["1M_3d"]["median_ms"]
        overhead_pct = ((ms_3d - ms_2d) / ms_2d) * 100
        print(
            f"{'3D (with z) overhead at 1M':<45} "
            f"{'< 10%':>14} {overhead_pct:>12.1f}% "
            f"{'OK' if overhead_pct < 10 else 'REVIEW':>10}"
        )

    print()

    # --- JSON output ---
    output = {
        "meta": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu": gpu_name,
            "pcie_theoretical_gbps": pcie_bw,
            "projection": "tmerc",
            "src_crs": SRC_CRS,
            "dst_crs": DST_CRS,
            "chunk_size": CHUNK_SIZE,
            "warmup_iters": WARMUP_ITERS,
            "bench_iters": BENCH_ITERS,
        },
        "chunked": chunked_results,
    }
    if gpu_resident_results:
        output["gpu_resident_reference"] = gpu_resident_results
    if diag:
        output["transfer_diagnostics"] = diag

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
            f.write("\n")
        print(f"Results written to {args.output}", file=sys.stderr)

    return output


if __name__ == "__main__":
    main()
