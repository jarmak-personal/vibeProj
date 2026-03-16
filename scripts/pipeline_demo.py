#!/usr/bin/env python3
"""End-to-end pipeline demo: synthetic coords → GPU transform → profile.

Generates random coordinates and transforms them through multiple projections,
profiling every step. Optionally reads GeoParquet fixtures if available.

Usage:
    uv run scripts/pipeline_demo.py
    uv run scripts/pipeline_demo.py /path/to/fixture.parquet
"""

from __future__ import annotations

import sys
import time

import numpy as np


def timer(label):
    """Context manager that prints elapsed time."""
    class Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *_):
            self.elapsed = time.perf_counter() - self.t0
            print(f"  {label:<45} {self.elapsed*1000:>8.2f} ms")
    return Timer()


def _load_coords_from_parquet(path):
    """Load coordinates from a GeoParquet file (requires geopandas + shapely)."""
    import geopandas as gpd
    import shapely

    gdf = gpd.read_parquet(path)
    print(f"  Rows: {len(gdf):,}  CRS: EPSG:{gdf.crs.to_epsg()}")
    coords = shapely.get_coordinates(gdf.geometry.values)
    raw_x, raw_y = coords[:, 0], coords[:, 1]
    # Scale synthetic [0, 1000] coords to realistic European range
    lon = (raw_x / 1000.0) * 40.0 - 10.0   # [-10, 30]
    lat = (raw_y / 1000.0) * 30.0 + 35.0    # [35, 65]
    return lat, lon


def _generate_synthetic_coords(n=500_000):
    """Generate random coordinates in a European-ish range."""
    rng = np.random.default_rng(42)
    lat = rng.uniform(35, 65, n).astype(np.float64)
    lon = rng.uniform(-10, 30, n).astype(np.float64)
    return lat, lon


def main():
    # ── Step 0: Imports ──────────────────────────────────────────────
    print("=" * 65)
    print("vibeProj Pipeline Demo")
    print("=" * 65)
    print()

    with timer("Import vibeproj"):
        from vibeproj import Transformer

    has_gpu = False
    try:
        with timer("Import cupy + detect GPU"):
            import cupy as cp
            dev = cp.cuda.Device(0)
            cc = dev.compute_capability
            ratio = dev.attributes.get("SingleToDoublePrecisionPerfRatio", "?")
            has_gpu = True
        print(f"  GPU: sm_{cc[0]}{cc[1]}, fp32:fp64 = {ratio}:1")
    except Exception:
        print("  No GPU detected — CPU-only mode")

    print()

    # ── Step 1: Load or generate coordinates ─────────────────────────
    print("── Step 1: Prepare coordinates ──")

    if len(sys.argv) > 1:
        fixture_path = sys.argv[1]
        with timer(f"Read {fixture_path}"):
            lat, lon = _load_coords_from_parquet(fixture_path)
    else:
        n = 500_000
        with timer(f"Generate {n:,} synthetic coordinates"):
            lat, lon = _generate_synthetic_coords(n)

    n_coords = len(lat)
    print(f"  Coordinates: {n_coords:,}")
    print(f"  lat: [{lat.min():.1f}, {lat.max():.1f}]  lon: [{lon.min():.1f}, {lon.max():.1f}]")

    # ── Step 2: GPU transfer ─────────────────────────────────────────
    if has_gpu:
        print()
        print("── Step 2: Transfer to GPU ──")

        with timer("Host → Device (lat, lon arrays)"):
            lat_gpu = cp.asarray(lat, dtype=cp.float64)
            lon_gpu = cp.asarray(lon, dtype=cp.float64)
            cp.cuda.Device(0).synchronize()
        mb = (lat_gpu.nbytes + lon_gpu.nbytes) / 1e6
        print(f"  Transferred: {mb:.1f} MB ({n_coords:,} x 2 x fp64)")
    else:
        lat_gpu, lon_gpu = lat, lon

    # ── Step 3: Transform through multiple projections ───────────────
    print()
    print("── Step 3: Projection transforms ──")

    projections = [
        ("WGS84 -> UTM 31N",         "EPSG:4326", "EPSG:32631"),
        ("WGS84 -> Web Mercator",     "EPSG:4326", "EPSG:3857"),
        ("WGS84 -> LAEA Europe",      "EPSG:4326", "EPSG:3035"),
        ("WGS84 -> France Lambert",   "EPSG:4326", "EPSG:2154"),
        ("WGS84 -> Albers CONUS",     "EPSG:4326", "EPSG:5070"),
        ("WGS84 -> Equal Earth",      "EPSG:4326", "EPSG:8857"),
        ("UTM 31N -> Web Mercator",   "EPSG:32631", "EPSG:3857"),
    ]

    results = {}

    for label, src, dst in projections:
        t = Transformer.from_crs(src, dst)

        # Use the right input for each source CRS
        if src == "EPSG:4326":
            in_x, in_y = lat_gpu, lon_gpu
        else:
            # Use previous result as input
            in_x, in_y = results[src]

        # Warmup
        t.transform(in_x[:100], in_y[:100])
        if has_gpu:
            cp.cuda.Device(0).synchronize()

        # Timed run
        with timer(f"{label} ({n_coords:,} coords)") as tm:
            out_x, out_y = t.transform(in_x, in_y)
            if has_gpu:
                cp.cuda.Device(0).synchronize()

        tput = n_coords / tm.elapsed / 1e6
        print(f"    -> {tput:.0f}M coords/sec")

        # Store result for chained transforms
        results[dst] = (out_x, out_y)

    # ── Step 4: Zero-copy buffer API ─────────────────────────────────
    if has_gpu:
        print()
        print("── Step 4: Zero-copy transform_buffers() API ──")

        t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
        out_x = cp.empty(n_coords, dtype=cp.float64)
        out_y = cp.empty(n_coords, dtype=cp.float64)

        # Warmup
        t.transform_buffers(lat_gpu, lon_gpu, out_x=out_x, out_y=out_y)
        cp.cuda.Device(0).synchronize()

        with timer(f"transform_buffers (pre-alloc, {n_coords:,} coords)") as tm:
            rx, ry = t.transform_buffers(lat_gpu, lon_gpu, out_x=out_x, out_y=out_y)
            cp.cuda.Device(0).synchronize()

        assert rx is out_x, "Output buffer not reused!"
        tput = n_coords / tm.elapsed / 1e6
        print(f"    -> {tput:.0f}M coords/sec (zero-alloc)")

    # ── Step 5: Roundtrip verification ───────────────────────────────
    print()
    print("── Step 5: Roundtrip accuracy verification ──")

    t_fwd = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    t_inv = Transformer.from_crs("EPSG:32631", "EPSG:4326")

    proj_x, proj_y = t_fwd.transform(lat_gpu, lon_gpu)
    lat2, lon2 = t_inv.transform(proj_x, proj_y)

    if has_gpu:
        err_lat = float(cp.max(cp.abs(lat2 - lat_gpu)))
        err_lon = float(cp.max(cp.abs(lon2 - lon_gpu)))
    else:
        err_lat = float(np.max(np.abs(lat2 - lat_gpu)))
        err_lon = float(np.max(np.abs(lon2 - lon_gpu)))

    err_meters = max(err_lat, err_lon) * 111_000  # rough deg -> m
    print(f"  Max roundtrip error: {err_meters:.2e} meters")
    print(f"  (lat: {err_lat:.2e} deg, lon: {err_lon:.2e} deg)")

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"Pipeline complete: {n_coords:,} coordinates through 7 transforms")
    if has_gpu:
        print("All transforms ran on GPU via fused NVRTC kernels")
    print("=" * 65)


if __name__ == "__main__":
    main()
