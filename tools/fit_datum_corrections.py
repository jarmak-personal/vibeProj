#!/usr/bin/env python3
"""Offline tool to fit SVD-compressed datum corrections.

Samples pyproj at a dense grid, subtracts our Helmert prediction (if any),
and fits a truncated SVD to the residual.  Outputs Python source code for
``DatumCorrectionData`` that can be pasted into ``_datum_corrections.py``.

Usage:
    uv run python tools/fit_datum_corrections.py \
        --src-crs EPSG:4267 --dst-crs EPSG:4269 \
        --n-lat 100 --n-lon 150 --rank 10 --target-accuracy 0.05
"""

from __future__ import annotations

import argparse
import math
import sys
import textwrap

import numpy as np
from numpy.linalg import svd
from pyproj import CRS, Transformer as ProjTransformer


def _get_coverage_bbox(src_crs: CRS, dst_crs: CRS) -> tuple[float, float, float, float]:
    """Determine the geographic bounding box for the transform.

    Returns (lat_min, lat_max, lon_min, lon_max).
    """
    # Try area of use from pyproj
    aou = src_crs.area_of_use
    if aou is not None:
        west, east = aou.west, aou.east
        # Handle antimeridian crossing (west > east in longitude)
        # Fall back to CONUS bbox for NAD27 which wraps Alaska
        if west > east:
            # NAD27: CONUS + southern Canada is the primary use case
            return (24.0, 50.0, -125.0, -66.0)
        return (aou.south, aou.north, west, east)

    # Fallback: CONUS
    return (24.0, 50.0, -125.0, -66.0)


def _sample_pyproj(
    src_crs: str,
    dst_crs: str,
    bbox: tuple[float, float, float, float],
    n_lat: int,
    n_lon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample pyproj at a regular grid.

    Returns (lat_grid, lon_grid, out_lat, out_lon, dlat_arcsec, dlon_arcsec).
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()

    # pyproj with always_xy=False: input is (lat, lon)
    pt = ProjTransformer.from_crs(src_crs, dst_crs, always_xy=False)
    out_lat, out_lon = pt.transform(lat_flat, lon_flat)

    # Shifts in arcseconds
    dlat_as = (out_lat - lat_flat) * 3600.0
    dlon_as = (out_lon - lon_flat) * 3600.0

    return (
        lat_grid,
        lon_grid,
        out_lat.reshape(n_lat, n_lon),
        out_lon.reshape(n_lat, n_lon),
        dlat_as.reshape(n_lat, n_lon),
        dlon_as.reshape(n_lat, n_lon),
    )


def _apply_our_helmert(
    lat_flat: np.ndarray,
    lon_flat: np.ndarray,
    src_crs: str,
    dst_crs: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Apply our Helmert and return (helmert_lat, helmert_lon) or None."""
    from vibeproj.crs import extract_helmert, parse_crs_input

    src = parse_crs_input(src_crs)
    dst = parse_crs_input(dst_crs)
    helmert = extract_helmert(src, dst)
    if helmert is None:
        return None

    from vibeproj.helmert import apply_helmert

    h_lat, h_lon = apply_helmert(lat_flat, lon_flat, helmert, np)
    return h_lat, h_lon


def _fit_svd(
    grid: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit truncated SVD to a 2D grid.

    Returns (S, U, Vt, reconstructed) where:
    - S: (rank,) singular values
    - U: (rank, n_lat) left singular vectors
    - Vt: (rank, n_lon) right singular vectors
    - reconstructed: (n_lat, n_lon) approximation
    """
    U_full, S_full, Vt_full = svd(grid, full_matrices=False)

    rank = min(rank, len(S_full))
    S = S_full[:rank]
    U = U_full[:, :rank].T  # (rank, n_lat)
    Vt = Vt_full[:rank, :]  # (rank, n_lon)

    # Reconstruct
    reconstructed = np.zeros_like(grid)
    for k in range(rank):
        reconstructed += S[k] * np.outer(U[k], Vt[k])

    return S, U, Vt, reconstructed


def _format_tuple(arr: np.ndarray, indent: int = 8) -> str:
    """Format a 1D array as a Python tuple literal, wrapped at 100 cols."""
    prefix = " " * indent
    items = [f"{v:.15e}" for v in arr]
    lines = []
    line = prefix + "("
    for i, item in enumerate(items):
        candidate = item + ("," if i < len(items) - 1 else ",)")
        if len(line) + len(candidate) + 1 > 100:
            lines.append(line)
            line = prefix + " " + candidate
        else:
            if line.endswith("("):
                line += candidate
            else:
                line += " " + candidate
    if not line.strip().endswith(")"):
        line += ")"
    lines.append(line)
    return "\n".join(lines)


def _format_tuple_of_tuples(arr2d: np.ndarray, indent: int = 8) -> str:
    """Format a 2D array as a tuple of tuples."""
    prefix = " " * indent
    inner = []
    for row in arr2d:
        inner.append(_format_tuple(row, indent=indent + 4))
    return prefix + "(\n" + ",\n".join(inner) + ",\n" + prefix + ")"


def main():
    parser = argparse.ArgumentParser(description="Fit SVD-compressed datum corrections")
    parser.add_argument("--src-crs", required=True, help="Source CRS (e.g. EPSG:4267)")
    parser.add_argument("--dst-crs", required=True, help="Destination CRS (e.g. EPSG:4269)")
    parser.add_argument("--n-lat", type=int, default=100, help="Grid rows (default: 100)")
    parser.add_argument("--n-lon", type=int, default=150, help="Grid columns (default: 150)")
    parser.add_argument("--rank", type=int, default=10, help="SVD rank (default: 10)")
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.05,
        help="Target accuracy in meters (default: 0.05)",
    )
    parser.add_argument(
        "--n-test-points",
        type=int,
        default=5000,
        help="Number of random test points for validation (default: 5000)",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Override bounding box (default: auto-detect from CRS area of use)",
    )
    args = parser.parse_args()

    src_crs_str = args.src_crs
    dst_crs_str = args.dst_crs
    n_lat = args.n_lat
    n_lon = args.n_lon
    rank = args.rank

    print(f"Fitting SVD correction: {src_crs_str} -> {dst_crs_str}")
    print(f"Grid: {n_lat} x {n_lon}, rank: {rank}")
    print()

    # Determine bbox
    src_crs = CRS.from_user_input(src_crs_str)
    dst_crs = CRS.from_user_input(dst_crs_str)
    if args.bbox is not None:
        bbox = tuple(args.bbox)
    else:
        bbox = _get_coverage_bbox(src_crs, dst_crs)
    print(f"Bounding box: lat [{bbox[0]:.1f}, {bbox[1]:.1f}], lon [{bbox[2]:.1f}, {bbox[3]:.1f}]")

    # Sample pyproj
    print("Sampling pyproj at grid points...")
    lat_grid, lon_grid, out_lat, out_lon, dlat_as, dlon_as = _sample_pyproj(
        src_crs_str, dst_crs_str, bbox, n_lat, n_lon
    )
    print(f"  dlat range: {dlat_as.min():.4f} to {dlat_as.max():.4f} arcsec")
    print(f"  dlon range: {dlon_as.min():.4f} to {dlon_as.max():.4f} arcsec")

    # Check if we have Helmert for this pair
    has_helmert = False
    print("Checking for Helmert parameters...")
    helmert_result = _apply_our_helmert(
        lat_grid.flatten(), lon_grid.flatten(), src_crs_str, dst_crs_str
    )
    if helmert_result is not None:
        has_helmert = True
        h_lat, h_lon = helmert_result
        h_dlat_as = (h_lat - lat_grid.flatten()) * 3600.0
        h_dlon_as = (h_lon - lon_grid.flatten()) * 3600.0

        # Residual = pyproj - helmert
        residual_lat = dlat_as - h_dlat_as.reshape(n_lat, n_lon)
        residual_lon = dlon_as - h_dlon_as.reshape(n_lat, n_lon)
        print(f"  Helmert found. Fitting residual.")
        print(f"  Helmert lat range: {h_dlat_as.min():.4f} to {h_dlat_as.max():.4f} arcsec")
        print(
            f"  Residual lat range: {residual_lat.min():.4f} to {residual_lat.max():.4f} arcsec"
        )
        fit_lat = residual_lat
        fit_lon = residual_lon
    else:
        has_helmert = False
        fit_lat = dlat_as
        fit_lon = dlon_as
        print("  No Helmert available. Fitting full correction.")

    print()

    # Fit SVD
    print(f"Fitting SVD (rank={rank})...")
    s_lat, u_lat, vt_lat, recon_lat = _fit_svd(fit_lat, rank)
    s_lon, u_lon, vt_lon, recon_lon = _fit_svd(fit_lon, rank)

    # Grid-level accuracy
    err_lat_as = fit_lat - recon_lat
    err_lon_as = fit_lon - recon_lon

    # Convert to meters at each grid point
    cos_lat = np.cos(np.radians(lat_grid))
    m_per_as = 111320.0 / 3600.0  # meters per arcsecond at equator
    err_lat_m = err_lat_as * m_per_as
    err_lon_m = err_lon_as * m_per_as * cos_lat
    err_total_m = np.sqrt(err_lat_m**2 + err_lon_m**2)

    print(f"  Grid max error: {err_total_m.max():.4f} m")
    print(f"  Grid RMS error: {np.sqrt(np.mean(err_total_m**2)):.4f} m")
    print(f"  Grid P95 error: {np.percentile(err_total_m, 95):.4f} m")
    print()

    # Validate on random test points
    print(f"Validating on {args.n_test_points} random test points...")
    rng = np.random.default_rng(42)
    lat_min, lat_max, lon_min, lon_max = bbox
    # Shrink bbox slightly to avoid edge effects
    margin_lat = (lat_max - lat_min) * 0.02
    margin_lon = (lon_max - lon_min) * 0.02
    test_lats = rng.uniform(lat_min + margin_lat, lat_max - margin_lat, args.n_test_points)
    test_lons = rng.uniform(lon_min + margin_lon, lon_max - margin_lon, args.n_test_points)

    # pyproj reference
    pt = ProjTransformer.from_crs(src_crs_str, dst_crs_str, always_xy=False)
    ref_lat, ref_lon = pt.transform(test_lats, test_lons)

    # Filter out points where pyproj returns zero shift (outside grid coverage).
    # These points are outside the NADCON5 grid and cannot be corrected.
    ref_dlat = np.abs(ref_lat - test_lats)
    ref_dlon = np.abs(ref_lon - test_lons)
    covered = (ref_dlat > 1e-12) | (ref_dlon > 1e-12)
    n_covered = covered.sum()
    n_uncovered = (~covered).sum()
    print(f"  {n_covered} points within grid coverage, {n_uncovered} outside (excluded)")
    test_lats = test_lats[covered]
    test_lons = test_lons[covered]
    ref_lat = ref_lat[covered]
    ref_lon = ref_lon[covered]

    # Our prediction: Helmert + SVD reconstruction
    if has_helmert:
        h_result = _apply_our_helmert(test_lats, test_lons, src_crs_str, dst_crs_str)
        pred_lat_base = h_result[0]
        pred_lon_base = h_result[1]
    else:
        pred_lat_base = test_lats.copy()
        pred_lon_base = test_lons.copy()

    # Evaluate SVD correction at test points using bilinear interpolation
    # of reconstructed grid node values (matches apply_svd_correction).
    test_lat_idx = (test_lats - lat_min) / (lat_max - lat_min) * (n_lat - 1)
    test_lon_idx = (test_lons - lon_min) / (lon_max - lon_min) * (n_lon - 1)
    test_lat_idx = np.clip(test_lat_idx, 0, n_lat - 1.0)
    test_lon_idx = np.clip(test_lon_idx, 0, n_lon - 1.0)

    lat_i0 = np.floor(test_lat_idx).astype(int)
    lon_i0 = np.floor(test_lon_idx).astype(int)
    lat_i1 = np.minimum(lat_i0 + 1, n_lat - 1)
    lon_i1 = np.minimum(lon_i0 + 1, n_lon - 1)
    lat_frac = test_lat_idx - lat_i0
    lon_frac = test_lon_idx - lon_i0

    n_test = len(test_lats)

    # Reconstruct at 4 corners then bilinear interpolate
    dlat_00 = np.zeros(n_test)
    dlat_01 = np.zeros(n_test)
    dlat_10 = np.zeros(n_test)
    dlat_11 = np.zeros(n_test)
    dlon_00 = np.zeros(n_test)
    dlon_01 = np.zeros(n_test)
    dlon_10 = np.zeros(n_test)
    dlon_11 = np.zeros(n_test)

    for k in range(rank):
        u0_lat = u_lat[k][lat_i0]
        u1_lat = u_lat[k][lat_i1]
        v0_lat = vt_lat[k][lon_i0]
        v1_lat = vt_lat[k][lon_i1]
        dlat_00 += s_lat[k] * u0_lat * v0_lat
        dlat_01 += s_lat[k] * u0_lat * v1_lat
        dlat_10 += s_lat[k] * u1_lat * v0_lat
        dlat_11 += s_lat[k] * u1_lat * v1_lat

        u0_lon = u_lon[k][lat_i0]
        u1_lon = u_lon[k][lat_i1]
        v0_lon = vt_lon[k][lon_i0]
        v1_lon = vt_lon[k][lon_i1]
        dlon_00 += s_lon[k] * u0_lon * v0_lon
        dlon_01 += s_lon[k] * u0_lon * v1_lon
        dlon_10 += s_lon[k] * u1_lon * v0_lon
        dlon_11 += s_lon[k] * u1_lon * v1_lon

    w00 = (1 - lat_frac) * (1 - lon_frac)
    w01 = (1 - lat_frac) * lon_frac
    w10 = lat_frac * (1 - lon_frac)
    w11 = lat_frac * lon_frac

    svd_dlat = w00 * dlat_00 + w01 * dlat_01 + w10 * dlat_10 + w11 * dlat_11
    svd_dlon = w00 * dlon_00 + w01 * dlon_01 + w10 * dlon_10 + w11 * dlon_11

    pred_lat = pred_lat_base + svd_dlat / 3600.0
    pred_lon = pred_lon_base + svd_dlon / 3600.0

    # Error vs pyproj
    err_lat_m_test = (pred_lat - ref_lat) * 111320.0
    cos_test = np.cos(np.radians(test_lats))
    err_lon_m_test = (pred_lon - ref_lon) * 111320.0 * cos_test
    err_total_test = np.sqrt(err_lat_m_test**2 + err_lon_m_test**2)

    print(f"  Max error:  {err_total_test.max():.4f} m  ({err_total_test.max() * 100:.2f} cm)")
    print(
        f"  RMS error:  {np.sqrt(np.mean(err_total_test**2)):.4f} m  "
        f"({np.sqrt(np.mean(err_total_test**2)) * 100:.2f} cm)"
    )
    print(
        f"  P95 error:  {np.percentile(err_total_test, 95):.4f} m  "
        f"({np.percentile(err_total_test, 95) * 100:.2f} cm)"
    )
    print()

    if err_total_test.max() > args.target_accuracy:
        print(
            f"WARNING: Max error {err_total_test.max():.4f} m exceeds "
            f"target {args.target_accuracy} m."
        )
        print("Consider increasing --rank or --n-lat/--n-lon.")
        print()

    # Determine singular value energy retention
    total_energy_lat = np.sum(s_lat**2) / np.sum(np.linalg.svd(fit_lat, compute_uv=False) ** 2)
    total_energy_lon = np.sum(s_lon**2) / np.sum(np.linalg.svd(fit_lon, compute_uv=False) ** 2)
    print(f"SVD energy retention: lat={total_energy_lat:.6f}, lon={total_energy_lon:.6f}")

    # Attribution
    if "4267" in src_crs_str and "4269" in dst_crs_str:
        source = (
            "NADCON5 (NOAA/NGS). US government public domain. "
            "SVD-compressed from pyproj reference transforms."
        )
    else:
        source = f"SVD-compressed from pyproj reference transforms ({src_crs_str} -> {dst_crs_str})."

    # Output Python source
    print()
    print("=" * 80)
    print("# Paste the following into src/vibeproj/_datum_corrections.py")
    print("=" * 80)
    print()

    var_name = f"_NAD27_NAD83" if "4267" in src_crs_str else f"_CORRECTION_{src_crs_str.replace(':', '_')}_{dst_crs_str.replace(':', '_')}"

    print(f'{var_name} = DatumCorrectionData(')
    print(f'    src_crs="{src_crs_str}",')
    print(f'    dst_crs="{dst_crs_str}",')
    print(f"    bbox=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}),")
    print(f"    n_lat={n_lat},")
    print(f"    n_lon={n_lon},")
    print(f"    rank={rank},")
    print(f"    s_lat={_format_tuple(s_lat, indent=4).strip()},")
    print(f"    u_lat={_format_tuple_of_tuples(u_lat, indent=4).strip()},")
    print(f"    vt_lat={_format_tuple_of_tuples(vt_lat, indent=4).strip()},")
    print(f"    s_lon={_format_tuple(s_lon, indent=4).strip()},")
    print(f"    u_lon={_format_tuple_of_tuples(u_lon, indent=4).strip()},")
    print(f"    vt_lon={_format_tuple_of_tuples(vt_lon, indent=4).strip()},")
    print(f'    source="{source}",')
    print(f"    has_helmert={has_helmert},")
    print(")")
    print(f"_register({var_name})")


if __name__ == "__main__":
    main()
