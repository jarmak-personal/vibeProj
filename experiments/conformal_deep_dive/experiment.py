"""
Conformal Mapping Deep Dive: Can complex polynomials replace NTv2 grids?

Experiment plan:
1. Sample pyproj's grid-based NAD27→NAD83 transform at a dense grid over CONUS
2. Sample our Helmert at the same points
3. Extract the residual (what the grid adds beyond Helmert)
4. Decompose residual into conformal (complex polynomial) + non-conformal parts
5. Measure accuracy vs degree, parameter count, and conformal fraction
"""

import numpy as np
import json
import sys
from pathlib import Path

# ── Phase 1: Sample the correction surfaces ─────────────────────────────

def sample_pyproj_nad27_nad83(n_lat=100, n_lon=150):
    """Sample pyproj grid-based NAD27→NAD83 over CONUS."""
    from pyproj import Transformer

    # CONUS bounding box (NAD27 coverage)
    lat_min, lat_max = 24.0, 50.0
    lon_min, lon_max = -125.0, -66.0

    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)
    grid_lon, grid_lat = np.meshgrid(lons, lats)  # shape (n_lat, n_lon)

    flat_lon = grid_lon.ravel()
    flat_lat = grid_lat.ravel()

    # pyproj transform (grid-based, high accuracy)
    t = Transformer.from_crs("EPSG:4267", "EPSG:4269", always_xy=True)
    out_lon, out_lat = t.transform(flat_lon, flat_lat)

    # Corrections in arcseconds (for interpretability)
    dlat_arcsec = (out_lat - flat_lat) * 3600.0
    dlon_arcsec = (out_lon - flat_lon) * 3600.0

    return {
        "lats": lats,
        "lons": lons,
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "dlat_arcsec": dlat_arcsec.reshape(n_lat, n_lon),
        "dlon_arcsec": dlon_arcsec.reshape(n_lat, n_lon),
        "dlat_deg": (out_lat - flat_lat).reshape(n_lat, n_lon),
        "dlon_deg": (out_lon - flat_lon).reshape(n_lat, n_lon),
    }


def sample_helmert_nad27_nad83(grid_lat, grid_lon):
    """Sample our 7-param Helmert at the same grid points."""
    # NAD27→NAD83 Helmert parameters (from EPSG, Position Vector convention)
    # These are approximate — the exact values come from pyproj pipeline parsing
    from pyproj import Transformer
    import re

    # Get the Helmert-only transform by querying pyproj for a parametric path
    # We'll use the well-known NAD27→NAD83 Helmert: dx=-8, dy=160, dz=176
    # (EPSG:1173, accuracy ~5m)
    tx, ty, tz = -8.0, 160.0, 176.0  # meters
    # No rotation or scale for this particular transform
    rx, ry, rz = 0.0, 0.0, 0.0
    ds = 1.0

    # Apply Helmert via ECEF
    a_src = 6378206.4    # Clarke 1866 semi-major
    f_src = 1 / 294.9786982
    a_dst = 6378137.0    # GRS80 semi-major
    f_dst = 1 / 298.257222101

    flat_lat = np.radians(grid_lat.ravel())
    flat_lon = np.radians(grid_lon.ravel())

    # Geodetic to ECEF (source ellipsoid)
    e2_src = 2 * f_src - f_src**2
    N_src = a_src / np.sqrt(1 - e2_src * np.sin(flat_lat)**2)
    X = N_src * np.cos(flat_lat) * np.cos(flat_lon)
    Y = N_src * np.cos(flat_lat) * np.sin(flat_lon)
    Z = N_src * (1 - e2_src) * np.sin(flat_lat)

    # Helmert transform
    X2 = ds * X + tx
    Y2 = ds * Y + ty
    Z2 = ds * Z + tz

    # ECEF to Geodetic (destination ellipsoid) — Bowring iterative
    e2_dst = 2 * f_dst - f_dst**2
    p = np.sqrt(X2**2 + Y2**2)
    lon_out = np.arctan2(Y2, X2)

    # Initial guess
    lat_out = np.arctan2(Z2, p * (1 - e2_dst))
    for _ in range(10):
        N_dst = a_dst / np.sqrt(1 - e2_dst * np.sin(lat_out)**2)
        lat_out = np.arctan2(Z2 + e2_dst * N_dst * np.sin(lat_out), p)

    out_lat = np.degrees(lat_out)
    out_lon = np.degrees(lon_out)
    in_lat = grid_lat.ravel()
    in_lon = grid_lon.ravel()

    dlat_arcsec = (out_lat - in_lat) * 3600.0
    dlon_arcsec = (out_lon - in_lon) * 3600.0

    n_lat, n_lon = grid_lat.shape
    return {
        "dlat_arcsec": dlat_arcsec.reshape(n_lat, n_lon),
        "dlon_arcsec": dlon_arcsec.reshape(n_lat, n_lon),
        "dlat_deg": (out_lat - in_lat).reshape(n_lat, n_lon),
        "dlon_deg": (out_lon - in_lon).reshape(n_lat, n_lon),
    }


# ── Phase 2: Complex polynomial fitting ─────────────────────────────────

def normalize_coords(lats, lons, lat_range, lon_range):
    """Normalize to [-1, 1] for numerical stability."""
    lat_mid = (lat_range[0] + lat_range[1]) / 2
    lon_mid = (lon_range[0] + lon_range[1]) / 2
    lat_scale = (lat_range[1] - lat_range[0]) / 2
    lon_scale = (lon_range[1] - lon_range[0]) / 2
    return (lats - lat_mid) / lat_scale, (lons - lon_mid) / lon_scale


def fit_conformal(z, w, degree):
    """Fit complex polynomial w(z) = sum c_k z^k via least-squares.

    z: complex array of normalized coordinates
    w: complex array of corrections (real=dlon, imag=dlat)
    Returns: complex coefficient array
    """
    V = np.column_stack([z**k for k in range(degree + 1)])
    coeffs, residuals, rank, sv = np.linalg.lstsq(V, w, rcond=None)
    return coeffs


def fit_general_polynomial(lat_n, lon_n, dlat, dlon, degree):
    """Fit separate real polynomials for lat and lon (non-conformal baseline).

    Total monomials for degree d in 2 variables: (d+1)(d+2)/2
    """
    monomials = []
    for total_deg in range(degree + 1):
        for i in range(total_deg + 1):
            j = total_deg - i
            monomials.append(lat_n**i * lon_n**j)

    V = np.column_stack(monomials)
    coeffs_lat, *_ = np.linalg.lstsq(V, dlat, rcond=None)
    coeffs_lon, *_ = np.linalg.lstsq(V, dlon, rcond=None)

    pred_lat = V @ coeffs_lat
    pred_lon = V @ coeffs_lon
    n_params = 2 * len(monomials)  # separate fits for lat and lon
    return pred_lat, pred_lon, n_params


def evaluate_conformal(z, coeffs):
    """Evaluate complex polynomial at z."""
    w = np.zeros_like(z)
    z_power = np.ones_like(z)
    for c in coeffs:
        w += c * z_power
        z_power *= z
    return w


# ── Phase 3: Analysis ────────────────────────────────────────────────────

def error_stats(true, pred):
    """Compute error statistics."""
    err = true - pred
    return {
        "max_abs": float(np.max(np.abs(err))),
        "rms": float(np.sqrt(np.mean(err**2))),
        "mean_abs": float(np.mean(np.abs(err))),
        "p95": float(np.percentile(np.abs(err), 95)),
        "p99": float(np.percentile(np.abs(err), 99)),
    }


def run_experiment():
    print("=" * 70)
    print("CONFORMAL MAPPING DEEP DIVE: NAD27 → NAD83 over CONUS")
    print("=" * 70)

    # ── Sample ───────────────────────────────────────────────────────
    print("\n[1/5] Sampling pyproj grid-based transform (100x150 = 15,000 points)...")
    pyproj_data = sample_pyproj_nad27_nad83(n_lat=100, n_lon=150)
    print(f"  dlat range: [{pyproj_data['dlat_arcsec'].min():.4f}, "
          f"{pyproj_data['dlat_arcsec'].max():.4f}] arcsec")
    print(f"  dlon range: [{pyproj_data['dlon_arcsec'].min():.4f}, "
          f"{pyproj_data['dlon_arcsec'].max():.4f}] arcsec")

    print("\n[2/5] Sampling Helmert transform at same points...")
    helmert_data = sample_helmert_nad27_nad83(
        pyproj_data["grid_lat"], pyproj_data["grid_lon"]
    )
    print(f"  dlat range: [{helmert_data['dlat_arcsec'].min():.4f}, "
          f"{helmert_data['dlat_arcsec'].max():.4f}] arcsec")
    print(f"  dlon range: [{helmert_data['dlon_arcsec'].min():.4f}, "
          f"{helmert_data['dlon_arcsec'].max():.4f}] arcsec")

    # Residual = pyproj - Helmert (what the grid adds)
    residual_dlat = pyproj_data["dlat_arcsec"] - helmert_data["dlat_arcsec"]
    residual_dlon = pyproj_data["dlon_arcsec"] - helmert_data["dlon_arcsec"]
    print(f"\n  RESIDUAL (grid correction beyond Helmert):")
    print(f"    dlat residual range: [{residual_dlat.min():.4f}, {residual_dlat.max():.4f}] arcsec")
    print(f"    dlon residual range: [{residual_dlon.min():.4f}, {residual_dlon.max():.4f}] arcsec")
    print(f"    dlat residual RMS:   {np.sqrt(np.mean(residual_dlat**2)):.4f} arcsec")
    print(f"    dlon residual RMS:   {np.sqrt(np.mean(residual_dlon**2)):.4f} arcsec")
    residual_meters = np.sqrt(residual_dlat**2 + residual_dlon**2) * 30.87  # ~30.87 m/arcsec at mid-lat
    print(f"    Residual magnitude:  RMS={np.sqrt(np.mean(residual_meters**2)):.2f} m, "
          f"max={residual_meters.max():.2f} m")

    # ── Normalize coordinates ────────────────────────────────────────
    lat_range = (24.0, 50.0)
    lon_range = (-125.0, -66.0)
    lat_n, lon_n = normalize_coords(
        pyproj_data["grid_lat"].ravel(),
        pyproj_data["grid_lon"].ravel(),
        lat_range, lon_range,
    )

    # Complex coordinate: z = lon_normalized + i * lat_normalized
    z = lon_n + 1j * lat_n

    # ── Phase 3: Fit conformal polynomials at increasing degree ──────
    print("\n[3/5] Fitting conformal (complex) polynomials to RESIDUAL...")
    print(f"  Fitting to residual after Helmert removal")
    print()

    # Target: residual in arcseconds
    # w = dlon + i*dlat (matching complex plane convention)
    w_residual = residual_dlon.ravel() + 1j * residual_dlat.ravel()
    # Also fit to FULL correction (no Helmert removal) for comparison
    w_full = pyproj_data["dlon_arcsec"].ravel() + 1j * pyproj_data["dlat_arcsec"].ravel()

    results = []
    max_degree = 30

    print(f"  {'Deg':>3} {'Params':>6} {'MaxErr(as)':>11} {'RMS(as)':>10} "
          f"{'P95(as)':>9} {'MaxErr(m)':>10} {'RMS(m)':>8} {'Conf%':>6}")
    print(f"  {'---':>3} {'------':>6} {'----------':>11} {'---------':>10} "
          f"{'--------':>9} {'---------':>10} {'------':>8} {'-----':>6}")

    for deg in list(range(1, 21)) + [25, 30]:
        n_params = 2 * (deg + 1)  # complex coefficients → real params
        coeffs = fit_conformal(z, w_residual, deg)
        w_pred = evaluate_conformal(z, coeffs)

        err_dlon = residual_dlon.ravel() - w_pred.real
        err_dlat = residual_dlat.ravel() - w_pred.imag

        # Combined error in arcseconds
        err_arcsec = np.sqrt(err_dlat**2 + err_dlon**2)
        # Convert to meters (~30.87 m per arcsec at 37°N midpoint)
        cos_mid = np.cos(np.radians(37.0))
        err_lat_m = err_dlat / 3600.0 * np.pi / 180.0 * 6371000
        err_lon_m = err_dlon / 3600.0 * np.pi / 180.0 * 6371000 * cos_mid
        err_m = np.sqrt(err_lat_m**2 + err_lon_m**2)

        # Conformal fraction: what % of total variance is captured?
        total_var = np.var(w_residual.real) + np.var(w_residual.imag)
        resid_var = np.var(err_dlon) + np.var(err_dlat)
        conf_pct = (1 - resid_var / total_var) * 100 if total_var > 0 else 0

        r = {
            "degree": deg,
            "n_params": n_params,
            "max_err_arcsec": float(np.max(err_arcsec)),
            "rms_arcsec": float(np.sqrt(np.mean(err_arcsec**2))),
            "p95_arcsec": float(np.percentile(err_arcsec, 95)),
            "max_err_m": float(np.max(err_m)),
            "rms_m": float(np.sqrt(np.mean(err_m**2))),
            "conformal_pct": conf_pct,
        }
        results.append(r)

        print(f"  {deg:3d} {n_params:6d} {r['max_err_arcsec']:11.6f} {r['rms_arcsec']:10.6f} "
              f"{r['p95_arcsec']:9.6f} {r['max_err_m']:10.4f} {r['rms_m']:8.4f} {conf_pct:5.1f}%")

    # ── Phase 4: Compare against non-conformal polynomial ────────────
    print("\n[4/5] Comparing conformal vs non-conformal polynomials...")
    print(f"\n  {'Type':>12} {'Deg':>3} {'Params':>6} {'MaxErr(m)':>10} {'RMS(m)':>8}")
    print(f"  {'----':>12} {'---':>3} {'------':>6} {'---------':>10} {'------':>8}")

    for deg in [5, 10, 15, 20]:
        # Conformal
        n_params_c = 2 * (deg + 1)
        coeffs_c = fit_conformal(z, w_residual, deg)
        w_pred_c = evaluate_conformal(z, coeffs_c)
        err_dlat_c = residual_dlat.ravel() - w_pred_c.imag
        err_dlon_c = residual_dlon.ravel() - w_pred_c.real
        cos_mid = np.cos(np.radians(37.0))
        err_m_c = np.sqrt(
            (err_dlat_c / 3600 * np.pi / 180 * 6371000)**2 +
            (err_dlon_c / 3600 * np.pi / 180 * 6371000 * cos_mid)**2
        )

        # Non-conformal (separate real polynomials)
        pred_lat_nc, pred_lon_nc, n_params_nc = fit_general_polynomial(
            lat_n, lon_n, residual_dlat.ravel(), residual_dlon.ravel(), deg
        )
        err_dlat_nc = residual_dlat.ravel() - pred_lat_nc
        err_dlon_nc = residual_dlon.ravel() - pred_lon_nc
        err_m_nc = np.sqrt(
            (err_dlat_nc / 3600 * np.pi / 180 * 6371000)**2 +
            (err_dlon_nc / 3600 * np.pi / 180 * 6371000 * cos_mid)**2
        )

        print(f"  {'conformal':>12} {deg:3d} {n_params_c:6d} {np.max(err_m_c):10.4f} "
              f"{np.sqrt(np.mean(err_m_c**2)):8.4f}")
        print(f"  {'general':>12} {deg:3d} {n_params_nc:6d} {np.max(err_m_nc):10.4f} "
              f"{np.sqrt(np.mean(err_m_nc**2)):8.4f}")

    # ── Phase 5: Conformal fraction analysis ─────────────────────────
    print("\n[5/5] Conformal decomposition analysis...")

    # Fit high-degree conformal to get the conformal component
    deg_high = 25
    coeffs_high = fit_conformal(z, w_residual, deg_high)
    w_conformal = evaluate_conformal(z, coeffs_high)

    # Non-conformal residual
    non_conformal_dlat = residual_dlat.ravel() - w_conformal.imag
    non_conformal_dlon = residual_dlon.ravel() - w_conformal.real

    total_energy_lat = np.sum(residual_dlat.ravel()**2)
    total_energy_lon = np.sum(residual_dlon.ravel()**2)
    conf_energy_lat = np.sum(w_conformal.imag**2)
    conf_energy_lon = np.sum(w_conformal.real**2)
    nonconf_energy_lat = np.sum(non_conformal_dlat**2)
    nonconf_energy_lon = np.sum(non_conformal_dlon**2)

    print(f"\n  At degree {deg_high}:")
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │ Component      │ dlat energy │ dlon energy  │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │ Total residual │ {total_energy_lat:11.2f} │ {total_energy_lon:11.2f}  │")
    print(f"  │ Conformal      │ {conf_energy_lat:11.2f} │ {conf_energy_lon:11.2f}  │")
    print(f"  │ Non-conformal  │ {nonconf_energy_lat:11.2f} │ {nonconf_energy_lon:11.2f}  │")
    print(f"  └─────────────────────────────────────────────┘")

    pct_conf = (1 - (nonconf_energy_lat + nonconf_energy_lon) /
                (total_energy_lat + total_energy_lon)) * 100
    print(f"\n  Conformal fraction of total residual energy: {pct_conf:.1f}%")

    cos_mid = np.cos(np.radians(37.0))
    nonconf_m = np.sqrt(
        (non_conformal_dlat / 3600 * np.pi / 180 * 6371000)**2 +
        (non_conformal_dlon / 3600 * np.pi / 180 * 6371000 * cos_mid)**2
    )
    print(f"  Non-conformal residual: RMS={np.sqrt(np.mean(nonconf_m**2)):.4f} m, "
          f"max={np.max(nonconf_m):.4f} m, P95={np.percentile(nonconf_m, 95):.4f} m")

    # ── Also try fitting to FULL correction (not just residual) ──────
    print("\n" + "=" * 70)
    print("BONUS: Conformal polynomial on FULL correction (no Helmert removal)")
    print("=" * 70)
    print(f"\n  {'Deg':>3} {'Params':>6} {'MaxErr(m)':>10} {'RMS(m)':>8} {'Conf%':>6}")
    print(f"  {'---':>3} {'------':>6} {'---------':>10} {'------':>8} {'-----':>6}")

    for deg in [5, 10, 15, 20, 25, 30]:
        n_params = 2 * (deg + 1)
        coeffs = fit_conformal(z, w_full, deg)
        w_pred = evaluate_conformal(z, coeffs)
        err_dlat = pyproj_data["dlat_arcsec"].ravel() - w_pred.imag
        err_dlon = pyproj_data["dlon_arcsec"].ravel() - w_pred.real
        err_m = np.sqrt(
            (err_dlat / 3600 * np.pi / 180 * 6371000)**2 +
            (err_dlon / 3600 * np.pi / 180 * 6371000 * cos_mid)**2
        )
        total_var = np.var(w_full.real) + np.var(w_full.imag)
        resid_var = np.var(err_dlon) + np.var(err_dlat)
        conf_pct = (1 - resid_var / total_var) * 100

        print(f"  {deg:3d} {n_params:6d} {np.max(err_m):10.4f} "
              f"{np.sqrt(np.mean(err_m**2)):8.4f} {conf_pct:5.1f}%")

    # ── SVD comparison ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SVD COMPRESSION COMPARISON (Tensor decomposition baseline)")
    print("=" * 70)

    for rank in [1, 3, 5, 10, 15, 20, 30]:
        U_lat, S_lat, Vt_lat = np.linalg.svd(residual_dlat, full_matrices=False)
        U_lon, S_lon, Vt_lon = np.linalg.svd(residual_dlon, full_matrices=False)

        recon_lat = (U_lat[:, :rank] * S_lat[:rank]) @ Vt_lat[:rank, :]
        recon_lon = (U_lon[:, :rank] * S_lon[:rank]) @ Vt_lon[:rank, :]

        err_dlat_svd = residual_dlat - recon_lat
        err_dlon_svd = residual_dlon - recon_lon
        err_m_svd = np.sqrt(
            (err_dlat_svd.ravel() / 3600 * np.pi / 180 * 6371000)**2 +
            (err_dlon_svd.ravel() / 3600 * np.pi / 180 * 6371000 * cos_mid)**2
        )

        n_lat, n_lon = residual_dlat.shape
        n_params_svd = 2 * rank * (n_lat + n_lon + 1)  # 2 components × rank × (U + V + sigma)

        print(f"  rank={rank:2d}  params={n_params_svd:5d}  "
              f"MaxErr={np.max(err_m_svd):.4f} m  RMS={np.sqrt(np.mean(err_m_svd**2)):.4f} m")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find degree needed for sub-cm
    for r in results:
        if r["max_err_m"] < 0.01:
            print(f"\n  Sub-centimeter (max < 1cm) achieved at degree {r['degree']} "
                  f"with {r['n_params']} real parameters")
            print(f"    Max error: {r['max_err_m']*100:.2f} cm")
            print(f"    RMS error: {r['rms_m']*100:.2f} cm")
            break
    else:
        # Find best achieved
        best = min(results, key=lambda r: r["max_err_m"])
        print(f"\n  Sub-centimeter NOT achieved up to degree {max_degree}.")
        print(f"  Best: degree {best['degree']}, max error {best['max_err_m']*100:.2f} cm, "
              f"RMS {best['rms_m']*100:.2f} cm, {best['n_params']} params")

    # Save results
    output = {
        "conformal_results": results,
        "residual_stats": {
            "dlat_rms_arcsec": float(np.sqrt(np.mean(residual_dlat**2))),
            "dlon_rms_arcsec": float(np.sqrt(np.mean(residual_dlon**2))),
            "conformal_fraction_pct": pct_conf,
        },
    }
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
