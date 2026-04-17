"""Shared accuracy test registries and helpers."""

from __future__ import annotations

import numpy as np

WGS84_A = 6378137.0


def make_lon_lat_grid(lat_range, lon_range, n=15):
    """Create an n x n grid and return (lon, lat) arrays in always_xy order."""
    lat = np.linspace(lat_range[0], lat_range[1], n)
    lon = np.linspace(lon_range[0], lon_range[1], n)
    lon_g, lat_g = np.meshgrid(lon, lat)
    return lon_g.ravel(), lat_g.ravel()


def finite_mask(*arrays):
    """Return a boolean mask where all arrays are finite."""
    mask = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


EPSG_SWEEP = [
    # (label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg)
    #
    # Ranges pushed to CRS area-of-use edges where projection math is most
    # stressed. Tolerances are the accepted budgets already used elsewhere
    # in the repo for these projection families.
    #
    # --- LCC (sign-sensitive n) ---
    ("lcc_nh_france", "EPSG:2154", (41.2, 51.5), (-9.8, 10.3), 0.01, 1e-9),
    ("lcc_sh_nz", "EPSG:3851", (-55, -26), (161, 179), 0.01, 1e-9),
    ("lcc_sh_australia", "EPSG:3112", (-44, -10), (112, 154), 0.01, 1e-9),
    #
    # --- AEA (sign-sensitive n) ---
    ("aea_nh_conus", "EPSG:5070", (24.5, 49.5), (-124.5, -66.5), 0.01, 1e-9),
    ("aea_sh_australia", "EPSG:3577", (-44, -10), (112, 154), 0.01, 1e-9),
    #
    # --- Polar Stereographic (sign flag) ---
    # Avoid exact poles (lon undefined at +/-90°; pyproj and vibeproj may
    # return different arbitrary longitudes, causing a spurious 180° diff).
    ("stere_south", "EPSG:3031", (-89.9, -60), (-180, 180), 0.1, 1e-8),
    ("stere_north", "EPSG:3413", (60, 89.9), (-180, 180), 0.1, 1e-8),
    #
    # --- LAEA (oblique) ---
    ("laea_oblique_nh", "EPSG:3035", (33, 73), (-32, 47), 0.01, 1e-7),
    #
    # --- Transverse Mercator (hemisphere) — full zone extent ---
    ("tmerc_nh", "EPSG:32631", (0, 84), (0, 6), 1e-3, 1e-9),
    ("tmerc_sh", "EPSG:32756", (-80, 0), (150, 156), 1e-3, 1e-9),
    ("tmerc_equatorial", "EPSG:32648", (-8, 8), (102, 108), 1e-3, 1e-9),
    #
    # --- Web Mercator ---
    ("webmerc", "EPSG:3857", (-85, 85), (-179, 179), 0.01, 1e-9),
    #
    # --- Mercator ---
    ("merc", "EPSG:3395", (-80, 80), (-179, 179), 0.01, 1e-9),
    #
    # --- Equidistant Cylindrical ---
    ("eqc", "EPSG:4087", (-80, 80), (-179, 179), 0.01, 1e-9),
    #
    # --- Equal Earth ---
    ("eqearth_greenwich", "EPSG:8857", (-85, 85), (-179, 179), 0.01, 1e-7),
    #
    # --- Cylindrical Equal Area ---
    ("cea_ease", "EPSG:6933", (-85, 85), (-179, 179), 0.1, 1e-7),
    #
    # --- Oblique Mercator ---
    ("omerc_malaysia", "EPSG:3168", (1, 7), (99.5, 104.5), 0.01, 1e-9),
    #
    # --- Oblique Stereographic (cross-datum) ---
    ("sterea_rd", "EPSG:28992", (50.7, 53.7), (3.2, 7.3), 10.0, 1e-4),
    #
    # --- Krovak (cross-datum) ---
    ("krovak_cz", "EPSG:5514", (48.5, 51.1), (12, 18.9), 17.0, 2e-4),
    #
    # --- Non-meter linear units (unit-normalization canaries) ---
    # ftUS
    ("lcc_ftus_ny_long_island", "EPSG:2263", (40.47, 41.3), (-74.26, -71.8), 0.01, 1e-9),
    ("tmerc_ftus_il_east", "EPSG:3435", (37.06, 42.5), (-89.27, -87.02), 1e-3, 1e-9),
    # international foot
    ("tmerc_ft_arizona_east", "EPSG:2222", (31.33, 37.01), (-111.71, -109.04), 1e-3, 1e-9),
    ("lcc_ft_michigan_north", "EPSG:2251", (45.08, 48.32), (-90.42, -83.44), 0.01, 1e-9),
]


UNIT_EQUIVALENT_PAIRS = [
    # (label, src_crs, dst_crs, lat_range, lon_range, tol_m)
    # Same area, same datum, same projection family, different projected units.
    # These are direct canaries for linear-unit normalization bugs.
    ("ny_long_island_ftus_to_m", "EPSG:2263", "EPSG:32118", (40.47, 41.3), (-74.26, -71.8), 0.01),
    ("illinois_east_ftus_to_m", "EPSG:3435", "EPSG:26971", (37.06, 42.5), (-89.27, -87.02), 0.01),
    ("arizona_east_ft_to_m", "EPSG:2222", "EPSG:26948", (31.33, 37.01), (-111.71, -109.04), 0.01),
    ("michigan_north_ft_to_m", "EPSG:2251", "EPSG:26988", (45.08, 48.32), (-90.42, -83.44), 0.01),
]
