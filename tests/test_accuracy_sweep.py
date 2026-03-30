"""Comprehensive accuracy sweep: every projection validated against pyproj.

Independent one-way forward AND inverse checks across both hemispheres
and parameter regimes.  Each test case is parametrized with curated
grid bounds and tolerance budgets.

Forward checks compare vibeProj output (meters) against pyproj via
Euclidean distance.  Inverse checks start from pyproj forward output,
then compare vibeProj inverse against pyproj inverse (degrees) so that
the inverse test is independent of vibeProj forward correctness.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyproj import Proj
from pyproj import Transformer as PyProjTransformer

from vibeproj import Transformer
from vibeproj.crs import ProjectionParams
from vibeproj.ellipsoid import WGS84
from vibeproj.pipeline import TransformPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WGS84_A = 6378137.0


def _make_grid(lat_range, lon_range, n=15):
    """Create an n x n grid and return (lon, lat) arrays in always_xy order."""
    lat = np.linspace(lat_range[0], lat_range[1], n)
    lon = np.linspace(lon_range[0], lon_range[1], n)
    lon_g, lat_g = np.meshgrid(lon, lat)
    return lon_g.ravel(), lat_g.ravel()


def _finite_mask(*arrays):
    """Return a boolean mask where all arrays are finite."""
    mask = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


# ---------------------------------------------------------------------------
# EPSG sweep registry
# ---------------------------------------------------------------------------

EPSG_SWEEP = [
    # (label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg)
    #
    # Ranges pushed to CRS area-of-use edges where projection math is most
    # stressed.  Tolerances calibrated against pyproj on 15x15 grids.
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
    # Avoid exact poles (lon undefined at ±90°; pyproj and vibeproj may
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
]


# ---------------------------------------------------------------------------
# +proj sweep registry (spherical projections)
# ---------------------------------------------------------------------------

PROJ_SWEEP = [
    # (label, proj_name, lat_0, lon_0, lat_range, lon_range, fwd_tol_m, inv_tol_deg)
    #
    # Ranges pushed toward visible-hemisphere edges (ortho/gnom) and high
    # latitudes (pseudocylindricals).
    ("ortho_nh", "ortho", 45, 0, (-40, 85), (-80, 80), 0.01, 1e-8),
    ("ortho_sh", "ortho", -45, 0, (-85, 40), (-80, 80), 0.01, 1e-8),
    ("gnom_nh", "gnom", 45, 0, (10, 75), (-30, 30), 0.01, 1e-8),
    ("gnom_sh", "gnom", -45, 0, (-75, -10), (-30, 30), 0.01, 1e-8),
    ("aeqd_nh", "aeqd", 45, 0, (-40, 85), (-80, 80), 0.01, 1e-8),
    ("aeqd_sh", "aeqd", -45, 0, (-85, 40), (-80, 80), 0.01, 1e-8),
    ("sinu", "sinu", 0, 0, (-85, 85), (-179, 179), 0.01, 1e-9),
    ("moll", "moll", 0, 0, (-85, 85), (-179, 179), 0.01, 1e-9),
    ("eck4", "eck4", 0, 0, (-85, 85), (-179, 179), 0.01, 1e-9),
    ("eck6", "eck6", 0, 0, (-85, 85), (-179, 179), 0.01, 1e-9),
    # robin: cubic polynomial interpolation matching PROJ (sub-meter forward).
    # Inverse vs pyproj limited by PROJ coefficient truncation (~1e-5 deg).
    ("robin", "robin", 0, 0, (-85, 85), (-179, 179), 1.0, 1e-4),
    # wintri: proper Newton-Raphson with analytical Jacobian.
    ("wintri", "wintri", 0, 0, (-85, 85), (-179, 179), 0.01, 1e-8),
    ("natearth", "natearth", 0, 0, (-85, 85), (-179, 179), 0.01, 1e-9),
]


# ---------------------------------------------------------------------------
# vibeProj helper: +proj-based forward/inverse via TransformPipeline
# ---------------------------------------------------------------------------


def _vp_proj_forward(proj_name, lon, lat, lat_0=0.0, lon_0=0.0):
    """vibeProj forward for spherical +proj projections.

    Input/output in always_xy order: (lon, lat) -> (x, y).
    """
    params = ProjectionParams(
        projection_name=proj_name,
        ellipsoid=WGS84,
        lon_0=lon_0,
        lat_0=lat_0,
        north_first=False,
        extra={},
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    pipe = TransformPipeline(src, params)
    # TransformPipeline._forward expects (lat, lon) when src is north_first=True.
    # Our src is north_first=True, so arg1=lat, arg2=lon.
    x, y = pipe.transform(lat, lon, np)
    return x, y


def _vp_proj_inverse(proj_name, x, y, lat_0=0.0, lon_0=0.0):
    """vibeProj inverse for spherical +proj projections.

    Input (x, y) -> output (lon, lat) in always_xy order.
    """
    params = ProjectionParams(
        projection_name=proj_name,
        ellipsoid=WGS84,
        lon_0=lon_0,
        lat_0=lat_0,
        north_first=False,
        extra={},
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    inv_pipe = TransformPipeline(params, src)
    # TransformPipeline._inverse returns (lon, lat) when dst north_first=True,
    # but dst here is src (longlat, north_first=True), so output is (lat, lon).
    lat, lon = inv_pipe.transform(x, y, np)
    return lon, lat


def _pp_proj_forward(proj_name, lon, lat, lat_0=0.0, lon_0=0.0):
    """pyproj forward for spherical +proj projections.

    pyproj.Proj takes (lon, lat) and returns (x, y).
    """
    proj_str = f"+proj={proj_name} +lon_0={lon_0} +lat_0={lat_0} +R={WGS84_A} +units=m"
    p = Proj(proj_str)
    return p(lon, lat)


def _pp_proj_inverse(proj_name, x, y, lat_0=0.0, lon_0=0.0):
    """pyproj inverse for spherical +proj projections.

    Returns (lon, lat).
    """
    proj_str = f"+proj={proj_name} +lon_0={lon_0} +lat_0={lat_0} +R={WGS84_A} +units=m"
    p = Proj(proj_str)
    return p(x, y, inverse=True)


# ---------------------------------------------------------------------------
# EPSG tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg",
    EPSG_SWEEP,
    ids=[e[0] for e in EPSG_SWEEP],
)
def test_forward_vs_pyproj_epsg(label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg):
    """Forward: vibeproj vs pyproj on an n x n grid (EPSG-based)."""
    lon, lat = _make_grid(lat_range, lon_range, n=15)

    vp = Transformer.from_crs("EPSG:4326", crs_spec)  # always_xy=True default
    pp = PyProjTransformer.from_crs("EPSG:4326", crs_spec, always_xy=True)

    vp_x, vp_y = vp.transform(lon, lat)
    pp_x, pp_y = pp.transform(lon, lat)

    mask = _finite_mask(vp_x, vp_y, pp_x, pp_y)
    assert mask.sum() > 0, f"No finite points for {label}"

    err = np.sqrt((vp_x[mask] - pp_x[mask]) ** 2 + (vp_y[mask] - pp_y[mask]) ** 2)
    max_err = float(np.max(err))
    assert max_err < fwd_tol_m, f"{label} forward error {max_err:.6e} m exceeds {fwd_tol_m} m"


@pytest.mark.parametrize(
    "label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg",
    EPSG_SWEEP,
    ids=[e[0] for e in EPSG_SWEEP],
)
def test_inverse_vs_pyproj_epsg(label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg):
    """Inverse: use pyproj forward as input, compare vibeproj inverse vs pyproj inverse."""
    lon, lat = _make_grid(lat_range, lon_range, n=15)

    vp = Transformer.from_crs("EPSG:4326", crs_spec)  # always_xy=True default
    pp = PyProjTransformer.from_crs("EPSG:4326", crs_spec, always_xy=True)

    # Generate projected coords via pyproj forward
    pp_x, pp_y = pp.transform(lon, lat)

    mask_fwd = _finite_mask(pp_x, pp_y)
    if mask_fwd.sum() == 0:
        pytest.skip(f"No finite forward points for {label}")

    pp_x_valid = pp_x[mask_fwd]
    pp_y_valid = pp_y[mask_fwd]

    # Inverse via both libraries
    from pyproj.enums import TransformDirection

    pp_inv_lon, pp_inv_lat = pp.transform(
        pp_x_valid, pp_y_valid, direction=TransformDirection.INVERSE
    )
    vp_inv_lon, vp_inv_lat = vp.transform(pp_x_valid, pp_y_valid, direction="INVERSE")

    mask_inv = _finite_mask(pp_inv_lon, pp_inv_lat, vp_inv_lon, vp_inv_lat)
    assert mask_inv.sum() > 0, f"No finite inverse points for {label}"

    err_lon = float(np.max(np.abs(vp_inv_lon[mask_inv] - pp_inv_lon[mask_inv])))
    err_lat = float(np.max(np.abs(vp_inv_lat[mask_inv] - pp_inv_lat[mask_inv])))
    max_err = max(err_lon, err_lat)
    assert max_err < inv_tol_deg, (
        f"{label} inverse error {max_err:.6e} deg exceeds {inv_tol_deg} deg "
        f"(lon={err_lon:.6e}, lat={err_lat:.6e})"
    )


# ---------------------------------------------------------------------------
# +proj tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, proj_name, lat_0, lon_0, lat_range, lon_range, fwd_tol_m, inv_tol_deg",
    PROJ_SWEEP,
    ids=[e[0] for e in PROJ_SWEEP],
)
def test_forward_vs_pyproj_proj(
    label, proj_name, lat_0, lon_0, lat_range, lon_range, fwd_tol_m, inv_tol_deg
):
    """Forward: vibeproj vs pyproj on an n x n grid (+proj-based spherical)."""
    lon, lat = _make_grid(lat_range, lon_range, n=15)

    vp_x, vp_y = _vp_proj_forward(proj_name, lon, lat, lat_0=lat_0, lon_0=lon_0)
    pp_x, pp_y = _pp_proj_forward(proj_name, lon, lat, lat_0=lat_0, lon_0=lon_0)

    mask = _finite_mask(vp_x, vp_y, pp_x, pp_y)
    assert mask.sum() > 0, f"No finite points for {label}"

    err = np.sqrt((vp_x[mask] - pp_x[mask]) ** 2 + (vp_y[mask] - pp_y[mask]) ** 2)
    max_err = float(np.max(err))
    assert max_err < fwd_tol_m, f"{label} forward error {max_err:.6e} m exceeds {fwd_tol_m} m"


@pytest.mark.parametrize(
    "label, proj_name, lat_0, lon_0, lat_range, lon_range, fwd_tol_m, inv_tol_deg",
    PROJ_SWEEP,
    ids=[e[0] for e in PROJ_SWEEP],
)
def test_inverse_vs_pyproj_proj(
    label, proj_name, lat_0, lon_0, lat_range, lon_range, fwd_tol_m, inv_tol_deg
):
    """Inverse: use pyproj forward as input, compare vibeproj inverse vs pyproj inverse."""
    lon, lat = _make_grid(lat_range, lon_range, n=15)

    # Generate projected coords via pyproj forward
    pp_x, pp_y = _pp_proj_forward(proj_name, lon, lat, lat_0=lat_0, lon_0=lon_0)

    mask_fwd = _finite_mask(pp_x, pp_y)
    if mask_fwd.sum() == 0:
        pytest.skip(f"No finite forward points for {label}")

    pp_x_valid = pp_x[mask_fwd]
    pp_y_valid = pp_y[mask_fwd]

    # Inverse via both libraries
    pp_inv_lon, pp_inv_lat = _pp_proj_inverse(
        proj_name, pp_x_valid, pp_y_valid, lat_0=lat_0, lon_0=lon_0
    )
    vp_inv_lon, vp_inv_lat = _vp_proj_inverse(
        proj_name, pp_x_valid, pp_y_valid, lat_0=lat_0, lon_0=lon_0
    )

    mask_inv = _finite_mask(pp_inv_lon, pp_inv_lat, vp_inv_lon, vp_inv_lat)
    assert mask_inv.sum() > 0, f"No finite inverse points for {label}"

    err_lon = float(np.max(np.abs(vp_inv_lon[mask_inv] - pp_inv_lon[mask_inv])))
    err_lat = float(np.max(np.abs(vp_inv_lat[mask_inv] - pp_inv_lat[mask_inv])))
    max_err = max(err_lon, err_lat)
    assert max_err < inv_tol_deg, (
        f"{label} inverse error {max_err:.6e} deg exceeds {inv_tol_deg} deg "
        f"(lon={err_lon:.6e}, lat={err_lat:.6e})"
    )
