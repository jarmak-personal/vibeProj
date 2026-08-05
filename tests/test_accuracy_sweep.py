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

from _accuracy_cases import EPSG_SWEEP, WGS84_A, finite_mask, make_lon_lat_grid
from vibeproj import Transformer
from vibeproj.crs import ProjectionParams
from vibeproj.ellipsoid import SPHERE
from vibeproj.pipeline import TransformPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# EPSG sweep registry
# ---------------------------------------------------------------------------


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
        ellipsoid=SPHERE,
        lon_0=lon_0,
        lat_0=lat_0,
        north_first=False,
        extra={},
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=SPHERE, north_first=True)
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
        ellipsoid=SPHERE,
        lon_0=lon_0,
        lat_0=lat_0,
        north_first=False,
        extra={},
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=SPHERE, north_first=True)
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
    lon, lat = make_lon_lat_grid(lat_range, lon_range, n=15)

    vp = Transformer.from_crs("EPSG:4326", crs_spec)  # always_xy=True default
    pp = PyProjTransformer.from_crs("EPSG:4326", crs_spec, always_xy=True)

    vp_x, vp_y = vp.transform(lon, lat)
    pp_x, pp_y = pp.transform(lon, lat)

    mask = finite_mask(vp_x, vp_y, pp_x, pp_y)
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
    lon, lat = make_lon_lat_grid(lat_range, lon_range, n=15)

    vp = Transformer.from_crs("EPSG:4326", crs_spec)  # always_xy=True default
    pp = PyProjTransformer.from_crs("EPSG:4326", crs_spec, always_xy=True)

    # Generate projected coords via pyproj forward
    pp_x, pp_y = pp.transform(lon, lat)

    mask_fwd = finite_mask(pp_x, pp_y)
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

    mask_inv = finite_mask(pp_inv_lon, pp_inv_lat, vp_inv_lon, vp_inv_lat)
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
    lon, lat = make_lon_lat_grid(lat_range, lon_range, n=15)

    vp_x, vp_y = _vp_proj_forward(proj_name, lon, lat, lat_0=lat_0, lon_0=lon_0)
    pp_x, pp_y = _pp_proj_forward(proj_name, lon, lat, lat_0=lat_0, lon_0=lon_0)

    mask = finite_mask(vp_x, vp_y, pp_x, pp_y)
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
    lon, lat = make_lon_lat_grid(lat_range, lon_range, n=15)

    # Generate projected coords via pyproj forward
    pp_x, pp_y = _pp_proj_forward(proj_name, lon, lat, lat_0=lat_0, lon_0=lon_0)

    mask_fwd = finite_mask(pp_x, pp_y)
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

    mask_inv = finite_mask(pp_inv_lon, pp_inv_lat, vp_inv_lon, vp_inv_lat)
    assert mask_inv.sum() > 0, f"No finite inverse points for {label}"

    err_lon = float(np.max(np.abs(vp_inv_lon[mask_inv] - pp_inv_lon[mask_inv])))
    err_lat = float(np.max(np.abs(vp_inv_lat[mask_inv] - pp_inv_lat[mask_inv])))
    max_err = max(err_lon, err_lat)
    assert max_err < inv_tol_deg, (
        f"{label} inverse error {max_err:.6e} deg exceeds {inv_tol_deg} deg "
        f"(lon={err_lon:.6e}, lat={err_lat:.6e})"
    )
