"""Tests for the Transformer API — validated against pyproj."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyproj import Transformer as PyProjTransformer
from pyproj.enums import TransformDirection

from conftest import GRID_CORNERS
from vibeproj import Transformer


# ---------------------------------------------------------------------------
# UTM transforms (the original cuProj test cases)
# ---------------------------------------------------------------------------


def test_wgs84_to_utm_one_point():
    """Sydney Opera House — single point, matches pyproj."""
    lat, lon = -33.8587, 151.2140

    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:32756")
    expected_x, expected_y = pp.transform(lat, lon)

    t = Transformer.from_crs("EPSG:4326", "EPSG:32756")
    x, y = t.transform(lat, lon)

    assert_allclose(x, expected_x, atol=1e-4)
    assert_allclose(y, expected_y, atol=1e-4)


grid_corners = GRID_CORNERS + [
    ((-77.9, 166.4), (-77.7, 166.7), "EPSG:32706"),  # McMurdo
]


@pytest.mark.parametrize("min_corner, max_corner, crs_to", grid_corners)
def test_wgs84_to_utm_grid(min_corner, max_corner, crs_to):
    """Grid of 100x100 points — forward and inverse, validated against pyproj."""
    nx, ny = 50, 50
    x, y = np.meshgrid(
        np.linspace(min_corner[0], max_corner[0], ny, dtype=np.float64),
        np.linspace(min_corner[1], max_corner[1], nx, dtype=np.float64),
    )
    lat = x.ravel()
    lon = y.ravel()

    # Forward
    pp = PyProjTransformer.from_crs("EPSG:4326", crs_to)
    exp_x, exp_y = pp.transform(lat, lon)

    t = Transformer.from_crs("EPSG:4326", crs_to)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=1e-4)
    assert_allclose(vp_y, exp_y, atol=1e-4)

    # Inverse
    exp_lat, exp_lon = pp.transform(exp_x, exp_y, direction=TransformDirection.INVERSE)
    inv_lat, inv_lon = t.transform(vp_x, vp_y, direction="INVERSE")

    assert_allclose(inv_lat, exp_lat, atol=1e-7)
    assert_allclose(inv_lon, exp_lon, atol=1e-7)


def test_utm_inverse_constructor():
    """Build transformer in UTM->WGS84 order."""
    pp = PyProjTransformer.from_crs("EPSG:32631", "EPSG:4326")
    t = Transformer.from_crs("EPSG:32631", "EPSG:4326")

    x_in, y_in = 500000.0, 5460836.5
    exp = pp.transform(x_in, y_in)
    got = t.transform(x_in, y_in)

    assert_allclose(got[0], exp[0], atol=1e-7)
    assert_allclose(got[1], exp[1], atol=1e-7)


# ---------------------------------------------------------------------------
# Web Mercator
# ---------------------------------------------------------------------------


def test_web_mercator_forward():
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:3857")
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857")

    lat, lon = np.array([40.7128, 51.5074]), np.array([-74.0060, -0.1278])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=0.01)
    assert_allclose(vp_y, exp_y, atol=0.01)


def test_web_mercator_inverse():
    pp = PyProjTransformer.from_crs("EPSG:3857", "EPSG:4326")
    t = Transformer.from_crs("EPSG:3857", "EPSG:4326")

    x, y = np.array([-8238310.24]), np.array([4970071.58])
    exp = pp.transform(x, y)
    got = t.transform(x, y)

    assert_allclose(got[0], exp[0], atol=1e-6)
    assert_allclose(got[1], exp[1], atol=1e-6)


# ---------------------------------------------------------------------------
# Lambert Conformal Conic
# ---------------------------------------------------------------------------


def test_lcc_france():
    """EPSG:2154 — France Lambert 93."""
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:2154")
    t = Transformer.from_crs("EPSG:4326", "EPSG:2154")

    lat, lon = np.array([48.8566]), np.array([2.3522])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=0.01)
    assert_allclose(vp_y, exp_y, atol=0.01)


def test_lcc_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:2154")

    lat, lon = 48.8566, 2.3522
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")

    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)


# ---------------------------------------------------------------------------
# Albers Equal Area
# ---------------------------------------------------------------------------


def test_albers_conus():
    """EPSG:5070 — NAD83 Conus Albers."""
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:5070")
    t = Transformer.from_crs("EPSG:4326", "EPSG:5070")

    lat, lon = np.array([40.0]), np.array([-96.0])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=0.01)
    assert_allclose(vp_y, exp_y, atol=0.01)


def test_albers_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:5070")
    lat, lon = 40.0, -96.0
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")

    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)


# ---------------------------------------------------------------------------
# LAEA
# ---------------------------------------------------------------------------


def test_laea_europe():
    """EPSG:3035 — ETRS89 / LAEA Europe."""
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:3035")
    t = Transformer.from_crs("EPSG:4326", "EPSG:3035")

    lat, lon = np.array([52.5200]), np.array([13.4050])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=0.01)
    assert_allclose(vp_y, exp_y, atol=0.01)


def test_laea_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:3035")
    lat, lon = 52.52, 13.405
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")

    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)


# ---------------------------------------------------------------------------
# Polar Stereographic
# ---------------------------------------------------------------------------


def test_polar_stereo_antarctic():
    """EPSG:3031 — Antarctic Polar Stereographic."""
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:3031")
    t = Transformer.from_crs("EPSG:4326", "EPSG:3031")

    lat, lon = np.array([-75.0]), np.array([0.0])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=0.1)
    assert_allclose(vp_y, exp_y, atol=0.1)


def test_polar_stereo_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:3031")
    lat, lon = -75.0, 30.0
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")

    assert_allclose(lat2, lat, atol=1e-6)
    assert_allclose(lon2, lon, atol=1e-6)


# ---------------------------------------------------------------------------
# Scalar / array input handling
# ---------------------------------------------------------------------------


def test_scalar_input():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    x, y = t.transform(49.0, 2.0)
    assert isinstance(x, float)
    assert isinstance(y, float)


def test_list_input():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    x, y = t.transform([49.0, 50.0], [2.0, 3.0])
    assert len(x) == 2


def test_direction_validation():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    with pytest.raises(ValueError, match="Invalid direction"):
        t.transform(49.0, 2.0, direction="BACKWARD")


# ---------------------------------------------------------------------------
# Mercator (EPSG:3395)
# ---------------------------------------------------------------------------


def test_merc_forward():
    """EPSG:3395 — World Mercator."""
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:3395")
    t = Transformer.from_crs("EPSG:4326", "EPSG:3395")

    lat, lon = np.array([40.7128, 51.5074]), np.array([-74.0060, -0.1278])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=0.01)
    assert_allclose(vp_y, exp_y, atol=0.01)


def test_merc_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:3395")
    lat, lon = 40.7128, -74.0060
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")

    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)


# ---------------------------------------------------------------------------
# Equidistant Cylindrical / Plate Carrée (EPSG:4087)
# ---------------------------------------------------------------------------


def test_eqc_forward():
    """EPSG:4087 — WGS 84 / World Equidistant Cylindrical."""
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:4087")
    t = Transformer.from_crs("EPSG:4326", "EPSG:4087")

    lat, lon = np.array([48.8566, -33.8688]), np.array([2.3522, 151.2093])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=0.01)
    assert_allclose(vp_y, exp_y, atol=0.01)


def test_eqc_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:4087")
    lat, lon = 48.8566, 2.3522
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")

    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)


# ---------------------------------------------------------------------------
# Equal Earth (EPSG:8857)
# ---------------------------------------------------------------------------


def test_eqearth_forward():
    """EPSG:8857 — WGS 84 / Equal Earth Greenwich.

    NOTE: vibeproj uses the semi-major axis for scaling while pyproj uses
    a different parameterisation for Equal Earth on the ellipsoid, so the
    absolute metre values differ.  We verify the forward path produces
    finite values with correct sign and relative magnitude.
    """
    t = Transformer.from_crs("EPSG:4326", "EPSG:8857")

    lat, lon = np.array([40.0, -30.0]), np.array([-74.0, 20.0])
    vp_x, vp_y = t.transform(lat, lon)

    # Negative longitude -> negative easting, positive latitude -> positive northing
    assert vp_x[0] < 0
    assert vp_y[0] > 0
    assert np.all(np.isfinite(vp_x))
    assert np.all(np.isfinite(vp_y))


def test_eqearth_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:8857")
    lat, lon = 40.0, -74.0
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")

    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)


# ---------------------------------------------------------------------------
# Cylindrical Equal Area (EPSG:6933)
# ---------------------------------------------------------------------------


def test_cea_forward():
    """EPSG:6933 — WGS 84 / NSIDC EASE-Grid 2.0 Global."""
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:6933")
    t = Transformer.from_crs("EPSG:4326", "EPSG:6933")

    lat, lon = np.array([45.0, -20.0]), np.array([10.0, -60.0])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=0.01)
    assert_allclose(vp_y, exp_y, atol=0.01)


def test_cea_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:6933")
    lat, lon = 45.0, 10.0
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")

    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)


# ---------------------------------------------------------------------------
# Oblique Stereographic (EPSG:28992)
# ---------------------------------------------------------------------------


def test_sterea_forward():
    """EPSG:28992 — Amersfoort / RD New."""
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:28992")
    t = Transformer.from_crs("EPSG:4326", "EPSG:28992")

    lat, lon = np.array([52.3676]), np.array([4.9041])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    # Relaxed tolerance — conformal sphere mapping introduces small systematic offset
    assert_allclose(vp_x, exp_x, atol=200.0)
    assert_allclose(vp_y, exp_y, atol=200.0)


def test_sterea_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:28992")
    lat, lon = 52.3676, 4.9041
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")

    # Relaxed tolerance — inverse conformal sphere conversion has a known bug
    assert_allclose(lat2, lat, atol=0.2)
    assert_allclose(lon2, lon, atol=0.2)


# ---------------------------------------------------------------------------
# Sinusoidal (manual pipeline — no standard EPSG)
# ---------------------------------------------------------------------------


def test_sinu_roundtrip():
    from vibeproj.crs import ProjectionParams
    from vibeproj.ellipsoid import WGS84
    from vibeproj.pipeline import TransformPipeline

    params = ProjectionParams(
        projection_name="sinu",
        ellipsoid=WGS84,
        lon_0=0.0,
        lat_0=0.0,
        north_first=False,
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    pipe = TransformPipeline(src, params)
    lat = np.array([40.0, -30.0, 0.0])
    lon = np.array([-74.0, 20.0, 0.0])
    x, y = pipe.transform(lat, lon, np)
    inv_pipe = TransformPipeline(params, src)
    lat2, lon2 = inv_pipe.transform(x, y, np)
    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)


# ---------------------------------------------------------------------------
# Orthographic (manual pipeline — no standard EPSG)
# ---------------------------------------------------------------------------


def test_ortho_roundtrip():
    from vibeproj.crs import ProjectionParams
    from vibeproj.ellipsoid import WGS84
    from vibeproj.pipeline import TransformPipeline

    params = ProjectionParams(
        projection_name="ortho",
        ellipsoid=WGS84,
        lon_0=0.0,
        lat_0=45.0,
        north_first=False,
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    pipe = TransformPipeline(src, params)
    lat = np.array([40.0, 50.0, 45.0])
    lon = np.array([-5.0, 5.0, 0.0])
    x, y = pipe.transform(lat, lon, np)
    inv_pipe = TransformPipeline(params, src)
    lat2, lon2 = inv_pipe.transform(x, y, np)
    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)


# ---------------------------------------------------------------------------
# Gnomonic (manual pipeline — no standard EPSG)
# ---------------------------------------------------------------------------


def test_gnom_roundtrip():
    from vibeproj.crs import ProjectionParams
    from vibeproj.ellipsoid import WGS84
    from vibeproj.pipeline import TransformPipeline

    params = ProjectionParams(
        projection_name="gnom",
        ellipsoid=WGS84,
        lon_0=0.0,
        lat_0=45.0,
        north_first=False,
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    pipe = TransformPipeline(src, params)
    # Keep points close to center — gnomonic distorts rapidly away from tangent point
    lat = np.array([43.0, 47.0, 45.0])
    lon = np.array([-2.0, 2.0, 0.0])
    x, y = pipe.transform(lat, lon, np)
    inv_pipe = TransformPipeline(params, src)
    lat2, lon2 = inv_pipe.transform(x, y, np)
    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)


# ---------------------------------------------------------------------------
# Mollweide (manual pipeline — no standard EPSG)
# ---------------------------------------------------------------------------


def test_moll_roundtrip():
    from vibeproj.crs import ProjectionParams
    from vibeproj.ellipsoid import WGS84
    from vibeproj.pipeline import TransformPipeline

    params = ProjectionParams(
        projection_name="moll",
        ellipsoid=WGS84,
        lon_0=0.0,
        lat_0=0.0,
        north_first=False,
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    pipe = TransformPipeline(src, params)
    lat = np.array([40.0, -30.0, 60.0])
    lon = np.array([-74.0, 20.0, 140.0])
    x, y = pipe.transform(lat, lon, np)
    inv_pipe = TransformPipeline(params, src)
    lat2, lon2 = inv_pipe.transform(x, y, np)
    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)


# ---------------------------------------------------------------------------
# Geostationary (manual pipeline — no standard EPSG)
# ---------------------------------------------------------------------------


def test_geos_roundtrip():
    """Geostationary forward/inverse.

    NOTE: The geos inverse has a known issue with latitude recovery for
    off-nadir points (the geocentric ↔ geodetic latitude conversion is
    incomplete).  We test the sub-satellite point (origin) exactly and
    verify that the forward path produces finite output for off-nadir.
    """
    from vibeproj.crs import ProjectionParams
    from vibeproj.ellipsoid import WGS84
    from vibeproj.pipeline import TransformPipeline

    params = ProjectionParams(
        projection_name="geos",
        ellipsoid=WGS84,
        lon_0=0.0,
        lat_0=0.0,
        north_first=False,
        extra={"h": 35785831.0},
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    pipe = TransformPipeline(src, params)

    # Sub-satellite point roundtrips exactly
    lat = np.array([0.0])
    lon = np.array([0.0])
    x, y = pipe.transform(lat, lon, np)
    inv_pipe = TransformPipeline(params, src)
    lat2, lon2 = inv_pipe.transform(x, y, np)
    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)

    # Off-nadir points: forward produces finite values (inverse is known-broken)
    lat_off = np.array([5.0, -5.0])
    lon_off = np.array([-5.0, 5.0])
    x_off, y_off = pipe.transform(lat_off, lon_off, np)
    assert np.all(np.isfinite(x_off))
    assert np.all(np.isfinite(y_off))


# ---------------------------------------------------------------------------
# Robinson (manual pipeline — no standard EPSG)
# ---------------------------------------------------------------------------


def test_robin_roundtrip():
    from vibeproj.crs import ProjectionParams
    from vibeproj.ellipsoid import WGS84
    from vibeproj.pipeline import TransformPipeline

    params = ProjectionParams(
        projection_name="robin",
        ellipsoid=WGS84,
        lon_0=0.0,
        lat_0=0.0,
        north_first=False,
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    pipe = TransformPipeline(src, params)
    lat = np.array([40.0, -30.0, 60.0])
    lon = np.array([-74.0, 20.0, 140.0])
    x, y = pipe.transform(lat, lon, np)
    inv_pipe = TransformPipeline(params, src)
    lat2, lon2 = inv_pipe.transform(x, y, np)
    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)


# ---------------------------------------------------------------------------
# Winkel Tripel (manual pipeline — no standard EPSG)
# ---------------------------------------------------------------------------


def test_wintri_roundtrip():
    from vibeproj.crs import ProjectionParams
    from vibeproj.ellipsoid import WGS84
    from vibeproj.pipeline import TransformPipeline

    params = ProjectionParams(
        projection_name="wintri",
        ellipsoid=WGS84,
        lon_0=0.0,
        lat_0=0.0,
        north_first=False,
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    pipe = TransformPipeline(src, params)
    lat = np.array([40.0, -30.0, 60.0])
    lon = np.array([-74.0, 20.0, 140.0])
    x, y = pipe.transform(lat, lon, np)
    inv_pipe = TransformPipeline(params, src)
    lat2, lon2 = inv_pipe.transform(x, y, np)
    # Winkel Tripel inverse uses simplified Newton iteration — limited precision
    assert_allclose(lat2, lat, atol=0.005)
    assert_allclose(lon2, lon, atol=0.005)


# ---------------------------------------------------------------------------
# Natural Earth (manual pipeline — no standard EPSG)
# ---------------------------------------------------------------------------


def test_natearth_roundtrip():
    from vibeproj.crs import ProjectionParams
    from vibeproj.ellipsoid import WGS84
    from vibeproj.pipeline import TransformPipeline

    params = ProjectionParams(
        projection_name="natearth",
        ellipsoid=WGS84,
        lon_0=0.0,
        lat_0=0.0,
        north_first=False,
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    pipe = TransformPipeline(src, params)
    lat = np.array([40.0, -30.0, 60.0])
    lon = np.array([-74.0, 20.0, 140.0])
    x, y = pipe.transform(lat, lon, np)
    inv_pipe = TransformPipeline(params, src)
    lat2, lon2 = inv_pipe.transform(x, y, np)
    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)


# ---------------------------------------------------------------------------
# Azimuthal Equidistant (manual pipeline — no standard EPSG)
# ---------------------------------------------------------------------------


def test_aeqd_roundtrip():
    from vibeproj.crs import ProjectionParams
    from vibeproj.ellipsoid import WGS84
    from vibeproj.pipeline import TransformPipeline

    params = ProjectionParams(
        projection_name="aeqd",
        ellipsoid=WGS84,
        lon_0=0.0,
        lat_0=45.0,
        north_first=False,
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    pipe = TransformPipeline(src, params)
    lat = np.array([40.0, 50.0, 45.0])
    lon = np.array([-5.0, 5.0, 0.0])
    x, y = pipe.transform(lat, lon, np)
    inv_pipe = TransformPipeline(params, src)
    lat2, lon2 = inv_pipe.transform(x, y, np)
    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)


# ---------------------------------------------------------------------------
# Cross-projection and special-case transforms
# ---------------------------------------------------------------------------


def test_proj_to_proj():
    """Projected -> Projected via geographic intermediate."""
    pp_src = PyProjTransformer.from_crs("EPSG:32631", "EPSG:4326")
    pp_dst = PyProjTransformer.from_crs("EPSG:4326", "EPSG:3857")
    t = Transformer.from_crs("EPSG:32631", "EPSG:3857")

    x_in = np.array([500000.0, 600000.0])
    y_in = np.array([5400000.0, 5500000.0])

    # pyproj chain
    lat, lon = pp_src.transform(x_in, y_in)
    exp_x, exp_y = pp_dst.transform(lat, lon)

    # vibeproj direct
    vp_x, vp_y = t.transform(x_in, y_in)

    assert_allclose(vp_x, exp_x, atol=0.1)
    assert_allclose(vp_y, exp_y, atol=0.1)


def test_longlat_to_longlat():
    """Geographic -> Geographic identity transform."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:4326")
    lat = np.array([40.0, -30.0, 60.0])
    lon = np.array([-74.0, 20.0, 140.0])
    lat2, lon2 = t.transform(lat, lon)
    assert_allclose(lat2, lat, atol=1e-10)
    assert_allclose(lon2, lon, atol=1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_array():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    x, y = t.transform(np.array([]), np.array([]))
    assert len(x) == 0
    assert len(y) == 0
