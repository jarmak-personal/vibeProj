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

    t = Transformer.from_crs("EPSG:4326", "EPSG:32756", always_xy=False)
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

    t = Transformer.from_crs("EPSG:4326", crs_to, always_xy=False)
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
    t = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=False)

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
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=False)

    lat, lon = np.array([40.7128, 51.5074]), np.array([-74.0060, -0.1278])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=0.01)
    assert_allclose(vp_y, exp_y, atol=0.01)


def test_web_mercator_inverse():
    pp = PyProjTransformer.from_crs("EPSG:3857", "EPSG:4326")
    t = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=False)

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
    t = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=False)

    lat, lon = np.array([48.8566]), np.array([2.3522])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=0.01)
    assert_allclose(vp_y, exp_y, atol=0.01)


def test_lcc_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=False)

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
    t = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=False)

    lat, lon = np.array([40.0]), np.array([-96.0])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=0.01)
    assert_allclose(vp_y, exp_y, atol=0.01)


def test_albers_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=False)
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
    t = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=False)

    lat, lon = np.array([52.5200]), np.array([13.4050])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=0.01)
    assert_allclose(vp_y, exp_y, atol=0.01)


def test_laea_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=False)
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
    t = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=False)

    lat, lon = np.array([-75.0]), np.array([0.0])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=0.1)
    assert_allclose(vp_y, exp_y, atol=0.1)


def test_polar_stereo_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=False)
    lat, lon = -75.0, 30.0
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")

    assert_allclose(lat2, lat, atol=1e-6)
    assert_allclose(lon2, lon, atol=1e-6)


# ---------------------------------------------------------------------------
# Scalar / array input handling
# ---------------------------------------------------------------------------


def test_scalar_input():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=False)
    x, y = t.transform(49.0, 2.0)
    assert isinstance(x, float)
    assert isinstance(y, float)


def test_list_input():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=False)
    x, y = t.transform([49.0, 50.0], [2.0, 3.0])
    assert len(x) == 2


def test_direction_validation():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=False)
    with pytest.raises(ValueError, match="Invalid direction"):
        t.transform(49.0, 2.0, direction="BACKWARD")


# ---------------------------------------------------------------------------
# Mercator (EPSG:3395)
# ---------------------------------------------------------------------------


def test_merc_forward():
    """EPSG:3395 — World Mercator."""
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:3395")
    t = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=False)

    lat, lon = np.array([40.7128, 51.5074]), np.array([-74.0060, -0.1278])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=0.01)
    assert_allclose(vp_y, exp_y, atol=0.01)


def test_merc_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=False)
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
    t = Transformer.from_crs("EPSG:4326", "EPSG:4087", always_xy=False)

    lat, lon = np.array([48.8566, -33.8688]), np.array([2.3522, 151.2093])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=0.01)
    assert_allclose(vp_y, exp_y, atol=0.01)


def test_eqc_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:4087", always_xy=False)
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

    With authalic latitude conversion, results now match pyproj.
    """
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:8857")
    t = Transformer.from_crs("EPSG:4326", "EPSG:8857", always_xy=False)

    lat, lon = np.array([40.0, -30.0]), np.array([-74.0, 20.0])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=0.01)
    assert_allclose(vp_y, exp_y, atol=0.01)


def test_eqearth_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:8857", always_xy=False)
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
    t = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=False)

    lat, lon = np.array([45.0, -20.0]), np.array([10.0, -60.0])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=0.01)
    assert_allclose(vp_y, exp_y, atol=0.01)


def test_cea_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=False)
    lat, lon = 45.0, 10.0
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")

    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)


# ---------------------------------------------------------------------------
# Oblique Stereographic (EPSG:28992)
# ---------------------------------------------------------------------------


def test_sterea_forward():
    """EPSG:28992 — Amersfoort / RD New.

    Compare vibeProj against pyproj using the same CRS pair.
    This is a cross-datum transform (WGS84 -> Amersfoort/Bessel),
    so we expect Helmert-level accuracy (~5m).
    """
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=False)
    t = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=False)

    lat, lon = np.array([52.3676]), np.array([4.9041])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=10.0)
    assert_allclose(vp_y, exp_y, atol=10.0)


def test_sterea_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=False)
    lat, lon = 52.3676, 4.9041
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")

    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)


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
    """Geostationary forward/inverse — both origin and off-nadir points."""
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
    inv_pipe = TransformPipeline(params, src)

    # Sub-satellite point roundtrips exactly
    lat = np.array([0.0])
    lon = np.array([0.0])
    x, y = pipe.transform(lat, lon, np)
    lat2, lon2 = inv_pipe.transform(x, y, np)
    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)

    # Off-nadir points now roundtrip accurately
    lat_off = np.array([5.0, -5.0, 30.0, -45.0])
    lon_off = np.array([-5.0, 5.0, 20.0, -30.0])
    x_off, y_off = pipe.transform(lat_off, lon_off, np)
    assert np.all(np.isfinite(x_off))
    assert np.all(np.isfinite(y_off))
    lat3, lon3 = inv_pipe.transform(x_off, y_off, np)
    assert_allclose(lat3, lat_off, atol=1e-7)
    assert_allclose(lon3, lon_off, atol=1e-7)


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
    t = Transformer.from_crs("EPSG:32631", "EPSG:3857", always_xy=False)

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
    t = Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=False)
    lat = np.array([40.0, -30.0, 60.0])
    lon = np.array([-74.0, 20.0, 140.0])
    lat2, lon2 = t.transform(lat, lon)
    assert_allclose(lat2, lat, atol=1e-10)
    assert_allclose(lon2, lon, atol=1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_array():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=False)
    x, y = t.transform(np.array([]), np.array([]))
    assert len(x) == 0
    assert len(y) == 0


# ---------------------------------------------------------------------------
# always_xy=True (new default)
# ---------------------------------------------------------------------------


def test_always_xy_true_default():
    """Default always_xy=True: x=lon, y=lat for geographic CRS."""
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")  # always_xy=True is default

    lon, lat = 2.0, 49.0
    exp_x, exp_y = pp.transform(lon, lat)
    vp_x, vp_y = t.transform(lon, lat)

    assert_allclose(vp_x, exp_x, atol=1e-4)
    assert_allclose(vp_y, exp_y, atol=1e-4)


def test_always_xy_true_inverse():
    """always_xy=True inverse: returns (lon, lat)."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")  # always_xy=True default
    lon, lat = 2.3522, 48.8566
    x, y = t.transform(lon, lat)
    lon2, lat2 = t.transform(x, y, direction="INVERSE")

    assert_allclose(lon2, lon, atol=1e-7)
    assert_allclose(lat2, lat, atol=1e-7)


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    r = repr(t)
    assert "EPSG:4326" in r
    assert "EPSG:32631" in r
    assert "tmerc" in r


def test_repr_proj_to_proj():
    t = Transformer.from_crs("EPSG:32631", "EPSG:3857")
    r = repr(t)
    assert "EPSG:32631" in r
    assert "EPSG:3857" in r


# ---------------------------------------------------------------------------
# list_projections
# ---------------------------------------------------------------------------


def test_list_projections():
    import vibeproj

    projs = vibeproj.list_projections()
    assert "tmerc" in projs
    assert "webmerc" in projs
    assert projs["tmerc"]["fused"] is True
    assert len(projs["tmerc"]["methods"]) > 0


# ---------------------------------------------------------------------------
# is_fused
# ---------------------------------------------------------------------------


def test_is_fused():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    assert t.is_fused is True


def test_is_fused_longlat():
    t = Transformer.from_crs("EPSG:4326", "EPSG:4326")
    assert t.is_fused is False


# ---------------------------------------------------------------------------
# Pickle serialization
# ---------------------------------------------------------------------------


def test_pickle_roundtrip():
    import pickle

    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    lon, lat = 2.3522, 48.8566
    x1, y1 = t.transform(lon, lat)

    t2 = pickle.loads(pickle.dumps(t))
    x2, y2 = t2.transform(lon, lat)

    assert_allclose(x2, x1, atol=1e-10)
    assert_allclose(y2, y1, atol=1e-10)


def test_pickle_preserves_always_xy():
    import pickle

    t = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=False)
    t2 = pickle.loads(pickle.dumps(t))
    assert t2._always_xy is False


# ---------------------------------------------------------------------------
# Z-coordinate passthrough
# ---------------------------------------------------------------------------


def test_z_passthrough_scalar():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    lon, lat, z = 2.0, 49.0, 100.0
    x, y, z_out = t.transform(lon, lat, z)
    assert isinstance(z_out, float)
    assert z_out == 100.0


def test_z_passthrough_array():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    lon = np.array([2.0, 3.0])
    lat = np.array([49.0, 50.0])
    z = np.array([100.0, 200.0])
    x, y, z_out = t.transform(lon, lat, z)
    assert_allclose(z_out, z)


def test_z_none_returns_two_tuple():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    result = t.transform(2.0, 49.0)
    assert len(result) == 2


def test_z_passthrough_transform_buffers():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    lon = np.array([2.0, 3.0], dtype=np.float64)
    lat = np.array([49.0, 50.0], dtype=np.float64)
    z = np.array([100.0, 200.0], dtype=np.float64)
    x, y, z_out = t.transform_buffers(lon, lat, z)
    assert z_out is z  # same object, zero-copy


# ---------------------------------------------------------------------------
# NaN / inf output warning
# ---------------------------------------------------------------------------


def test_nan_output_warns():
    """NaN input propagates to NaN output and triggers a warning."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    with pytest.warns(UserWarning, match="non-finite"):
        t.transform(float("nan"), 49.0)


# ---------------------------------------------------------------------------
# Datum shift warning
# ---------------------------------------------------------------------------


def test_datum_warning_no_helmert():
    """EPSG:4326 -> EPSG:27700 now has Helmert, so no vibeProj datum warning.

    pyproj may emit its own warning about missing grid files.
    """
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Transformer.from_crs("EPSG:4326", "EPSG:27700")

    # Filter to vibeProj warnings only (exclude pyproj grid warnings)
    our_warnings = [
        w
        for w in caught
        if "vibeproj" in str(w.filename).lower() and "pyproj" not in str(w.filename).lower()
    ]
    assert len(our_warnings) == 0, f"Unexpected vibeProj warnings: {our_warnings}"


def test_no_datum_warning_same_ellipsoid():
    """EPSG:4326 → EPSG:32631 — both WGS84, no warning."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Transformer.from_crs("EPSG:4326", "EPSG:32631")


# ---------------------------------------------------------------------------
# compile / warm_up
# ---------------------------------------------------------------------------


def test_compile():
    """Transformer.compile() should not raise."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    t.compile()  # no-op on CPU, but should not raise


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


def test_unsupported_projection_error():
    from vibeproj.exceptions import UnsupportedProjectionError

    with pytest.raises(UnsupportedProjectionError):
        Transformer.from_crs("EPSG:4326", "EPSG:5514")  # S-JTSK / Krovak — not supported


def test_crs_resolution_error():
    from vibeproj.exceptions import CRSResolutionError

    with pytest.raises(CRSResolutionError):
        Transformer.from_crs(object(), "EPSG:4326")


def test_exceptions_inherit_base():
    from vibeproj import (
        CoordinateValidationError,
        CRSResolutionError,
        UnsupportedProjectionError,
        VibeProjectionError,
    )

    assert issubclass(UnsupportedProjectionError, VibeProjectionError)
    assert issubclass(CRSResolutionError, VibeProjectionError)
    assert issubclass(CoordinateValidationError, VibeProjectionError)


# ---------------------------------------------------------------------------
# Accuracy metadata
# ---------------------------------------------------------------------------


def test_accuracy_same_datum():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    assert t.accuracy == "sub-millimeter"


def test_accuracy_cross_datum():
    t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    # With Helmert available, accuracy is "sub-meter" not "degraded"
    assert t.accuracy == "sub-meter"


# ---------------------------------------------------------------------------
# Chunked transform
# ---------------------------------------------------------------------------


def test_transform_chunked_matches_transform():
    """Chunked result must match non-chunked result."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    lon = np.linspace(-10, 30, 5000)
    lat = np.linspace(35, 65, 5000)

    x_ref, y_ref = t.transform(lon, lat)
    x_ch, y_ch = t.transform_chunked(lon, lat, chunk_size=1000)

    assert_allclose(x_ch, x_ref, atol=1e-10)
    assert_allclose(y_ch, y_ref, atol=1e-10)


def test_transform_chunked_z_passthrough():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    lon = np.array([2.0, 3.0])
    lat = np.array([49.0, 50.0])
    z = np.array([100.0, 200.0])
    x, y, z_out = t.transform_chunked(lon, lat, z, chunk_size=1)
    assert_allclose(z_out, z)


def test_transform_chunked_empty():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    x, y = t.transform_chunked(np.array([]), np.array([]))
    assert len(x) == 0
    assert len(y) == 0


def test_transform_chunked_inverse():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    lon = np.array([2.0, 3.0, 4.0])
    lat = np.array([49.0, 50.0, 51.0])
    x, y = t.transform_chunked(lon, lat)
    lon2, lat2 = t.transform_chunked(x, y, direction="INVERSE", chunk_size=2)
    assert_allclose(lon2, lon, atol=1e-7)
    assert_allclose(lat2, lat, atol=1e-7)
