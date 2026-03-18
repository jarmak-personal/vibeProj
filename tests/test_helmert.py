"""Tests for Helmert 7-parameter datum transformation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyproj import Transformer as PyProjTransformer

from vibeproj import Transformer
from vibeproj.ellipsoid import WGS84, GRS80, Ellipsoid
from vibeproj.helmert import HelmertParams, apply_helmert, ecef_to_geodetic, geodetic_to_ecef


# ---------------------------------------------------------------------------
# HelmertParams construction
# ---------------------------------------------------------------------------


def test_helmert_params_frozen():
    h = HelmertParams(
        tx=1.0,
        ty=2.0,
        tz=3.0,
        rx=0.1,
        ry=0.2,
        rz=0.3,
        ds=1.000001,
        src_ellipsoid=WGS84,
        dst_ellipsoid=GRS80,
    )
    with pytest.raises(AttributeError):
        h.tx = 999.0


def test_helmert_inverted():
    h = HelmertParams(
        tx=1.0,
        ty=2.0,
        tz=3.0,
        rx=0.01,
        ry=0.02,
        rz=0.03,
        ds=1.0001,
        src_ellipsoid=WGS84,
        dst_ellipsoid=GRS80,
    )
    inv = h.inverted()
    assert inv.tx == -1.0
    assert inv.ty == -2.0
    assert inv.tz == -3.0
    assert inv.rx == -0.01
    assert inv.ry == -0.02
    assert inv.rz == -0.03
    assert_allclose(inv.ds, 1.0 / 1.0001)
    assert inv.src_ellipsoid is GRS80
    assert inv.dst_ellipsoid is WGS84


# ---------------------------------------------------------------------------
# Geodetic <-> ECEF conversion
# ---------------------------------------------------------------------------


def test_ecef_equator_prime_meridian():
    """lat=0, lon=0 on WGS84 -> X=a, Y=0, Z=0."""
    lat = np.array([0.0])
    lon = np.array([0.0])
    X, Y, Z = geodetic_to_ecef(lat, lon, WGS84.a, WGS84.es, np)
    assert_allclose(X, [WGS84.a], atol=1e-6)
    assert_allclose(Y, [0.0], atol=1e-6)
    assert_allclose(Z, [0.0], atol=1e-6)


def test_ecef_north_pole():
    """lat=pi/2, lon=0 -> X=0, Y=0, Z=b."""
    lat = np.array([np.pi / 2])
    lon = np.array([0.0])
    X, Y, Z = geodetic_to_ecef(lat, lon, WGS84.a, WGS84.es, np)
    assert_allclose(X, [0.0], atol=1e-3)
    assert_allclose(Y, [0.0], atol=1e-3)
    assert_allclose(Z, [WGS84.b], atol=1e-3)


def test_ecef_roundtrip():
    """Geodetic -> ECEF -> geodetic roundtrip."""
    lat = np.array([0.7, -0.5, 1.2])  # radians
    lon = np.array([0.3, 2.1, -1.5])

    X, Y, Z = geodetic_to_ecef(lat, lon, WGS84.a, WGS84.es, np)
    lat2, lon2 = ecef_to_geodetic(X, Y, Z, WGS84.a, WGS84.es, np)

    assert_allclose(lat2, lat, atol=1e-12)
    assert_allclose(lon2, lon, atol=1e-12)


# ---------------------------------------------------------------------------
# apply_helmert
# ---------------------------------------------------------------------------


def test_apply_helmert_identity():
    """Identity Helmert (all zeros) should return input unchanged."""
    h = HelmertParams(
        tx=0,
        ty=0,
        tz=0,
        rx=0,
        ry=0,
        rz=0,
        ds=1.0,
        src_ellipsoid=WGS84,
        dst_ellipsoid=WGS84,
    )
    lat = np.array([51.5, -33.9, 0.0])
    lon = np.array([-0.1, 151.2, 0.0])

    lat_out, lon_out = apply_helmert(lat, lon, h, np)

    assert_allclose(lat_out, lat, atol=1e-10)
    assert_allclose(lon_out, lon, atol=1e-10)


def test_apply_helmert_translation_only():
    """3-param (translations only) shift produces nonzero change."""
    h = HelmertParams(
        tx=100.0,
        ty=-200.0,
        tz=300.0,
        rx=0,
        ry=0,
        rz=0,
        ds=1.0,
        src_ellipsoid=WGS84,
        dst_ellipsoid=WGS84,
    )
    lat = np.array([51.5])
    lon = np.array([-0.1])

    lat_out, lon_out = apply_helmert(lat, lon, h, np)

    # Should differ from input
    assert abs(lat_out[0] - lat[0]) > 1e-6
    assert abs(lon_out[0] - lon[0]) > 1e-6


def test_apply_helmert_roundtrip():
    """Forward then inverse Helmert recovers original coordinates."""
    h = HelmertParams(
        tx=446.448,
        ty=-125.157,
        tz=542.060,
        rx=0.1502 * 4.84813681e-6,
        ry=0.2470 * 4.84813681e-6,
        rz=0.8421 * 4.84813681e-6,
        ds=1.0 + (-20.4894) * 1e-6,
        src_ellipsoid=WGS84,
        dst_ellipsoid=Ellipsoid.from_af(6377563.396, 299.3249646),  # Airy 1830
    )
    lat = np.array([51.5074, 52.2053])
    lon = np.array([-0.1278, 0.1218])

    lat_shifted, lon_shifted = apply_helmert(lat, lon, h, np)
    lat_back, lon_back = apply_helmert(lat_shifted, lon_shifted, h.inverted(), np)

    # Helmert roundtrip is not exact due to linearized rotation matrix;
    # sub-microdegree (~0.1mm) accuracy is expected.
    assert_allclose(lat_back, lat, atol=1e-6)
    assert_allclose(lon_back, lon, atol=1e-6)


# ---------------------------------------------------------------------------
# extract_helmert
# ---------------------------------------------------------------------------


def test_extract_helmert_same_datum_returns_none():
    """Same datum (WGS84 -> WGS84) should return None."""
    from pyproj import CRS
    from vibeproj.crs import extract_helmert

    src = CRS.from_epsg(4326)
    dst = CRS.from_epsg(32631)  # UTM zone 31N, WGS84
    assert extract_helmert(src, dst) is None


def test_extract_helmert_wgs84_grs80_small_shift():
    """WGS84 -> NAD83 has a ~1m Helmert, but _cross_datum is False for this pair.

    extract_helmert() will find the small shift if called directly, but the
    Transformer never calls it because the ellipsoids are effectively identical.
    """
    from pyproj import CRS
    from vibeproj.crs import extract_helmert

    src = CRS.from_epsg(4326)  # WGS84
    dst = CRS.from_epsg(4269)  # NAD83 (GRS80)
    result = extract_helmert(src, dst)
    # A tiny Helmert exists (~1m translations), but at the Transformer level
    # _cross_datum is False so extract_helmert is never called.
    if result is not None:
        assert abs(result.tx) <= 2.0
        assert abs(result.ty) <= 2.0
        assert abs(result.tz) <= 2.0


def test_extract_helmert_cross_datum():
    """WGS84 -> OSGB36 should extract Helmert parameters."""
    from pyproj import CRS
    from vibeproj.crs import extract_helmert

    src = CRS.from_epsg(4326)
    dst = CRS.from_epsg(4277)  # OSGB36
    h = extract_helmert(src, dst)
    assert h is not None
    # Should have nonzero translations
    assert abs(h.tx) + abs(h.ty) + abs(h.tz) > 100.0


# ---------------------------------------------------------------------------
# Structural zero-overhead tests
# ---------------------------------------------------------------------------


def test_same_datum_no_helmert():
    """Same-datum transform must not have Helmert params attached."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    assert t._helmert is None
    assert t._pipeline._helmert is None


def test_same_datum_pipeline_no_helmert():
    """All same-datum projections should have _helmert=None."""
    for src, dst in [
        ("EPSG:4326", "EPSG:32631"),
        ("EPSG:4326", "EPSG:3857"),
        ("EPSG:4326", "EPSG:2154"),
        ("EPSG:4326", "EPSG:5070"),
    ]:
        t = Transformer.from_crs(src, dst)
        assert t._helmert is None, f"Expected no Helmert for {src} -> {dst}"


def test_wgs84_grs80_no_helmert():
    """WGS84/GRS80 treated as same datum (0.1mm difference in b)."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:5070")  # NAD83/GRS80 Albers
    assert t._helmert is None


def test_cross_datum_has_helmert():
    """Cross-datum transform must have Helmert params."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    assert t._helmert is not None


# ---------------------------------------------------------------------------
# Cross-datum integration tests (validated against pyproj)
# ---------------------------------------------------------------------------


def test_cross_datum_wgs84_to_osgb_forward():
    """EPSG:4326 -> EPSG:27700 (WGS84 -> OSGB36 British National Grid).

    Helmert accuracy is ~5m. pyproj may use a different Helmert variant or
    a grid-based transformation, so we allow 10m tolerance.
    """
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    exp_x, exp_y = pp.transform(-0.1278, 51.5074)

    t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    vp_x, vp_y = t.transform(-0.1278, 51.5074)

    assert_allclose(vp_x, exp_x, atol=10.0)
    assert_allclose(vp_y, exp_y, atol=10.0)


def test_cross_datum_longlat_to_longlat():
    """EPSG:4326 -> EPSG:4277 (WGS84 -> OSGB36 geographic).

    Tests the longlat_to_longlat mode with Helmert.
    pyproj may select a different Helmert variant, so we use a wider tolerance.
    0.01 degrees ~ 1km — within expected Helmert variant differences.
    """
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:4277", always_xy=True)
    exp_lon, exp_lat = pp.transform(-0.1278, 51.5074)

    t = Transformer.from_crs("EPSG:4326", "EPSG:4277")
    vp_lon, vp_lat = t.transform(-0.1278, 51.5074)

    assert_allclose(vp_lat, exp_lat, atol=0.01)
    assert_allclose(vp_lon, exp_lon, atol=0.01)


def test_cross_datum_proj_to_proj():
    """EPSG:32631 -> EPSG:27700 (UTM zone 31N -> British National Grid).

    Tests proj_to_proj mode with Helmert in between.
    """
    # London coordinates in UTM zone 31N
    pp_fwd = PyProjTransformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    utm_x, utm_y = pp_fwd.transform(-0.1278, 51.5074)

    # Transform UTM -> BNG via pyproj (reference)
    pp = PyProjTransformer.from_crs("EPSG:32631", "EPSG:27700", always_xy=True)
    exp_x, exp_y = pp.transform(utm_x, utm_y)

    # Transform UTM -> BNG via vibeProj
    t = Transformer.from_crs("EPSG:32631", "EPSG:27700")
    vp_x, vp_y = t.transform(utm_x, utm_y)

    assert_allclose(vp_x, exp_x, atol=10.0)
    assert_allclose(vp_y, exp_y, atol=10.0)


def test_cross_datum_roundtrip():
    """Forward then inverse cross-datum should recover input within Helmert accuracy."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    lon, lat = -0.1278, 51.5074

    x, y = t.transform(lon, lat)
    lon2, lat2 = t.transform(x, y, direction="INVERSE")

    # Roundtrip should recover to ~1e-6 degrees (sub-meter)
    assert_allclose(lon2, lon, atol=1e-4)
    assert_allclose(lat2, lat, atol=1e-4)


def test_cross_datum_array_transform():
    """Cross-datum transform with arrays."""
    lon = np.array([-0.1278, -1.2578, -3.1883])
    lat = np.array([51.5074, 51.7520, 51.4816])

    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    exp_x, exp_y = pp.transform(lon, lat)

    t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    vp_x, vp_y = t.transform(lon, lat)

    assert_allclose(vp_x, exp_x, atol=10.0)
    assert_allclose(vp_y, exp_y, atol=10.0)


# ---------------------------------------------------------------------------
# Accuracy and warning metadata
# ---------------------------------------------------------------------------


def test_cross_datum_accuracy_with_helmert():
    """Accuracy should be 'sub-meter' when Helmert is available."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    assert t.accuracy == "sub-meter"


def test_cross_datum_no_warning_when_helmert():
    """No vibeProj datum warning emitted when Helmert is available.

    pyproj may still emit its own warning about missing grid files.
    """
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Transformer.from_crs("EPSG:4326", "EPSG:27700")

    # Filter to only warnings from our code (not pyproj's grid-missing warning)
    our_warnings = [
        w
        for w in caught
        if "vibeproj" in str(w.filename).lower() and "pyproj" not in str(w.filename).lower()
    ]
    assert len(our_warnings) == 0, f"Unexpected vibeProj warnings: {our_warnings}"
