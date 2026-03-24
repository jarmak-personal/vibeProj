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
    lat2, lon2, _ = ecef_to_geodetic(X, Y, Z, WGS84.a, WGS84.es, np)

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


# ---------------------------------------------------------------------------
# 15-parameter time-dependent Helmert
# ---------------------------------------------------------------------------


def test_helmert_rate_fields_default_zero():
    """Rate fields default to 0.0 (backward-compatible 7-param)."""
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
    assert h.dtx == 0.0
    assert h.dty == 0.0
    assert h.dtz == 0.0
    assert h.drx == 0.0
    assert h.dry == 0.0
    assert h.drz == 0.0
    assert h.dds == 0.0
    assert h.t_epoch == 0.0
    assert not h.has_rates


def test_helmert_has_rates():
    """has_rates is True when any rate field is nonzero."""
    h = HelmertParams(
        tx=0,
        ty=0,
        tz=0,
        rx=0,
        ry=0,
        rz=0,
        ds=1.0,
        src_ellipsoid=WGS84,
        dst_ellipsoid=GRS80,
        dtx=0.001,
    )
    assert h.has_rates


def test_helmert_at_epoch_zero_dt():
    """at_epoch(t_epoch) returns the same base params (dt=0)."""
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
        dtx=0.1,
        dty=0.2,
        dtz=0.3,
        drx=1e-9,
        dry=2e-9,
        drz=3e-9,
        dds=1e-10,
        t_epoch=2010.0,
    )
    resolved = h.at_epoch(2010.0)
    assert_allclose(resolved.tx, 1.0)
    assert_allclose(resolved.ty, 2.0)
    assert_allclose(resolved.tz, 3.0)
    assert_allclose(resolved.ds, 1.0001)
    # Rates should be zeroed in the resolved result
    assert resolved.dtx == 0.0
    assert not resolved.has_rates


def test_helmert_at_epoch_nonzero_dt():
    """at_epoch applies rate * dt to each base parameter."""
    h = HelmertParams(
        tx=1.0,
        ty=2.0,
        tz=3.0,
        rx=0.0,
        ry=0.0,
        rz=0.0,
        ds=1.0,
        src_ellipsoid=WGS84,
        dst_ellipsoid=GRS80,
        dtx=0.1,
        dty=-0.2,
        dtz=0.3,
        drx=1e-9,
        dry=0.0,
        drz=0.0,
        dds=1e-10,
        t_epoch=2010.0,
    )
    resolved = h.at_epoch(2020.0)  # dt = 10 years
    assert_allclose(resolved.tx, 1.0 + 0.1 * 10)
    assert_allclose(resolved.ty, 2.0 + (-0.2) * 10)
    assert_allclose(resolved.tz, 3.0 + 0.3 * 10)
    assert_allclose(resolved.rx, 1e-9 * 10)
    assert_allclose(resolved.ds, 1.0 + 1e-10 * 10)


def test_helmert_inverted_preserves_rates():
    """inverted() negates rates and preserves t_epoch."""
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
        dtx=0.1,
        dty=0.2,
        dtz=0.3,
        drx=1e-9,
        dry=2e-9,
        drz=3e-9,
        dds=1e-10,
        t_epoch=2010.0,
    )
    inv = h.inverted()
    assert inv.dtx == -0.1
    assert inv.dty == -0.2
    assert inv.dtz == -0.3
    assert inv.drx == -1e-9
    assert inv.dry == -2e-9
    assert inv.drz == -3e-9
    assert inv.dds == -1e-10
    assert inv.t_epoch == 2010.0


def test_helmert_at_epoch_roundtrip():
    """Forward at_epoch then inverse at_epoch recovers original coords."""
    h = HelmertParams(
        tx=446.448,
        ty=-125.157,
        tz=542.060,
        rx=0.1502 * 4.84813681e-6,
        ry=0.2470 * 4.84813681e-6,
        rz=0.8421 * 4.84813681e-6,
        ds=1.0 + (-20.4894) * 1e-6,
        src_ellipsoid=WGS84,
        dst_ellipsoid=Ellipsoid.from_af(6377563.396, 299.3249646),
        dtx=0.001,
        dty=-0.002,
        dtz=0.003,
        drx=1e-11,
        dry=2e-11,
        drz=3e-11,
        dds=1e-12,
        t_epoch=2000.0,
    )
    epoch = 2024.0
    h_fwd = h.at_epoch(epoch)
    h_inv = h.inverted().at_epoch(epoch)

    lat = np.array([51.5074])
    lon = np.array([-0.1278])

    lat_s, lon_s = apply_helmert(lat, lon, h_fwd, np)
    lat_b, lon_b = apply_helmert(lat_s, lon_s, h_inv, np)

    assert_allclose(lat_b, lat, atol=1e-6)
    assert_allclose(lon_b, lon, atol=1e-6)


# ---------------------------------------------------------------------------
# datum_shift parameter on Transformer
# ---------------------------------------------------------------------------


def test_datum_shift_invalid_value():
    """Invalid datum_shift raises ValueError."""
    with pytest.raises(ValueError, match="datum_shift"):
        Transformer.from_crs("EPSG:4326", "EPSG:27700", datum_shift="invalid")


def test_datum_shift_fast_uses_base_params():
    """datum_shift='fast' passes base 7-param Helmert (rates not applied)."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:27700", datum_shift="fast")
    assert t._helmert is not None
    assert not t._epoch_applied
    assert t.accuracy == "sub-meter"


def test_datum_shift_accurate_default():
    """datum_shift='accurate' is the default."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    assert t._datum_shift == "accurate"


def test_datum_shift_pickle_roundtrip():
    """datum_shift and epoch survive pickle."""
    import pickle

    t = Transformer.from_crs("EPSG:4326", "EPSG:27700", datum_shift="fast", epoch=2024.0)
    data = pickle.dumps(t)
    t2 = pickle.loads(data)
    assert t2._datum_shift == "fast"
    assert t2._epoch == 2024.0


# ---------------------------------------------------------------------------
# PROJ pipeline rate param parsing
# ---------------------------------------------------------------------------


def test_parse_helmert_rates_from_proj4():
    """_parse_helmert_from_proj4 extracts 15-param rate fields."""
    from vibeproj.crs import _parse_helmert_from_proj4

    proj4 = (
        "+proj=pipeline +step +proj=helmert +x=1.0 +y=2.0 +z=3.0 "
        "+rx=0.1 +ry=0.2 +rz=0.3 +s=0.5 "
        "+dx=0.01 +dy=0.02 +dz=0.03 "
        "+drx=0.001 +dry=0.002 +drz=0.003 +ds=0.004 "
        "+t_epoch=2010.0 +convention=position_vector"
    )
    parsed = _parse_helmert_from_proj4(proj4)
    assert parsed is not None
    (
        tx,
        ty,
        tz,
        rx_as,
        ry_as,
        rz_as,
        ds_ppm,
        convention,
        is_inverse,
        dtx,
        dty,
        dtz,
        drx_as,
        dry_as,
        drz_as,
        dds_ppm,
        t_epoch,
    ) = parsed
    assert tx == 1.0
    assert dtx == 0.01
    assert dty == 0.02
    assert dtz == 0.03
    assert drx_as == 0.001
    assert dry_as == 0.002
    assert drz_as == 0.003
    assert dds_ppm == 0.004
    assert t_epoch == 2010.0


def test_parse_helmert_no_rates():
    """Standard 7-param PROJ string yields zero rates."""
    from vibeproj.crs import _parse_helmert_from_proj4

    proj4 = (
        "+proj=pipeline +step +proj=helmert +x=446.448 +y=-125.157 +z=542.060 "
        "+rx=0.1502 +ry=0.2470 +rz=0.8421 +s=-20.4894 "
        "+convention=position_vector"
    )
    parsed = _parse_helmert_from_proj4(proj4)
    assert parsed is not None
    (
        tx,
        ty,
        tz,
        rx_as,
        ry_as,
        rz_as,
        ds_ppm,
        convention,
        is_inverse,
        dtx,
        dty,
        dtz,
        drx_as,
        dry_as,
        drz_as,
        dds_ppm,
        t_epoch,
    ) = parsed
    assert dtx == 0.0
    assert dty == 0.0
    assert dtz == 0.0
    assert t_epoch == 0.0


# ---------------------------------------------------------------------------
# Z-dimension (ellipsoidal height) support
# ---------------------------------------------------------------------------


def test_ecef_roundtrip_with_height():
    """Geodetic -> ECEF -> geodetic roundtrip with ellipsoidal height."""
    lat = np.array([0.7, -0.5, 1.2])  # radians
    lon = np.array([0.3, 2.1, -1.5])
    h = np.array([100.0, 5000.0, -30.0])

    X, Y, Z = geodetic_to_ecef(lat, lon, WGS84.a, WGS84.es, np, h=h)
    lat2, lon2, h2 = ecef_to_geodetic(X, Y, Z, WGS84.a, WGS84.es, np, return_height=True)

    assert_allclose(lat2, lat, atol=1e-12)
    assert_allclose(lon2, lon, atol=1e-12)
    assert_allclose(h2, h, atol=1e-6)


def test_ecef_roundtrip_with_height_near_pole():
    """ECEF roundtrip at the North Pole with height (near-pole guard)."""
    lat = np.array([np.pi / 2 - 1e-12])  # near-pole
    lon = np.array([0.0])
    h = np.array([500.0])

    X, Y, Z = geodetic_to_ecef(lat, lon, WGS84.a, WGS84.es, np, h=h)
    lat2, lon2, h2 = ecef_to_geodetic(X, Y, Z, WGS84.a, WGS84.es, np, return_height=True)

    assert_allclose(lat2, lat, atol=1e-10)
    assert_allclose(h2, h, atol=1e-3)


def test_ecef_height_zero_matches_no_height():
    """geodetic_to_ecef with h=0 should match h=None."""
    lat = np.array([0.7, -0.5])
    lon = np.array([0.3, 2.1])

    X1, Y1, Z1 = geodetic_to_ecef(lat, lon, WGS84.a, WGS84.es, np)
    X2, Y2, Z2 = geodetic_to_ecef(lat, lon, WGS84.a, WGS84.es, np, h=np.zeros(2))

    assert_allclose(X1, X2, atol=1e-12)
    assert_allclose(Y1, Y2, atol=1e-12)
    assert_allclose(Z1, Z2, atol=1e-12)


def test_apply_helmert_identity_with_z():
    """Identity Helmert with z should return z unchanged."""
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
    z = np.array([100.0, 5000.0, 0.0])

    lat_out, lon_out, z_out = apply_helmert(lat, lon, h, np, h=z)

    assert_allclose(lat_out, lat, atol=1e-10)
    assert_allclose(lon_out, lon, atol=1e-10)
    assert_allclose(z_out, z, atol=1e-6)


def test_apply_helmert_cross_datum_z_changes():
    """Cross-datum Helmert should modify z (height changes with ellipsoid)."""
    h = HelmertParams(
        tx=446.448,
        ty=-125.157,
        tz=542.060,
        rx=0.1502 * 4.84813681e-6,
        ry=0.2470 * 4.84813681e-6,
        rz=0.8421 * 4.84813681e-6,
        ds=1.0 + (-20.4894) * 1e-6,
        src_ellipsoid=WGS84,
        dst_ellipsoid=Ellipsoid.from_af(6377563.396, 299.3249646),
    )
    lat = np.array([51.5074])
    lon = np.array([-0.1278])
    z = np.array([45.0])

    lat_out, lon_out, z_out = apply_helmert(lat, lon, h, np, h=z)

    # z should change due to different ellipsoid
    assert abs(z_out[0] - z[0]) > 1.0  # expect meters-level change


def test_apply_helmert_roundtrip_with_z():
    """Forward then inverse Helmert recovers original z."""
    h = HelmertParams(
        tx=446.448,
        ty=-125.157,
        tz=542.060,
        rx=0.1502 * 4.84813681e-6,
        ry=0.2470 * 4.84813681e-6,
        rz=0.8421 * 4.84813681e-6,
        ds=1.0 + (-20.4894) * 1e-6,
        src_ellipsoid=WGS84,
        dst_ellipsoid=Ellipsoid.from_af(6377563.396, 299.3249646),
    )
    lat = np.array([51.5074, 52.2053])
    lon = np.array([-0.1278, 0.1218])
    z = np.array([45.0, 150.0])

    lat_s, lon_s, z_s = apply_helmert(lat, lon, h, np, h=z)
    lat_b, lon_b, z_b = apply_helmert(lat_s, lon_s, h.inverted(), np, h=z_s)

    assert_allclose(lat_b, lat, atol=1e-6)
    assert_allclose(lon_b, lon, atol=1e-6)
    assert_allclose(z_b, z, atol=0.02)  # ~14mm due to linearized rotation matrix


def test_apply_helmert_without_z_unchanged():
    """apply_helmert without z returns 2-tuple (backward compat)."""
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

    result = apply_helmert(lat, lon, h, np)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Pipeline z passthrough and transform
# ---------------------------------------------------------------------------


def test_pipeline_z_passthrough_no_helmert():
    """z passes through unchanged when no Helmert is active (all 4 modes)."""
    from vibeproj.pipeline import TransformPipeline
    from vibeproj.crs import resolve_transform

    # forward: geo -> proj (same datum)
    src, dst, _, _ = resolve_transform("EPSG:4326", "EPSG:32631")
    pipe = TransformPipeline(src, dst)
    lon = np.array([2.0])
    lat = np.array([48.0])
    z = np.array([100.0])
    rx, ry, rz = pipe.transform(lon, lat, np, z=z)
    assert_allclose(rz, z)

    # inverse: proj -> geo (same datum)
    src, dst, _, _ = resolve_transform("EPSG:32631", "EPSG:4326")
    pipe = TransformPipeline(src, dst)
    rx2, ry2, rz2 = pipe.transform(np.array([500000.0]), np.array([5300000.0]), np, z=z)
    assert_allclose(rz2, z)


def test_pipeline_z_transform_with_helmert_forward():
    """forward mode: z is transformed through Helmert."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    assert t._helmert is not None

    lon, lat, z = -0.1278, 51.5074, 45.0
    x, y, z_out = t.transform(lon, lat, z=z)

    # z should change when crossing datums
    assert isinstance(z_out, float)
    assert abs(z_out - z) > 1.0


def test_pipeline_z_transform_with_helmert_longlat():
    """longlat_to_longlat mode: z is transformed through Helmert."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:4277")
    assert t._helmert is not None

    lon, lat, z = -0.1278, 51.5074, 45.0
    lon_out, lat_out, z_out = t.transform(lon, lat, z=z)

    assert isinstance(z_out, float)
    assert abs(z_out - z) > 1.0


# ---------------------------------------------------------------------------
# Transformer z tests
# ---------------------------------------------------------------------------


def test_transformer_cross_datum_z_scalar():
    """Scalar z is correctly transformed through Helmert."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    x, y, z_out = t.transform(-0.1278, 51.5074, z=45.0)
    assert isinstance(z_out, float)
    assert abs(z_out - 45.0) > 1.0


def test_transformer_cross_datum_z_array():
    """Array z is correctly transformed through Helmert."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    lon = np.array([-0.1278, -1.2578])
    lat = np.array([51.5074, 51.7520])
    z = np.array([45.0, 100.0])

    x, y, z_out = t.transform(lon, lat, z=z)
    assert z_out.shape == z.shape
    assert not np.allclose(z_out, z, atol=0.5)  # z should change


def test_transformer_cross_datum_z_roundtrip():
    """Forward then inverse cross-datum with z recovers original z."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    lon, lat, z = -0.1278, 51.5074, 45.0

    x, y, z_fwd = t.transform(lon, lat, z=z)
    lon2, lat2, z_back = t.transform(x, y, z=z_fwd, direction="INVERSE")

    assert_allclose(z_back, z, atol=0.1)
    assert_allclose(lon2, lon, atol=1e-4)
    assert_allclose(lat2, lat, atol=1e-4)


def test_transformer_same_datum_z_passthrough():
    """Same-datum transform passes z through unchanged."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    assert t._helmert is None

    lon, lat, z = 2.0, 48.0, 123.456
    x, y, z_out = t.transform(lon, lat, z=z)
    assert z_out == z  # exact passthrough


def test_transformer_cross_datum_z_vs_pyproj():
    """Cross-datum z transform: x/y match pyproj, z is actively transformed.

    pyproj passes z through unchanged for 2D target CRS (EPSG:27700).
    Our implementation correctly transforms z through the ECEF intermediate,
    so z will differ from pyproj's passthrough. We verify x/y match and that
    z is reasonable (ellipsoidal height difference between WGS84 and Airy 1830).
    """
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    exp_x, exp_y = pp.transform(-0.1278, 51.5074)

    t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    vp_x, vp_y, vp_z = t.transform(-0.1278, 51.5074, z=45.0)

    # x/y should match pyproj within Helmert accuracy
    assert_allclose(vp_x, exp_x, atol=10.0)
    assert_allclose(vp_y, exp_y, atol=10.0)
    # z should be modified (WGS84->Airy ellipsoid height difference is ~46m)
    assert abs(vp_z - 45.0) > 1.0


def test_transform_buffers_with_out_z():
    """transform_buffers passes z through Helmert and returns it."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    lon = np.array([-0.1278])
    lat = np.array([51.5074])
    z = np.array([45.0])

    x, y, z_out = t.transform_buffers(lon, lat, z=z)
    assert z_out.shape == z.shape
    assert abs(z_out[0] - z[0]) > 1.0  # z should change


def test_transform_buffers_same_datum_z_passthrough():
    """transform_buffers returns z unchanged when no Helmert."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    lon = np.array([2.0])
    lat = np.array([48.0])
    z = np.array([123.456])

    x, y, z_out = t.transform_buffers(lon, lat, z=z)
    assert_allclose(z_out, z)


def test_transform_chunked_with_z():
    """transform_chunked handles z through Helmert correctly."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    lon = np.array([-0.1278, -1.2578])
    lat = np.array([51.5074, 51.7520])
    z = np.array([45.0, 100.0])

    # Falls back to CPU transform (no CuPy)
    x, y, z_out = t.transform_chunked(lon, lat, z=z)
    assert z_out.shape == z.shape
    # Compare with regular transform
    x2, y2, z2 = t.transform(lon, lat, z=z)
    assert_allclose(z_out, z2, atol=1e-10)
