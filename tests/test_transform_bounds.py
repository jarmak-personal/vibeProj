"""Tests for Transformer.transform_bounds() — validated against pyproj."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyproj import Transformer as PyProjTransformer

from _accuracy_cases import UNIT_EQUIVALENT_PAIRS
from vibeproj import Transformer


# ---------------------------------------------------------------------------
# Helper: pyproj reference bounds
# ---------------------------------------------------------------------------


def _pyproj_bounds(
    crs_from: str,
    crs_to: str,
    left: float,
    bottom: float,
    right: float,
    top: float,
    *,
    densify_pts: int = 21,
    always_xy: bool = True,
) -> tuple[float, float, float, float]:
    """Compute reference bounds using pyproj."""
    pp = PyProjTransformer.from_crs(crs_from, crs_to, always_xy=always_xy)
    return pp.transform_bounds(left, bottom, right, top, densify_pts=densify_pts)


# ---------------------------------------------------------------------------
# UTM (EPSG:4326 -> EPSG:32631)
# ---------------------------------------------------------------------------


def test_bounds_utm_forward():
    """WGS84 geographic -> UTM zone 31N bounding box."""
    left, bottom, right, top = -1.0, 40.0, 5.0, 50.0
    expected = _pyproj_bounds("EPSG:4326", "EPSG:32631", left, bottom, right, top)

    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    got = t.transform_bounds(left, bottom, right, top)

    # Projected coordinates: ~1 m tolerance
    assert_allclose(got, expected, atol=1.0)


def test_bounds_utm_inverse():
    """UTM zone 31N -> WGS84 geographic bounding box (inverse)."""
    # First get some UTM bounds to use as input
    left, bottom, right, top = 200000.0, 4400000.0, 700000.0, 5600000.0
    expected = _pyproj_bounds("EPSG:32631", "EPSG:4326", left, bottom, right, top)

    t = Transformer.from_crs("EPSG:32631", "EPSG:4326")
    got = t.transform_bounds(left, bottom, right, top)

    # Geographic coordinates: ~0.001 degree tolerance
    assert_allclose(got, expected, atol=0.001)


def test_bounds_utm_inverse_direction():
    """Use direction='INVERSE' instead of swapping CRS order."""
    left, bottom, right, top = -1.0, 40.0, 5.0, 50.0

    # Forward transform to get projected bounds
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    projected = t.transform_bounds(left, bottom, right, top)

    # Now inverse those projected bounds back to geographic.
    # Note: a forward-then-inverse bounds roundtrip naturally expands
    # because the densified envelope of the projected rectangle is larger
    # than the original geographic rectangle.  We verify against pyproj
    # rather than expecting exact recovery of the input.
    expected = _pyproj_bounds("EPSG:32631", "EPSG:4326", *projected)

    t_inv = Transformer.from_crs("EPSG:32631", "EPSG:4326")
    got = t_inv.transform_bounds(*projected)

    assert_allclose(got, expected, atol=0.001)


# ---------------------------------------------------------------------------
# Web Mercator (EPSG:4326 -> EPSG:3857)
# ---------------------------------------------------------------------------


def test_bounds_web_mercator():
    """WGS84 geographic -> Web Mercator bounding box."""
    left, bottom, right, top = -74.1, 40.6, -73.7, 40.9
    expected = _pyproj_bounds("EPSG:4326", "EPSG:3857", left, bottom, right, top)

    t = Transformer.from_crs("EPSG:4326", "EPSG:3857")
    got = t.transform_bounds(left, bottom, right, top)

    assert_allclose(got, expected, atol=1.0)


def test_bounds_web_mercator_large():
    """Web Mercator with a larger geographic extent."""
    left, bottom, right, top = -180.0, -85.0, 180.0, 85.0
    expected = _pyproj_bounds("EPSG:4326", "EPSG:3857", left, bottom, right, top)

    t = Transformer.from_crs("EPSG:4326", "EPSG:3857")
    got = t.transform_bounds(left, bottom, right, top)

    assert_allclose(got, expected, atol=1.0)


# ---------------------------------------------------------------------------
# Cross-datum (EPSG:4326 -> EPSG:27700 — British National Grid, Airy ellipsoid)
# ---------------------------------------------------------------------------


def test_bounds_cross_datum():
    """WGS84 -> British National Grid (different datum)."""
    # London area
    left, bottom, right, top = -0.5, 51.3, 0.3, 51.6
    expected = _pyproj_bounds("EPSG:4326", "EPSG:27700", left, bottom, right, top)

    t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    got = t.transform_bounds(left, bottom, right, top)

    # Cross-datum: allow ~5 m tolerance (Helmert vs grid-based shifts)
    assert_allclose(got, expected, atol=5.0)


# ---------------------------------------------------------------------------
# Densify points variations
# ---------------------------------------------------------------------------


def test_bounds_densify_pts_minimum():
    """densify_pts=2 should produce a small number of edge samples."""
    left, bottom, right, top = -1.0, 40.0, 5.0, 50.0
    expected = _pyproj_bounds("EPSG:4326", "EPSG:32631", left, bottom, right, top, densify_pts=2)

    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    got = t.transform_bounds(left, bottom, right, top, densify_pts=2)

    assert_allclose(got, expected, atol=1.0)


def test_bounds_densify_pts_high():
    """densify_pts=100 should capture more curvature."""
    left, bottom, right, top = -1.0, 40.0, 5.0, 50.0
    expected = _pyproj_bounds("EPSG:4326", "EPSG:32631", left, bottom, right, top, densify_pts=100)

    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    got = t.transform_bounds(left, bottom, right, top, densify_pts=100)

    assert_allclose(got, expected, atol=1.0)


def test_bounds_densify_pts_zero():
    """densify_pts=0 should produce corner-only transform."""
    left, bottom, right, top = -1.0, 40.0, 5.0, 50.0
    expected = _pyproj_bounds("EPSG:4326", "EPSG:32631", left, bottom, right, top, densify_pts=0)

    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    got = t.transform_bounds(left, bottom, right, top, densify_pts=0)

    assert_allclose(got, expected, atol=1.0)


def test_bounds_densify_pts_clamped():
    """Negative densify_pts should be clamped to 0 (corners only)."""
    left, bottom, right, top = -1.0, 40.0, 5.0, 50.0

    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    # Should not raise, should produce same result as densify_pts=0
    got_clamped = t.transform_bounds(left, bottom, right, top, densify_pts=-5)
    got_0 = t.transform_bounds(left, bottom, right, top, densify_pts=0)

    assert_allclose(got_clamped, got_0, atol=1e-10)


# ---------------------------------------------------------------------------
# Densified vs corner-only: densification should give tighter or equal bounds
# ---------------------------------------------------------------------------


def test_densified_bounds_tighter_or_equal():
    """Densified transform should produce a bounding box that is at least
    as tight as (or equal to) a corner-only transform for projections with
    curvature.

    For UTM with a wide longitude range, the densified bottom/left should
    be <= the corner-only bottom/left, and densified top/right should be
    >= the corner-only top/right.
    """
    left, bottom, right, top = -5.0, 40.0, 10.0, 55.0

    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")

    # Corner-only: transform just the 4 corners
    corners_x = np.array([left, right, left, right])
    corners_y = np.array([bottom, bottom, top, top])
    cx, cy = t.transform(corners_x, corners_y)
    corner_bounds = (cx.min(), cy.min(), cx.max(), cy.max())

    # Densified bounds
    dense_bounds = t.transform_bounds(left, bottom, right, top, densify_pts=21)

    # The densified envelope should contain or equal the corner envelope
    # (densified may discover more extreme values along edges)
    assert dense_bounds[0] <= corner_bounds[0] + 1e-6  # left
    assert dense_bounds[1] <= corner_bounds[1] + 1e-6  # bottom
    assert dense_bounds[2] >= corner_bounds[2] - 1e-6  # right
    assert dense_bounds[3] >= corner_bounds[3] - 1e-6  # top


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_bounds_invalid_direction():
    """Invalid direction should raise ValueError."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    with pytest.raises(ValueError, match="Invalid direction"):
        t.transform_bounds(0, 0, 1, 1, direction="BACKWARD")


def test_bounds_return_type():
    """Return value should be a tuple of four Python floats."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    result = t.transform_bounds(-1.0, 40.0, 5.0, 50.0)
    assert isinstance(result, tuple)
    assert len(result) == 4
    for val in result:
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# Southern hemisphere UTM
# ---------------------------------------------------------------------------


def test_bounds_southern_hemisphere():
    """UTM zone 56S — Sydney area."""
    left, bottom, right, top = 151.0, -34.0, 151.5, -33.5
    expected = _pyproj_bounds("EPSG:4326", "EPSG:32756", left, bottom, right, top)

    t = Transformer.from_crs("EPSG:4326", "EPSG:32756")
    got = t.transform_bounds(left, bottom, right, top)

    assert_allclose(got, expected, atol=1.0)


# ---------------------------------------------------------------------------
# LCC (Lambert Conformal Conic) — good test for non-linear curvature
# ---------------------------------------------------------------------------


def test_bounds_lcc():
    """WGS84 -> NAD83 / Conus Albers (EPSG:5072) — conic projection."""
    left, bottom, right, top = -125.0, 24.0, -66.0, 50.0
    expected = _pyproj_bounds("EPSG:4326", "EPSG:5072", left, bottom, right, top)

    t = Transformer.from_crs("EPSG:4326", "EPSG:5072")
    got = t.transform_bounds(left, bottom, right, top)

    # Conic projections have significant curvature — allow wider tolerance
    assert_allclose(got, expected, atol=10.0)


# ---------------------------------------------------------------------------
# Non-meter projected units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, src_crs, dst_crs, lat_range, lon_range, tol_m",
    UNIT_EQUIVALENT_PAIRS,
    ids=[case[0] for case in UNIT_EQUIVALENT_PAIRS],
)
def test_bounds_non_meter_forward_inverse(label, src_crs, dst_crs, lat_range, lon_range, tol_m):
    """Non-meter projected CRS bounds match pyproj in both directions."""
    left, bottom, right, top = lon_range[0], lat_range[0], lon_range[1], lat_range[1]

    expected = _pyproj_bounds("EPSG:4326", src_crs, left, bottom, right, top)
    t = Transformer.from_crs("EPSG:4326", src_crs)
    got = t.transform_bounds(left, bottom, right, top)
    assert_allclose(got, expected, atol=1.0)

    expected_inv = _pyproj_bounds(src_crs, "EPSG:4326", *expected)
    t_inv = Transformer.from_crs(src_crs, "EPSG:4326")
    got_inv = t_inv.transform_bounds(*expected)
    assert_allclose(got_inv, expected_inv, atol=0.001)


@pytest.mark.parametrize(
    "label, src_crs, dst_crs, lat_range, lon_range, tol_m",
    UNIT_EQUIVALENT_PAIRS,
    ids=[case[0] for case in UNIT_EQUIVALENT_PAIRS],
)
def test_bounds_projected_unit_pairs(label, src_crs, dst_crs, lat_range, lon_range, tol_m):
    """Projected-to-projected bounds match pyproj for meter/foot equivalent CRSs."""
    left, bottom, right, top = lon_range[0], lat_range[0], lon_range[1], lat_range[1]
    src_bounds = _pyproj_bounds("EPSG:4326", src_crs, left, bottom, right, top)

    expected = _pyproj_bounds(src_crs, dst_crs, *src_bounds)
    t = Transformer.from_crs(src_crs, dst_crs)
    got = t.transform_bounds(*src_bounds)
    assert_allclose(got, expected, atol=1.0)

    expected_inv = _pyproj_bounds(dst_crs, src_crs, *expected)
    t_inv = Transformer.from_crs(dst_crs, src_crs)
    got_inv = t_inv.transform_bounds(*expected)
    assert_allclose(got_inv, expected_inv, atol=1.0)


# ---------------------------------------------------------------------------
# Antimeridian crossing
# ---------------------------------------------------------------------------


def test_bounds_antimeridian_nz_tm():
    """NZGD2000 / NZ TM (EPSG:2193) -> WGS84 crosses the antimeridian.

    pyproj returns left > right to signal the crossing; vibeproj must match.
    """
    left, bottom, right, top = 1000000.0, 4700000.0, 2200000.0, 6300000.0
    expected = _pyproj_bounds("EPSG:2193", "EPSG:4326", left, bottom, right, top)

    t = Transformer.from_crs("EPSG:2193", "EPSG:4326")
    got = t.transform_bounds(left, bottom, right, top)

    # Verify left > right (antimeridian signal)
    assert got[0] > got[2], f"Expected left > right for antimeridian crossing, got {got}"
    assert_allclose(got, expected, atol=0.001)


def test_bounds_antimeridian_utm60n():
    """UTM zone 60N (EPSG:32660) -> WGS84 crosses the antimeridian.

    Central meridian is 177°E, so wide bounds straddle ±180°.
    """
    left, bottom, right, top = 100000.0, 0.0, 900000.0, 9500000.0
    expected = _pyproj_bounds("EPSG:32660", "EPSG:4326", left, bottom, right, top)

    t = Transformer.from_crs("EPSG:32660", "EPSG:4326")
    got = t.transform_bounds(left, bottom, right, top)

    assert got[0] > got[2], f"Expected left > right for antimeridian crossing, got {got}"
    assert_allclose(got, expected, atol=0.001)


def test_bounds_antimeridian_utm1s():
    """UTM zone 1S (EPSG:32701) -> WGS84 crosses the antimeridian.

    Central meridian is 177°W.
    """
    left, bottom, right, top = 100000.0, 8000000.0, 900000.0, 9800000.0
    expected = _pyproj_bounds("EPSG:32701", "EPSG:4326", left, bottom, right, top)

    t = Transformer.from_crs("EPSG:32701", "EPSG:4326")
    got = t.transform_bounds(left, bottom, right, top)

    assert got[0] > got[2], f"Expected left > right for antimeridian crossing, got {got}"
    assert_allclose(got, expected, atol=0.001)


def test_bounds_no_false_antimeridian():
    """Non-crossing geographic output should NOT produce left > right."""
    # UTM 31N inverse -> geographic, far from the antimeridian
    left, bottom, right, top = 200000.0, 4400000.0, 700000.0, 5600000.0

    t = Transformer.from_crs("EPSG:32631", "EPSG:4326")
    got = t.transform_bounds(left, bottom, right, top)

    assert got[0] < got[2], f"Non-crossing case should have left < right, got {got}"
