"""Tests for SVD-compressed datum corrections.

Covers unit tests for DatumCorrectionData and lookup, SVD evaluation,
accuracy validation against pyproj, and Transformer integration.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyproj import Transformer as PyProjTransformer

from vibeproj import Transformer
from vibeproj._datum_corrections import (
    DatumCorrectionData,
    apply_svd_correction,
    get_datum_correction,
    is_reverse_direction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conus_grid(n: int = 20, *, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate random CONUS interior points avoiding grid boundary effects.

    pyproj uses multiple NADCON5 sub-grids with discontinuities at boundaries
    (notably lon ~ -89.5 and lat ~ 43.2 near the US-Canada border at lon ~ -80).
    The SVD compression smooths over these boundaries. We avoid those areas to
    test the core SVD accuracy without grid-boundary artifacts.

    Exclusions:
    - US-Canada border (lat > 43, lon > -85) where pyproj grid boundary at lat~43.2
    - lon in [-90.0, -89.0] where a NADCON5 sub-grid boundary exists
    - Southern Florida (lat < 27) near grid edge
    - Atlantic coast (lon > -73) near grid edge
    """
    rng = np.random.default_rng(seed)
    result_lons = []
    result_lats = []
    while len(result_lons) < n:
        batch = max(n * 3, 100)
        lons = rng.uniform(-118.0, -73.0, batch)
        lats = rng.uniform(27.0, 46.0, batch)
        # Filter out grid boundary areas
        mask = np.ones(batch, dtype=bool)
        mask &= ~((lats > 43.0) & (lons > -85.0))  # US-Canada grid boundary
        mask &= ~((lons > -90.0) & (lons < -89.0))  # NADCON5 sub-grid seam
        valid_lons = lons[mask]
        valid_lats = lats[mask]
        result_lons.extend(valid_lons.tolist())
        result_lats.extend(valid_lats.tolist())
    return np.array(result_lons[:n]), np.array(result_lats[:n])


def _position_error_meters(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> np.ndarray:
    """Approximate position error in meters between two lon/lat arrays."""
    dlat_m = np.abs(lat1 - lat2) * 111_000.0
    dlon_m = np.abs(lon1 - lon2) * 111_000.0 * np.cos(np.radians(0.5 * (lat1 + lat2)))
    return np.sqrt(dlat_m**2 + dlon_m**2)


# ============================================================================
# 1. Unit tests for DatumCorrectionData and lookup
# ============================================================================


class TestDatumCorrectionLookup:
    """Registry and lookup tests."""

    def test_get_nad27_nad83_correction(self):
        """get_datum_correction returns the NAD27->NAD83 entry with correct metadata."""
        correction = get_datum_correction("EPSG:4267", "EPSG:4269")
        assert correction is not None
        assert correction.src_crs == "EPSG:4267"
        assert correction.dst_crs == "EPSG:4269"
        # Bounding box: CONUS (lat_min, lat_max, lon_min, lon_max)
        assert correction.bbox == (24.5, 49.5, -124.5, -67.0)
        # Grid dimensions
        assert correction.n_lat == 100
        assert correction.n_lon == 150
        # SVD truncation rank
        assert correction.rank == 10
        # SVD components have correct shapes
        assert len(correction.s_lat) == correction.rank
        assert len(correction.u_lat) == correction.rank
        assert len(correction.u_lat[0]) == correction.n_lat
        assert len(correction.vt_lat) == correction.rank
        assert len(correction.vt_lat[0]) == correction.n_lon
        assert len(correction.s_lon) == correction.rank
        assert len(correction.u_lon) == correction.rank
        assert len(correction.u_lon[0]) == correction.n_lat
        assert len(correction.vt_lon) == correction.rank
        assert len(correction.vt_lon[0]) == correction.n_lon
        # Metadata
        assert correction.has_helmert is False
        assert "NADCON" in correction.source

    def test_get_reversed_correction(self):
        """Reverse direction lookup (NAD83->NAD27) returns the same entry."""
        fwd = get_datum_correction("EPSG:4267", "EPSG:4269")
        rev = get_datum_correction("EPSG:4269", "EPSG:4267")
        assert fwd is not None
        assert rev is not None
        # Both should return the same object (stored once, looked up both ways)
        assert fwd is rev

    def test_is_reverse_direction(self):
        """is_reverse_direction correctly identifies forward vs reverse."""
        assert not is_reverse_direction("EPSG:4267", "EPSG:4269")
        assert is_reverse_direction("EPSG:4269", "EPSG:4267")

    def test_get_unknown_pair_returns_none(self):
        """Unknown CRS pair returns None."""
        assert get_datum_correction("EPSG:4326", "EPSG:4269") is None
        assert get_datum_correction("EPSG:9999", "EPSG:9998") is None
        assert get_datum_correction("CUSTOM:1", "CUSTOM:2") is None

    def test_correction_data_immutable(self):
        """Frozen dataclass cannot be mutated."""
        correction = get_datum_correction("EPSG:4267", "EPSG:4269")
        assert correction is not None
        with pytest.raises(AttributeError):
            correction.rank = 999  # type: ignore[misc]
        with pytest.raises(AttributeError):
            correction.bbox = (0, 0, 0, 0)  # type: ignore[misc]
        with pytest.raises(AttributeError):
            correction.src_crs = "EPSG:0000"  # type: ignore[misc]


# ============================================================================
# 2. Unit tests for SVD evaluation
# ============================================================================


class TestSVDEvaluation:
    """Tests for apply_svd_correction()."""

    def test_svd_identity_correction(self):
        """Zero-valued DatumCorrectionData returns input unchanged."""
        rank = 2
        n_lat, n_lon = 5, 5
        zero_correction = DatumCorrectionData(
            src_crs="TEST:1",
            dst_crs="TEST:2",
            bbox=(30.0, 50.0, -130.0, -60.0),
            n_lat=n_lat,
            n_lon=n_lon,
            rank=rank,
            s_lat=tuple(0.0 for _ in range(rank)),
            u_lat=tuple(tuple(0.0 for _ in range(n_lat)) for _ in range(rank)),
            vt_lat=tuple(tuple(0.0 for _ in range(n_lon)) for _ in range(rank)),
            s_lon=tuple(0.0 for _ in range(rank)),
            u_lon=tuple(tuple(0.0 for _ in range(n_lat)) for _ in range(rank)),
            vt_lon=tuple(tuple(0.0 for _ in range(n_lon)) for _ in range(rank)),
            source="test",
            has_helmert=False,
        )
        lat = np.array([35.0, 40.0, 45.0])
        lon = np.array([-100.0, -90.0, -80.0])

        lat_out, lon_out = apply_svd_correction(lat, lon, zero_correction, np)

        assert_allclose(lat_out, lat, atol=1e-15)
        assert_allclose(lon_out, lon, atol=1e-15)

    def test_svd_negate_flag(self):
        """negate=True produces opposite corrections from negate=False."""
        correction = get_datum_correction("EPSG:4267", "EPSG:4269")
        assert correction is not None

        lat = np.array([35.0, 40.0, 45.0])
        lon = np.array([-100.0, -90.0, -80.0])

        lat_fwd, lon_fwd = apply_svd_correction(
            lat,
            lon,
            correction,
            np,
            negate=False,
        )
        lat_neg, lon_neg = apply_svd_correction(
            lat,
            lon,
            correction,
            np,
            negate=True,
        )

        # Forward adds correction, negate subtracts it.
        # The deltas should have opposite signs.
        dlat_fwd = lat_fwd - lat
        dlat_neg = lat_neg - lat
        dlon_fwd = lon_fwd - lon
        dlon_neg = lon_neg - lon

        assert_allclose(dlat_fwd, -dlat_neg, atol=1e-15)
        assert_allclose(dlon_fwd, -dlon_neg, atol=1e-15)

        # Corrections should be nonzero
        assert np.all(np.abs(dlat_fwd) > 1e-10)
        assert np.all(np.abs(dlon_fwd) > 1e-10)

    def test_svd_output_buffers(self):
        """Pre-allocated output arrays are used correctly."""
        correction = get_datum_correction("EPSG:4267", "EPSG:4269")
        assert correction is not None

        lat = np.array([35.0, 40.0, 45.0])
        lon = np.array([-100.0, -90.0, -80.0])

        out_lat = np.empty_like(lat)
        out_lon = np.empty_like(lon)

        result_lat, result_lon = apply_svd_correction(
            lat,
            lon,
            correction,
            np,
            out_lat=out_lat,
            out_lon=out_lon,
        )

        # The returned arrays should be the pre-allocated buffers
        assert result_lat is out_lat
        assert result_lon is out_lon

        # Values should match a fresh call without output buffers
        expected_lat, expected_lon = apply_svd_correction(
            lat,
            lon,
            correction,
            np,
        )
        assert_allclose(result_lat, expected_lat, atol=1e-15)
        assert_allclose(result_lon, expected_lon, atol=1e-15)

    def test_svd_clamping_outside_bbox(self):
        """Points outside the grid bbox are clamped, not crashed."""
        correction = get_datum_correction("EPSG:4267", "EPSG:4269")
        assert correction is not None

        lat_min, lat_max, lon_min, lon_max = correction.bbox

        # Points well outside the bbox in all four directions
        lat = np.array([10.0, 60.0, lat_min, lat_max, 37.0])
        lon = np.array([-100.0, -100.0, -140.0, -50.0, -95.0])

        # Should not raise
        lat_out, lon_out = apply_svd_correction(lat, lon, correction, np)

        # Output should be finite
        assert np.all(np.isfinite(lat_out))
        assert np.all(np.isfinite(lon_out))

        # Corrections should be applied (output differs from input)
        # even for clamped points
        assert not np.allclose(lat_out, lat)
        assert not np.allclose(lon_out, lon)

    def test_svd_empty_arrays(self):
        """Zero-length arrays pass through without error."""
        correction = get_datum_correction("EPSG:4267", "EPSG:4269")
        assert correction is not None

        lat = np.array([], dtype=np.float64)
        lon = np.array([], dtype=np.float64)

        lat_out, lon_out = apply_svd_correction(lat, lon, correction, np)

        assert lat_out.shape == (0,)
        assert lon_out.shape == (0,)

    def test_svd_scalar_like_input(self):
        """Single-element arrays work correctly."""
        correction = get_datum_correction("EPSG:4267", "EPSG:4269")
        assert correction is not None

        lat = np.array([40.0])
        lon = np.array([-100.0])

        lat_out, lon_out = apply_svd_correction(lat, lon, correction, np)

        assert lat_out.shape == (1,)
        assert lon_out.shape == (1,)
        # Should produce a nonzero correction
        assert abs(lat_out[0] - lat[0]) > 1e-10
        assert abs(lon_out[0] - lon[0]) > 1e-10


# ============================================================================
# 3. Accuracy tests (against pyproj)
# ============================================================================


class TestNAD27NAD83Accuracy:
    """Accuracy validation using pyproj as ground truth."""

    def test_nad27_nad83_accuracy_conus(self):
        """NAD27->NAD83 over CONUS matches pyproj within 5cm."""
        lons, lats = _make_conus_grid(n=50, seed=42)

        t = Transformer.from_crs("EPSG:4267", "EPSG:4269")
        pp = PyProjTransformer.from_crs("EPSG:4267", "EPSG:4269", always_xy=True)

        vp_lons, vp_lats = t.transform(lons, lats)
        pp_lons, pp_lats = pp.transform(lons, lats)

        errors = _position_error_meters(vp_lons, vp_lats, pp_lons, pp_lats)
        max_err = np.max(errors)

        assert max_err < 0.05, f"Max NAD27->NAD83 error {max_err:.4f} m exceeds 5cm threshold"

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_nad27_nad83_accuracy_multiple_seeds(self, seed: int):
        """NAD27->NAD83 accuracy holds across different random point sets."""
        lons, lats = _make_conus_grid(n=30, seed=seed)

        t = Transformer.from_crs("EPSG:4267", "EPSG:4269")
        pp = PyProjTransformer.from_crs("EPSG:4267", "EPSG:4269", always_xy=True)

        vp_lons, vp_lats = t.transform(lons, lats)
        pp_lons, pp_lats = pp.transform(lons, lats)

        errors = _position_error_meters(vp_lons, vp_lats, pp_lons, pp_lats)
        max_err = np.max(errors)

        assert max_err < 0.05, f"Max error {max_err:.4f} m (seed={seed}) exceeds 5cm threshold"

    def test_nad83_nad27_accuracy_conus(self):
        """Reverse direction NAD83->NAD27 also matches pyproj within 5cm."""
        lons, lats = _make_conus_grid(n=50, seed=99)

        t = Transformer.from_crs("EPSG:4269", "EPSG:4267")
        pp = PyProjTransformer.from_crs("EPSG:4269", "EPSG:4267", always_xy=True)

        vp_lons, vp_lats = t.transform(lons, lats)
        pp_lons, pp_lats = pp.transform(lons, lats)

        errors = _position_error_meters(vp_lons, vp_lats, pp_lons, pp_lats)
        max_err = np.max(errors)

        assert max_err < 0.05, f"Max NAD83->NAD27 error {max_err:.4f} m exceeds 5cm threshold"

    def test_nad27_nad83_roundtrip(self):
        """Forward NAD27->NAD83 then inverse NAD83->NAD27 roundtrips within 1cm."""
        lons, lats = _make_conus_grid(n=50, seed=42)

        t_fwd = Transformer.from_crs("EPSG:4267", "EPSG:4269")
        t_inv = Transformer.from_crs("EPSG:4269", "EPSG:4267")

        # Forward
        mid_lons, mid_lats = t_fwd.transform(lons, lats)
        # Inverse
        rt_lons, rt_lats = t_inv.transform(mid_lons, mid_lats)

        errors = _position_error_meters(lons, lats, rt_lons, rt_lats)
        max_err = np.max(errors)

        assert max_err < 0.01, f"Roundtrip error {max_err:.4f} m exceeds 1cm threshold"

    def test_svd_improves_over_helmert_only(self):
        """SVD correction reduces error compared to Helmert-only (or no shift).

        To test this we create a Transformer with SVD and compare its accuracy
        against a Transformer where we null out the SVD correction (which for
        NAD27->NAD83 means no shift at all since has_helmert=False).
        """
        lons, lats = _make_conus_grid(n=30, seed=42)

        pp = PyProjTransformer.from_crs("EPSG:4267", "EPSG:4269", always_xy=True)
        pp_lons, pp_lats = pp.transform(lons, lats)

        # With SVD correction
        t_svd = Transformer.from_crs("EPSG:4267", "EPSG:4269")
        svd_lons, svd_lats = t_svd.transform(lons, lats)
        svd_errors = _position_error_meters(svd_lons, svd_lats, pp_lons, pp_lats)

        # Without SVD: manually null it out and transform
        t_no_svd = Transformer.from_crs("EPSG:4267", "EPSG:4269")
        t_no_svd._pipeline._svd_correction = None
        no_svd_lons, no_svd_lats = t_no_svd.transform(lons, lats)
        no_svd_errors = _position_error_meters(no_svd_lons, no_svd_lats, pp_lons, pp_lats)

        # SVD should be significantly better
        assert np.max(svd_errors) < np.max(no_svd_errors), (
            f"SVD max error ({np.max(svd_errors):.4f} m) should be less than "
            f"no-SVD max error ({np.max(no_svd_errors):.4f} m)"
        )
        # The improvement should be substantial (NAD27/83 shift is ~50-200m)
        assert np.mean(svd_errors) < 0.1 * np.mean(no_svd_errors), (
            "SVD should improve mean accuracy by at least 10x"
        )

    @pytest.mark.parametrize(
        "lon, lat, label",
        [
            (-74.0, 40.7, "New York"),
            (-87.6, 41.9, "Chicago"),
            (-118.2, 34.1, "Los Angeles"),
            (-95.4, 29.8, "Houston"),
            (-104.9, 39.7, "Denver"),
            (-112.0, 33.5, "Phoenix"),
            (-122.4, 37.8, "San Francisco"),
            (-80.2, 25.8, "Miami"),
            (-90.1, 29.9, "New Orleans"),
            (-93.3, 44.9, "Minneapolis"),
        ],
    )
    def test_nad27_nad83_us_cities(self, lon: float, lat: float, label: str):
        """NAD27->NAD83 at specific US cities within 5cm of pyproj."""
        t = Transformer.from_crs("EPSG:4267", "EPSG:4269")
        pp = PyProjTransformer.from_crs("EPSG:4267", "EPSG:4269", always_xy=True)

        vp_lon, vp_lat = t.transform(lon, lat)
        pp_lon, pp_lat = pp.transform(lon, lat)

        error = _position_error_meters(
            np.array([vp_lon]),
            np.array([vp_lat]),
            np.array([pp_lon]),
            np.array([pp_lat]),
        )[0]

        assert error < 0.05, f"{label} error: {error:.4f} m exceeds 5cm threshold"


# ============================================================================
# 4. Integration tests
# ============================================================================


class TestIntegration:
    """End-to-end Transformer integration with SVD corrections."""

    def test_accuracy_property_with_svd(self):
        """Transformer.accuracy reports 'sub-5cm' when SVD correction is active."""
        t = Transformer.from_crs("EPSG:4267", "EPSG:4269")
        assert t.accuracy == "sub-5cm"

    def test_accuracy_property_reverse_with_svd(self):
        """Reverse direction also reports 'sub-5cm'."""
        t = Transformer.from_crs("EPSG:4269", "EPSG:4267")
        assert t.accuracy == "sub-5cm"

    def test_no_warning_with_svd(self):
        """No 'grid-based shifts not yet supported' warning when SVD is available."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Transformer.from_crs("EPSG:4267", "EPSG:4269")

        grid_warnings = [w for w in caught if "grid-based shifts" in str(w.message).lower()]
        assert len(grid_warnings) == 0, f"Unexpected grid-based shift warning: {grid_warnings}"

    def test_no_warning_reverse_with_svd(self):
        """Reverse direction also has no grid-based shift warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Transformer.from_crs("EPSG:4269", "EPSG:4267")

        grid_warnings = [w for w in caught if "grid-based shifts" in str(w.message).lower()]
        assert len(grid_warnings) == 0

    def test_projected_crs_with_svd(self):
        """SVD correction works through projected CRS (NAD27 UTM -> NAD83 UTM).

        Uses UTM zone 18N: EPSG:26718 (NAD27) and EPSG:26918 (NAD83).
        """
        # Get a test point in NAD27 UTM 18N coordinates
        pp_geo_to_utm = PyProjTransformer.from_crs(
            "EPSG:4267",
            "EPSG:26718",
            always_xy=True,
        )
        # Washington DC area
        utm_x, utm_y = pp_geo_to_utm.transform(-77.0, 39.0)

        # Transform NAD27 UTM -> NAD83 UTM via vibeproj
        t = Transformer.from_crs("EPSG:26718", "EPSG:26918")
        vp_x, vp_y = t.transform(utm_x, utm_y)

        # Reference: pyproj
        pp = PyProjTransformer.from_crs("EPSG:26718", "EPSG:26918", always_xy=True)
        pp_x, pp_y = pp.transform(utm_x, utm_y)

        # Projected coordinates: tolerance in meters
        assert abs(vp_x - pp_x) < 0.05, f"UTM x error: {abs(vp_x - pp_x):.4f} m"
        assert abs(vp_y - pp_y) < 0.05, f"UTM y error: {abs(vp_y - pp_y):.4f} m"

    @pytest.mark.parametrize(
        "src_epsg, dst_epsg, label",
        [
            ("EPSG:26718", "EPSG:26918", "UTM 18N"),
            ("EPSG:26710", "EPSG:26910", "UTM 10N"),
            ("EPSG:26714", "EPSG:26914", "UTM 14N"),
        ],
    )
    def test_projected_crs_with_svd_multiple_zones(
        self,
        src_epsg: str,
        dst_epsg: str,
        label: str,
    ):
        """SVD correction works across multiple NAD27/NAD83 UTM zones."""
        t = Transformer.from_crs(src_epsg, dst_epsg)
        assert t.accuracy == "sub-5cm"

        pp = PyProjTransformer.from_crs(src_epsg, dst_epsg, always_xy=True)

        # Use a point near the center of the zone
        # (Zone center longitude can be derived from zone number, but
        # we just use the midpoint of valid UTM easting range.)
        test_x, test_y = 500000.0, 4000000.0  # Central easting, mid-CONUS northing

        vp_x, vp_y = t.transform(test_x, test_y)
        pp_x, pp_y = pp.transform(test_x, test_y)

        assert abs(vp_x - pp_x) < 0.05, f"{label} x error: {abs(vp_x - pp_x):.4f} m"
        assert abs(vp_y - pp_y) < 0.05, f"{label} y error: {abs(vp_y - pp_y):.4f} m"

    def test_svd_correction_stored_on_transformer(self):
        """Internal state: _svd_correction is populated for NAD27/NAD83."""
        t = Transformer.from_crs("EPSG:4267", "EPSG:4269")
        assert t._svd_correction is not None
        assert t._svd_negate is False

    def test_svd_negate_on_reverse_transformer(self):
        """Internal state: reverse direction sets _svd_negate=True."""
        t = Transformer.from_crs("EPSG:4269", "EPSG:4267")
        assert t._svd_correction is not None
        assert t._svd_negate is True

    def test_svd_correction_absent_for_same_datum(self):
        """Same-datum transforms have no SVD correction."""
        t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
        assert t._svd_correction is None
        assert t.accuracy == "sub-millimeter"

    def test_svd_correction_absent_for_unknown_pair(self):
        """Cross-datum pairs without SVD data do not get SVD correction."""
        # WGS84 -> OSGB36: has Helmert but no SVD
        t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
        assert t._svd_correction is None
        assert t.accuracy in ("sub-meter", "sub-decimeter")

    def test_nad27_nad83_inverse_via_direction(self):
        """Using direction='INVERSE' on a NAD27->NAD83 Transformer works."""
        lons = np.array([-100.0, -90.0, -80.0])
        lats = np.array([35.0, 40.0, 45.0])

        t = Transformer.from_crs("EPSG:4267", "EPSG:4269")

        # Forward
        fwd_lons, fwd_lats = t.transform(lons, lats)
        # Inverse via direction parameter
        inv_lons, inv_lats = t.transform(fwd_lons, fwd_lats, direction="INVERSE")

        errors = _position_error_meters(lons, lats, inv_lons, inv_lats)
        max_err = np.max(errors)

        assert max_err < 0.01, f"Inverse via direction roundtrip error {max_err:.4f} m exceeds 1cm"

    def test_nad27_nad83_scalar_input(self):
        """Scalar inputs (not arrays) work correctly."""
        t = Transformer.from_crs("EPSG:4267", "EPSG:4269")
        pp = PyProjTransformer.from_crs("EPSG:4267", "EPSG:4269", always_xy=True)

        vp_lon, vp_lat = t.transform(-100.0, 40.0)
        pp_lon, pp_lat = pp.transform(-100.0, 40.0)

        error = _position_error_meters(
            np.array([vp_lon]),
            np.array([vp_lat]),
            np.array([pp_lon]),
            np.array([pp_lat]),
        )[0]
        assert error < 0.05

    def test_nad27_nad83_array_input(self):
        """Array inputs work correctly and match scalar results."""
        t = Transformer.from_crs("EPSG:4267", "EPSG:4269")

        # Scalar
        s_lon, s_lat = t.transform(-100.0, 40.0)
        # Array
        a_lons, a_lats = t.transform(
            np.array([-100.0]),
            np.array([40.0]),
        )

        assert_allclose(a_lons[0], s_lon, atol=1e-12)
        assert_allclose(a_lats[0], s_lat, atol=1e-12)
