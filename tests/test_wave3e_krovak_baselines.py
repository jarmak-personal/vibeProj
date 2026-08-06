"""Wave 3E correctness and domain baselines for native Krovak."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pyproj import CRS
from pyproj import Transformer as PyProjTransformer

from vibeproj import Transformer
from vibeproj.crs import resolve_projection_params
from vibeproj.exceptions import UnsupportedProjectionError
from vibeproj.transcendentals import (
    projection_strategy_domain,
    projection_strategy_domains,
)


REGULAR_KROVAK_CODES = (2065, 5513, 8352)
NORTH_ORIENTED_KROVAK_CODES = (5221, 5514, 8353)
SUPPORTED_KROVAK_CODES = REGULAR_KROVAK_CODES + NORTH_ORIENTED_KROVAK_CODES
MODIFIED_KROVAK_CODES = (5224, 5225, 5515, 5516)

BESSEL_A = 6_377_397.155
BESSEL_ES = 0.006674372231802145
BESSEL_E = 0.08169683122252751
STANDARD_ALPHA_C = 30.28813975277778
STANDARD_PHI_P = 78.5


def _cupy_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device not available")
    except Exception as exc:
        pytest.skip(f"CUDA device not available: {exc}")
    return cp


def _same_datum_geographic_points(target: CRS) -> tuple[np.ndarray, np.ndarray]:
    """Return Czech/Slovak longitudes in the target geodetic prime meridian."""
    greenwich_longitude = np.array([12.1, 14.5, 16.8, 18.9])
    longitude = greenwich_longitude - target.geodetic_crs.prime_meridian.longitude
    latitude = np.array([48.4, 49.5, 50.2, 51.1])
    return longitude, latitude


@pytest.mark.parametrize("epsg", SUPPORTED_KROVAK_CODES)
def test_supported_krovak_crss_pin_bessel_setup_and_launch_uniform_scalars(epsg):
    target = CRS.from_epsg(epsg)
    params = resolve_projection_params(target)
    transformer = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)
    computed = transformer._pipeline.computed
    regular = epsg in REGULAR_KROVAK_CODES

    assert params.operation_method == ("Krovak" if regular else "Krovak (North Orientated)")
    assert params.north_first is regular
    assert params.visualization_north_first is regular
    assert params.easting_axis_sign == (-1.0 if regular else 1.0)
    assert params.northing_axis_sign == (-1.0 if regular else 1.0)
    assert params.ellipsoid.a == pytest.approx(BESSEL_A, rel=0.0, abs=1e-9)
    assert params.ellipsoid.es == pytest.approx(BESSEL_ES, rel=0.0, abs=2e-18)
    assert params.ellipsoid.e == pytest.approx(BESSEL_E, rel=0.0, abs=2e-16)
    assert params.lat_0 == pytest.approx(49.5, rel=0.0, abs=0.0)
    assert params.k_0 == pytest.approx(0.9999, rel=0.0, abs=0.0)
    assert params.extra["alpha_c"] == pytest.approx(STANDARD_ALPHA_C, rel=0.0, abs=5e-14)
    assert params.extra["phi_p"] == pytest.approx(STANDARD_PHI_P, rel=0.0, abs=0.0)

    expected_scalars = {
        "a": BESSEL_A,
        "e": BESSEL_E,
        "B": 1.0005974983716484,
        "k": 1.0034191639671806,
        "n": 0.9799247046208296,
        "r_0_norm": 0.20353742648834952,
        "tan_half_p": 9.931008767325844,
        "sin_alpha_c": 0.5043488898136432,
        "cos_alpha_c": 0.8634999695099853,
    }
    for name, expected in expected_scalars.items():
        assert math.isfinite(computed[name])
        assert computed[name] == pytest.approx(expected, rel=2e-15, abs=2e-15)
    assert computed["B"] > 0.0
    assert computed["k"] > 0.0
    assert computed["n"] > 0.0
    assert computed["r_0_norm"] > 0.0
    assert computed["tan_half_p"] > 0.0
    assert math.isfinite(computed["lam0"])
    assert math.isfinite(computed["x0"])
    assert math.isfinite(computed["y0"])
    assert computed["cgb"] == pytest.approx(
        (
            0.003346491640862642,
            6.53254012663727e-06,
            1.7488220271061576e-08,
            5.3234553990234705e-11,
            1.737731799565109e-13,
            5.947923104361088e-16,
        ),
        rel=2e-15,
        abs=2e-15,
    )
    assert computed["log_k"] == pytest.approx(0.0034133319161083263, rel=2e-15, abs=2e-15)

    orientation = "regular" if regular else "north_oriented"
    semantic_domain = f"ellipsoidal.standard_bessel.{orientation}"
    assert computed["_strategy_krovak_semantics"] == f"standard_bessel.{orientation}"
    assert projection_strategy_domain("krovak", "forward", computed) == (
        f"krovak.forward.{semantic_domain}"
    )
    assert projection_strategy_domain("krovak", "inverse", computed) == (
        f"krovak.inverse.{semantic_domain}"
    )


@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_krovak_warmup_domains_cover_exact_setup_semantics(direction):
    assert projection_strategy_domains("krovak", direction) == (
        f"krovak.{direction}.ellipsoidal.standard_bessel.regular",
        f"krovak.{direction}.ellipsoidal.standard_bessel.north_oriented",
        f"krovak.{direction}.ellipsoidal.custom",
        f"krovak.{direction}.ellipsoidal.invalid_setup",
        f"krovak.{direction}.spherical",
        f"krovak.{direction}.unspecified",
    )


@pytest.mark.parametrize("epsg", SUPPORTED_KROVAK_CODES)
@pytest.mark.parametrize("always_xy", [False, True])
def test_all_regular_krovak_epsg_variants_match_same_datum_proj_and_roundtrip(epsg, always_xy):
    target = CRS.from_epsg(epsg)
    source = target.geodetic_crs
    longitude, latitude = _same_datum_geographic_points(target)
    first, second = (longitude, latitude) if always_xy else (latitude, longitude)
    expected_forward = PyProjTransformer.from_crs(source, target, always_xy=always_xy)
    expected_inverse = PyProjTransformer.from_crs(target, source, always_xy=always_xy)
    actual = Transformer.from_crs(source, target, always_xy=always_xy)

    expected_x, expected_y = expected_forward.transform(first, second)
    actual_x, actual_y = actual.transform(first, second)
    assert_allclose(actual_x, expected_x, rtol=0.0, atol=2e-6)
    assert_allclose(actual_y, expected_y, rtol=0.0, atol=2e-6)

    expected_first, expected_second = expected_inverse.transform(expected_x, expected_y)
    actual_first, actual_second = actual.transform(actual_x, actual_y, direction="INVERSE")
    assert_allclose(actual_first, expected_first, rtol=0.0, atol=1e-11)
    assert_allclose(actual_second, expected_second, rtol=0.0, atol=1e-11)
    assert_allclose(actual_first, first, rtol=0.0, atol=1e-11)
    assert_allclose(actual_second, second, rtol=0.0, atol=1e-11)


@pytest.mark.parametrize("epsg", MODIFIED_KROVAK_CODES)
def test_modified_krovak_methods_remain_explicitly_unsupported(epsg):
    target = CRS.from_epsg(epsg)
    with pytest.raises(UnsupportedProjectionError, match="Krovak Modified"):
        Transformer.from_crs(target.geodetic_crs, target, always_xy=True)


def test_custom_bessel_setup_with_standard_cone_axis_matches_proj_pipeline():
    definition = (
        "+proj=krovak +lat_0=40 +lon_0=10 +alpha=30.28813975277778 +k=.9 +ellps=bessel +units=m"
    )
    target = CRS.from_user_input(f"{definition} +type=crs")
    actual = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)
    expected = PyProjTransformer.from_pipeline(definition)
    longitude = np.array([5.0, 10.0, 15.0])
    latitude = np.array([35.0, 40.0, 50.0])

    actual_x, actual_y = actual.transform(longitude, latitude)
    expected_x, expected_y = expected.transform(longitude, latitude)
    assert_allclose(actual_x, expected_x, rtol=0.0, atol=2e-6)
    assert_allclose(actual_y, expected_y, rtol=0.0, atol=2e-6)
    got_lon, got_lat = actual.transform(actual_x, actual_y, direction="INVERSE")
    assert_allclose(got_lon, longitude, rtol=0.0, atol=2e-12)
    assert_allclose(got_lat, latitude, rtol=0.0, atol=2e-12)


@pytest.mark.parametrize(
    ("earth", "minimum_disagreement_m"),
    [("+ellps=WGS84", 100.0), ("+R=6371000", 2_000.0)],
)
def test_non_bessel_custom_krovak_is_a_documented_proj_compatibility_gap(
    earth, minimum_disagreement_m
):
    definition = (
        "+proj=krovak +lat_0=49.5 +lon_0=24.83333333333333 "
        "+alpha=30.28813975277778 +k=.9999 "
        f"{earth} +units=m"
    )
    target = CRS.from_user_input(f"{definition} +type=crs")
    actual = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)
    expected = PyProjTransformer.from_pipeline(definition)
    longitude = np.array([13.0, 14.0, 17.5])
    latitude = np.array([49.0, 50.0, 50.7])

    actual_x, actual_y = actual.transform(longitude, latitude)
    expected_x, expected_y = expected.transform(longitude, latitude)
    disagreement = np.hypot(actual_x - expected_x, actual_y - expected_y)
    assert float(np.max(disagreement)) > minimum_disagreement_m

    # The declared-geometry implementation is internally symmetric, but PROJ's
    # Krovak operation remains Bessel-defined. This excludes non-Bessel setup
    # from any PROJ-compatible accelerated qualification until validation or
    # projection semantics are changed deliberately.
    got_lon, got_lat = actual.transform(actual_x, actual_y, direction="INVERSE")
    assert_allclose(got_lon, longitude, rtol=0.0, atol=2e-12)
    assert_allclose(got_lat, latitude, rtol=0.0, atol=2e-12)


def test_nonstandard_cone_axis_is_a_documented_proj_compatibility_gap():
    definition = (
        "+proj=krovak +lat_0=49.5 +lon_0=24.83333333333333 "
        "+alpha=20 +k=.9999 +ellps=bessel +units=m"
    )
    target = CRS.from_user_input(f"{definition} +type=crs")
    actual = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)
    expected = PyProjTransformer.from_pipeline(definition)
    longitude = np.array([13.0, 14.0, 17.5])
    latitude = np.array([49.0, 50.0, 50.7])
    actual_xy = actual.transform(longitude, latitude)
    expected_xy = expected.transform(longitude, latitude)
    disagreement = np.hypot(actual_xy[0] - expected_xy[0], actual_xy[1] - expected_xy[1])
    assert float(np.max(disagreement)) > 1_000_000.0


def test_forward_poles_outside_latitudes_and_fused_native_classes_are_stable():
    cp = _cupy_or_skip()
    target = CRS.from_epsg(5514)
    source = target.geodetic_crs
    actual = Transformer.from_crs(source, target, always_xy=True)
    expected = PyProjTransformer.from_crs(source, target, always_xy=True)
    longitude = np.full(7, 14.0)
    latitude = np.array([-100.0, -90.0, -89.999999, 0.0, 89.999999, 90.0, 100.0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cpu = actual.transform(longitude, latitude, transcendentals="native")
    proj = expected.transform(longitude, latitude)
    gpu = actual.transform_buffers(
        cp.asarray(longitude),
        cp.asarray(latitude),
        precision="fp64",
        transcendentals="native",
    )
    gpu_host = cp.asnumpy(gpu[0]), cp.asnumpy(gpu[1])

    valid = np.array([False, True, True, True, True, True, False])
    assert_allclose(cpu[0][valid], proj[0][valid], rtol=0.0, atol=6e-6)
    assert_allclose(cpu[1][valid], proj[1][valid], rtol=0.0, atol=6e-6)
    assert np.isnan(cpu[0][~valid]).all() and np.isnan(cpu[1][~valid]).all()
    assert np.isposinf(proj[0][~valid]).all() and np.isposinf(proj[1][~valid]).all()
    assert_allclose(gpu_host[0][valid], cpu[0][valid], rtol=0.0, atol=2e-8)
    assert_allclose(gpu_host[1][valid], cpu[1][valid], rtol=0.0, atol=2e-8)
    assert_array_equal(np.isnan(gpu_host[0]), np.isnan(cpu[0]))
    assert_array_equal(np.isnan(gpu_host[1]), np.isnan(cpu[1]))


def test_wrapped_antimeridian_seam_pins_native_cpu_fused_branch_and_proj_gap():
    cp = _cupy_or_skip()
    target = CRS.from_epsg(5514)
    source = target.geodetic_crs
    actual = Transformer.from_crs(source, target, always_xy=True)
    expected = PyProjTransformer.from_crs(source, target, always_xy=True)
    lon0 = 24.83333333333333
    longitude = np.array(
        [
            math.nextafter(lon0 - 180.0, -math.inf),
            lon0 - 180.0,
            math.nextafter(lon0 - 180.0, math.inf),
            math.nextafter(lon0 + 180.0, -math.inf),
            lon0 + 180.0,
            math.nextafter(lon0 + 180.0, math.inf),
        ]
    )
    latitude = np.zeros(longitude.size)
    cpu = actual.transform(longitude, latitude, transcendentals="native")
    proj = expected.transform(longitude, latitude)
    gpu = actual.transform_buffers(
        cp.asarray(longitude),
        cp.asarray(latitude),
        precision="fp64",
        transcendentals="native",
    )
    gpu_host = cp.asnumpy(gpu[0]), cp.asnumpy(gpu[1])

    assert_allclose(gpu_host[0], cpu[0], rtol=0.0, atol=2e-8)
    assert_allclose(gpu_host[1], cpu[1], rtol=0.0, atol=2e-8)
    assert_array_equal(np.signbit(cpu[0]), [True, True, False, True, True, False])
    assert_allclose(np.abs(cpu[0]), np.abs(proj[0]), rtol=0.0, atol=2e-5)
    assert_allclose(cpu[1], proj[1], rtol=0.0, atol=2e-5)
    assert cpu[0][1] < 0.0 < proj[0][1]


def test_inverse_center_tiny_radius_axes_and_huge_finite_values_match_proj_and_fused():
    cp = _cupy_or_skip()
    target = CRS.from_epsg(5514)
    source = target.geodetic_crs
    actual = Transformer.from_crs(source, target, always_xy=True)
    expected = PyProjTransformer.from_crs(target, source, always_xy=True)
    x = np.array([0.0, -0.0, 1e-320, -1e-320, 1e-24, -1e-24, 1.0, -1.0, 1e20, -1e20])
    y = np.array([0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e20, -1e20])

    cpu = actual.transform(x, y, direction="INVERSE", transcendentals="native")
    proj = expected.transform(x, y)
    gpu = actual.transform_buffers(
        cp.asarray(x),
        cp.asarray(y),
        direction="INVERSE",
        precision="fp64",
        transcendentals="native",
    )
    gpu_host = cp.asnumpy(gpu[0]), cp.asnumpy(gpu[1])

    assert_allclose(cpu[0], proj[0], rtol=0.0, atol=5e-11)
    assert_allclose(cpu[1], proj[1], rtol=0.0, atol=5e-11)
    assert_allclose(gpu_host[0], cpu[0], rtol=0.0, atol=6e-14)
    assert_allclose(gpu_host[1], cpu[1], rtol=0.0, atol=6e-14)
    assert_allclose(cpu[0][:6], 24.833333333333332, rtol=0.0, atol=2e-14)
    assert_allclose(cpu[1][:6], 59.75759856306633, rtol=0.0, atol=2e-14)


@pytest.mark.parametrize("epsg", [5513, 5514])
@pytest.mark.parametrize("direction", ["FORWARD", "INVERSE"])
def test_full_nonfinite_matrix_is_atomic_and_cpu_fused_native_stable(epsg, direction):
    cp = _cupy_or_skip()
    target = CRS.from_epsg(epsg)
    source = target.geodetic_crs
    transformer = Transformer.from_crs(source, target, always_xy=True)
    values = np.array([0.0, 1.0, math.inf, -math.inf, math.nan])
    first = np.tile(values, values.size)
    second = np.repeat(values, values.size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cpu = transformer.transform(first, second, direction=direction, transcendentals="native")
    gpu = transformer.transform_buffers(
        cp.asarray(first),
        cp.asarray(second),
        direction=direction,
        precision="fp64",
        transcendentals="native",
    )
    gpu_host = cp.asnumpy(gpu[0]), cp.asnumpy(gpu[1])

    any_nan = np.isnan(first) | np.isnan(second)
    any_infinite = (np.isinf(first) | np.isinf(second)) & ~any_nan
    finite_input = ~(any_nan | any_infinite)
    assert np.isnan(cpu[0][any_nan]).all() and np.isnan(cpu[1][any_nan]).all()
    assert np.isposinf(cpu[0][any_infinite]).all()
    assert np.isposinf(cpu[1][any_infinite]).all()
    for actual_output, expected_output in zip(gpu_host, cpu, strict=True):
        assert_array_equal(np.isnan(actual_output), np.isnan(expected_output))
        assert_array_equal(np.isposinf(actual_output), np.isposinf(expected_output))
        assert_array_equal(np.isneginf(actual_output), np.isneginf(expected_output))
        tolerance = 1e-8 if direction == "FORWARD" else 1e-13
        assert_allclose(
            actual_output[finite_input],
            expected_output[finite_input],
            rtol=0.0,
            atol=tolerance,
        )
