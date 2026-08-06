"""Wave 3 correctness baselines for optional parameters and Gnomonic."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pyproj import CRS
from pyproj import Transformer as PyProjTransformer

from vibeproj import Transformer, UnsupportedProjectionError
from vibeproj.crs import ProjectionParams, resolve_projection_params
from vibeproj.ellipsoid import SPHERE, WGS84
from vibeproj.exceptions import CRSResolutionError
from vibeproj.pipeline import TransformPipeline
from vibeproj.projections.albers_equal_area import AlbersEqualArea
from vibeproj.projections.cylindrical_equal_area import CylindricalEqualArea
from vibeproj.projections.gnomonic import GNOM_MIN_COS_C
from vibeproj.projections.lambert_conformal_conic import LambertConformalConic
from vibeproj.projections.mercator import Mercator
from vibeproj.projections.plate_carree import PlateCarree
from vibeproj.projections.winkel_tripel import WinkelTripel
from vibeproj.transcendentals import (
    NATIVE_LIBDEVICE,
    DeviceCapability,
    TranscendentalOperation,
    projection_strategy_domain,
    projection_strategy_domains,
)


GNOM_EQ = CRS.from_user_input("+proj=gnom +lat_0=0 +lon_0=0 +R=6378137 +units=m +type=crs")
GNOM_OBLIQUE = CRS.from_user_input("+proj=gnom +lat_0=45 +lon_0=12 +R=6378137 +units=m +type=crs")


def _cupy_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device not available")
    except Exception as exc:
        pytest.skip(f"CUDA device not available: {exc}")
    return cp


def test_projection_params_optional_standard_parallels_preserve_positional_order():
    params = ProjectionParams("eqc", SPHERE, 1.0, 2.0, 0.0, 3.0, 0.9, 4.0, 5.0)
    assert params.lon_0 == 1.0
    assert params.lat_0 == 2.0
    assert params.lat_1 == 0.0
    assert params.lat_2 == 3.0
    assert params.k_0 == 0.9
    assert ProjectionParams("eqc", SPHERE).lat_1 is None
    assert ProjectionParams("eqc", SPHERE).lat_2 is None


@pytest.mark.parametrize(
    ("definition", "projection_type"),
    [
        (
            "+proj=lcc +lat_0=20 +lat_1=0 +lat_2=30 +lon_0=0 +R=6371000 +units=m +type=crs",
            LambertConformalConic,
        ),
        (
            "+proj=aea +lat_0=20 +lat_1=0 +lat_2=30 +lon_0=0 +R=6371000 +units=m +type=crs",
            AlbersEqualArea,
        ),
    ],
)
def test_conic_crs_resolution_and_setup_preserve_explicit_zero(definition, projection_type):
    target = CRS.from_user_input(definition)
    params = resolve_projection_params(target)
    assert params.lat_1 == 0.0
    assert params.lat_2 == 30.0
    computed = projection_type().setup(params)
    assert math.isfinite(computed["n"])
    assert computed["n"] != pytest.approx(math.sin(math.radians(params.lat_0)))

    lon = np.array([-15.0, 0.0, 20.0])
    lat = np.array([-5.0, 20.0, 45.0])
    expected = PyProjTransformer.from_crs(target.geodetic_crs, target, always_xy=True)
    actual = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)
    expected_x, expected_y = expected.transform(lon, lat)
    actual_x, actual_y = actual.transform(lon, lat)
    assert_allclose(actual_x, expected_x, rtol=2e-15, atol=2e-8)
    assert_allclose(actual_y, expected_y, rtol=2e-15, atol=2e-8)


def test_conic_missing_standard_parallels_fall_back_by_presence():
    lcc = LambertConformalConic().setup(ProjectionParams("lcc", SPHERE, lat_0=30.0))
    aea = AlbersEqualArea().setup(ProjectionParams("aea", SPHERE, lat_0=30.0))
    assert lcc["n"] == pytest.approx(0.5)
    assert aea["n"] == pytest.approx(0.5)


def test_winkel_default_and_explicit_zero_are_distinct():
    default = WinkelTripel().setup(ProjectionParams("wintri", SPHERE))
    explicit_zero = WinkelTripel().setup(ProjectionParams("wintri", SPHERE, lat_1=0.0))
    assert default["cos_phi1"] == pytest.approx(2.0 / math.pi)
    assert explicit_zero["cos_phi1"] == 1.0


def test_cylindrical_omitted_standard_parallel_defaults_to_zero():
    cea = CylindricalEqualArea().setup(ProjectionParams("cea", SPHERE))
    eqc = PlateCarree().setup(ProjectionParams("eqc", SPHERE))
    assert cea["k0"] == 1.0
    assert eqc["cos_lat_ts"] == 1.0


def test_mercator_variant_b_requires_parameter_but_accepts_explicit_zero():
    missing = ProjectionParams("merc", SPHERE, operation_method="Mercator (variant B)")
    with pytest.raises(CRSResolutionError, match="requires an explicit latitude"):
        Mercator().setup(missing)

    explicit = ProjectionParams("merc", SPHERE, lat_1=0.0, operation_method="Mercator (variant B)")
    assert Mercator().setup(explicit)["k0"] == 1.0

    target = CRS.from_user_input("+proj=merc +lat_ts=0 +R=6371000 +units=m +type=crs")
    resolved = resolve_projection_params(target)
    assert resolved.operation_method == "Mercator (variant B)"
    assert resolved.lat_1 == 0.0
    assert resolved.k_0 == 1.0


def test_public_gnomonic_rejects_ellipsoid_and_accepts_explicit_sphere():
    ellipsoidal = CRS.from_user_input(
        "+proj=gnom +lat_0=45 +lon_0=12 +datum=WGS84 +units=m +type=crs"
    )
    with pytest.raises(UnsupportedProjectionError, match="spherical Gnomonic only"):
        resolve_projection_params(ellipsoidal)
    with pytest.raises(UnsupportedProjectionError, match="spherical Gnomonic only"):
        Transformer.from_crs(ellipsoidal.geodetic_crs, ellipsoidal, always_xy=True)
    with pytest.raises(UnsupportedProjectionError, match="Ellipsoidal Gnomonic"):
        TransformPipeline(
            ProjectionParams("longlat", WGS84),
            ProjectionParams("gnom", WGS84, lat_0=45.0),
        )

    params = resolve_projection_params(GNOM_OBLIQUE)
    assert params.ellipsoid.es == 0.0
    Transformer.from_crs(GNOM_OBLIQUE.geodetic_crs, GNOM_OBLIQUE, always_xy=True)


def test_gnomonic_randomized_forward_inverse_matches_pyproj_and_roundtrips():
    random = np.random.default_rng(20260806)
    lon = 12.0 + random.uniform(-35.0, 35.0, 4096)
    lat = 45.0 + random.uniform(-30.0, 30.0, 4096)
    expected = PyProjTransformer.from_crs(GNOM_OBLIQUE.geodetic_crs, GNOM_OBLIQUE, always_xy=True)
    actual = Transformer.from_crs(GNOM_OBLIQUE.geodetic_crs, GNOM_OBLIQUE, always_xy=True)

    expected_x, expected_y = expected.transform(lon, lat)
    actual_x, actual_y = actual.transform(lon, lat)
    assert_allclose(actual_x, expected_x, rtol=2e-15, atol=2e-8)
    assert_allclose(actual_y, expected_y, rtol=2e-15, atol=2e-8)

    actual_lon, actual_lat = actual.transform(actual_x, actual_y, direction="INVERSE")
    assert_allclose(actual_lon, lon, rtol=0.0, atol=3e-14)
    assert_allclose(actual_lat, lat, rtol=0.0, atol=5e-14)


def test_gnomonic_horizon_constant_exact_and_nextafter_match_pyproj():
    boundary = math.degrees(math.acos(GNOM_MIN_COS_C))
    longitude = np.array(
        [
            math.nextafter(boundary, -math.inf),
            boundary,
            math.nextafter(boundary, math.inf),
            90.0,
            91.0,
            180.0,
        ]
    )
    latitude = np.zeros_like(longitude)
    expected = PyProjTransformer.from_crs(GNOM_EQ.geodetic_crs, GNOM_EQ, always_xy=True)
    actual = Transformer.from_crs(GNOM_EQ.geodetic_crs, GNOM_EQ, always_xy=True)

    expected_x, expected_y = expected.transform(longitude, latitude)
    with pytest.warns(UserWarning, match="non-finite"):
        actual_x, actual_y = actual.transform(longitude, latitude)
    assert_array_equal(np.isfinite(actual_x), np.isfinite(expected_x))
    assert_array_equal(np.isfinite(actual_y), np.isfinite(expected_y))
    assert_allclose(actual_x[:2], expected_x[:2], rtol=2e-15, atol=0.0)
    assert_allclose(actual_y[:2], expected_y[:2], rtol=0.0, atol=0.0)
    assert np.all(np.isposinf(actual_x[2:]))
    assert np.all(np.isposinf(actual_y[2:]))


def test_gnomonic_inverse_center_axes_huge_finite_and_nonfinite_classification():
    points_x = np.array([0.0, 1e300, 0.0, 1e300, math.nan, math.inf])
    points_y = np.array([0.0, 0.0, 1e300, 1e300, 0.0, 0.0])
    expected = PyProjTransformer.from_crs(GNOM_OBLIQUE, GNOM_OBLIQUE.geodetic_crs, always_xy=True)
    actual = Transformer.from_crs(GNOM_OBLIQUE.geodetic_crs, GNOM_OBLIQUE, always_xy=True)

    expected_lon, expected_lat = expected.transform(points_x, points_y)
    with pytest.warns(UserWarning, match="non-finite"):
        actual_lon, actual_lat = actual.transform(points_x, points_y, direction="INVERSE")
    assert_allclose(actual_lon[:4], expected_lon[:4], rtol=0.0, atol=4e-14)
    assert_allclose(actual_lat[:4], expected_lat[:4], rtol=0.0, atol=4e-14)
    assert_array_equal(np.isnan(actual_lon[4:]), np.isnan(expected_lon[4:]))
    assert_array_equal(np.isnan(actual_lat[4:]), np.isnan(expected_lat[4:]))
    assert_array_equal(np.isinf(actual_lon[4:]), np.isinf(expected_lon[4:]))
    assert_array_equal(np.isinf(actual_lat[4:]), np.isinf(expected_lat[4:]))


@pytest.mark.parametrize(
    ("latitude_origin", "mode"),
    [(0.0, "equatorial"), (90.0, "north_pole"), (37.0, "oblique"), (-90.0, "south_pole")],
)
def test_gnomonic_strategy_domains_are_exact_and_native_only(latitude_origin, mode):
    pipeline = TransformPipeline(
        ProjectionParams("longlat", SPHERE, north_first=False),
        ProjectionParams("gnom", SPHERE, lat_0=latitude_origin, north_first=False),
    )
    domain = projection_strategy_domain("gnom", "forward", pipeline.computed)
    assert domain == f"gnom.forward.spherical.{mode}"
    explanation = pipeline.build_execution_context(
        precision="fp64",
        transcendentals="accelerated",
        device=DeviceCapability(backend="cpu", name="test CPU"),
        workload_size=1_000_000,
    )
    decision = next(
        item
        for item in explanation.decisions
        if item.operation is TranscendentalOperation.PROJECTION
    )
    assert decision.domain == domain
    assert decision.implementation_id == NATIVE_LIBDEVICE


def test_gnomonic_strategy_warmup_domains_cover_all_spherical_origins():
    assert projection_strategy_domains("gnom", "forward") == (
        "gnom.forward.spherical.equatorial",
        "gnom.forward.spherical.north_pole",
        "gnom.forward.spherical.oblique",
        "gnom.forward.spherical.south_pole",
    )


def test_gnomonic_gpu_parity_buffers_and_stream():
    cp = _cupy_or_skip()
    params = resolve_projection_params(GNOM_OBLIQUE)
    source = ProjectionParams("longlat", SPHERE, north_first=False)
    pipeline = TransformPipeline(source, params)
    lon_np = np.linspace(-10.0, 34.0, 4096)
    lat_np = np.linspace(20.0, 70.0, 4096)
    expected_x, expected_y = pipeline.transform(lon_np, lat_np, np)
    lon = cp.asarray(lon_np)
    lat = cp.asarray(lat_np)
    out_x = cp.empty_like(lon)
    out_y = cp.empty_like(lat)
    stream = cp.cuda.Stream(non_blocking=True)
    actual_x, actual_y = pipeline.transform(lon, lat, cp, out_x=out_x, out_y=out_y, stream=stream)
    stream.synchronize()
    assert actual_x is out_x
    assert actual_y is out_y
    assert_allclose(cp.asnumpy(actual_x), expected_x, rtol=2e-14, atol=2e-8)
    assert_allclose(cp.asnumpy(actual_y), expected_y, rtol=2e-14, atol=2e-8)

    boundary = math.degrees(math.acos(GNOM_MIN_COS_C))
    edge_lon_np = np.array(
        [
            math.nextafter(boundary, -math.inf),
            boundary,
            math.nextafter(boundary, math.inf),
            90.0,
            math.nan,
            math.inf,
        ]
    )
    edge_lat_np = np.zeros_like(edge_lon_np)
    equatorial = TransformPipeline(
        source,
        resolve_projection_params(GNOM_EQ),
    )
    cpu_edge_x, cpu_edge_y = equatorial.transform(edge_lon_np, edge_lat_np, np)
    gpu_edge_x, gpu_edge_y = equatorial.transform(
        cp.asarray(edge_lon_np), cp.asarray(edge_lat_np), cp
    )
    assert_array_equal(cp.asnumpy(gpu_edge_x), cpu_edge_x)
    assert_array_equal(cp.asnumpy(gpu_edge_y), cpu_edge_y)

    inverse = TransformPipeline(params, source)
    huge_x_np = np.array([0.0, 1e300, 0.0, 1e300, math.nan, math.inf])
    huge_y_np = np.array([0.0, 0.0, 1e300, 1e300, 0.0, 0.0])
    cpu_lon, cpu_lat = inverse.transform(huge_x_np, huge_y_np, np)
    gpu_lon, gpu_lat = inverse.transform(cp.asarray(huge_x_np), cp.asarray(huge_y_np), cp)
    gpu_lon = cp.asnumpy(gpu_lon)
    gpu_lat = cp.asnumpy(gpu_lat)
    assert_allclose(gpu_lon[:4], cpu_lon[:4], rtol=0.0, atol=4e-14)
    assert_allclose(gpu_lat[:4], cpu_lat[:4], rtol=0.0, atol=4e-14)
    assert_array_equal(np.isfinite(gpu_lon[4:]), np.isfinite(cpu_lon[4:]))
    assert_array_equal(np.isfinite(gpu_lat[4:]), np.isfinite(cpu_lat[4:]))
