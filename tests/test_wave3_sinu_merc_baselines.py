"""Wave 3 correctness baselines for Sinusoidal and Mercator."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pyproj import CRS
from pyproj import Transformer as PyProjTransformer

from vibeproj import Transformer, UnsupportedProjectionError
from vibeproj._conformal import conformal_to_geodetic_coefficients
from vibeproj.crs import ProjectionParams, resolve_projection_params
from vibeproj.ellipsoid import SPHERE, Ellipsoid
from vibeproj.pipeline import TransformPipeline
from vibeproj.projections.sinusoidal import (
    SINU_MAX_ECCENTRICITY_SQUARED,
    SINU_MAX_SEMI_MAJOR_AXIS_M,
    Sinusoidal,
)
from vibeproj.transcendentals import (
    NATIVE_LIBDEVICE,
    SINU_FORWARD_FIXED_Q62,
    SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS,
    DeviceCapability,
    TranscendentalOperation,
    projection_strategy_domain,
    projection_strategy_domains,
    resolve_transcendental_strategy,
)


CPU = DeviceCapability(backend="cpu", name="test CPU")
ADA = DeviceCapability(
    backend="cuda",
    compute_capability=(8, 9),
    fp32_to_fp64_ratio=64,
    name="test Ada",
)


def _cupy_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device not available")
    except Exception as exc:
        pytest.skip(f"CUDA device not available: {exc}")
    return cp


def _projection_domain(
    transformer: Transformer,
    *,
    direction: str = "FORWARD",
    device: DeviceCapability = CPU,
    transcendentals: str = "native",
    workload_size: int | None = None,
):
    explanation = transformer.explain_strategy(
        direction=direction,
        device=device,
        transcendentals=transcendentals,
        workload_size=workload_size,
    )
    return next(
        decision
        for decision in explanation.decisions
        if decision.operation is TranscendentalOperation.PROJECTION
    )


@pytest.mark.parametrize(
    "target_input",
    [
        "ESRI:54008",
        "EPSG:2934",
        "EPSG:3994",
        "EPSG:3388",
        "EPSG:3395",
        "+proj=merc +lat_ts=33 +lon_0=12 +R=6371000 +units=m +type=crs",
    ],
)
def test_sinu_and_merc_randomized_cpu_forward_inverse_match_pyproj(target_input):
    target = CRS.from_user_input(target_input)
    source = target.geodetic_crs
    params = resolve_projection_params(target)
    random = np.random.default_rng(20260806)
    lon = params.lon_0 + random.uniform(-70.0, 70.0, 4096)
    lat = random.uniform(-82.0, 82.0, 4096)

    expected_forward = PyProjTransformer.from_crs(source, target, always_xy=True)
    expected_inverse = PyProjTransformer.from_crs(target, source, always_xy=True)
    transformer = Transformer.from_crs(source, target, always_xy=True)

    expected_x, expected_y = expected_forward.transform(lon, lat)
    actual_x, actual_y = transformer.transform(lon, lat)
    assert_allclose(actual_x, expected_x, rtol=0.0, atol=2e-8)
    assert_allclose(actual_y, expected_y, rtol=0.0, atol=2e-8)

    expected_lon, expected_lat = expected_inverse.transform(expected_x, expected_y)
    actual_lon, actual_lat = transformer.transform(actual_x, actual_y, direction="INVERSE")
    assert_allclose(actual_lon, expected_lon, rtol=0.0, atol=2e-12)
    assert_allclose(actual_lat, expected_lat, rtol=0.0, atol=2e-12)
    assert_allclose(actual_lon, lon, rtol=0.0, atol=2e-12)
    assert_allclose(actual_lat, lat, rtol=0.0, atol=2e-12)


@pytest.mark.parametrize(
    ("target_input", "expected_k0"),
    [
        ("EPSG:2934", 0.997),
        ("EPSG:3994", 0.7557992272019596),
        ("EPSG:3388", 0.7442608941715082),
        ("EPSG:3395", 1.0),
    ],
)
def test_mercator_crs_resolution_honors_variant_scale(target_input, expected_k0):
    params = resolve_projection_params(CRS.from_user_input(target_input))
    assert params.k_0 == pytest.approx(expected_k0, rel=0.0, abs=2e-15)


def test_mercator_reuses_transverse_mercator_conformal_inverse_coefficients():
    mercator = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=True)
    transverse = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)

    transverse_coefficients = transverse._pipeline.computed["cgb"]
    assert isinstance(transverse_coefficients, list)
    assert mercator._pipeline.computed["conformal_to_geodetic"] == tuple(transverse_coefficients)

    sphere = Transformer.from_crs(
        "+proj=longlat +R=6371000 +type=crs",
        "+proj=merc +R=6371000 +type=crs",
        always_xy=True,
    )
    assert sphere._pipeline.computed["conformal_to_geodetic"] == (0.0,) * 6


@pytest.mark.parametrize("n", [0.0, 1 / 3.0, 0.00363341959, 1 / 298.257223563])
def test_shared_conformal_coefficients_are_bitwise_legacy_tmerc(n):
    power = n
    expected = [
        n
        * (2 + n * (-2 / 3.0 + n * (-2 + n * (116 / 45.0 + n * (26 / 45.0 + n * (-2854 / 675.0))))))
    ]
    power *= n
    expected.append(
        power
        * (7 / 3.0 + n * (-8 / 5.0 + n * (-227 / 45.0 + n * (2704 / 315.0 + n * (2323 / 945.0)))))
    )
    power *= n
    expected.append(
        power * (56 / 15.0 + n * (-136 / 35.0 + n * (-1262 / 105.0 + n * (73814 / 2835.0))))
    )
    power *= n
    expected.append(power * (4279 / 630.0 + n * (-332 / 35.0 + n * (-399572 / 14175.0))))
    power *= n
    expected.append(power * (4174 / 315.0 + n * (-144838 / 6237.0)))
    power *= n
    expected.append(power * (601676 / 22275.0))

    actual_bits = np.asarray(conformal_to_geodetic_coefficients(n)).view(np.uint64)
    expected_bits = np.asarray(expected).view(np.uint64)
    assert_array_equal(actual_bits, expected_bits)


@pytest.mark.parametrize(
    ("target_definition", "expected_domain"),
    [
        ("+proj=sinu +R=6378137 +units=m +type=crs", "sinu.forward.spherical"),
        ("ESRI:54008", "sinu.forward.ellipsoidal"),
        ("EPSG:3395", "merc.forward.ellipsoidal.variant_a"),
        ("EPSG:3994", "merc.forward.ellipsoidal.variant_b"),
        (
            "+proj=merc +k=0.87 +R=6371000 +units=m +type=crs",
            "merc.forward.spherical.variant_a",
        ),
        ("EPSG:3857", "webmerc.forward.spherical.pseudo"),
    ],
)
def test_sinu_merc_and_webmerc_strategy_domains_are_exact(target_definition, expected_domain):
    target = CRS.from_user_input(target_definition)
    transformer = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)
    assert _projection_domain(transformer).domain == expected_domain
    assert _projection_domain(transformer, direction="INVERSE").domain == expected_domain.replace(
        ".forward.", ".inverse."
    )


def test_sinu_fixed_q62_is_registered_only_for_the_spherical_domain():
    sphere = CRS.from_user_input("+proj=sinu +R=6378137 +units=m +type=crs")
    ellipsoid = CRS.from_user_input("ESRI:54008")
    sphere_transformer = Transformer.from_crs(sphere.geodetic_crs, sphere, always_xy=True)
    ellipsoid_transformer = Transformer.from_crs(ellipsoid.geodetic_crs, ellipsoid, always_xy=True)

    sphere_decision = _projection_domain(
        sphere_transformer,
        device=ADA,
        transcendentals="auto",
        workload_size=SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS,
    )
    ellipsoid_decision = _projection_domain(
        ellipsoid_transformer,
        device=ADA,
        transcendentals="accelerated",
        workload_size=SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS,
    )
    assert sphere_decision.implementation_id == SINU_FORWARD_FIXED_Q62
    assert ellipsoid_decision.implementation_id == NATIVE_LIBDEVICE
    assert projection_strategy_domains("sinu", "forward") == ("sinu.forward.spherical",)


def test_custom_mercator_domain_is_exact_and_native_only():
    geographic = ProjectionParams("longlat", SPHERE, north_first=False)
    projected = ProjectionParams("merc", SPHERE, k_0=0.87, north_first=False)
    pipeline = TransformPipeline(geographic, projected)
    domain = projection_strategy_domain("merc", "forward", pipeline.computed)
    decision = resolve_transcendental_strategy(
        TranscendentalOperation.PROJECTION,
        "accelerated",
        device=ADA,
        domain=domain,
        precision="fp64",
    )
    assert domain == "merc.forward.spherical.custom"
    assert decision.implementation_id == NATIVE_LIBDEVICE
    assert decision.fallback is True


def test_mercator_pole_clamp_exact_and_nextafter_boundaries():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=True)
    boundary = 89.999
    below = math.nextafter(boundary, -math.inf)
    above = math.nextafter(boundary, math.inf)

    _, exact_y = transformer.transform(np.array([0.0]), np.array([boundary]))
    _, below_y = transformer.transform(np.array([0.0]), np.array([below]))
    with pytest.warns(RuntimeWarning, match="Latitude values clamped"):
        _, above_y = transformer.transform(np.array([0.0]), np.array([above]))

    assert below_y[0] < exact_y[0]
    assert_array_equal(above_y, exact_y)


@pytest.mark.parametrize("target_input", ["ESRI:54008", "EPSG:3994"])
def test_sinu_and_merc_native_axis_order_matches_pyproj(target_input):
    target = CRS.from_user_input(target_input)
    source = target.geodetic_crs
    expected = PyProjTransformer.from_crs(source, target, always_xy=False)
    transformer = Transformer.from_crs(source, target, always_xy=False)
    first_is_latitude = source.axis_info[0].direction.lower() in ("north", "south")
    lat = np.array([-35.0, 0.0, 60.0])
    lon = np.array([-10.0, 0.0, 25.0]) + resolve_projection_params(target).lon_0
    first, second = (lat, lon) if first_is_latitude else (lon, lat)

    expected_first, expected_second = expected.transform(first, second)
    actual_first, actual_second = transformer.transform(first, second)
    assert_allclose(actual_first, expected_first, rtol=0.0, atol=2e-8)
    assert_allclose(actual_second, expected_second, rtol=0.0, atol=2e-8)


def test_ellipsoidal_sinu_inverse_meridional_pole_boundary():
    target = CRS.from_user_input("ESRI:54008")
    transformer = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)
    _, pole_y = PyProjTransformer.from_crs(target.geodetic_crs, target, always_xy=True).transform(
        0.0, 90.0
    )
    below = math.nextafter(pole_y, -math.inf)
    above = math.nextafter(pole_y, math.inf)

    with pytest.warns(UserWarning, match="non-finite"):
        lon, lat = transformer.transform(
            np.zeros(3), np.array([below, pole_y, above]), direction="INVERSE"
        )
    assert np.all(np.isfinite(lon[:2]))
    assert np.all(np.isfinite(lat[:2]))
    assert lat[1] == pytest.approx(90.0, abs=2e-13)
    assert math.isinf(lon[2]) and math.isinf(lat[2])


def test_ellipsoidal_sinu_supported_eccentricity_boundary_matches_pyproj():
    target = CRS.from_user_input(
        f"+proj=sinu +a=6400000 +es={SINU_MAX_ECCENTRICITY_SQUARED} +units=m +type=crs"
    )
    source = target.geodetic_crs
    expected = PyProjTransformer.from_crs(source, target, always_xy=True)
    transformer = Transformer.from_crs(source, target, always_xy=True)
    lat = np.linspace(-89.999, 89.999, 16_385)
    lon = np.linspace(-179.0, 179.0, lat.size)

    expected_x, expected_y = expected.transform(lon, lat)
    actual_x, actual_y = transformer.transform(lon, lat)
    forward_error_m = np.hypot(actual_x - expected_x, actual_y - expected_y)
    assert np.max(forward_error_m) < 1e-8
    assert np.quantile(forward_error_m, 0.99) < 1e-8

    actual_lon, actual_lat = transformer.transform(actual_x, actual_y, direction="INVERSE")
    assert_allclose(actual_lat, lat, rtol=0.0, atol=3e-12)
    latitude_error_m = np.deg2rad(actual_lat - lat) * 6_400_000.0
    longitude_error_m = np.deg2rad(actual_lon - lon) * 6_400_000.0 * np.cos(np.deg2rad(lat))
    assert np.max(np.hypot(longitude_error_m, latitude_error_m)) < 1e-8

    for eccentricity_squared in (
        math.nextafter(SINU_MAX_ECCENTRICITY_SQUARED, -math.inf),
        SINU_MAX_ECCENTRICITY_SQUARED,
    ):
        Sinusoidal().setup(ProjectionParams("sinu", _ellipsoid_with_es(eccentricity_squared)))


def _sphere_with_radius(radius: float) -> Ellipsoid:
    return Ellipsoid(a=radius, b=radius, f=0.0, e=0.0, es=0.0, n=0.0)


def _ellipsoid_with_es(es: float) -> Ellipsoid:
    radius = 6_400_000.0
    eccentricity = math.sqrt(es) if es >= 0.0 else math.nan
    flattening = 1.0 - math.sqrt(1.0 - es) if 0.0 <= es <= 1.0 else math.nan
    semi_minor = radius * (1.0 - flattening)
    third_flattening = flattening / (2.0 - flattening)
    return Ellipsoid(
        a=radius,
        b=semi_minor,
        f=flattening,
        e=eccentricity,
        es=es,
        n=third_flattening,
    )


@pytest.mark.parametrize(
    "eccentricity_squared",
    [
        math.nextafter(SINU_MAX_ECCENTRICITY_SQUARED, math.inf),
        0.19,
        -math.ulp(0.0),
        math.nan,
        math.inf,
    ],
)
def test_sinu_rejects_unqualified_custom_eccentricity_before_dispatch(eccentricity_squared):
    ellipsoid = _ellipsoid_with_es(eccentricity_squared)
    projected = ProjectionParams("sinu", ellipsoid, north_first=False)
    geographic = ProjectionParams("longlat", ellipsoid, north_first=False)

    with pytest.raises(UnsupportedProjectionError, match="eccentricity squared"):
        Sinusoidal().setup(projected)
    with pytest.raises(UnsupportedProjectionError, match="eccentricity squared"):
        TransformPipeline(geographic, projected)


@pytest.mark.parametrize("inverse_flattening", [10.0, 5.0])
def test_public_sinu_rejects_high_eccentricity_crs_during_construction(inverse_flattening):
    target = CRS.from_user_input(
        f"+proj=sinu +a=6400000 +rf={inverse_flattening} +units=m +type=crs"
    )
    with pytest.raises(UnsupportedProjectionError, match="eccentricity squared"):
        Transformer.from_crs(target.geodetic_crs, target, always_xy=True)


@pytest.mark.parametrize(
    "semi_major_axis",
    [
        math.nextafter(SINU_MAX_SEMI_MAJOR_AXIS_M, math.inf),
        1.0e9,
        1.0e12,
        0.0,
        -1.0,
        math.nan,
        math.inf,
    ],
)
def test_sinu_rejects_unqualified_scale_before_dispatch(semi_major_axis):
    ellipsoid = _sphere_with_radius(semi_major_axis)
    projected = ProjectionParams("sinu", ellipsoid, north_first=False)
    geographic = ProjectionParams("longlat", ellipsoid, north_first=False)

    with pytest.raises(UnsupportedProjectionError, match="semi-major axes"):
        Sinusoidal().setup(projected)
    with pytest.raises(UnsupportedProjectionError, match="semi-major axes"):
        TransformPipeline(geographic, projected)


def test_sinu_scale_exact_boundary_and_nextafter_public_construction():
    for radius in (
        math.nextafter(SINU_MAX_SEMI_MAJOR_AXIS_M, -math.inf),
        SINU_MAX_SEMI_MAJOR_AXIS_M,
    ):
        ellipsoid = _sphere_with_radius(radius)
        projected = ProjectionParams("sinu", ellipsoid, north_first=False)
        geographic = ProjectionParams("longlat", ellipsoid, north_first=False)
        Sinusoidal().setup(projected)
        TransformPipeline(geographic, projected)

    at_ceiling = CRS.from_user_input(
        f"+proj=sinu +a={SINU_MAX_SEMI_MAJOR_AXIS_M} +es={SINU_MAX_ECCENTRICITY_SQUARED} "
        "+units=m +type=crs"
    )
    Transformer.from_crs(at_ceiling.geodetic_crs, at_ceiling, always_xy=True)

    above_ceiling = math.nextafter(SINU_MAX_SEMI_MAJOR_AXIS_M, math.inf)
    outside = CRS.from_user_input(
        f"+proj=sinu +a={above_ceiling!r} +es={SINU_MAX_ECCENTRICITY_SQUARED} +units=m +type=crs"
    )
    with pytest.raises(UnsupportedProjectionError, match="semi-major axes"):
        Transformer.from_crs(outside.geodetic_crs, outside, always_xy=True)


def test_sinu_and_merc_nonfinite_cpu_behavior_is_stable():
    target = CRS.from_user_input("ESRI:54008")
    projected = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)
    with pytest.warns((RuntimeWarning, UserWarning)):
        x, y = projected.transform(
            np.array([math.nan, math.inf, 0.0]), np.array([0.0, 0.0, math.nan])
        )
    assert math.isnan(x[0]) and math.isinf(x[1]) and math.isnan(y[2])

    with pytest.warns(UserWarning, match="non-finite"):
        lon, lat = projected.transform(
            np.array([0.0, math.nan]),
            np.array([math.inf, math.nan]),
            direction="INVERSE",
        )
    assert math.isinf(lon[0]) and math.isinf(lat[0])
    assert math.isnan(lon[1]) and math.isnan(lat[1])


@pytest.mark.parametrize(
    "target_input",
    [
        "ESRI:54008",
        "EPSG:2934",
        "EPSG:3994",
        "EPSG:3388",
        "EPSG:3395",
        "+proj=merc +lat_ts=33 +lon_0=12 +R=6371000 +units=m +type=crs",
    ],
)
def test_sinu_and_merc_fused_fp64_forward_inverse_match_pyproj(target_input):
    cp = _cupy_or_skip()
    target = CRS.from_user_input(target_input)
    source = target.geodetic_crs
    params = resolve_projection_params(target)
    lon = np.linspace(params.lon_0 - 60.0, params.lon_0 + 60.0, 2048)
    lat = np.linspace(-80.0, 80.0, 2048)
    expected = PyProjTransformer.from_crs(source, target, always_xy=True)
    transformer = Transformer.from_crs(source, target, always_xy=True)

    expected_x, expected_y = expected.transform(lon, lat)
    actual_x, actual_y = transformer.transform(
        cp.asarray(lon), cp.asarray(lat), precision="fp64", transcendentals="native"
    )
    assert_allclose(cp.asnumpy(actual_x), expected_x, rtol=0.0, atol=2e-8)
    assert_allclose(cp.asnumpy(actual_y), expected_y, rtol=0.0, atol=2e-8)

    actual_lon, actual_lat = transformer.transform(
        actual_x,
        actual_y,
        direction="INVERSE",
        precision="fp64",
        transcendentals="native",
    )
    assert_allclose(cp.asnumpy(actual_lon), lon, rtol=0.0, atol=2e-12)
    assert_allclose(cp.asnumpy(actual_lat), lat, rtol=0.0, atol=2e-12)


@pytest.mark.parametrize("target_input", ["ESRI:54008", "EPSG:3994"])
def test_sinu_and_merc_buffers_honor_outputs_and_stream(target_input):
    cp = _cupy_or_skip()
    target = CRS.from_user_input(target_input)
    transformer = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)
    lon = cp.linspace(-20.0, 20.0, 1024, dtype=cp.float64)
    lat = cp.linspace(-70.0, 70.0, 1024, dtype=cp.float64)
    out_x = cp.empty_like(lon)
    out_y = cp.empty_like(lat)
    stream = cp.cuda.Stream(non_blocking=True)

    result = transformer.transform_buffers(
        lon,
        lat,
        out_x=out_x,
        out_y=out_y,
        precision="fp64",
        transcendentals="native",
        stream=stream,
    )
    stream.synchronize()
    assert result[0] is out_x
    assert result[1] is out_y
    assert bool(cp.all(cp.isfinite(out_x)))
    assert bool(cp.all(cp.isfinite(out_y)))


def test_mercator_fused_pole_clamp_and_nonfinite_semantics_match_cpu():
    cp = _cupy_or_skip()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=True)
    boundary = 89.999
    latitude = np.array(
        [
            math.nextafter(boundary, -math.inf),
            boundary,
            math.nextafter(boundary, math.inf),
            math.inf,
            -math.inf,
            math.nan,
        ]
    )
    longitude = np.zeros_like(latitude)
    with pytest.warns((RuntimeWarning, UserWarning)):
        expected_x, expected_y = transformer.transform(longitude, latitude)
    actual_x, actual_y = transformer.transform(
        cp.asarray(longitude),
        cp.asarray(latitude),
        precision="fp64",
        transcendentals="native",
    )

    assert_allclose(cp.asnumpy(actual_x), expected_x, rtol=0.0, atol=0.0, equal_nan=True)
    assert_allclose(cp.asnumpy(actual_y), expected_y, rtol=0.0, atol=2e-8, equal_nan=True)
    assert cp.asnumpy(actual_y)[2] == cp.asnumpy(actual_y)[1]


def test_ellipsoidal_sinu_fused_inverse_meridional_pole_boundary():
    cp = _cupy_or_skip()
    target = CRS.from_user_input("ESRI:54008")
    transformer = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)
    _, pole_y = PyProjTransformer.from_crs(target.geodetic_crs, target, always_xy=True).transform(
        0.0, 90.0
    )
    y = cp.asarray([math.nextafter(pole_y, -math.inf), pole_y, math.nextafter(pole_y, math.inf)])
    lon, lat = transformer.transform(cp.zeros_like(y), y, direction="INVERSE")
    lon_host, lat_host = cp.asnumpy(lon), cp.asnumpy(lat)
    assert np.all(np.isfinite(lon_host[:2]))
    assert np.all(np.isfinite(lat_host[:2]))
    assert lat_host[1] == pytest.approx(90.0, abs=2e-13)
    assert math.isinf(lon_host[2]) and math.isinf(lat_host[2])


def test_ellipsoidal_sinu_supported_eccentricity_boundary_fused_matches_pyproj():
    cp = _cupy_or_skip()
    target = CRS.from_user_input(
        f"+proj=sinu +a=6400000 +es={SINU_MAX_ECCENTRICITY_SQUARED} +units=m +type=crs"
    )
    source = target.geodetic_crs
    expected = PyProjTransformer.from_crs(source, target, always_xy=True)
    transformer = Transformer.from_crs(source, target, always_xy=True)
    lat = np.linspace(-89.999, 89.999, 16_385)
    lon = np.linspace(-179.0, 179.0, lat.size)
    expected_x, expected_y = expected.transform(lon, lat)

    actual_x, actual_y = transformer.transform(
        cp.asarray(lon), cp.asarray(lat), precision="fp64", transcendentals="native"
    )
    x_host, y_host = cp.asnumpy(actual_x), cp.asnumpy(actual_y)
    forward_error_m = np.hypot(x_host - expected_x, y_host - expected_y)
    assert np.max(forward_error_m) < 1e-8
    assert np.quantile(forward_error_m, 0.99) < 1e-8

    actual_lon, actual_lat = transformer.transform(
        actual_x,
        actual_y,
        direction="INVERSE",
        precision="fp64",
        transcendentals="native",
    )
    lon_host, lat_host = cp.asnumpy(actual_lon), cp.asnumpy(actual_lat)
    inverse_error_m = np.hypot(
        np.deg2rad(lon_host - lon) * 6_400_000.0 * np.cos(np.deg2rad(lat)),
        np.deg2rad(lat_host - lat) * 6_400_000.0,
    )
    assert np.max(inverse_error_m) < 1e-8
    assert np.quantile(inverse_error_m, 0.99) < 1e-8
