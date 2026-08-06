"""Wave 3F native-contract baselines for Transverse Mercator inverse."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pyproj import CRS
from pyproj import Transformer as PyProjTransformer

from vibeproj import Transformer
from vibeproj.crs import ProjectionParams
from vibeproj.ellipsoid import SPHERE, WGS84
from vibeproj.pipeline import TransformPipeline
from vibeproj.transcendentals import (
    NATIVE_LIBDEVICE,
    projection_strategy_domain,
    projection_strategy_domains,
)


# PROJ's Poder/Engsager inverse accepts this closed normalized-easting interval.
# This is a data-domain boundary, not a launch-uniform setup parameter.
PROJ_ETMERC_CE_LIMIT = 2.623395162778


def _cupy_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device not available")
    except Exception as exc:
        pytest.skip(f"CUDA device not available: {exc}")
    return cp


def _inverse_transformer(target: CRS, *, always_xy: bool = True) -> Transformer:
    return Transformer.from_crs(target, target.geodetic_crs, always_xy=always_xy)


def _angular_difference_degrees(actual, expected):
    return (np.asarray(actual) - np.asarray(expected) + 180.0) % 360.0 - 180.0


@pytest.mark.parametrize(
    ("epsg", "zone", "central_meridian", "false_northing"),
    [
        (32601, 1, -177.0, 0.0),
        (32660, 60, 177.0, 0.0),
        (32701, 1, -177.0, 10_000_000.0),
        (32760, 60, 177.0, 10_000_000.0),
    ],
)
def test_utm_zone_and_hemisphere_boundaries_pin_launch_uniform_setup(
    epsg, zone, central_meridian, false_northing
):
    target = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)
    computed = transformer._pipeline.computed

    assert transformer._pipeline.proj_params.utm_zone == zone
    assert computed["is_utm"] is True
    assert computed["_strategy_geometry"] == "ellipsoidal"
    assert computed["_strategy_operation_method"] == "Transverse Mercator"
    assert math.degrees(computed["lam0"]) == pytest.approx(central_meridian, abs=2e-14)
    assert computed["k0"] == 0.9996
    assert computed["x0"] == 500_000.0
    assert computed["y0"] == false_northing
    assert computed["Qn"] == pytest.approx(0.9979249687118802, rel=0.0, abs=0.0)
    assert computed["Zb"] == 0.0
    assert math.copysign(1.0, computed["Zb"]) == -1.0
    for coefficients in ("cgb", "cbg", "utg", "gtu"):
        assert len(computed[coefficients]) == 6
        assert np.isfinite(computed[coefficients]).all()


@pytest.mark.parametrize(
    ("target_definition", "geometry", "is_utm"),
    [
        ("EPSG:32631", "ellipsoidal", True),
        ("EPSG:2193", "ellipsoidal", False),
        ("EPSG:2222", "ellipsoidal", False),
        (
            "+proj=tmerc +lat_0=10 +lon_0=20 +k=.8 +x_0=12345 +y_0=-67890 "
            "+R=6371000 +units=m +type=crs",
            "spherical",
            False,
        ),
    ],
)
def test_inverse_setup_matrix_pins_sphere_ellipsoid_utm_and_general_semantics(
    target_definition, geometry, is_utm
):
    target = CRS.from_user_input(target_definition)
    computed = _inverse_transformer(target)._pipeline.computed

    assert computed["_strategy_geometry"] == geometry
    assert computed["is_utm"] is is_utm
    setup_scalars = (
        "Qn",
        "Zb",
        "lam0",
        "k0",
        "x0",
        "y0",
        "a",
        "x_unit_to_m",
        "y_unit_to_m",
        "easting_axis_sign",
        "northing_axis_sign",
    )
    assert all(math.isfinite(float(computed[name])) for name in setup_scalars)
    if geometry == "spherical":
        for coefficients in ("cgb", "cbg", "utg", "gtu"):
            assert_array_equal(computed[coefficients], np.zeros(6))
    else:
        for coefficients in ("cgb", "cbg", "utg", "gtu"):
            assert np.any(np.asarray(computed[coefficients]) != 0.0)


def test_all_inverse_setup_semantics_currently_collapse_to_one_dispatch_domain():
    utm = _inverse_transformer(CRS.from_epsg(32631))._pipeline.computed
    general = _inverse_transformer(CRS.from_epsg(2193))._pipeline.computed
    sphere = _inverse_transformer(
        CRS.from_user_input("+proj=tmerc +lat_0=10 +lon_0=20 +k=.8 +R=6371000 +units=m +type=crs")
    )._pipeline.computed
    invalid = dict(utm, Qn=0.0, a=0.0, x_unit_to_m=0.0, utg=[math.nan] * 6)

    # This deliberately exposes the pre-acceleration planning blocker: inverse
    # TM has no exact setup taxonomy yet. UTM/general, sphere/ellipsoid, and
    # invalid setup all warm and resolve through the same generic domain.
    for computed in (utm, general, sphere, invalid):
        assert projection_strategy_domain("tmerc", "inverse", computed) == "tmerc.inverse"
    assert projection_strategy_domains("tmerc", "inverse") == ("tmerc.inverse",)


def test_direct_invalid_setup_is_not_rejected_before_inverse_math():
    geographic = ProjectionParams("longlat", WGS84, north_first=False)
    projected = ProjectionParams("tmerc", WGS84, k_0=0.0, north_first=False)
    pipeline = TransformPipeline(projected, geographic)

    assert pipeline.computed["Qn"] == 0.0
    assert projection_strategy_domain("tmerc", "inverse", pipeline.computed) == "tmerc.inverse"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        longitude, latitude = pipeline.transform(np.array([0.0]), np.array([0.0]), np)
    assert np.isnan(longitude).all()
    assert np.isnan(latitude).all()


@pytest.mark.parametrize(
    ("epsg", "central_meridian", "latitude"),
    [
        (32601, -177.0, np.array([0.0, 40.0, 84.0])),
        (32660, 177.0, np.array([0.0, 40.0, 84.0])),
        (32701, -177.0, np.array([-80.0, -40.0, 0.0])),
        (32760, 177.0, np.array([-80.0, -40.0, 0.0])),
    ],
)
def test_utm_inverse_zone_edges_and_central_meridian_match_proj_and_roundtrip(
    epsg, central_meridian, latitude
):
    target = CRS.from_epsg(epsg)
    longitude = central_meridian + np.array([-3.0, 0.0, 3.0])
    expected_forward = PyProjTransformer.from_crs(target.geodetic_crs, target, always_xy=True)
    expected_inverse = PyProjTransformer.from_crs(target, target.geodetic_crs, always_xy=True)
    actual = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)
    easting, northing = expected_forward.transform(longitude, latitude)

    expected_lon, expected_lat = expected_inverse.transform(easting, northing)
    actual_lon, actual_lat = actual.transform(easting, northing, direction="INVERSE")

    assert_allclose(_angular_difference_degrees(actual_lon, expected_lon), 0.0, atol=4e-14)
    assert_allclose(actual_lat, expected_lat, rtol=0.0, atol=2e-14)
    assert_allclose(_angular_difference_degrees(actual_lon, longitude), 0.0, atol=4e-14)
    assert_allclose(actual_lat, latitude, rtol=0.0, atol=2e-14)


@pytest.mark.parametrize(
    ("target_definition", "longitude", "latitude"),
    [
        (
            "EPSG:2193",
            np.array([166.0, 170.0, 173.0, 176.0, 179.0]),
            np.array([-47.0, -44.0, -41.0, -38.0, -35.0]),
        ),
        (
            "EPSG:2222",
            np.array([-111.2, -110.8, -110.4, -109.9, -109.3]),
            np.array([31.75, 33.0, 34.0, 35.0, 36.5]),
        ),
        (
            "+proj=tmerc +lat_0=10 +lon_0=20 +k=.8 +x_0=12345 +y_0=-67890 "
            "+R=6371000 +units=m +type=crs",
            np.array([10.0, 15.0, 20.0, 25.0, 30.0]),
            np.array([-60.0, -20.0, 10.0, 45.0, 70.0]),
        ),
    ],
)
def test_general_ellipsoidal_and_spherical_inverse_match_proj_and_roundtrip(
    target_definition, longitude, latitude
):
    target = CRS.from_user_input(target_definition)
    expected_forward = PyProjTransformer.from_crs(target.geodetic_crs, target, always_xy=True)
    expected_inverse = PyProjTransformer.from_crs(target, target.geodetic_crs, always_xy=True)
    actual = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)
    easting, northing = expected_forward.transform(longitude, latitude)

    expected_lon, expected_lat = expected_inverse.transform(easting, northing)
    actual_lon, actual_lat = actual.transform(easting, northing, direction="INVERSE")

    assert_allclose(_angular_difference_degrees(actual_lon, expected_lon), 0.0, atol=4e-12)
    assert_allclose(actual_lat, expected_lat, rtol=0.0, atol=3e-12)
    assert_allclose(_angular_difference_degrees(actual_lon, longitude), 0.0, atol=4e-12)
    assert_allclose(actual_lat, latitude, rtol=0.0, atol=3e-12)


@pytest.mark.parametrize("unit", ["m", "us-ft"])
@pytest.mark.parametrize("axis", ["enu", "neu", "swu"])
@pytest.mark.parametrize("always_xy", [False, True])
def test_custom_units_offsets_axis_signs_and_order_match_proj(unit, axis, always_xy):
    target = CRS.from_user_input(
        "+proj=tmerc +lat_0=11 +lon_0=27 +k=.87 +x_0=123 +y_0=-456 "
        f"+ellps=WGS84 +axis={axis} +units={unit} +type=crs"
    )
    source = target.geodetic_crs
    expected_forward = PyProjTransformer.from_crs(source, target, always_xy=always_xy)
    expected_inverse = PyProjTransformer.from_crs(target, source, always_xy=always_xy)
    actual = Transformer.from_crs(source, target, always_xy=always_xy)
    longitude = np.array([20.0, 24.0, 27.0, 30.0, 34.0])
    latitude = np.array([-50.0, -20.0, 11.0, 40.0, 70.0])
    if always_xy:
        first, second = longitude, latitude
    else:
        source_north_first = source.axis_info[0].direction.lower() in {"north", "south"}
        first, second = (latitude, longitude) if source_north_first else (longitude, latitude)

    expected_x, expected_y = expected_forward.transform(first, second)
    actual_x, actual_y = actual.transform(first, second)
    assert_allclose(actual_x, expected_x, rtol=0.0, atol=8e-8)
    assert_allclose(actual_y, expected_y, rtol=0.0, atol=8e-8)

    expected_first, expected_second = expected_inverse.transform(expected_x, expected_y)
    actual_first, actual_second = actual.transform(actual_x, actual_y, direction="INVERSE")
    assert_allclose(actual_first, expected_first, rtol=0.0, atol=4e-12)
    assert_allclose(actual_second, expected_second, rtol=0.0, atol=4e-12)
    assert_allclose(actual_first, first, rtol=0.0, atol=4e-12)
    assert_allclose(actual_second, second, rtol=0.0, atol=4e-12)


def test_inverse_poles_and_adjacent_branch_sides_match_proj_and_fused():
    cp = _cupy_or_skip()
    target = CRS.from_epsg(32631)
    actual = _inverse_transformer(target)
    expected = PyProjTransformer.from_crs(target, target.geodetic_crs, always_xy=True)
    computed = actual._pipeline.computed
    pole_y = computed["y0"] + computed["a"] * (computed["Zb"] + computed["Qn"] * math.pi / 2.0)
    northing = np.array(
        [
            math.nextafter(-pole_y, -math.inf),
            -pole_y,
            math.nextafter(-pole_y, math.inf),
            math.nextafter(pole_y, -math.inf),
            pole_y,
            math.nextafter(pole_y, math.inf),
        ]
    )
    easting = np.full(northing.size, computed["x0"])

    cpu = actual.transform(easting, northing, transcendentals="native")
    proj = expected.transform(easting, northing)
    gpu = actual.transform_buffers(
        cp.asarray(easting),
        cp.asarray(northing),
        precision="fp64",
        transcendentals="native",
    )
    gpu_host = cp.asnumpy(gpu[0]), cp.asnumpy(gpu[1])

    assert_allclose(_angular_difference_degrees(cpu[0], proj[0]), 0.0, atol=4e-14)
    assert_allclose(cpu[1], proj[1], rtol=0.0, atol=4e-14)
    assert_allclose(gpu_host[0], cpu[0], rtol=0.0, atol=3e-14)
    assert_allclose(gpu_host[1], cpu[1], rtol=0.0, atol=3e-14)
    assert cpu[0][1] == pytest.approx(3.0, abs=1e-14)
    assert cpu[0][4] == pytest.approx(3.0, abs=1e-14)
    assert abs(cpu[1][1]) == 90.0
    assert abs(cpu[1][4]) == 90.0
    assert cpu[0][0] == -177.0
    assert cpu[0][5] == -177.0


def test_inverse_center_tiny_values_signed_zero_and_fused_parity():
    cp = _cupy_or_skip()
    target = CRS.from_user_input(
        "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +type=crs"
    )
    actual = _inverse_transformer(target)
    expected = PyProjTransformer.from_crs(target, target.geodetic_crs, always_xy=True)
    tiny = np.nextafter(0.0, 1.0)
    easting = np.array([0.0, -0.0, tiny, -tiny, 1e-300, -1e-300, 1e-20, -1e-20])
    northing = np.array([0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 1e-20, -1e-20])

    cpu = actual.transform(easting, northing, transcendentals="native")
    proj = expected.transform(easting, northing)
    gpu = actual.transform_buffers(
        cp.asarray(easting),
        cp.asarray(northing),
        precision="fp64",
        transcendentals="native",
    )
    gpu_host = cp.asnumpy(gpu[0]), cp.asnumpy(gpu[1])

    assert_allclose(cpu[0], proj[0], rtol=0.0, atol=2e-320)
    assert_allclose(cpu[1], proj[1], rtol=0.0, atol=2e-320)
    assert_array_equal(gpu_host[0], cpu[0])
    assert_array_equal(gpu_host[1], cpu[1])
    assert_array_equal(np.signbit(cpu[0]), [False, False, False, False, False, True, False, True])
    assert_array_equal(np.signbit(cpu[1]), [False, False, False, False, False, False, False, True])


def test_closed_proj_normalized_easting_limit_and_missing_outside_guard():
    cp = _cupy_or_skip()
    target = CRS.from_epsg(32631)
    actual = _inverse_transformer(target)
    expected = PyProjTransformer.from_crs(target, target.geodetic_crs, always_xy=True)
    computed = actual._pipeline.computed
    ce = np.array(
        [
            -PROJ_ETMERC_CE_LIMIT,
            PROJ_ETMERC_CE_LIMIT,
            math.nextafter(PROJ_ETMERC_CE_LIMIT, math.inf),
        ]
    )
    easting = computed["x0"] + computed["a"] * computed["Qn"] * ce
    northing = np.full(ce.size, computed["y0"] + computed["a"] * computed["Zb"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cpu = actual.transform(easting, northing, transcendentals="native")
    proj = expected.transform(easting, northing)
    gpu = actual.transform_buffers(
        cp.asarray(easting),
        cp.asarray(northing),
        precision="fp64",
        transcendentals="native",
    )
    gpu_host = cp.asnumpy(gpu[0]), cp.asnumpy(gpu[1])

    assert_allclose(cpu[0][:2], proj[0][:2], rtol=0.0, atol=3e-14)
    assert_allclose(cpu[1][:2], proj[1][:2], rtol=0.0, atol=0.0)
    assert_allclose(gpu_host[0], cpu[0], rtol=0.0, atol=2e-14)
    assert_allclose(gpu_host[1], cpu[1], rtol=0.0, atol=0.0)
    # Pre-existing blocker: the core omits PROJ's closed |Ce| guard, so the
    # first representable value outside it looks valid instead of failing.
    assert np.isfinite(cpu[0][2]) and np.isfinite(cpu[1][2])
    assert math.isinf(proj[0][2]) and proj[0][2] > 0.0
    assert math.isinf(proj[1][2]) and proj[1][2] > 0.0


def test_extended_and_huge_finite_northings_pin_native_parity_and_proj_gap():
    cp = _cupy_or_skip()
    target = CRS.from_epsg(32631)
    actual = _inverse_transformer(target)
    expected = PyProjTransformer.from_crs(target, target.geodetic_crs, always_xy=True)
    computed = actual._pipeline.computed
    northing = np.array([1e8, -1e8, 1e20, -1e20, 1e100, -1e100, 1e300, -1e300, np.finfo(float).max])
    easting = np.full(northing.size, computed["x0"])

    cpu = actual.transform(easting, northing, transcendentals="native")
    proj = expected.transform(easting, northing)
    gpu = actual.transform_buffers(
        cp.asarray(easting),
        cp.asarray(northing),
        precision="fp64",
        transcendentals="native",
    )
    gpu_host = cp.asnumpy(gpu[0]), cp.asnumpy(gpu[1])

    assert np.isfinite(cpu[0]).all() and np.isfinite(cpu[1]).all()
    assert_allclose(gpu_host[0], cpu[0], rtol=0.0, atol=0.0)
    assert_allclose(gpu_host[1], cpu[1], rtol=0.0, atol=2e-14)
    assert_allclose(_angular_difference_degrees(cpu[0][:2], proj[0][:2]), 0.0, atol=4e-14)
    assert_allclose(cpu[1][:2], proj[1][:2], rtol=0.0, atol=2e-13)
    # Argument reduction for unbounded Cn is not a PROJ-compatible domain.
    assert abs(cpu[1][2] - proj[1][2]) > 0.1
    assert abs(_angular_difference_degrees(cpu[0][6], proj[0][6])) > 170.0
    assert abs(cpu[1][6] - proj[1][6]) > 60.0


def test_spherical_large_easting_exp_overflow_is_a_proj_compatibility_gap():
    cp = _cupy_or_skip()
    target = CRS.from_user_input(
        "+proj=tmerc +lat_0=10 +lon_0=20 +k=.8 +x_0=12345 +y_0=-67890 +R=6371000 +units=m +type=crs"
    )
    actual = _inverse_transformer(target)
    expected = PyProjTransformer.from_crs(target, target.geodetic_crs, always_xy=True)
    computed = actual._pipeline.computed
    # exp(2*Ce) is finite at 354 and overflows at 355. With every spherical
    # series coefficient zero, PROJ still reaches the finite +/-90-degree
    # longitude asymptotes; the shared Engsager expression instead creates
    # zero-times-infinity intermediates and returns NaN atomically.
    ce = np.array([-355.0, -354.0, 354.0, 355.0])
    easting = computed["x0"] + computed["a"] * computed["Qn"] * ce
    northing = np.full(ce.size, computed["y0"] + computed["a"] * computed["Zb"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cpu = actual.transform(easting, northing, transcendentals="native")
    proj = expected.transform(easting, northing)
    gpu = actual.transform_buffers(
        cp.asarray(easting),
        cp.asarray(northing),
        precision="fp64",
        transcendentals="native",
    )
    gpu_host = cp.asnumpy(gpu[0]), cp.asnumpy(gpu[1])

    assert_allclose(cpu[0][1:3], proj[0][1:3], rtol=0.0, atol=0.0)
    assert_allclose(cpu[1][1:3], proj[1][1:3], rtol=0.0, atol=0.0)
    assert np.isnan(cpu[0][[0, 3]]).all() and np.isnan(cpu[1][[0, 3]]).all()
    assert np.isfinite(proj[0]).all() and np.isfinite(proj[1]).all()
    assert_array_equal(np.isnan(gpu_host[0]), np.isnan(cpu[0]))
    assert_array_equal(np.isnan(gpu_host[1]), np.isnan(cpu[1]))
    assert_allclose(gpu_host[0][1:3], cpu[0][1:3], rtol=0.0, atol=0.0)
    assert_allclose(gpu_host[1][1:3], cpu[1][1:3], rtol=0.0, atol=0.0)


def test_full_nan_inf_matrix_pins_cpu_fused_classes_and_proj_difference():
    cp = _cupy_or_skip()
    target = CRS.from_epsg(32631)
    actual = _inverse_transformer(target)
    expected = PyProjTransformer.from_crs(target, target.geodetic_crs, always_xy=True)
    computed = actual._pipeline.computed
    x_values = np.array([computed["x0"], math.inf, -math.inf, math.nan])
    y_values = np.array([computed["y0"], math.inf, -math.inf, math.nan])
    easting = np.tile(x_values, y_values.size)
    northing = np.repeat(y_values, x_values.size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cpu = actual.transform(easting, northing, transcendentals="native")
    proj = expected.transform(easting, northing)
    gpu = actual.transform_buffers(
        cp.asarray(easting),
        cp.asarray(northing),
        precision="fp64",
        transcendentals="native",
    )
    gpu_host = cp.asnumpy(gpu[0]), cp.asnumpy(gpu[1])

    assert_allclose(gpu_host[0], cpu[0], rtol=0.0, atol=0.0, equal_nan=True)
    assert_allclose(gpu_host[1], cpu[1], rtol=0.0, atol=0.0, equal_nan=True)
    assert np.isfinite(cpu[0][0]) and np.isfinite(cpu[1][0])
    assert np.isnan(cpu[0][1:]).all() and np.isnan(cpu[1][1:]).all()
    # PROJ uses positive infinity for material infinite inputs, while NaN has
    # precedence when either component is NaN. vibeProj currently returns NaN
    # atomically for every non-finite pair.
    assert math.isinf(proj[0][1]) and math.isinf(proj[1][1])
    assert math.isinf(proj[0][4]) and math.isinf(proj[1][4])
    nan_input = np.isnan(easting) | np.isnan(northing)
    assert np.isnan(proj[0][nan_input]).all()
    assert np.isnan(proj[1][nan_input]).all()


def test_inverse_transcendental_policies_all_resolve_native():
    target = CRS.from_epsg(32631)
    transformer = _inverse_transformer(target)
    pipeline = transformer._pipeline
    for policy in ("auto", "native", "accelerated"):
        context = pipeline.build_execution_context(
            precision="fp64",
            transcendentals=policy,
            device=transformer._input_device_capability(np.array([0.0]), np),
            workload_size=1_000_000,
        )
        assert len(context.projection_implementations) == 1
        implementation = context.projection_implementations[0]
        assert implementation.projection == "tmerc"
        assert implementation.direction == "inverse"
        assert implementation.domain == "tmerc.inverse"
        assert implementation.implementation_id == NATIVE_LIBDEVICE


def test_spherical_core_inverse_roundtrip_is_reachable_without_crs_resolution():
    projected = ProjectionParams("tmerc", SPHERE, lon_0=9.0, lat_0=12.0, k_0=0.9)
    geographic = ProjectionParams("longlat", SPHERE, north_first=False)
    forward = TransformPipeline(geographic, projected)
    inverse = TransformPipeline(projected, geographic)
    longitude = np.array([-0.1, 9.0, 18.1])
    latitude = np.array([-70.0, 12.0, 70.0])

    easting, northing = forward.transform(longitude, latitude, np)
    actual_lon, actual_lat = inverse.transform(easting, northing, np)

    assert_allclose(actual_lon, longitude, rtol=0.0, atol=4e-14)
    assert_allclose(actual_lat, latitude, rtol=0.0, atol=2e-14)
