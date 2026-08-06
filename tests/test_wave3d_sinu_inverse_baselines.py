"""Wave 3D native-contract baselines for ellipsoidal Sinusoidal inverse."""

from __future__ import annotations

import ast
import inspect
import math
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyproj import CRS
from pyproj import Transformer as PyProjTransformer

from vibeproj import Transformer
from vibeproj.crs import ProjectionParams, resolve_projection_params
from vibeproj.ellipsoid import Ellipsoid
import vibeproj.fused_kernels as fused_module
from vibeproj.projections.sinusoidal import (
    SINU_MAX_ECCENTRICITY_SQUARED,
    SINU_MAX_SEMI_MAJOR_AXIS_M,
    Sinusoidal,
    _meridional_arc,
)


ELLIPSOIDAL_SINU = "ESRI:54008"


def _cupy_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device not available")
    except Exception as exc:
        pytest.skip(f"CUDA device not available: {exc}")
    return cp


def _transformer(*, always_xy: bool = True) -> tuple[CRS, Transformer]:
    target = CRS.from_user_input(ELLIPSOIDAL_SINU)
    return target, Transformer.from_crs(target.geodetic_crs, target, always_xy=always_xy)


def _public_pole_y(transformer: Transformer, sign: float) -> float:
    computed = transformer._pipeline.computed
    physical_y = computed["y0"] + sign * computed["a"] * computed["meridional_pole"]
    return physical_y / (computed.get("northing_axis_sign", 1.0) * computed.get("y_unit_to_m", 1.0))


def _ellipsoid_with_es(es: float, *, a: float = SINU_MAX_SEMI_MAJOR_AXIS_M) -> Ellipsoid:
    eccentricity = math.sqrt(es)
    flattening = 1.0 - math.sqrt(1.0 - es)
    return Ellipsoid(
        a=a,
        b=a * (1.0 - flattening),
        f=flattening,
        e=eccentricity,
        es=es,
        n=flattening / (2.0 - flattening),
    )


def test_sinu_setup_precomputes_recurrence_hot_meridional_limit():
    _, transformer = _transformer()
    computed = transformer._pipeline.computed
    expected = _meridional_arc(math.radians(89.9), computed["meridional_coefficients"], math)

    assert computed["meridional_recurrence_hot_cy_limit"] == expected


def test_sinu_dispatch_isolates_recurrence_only_host_packing():
    source = inspect.getsource(fused_module.fused_transform)
    direct_packing = """args = _with_units(
                    real_t(computed["es"]),
                    *(real_t(value) for value in computed["meridional_coefficients"]),
                    real_t(computed["lam0"]),
                    real_t(computed["a"]),
                    real_t(computed["x0"]),
                    real_t(computed["y0"]),
                )"""
    tree = ast.parse(source)
    sinu_branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "projection_name"
        and any(
            isinstance(comparator, ast.Constant) and comparator.value == "sinu"
            for comparator in node.test.comparators
        )
    )
    sinu_body = ast.Module(body=sinu_branch.body, type_ignores=[])

    assert not any(isinstance(node, ast.List) for node in ast.walk(sinu_body))
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sin"
        for node in ast.walk(sinu_body)
    )
    assert source.count(direct_packing) == 1
    assert source.count('computed["meridional_recurrence_hot_cy_limit"]') == 1


@pytest.mark.parametrize("pole_sign", [-1.0, 1.0])
def test_ellipsoidal_sinu_inverse_both_meridional_pole_boundaries(pole_sign):
    _, transformer = _transformer()
    pole_y = _public_pole_y(transformer, pole_sign)
    inside = math.nextafter(pole_y, 0.0)
    outside = math.nextafter(pole_y, math.copysign(math.inf, pole_sign))

    with pytest.warns(UserWarning, match="non-finite"):
        longitude, latitude = transformer.transform(
            np.zeros(3), np.array([inside, pole_y, outside]), direction="INVERSE"
        )

    assert np.all(np.isfinite(longitude[:2]))
    assert np.all(np.isfinite(latitude[:2]))
    assert latitude[1] == pytest.approx(pole_sign * 90.0, abs=2e-13)
    assert math.isinf(longitude[2]) and longitude[2] > 0.0
    assert math.isinf(latitude[2]) and latitude[2] > 0.0


@pytest.mark.parametrize("pole_sign", [-1.0, 1.0])
def test_ellipsoidal_sinu_inverse_nonzero_x_at_and_adjacent_to_poles_is_stable(pole_sign):
    _, transformer = _transformer()
    pole_y = _public_pole_y(transformer, pole_sign)
    adjacent_y = math.nextafter(pole_y, 0.0)
    easting = np.array([0.0, 1.0, 10_000_000.0, 1.0e20])

    pole_longitude, pole_latitude = transformer.transform(
        easting, np.full(easting.size, pole_y), direction="INVERSE"
    )
    adjacent_longitude, adjacent_latitude = transformer.transform(
        easting, np.full(easting.size, adjacent_y), direction="INVERSE"
    )

    # The native contract accepts the exact meridional endpoint for every
    # finite x. Longitude is ill-conditioned there but remains finite and is
    # normalized by the common inverse postamble.
    assert np.all(np.isfinite(pole_longitude))
    assert np.all(np.isfinite(pole_latitude))
    assert_allclose(pole_latitude, pole_sign * 90.0, rtol=0.0, atol=2e-13)
    assert np.all(np.isfinite(adjacent_longitude))
    assert np.all(np.isfinite(adjacent_latitude))
    assert np.all(np.abs(adjacent_latitude) <= 90.0)
    assert np.all(np.signbit(adjacent_latitude) == (pole_sign < 0.0))


def test_ellipsoidal_sinu_inverse_arbitrary_valid_image_matches_pyproj():
    target, transformer = _transformer()
    expected = PyProjTransformer.from_crs(target, target.geodetic_crs, always_xy=True)
    computed = transformer._pipeline.computed
    latitude = np.array([-89.999999, -80.0, -45.0, 0.0, 45.0, 80.0, 89.999999])
    longitude = np.array([-179.999, -140.0, -30.0, 0.0, 75.0, 140.0, 179.999])
    phi = np.deg2rad(latitude)
    lam = np.deg2rad(longitude)
    sin_phi = np.sin(phi)
    x_normalized = lam * np.cos(phi) / np.sqrt(1.0 - computed["es"] * sin_phi * sin_phi)
    y_normalized = _meridional_arc(phi, computed["meridional_coefficients"], np)
    easting = computed["x0"] + computed["a"] * x_normalized
    northing = computed["y0"] + computed["a"] * y_normalized

    expected_lon, expected_lat = expected.transform(easting, northing)
    actual_lon, actual_lat = transformer.transform(easting, northing, direction="INVERSE")

    assert_allclose(actual_lon, expected_lon, rtol=0.0, atol=3e-11)
    assert_allclose(actual_lat, expected_lat, rtol=0.0, atol=3e-12)
    assert_allclose(actual_lon, longitude, rtol=0.0, atol=3e-11)
    assert_allclose(actual_lat, latitude, rtol=0.0, atol=3e-12)


def test_ellipsoidal_sinu_inverse_longitude_branch_and_wrap_contract():
    _, transformer = _transformer()
    computed = transformer._pipeline.computed
    relative_longitude = np.array(
        [
            math.nextafter(-math.pi, -math.inf),
            -math.pi,
            math.nextafter(-math.pi, math.inf),
            math.nextafter(math.pi, -math.inf),
            math.pi,
            math.nextafter(math.pi, math.inf),
            -3.0 * math.pi,
            3.0 * math.pi,
        ]
    )
    easting = computed["x0"] + computed["a"] * relative_longitude
    northing = np.full(easting.size, computed["y0"])

    longitude, latitude = transformer.transform(easting, northing, direction="INVERSE")
    # Public scale removal reconstructs normalized x after the `a * x`
    # multiplication above. At exact +/-pi that rounding can land on the
    # adjacent side of the branch, so pin the value that reaches the core.
    reconstructed = (easting - computed["x0"]) / computed["a"]
    wrapped = reconstructed - 2.0 * math.pi * np.round(reconstructed / (2.0 * math.pi))

    assert_allclose(longitude, np.rad2deg(wrapped), rtol=0.0, atol=6e-14)
    assert_allclose(latitude, 0.0, rtol=0.0, atol=0.0)
    assert longitude[1] == pytest.approx(180.0, abs=6e-14)
    assert longitude[4] == pytest.approx(-180.0, abs=6e-14)
    assert longitude[6] == 180.0
    assert longitude[7] == -180.0


def test_ellipsoidal_sinu_inverse_full_nonfinite_matrix_pins_native_precedence():
    _, transformer = _transformer()
    pole_y = _public_pole_y(transformer, 1.0)
    x_values = np.array([0.0, 1.0, math.inf, -math.inf, math.nan])
    y_values = np.array([0.0, math.nextafter(pole_y, 0.0), math.inf, -math.inf, math.nan])
    easting = np.tile(x_values, y_values.size)
    northing = np.repeat(y_values, x_values.size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        longitude, latitude = transformer.transform(easting, northing, direction="INVERSE")

    for index, (x_value, y_value) in enumerate(zip(easting, northing, strict=True)):
        if math.isinf(y_value):
            # The material northing-domain guard wins over x NaN/Inf and is
            # deliberately atomic and positive.
            assert math.isinf(longitude[index]) and longitude[index] > 0.0
            assert math.isinf(latitude[index]) and latitude[index] > 0.0
        elif math.isnan(y_value):
            assert math.isnan(longitude[index])
            assert math.isnan(latitude[index])
        elif math.isnan(x_value):
            assert math.isnan(longitude[index])
            assert math.isfinite(latitude[index])
        elif math.isinf(x_value):
            assert longitude[index] == x_value
            assert math.isfinite(latitude[index])
        else:
            assert math.isfinite(longitude[index])
            assert math.isfinite(latitude[index])


@pytest.mark.parametrize("unit", ["m", "us-ft"])
@pytest.mark.parametrize("axis", ["enu", "swu"])
@pytest.mark.parametrize("always_xy", [False, True])
def test_custom_ellipsoidal_sinu_units_offsets_axis_signs_and_order_match_pyproj(
    unit, axis, always_xy
):
    target = CRS.from_user_input(
        f"+proj=sinu +lon_0=12 +x_0=100 +y_0=-200 +ellps=WGS84 +axis={axis} +units={unit} +type=crs"
    )
    source = target.geodetic_crs
    params = resolve_projection_params(target)
    expected_forward = PyProjTransformer.from_crs(source, target, always_xy=always_xy)
    expected_inverse = PyProjTransformer.from_crs(target, source, always_xy=always_xy)
    transformer = Transformer.from_crs(source, target, always_xy=always_xy)
    longitude = np.array([-160.0, -20.0, 12.0, 80.0, 170.0])
    latitude = np.array([-80.0, -35.0, 0.0, 45.0, 80.0])
    if always_xy:
        first, second = longitude, latitude
    else:
        source_north_first = source.axis_info[0].direction.lower() in {"north", "south"}
        first, second = (latitude, longitude) if source_north_first else (longitude, latitude)

    expected_first, expected_second = expected_forward.transform(first, second)
    actual_first, actual_second = transformer.transform(first, second)
    assert_allclose(actual_first, expected_first, rtol=0.0, atol=8e-8)
    assert_allclose(actual_second, expected_second, rtol=0.0, atol=8e-8)

    expected_geo = expected_inverse.transform(expected_first, expected_second)
    actual_geo = transformer.transform(actual_first, actual_second, direction="INVERSE")
    assert_allclose(actual_geo[0], expected_geo[0], rtol=0.0, atol=3e-12)
    assert_allclose(actual_geo[1], expected_geo[1], rtol=0.0, atol=3e-12)
    assert_allclose(actual_geo[0], first, rtol=0.0, atol=3e-12)
    assert_allclose(actual_geo[1], second, rtol=0.0, atol=3e-12)

    expected_sign = -1.0 if axis == "swu" else 1.0
    assert params.easting_axis_sign == expected_sign
    assert params.northing_axis_sign == expected_sign
    assert params.visualization_north_first is (axis == "swu")


@pytest.mark.parametrize(
    "eccentricity_squared",
    [
        math.nextafter(0.0, math.inf),
        1.0e-16,
        0.0066943799901413165,
        math.nextafter(SINU_MAX_ECCENTRICITY_SQUARED, -math.inf),
        SINU_MAX_ECCENTRICITY_SQUARED,
    ],
)
@pytest.mark.parametrize(
    "semi_major_axis",
    [math.nextafter(0.0, math.inf), 1.0, SINU_MAX_SEMI_MAJOR_AXIS_M],
)
def test_ellipsoidal_sinu_inverse_geometry_seams_roundtrip_normalized_math(
    eccentricity_squared, semi_major_axis
):
    ellipsoid = _ellipsoid_with_es(eccentricity_squared, a=semi_major_axis)
    params = ProjectionParams("sinu", ellipsoid)
    projection = Sinusoidal()
    computed = projection.setup(params)
    longitude = np.deg2rad(np.array([-170.0, -30.0, 0.0, 75.0, 170.0]))
    latitude = np.deg2rad(np.array([-89.999, -45.0, 0.0, 45.0, 89.999]))

    x, y = projection.forward(longitude, latitude, params, computed, np)
    actual_lon, actual_lat = projection.inverse(x, y, params, computed, np)

    assert_allclose(actual_lon, longitude, rtol=0.0, atol=8e-13)
    assert_allclose(actual_lat, latitude, rtol=0.0, atol=8e-16)
    assert computed["a"] == semi_major_axis
    assert computed["es"] == eccentricity_squared


def test_ellipsoidal_sinu_inverse_gpu_native_guards_buffers_and_stream_match_cpu():
    cp = _cupy_or_skip()
    _, transformer = _transformer()
    north_pole = _public_pole_y(transformer, 1.0)
    south_pole = _public_pole_y(transformer, -1.0)
    easting = np.array([0.0, 1.0, 1.0e7, math.inf, -math.inf, math.nan, math.nan])
    northing = np.array(
        [
            0.0,
            math.nextafter(north_pole, 0.0),
            math.nextafter(south_pole, 0.0),
            0.0,
            0.0,
            math.nan,
            math.inf,
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected_lon, expected_lat = transformer.transform(
            easting, northing, direction="INVERSE", transcendentals="native"
        )

    gpu_x = cp.asarray(easting)
    gpu_y = cp.asarray(northing)
    out_lon = cp.empty_like(gpu_x)
    out_lat = cp.empty_like(gpu_y)
    stream = cp.cuda.Stream(non_blocking=True)
    actual_lon, actual_lat = transformer.transform_buffers(
        gpu_x,
        gpu_y,
        direction="INVERSE",
        precision="fp64",
        transcendentals="native",
        out_x=out_lon,
        out_y=out_lat,
        stream=stream,
    )
    stream.synchronize()

    assert actual_lon is out_lon
    assert actual_lat is out_lat
    actual_lon_host = cp.asnumpy(actual_lon)
    actual_lat_host = cp.asnumpy(actual_lat)
    assert np.array_equal(np.isfinite(actual_lon_host), np.isfinite(expected_lon))
    assert np.array_equal(np.isnan(actual_lon_host), np.isnan(expected_lon))
    assert np.array_equal(np.signbit(actual_lon_host), np.signbit(expected_lon))
    assert_allclose(actual_lat_host, expected_lat, rtol=0.0, atol=3e-12, equal_nan=True)

    # One ulp inside a pole, tiny CPU/GPU latitude differences make raw
    # longitude ill-conditioned. Compare physical horizontal error there. Any
    # accelerated implementation that promises exact GPU-native longitude must
    # complete-warp fallback throughout this conditioning band.
    finite = np.isfinite(actual_lon_host) & np.isfinite(expected_lon)
    computed = transformer._pipeline.computed
    horizontal_error_m = (
        np.deg2rad(actual_lon_host[finite] - expected_lon[finite])
        * computed["a"]
        * np.cos(np.deg2rad(expected_lat[finite]))
    )
    assert np.max(np.abs(horizontal_error_m)) < 1e-8
