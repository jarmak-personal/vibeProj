"""CUDA qualification tests for the orthographic inverse reframe."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device not available", allow_module_level=True)
except Exception as exc:  # pragma: no cover - import-time environment guard
    pytest.skip(f"CUDA device not available: {exc}", allow_module_level=True)

import vibeproj.fused_kernels as fused_module  # noqa: E402
from vibeproj.crs import ProjectionParams  # noqa: E402
from vibeproj.ellipsoid import Ellipsoid  # noqa: E402
from vibeproj.pipeline import TransformPipeline  # noqa: E402
from vibeproj.transcendentals import (  # noqa: E402
    NATIVE_LIBDEVICE,
    ORTHO_INVERSE_GUARDED_REFRAME,
    PROJECTION_FIXED_Q62_MAX_SCALE_M,
)


RADIUS_M = 6_378_137.0


def _pipeline(*, lat_0: float = 0.0, radius: float = RADIUS_M):
    sphere = Ellipsoid(a=radius, b=radius, f=0.0, e=0.0, es=0.0, n=0.0)
    geographic = ProjectionParams(
        projection_name="longlat",
        ellipsoid=sphere,
        north_first=False,
    )
    projected = ProjectionParams(
        projection_name="ortho",
        ellipsoid=sphere,
        lat_0=lat_0,
        north_first=False,
    )
    return TransformPipeline(projected, geographic)


def _run(pipeline, x, y, implementation_id):
    return fused_module.fused_transform(
        x,
        y,
        projection_name="ortho",
        direction="inverse",
        computed=pipeline.computed,
        src_north_first=pipeline.src_north_first,
        dst_north_first=pipeline.dst_north_first,
        xp=cp,
        precision="fp64",
        transcendental_impl=implementation_id,
    )


def _geographic_error_m(actual, reference, radius):
    actual_lon = cp.asnumpy(actual[0])
    actual_lat = cp.asnumpy(actual[1])
    reference_lon = cp.asnumpy(reference[0])
    reference_lat = cp.asnumpy(reference[1])
    delta_lon = (actual_lon - reference_lon + 180.0) % 360.0 - 180.0
    east = np.deg2rad(delta_lon) * radius * np.cos(np.deg2rad(reference_lat))
    north = np.deg2rad(actual_lat - reference_lat) * radius
    return np.hypot(east, north)


def test_ortho_inverse_reframe_compiles_as_distinct_exact_fp64_variant():
    fused_module._kernel_cache.clear()
    native = fused_module._get_kernel(
        "ortho", "inverse", "float64", transcendental_impl=NATIVE_LIBDEVICE
    )
    accelerated = fused_module._get_kernel(
        "ortho",
        "inverse",
        "float64",
        transcendental_impl=ORTHO_INVERSE_GUARDED_REFRAME,
    )
    native.compile()
    accelerated.compile()

    assert native is not accelerated
    assert accelerated.attributes["shared_size_bytes"] == 0
    assert accelerated.attributes["max_threads_per_block"] == 256


def test_ortho_inverse_reframe_rejects_wrong_precision_and_operation():
    for compute_dtype in ("float32", "ds"):
        with pytest.raises(ValueError, match="qualified only"):
            fused_module._get_kernel(
                "ortho",
                "inverse",
                compute_dtype,
                transcendental_impl=ORTHO_INVERSE_GUARDED_REFRAME,
            )
    with pytest.raises(ValueError, match="qualified only"):
        fused_module._get_kernel(
            "ortho",
            "forward",
            "float64",
            transcendental_impl=ORTHO_INVERSE_GUARDED_REFRAME,
        )


@pytest.mark.parametrize(
    "radius",
    [
        1.0,
        3_396_190.0,
        RADIUS_M,
        np.nextafter(PROJECTION_FIXED_Q62_MAX_SCALE_M, 0.0),
        PROJECTION_FIXED_Q62_MAX_SCALE_M,
    ],
)
def test_ortho_equatorial_reframe_meets_ten_nanometer_contract_at_qualified_scales(radius):
    rng = np.random.default_rng(20260805)
    count = 300_000
    rho = np.sqrt(rng.uniform(1e-16, 0.99, count))
    azimuth = rng.uniform(-math.pi, math.pi, count)
    x = cp.asarray(rho * np.cos(azimuth) * radius)
    y = cp.asarray(rho * np.sin(azimuth) * radius)
    pipeline = _pipeline(radius=radius)
    native = _run(pipeline, x, y, NATIVE_LIBDEVICE)
    accelerated = _run(pipeline, x, y, ORTHO_INVERSE_GUARDED_REFRAME)

    error = _geographic_error_m(accelerated, native, radius)
    assert float(np.max(error)) <= 1e-8
    assert float(np.percentile(error, 99)) <= 1e-8


@pytest.mark.parametrize(
    "radius",
    [
        np.nextafter(PROJECTION_FIXED_Q62_MAX_SCALE_M, math.inf),
        10_000_000.0,
        math.inf,
        math.nan,
    ],
)
def test_ortho_reframe_above_scale_domain_is_bit_exact_native(radius):
    input_scale = RADIUS_M if not math.isfinite(radius) else radius
    pipeline = _pipeline(radius=radius)
    x = cp.asarray([0.1, -0.3, 0.7], dtype=cp.float64) * input_scale
    y = cp.asarray([0.2, 0.4, -0.1], dtype=cp.float64) * input_scale
    native = _run(pipeline, x, y, NATIVE_LIBDEVICE)
    accelerated = _run(pipeline, x, y, ORTHO_INVERSE_GUARDED_REFRAME)

    for expected, actual in zip(native, accelerated, strict=True):
        assert_array_equal(cp.asnumpy(actual).view(np.uint64), cp.asnumpy(expected).view(np.uint64))


@pytest.mark.parametrize("lat_0", [-90.0, -45.0, 0.0, 45.0, 90.0])
def test_ortho_guarded_reframe_property_holds_across_origin_modes(lat_0):
    rng = np.random.default_rng(20260806 + int(lat_0))
    count = 100_000
    rho = np.sqrt(rng.uniform(1e-16, 0.99, count))
    azimuth = rng.uniform(-math.pi, math.pi, count)
    x = cp.asarray(rho * np.cos(azimuth) * RADIUS_M)
    y = cp.asarray(rho * np.sin(azimuth) * RADIUS_M)
    pipeline = _pipeline(lat_0=lat_0)
    native = _run(pipeline, x, y, NATIVE_LIBDEVICE)
    accelerated = _run(pipeline, x, y, ORTHO_INVERSE_GUARDED_REFRAME)

    assert float(np.max(_geographic_error_m(accelerated, native, RADIUS_M))) <= 1e-8


def test_ortho_guard_edges_axes_signed_zero_nonfinite_and_invalid_are_bit_exact():
    guard_radius = math.sqrt(0.99) * RADIUS_M
    x_host = np.array(
        [
            0.0,
            -0.0,
            0.0,
            -0.0,
            0.5 * RADIUS_M,
            -0.5 * RADIUS_M,
            0.0,
            0.0,
            guard_radius,
            np.nextafter(guard_radius, math.inf),
            RADIUS_M,
            np.nextafter(RADIUS_M, math.inf),
            1.1 * RADIUS_M,
            math.inf,
            -math.inf,
            math.nan,
        ],
        dtype=np.float64,
    )
    y_host = np.array(
        [0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.5 * RADIUS_M, -0.5 * RADIUS_M] + [0.0] * 8,
        dtype=np.float64,
    )
    x = cp.asarray(x_host)
    y = cp.asarray(y_host)
    pipeline = _pipeline()
    native = _run(pipeline, x, y, NATIVE_LIBDEVICE)
    accelerated = _run(pipeline, x, y, ORTHO_INVERSE_GUARDED_REFRAME)

    for expected, actual in zip(native, accelerated, strict=True):
        assert_array_equal(cp.asnumpy(actual).view(np.uint64), cp.asnumpy(expected).view(np.uint64))


def test_ortho_near_center_rho_squared_guard_exact_and_adjacent_values():
    pipeline = _pipeline(radius=1.0)
    component = np.float64(math.sqrt(0.5e-16))
    probes = (
        (np.nextafter(component, 0.0), np.nextafter(1e-16, 0.0), False),
        (component, np.float64(1e-16), False),
        (np.nextafter(component, math.inf), np.nextafter(1e-16, math.inf), True),
    )
    for x_value, expected_rho_squared, uses_fast_path in probes:
        actual_rho_squared = np.float64(x_value * x_value + component * component)
        assert actual_rho_squared == expected_rho_squared
        x = cp.asarray([x_value])
        y = cp.asarray([component])
        native = _run(pipeline, x, y, NATIVE_LIBDEVICE)
        accelerated = _run(pipeline, x, y, ORTHO_INVERSE_GUARDED_REFRAME)
        native_bits = tuple(cp.asnumpy(value).view(np.uint64) for value in native)
        accelerated_bits = tuple(cp.asnumpy(value).view(np.uint64) for value in accelerated)
        if uses_fast_path:
            assert any(
                not np.array_equal(actual, expected)
                for actual, expected in zip(accelerated_bits, native_bits, strict=True)
            )
            assert float(np.max(_geographic_error_m(accelerated, native, 1.0))) <= 1e-8
        else:
            for expected, actual in zip(native_bits, accelerated_bits, strict=True):
                assert_array_equal(actual, expected)


@pytest.mark.parametrize("x_sign", [-1.0, 1.0])
@pytest.mark.parametrize("y_sign", [-1.0, 1.0])
def test_ortho_upper_guard_and_latitude_corner_exact_and_adjacent_values(x_sign, y_sign):
    pipeline = _pipeline(radius=1.0)
    y_value = np.float64(y_sign * 0.95)
    x_values = (
        np.float64(x_sign * 0.29580398915498063),
        np.float64(x_sign * 0.29580398915498085),
        np.float64(x_sign * 0.295803989154981),
    )
    expected_rho_squared = (
        np.nextafter(0.99, 0.0),
        np.float64(0.99),
        np.nextafter(0.99, math.inf),
    )
    for index, (x_value, expected_rho) in enumerate(
        zip(x_values, expected_rho_squared, strict=True)
    ):
        actual_rho = np.float64(x_value * x_value + y_value * y_value)
        assert actual_rho == expected_rho
        x = cp.asarray([x_value])
        y = cp.asarray([y_value])
        native = _run(pipeline, x, y, NATIVE_LIBDEVICE)
        accelerated = _run(pipeline, x, y, ORTHO_INVERSE_GUARDED_REFRAME)
        if index == 2:
            for expected, actual in zip(native, accelerated, strict=True):
                assert_array_equal(
                    cp.asnumpy(actual).view(np.uint64), cp.asnumpy(expected).view(np.uint64)
                )
        else:
            assert float(np.max(_geographic_error_m(accelerated, native, 1.0))) <= 1e-8
