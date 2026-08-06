"""CUDA qualification for broad regular-Mercator transcendental variants."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pyproj import CRS
from pyproj import Transformer as PyProjTransformer

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device not available", allow_module_level=True)
except Exception as exc:  # pragma: no cover - import-time environment guard
    pytest.skip(f"CUDA device not available: {exc}", allow_module_level=True)

import vibeproj.fused_kernels as fused_module  # noqa: E402
from vibeproj import Transformer  # noqa: E402
from vibeproj._conformal import conformal_to_geodetic_coefficients  # noqa: E402
from vibeproj.transcendentals import (  # noqa: E402
    MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY,
    MERC_FORWARD_SPHERICAL_PRODUCT_POLY,
    MERC_FORWARD_SPHERICAL_PRODUCT_POLY_MIN_ELEMENTS,
    MERC_INVERSE_EXP_SERIES,
    MERC_INVERSE_EXP_SERIES_MIN_ELEMENTS,
    NATIVE_LIBDEVICE,
    PROJECTION_FIXED_Q62_MAX_SCALE_M,
)


TARGETS = {
    "ellipsoidal_a": "EPSG:3395",
    "ellipsoidal_b": "EPSG:3994",
    "spherical_a": "+proj=merc +k=0.87 +lon_0=12 +R=6371000 +units=m +type=crs",
    "spherical_b": "+proj=merc +lat_ts=33 +lon_0=12 +R=6371000 +units=m +type=crs",
}


def _transformer(target_definition: str) -> tuple[Transformer, CRS]:
    target = CRS.from_user_input(target_definition)
    return Transformer.from_crs(target.geodetic_crs, target, always_xy=True), target


def _run_exact(
    transformer: Transformer,
    first,
    second,
    direction: str,
    implementation_id: str,
    *,
    computed: dict | None = None,
    out_x=None,
    out_y=None,
    stream=None,
):
    pipeline = transformer._pipeline_for_direction(direction.upper())
    return fused_module.fused_transform(
        first,
        second,
        projection_name="merc",
        direction=direction,
        computed=pipeline.computed if computed is None else computed,
        src_north_first=pipeline.src_north_first,
        dst_north_first=pipeline.dst_north_first,
        xp=cp,
        out_x=out_x,
        out_y=out_y,
        precision="fp64",
        stream=stream,
        transcendental_impl=implementation_id,
    )


def _assert_bitwise_equal(actual, expected) -> None:
    for got, want in zip(actual, expected, strict=True):
        assert_array_equal(cp.asnumpy(got).view(np.uint64), cp.asnumpy(want).view(np.uint64))


def _projected_error_m(actual, expected) -> np.ndarray:
    return np.hypot(
        cp.asnumpy(actual[0] - expected[0]),
        cp.asnumpy(actual[1] - expected[1]),
    )


def _geographic_error_m(actual, expected, scale: float) -> np.ndarray:
    actual_lon, actual_lat = (cp.asnumpy(value) for value in actual)
    expected_lon, expected_lat = (cp.asnumpy(value) for value in expected)
    delta_lon = (actual_lon - expected_lon + 180.0) % 360.0 - 180.0
    east = np.deg2rad(delta_lon) * scale * np.cos(np.deg2rad(expected_lat))
    north = np.deg2rad(actual_lat - expected_lat) * scale
    return np.hypot(east, north)


def test_merc_variants_compile_cache_resources_and_native_inverse_abi():
    fused_module._kernel_cache.clear()
    kernels = {}
    for direction, implementation_id in (
        ("forward", MERC_FORWARD_SPHERICAL_PRODUCT_POLY),
        ("forward", MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY),
        ("inverse", MERC_INVERSE_EXP_SERIES),
    ):
        kernel = fused_module._get_kernel(
            "merc", direction, "float64", transcendental_impl=implementation_id
        )
        kernel.compile()
        kernels[implementation_id] = kernel
        assert kernel.attributes["num_regs"] <= 38
        assert kernel.attributes["local_size_bytes"] <= 40
        assert kernel.attributes["shared_size_bytes"] == 0

    native_source = fused_module._inject_linear_unit_args(
        fused_module._MERC_INVERSE_SOURCE.format(
            real_t="double",
            pi=fused_module._PI_LITERALS["float64"],
            tol=fused_module._TOL_LITERALS["float64"],
        )
    )
    accelerated_source, accelerated_name = fused_module._build_projection_guarded_source(
        "merc", "inverse", MERC_INVERSE_EXP_SERIES
    )
    assert "cgb0" not in native_source
    assert "cgb0" in accelerated_source
    assert accelerated_name == "merc_inverse_exp_series"
    assert len({id(kernel) for kernel in kernels.values()}) == 3


@pytest.mark.parametrize(
    "implementation_id",
    [
        MERC_FORWARD_SPHERICAL_PRODUCT_POLY,
        MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY,
        MERC_INVERSE_EXP_SERIES,
    ],
)
def test_merc_variants_reject_wrong_precision_and_operation(implementation_id):
    expected_direction = "inverse" if implementation_id == MERC_INVERSE_EXP_SERIES else "forward"
    for precision in ("float32", "ds"):
        with pytest.raises(ValueError, match="qualified only"):
            fused_module._get_kernel(
                "merc", expected_direction, precision, transcendental_impl=implementation_id
            )
    wrong_direction = "inverse" if expected_direction == "forward" else "forward"
    with pytest.raises(ValueError, match="qualified only"):
        fused_module._get_kernel(
            "merc", wrong_direction, "float64", transcendental_impl=implementation_id
        )


@pytest.mark.parametrize(
    ("target_name", "wrong_implementation_id", "latitude"),
    [
        ("spherical_a", MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY, -17.0),
        ("ellipsoidal_a", MERC_FORWARD_SPHERICAL_PRODUCT_POLY, 89.95),
    ],
)
def test_merc_forward_geometry_ids_fallback_bitwise_when_called_cross_geometry(
    target_name, wrong_implementation_id, latitude
):
    transformer, _ = _transformer(TARGETS[target_name])
    longitude = cp.full(32, 23.5, dtype=cp.float64)
    latitude_values = cp.full(32, latitude, dtype=cp.float64)
    native = _run_exact(transformer, longitude, latitude_values, "forward", NATIVE_LIBDEVICE)
    wrong = _run_exact(
        transformer,
        longitude,
        latitude_values,
        "forward",
        wrong_implementation_id,
    )
    _assert_bitwise_equal(wrong, native)


@pytest.mark.parametrize("target_definition", TARGETS.values(), ids=TARGETS)
def test_merc_full_domain_forward_inverse_meet_native_and_pyproj_contract(target_definition):
    transformer, target = _transformer(target_definition)
    computed = transformer._pipeline.computed
    scale = float(computed["a"])
    count = 262_144
    longitude = cp.linspace(-179.0, 179.0, count, dtype=cp.float64) + math.degrees(computed["lam0"])
    latitude = cp.linspace(-89.999, 89.999, count, dtype=cp.float64)

    native_forward = _run_exact(transformer, longitude, latitude, "forward", NATIVE_LIBDEVICE)
    forward_id = (
        MERC_FORWARD_SPHERICAL_PRODUCT_POLY
        if computed["e"] == 0.0
        else MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY
    )
    accelerated_forward = _run_exact(transformer, longitude, latitude, "forward", forward_id)
    forward_error_m = np.hypot(
        cp.asnumpy(accelerated_forward[0] - native_forward[0]) * abs(computed["x_unit_to_m"]),
        cp.asnumpy(accelerated_forward[1] - native_forward[1]) * abs(computed["y_unit_to_m"]),
    )
    assert float(np.max(forward_error_m)) <= 1e-8

    native_inverse = _run_exact(transformer, *native_forward, "inverse", NATIVE_LIBDEVICE)
    accelerated_inverse = _run_exact(
        transformer, *native_forward, "inverse", MERC_INVERSE_EXP_SERIES
    )
    assert float(np.max(_geographic_error_m(accelerated_inverse, native_inverse, scale))) <= 1e-8

    oracle_forward = PyProjTransformer.from_crs(target.geodetic_crs, target, always_xy=True)
    sample = np.linspace(0, count - 1, 4096, dtype=np.int64)
    lon_host = cp.asnumpy(longitude)[sample]
    lat_host = cp.asnumpy(latitude)[sample]
    oracle_x, oracle_y = oracle_forward.transform(lon_host, lat_host)
    native_x = cp.asnumpy(native_forward[0])[sample]
    native_y = cp.asnumpy(native_forward[1])[sample]
    accelerated_x = cp.asnumpy(accelerated_forward[0])[sample]
    accelerated_y = cp.asnumpy(accelerated_forward[1])[sample]
    native_oracle_error = np.hypot(native_x - oracle_x, native_y - oracle_y)
    accelerated_oracle_error = np.hypot(accelerated_x - oracle_x, accelerated_y - oracle_y)
    # Near-pole native Mercator differs from pyproj by micrometers. Preserve
    # that reference semantics and permit at most the qualified 10 nm delta.
    assert np.max(accelerated_oracle_error) <= np.max(native_oracle_error) + 1e-8


@pytest.mark.parametrize("direction", ["forward", "inverse"])
@pytest.mark.parametrize("component", [0, 1])
@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("mixed", [False, True])
def test_merc_nonfinite_components_use_warp_atomic_bitwise_native_fallback(
    direction, component, invalid, mixed
):
    transformer, _ = _transformer(TARGETS["ellipsoidal_a"])
    implementation_id = (
        MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY if direction == "forward" else MERC_INVERSE_EXP_SERIES
    )
    if direction == "forward":
        first = cp.full(32, 120.15606267819464, dtype=cp.float64)
        second = cp.full(32, -17.145922217462868, dtype=cp.float64)
    else:
        first = cp.full(32, 11_478_839.31124818, dtype=cp.float64)
        second = cp.full(32, 4_691_815.763041037, dtype=cp.float64)
    target = first if component == 0 else second
    if mixed:
        target[0] = invalid
    else:
        target.fill(invalid)
    native = _run_exact(transformer, first, second, direction, NATIVE_LIBDEVICE)
    accelerated = _run_exact(transformer, first, second, direction, implementation_id)
    _assert_bitwise_equal(accelerated, native)


@pytest.mark.parametrize("direction", ["forward", "inverse"])
@pytest.mark.parametrize("signed_zero", [0.0, -0.0])
def test_merc_signed_zero_is_in_the_hot_accuracy_domain(direction, signed_zero):
    transformer, _ = _transformer(TARGETS["ellipsoidal_a"])
    computed = transformer._pipeline.computed
    if direction == "forward":
        first = cp.full(32, 12.0, dtype=cp.float64)
        second = cp.full(32, signed_zero, dtype=cp.float64)
        implementation_id = MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY
    else:
        first = cp.full(32, computed["x0"] + computed["a"], dtype=cp.float64)
        second = cp.full(32, computed["y0"] + signed_zero, dtype=cp.float64)
        implementation_id = MERC_INVERSE_EXP_SERIES
    native = _run_exact(transformer, first, second, direction, NATIVE_LIBDEVICE)
    accelerated = _run_exact(transformer, first, second, direction, implementation_id)
    error = (
        _projected_error_m(accelerated, native)
        if direction == "forward"
        else _geographic_error_m(accelerated, native, computed["a"])
    )
    assert float(np.max(error)) <= 1e-8


@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_merc_hot_branch_is_nonvacuous(direction):
    transformer, _ = _transformer(TARGETS["ellipsoidal_a"])
    if direction == "forward":
        first = cp.full(32, 23.5, dtype=cp.float64)
        second = cp.full(32, -89.35820071200712, dtype=cp.float64)
        implementation_id = MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY
    else:
        first = cp.full(32, 11_478_839.31124818, dtype=cp.float64)
        second = cp.full(32, 4_691_815.763041037, dtype=cp.float64)
        implementation_id = MERC_INVERSE_EXP_SERIES
    native = _run_exact(transformer, first, second, direction, NATIVE_LIBDEVICE)
    accelerated = _run_exact(transformer, first, second, direction, implementation_id)
    assert any(
        not np.array_equal(cp.asnumpy(got).view(np.uint64), cp.asnumpy(want).view(np.uint64))
        for got, want in zip(accelerated, native, strict=True)
    )
    error = (
        _projected_error_m(accelerated, native)
        if direction == "forward"
        else _geographic_error_m(accelerated, native, 6_378_137.0)
    )
    assert float(np.max(error)) <= 1e-8


@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_merc_invalid_setup_is_bitwise_native(direction):
    transformer, _ = _transformer(TARGETS["ellipsoidal_a"])
    pipeline = transformer._pipeline_for_direction(direction.upper())
    computed = pipeline.computed
    first = cp.full(32, 120.0 if direction == "forward" else 1e300, dtype=cp.float64)
    second = cp.full(32, 45.0 if direction == "forward" else 1e300, dtype=cp.float64)
    common = (
        {"e": math.nextafter(0.1, math.inf)},
        {"e": -math.ulp(0.0)},
        {"e": math.nan},
        {"e": math.inf},
        {"a": math.nextafter(PROJECTION_FIXED_Q62_MAX_SCALE_M, math.inf)},
        {"a": 0.0},
        {"a": -1.0},
        {"a": math.nan},
        {"a": math.inf},
        {"lam0": math.inf},
        {"lam0": math.nan},
        {"x0": math.inf},
        {"x0": math.nan},
        {"y0": math.inf},
        {"y0": math.nan},
        {"x_unit_to_m": math.inf},
        {"x_unit_to_m": math.nan},
        {"x_unit_to_m": 0.0},
        {"y_unit_to_m": math.inf},
        {"y_unit_to_m": math.nan},
        {"y_unit_to_m": 0.0},
    )
    directional = (
        (
            {"k0": math.nextafter(1.0, math.inf)},
            {"k0": 0.0},
            {"k0": -1.0},
            {"k0": math.nan},
            {"k0": math.inf},
        )
        if direction == "forward"
        else (
            {"k0": math.ulp(0.0)},
            {"k0": 0.0},
            {"k0": -1.0},
            {"k0": math.nan},
            {"k0": math.inf},
            {"conformal_to_geodetic": (math.nan,) * 6},
        )
    )
    implementation_id = (
        MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY if direction == "forward" else MERC_INVERSE_EXP_SERIES
    )
    for mutation in (*common, *directional):
        mutated = {**computed, **mutation}
        native = _run_exact(
            transformer, first, second, direction, NATIVE_LIBDEVICE, computed=mutated
        )
        accelerated = _run_exact(
            transformer, first, second, direction, implementation_id, computed=mutated
        )
        _assert_bitwise_equal(accelerated, native)


def test_merc_forward_finite_raw_derived_longitude_overflow_is_bitwise_native():
    transformer, _ = _transformer(TARGETS["ellipsoidal_a"])
    computed = {
        **transformer._pipeline.computed,
        "lam0": np.finfo(np.float64).max,
    }
    longitude = cp.full(32, -np.finfo(np.float64).max, dtype=cp.float64)
    latitude = cp.full(32, 45.0, dtype=cp.float64)
    native = _run_exact(
        transformer, longitude, latitude, "forward", NATIVE_LIBDEVICE, computed=computed
    )
    accelerated = _run_exact(
        transformer,
        longitude,
        latitude,
        "forward",
        MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY,
        computed=computed,
    )
    _assert_bitwise_equal(accelerated, native)


@pytest.mark.parametrize(
    "latitude",
    [
        89.96231640026328,
        -89.96231640026328,
        89.99008855026355,
        -89.99008855026355,
        89.999,
        -89.999,
    ],
)
def test_merc_forward_polar_accuracy_cap_is_bitwise_native(latitude):
    transformer, _ = _transformer(TARGETS["ellipsoidal_a"])
    longitude = cp.full(32, 66.84738314058617, dtype=cp.float64)
    latitude_values = cp.full(32, latitude, dtype=cp.float64)
    native = _run_exact(transformer, longitude, latitude_values, "forward", NATIVE_LIBDEVICE)
    accelerated = _run_exact(
        transformer,
        longitude,
        latitude_values,
        "forward",
        MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY,
    )
    _assert_bitwise_equal(accelerated, native)


def test_merc_exact_setup_boundaries_remain_accurate():
    transformer, _ = _transformer(TARGETS["ellipsoidal_a"])
    base = transformer._pipeline.computed
    eccentricity = 0.1
    flattening = 1.0 - math.sqrt(1.0 - eccentricity * eccentricity)
    third_flattening = flattening / (2.0 - flattening)
    computed = {
        **base,
        "e": eccentricity,
        "a": PROJECTION_FIXED_Q62_MAX_SCALE_M,
        "k0": 1.0,
        "x0": 123_456.75,
        "y0": -654_321.25,
        "x_unit_to_m": -2.0,
        "y_unit_to_m": -0.5,
        "conformal_to_geodetic": conformal_to_geodetic_coefficients(third_flattening),
    }
    longitude = cp.linspace(-179.0, 179.0, 65_536, dtype=cp.float64)
    latitude = cp.linspace(-89.999, 89.999, 65_536, dtype=cp.float64)
    native_forward = _run_exact(
        transformer, longitude, latitude, "forward", NATIVE_LIBDEVICE, computed=computed
    )
    accelerated_forward = _run_exact(
        transformer,
        longitude,
        latitude,
        "forward",
        MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY,
        computed=computed,
    )
    forward_error_m = np.hypot(
        cp.asnumpy(accelerated_forward[0] - native_forward[0]) * abs(computed["x_unit_to_m"]),
        cp.asnumpy(accelerated_forward[1] - native_forward[1]) * abs(computed["y_unit_to_m"]),
    )
    assert float(np.max(forward_error_m)) <= 1e-8
    native_inverse = _run_exact(
        transformer, *native_forward, "inverse", NATIVE_LIBDEVICE, computed=computed
    )
    accelerated_inverse = _run_exact(
        transformer,
        *native_forward,
        "inverse",
        MERC_INVERSE_EXP_SERIES,
        computed=computed,
    )
    assert (
        float(
            np.max(
                _geographic_error_m(
                    accelerated_inverse,
                    native_inverse,
                    PROJECTION_FIXED_Q62_MAX_SCALE_M,
                )
            )
        )
        <= 1e-8
    )


def test_public_merc_compile_outputs_stream_thresholds_and_cache_concurrency():
    transformer, _ = _transformer(TARGETS["ellipsoidal_a"])
    transformer.compile(precision="fp64", transcendentals="accelerated")
    assert (
        "merc",
        "forward",
        "float64",
        MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY,
    ) in fused_module._kernel_cache
    assert (
        "merc",
        "inverse",
        "float64",
        MERC_INVERSE_EXP_SERIES,
    ) in fused_module._kernel_cache

    for workload_size in (1, 5_000_000):
        explanation = transformer.explain_strategy(
            direction="FORWARD",
            transcendentals="auto",
            precision="fp64",
            workload_size=workload_size,
        )
        assert explanation.decisions[0].implementation_id == NATIVE_LIBDEVICE
    explicit = transformer.explain_strategy(
        direction="FORWARD",
        transcendentals="accelerated",
        precision="fp64",
        workload_size=5_000_000,
    )
    assert explicit.decisions[0].implementation_id == MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY

    sphere, _ = _transformer(TARGETS["spherical_a"])
    sphere.compile(precision="fp64", transcendentals="accelerated")
    assert (
        "merc",
        "forward",
        "float64",
        MERC_FORWARD_SPHERICAL_PRODUCT_POLY,
    ) in fused_module._kernel_cache
    for workload_size, expected in (
        (MERC_FORWARD_SPHERICAL_PRODUCT_POLY_MIN_ELEMENTS - 1, NATIVE_LIBDEVICE),
        (
            MERC_FORWARD_SPHERICAL_PRODUCT_POLY_MIN_ELEMENTS,
            MERC_FORWARD_SPHERICAL_PRODUCT_POLY,
        ),
    ):
        explanation = sphere.explain_strategy(
            direction="FORWARD",
            transcendentals="auto",
            precision="fp64",
            workload_size=workload_size,
        )
        assert explanation.decisions[0].implementation_id == expected

    for direction, threshold, implementation_id in (
        ("INVERSE", MERC_INVERSE_EXP_SERIES_MIN_ELEMENTS, MERC_INVERSE_EXP_SERIES),
    ):
        below = transformer.explain_strategy(
            direction=direction,
            transcendentals="auto",
            precision="fp64",
            workload_size=threshold - 1,
        )
        at = transformer.explain_strategy(
            direction=direction,
            transcendentals="auto",
            precision="fp64",
            workload_size=threshold,
        )
        assert below.decisions[0].implementation_id == NATIVE_LIBDEVICE
        assert at.decisions[0].implementation_id == implementation_id

    longitude = cp.linspace(-170.0, 170.0, 4096, dtype=cp.float64)
    latitude = cp.linspace(-80.0, 80.0, 4096, dtype=cp.float64)
    projected = transformer.transform_buffers(
        longitude,
        latitude,
        precision="fp64",
        transcendentals="native",
    )
    public_inputs = {"FORWARD": (longitude, latitude), "INVERSE": projected}
    public_results = {}
    for direction, inputs in public_inputs.items():
        out_x = cp.empty_like(longitude)
        out_y = cp.empty_like(latitude)
        stream = cp.cuda.Stream(non_blocking=True)
        result = transformer.transform_buffers(
            *inputs,
            direction=direction,
            out_x=out_x,
            out_y=out_y,
            precision="fp64",
            transcendentals="accelerated",
            stream=stream,
        )
        stream.synchronize()
        assert result[0] is out_x
        assert result[1] is out_y
        public_results[direction] = tuple(cp.asnumpy(value) for value in result)

    fused_module._kernel_cache.clear()
    with ThreadPoolExecutor(max_workers=8) as executor:
        kernels = list(
            executor.map(
                lambda _: fused_module._get_kernel(
                    "merc",
                    "inverse",
                    "float64",
                    transcendental_impl=MERC_INVERSE_EXP_SERIES,
                ),
                range(32),
            )
        )
    assert all(kernel is kernels[0] for kernel in kernels)

    def execute_public(direction):
        inputs = public_inputs[direction]
        out_x = cp.empty_like(inputs[0])
        out_y = cp.empty_like(inputs[1])
        stream = cp.cuda.Stream(non_blocking=True)
        result = transformer.transform_buffers(
            *inputs,
            direction=direction,
            out_x=out_x,
            out_y=out_y,
            precision="fp64",
            transcendentals="accelerated",
            stream=stream,
        )
        stream.synchronize()
        return direction, tuple(cp.asnumpy(value) for value in result)

    directions = ["FORWARD", "INVERSE"] * 4
    with ThreadPoolExecutor(max_workers=8) as executor:
        concurrent_results = list(executor.map(execute_public, directions))
    for direction, result in concurrent_results:
        for actual, expected in zip(result, public_results[direction], strict=True):
            assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
