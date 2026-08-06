"""GPU correctness, fallback, and residency gates for Wave 3C LCC."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import re

import numpy as np
import pytest

from vibeproj import Transformer
import vibeproj.fused_kernels as fused_module
from vibeproj.fused_kernels import fused_transform
from vibeproj.transcendentals import (
    LCC_FORWARD_CONFORMAL_REFRAME,
    LCC_INVERSE_CONFORMAL_REFRAME,
    NATIVE_LIBDEVICE,
)


cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device not available", allow_module_level=True)
except Exception as exc:  # pragma: no cover - runtime-specific
    pytest.skip(f"CUDA device not available: {exc}", allow_module_level=True)


TARGETS = (
    "+proj=lcc +lat_0=40 +lat_1=40 +lon_0=-96 +R=6371000 +units=m +type=crs",
    "+proj=lcc +lat_0=23 +lat_1=33 +lat_2=45 +lon_0=-96 +R=6371000 +units=m +type=crs",
    "+proj=lcc +lat_0=40 +lat_1=40 +lon_0=-96 +ellps=WGS84 +units=m +type=crs",
    "EPSG:2154",
    "EPSG:3851",
    "+proj=lcc +lat_0=10 +lat_1=0 +lat_2=30 +lon_0=-96 +ellps=WGS84 +units=m +type=crs",
    "+proj=lcc +lat_0=23 +lat_1=33 +lat_2=45 +lon_0=-96 +a=6400000 +es=.01 +units=m +type=crs",
)


def _transformer(target_definition: str) -> Transformer:
    from pyproj import CRS

    target = CRS.from_user_input(target_definition)
    return Transformer.from_crs(target.geodetic_crs, target, always_xy=True)


def _bitwise_equal(left, right) -> bool:
    return all(
        bool(cp.all(actual.view(cp.uint64) == expected.view(cp.uint64)).get())
        for actual, expected in zip(left, right, strict=True)
    )


def _projected_error_m(actual, expected) -> float:
    return float(cp.max(cp.hypot(actual[0] - expected[0], actual[1] - expected[1])).get())


def _angular_error_m(actual, expected, radius: float) -> float:
    dlon = cp.mod(actual[0] - expected[0] + 180.0, 360.0) - 180.0
    dx = cp.deg2rad(dlon) * radius * cp.cos(cp.deg2rad(expected[1]))
    dy = cp.deg2rad(actual[1] - expected[1]) * radius
    return float(cp.max(cp.hypot(dx, dy)).get())


def _run_exact(transformer, first, second, direction: str, implementation: str, **kwargs):
    pipeline = transformer._pipeline
    return fused_transform(
        first,
        second,
        projection_name="lcc",
        direction=direction,
        computed=pipeline.computed,
        src_north_first=pipeline.src_north_first,
        dst_north_first=pipeline.dst_north_first,
        xp=cp,
        precision="fp64",
        transcendental_impl=implementation,
        **kwargs,
    )


@pytest.mark.parametrize("target_definition", TARGETS)
def test_lcc_public_accelerated_forward_inverse_stays_below_10nm(target_definition):
    transformer = _transformer(target_definition)
    computed = transformer._pipeline.computed
    longitude_origin = math.degrees(float(computed["lam0"]))
    if computed["n"] < 0.0:
        latitude = cp.linspace(-75.0, -5.0, 16_384, dtype=cp.float64)
    else:
        latitude = cp.linspace(5.0, 75.0, 16_384, dtype=cp.float64)
    longitude = cp.linspace(
        longitude_origin - 45.0,
        longitude_origin + 45.0,
        latitude.size,
        dtype=cp.float64,
    )
    native_forward = transformer.transform_buffers(
        longitude, latitude, precision="fp64", transcendentals="native"
    )
    accelerated_forward = transformer.transform_buffers(
        longitude, latitude, precision="fp64", transcendentals="accelerated"
    )
    assert _projected_error_m(accelerated_forward, native_forward) <= 1e-8
    if math.isclose(float(computed["e"]), 0.1, rel_tol=0.0, abs_tol=1e-15):
        assert not _bitwise_equal(accelerated_forward, native_forward)

    native_inverse = transformer.transform_buffers(
        *native_forward,
        direction="INVERSE",
        precision="fp64",
        transcendentals="native",
    )
    accelerated_inverse = transformer.transform_buffers(
        *native_forward,
        direction="INVERSE",
        precision="fp64",
        transcendentals="accelerated",
    )
    assert _angular_error_m(accelerated_inverse, native_inverse, float(computed["a"])) <= 1e-8
    if math.isclose(float(computed["e"]), 0.1, rel_tol=0.0, abs_tol=1e-15):
        assert not _bitwise_equal(accelerated_inverse, native_inverse)


@pytest.mark.parametrize("fraction", [0.0, 0.001, 0.01, 0.1, 0.5, 1.0])
def test_lcc_forward_opposite_cone_extreme_mixtures_remain_hot_and_accurate(fraction):
    transformer = _transformer("EPSG:2154")
    n = 65_536
    rng = np.random.default_rng(20260806)
    longitude = rng.uniform(-40.0, 20.0, n)
    latitude = rng.uniform(5.0, 75.0, n)
    count = round(fraction * n)
    if count:
        latitude[rng.choice(n, count, replace=False)] = -89.999
    lon = cp.asarray(longitude)
    lat = cp.asarray(latitude)
    native = transformer.transform_buffers(lon, lat, transcendentals="native")
    accelerated = transformer.transform_buffers(lon, lat, transcendentals="accelerated")
    assert _projected_error_m(accelerated, native) <= 1e-8
    decision = transformer.explain_strategy(
        transcendentals="accelerated", workload_size=n
    ).decisions[-1]
    assert decision.implementation_id == LCC_FORWARD_CONFORMAL_REFRAME


@pytest.mark.parametrize(
    ("longitude", "latitude"),
    [
        (3.0, 90.0),
        (3.0, -90.0),
        (math.nan, 45.0),
        (math.inf, 45.0),
        (-math.inf, 45.0),
        (3.0, math.nan),
        (3.0, math.inf),
        (3.0, -math.inf),
    ],
)
def test_lcc_forward_homogeneous_cold_warps_are_bitwise_native(longitude, latitude):
    transformer = _transformer("EPSG:2154")
    lon = cp.full(64, longitude, dtype=cp.float64)
    lat = cp.full(64, latitude, dtype=cp.float64)
    native = _run_exact(transformer, lon, lat, "forward", NATIVE_LIBDEVICE)
    accelerated = _run_exact(transformer, lon, lat, "forward", LCC_FORWARD_CONFORMAL_REFRAME)
    assert _bitwise_equal(accelerated, native)


@pytest.mark.parametrize(
    ("edge", "component"),
    [
        (90.0, "latitude"),
        (-90.0, "latitude"),
        (math.nan, "longitude"),
        (math.inf, "longitude"),
        (math.nan, "latitude"),
        (math.inf, "latitude"),
    ],
)
def test_lcc_forward_one_cold_lane_makes_mixed_warp_bitwise_native(edge, component):
    transformer = _transformer("EPSG:2154")
    lon = cp.linspace(-15.0, 15.0, 32, dtype=cp.float64)
    lat = cp.linspace(30.0, 60.0, 32, dtype=cp.float64)
    (lon if component == "longitude" else lat)[0] = edge
    native = _run_exact(transformer, lon, lat, "forward", NATIVE_LIBDEVICE)
    accelerated = _run_exact(transformer, lon, lat, "forward", LCC_FORWARD_CONFORMAL_REFRAME)
    assert _bitwise_equal(accelerated, native)


@pytest.mark.parametrize("target_definition", ["EPSG:2154", "EPSG:3851"])
def test_lcc_forward_north_and_south_hot_witnesses_are_nonvacuous(target_definition):
    transformer = _transformer(target_definition)
    if target_definition == "EPSG:3851":
        # This fixed southern-cone witness differs from native by one binary64
        # ULP and is also used by the central qualification guard audit.
        lon = cp.full(4096, 169.3024197984506, dtype=cp.float64)
        lat = cp.full(4096, -9.308100979853066, dtype=cp.float64)
    else:
        longitude_origin = math.degrees(float(transformer._pipeline.computed["lam0"]))
        lon = cp.linspace(longitude_origin - 45.0, longitude_origin + 45.0, 4096)
        lat = cp.linspace(5.0, 75.0, 4096)
    native = _run_exact(transformer, lon, lat, "forward", NATIVE_LIBDEVICE)
    accelerated = _run_exact(transformer, lon, lat, "forward", LCC_FORWARD_CONFORMAL_REFRAME)
    assert not _bitwise_equal(accelerated, native)
    assert _projected_error_m(accelerated, native) <= 1e-8


def _inverse_apex(transformer):
    computed = transformer._pipeline.computed
    return (
        float(computed["x0"]),
        float(computed["y0"] + computed["a"] * computed["rho0"]),
    )


@pytest.mark.parametrize(
    "kind",
    ["apex", "nan_x", "positive_inf_x", "negative_inf_x", "nan_y", "positive_inf_y"],
)
def test_lcc_inverse_homogeneous_cold_warps_are_bitwise_native(kind):
    transformer = _transformer("EPSG:2154")
    apex_x, apex_y = _inverse_apex(transformer)
    values = {
        "apex": (apex_x, apex_y),
        "nan_x": (math.nan, apex_y - 1_000_000.0),
        "positive_inf_x": (math.inf, apex_y - 1_000_000.0),
        "negative_inf_x": (-math.inf, apex_y - 1_000_000.0),
        "nan_y": (apex_x + 100_000.0, math.nan),
        "positive_inf_y": (apex_x + 100_000.0, math.inf),
    }
    first, second = values[kind]
    x = cp.full(64, first, dtype=cp.float64)
    y = cp.full(64, second, dtype=cp.float64)
    native = _run_exact(transformer, x, y, "inverse", NATIVE_LIBDEVICE)
    accelerated = _run_exact(transformer, x, y, "inverse", LCC_INVERSE_CONFORMAL_REFRAME)
    assert _bitwise_equal(accelerated, native)


@pytest.mark.parametrize("kind", ["apex", "nan_x", "positive_inf_x", "nan_y"])
def test_lcc_inverse_one_cold_lane_makes_mixed_warp_bitwise_native(kind):
    transformer = _transformer("EPSG:2154")
    apex_x, apex_y = _inverse_apex(transformer)
    x = cp.linspace(apex_x - 100_000.0, apex_x + 100_000.0, 32, dtype=cp.float64)
    y = cp.linspace(apex_y - 2_000_000.0, apex_y - 1_000_000.0, 32, dtype=cp.float64)
    values = {
        "apex": (apex_x, apex_y),
        "nan_x": (math.nan, apex_y - 1_000_000.0),
        "positive_inf_x": (math.inf, apex_y - 1_000_000.0),
        "nan_y": (apex_x + 100_000.0, math.nan),
    }
    x[0], y[0] = values[kind]
    native = _run_exact(transformer, x, y, "inverse", NATIVE_LIBDEVICE)
    accelerated = _run_exact(transformer, x, y, "inverse", LCC_INVERSE_CONFORMAL_REFRAME)
    assert _bitwise_equal(accelerated, native)


def test_lcc_inverse_hot_warp_is_nonvacuous_and_below_10nm():
    transformer = _transformer("EPSG:2154")
    lon = cp.linspace(-15.0, 15.0, 4096, dtype=cp.float64)
    lat = cp.linspace(30.0, 60.0, 4096, dtype=cp.float64)
    projected = transformer.transform_buffers(lon, lat, transcendentals="native")
    native = _run_exact(transformer, *projected, "inverse", NATIVE_LIBDEVICE)
    accelerated = _run_exact(transformer, *projected, "inverse", LCC_INVERSE_CONFORMAL_REFRAME)
    assert not _bitwise_equal(accelerated, native)
    assert _angular_error_m(accelerated, native, float(transformer._pipeline.computed["a"])) <= 1e-8


@pytest.mark.parametrize("direction", ["forward", "inverse"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n", 0.0),
        ("n", math.nan),
        ("F", 0.0),
        ("F", math.nan),
        ("rho0", math.inf),
        ("rho0", math.nan),
        ("e", math.nextafter(0.0, -math.inf)),
        ("e", math.nextafter(0.1, math.inf)),
        ("e", math.nan),
        ("k0", math.nextafter(1.0, 0.0)),
        ("k0", math.nextafter(1.0, math.inf)),
        ("lam0", math.inf),
        ("a", 0.0),
        ("a", math.nextafter(6_400_000.0, math.inf)),
        ("a", math.nan),
        ("x0", math.inf),
        ("y0", math.nan),
        ("x_unit_to_m", 0.0),
        ("x_unit_to_m", math.inf),
        ("y_unit_to_m", 0.0),
        ("y_unit_to_m", math.nan),
    ],
)
def test_lcc_device_setup_guard_is_exhaustively_bitwise_native(direction, field, value):
    transformer = _transformer("EPSG:2154")
    computed = transformer._pipeline.computed
    original = computed[field]
    lon = cp.linspace(-10.0, 10.0, 64, dtype=cp.float64)
    lat = cp.linspace(35.0, 55.0, 64, dtype=cp.float64)
    if direction == "forward":
        first, second = lon, lat
        implementation = LCC_FORWARD_CONFORMAL_REFRAME
    else:
        first, second = transformer.transform_buffers(lon, lat, transcendentals="native")
        implementation = LCC_INVERSE_CONFORMAL_REFRAME
    computed[field] = value
    try:
        native = _run_exact(transformer, first, second, direction, NATIVE_LIBDEVICE)
        accelerated = _run_exact(transformer, first, second, direction, implementation)
        assert _bitwise_equal(accelerated, native)
    finally:
        computed[field] = original


def test_lcc_inverse_negative_radius_ratio_is_bitwise_native():
    transformer = _transformer("EPSG:2154")
    computed = transformer._pipeline.computed
    original = computed["F"]
    lon = cp.linspace(-10.0, 10.0, 64, dtype=cp.float64)
    lat = cp.linspace(35.0, 55.0, 64, dtype=cp.float64)
    projected = transformer.transform_buffers(lon, lat, transcendentals="native")
    computed["F"] = -abs(float(original))
    try:
        native = _run_exact(transformer, *projected, "inverse", NATIVE_LIBDEVICE)
        accelerated = _run_exact(transformer, *projected, "inverse", LCC_INVERSE_CONFORMAL_REFRAME)
        assert _bitwise_equal(accelerated, native)
    finally:
        computed["F"] = original


def test_lcc_forward_unbounded_cone_angle_is_bitwise_native():
    transformer = _transformer("EPSG:2154")
    computed = transformer._pipeline.computed
    original = computed["n"]
    longitude_origin = math.degrees(float(computed["lam0"]))
    lon = cp.full(64, longitude_origin + 170.0, dtype=cp.float64)
    lat = cp.linspace(35.0, 55.0, 64, dtype=cp.float64)
    computed["n"] = 2.0
    try:
        native = _run_exact(transformer, lon, lat, "forward", NATIVE_LIBDEVICE)
        accelerated = _run_exact(transformer, lon, lat, "forward", LCC_FORWARD_CONFORMAL_REFRAME)
        assert _bitwise_equal(accelerated, native)
    finally:
        computed["n"] = original


@pytest.mark.parametrize("direction", ["forward", "inverse"])
@pytest.mark.parametrize("field", ["x_unit_to_m", "y_unit_to_m"])
def test_lcc_negative_finite_unit_factors_remain_hot(direction, field):
    transformer = _transformer("EPSG:2154")
    computed = transformer._pipeline.computed
    original = computed[field]
    lon = cp.linspace(-10.0, 10.0, 256, dtype=cp.float64)
    lat = cp.linspace(35.0, 55.0, 256, dtype=cp.float64)
    if direction == "forward":
        first, second = lon, lat
        implementation = LCC_FORWARD_CONFORMAL_REFRAME
    else:
        first, second = transformer.transform_buffers(lon, lat, transcendentals="native")
        implementation = LCC_INVERSE_CONFORMAL_REFRAME
    computed[field] = -1.0
    try:
        native = _run_exact(transformer, first, second, direction, NATIVE_LIBDEVICE)
        accelerated = _run_exact(transformer, first, second, direction, implementation)
        assert not _bitwise_equal(accelerated, native)
        error = (
            _projected_error_m(accelerated, native)
            if direction == "forward"
            else _angular_error_m(accelerated, native, float(computed["a"]))
        )
        assert error <= 1e-8
    finally:
        computed[field] = original


@pytest.mark.parametrize(
    ("n", "expects_native"),
    [
        (math.nextafter(0.2, 0.0), True),
        (0.2, False),
        (math.nextafter(0.2, math.inf), False),
        (math.nextafter(-0.2, 0.0), True),
        (-0.2, False),
        (math.nextafter(-0.2, -math.inf), False),
    ],
)
def test_lcc_forward_device_regular_cone_boundary_is_exact(n, expects_native):
    # The spherical hot path uses log/exp in place of native pow, which makes
    # this a non-vacuous branch witness on both signs of the exact boundary.
    transformer = _transformer(TARGETS[0])
    computed = transformer._pipeline.computed
    original = computed["n"]
    lon = cp.linspace(-10.0, 10.0, 256, dtype=cp.float64)
    lat = cp.linspace(35.0, 55.0, 256, dtype=cp.float64)
    computed["n"] = n
    try:
        native = _run_exact(transformer, lon, lat, "forward", NATIVE_LIBDEVICE)
        accelerated = _run_exact(transformer, lon, lat, "forward", LCC_FORWARD_CONFORMAL_REFRAME)
        assert _bitwise_equal(accelerated, native) is expects_native
        if not expects_native:
            assert _projected_error_m(accelerated, native) <= 1e-8
    finally:
        computed["n"] = original


@pytest.mark.parametrize("target_definition", ["EPSG:2154", "EPSG:3851"])
def test_lcc_inverse_huge_finite_inputs_take_finite_pole_limit(target_definition):
    transformer = _transformer(target_definition)
    values = cp.asarray([1e200, -1e200, 1e300, -1e300], dtype=cp.float64)
    native = transformer.transform_buffers(
        values, values[::-1], direction="INVERSE", transcendentals="native"
    )
    accelerated = transformer.transform_buffers(
        values, values[::-1], direction="INVERSE", transcendentals="accelerated"
    )
    assert bool(cp.all(cp.isfinite(accelerated[0])).get())
    assert bool(cp.all(cp.isfinite(accelerated[1])).get())
    assert _angular_error_m(accelerated, native, float(transformer._pipeline.computed["a"])) <= 1e-8


def test_lcc_public_us_survey_foot_offsets_stream_output_identity_and_concurrency():
    transformer = _transformer("EPSG:2263")
    lon = cp.linspace(-74.2, -72.0, 4096, dtype=cp.float64)
    lat = cp.linspace(40.5, 41.2, 4096, dtype=cp.float64)
    native = transformer.transform_buffers(lon, lat, transcendentals="native")
    out_x = cp.empty_like(lon)
    out_y = cp.empty_like(lat)
    stream = cp.cuda.Stream(non_blocking=True)
    accelerated = transformer.transform_buffers(
        lon,
        lat,
        out_x=out_x,
        out_y=out_y,
        transcendentals="accelerated",
        stream=stream,
    )
    stream.synchronize()
    assert accelerated[0] is out_x
    assert accelerated[1] is out_y
    unit = float(transformer._pipeline.computed["x_unit_to_m"])
    assert _projected_error_m(accelerated, native) * unit <= 1e-8

    fused_module._kernel_cache.clear()
    with ThreadPoolExecutor(max_workers=8) as executor:
        kernels = list(
            executor.map(
                lambda _: fused_module._get_kernel(
                    "lcc",
                    "forward",
                    "float64",
                    transcendental_impl=LCC_FORWARD_CONFORMAL_REFRAME,
                ),
                range(32),
            )
        )
    assert all(kernel is kernels[0] for kernel in kernels)


def test_lcc_public_forward_inverse_run_on_independent_streams_with_preallocated_outputs():
    transformer = _transformer("EPSG:2154")
    lon = cp.linspace(-15.0, 15.0, 65_536, dtype=cp.float64)
    lat = cp.linspace(30.0, 60.0, 65_536, dtype=cp.float64)
    projected = transformer.transform_buffers(lon, lat, transcendentals="native")
    expected_forward = projected
    expected_inverse = transformer.transform_buffers(
        *projected, direction="INVERSE", transcendentals="native"
    )
    cp.cuda.get_current_stream().synchronize()
    forward_out = (cp.empty_like(lon), cp.empty_like(lat))
    inverse_out = (cp.empty_like(lon), cp.empty_like(lat))
    forward_stream = cp.cuda.Stream(non_blocking=True)
    inverse_stream = cp.cuda.Stream(non_blocking=True)

    with forward_stream:
        actual_forward = transformer.transform_buffers(
            lon,
            lat,
            out_x=forward_out[0],
            out_y=forward_out[1],
            transcendentals="accelerated",
            stream=forward_stream,
        )
    with inverse_stream:
        actual_inverse = transformer.transform_buffers(
            *projected,
            direction="INVERSE",
            out_x=inverse_out[0],
            out_y=inverse_out[1],
            transcendentals="accelerated",
            stream=inverse_stream,
        )
    forward_stream.synchronize()
    inverse_stream.synchronize()

    assert actual_forward[0] is forward_out[0] and actual_forward[1] is forward_out[1]
    assert actual_inverse[0] is inverse_out[0] and actual_inverse[1] is inverse_out[1]
    assert _projected_error_m(actual_forward, expected_forward) <= 1e-8
    assert (
        _angular_error_m(
            actual_inverse, expected_inverse, float(transformer._pipeline.computed["a"])
        )
        <= 1e-8
    )


@pytest.mark.parametrize(
    ("direction", "implementation"),
    [
        ("forward", LCC_FORWARD_CONFORMAL_REFRAME),
        ("inverse", LCC_INVERSE_CONFORMAL_REFRAME),
    ],
)
def test_lcc_accelerated_source_preserves_native_abi(direction, implementation):
    template, native_name = fused_module._SOURCE_MAP[("lcc", direction)]
    native_source = fused_module._inject_linear_unit_args(
        template.format(
            real_t="double",
            pi=fused_module._PI_LITERALS["float64"],
            tol=fused_module._TOL_LITERALS["float64"],
        )
    )
    accelerated_source, accelerated_name = fused_module._build_projection_guarded_source(
        "lcc", direction, implementation
    )

    def parameters(source, name):
        match = re.search(
            rf'extern "C" __global__ void __launch_bounds__\(256\) '
            rf"{re.escape(name)}\((.*?)\n\) \{{",
            source,
            flags=re.DOTALL,
        )
        assert match is not None
        return match.group(1)

    native_parameters = parameters(native_source, native_name)
    assert parameters(accelerated_source, accelerated_name) == native_parameters
    assert native_name != accelerated_name
    assert "vp_lcc_setup_is_qualified" not in native_source
    assert "conformal_reframe" not in native_source
    assert "vp_lcc_setup_is_qualified" in accelerated_source
    assert "conformal_reframe" in accelerated_name
    assert "const double contraction" not in native_source
    if direction == "inverse":
        assert "if (i == 5 && fabs(dphi) >= 1e-14)" in accelerated_source
        assert "step /= 1.0 - contraction" in accelerated_source
    else:
        assert "const double contraction" not in accelerated_source
    assert native_parameters.count("double") == 15
    assert "double nn, double F, double rho0, double e, double k0" in native_parameters
    assert native_parameters.rstrip().endswith("int src_north_first, int dst_north_first, int n")


@pytest.mark.parametrize(
    ("projection", "direction", "dtype", "implementation"),
    [
        ("lcc", "inverse", "float64", LCC_FORWARD_CONFORMAL_REFRAME),
        ("lcc", "forward", "float64", LCC_INVERSE_CONFORMAL_REFRAME),
        ("lcc", "forward", "float32", LCC_FORWARD_CONFORMAL_REFRAME),
        ("lcc", "inverse", "float32", LCC_INVERSE_CONFORMAL_REFRAME),
    ],
)
def test_lcc_accelerated_implementation_rejects_wrong_direction_or_precision(
    projection, direction, dtype, implementation
):
    with pytest.raises(ValueError, match="qualified only"):
        fused_module._validate_projection_implementation(
            projection, direction, dtype, implementation
        )


@pytest.mark.parametrize(
    ("direction", "implementation"),
    [
        ("forward", LCC_FORWARD_CONFORMAL_REFRAME),
        ("inverse", LCC_INVERSE_CONFORMAL_REFRAME),
    ],
)
def test_lcc_accelerated_kernel_cache_resources_and_single_graph_node(direction, implementation):
    fused_module._kernel_cache.clear()
    native_kernel = fused_module._get_kernel(
        "lcc", direction, "float64", transcendental_impl=NATIVE_LIBDEVICE
    )
    kernel = fused_module._get_kernel(
        "lcc", direction, "float64", transcendental_impl=implementation
    )
    assert native_kernel is not kernel
    assert native_kernel.name != kernel.name
    kernel.compile()
    assert kernel.attributes["num_regs"] <= 40
    assert kernel.attributes["shared_size_bytes"] == 0
    assert kernel.attributes["local_size_bytes"] <= 64

    transformer = _transformer("EPSG:2154")
    values = cp.linspace(-1.0, 1.0, 4096, dtype=cp.float64)
    if direction == "forward":
        first, second = values, cp.linspace(40.0, 50.0, values.size)
    else:
        projected = transformer.transform_buffers(
            values, cp.linspace(40.0, 50.0, values.size), transcendentals="native"
        )
        first, second = projected
    out_x = cp.empty_like(first)
    out_y = cp.empty_like(second)
    transformer.transform_buffers(
        first,
        second,
        direction=direction.upper(),
        out_x=out_x,
        out_y=out_y,
        transcendentals="accelerated",
    )
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    stream.begin_capture()
    with stream:
        result = transformer.transform_buffers(
            first,
            second,
            direction=direction.upper(),
            out_x=out_x,
            out_y=out_y,
            transcendentals="accelerated",
            stream=stream,
        )
    graph = stream.end_capture()
    dot = graph.debug_dot_str()
    assert result[0] is out_x and result[1] is out_y
    assert dot.count(implementation.replace(".", "_")) == 1
    assert "memcpy" not in dot.lower()
    assert "memset" not in dot.lower()
