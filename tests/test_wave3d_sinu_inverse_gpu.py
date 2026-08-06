"""GPU production gates for Wave 3D ellipsoidal Sinusoidal inverse."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import re

import pytest

from vibeproj import Transformer
import vibeproj.fused_kernels as fused_module
from vibeproj.fused_kernels import fused_transform
from vibeproj.transcendentals import (
    NATIVE_LIBDEVICE,
    SINU_INVERSE_CONVERGENT_NEWTON,
    SINU_INVERSE_MERIDIONAL_RECURRENCE,
)


cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device not available", allow_module_level=True)
except Exception as exc:  # pragma: no cover - runtime-specific
    pytest.skip(f"CUDA device not available: {exc}", allow_module_level=True)


TARGETS = (
    "ESRI:54008",
    "+proj=sinu +lon_0=0 +a=6400000 +es=.012 +units=m +type=crs",
)


def _transformer(target_definition: str = "ESRI:54008") -> Transformer:
    from pyproj import CRS

    target = CRS.from_user_input(target_definition)
    return Transformer.from_crs(target.geodetic_crs, target, always_xy=True)


def _run_exact(transformer, first, second, implementation, **kwargs):
    pipeline = transformer._pipeline
    return fused_transform(
        first,
        second,
        projection_name="sinu",
        direction="inverse",
        computed=pipeline.computed,
        src_north_first=pipeline.src_north_first,
        dst_north_first=pipeline.dst_north_first,
        xp=cp,
        precision="fp64",
        transcendental_impl=implementation,
        **kwargs,
    )


def _bitwise_equal(left, right) -> bool:
    return all(
        bool(cp.all(actual.view(cp.uint64) == expected.view(cp.uint64)).get())
        for actual, expected in zip(left, right, strict=True)
    )


def _angular_error_m(actual, expected, radius: float) -> float:
    finite = cp.isfinite(actual[0]) & cp.isfinite(actual[1])
    finite &= cp.isfinite(expected[0]) & cp.isfinite(expected[1])
    if not bool(cp.any(finite).get()):
        return 0.0
    dlon = cp.mod(actual[0][finite] - expected[0][finite] + 180.0, 360.0) - 180.0
    dx = cp.deg2rad(dlon) * radius * cp.cos(cp.deg2rad(expected[1][finite]))
    dy = cp.deg2rad(actual[1][finite] - expected[1][finite]) * radius
    return float(cp.max(cp.hypot(dx, dy)).get())


def _projected_hot(transformer, n=16_384):
    longitude = cp.linspace(-179.0, 179.0, n, dtype=cp.float64)
    latitude = cp.linspace(-89.8, 89.8, n, dtype=cp.float64)
    projected = transformer.transform_buffers(
        longitude, latitude, precision="fp64", transcendentals="native"
    )
    return longitude, latitude, projected


@pytest.mark.parametrize("target_definition", TARGETS)
@pytest.mark.parametrize(
    "implementation",
    [SINU_INVERSE_CONVERGENT_NEWTON, SINU_INVERSE_MERIDIONAL_RECURRENCE],
)
def test_sinu_inverse_qualified_boundaries_stay_below_10nm(target_definition, implementation):
    transformer = _transformer(target_definition)
    _, _, projected = _projected_hot(transformer)
    native = _run_exact(transformer, *projected, NATIVE_LIBDEVICE)
    candidate = _run_exact(transformer, *projected, implementation)
    assert not _bitwise_equal(candidate, native)
    assert _angular_error_m(candidate, native, transformer._pipeline.computed["a"]) <= 1e-8


@pytest.mark.parametrize("kind", ["north_pole", "south_pole", "outside", "nan_x", "inf_x"])
def test_recurrence_homogeneous_cold_warps_are_bitwise_native(kind):
    transformer = _transformer()
    computed = transformer._pipeline.computed
    pole_y = computed["y0"] + computed["a"] * computed["meridional_pole"]
    values = {
        "north_pole": (1.0, pole_y),
        "south_pole": (1.0, -pole_y),
        "outside": (1.0, math.nextafter(pole_y, math.inf)),
        "nan_x": (math.nan, 0.0),
        "inf_x": (math.inf, 0.0),
    }
    first, second = values[kind]
    x = cp.full(64, first, dtype=cp.float64)
    y = cp.full(64, second, dtype=cp.float64)
    native = _run_exact(transformer, x, y, NATIVE_LIBDEVICE)
    candidate = _run_exact(transformer, x, y, SINU_INVERSE_MERIDIONAL_RECURRENCE)
    assert _bitwise_equal(candidate, native)


@pytest.mark.parametrize("kind", ["north_pole", "adjacent", "outside", "nan_x", "inf_y"])
def test_recurrence_one_cold_lane_makes_complete_warp_bitwise_native(kind):
    transformer = _transformer()
    witness_longitude = cp.full(32, -57.306659342000856, dtype=cp.float64)
    witness_latitude = cp.full(32, -28.749374351461878, dtype=cp.float64)
    projected = transformer.transform_buffers(
        witness_longitude,
        witness_latitude,
        precision="fp64",
        transcendentals="native",
    )
    x, y = projected[0].copy(), projected[1].copy()
    computed = transformer._pipeline.computed
    pole_y = computed["y0"] + computed["a"] * computed["meridional_pole"]
    values = {
        "north_pole": (1.0, pole_y),
        "adjacent": (1.0, math.nextafter(pole_y, 0.0)),
        "outside": (1.0, math.nextafter(pole_y, math.inf)),
        "nan_x": (math.nan, 0.0),
        "inf_y": (0.0, math.inf),
    }
    x[0], y[0] = values[kind]
    native = _run_exact(transformer, x, y, NATIVE_LIBDEVICE)
    candidate = _run_exact(transformer, x, y, SINU_INVERSE_MERIDIONAL_RECURRENCE)
    assert _bitwise_equal(candidate, native)


def test_recurrence_post_candidate_longitude_guard_makes_complete_warp_native():
    transformer = _transformer()
    witness_longitude = cp.full(32, -57.306659342000856, dtype=cp.float64)
    witness_latitude = cp.full(32, -28.749374351461878, dtype=cp.float64)
    projected = transformer.transform_buffers(
        witness_longitude,
        witness_latitude,
        precision="fp64",
        transcendentals="native",
    )
    x, y = projected[0].copy(), projected[1].copy()
    computed = transformer._pipeline.computed
    phi = math.radians(45.0)
    sin_phi = math.sin(phi)
    denominator = math.cos(phi) / math.sqrt(1.0 - computed["es"] * sin_phi * sin_phi)
    x[0] = computed["x0"] + computed["a"] * math.nextafter(math.pi, math.inf) * denominator
    y[0] = computed["y0"] + computed["a"] * sum(
        coefficient * (phi if harmonic == 0 else math.sin(2.0 * harmonic * phi))
        for harmonic, coefficient in enumerate(computed["meridional_coefficients"])
    )
    native = _run_exact(transformer, x, y, NATIVE_LIBDEVICE)
    candidate = _run_exact(transformer, x, y, SINU_INVERSE_MERIDIONAL_RECURRENCE)
    assert _bitwise_equal(candidate, native)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_recurrence_899_degree_physical_y_boundary_is_nonvacuous(sign):
    transformer = _transformer()
    computed = transformer._pipeline.computed
    hot_cy_limit = computed["meridional_recurrence_hot_cy_limit"]

    boundary_y = computed["y0"] + sign * computed["a"] * hot_cy_limit

    def normalized_y(public_y):
        return (public_y - computed["y0"]) / computed["a"]

    inside_y = boundary_y
    while abs(normalized_y(inside_y)) > hot_cy_limit:
        inside_y = math.nextafter(inside_y, computed["y0"])
    outside_y = boundary_y
    away = math.copysign(math.inf, sign)
    while abs(normalized_y(outside_y)) <= hot_cy_limit:
        outside_y = math.nextafter(outside_y, away)

    assert abs(normalized_y(inside_y)) <= hot_cy_limit
    assert abs(normalized_y(outside_y)) > hot_cy_limit
    assert math.nextafter(inside_y, away) == outside_y

    witness_longitude = cp.full(32, -57.306659342000856, dtype=cp.float64)
    witness_latitude = cp.full(32, -28.749374351461878, dtype=cp.float64)
    projected = transformer.transform_buffers(
        witness_longitude,
        witness_latitude,
        precision="fp64",
        transcendentals="native",
    )
    ordinary_native = _run_exact(transformer, *projected, NATIVE_LIBDEVICE)
    ordinary_candidate = _run_exact(transformer, *projected, SINU_INVERSE_MERIDIONAL_RECURRENCE)
    assert not _bitwise_equal(ordinary_candidate, ordinary_native)

    for public_y, expects_native in ((inside_y, False), (outside_y, True)):
        x, y = projected[0].copy(), projected[1].copy()
        x[0] = computed["x0"]
        y[0] = public_y
        native = _run_exact(transformer, x, y, NATIVE_LIBDEVICE)
        candidate = _run_exact(transformer, x, y, SINU_INVERSE_MERIDIONAL_RECURRENCE)
        assert _bitwise_equal(candidate, native) is expects_native
        if not expects_native:
            assert _angular_error_m(candidate, native, computed["a"]) <= 1e-8


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("es", 0.0),
        ("es", math.nextafter(0.012, math.inf)),
        ("es", math.nan),
        ("lam0", math.inf),
        ("a", 0.0),
        ("a", math.nextafter(6_400_000.0, math.inf)),
        ("x0", math.inf),
        ("y0", math.nan),
        ("x_unit_to_m", 0.0),
        ("x_unit_to_m", math.inf),
        ("y_unit_to_m", 0.0),
        ("y_unit_to_m", math.nan),
        ("c0", 0.0),
        ("c1", math.nan),
        ("c7", math.inf),
    ],
)
def test_recurrence_device_setup_guards_are_bitwise_native(field, value):
    transformer = _transformer()
    computed = transformer._pipeline.computed
    _, _, projected = _projected_hot(transformer, 64)
    if field.startswith("c"):
        index = int(field[1:])
        original = computed["meridional_coefficients"]
        coefficients = list(original)
        coefficients[index] = value
        computed["meridional_coefficients"] = tuple(coefficients)
        restore_field, restore_value = "meridional_coefficients", original
    else:
        restore_field, restore_value = field, computed[field]
        computed[field] = value
    try:
        native = _run_exact(transformer, *projected, NATIVE_LIBDEVICE)
        candidate = _run_exact(transformer, *projected, SINU_INVERSE_MERIDIONAL_RECURRENCE)
        assert _bitwise_equal(candidate, native)
    finally:
        computed[restore_field] = restore_value


def test_convergent_newton_preserves_full_native_nonfinite_and_pole_classes():
    transformer = _transformer()
    computed = transformer._pipeline.computed
    pole_y = computed["y0"] + computed["a"] * computed["meridional_pole"]
    x = cp.asarray([0.0, 1.0, math.inf, -math.inf, math.nan] * 5, dtype=cp.float64)
    y = cp.repeat(
        cp.asarray(
            [0.0, math.nextafter(pole_y, 0.0), math.inf, -math.inf, math.nan],
            dtype=cp.float64,
        ),
        5,
    )
    native = _run_exact(transformer, x, y, NATIVE_LIBDEVICE)
    candidate = _run_exact(transformer, x, y, SINU_INVERSE_CONVERGENT_NEWTON)
    for actual, expected in zip(candidate, native, strict=True):
        assert bool(cp.all(cp.isfinite(actual) == cp.isfinite(expected)).get())
        assert bool(cp.all(cp.isnan(actual) == cp.isnan(expected)).get())
        assert bool(cp.all(cp.signbit(actual) == cp.signbit(expected)).get())
    assert _angular_error_m(candidate, native, computed["a"]) <= 1e-8


def test_public_policy_ids_buffers_stream_and_concurrency():
    transformer = _transformer()
    _, _, projected = _projected_hot(transformer, 4096)
    expected = transformer.transform_buffers(
        *projected, direction="INVERSE", precision="fp64", transcendentals="native"
    )
    stream = cp.cuda.Stream(non_blocking=True)
    out_x, out_y = cp.empty_like(projected[0]), cp.empty_like(projected[1])
    actual = transformer.transform_buffers(
        *projected,
        direction="INVERSE",
        precision="fp64",
        transcendentals="accelerated",
        out_x=out_x,
        out_y=out_y,
        stream=stream,
    )
    stream.synchronize()
    assert actual[0] is out_x and actual[1] is out_y
    assert _angular_error_m(actual, expected, transformer._pipeline.computed["a"]) <= 1e-8

    automatic = transformer.explain_strategy(
        direction="INVERSE", precision="fp64", transcendentals="auto", workload_size=1
    )
    explicit = transformer.explain_strategy(
        direction="INVERSE", precision="fp64", transcendentals="accelerated", workload_size=1
    )
    assert automatic.decisions[-1].implementation_id == SINU_INVERSE_CONVERGENT_NEWTON
    assert explicit.decisions[-1].implementation_id == SINU_INVERSE_MERIDIONAL_RECURRENCE

    auto_stream = cp.cuda.Stream(non_blocking=True)
    recurrence_stream = cp.cuda.Stream(non_blocking=True)
    auto_out = (cp.empty_like(projected[0]), cp.empty_like(projected[1]))
    recurrence_out = (cp.empty_like(projected[0]), cp.empty_like(projected[1]))
    with auto_stream:
        auto_result = transformer.transform_buffers(
            *projected,
            direction="INVERSE",
            precision="fp64",
            transcendentals="auto",
            out_x=auto_out[0],
            out_y=auto_out[1],
            stream=auto_stream,
        )
    with recurrence_stream:
        recurrence_result = transformer.transform_buffers(
            *projected,
            direction="INVERSE",
            precision="fp64",
            transcendentals="accelerated",
            out_x=recurrence_out[0],
            out_y=recurrence_out[1],
            stream=recurrence_stream,
        )
    auto_stream.synchronize()
    recurrence_stream.synchronize()
    assert auto_result[0] is auto_out[0] and auto_result[1] is auto_out[1]
    assert recurrence_result[0] is recurrence_out[0]
    assert recurrence_result[1] is recurrence_out[1]
    assert _angular_error_m(auto_result, expected, transformer._pipeline.computed["a"]) <= 1e-8
    assert (
        _angular_error_m(recurrence_result, expected, transformer._pipeline.computed["a"]) <= 1e-8
    )

    fused_module._kernel_cache.clear()
    with ThreadPoolExecutor(max_workers=8) as executor:
        kernels = list(
            executor.map(
                lambda _: fused_module._get_kernel(
                    "sinu",
                    "inverse",
                    "float64",
                    transcendental_impl=SINU_INVERSE_MERIDIONAL_RECURRENCE,
                ),
                range(32),
            )
        )
    assert all(kernel is kernels[0] for kernel in kernels)


@pytest.mark.parametrize("always_xy", [False, True])
@pytest.mark.parametrize("axis", ["enu", "swu"])
@pytest.mark.parametrize("unit", ["m", "us-ft"])
def test_public_candidates_preserve_custom_units_offsets_axes_and_order(always_xy, axis, unit):
    from pyproj import CRS

    target = CRS.from_user_input(
        f"+proj=sinu +lon_0=12 +x_0=100 +y_0=-200 +ellps=WGS84 +axis={axis} +units={unit} +type=crs"
    )
    transformer = Transformer.from_crs(target.geodetic_crs, target, always_xy=always_xy)
    longitude = cp.linspace(-160.0, 170.0, 4096, dtype=cp.float64)
    latitude = cp.linspace(-80.0, 80.0, 4096, dtype=cp.float64)
    if always_xy:
        first, second = longitude, latitude
        north_first = False
    else:
        north_first = target.geodetic_crs.axis_info[0].direction.lower() in {"north", "south"}
        first, second = (latitude, longitude) if north_first else (longitude, latitude)
    projected = transformer.transform_buffers(
        first, second, precision="fp64", transcendentals="native"
    )
    native = transformer.transform_buffers(
        *projected, direction="INVERSE", precision="fp64", transcendentals="native"
    )
    for policy in ("auto", "accelerated"):
        candidate = transformer.transform_buffers(
            *projected,
            direction="INVERSE",
            precision="fp64",
            transcendentals=policy,
        )
        candidate_lon_lat = (candidate[1], candidate[0]) if north_first else candidate
        native_lon_lat = (native[1], native[0]) if north_first else native
        assert (
            _angular_error_m(
                candidate_lon_lat,
                native_lon_lat,
                transformer._pipeline.computed["a"],
            )
            <= 1e-8
        )


def test_sinu_inverse_sources_preserve_native_source_and_expected_abis():
    template, native_name = fused_module._SOURCE_MAP[("sinu", "inverse")]
    native_source = fused_module._inject_linear_unit_args(
        template.format(
            real_t="double",
            pi=fused_module._PI_LITERALS["float64"],
            tol=fused_module._TOL_LITERALS["float64"],
        )
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
    early_source, early_name = fused_module._build_projection_guarded_source(
        "sinu", "inverse", SINU_INVERSE_CONVERGENT_NEWTON
    )
    recurrence_source, recurrence_name = fused_module._build_projection_guarded_source(
        "sinu", "inverse", SINU_INVERSE_MERIDIONAL_RECURRENCE
    )
    assert parameters(early_source, early_name) == native_parameters
    assert "hot_cy_limit" not in native_source
    assert "hot_cy_limit" not in parameters(early_source, early_name)
    assert "double es, double hot_cy_limit" in parameters(recurrence_source, recurrence_name)
    assert "if (fabs(delta) < 1e-14) break;" in early_source
    assert "__any_sync" in recurrence_source
    assert "vp_sinu_inverse_native_ellipsoid" in recurrence_source
    assert "meridional_recurrence" not in native_source
    assert "convergent_newton" not in native_source


@pytest.mark.parametrize(
    ("direction", "dtype", "implementation"),
    [
        ("forward", "float64", SINU_INVERSE_CONVERGENT_NEWTON),
        ("forward", "float64", SINU_INVERSE_MERIDIONAL_RECURRENCE),
        ("inverse", "float32", SINU_INVERSE_CONVERGENT_NEWTON),
        ("inverse", "float32", SINU_INVERSE_MERIDIONAL_RECURRENCE),
    ],
)
def test_sinu_inverse_implementations_reject_wrong_direction_or_precision(
    direction, dtype, implementation
):
    with pytest.raises(ValueError, match="qualified only"):
        fused_module._validate_projection_implementation("sinu", direction, dtype, implementation)


@pytest.mark.parametrize(
    "implementation",
    [SINU_INVERSE_CONVERGENT_NEWTON, SINU_INVERSE_MERIDIONAL_RECURRENCE],
)
def test_sinu_inverse_kernel_resources_cache_and_graph_residency(implementation):
    fused_module._kernel_cache.clear()
    native = fused_module._get_kernel(
        "sinu", "inverse", "float64", transcendental_impl=NATIVE_LIBDEVICE
    )
    candidate = fused_module._get_kernel(
        "sinu", "inverse", "float64", transcendental_impl=implementation
    )
    assert native is not candidate
    candidate.compile()
    assert candidate.attributes["num_regs"] <= 38
    assert candidate.attributes["local_size_bytes"] <= 56
    assert candidate.attributes["shared_size_bytes"] == 0

    transformer = _transformer()
    _, _, projected = _projected_hot(transformer, 4096)
    out_x, out_y = cp.empty_like(projected[0]), cp.empty_like(projected[1])
    stream = cp.cuda.Stream(non_blocking=True)
    stream.begin_capture()
    with stream:
        result = _run_exact(
            transformer,
            *projected,
            implementation,
            out_x=out_x,
            out_y=out_y,
            stream=stream,
        )
    graph = stream.end_capture()
    dot = graph.debug_dot_str().lower()
    assert result[0] is out_x and result[1] is out_y
    assert dot.count(implementation.replace(".", "_")) == 1
    assert "memcpy" not in dot
    assert "memset" not in dot
