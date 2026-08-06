"""CUDA qualification for the explicit Gnomonic inverse reframe."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

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
from vibeproj import Transformer  # noqa: E402
from vibeproj.crs import ProjectionParams  # noqa: E402
from vibeproj.ellipsoid import Ellipsoid  # noqa: E402
from vibeproj.pipeline import TransformPipeline  # noqa: E402
from vibeproj.transcendentals import (  # noqa: E402
    GNOM_INVERSE_GUARDED_RSQRT_REFRAME,
    NATIVE_LIBDEVICE,
    PROJECTION_FIXED_Q62_MAX_SCALE_M,
)


RADIUS_M = 6_378_137.0


def _pipeline(
    *,
    lat_0: float = 0.0,
    radius: float = RADIUS_M,
    x_0: float = 0.0,
    y_0: float = 0.0,
    x_unit_to_m: float = 1.0,
    y_unit_to_m: float = 1.0,
    easting_axis_sign: float = 1.0,
    northing_axis_sign: float = 1.0,
) -> TransformPipeline:
    sphere = Ellipsoid(a=radius, b=radius, f=0.0, e=0.0, es=0.0, n=0.0)
    geographic = ProjectionParams("longlat", sphere, north_first=False)
    projected = ProjectionParams(
        "gnom",
        sphere,
        lat_0=lat_0,
        x_0=x_0,
        y_0=y_0,
        x_unit_to_m=x_unit_to_m,
        y_unit_to_m=y_unit_to_m,
        easting_axis_sign=easting_axis_sign,
        northing_axis_sign=northing_axis_sign,
        north_first=False,
    )
    return TransformPipeline(projected, geographic)


def _run(
    pipeline,
    x,
    y,
    implementation_id,
    *,
    out_x=None,
    out_y=None,
    stream=None,
    computed=None,
):
    return fused_module.fused_transform(
        x,
        y,
        projection_name="gnom",
        direction="inverse",
        computed=pipeline.computed if computed is None else computed,
        src_north_first=pipeline.src_north_first,
        dst_north_first=pipeline.dst_north_first,
        xp=cp,
        out_x=out_x,
        out_y=out_y,
        stream=stream,
        precision="fp64",
        transcendental_impl=implementation_id,
    )


def _assert_bitwise_equal(actual, expected) -> None:
    for got, want in zip(actual, expected, strict=True):
        assert_array_equal(cp.asnumpy(got).view(np.uint64), cp.asnumpy(want).view(np.uint64))


def _error_m(actual, expected, radius: float) -> np.ndarray:
    actual_lon, actual_lat = (cp.asnumpy(value) for value in actual)
    expected_lon, expected_lat = (cp.asnumpy(value) for value in expected)
    delta_lon = (actual_lon - expected_lon + 180.0) % 360.0 - 180.0
    east = np.deg2rad(delta_lon) * radius * np.cos(np.deg2rad(expected_lat))
    north = np.deg2rad(actual_lat - expected_lat) * radius
    return np.hypot(east, north)


def test_gnom_inverse_variant_compile_cache_and_resources():
    fused_module._kernel_cache.clear()
    native = fused_module._get_kernel(
        "gnom", "inverse", "float64", transcendental_impl=NATIVE_LIBDEVICE
    )
    accelerated = fused_module._get_kernel(
        "gnom",
        "inverse",
        "float64",
        transcendental_impl=GNOM_INVERSE_GUARDED_RSQRT_REFRAME,
    )
    native.compile()
    accelerated.compile()

    assert native is not accelerated
    assert accelerated.attributes["num_regs"] <= 46
    assert accelerated.attributes["local_size_bytes"] <= 16
    assert accelerated.attributes["shared_size_bytes"] == 0


def test_gnom_inverse_variant_rejects_wrong_operation_and_precision():
    for precision in ("float32", "ds"):
        with pytest.raises(ValueError, match="qualified only"):
            fused_module._get_kernel(
                "gnom",
                "inverse",
                precision,
                transcendental_impl=GNOM_INVERSE_GUARDED_RSQRT_REFRAME,
            )
    with pytest.raises(ValueError, match="qualified only"):
        fused_module._get_kernel(
            "gnom",
            "forward",
            "float64",
            transcendental_impl=GNOM_INVERSE_GUARDED_RSQRT_REFRAME,
        )


@pytest.mark.parametrize("lat_0", [0.0, 45.0, 60.0])
def test_gnom_inverse_hot_domain_meets_ten_nanometer_contract(lat_0):
    rng = np.random.default_rng(20260806 + int(lat_0))
    count = 300_000
    rho = np.sqrt(rng.uniform(1e-12, np.nextafter(0.02, 0.0), count))
    azimuth = rng.uniform(-math.pi, math.pi, count)
    pipeline = _pipeline(lat_0=lat_0)
    x = cp.asarray(rho * np.cos(azimuth) * RADIUS_M)
    y = cp.asarray(rho * np.sin(azimuth) * RADIUS_M)
    native = _run(pipeline, x, y, NATIVE_LIBDEVICE)
    accelerated = _run(pipeline, x, y, GNOM_INVERSE_GUARDED_RSQRT_REFRAME)

    assert float(np.max(_error_m(accelerated, native, RADIUS_M))) <= 1e-8


def test_gnom_inverse_cold_boundaries_axes_signed_zero_and_nonfinite_are_bit_exact():
    x_host = np.asarray(
        [
            0.0,
            -0.0,
            0.05,
            0.0,
            math.inf,
            -math.inf,
            math.nan,
        ],
        dtype=np.float64,
    )
    y_host = np.asarray(
        [0.0, -0.0, 0.0, 0.05] + [0.0] * 3,
        dtype=np.float64,
    )
    pipeline = _pipeline(radius=1.0)
    x = cp.asarray(x_host)
    y = cp.asarray(y_host)
    native = _run(pipeline, x, y, NATIVE_LIBDEVICE)
    accelerated = _run(pipeline, x, y, GNOM_INVERSE_GUARDED_RSQRT_REFRAME)

    _assert_bitwise_equal(accelerated, native)


@pytest.mark.parametrize(
    ("boundary", "x_value", "y_value", "hot"),
    [
        ("lower-below", 6.324555320336758e-13, 7.745966692414833e-13, False),
        ("lower-exact", 6.324555320336758e-13, 7.745966692414834e-13, False),
        ("lower-above", 6.324555320336758e-13, 7.745966692414835e-13, True),
        ("upper-below", 0.08944271909999159, 0.10954451150103321, True),
        ("upper-exact", 0.08944271909999159, 0.10954451150103323, True),
        ("upper-above", 0.08944271909999159, 0.10954451150103324, False),
    ],
)
def test_gnom_inverse_rho_squared_boundaries_use_homogeneous_warps(boundary, x_value, y_value, hot):
    del boundary
    pipeline = _pipeline(radius=1.0)
    x = cp.full(256, x_value, dtype=cp.float64)
    y = cp.full(256, y_value, dtype=cp.float64)
    native = _run(pipeline, x, y, NATIVE_LIBDEVICE)
    accelerated = _run(pipeline, x, y, GNOM_INVERSE_GUARDED_RSQRT_REFRAME)

    if hot:
        assert float(np.max(_error_m(accelerated, native, 1.0))) <= 1e-8
    else:
        _assert_bitwise_equal(accelerated, native)


def test_shared_rsqrt_helper_selects_exact_and_adjacent_rho_squared_boundaries():
    source = (
        fused_module._PROJECTION_GUARDED_NATIVE_FALLBACK_HELPERS
        + r"""
extern "C" __global__ void vp_gnom_rho_guard_probe(
    const double* x, const double* y, double* rho_squared, int* selected, int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double inverse_norm;
    selected[i] = vp_projection_guarded_rsqrt_norm(
        x[i], y[i], 1e-24, 0.02, &rho_squared[i], &inverse_norm
    );
}
"""
    )
    probe = cp.RawKernel(source, "vp_gnom_rho_guard_probe", options=("--std=c++11",))
    x = cp.asarray(
        [
            6.324555320336758e-13,
            6.324555320336758e-13,
            6.324555320336758e-13,
            0.08944271909999159,
            0.08944271909999159,
            0.08944271909999159,
        ]
    )
    y = cp.asarray(
        [
            7.745966692414833e-13,
            7.745966692414834e-13,
            7.745966692414835e-13,
            0.10954451150103321,
            0.10954451150103323,
            0.10954451150103324,
        ]
    )
    rho_squared = cp.empty_like(x)
    selected = cp.empty(x.size, dtype=cp.int32)
    probe((1,), (32,), (x, y, rho_squared, selected, np.int32(x.size)))

    assert_array_equal(cp.asnumpy(selected), np.asarray([0, 0, 1, 1, 1, 0]))
    assert_array_equal(
        cp.asnumpy(rho_squared).view(np.uint64),
        np.asarray(
            [
                np.nextafter(1e-24, 0.0),
                1e-24,
                np.nextafter(1e-24, math.inf),
                np.nextafter(0.02, 0.0),
                0.02,
                np.nextafter(0.02, math.inf),
            ],
            dtype=np.float64,
        ).view(np.uint64),
    )


@pytest.mark.parametrize(
    ("cos_phi0", "qualified"),
    [(0.5, True), (np.nextafter(0.5, 0.0), False), (0.0, False)],
)
def test_gnom_inverse_cos_phi0_exact_boundary_and_nextbelow(cos_phi0, qualified):
    pipeline = _pipeline(radius=1.0, lat_0=60.0)
    computed = dict(pipeline.computed)
    computed["cos_phi0"] = cos_phi0
    computed["sin_phi0"] = math.sqrt(1.0 - cos_phi0 * cos_phi0)
    x = cp.linspace(0.01, 0.09, 256, dtype=cp.float64)
    y = cp.linspace(0.09, 0.01, 256, dtype=cp.float64)
    native = _run(pipeline, x, y, NATIVE_LIBDEVICE, computed=computed)
    accelerated = _run(
        pipeline,
        x,
        y,
        GNOM_INVERSE_GUARDED_RSQRT_REFRAME,
        computed=computed,
    )

    if qualified:
        assert float(np.max(_error_m(accelerated, native, 1.0))) <= 1e-8
    else:
        _assert_bitwise_equal(accelerated, native)


@pytest.mark.parametrize(
    ("radius", "lat_0"),
    [
        (np.nextafter(PROJECTION_FIXED_Q62_MAX_SCALE_M, math.inf), 0.0),
        (math.inf, 0.0),
        (math.nan, 0.0),
        (RADIUS_M, np.nextafter(60.0, math.inf)),
        (RADIUS_M, 90.0),
        (RADIUS_M, -90.0),
    ],
)
def test_gnom_inverse_unqualified_scale_and_origin_are_bit_exact(radius, lat_0):
    input_scale = RADIUS_M if not math.isfinite(radius) else radius
    pipeline = _pipeline(radius=radius, lat_0=lat_0)
    x = cp.asarray([0.04, -0.08, 0.1]) * input_scale
    y = cp.asarray([0.06, 0.03, -0.02]) * input_scale
    native = _run(pipeline, x, y, NATIVE_LIBDEVICE)
    accelerated = _run(pipeline, x, y, GNOM_INVERSE_GUARDED_RSQRT_REFRAME)

    _assert_bitwise_equal(accelerated, native)


def test_gnom_inverse_exact_maximum_scale_remains_qualified():
    radius = PROJECTION_FIXED_Q62_MAX_SCALE_M
    pipeline = _pipeline(radius=radius, lat_0=45.0)
    x = cp.full(256, 0.08 * radius, dtype=cp.float64)
    y = cp.full(256, 0.06 * radius, dtype=cp.float64)
    native = _run(pipeline, x, y, NATIVE_LIBDEVICE)
    accelerated = _run(pipeline, x, y, GNOM_INVERSE_GUARDED_RSQRT_REFRAME)

    assert float(np.max(_error_m(accelerated, native, radius))) <= 1e-8


def test_gnom_lane_guard_selects_exact_cosine_and_scale_boundaries():
    source = (
        fused_module._PROJECTION_SCALE_GUARD_DEVICE_FNS
        + fused_module._PROJECTION_GUARDED_NATIVE_FALLBACK_HELPERS
        + r"""
extern "C" __global__ void vp_gnom_lane_guard_probe(
    double cos_phi0, double physical_scale, int* selected
) {
    double rho_squared, inverse_norm;
    const double cx = 0.08;
    const double cy = 0.06;
    const double sin_phi0 = sqrt(1.0 - cos_phi0 * cos_phi0);
    const bool radial_qualified = vp_projection_guarded_rsqrt_norm(
        cx, cy, 1e-24, 0.02, &rho_squared, &inverse_norm
    );
    const double phi_argument = fma(cy, cos_phi0, sin_phi0) * inverse_norm;
    const double lam_denominator = fma(-cy, sin_phi0, cos_phi0);
    selected[0] = radial_qualified && cx != 0.0 && cy != 0.0
        && isfinite(sin_phi0) && isfinite(cos_phi0)
        && fabs(cos_phi0) >= 0.5
        && vp_projection_fixed_scale_is_qualified(physical_scale)
        && fabs(phi_argument) <= 1.0 && isfinite(lam_denominator);
}
"""
    )
    probe = cp.RawKernel(source, "vp_gnom_lane_guard_probe", options=("--std=c++11",))

    def selected(cos_phi0: float, scale: float) -> int:
        output = cp.empty(1, dtype=cp.int32)
        probe((1,), (1,), (cp.float64(cos_phi0), cp.float64(scale), output))
        return int(output[0].get())

    assert selected(0.5, PROJECTION_FIXED_Q62_MAX_SCALE_M) == 1
    assert selected(np.nextafter(0.5, 0.0), PROJECTION_FIXED_Q62_MAX_SCALE_M) == 0
    assert selected(0.5, np.nextafter(PROJECTION_FIXED_Q62_MAX_SCALE_M, math.inf)) == 0


def test_public_gnom_inverse_compile_preallocated_nondefault_stream_and_auto_native():
    projected = "+proj=gnom +lat_0=45 +lon_0=0 +R=6378137 +units=m +type=crs"
    geographic = "+proj=longlat +R=6378137 +type=crs"
    transformer = Transformer.from_crs(projected, geographic, always_xy=True)
    transformer.compile(precision="fp64", transcendentals="accelerated")
    assert (
        "gnom",
        "inverse",
        "float64",
        GNOM_INVERSE_GUARDED_RSQRT_REFRAME,
    ) in fused_module._kernel_cache

    x = cp.linspace(0.02, 0.10, 4096, dtype=cp.float64) * RADIUS_M
    y = cp.linspace(-0.10, -0.02, 4096, dtype=cp.float64) * RADIUS_M
    native = transformer.transform_buffers(x, y, precision="fp64", transcendentals="native")
    out_x = cp.empty_like(x)
    out_y = cp.empty_like(y)
    stream = cp.cuda.Stream(non_blocking=True)
    accelerated = transformer.transform_buffers(
        x,
        y,
        out_x=out_x,
        out_y=out_y,
        precision="fp64",
        transcendentals="accelerated",
        stream=stream,
    )
    automatic = transformer.transform_buffers(
        x,
        y,
        precision="fp64",
        transcendentals="auto",
        stream=stream,
    )
    stream.synchronize()

    assert accelerated[0] is out_x
    assert accelerated[1] is out_y
    assert float(np.max(_error_m(accelerated, native, RADIUS_M))) <= 1e-8
    _assert_bitwise_equal(automatic, native)


def test_gnom_inverse_signed_units_offsets_preallocated_stream_and_concurrency():
    pipeline = _pipeline(
        lat_0=45.0,
        x_0=321_000.0,
        y_0=-654_000.0,
        x_unit_to_m=0.3048,
        y_unit_to_m=0.3048,
        easting_axis_sign=-1.0,
        northing_axis_sign=-1.0,
    )
    normalized_x = np.linspace(0.01, 0.10, 4096)
    normalized_y = np.linspace(-0.10, -0.01, 4096)
    x = cp.asarray((normalized_x * RADIUS_M + 321_000.0) / -0.3048)
    y = cp.asarray((normalized_y * RADIUS_M - 654_000.0) / -0.3048)
    native = _run(pipeline, x, y, NATIVE_LIBDEVICE)
    out_x = cp.empty_like(x)
    out_y = cp.empty_like(y)
    stream = cp.cuda.Stream(non_blocking=True)
    accelerated = _run(
        pipeline,
        x,
        y,
        GNOM_INVERSE_GUARDED_RSQRT_REFRAME,
        out_x=out_x,
        out_y=out_y,
        stream=stream,
    )
    stream.synchronize()

    assert accelerated[0] is out_x
    assert accelerated[1] is out_y
    assert float(np.max(_error_m(accelerated, native, RADIUS_M))) <= 1e-8

    fused_module._kernel_cache.clear()
    with ThreadPoolExecutor(max_workers=8) as executor:
        kernels = list(
            executor.map(
                lambda _: fused_module._get_kernel(
                    "gnom",
                    "inverse",
                    "float64",
                    transcendental_impl=GNOM_INVERSE_GUARDED_RSQRT_REFRAME,
                ),
                range(32),
            )
        )
    assert all(kernel is kernels[0] for kernel in kernels)
