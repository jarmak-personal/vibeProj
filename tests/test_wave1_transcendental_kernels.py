"""CUDA qualification tests for Wave 1 projection transcendental candidates."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device not available", allow_module_level=True)
except Exception as exc:  # pragma: no cover - import-time environment guard
    pytest.skip(f"CUDA device not available: {exc}", allow_module_level=True)

from vibeproj.crs import ProjectionParams  # noqa: E402
from vibeproj.ellipsoid import Ellipsoid, WGS84  # noqa: E402
from vibeproj._transcendental_device_fns import (  # noqa: E402
    PROJECTION_BOUNDED_Q62_DEVICE_FNS,
    PROJECTION_FIXED_Q62_MAX_SCALE_M,
)
import vibeproj.fused_kernels as fused_module  # noqa: E402
from vibeproj.fused_kernels import (  # noqa: E402
    NATIVE_LIBDEVICE,
    ORTHO_FORWARD_FIXED_Q62,
    SINU_FORWARD_FIXED_Q62,
    _get_kernel,
    _kernel_cache,
    compile_kernels,
    fused_transform,
)
from vibeproj.pipeline import TransformPipeline  # noqa: E402


_FIXED_COS_GUARD_TEST_SOURCE = (
    PROJECTION_BOUNDED_Q62_DEVICE_FNS
    + r"""
extern "C" __global__ void projection_fixed_cos_guard_test(
    const double* angles,
    double caller_bound,
    double physical_scale,
    double* accelerated,
    double* native,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    accelerated[idx] = vp_projection_fixed_cos(
        angles[idx], caller_bound, physical_scale
    );
    native[idx] = cos(angles[idx]);
}
"""
)


def _pipeline(
    projection: str,
    direction: str,
    *,
    lat_0: float = 0.0,
    lon_0: float = 0.0,
    radius: float = WGS84.a,
    x_0: float = 0.0,
    y_0: float = 0.0,
    x_unit_to_m: float = 1.0,
    y_unit_to_m: float = 1.0,
    src_north_first: bool = True,
    dst_north_first: bool = False,
):
    sphere = Ellipsoid(
        a=radius,
        b=radius,
        f=0.0,
        e=0.0,
        es=0.0,
        n=0.0,
    )
    geographic = ProjectionParams(
        projection_name="longlat",
        ellipsoid=sphere,
        north_first=src_north_first,
    )
    projected = ProjectionParams(
        projection_name=projection,
        ellipsoid=sphere,
        lon_0=lon_0,
        lat_0=lat_0,
        x_0=x_0,
        y_0=y_0,
        x_unit_to_m=x_unit_to_m,
        y_unit_to_m=y_unit_to_m,
        north_first=dst_north_first,
    )
    if direction == "forward":
        return TransformPipeline(geographic, projected)
    return TransformPipeline(projected, geographic)


def _run_exact(
    pipeline,
    arg1,
    arg2,
    projection: str,
    direction: str,
    implementation_id: str,
    *,
    out_x=None,
    out_y=None,
    stream=None,
    computed=None,
):
    return fused_transform(
        arg1,
        arg2,
        projection_name=projection,
        direction=direction,
        computed=pipeline.computed if computed is None else computed,
        src_north_first=pipeline.src_north_first,
        dst_north_first=pipeline.dst_north_first,
        xp=cp,
        out_x=out_x,
        out_y=out_y,
        precision="fp64",
        transcendental_impl=implementation_id,
        stream=stream,
    )


def test_projection_fixed_cos_retains_hard_pi_guard_for_overbroad_caller_bound():
    angles_host = np.array(
        [
            np.nextafter(-np.pi, -np.inf),
            -2.0 * np.pi,
            np.nextafter(np.pi, np.inf),
            2.0 * np.pi,
            -np.inf,
            np.inf,
            np.nan,
        ],
        dtype=np.float64,
    )
    angles = cp.asarray(angles_host)
    accelerated = cp.empty_like(angles)
    native = cp.empty_like(angles)
    kernel = cp.RawKernel(_FIXED_COS_GUARD_TEST_SOURCE, "projection_fixed_cos_guard_test")
    kernel(
        (1,),
        (256,),
        (
            angles,
            np.float64(100.0),
            np.float64(WGS84.a),
            accelerated,
            native,
            np.int32(angles.size),
        ),
    )

    assert_array_equal(
        cp.asnumpy(accelerated).view(np.uint64),
        cp.asnumpy(native).view(np.uint64),
    )


@pytest.mark.parametrize(
    ("projection", "direction", "implementation_id"),
    [
        ("sinu", "forward", SINU_FORWARD_FIXED_Q62),
        ("ortho", "forward", ORTHO_FORWARD_FIXED_Q62),
    ],
)
def test_wave1_exact_implementation_variants_compile_and_have_distinct_cache_keys(
    projection,
    direction,
    implementation_id,
):
    _kernel_cache.clear()
    native = _get_kernel(
        projection,
        direction,
        "float64",
        transcendental_impl=NATIVE_LIBDEVICE,
    )
    accelerated = _get_kernel(
        projection,
        direction,
        "float64",
        transcendental_impl=implementation_id,
    )
    native.compile()
    accelerated.compile()

    assert native is not accelerated
    assert (projection, direction, "float64", NATIVE_LIBDEVICE) in _kernel_cache
    assert (projection, direction, "float64", implementation_id) in _kernel_cache


@pytest.mark.parametrize(
    ("projection", "direction", "wrong_projection", "wrong_direction", "implementation_id"),
    [
        ("sinu", "forward", "sinu", "inverse", SINU_FORWARD_FIXED_Q62),
        ("ortho", "forward", "ortho", "inverse", ORTHO_FORWARD_FIXED_Q62),
    ],
)
def test_wave1_exact_implementation_rejects_wrong_precision_and_operation(
    projection,
    direction,
    wrong_projection,
    wrong_direction,
    implementation_id,
):
    for compute_dtype in ("float32", "ds"):
        with pytest.raises(ValueError, match="qualified only"):
            _get_kernel(
                projection,
                direction,
                compute_dtype,
                transcendental_impl=implementation_id,
            )
    with pytest.raises(ValueError, match="qualified only"):
        _get_kernel(
            wrong_projection,
            wrong_direction,
            "float64",
            transcendental_impl=implementation_id,
        )


def test_compile_kernels_consumes_concrete_projection_variants(monkeypatch):
    observed = []

    class CompileSpy:
        def compile(self):
            return None

    def fake_get_kernel(projection, direction, compute_dtype, **kwargs):
        observed.append((projection, direction, compute_dtype, kwargs["transcendental_impl"]))
        return CompileSpy()

    monkeypatch.setattr(fused_module, "_get_kernel", fake_get_kernel)
    compile_kernels(
        precision="fp64",
        projection_variants=(
            ("sinu", "forward", SINU_FORWARD_FIXED_Q62),
            ("sinu", "inverse", NATIVE_LIBDEVICE),
            ("ortho", "forward", ORTHO_FORWARD_FIXED_Q62),
            ("ortho", "inverse", NATIVE_LIBDEVICE),
        ),
    )

    assert observed == [
        ("sinu", "forward", "float64", SINU_FORWARD_FIXED_Q62),
        ("sinu", "inverse", "float64", NATIVE_LIBDEVICE),
        ("ortho", "forward", "float64", ORTHO_FORWARD_FIXED_Q62),
        ("ortho", "inverse", "float64", NATIVE_LIBDEVICE),
    ]


def test_sinu_forward_fixed_q62_matches_native_over_global_valid_domain():
    rng = np.random.default_rng(20260805)
    lat = cp.asarray(rng.uniform(-90.0, 90.0, 200_000), dtype=cp.float64)
    lon = cp.asarray(rng.uniform(-180.0, 180.0, lat.size), dtype=cp.float64)
    pipeline = _pipeline("sinu", "forward")

    native = _run_exact(
        pipeline,
        lat,
        lon,
        "sinu",
        "forward",
        NATIVE_LIBDEVICE,
    )
    accelerated = _run_exact(
        pipeline,
        lat,
        lon,
        "sinu",
        "forward",
        SINU_FORWARD_FIXED_Q62,
    )

    radial_error = cp.hypot(accelerated[0] - native[0], accelerated[1] - native[1])
    assert float(cp.max(radial_error).item()) <= 1e-8


def test_ortho_forward_fixed_q62_matches_native_on_front_and_rear_hemispheres():
    rng = np.random.default_rng(20260806)
    lat_host = np.concatenate(
        (
            rng.uniform(-90.0, 90.0, 200_000),
            np.array([-80.0, -45.0, 0.0, 45.0, 80.0]),
        )
    )
    lon_host = np.concatenate(
        (
            rng.uniform(-180.0, 180.0, 200_000),
            np.array([-179.0, -120.0, 100.0, 135.0, 179.0]),
        )
    )
    lat = cp.asarray(lat_host)
    lon = cp.asarray(lon_host)
    pipeline = _pipeline("ortho", "forward", lat_0=45.0)

    native = _run_exact(
        pipeline,
        lat,
        lon,
        "ortho",
        "forward",
        NATIVE_LIBDEVICE,
    )
    accelerated = _run_exact(
        pipeline,
        lat,
        lon,
        "ortho",
        "forward",
        ORTHO_FORWARD_FIXED_Q62,
    )

    radial_error = cp.hypot(accelerated[0] - native[0], accelerated[1] - native[1])
    assert float(cp.max(radial_error).item()) <= 1e-8
    assert bool(cp.all(cp.isfinite(accelerated[0][-5:])).item())
    assert bool(cp.all(cp.isfinite(accelerated[1][-5:])).item())


@pytest.mark.parametrize("lat_0", [-90.0, -45.0, 0.0, 45.0, 90.0])
def test_ortho_forward_fixed_q62_preserves_horizon_antipode_and_rear_behavior(lat_0):
    horizon_inside = np.nextafter(90.0, 0.0)
    horizon_outside = np.nextafter(90.0, np.inf)
    lat = cp.asarray([0.0, 0.0, 0.0, 0.0, -lat_0], dtype=cp.float64)
    lon = cp.asarray(
        [horizon_inside, 90.0, horizon_outside, 120.0, 180.0],
        dtype=cp.float64,
    )
    pipeline = _pipeline("ortho", "forward", lat_0=lat_0)
    native = _run_exact(
        pipeline,
        lat,
        lon,
        "ortho",
        "forward",
        NATIVE_LIBDEVICE,
    )
    accelerated = _run_exact(
        pipeline,
        lat,
        lon,
        "ortho",
        "forward",
        ORTHO_FORWARD_FIXED_Q62,
    )

    radial_error = cp.hypot(accelerated[0] - native[0], accelerated[1] - native[1])
    assert float(cp.max(radial_error).item()) <= 1e-8
    assert bool(cp.all(cp.isfinite(accelerated[0])).item())
    assert bool(cp.all(cp.isfinite(accelerated[1])).item())


@pytest.mark.parametrize(
    ("projection", "implementation_id", "lat_0"),
    [
        ("sinu", SINU_FORWARD_FIXED_Q62, 0.0),
        ("ortho", ORTHO_FORWARD_FIXED_Q62, 45.0),
    ],
)
def test_wave1_central_meridian_and_antimeridian_wrapping_match_native(
    projection,
    implementation_id,
    lat_0,
):
    lat = cp.asarray([-80.0, -45.0, 0.0, 45.0, 80.0], dtype=cp.float64)
    lon = cp.asarray(
        [-180.0, np.nextafter(-180.0, -np.inf), 179.0, 180.0, 359.0],
        dtype=cp.float64,
    )
    pipeline = _pipeline(projection, "forward", lat_0=lat_0, lon_0=179.0)
    native = _run_exact(
        pipeline,
        lat,
        lon,
        projection,
        "forward",
        NATIVE_LIBDEVICE,
    )
    accelerated = _run_exact(
        pipeline,
        lat,
        lon,
        projection,
        "forward",
        implementation_id,
    )

    radial_error = cp.hypot(accelerated[0] - native[0], accelerated[1] - native[1])
    assert float(cp.max(radial_error).item()) <= 1e-8


@pytest.mark.parametrize(
    ("projection", "implementation_id", "lat_0"),
    [
        ("sinu", SINU_FORWARD_FIXED_Q62, 0.0),
        ("ortho", ORTHO_FORWARD_FIXED_Q62, 45.0),
    ],
)
@pytest.mark.parametrize(
    ("radius", "x_unit_to_m", "y_unit_to_m"),
    [
        (1.0, 1.0, 1.0),
        (3_396_190.0, 0.3048, 0.3048),
        (6_378_137.0, 1000.0, 1000.0),
    ],
)
def test_wave1_radii_units_offsets_and_axis_order_match_native(
    projection,
    implementation_id,
    lat_0,
    radius,
    x_unit_to_m,
    y_unit_to_m,
):
    lat = cp.asarray([-80.0, -45.0, 0.0, 45.0, 80.0], dtype=cp.float64)
    lon = cp.asarray([-170.0, -90.0, 0.0, 100.0, 170.0], dtype=cp.float64)
    pipeline = _pipeline(
        projection,
        "forward",
        lat_0=lat_0,
        lon_0=17.0,
        radius=radius,
        x_0=123_456.75,
        y_0=-987_654.25,
        x_unit_to_m=x_unit_to_m,
        y_unit_to_m=y_unit_to_m,
        src_north_first=False,
        dst_north_first=True,
    )
    # Source is longitude-first and destination is northing-first.
    native = _run_exact(
        pipeline,
        lon,
        lat,
        projection,
        "forward",
        NATIVE_LIBDEVICE,
    )
    accelerated = _run_exact(
        pipeline,
        lon,
        lat,
        projection,
        "forward",
        implementation_id,
    )

    for expected, actual in zip(native, accelerated, strict=True):
        assert_allclose(
            cp.asnumpy(actual),
            cp.asnumpy(expected),
            rtol=0.0,
            atol=max(1e-8 / min(x_unit_to_m, y_unit_to_m), 1e-12),
        )


@pytest.mark.parametrize(
    ("projection", "implementation_id", "lat_0"),
    [
        ("sinu", SINU_FORWARD_FIXED_Q62, 0.0),
        ("ortho", ORTHO_FORWARD_FIXED_Q62, 45.0),
    ],
)
@pytest.mark.parametrize(
    "radius",
    [
        np.nextafter(PROJECTION_FIXED_Q62_MAX_SCALE_M, 0.0),
        PROJECTION_FIXED_Q62_MAX_SCALE_M,
    ],
)
def test_wave1_max_qualified_scale_preserves_coordinate_contract(
    projection,
    implementation_id,
    lat_0,
    radius,
):
    rng = np.random.default_rng(20260807)
    lat = cp.asarray(rng.uniform(-90.0, 90.0, 200_000), dtype=cp.float64)
    lon = cp.asarray(rng.uniform(-180.0, 180.0, lat.size), dtype=cp.float64)
    pipeline = _pipeline(
        projection,
        "forward",
        lat_0=lat_0,
        lon_0=17.0,
        radius=radius,
    )

    native = _run_exact(
        pipeline,
        lat,
        lon,
        projection,
        "forward",
        NATIVE_LIBDEVICE,
    )
    accelerated = _run_exact(
        pipeline,
        lat,
        lon,
        projection,
        "forward",
        implementation_id,
    )

    radial_error = cp.hypot(accelerated[0] - native[0], accelerated[1] - native[1])
    assert float(cp.max(radial_error).item()) <= 1e-8


@pytest.mark.parametrize(
    ("projection", "implementation_id", "lat_0"),
    [
        ("sinu", SINU_FORWARD_FIXED_Q62, 0.0),
        ("ortho", ORTHO_FORWARD_FIXED_Q62, 45.0),
    ],
)
@pytest.mark.parametrize(
    "physical_scale",
    [
        np.nextafter(PROJECTION_FIXED_Q62_MAX_SCALE_M, np.inf),
        1.0e12,
        0.0,
        -1.0,
        np.inf,
        np.nan,
    ],
)
def test_wave1_unqualified_scale_is_bitwise_native(
    projection,
    implementation_id,
    lat_0,
    physical_scale,
):
    lat = cp.asarray([-89.0, -45.0, -0.0, 0.0, 45.0, 89.0], dtype=cp.float64)
    lon = cp.asarray([-179.0, -91.0, -0.0, 0.0, 91.0, 179.0], dtype=cp.float64)
    pipeline = _pipeline(projection, "forward", lat_0=lat_0, lon_0=17.0)
    computed = dict(pipeline.computed)
    computed["a"] = physical_scale

    native = _run_exact(
        pipeline,
        lat,
        lon,
        projection,
        "forward",
        NATIVE_LIBDEVICE,
        computed=computed,
    )
    accelerated = _run_exact(
        pipeline,
        lat,
        lon,
        projection,
        "forward",
        implementation_id,
        computed=computed,
    )

    for expected, actual in zip(native, accelerated, strict=True):
        assert_array_equal(
            cp.asnumpy(actual).view(np.uint64),
            cp.asnumpy(expected).view(np.uint64),
        )


@pytest.mark.parametrize(
    ("projection", "implementation_id", "lat_0"),
    [
        ("sinu", SINU_FORWARD_FIXED_Q62, 0.0),
        ("ortho", ORTHO_FORWARD_FIXED_Q62, 45.0),
    ],
)
def test_wave1_valid_domain_boundaries_match_native_with_nanometer_error(
    projection,
    implementation_id,
    lat_0,
):
    lat_host = np.array(
        [
            -90.0,
            np.nextafter(-90.0, np.inf),
            -45.0,
            -0.0,
            0.0,
            45.0,
            np.nextafter(90.0, -np.inf),
            90.0,
        ],
        dtype=np.float64,
    )
    lon_host = np.array(
        [-180.0, -179.999999, -120.0, -0.0, 0.0, 100.0, 179.999999, 180.0],
        dtype=np.float64,
    )
    lat = cp.asarray(lat_host)
    lon = cp.asarray(lon_host)
    pipeline = _pipeline(projection, "forward", lat_0=lat_0)
    native = _run_exact(
        pipeline,
        lat,
        lon,
        projection,
        "forward",
        NATIVE_LIBDEVICE,
    )
    accelerated = _run_exact(
        pipeline,
        lat,
        lon,
        projection,
        "forward",
        implementation_id,
    )

    radial_error = cp.hypot(accelerated[0] - native[0], accelerated[1] - native[1])
    assert float(cp.max(radial_error).item()) <= 1e-8


@pytest.mark.parametrize(
    ("projection", "implementation_id", "lat_0"),
    [
        ("sinu", SINU_FORWARD_FIXED_Q62, 0.0),
        ("ortho", ORTHO_FORWARD_FIXED_Q62, 45.0),
    ],
)
def test_wave1_out_of_domain_and_nonfinite_inputs_are_bitwise_native(
    projection,
    implementation_id,
    lat_0,
):
    lat_host = np.array(
        [
            -np.inf,
            -360.0,
            -181.0,
            -100.0,
            np.nextafter(-90.0, -np.inf),
            np.nextafter(90.0, np.inf),
            100.0,
            181.0,
            360.0,
            np.inf,
            np.nan,
            40.0,
            40.0,
        ],
        dtype=np.float64,
    )
    lon_host = np.array(
        [0.0] * 11 + [np.inf, np.nan],
        dtype=np.float64,
    )
    lat = cp.asarray(lat_host)
    lon = cp.asarray(lon_host)
    pipeline = _pipeline(projection, "forward", lat_0=lat_0)

    native = _run_exact(
        pipeline,
        lat,
        lon,
        projection,
        "forward",
        NATIVE_LIBDEVICE,
    )
    accelerated = _run_exact(
        pipeline,
        lat,
        lon,
        projection,
        "forward",
        implementation_id,
    )

    for expected, actual in zip(native, accelerated, strict=True):
        assert_array_equal(
            cp.asnumpy(actual).view(np.uint64),
            cp.asnumpy(expected).view(np.uint64),
        )


@pytest.mark.parametrize(
    ("projection", "implementation_id", "lat_0"),
    [
        ("sinu", SINU_FORWARD_FIXED_Q62, 0.0),
        ("ortho", ORTHO_FORWARD_FIXED_Q62, 45.0),
    ],
)
def test_wave1_variants_honor_stream_and_preallocated_output_identity(
    projection,
    implementation_id,
    lat_0,
):
    lat = cp.asarray([-90.0, -45.0, 0.0, 45.0, 90.0], dtype=cp.float64)
    lon = cp.asarray([-180.0, -120.0, 0.0, 120.0, 180.0], dtype=cp.float64)
    out_x = cp.empty_like(lat)
    out_y = cp.empty_like(lon)
    stream = cp.cuda.Stream(non_blocking=True)
    pipeline = _pipeline(projection, "forward", lat_0=lat_0)

    result = _run_exact(
        pipeline,
        lat,
        lon,
        projection,
        "forward",
        implementation_id,
        out_x=out_x,
        out_y=out_y,
        stream=stream,
    )
    stream.synchronize()

    assert result[0] is out_x
    assert result[1] is out_y
    assert_allclose(cp.asnumpy(out_x), cp.asnumpy(result[0]), rtol=0.0, atol=0.0)
    assert_allclose(cp.asnumpy(out_y), cp.asnumpy(result[1]), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("projection", "implementation_id", "kernel_name", "lat_0"),
    [
        ("sinu", SINU_FORWARD_FIXED_Q62, "sinu_forward_fixed_q62", 0.0),
        ("ortho", ORTHO_FORWARD_FIXED_Q62, "ortho_forward_fixed_q62", 45.0),
    ],
)
def test_wave1_variants_capture_as_one_allocation_free_kernel(
    monkeypatch,
    projection,
    implementation_id,
    kernel_name,
    lat_0,
):
    lat = cp.linspace(-80.0, 80.0, 1024, dtype=cp.float64)
    lon = cp.linspace(-170.0, 170.0, 1024, dtype=cp.float64)
    out_x = cp.empty_like(lat)
    out_y = cp.empty_like(lon)
    pipeline = _pipeline(projection, "forward", lat_0=lat_0)
    stream = cp.cuda.Stream(non_blocking=True)

    _run_exact(
        pipeline,
        lat,
        lon,
        projection,
        "forward",
        implementation_id,
        out_x=out_x,
        out_y=out_y,
        stream=stream,
    )
    stream.synchronize()

    allocation_calls = []
    original_empty = cp.empty

    def counted_empty(*args, **kwargs):
        allocation_calls.append((args, kwargs))
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(cp, "empty", counted_empty)
    stream.begin_capture()
    _run_exact(
        pipeline,
        lat,
        lon,
        projection,
        "forward",
        implementation_id,
        out_x=out_x,
        out_y=out_y,
        stream=stream,
    )
    graph = stream.end_capture()
    dot = graph.debug_dot_str()

    assert allocation_calls == []
    assert kernel_name in dot
    assert dot.count('shape="octagon"') == 1
    assert "memcpy" not in dot.lower()
