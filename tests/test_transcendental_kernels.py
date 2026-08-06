"""Focused compilation, dispatch, and accuracy tests for CUDA math variants."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device not available", allow_module_level=True)
except Exception as exc:  # pragma: no cover - import-time environment guard
    pytest.skip(f"CUDA device not available: {exc}", allow_module_level=True)

from vibeproj import Transformer  # noqa: E402
from vibeproj._transcendental_device_fns import (  # noqa: E402
    TM_UTM_QUALIFIED_DEVICE_FNS,
)
import vibeproj.fused_kernels as fused_module  # noqa: E402
from vibeproj.fused_kernels import (  # noqa: E402
    _SUPPORTED,
    _get_helmert_kernel,
    _get_kernel,
    _get_svd_device_arrays,
    _get_svd_kernel,
    _helmert_kernel_cache,
    _kernel_cache,
    _svd_device_cache,
    _svd_device_cache_key,
    compile_helmert_kernel,
    compile_kernels,
    compile_svd_kernel,
    fused_svd_correction,
    fused_transform,
)
from vibeproj.transcendentals import (  # noqa: E402
    GNOM_INVERSE_GUARDED_RSQRT_REFRAME,
    HELMERT_FIXED_Q62,
    NATIVE_LIBDEVICE,
    TMERC_FIXED_Q62,
)
from vibeproj.exceptions import CoordinateValidationError  # noqa: E402


_PAIRED_NATIVE_CASES = (
    ("tmerc", "forward"),
    ("tmerc", "inverse"),
    ("lcc", "forward"),
    ("stere", "forward"),
    ("aea", "forward"),
    ("aea", "inverse"),
    ("laea", "forward"),
    ("laea", "inverse"),
    ("cea", "inverse"),
    ("ortho", "forward"),
    ("ortho", "inverse"),
    ("gnom", "forward"),
    ("gnom", "inverse"),
    ("moll", "forward"),
    ("omerc", "forward"),
    ("omerc", "inverse"),
    ("krovak", "forward"),
    ("krovak", "inverse"),
    ("eck4", "forward"),
    ("eck4", "inverse"),
    ("eck6", "forward"),
    ("eck6", "inverse"),
    ("sterea", "forward"),
    ("sterea", "inverse"),
    ("geos", "forward"),
    ("geos", "inverse"),
    ("wintri", "forward"),
    ("wintri", "inverse"),
    ("aeqd", "forward"),
    ("aeqd", "inverse"),
)


@pytest.mark.parametrize("compute_dtype", ["float64", "float32"])
@pytest.mark.parametrize(("projection", "direction"), _PAIRED_NATIVE_CASES)
def test_native_paired_sincos_kernels_compile(projection, direction, compute_dtype):
    kernel = _get_kernel(
        projection,
        direction,
        compute_dtype,
        transcendental_impl=NATIVE_LIBDEVICE,
    )
    kernel.compile()


def test_all_103_production_rawkernel_variants_compile():
    compiled = []
    for compute_dtype in ("float64", "float32"):
        for projection, direction in sorted(_SUPPORTED):
            kernel = _get_kernel(
                projection,
                direction,
                compute_dtype,
                transcendental_impl=NATIVE_LIBDEVICE,
            )
            kernel.compile()
            compiled.append((projection, direction, compute_dtype, NATIVE_LIBDEVICE))

    for direction in ("forward", "inverse"):
        _get_kernel("tmerc", direction, "ds", transcendental_impl=NATIVE_LIBDEVICE).compile()
        compiled.append(("tmerc", direction, "ds", NATIVE_LIBDEVICE))

    _get_kernel("tmerc", "forward", "float64", transcendental_impl=TMERC_FIXED_Q62).compile()
    compiled.append(("tmerc", "forward", "float64", TMERC_FIXED_Q62))

    _get_kernel(
        "gnom",
        "inverse",
        "float64",
        transcendental_impl=GNOM_INVERSE_GUARDED_RSQRT_REFRAME,
    ).compile()
    compiled.append(("gnom", "inverse", "float64", GNOM_INVERSE_GUARDED_RSQRT_REFRAME))

    for implementation_id in (NATIVE_LIBDEVICE, HELMERT_FIXED_Q62):
        _get_helmert_kernel(implementation_id).compile()
        compiled.append(("helmert", "forward", "float64", implementation_id))

    _get_svd_kernel().compile()
    compiled.append(("svd", "correction", "float64", NATIVE_LIBDEVICE))
    assert len(compiled) == 103


def test_tmerc_implementation_id_separates_rawkernel_cache():
    _kernel_cache.clear()
    native = _get_kernel("tmerc", "forward", "float64", transcendental_impl=NATIVE_LIBDEVICE)
    accelerated = _get_kernel("tmerc", "forward", "float64", transcendental_impl=TMERC_FIXED_Q62)

    assert native is not accelerated
    assert ("tmerc", "forward", "float64", NATIVE_LIBDEVICE) in _kernel_cache
    assert ("tmerc", "forward", "float64", TMERC_FIXED_Q62) in _kernel_cache


def test_helmert_implementation_id_separates_rawkernel_cache():
    _helmert_kernel_cache.clear()
    native = _get_helmert_kernel(NATIVE_LIBDEVICE)
    accelerated = _get_helmert_kernel(HELMERT_FIXED_Q62)

    assert native is not accelerated
    assert set(_helmert_kernel_cache) == {NATIVE_LIBDEVICE, HELMERT_FIXED_Q62}


@pytest.mark.parametrize("implementation_id", [NATIVE_LIBDEVICE, TMERC_FIXED_Q62])
def test_tmerc_variant_cache_is_thread_safe(implementation_id):
    _kernel_cache.clear()

    def load_kernel(_):
        return _get_kernel(
            "tmerc",
            "forward",
            "float64",
            transcendental_impl=implementation_id,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        kernels = list(executor.map(load_kernel, range(32)))

    assert all(kernel is kernels[0] for kernel in kernels)


@pytest.mark.parametrize("implementation_id", [NATIVE_LIBDEVICE, HELMERT_FIXED_Q62])
def test_helmert_variant_cache_is_thread_safe(implementation_id):
    _helmert_kernel_cache.clear()

    with ThreadPoolExecutor(max_workers=8) as executor:
        kernels = list(executor.map(_get_helmert_kernel, [implementation_id] * 32))

    assert all(kernel is kernels[0] for kernel in kernels)


@pytest.mark.parametrize(
    ("projection", "direction", "compute_dtype"),
    [
        ("tmerc", "inverse", "float64"),
        ("tmerc", "forward", "float32"),
        ("tmerc", "forward", "ds"),
        ("merc", "forward", "float64"),
    ],
)
def test_tmerc_accelerated_id_rejects_unsupported_kernel_surfaces(
    projection, direction, compute_dtype
):
    with pytest.raises(ValueError, match="qualified only"):
        _get_kernel(
            projection,
            direction,
            compute_dtype,
            transcendental_impl=TMERC_FIXED_Q62,
        )


_TM_MATH_TEST_SOURCE = (
    TM_UTM_QUALIFIED_DEVICE_FNS
    + r"""
extern "C" __global__ void tm_qualified_math_test(
    const double* __restrict__ gaussian_lat,
    const double* __restrict__ longitude_offset,
    const double* __restrict__ asinh_input,
    double* __restrict__ latitude_out,
    double* __restrict__ asinh_out,
    double* __restrict__ sin_out,
    double* __restrict__ cos_out,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const double lat = gaussian_lat[idx];
    const double lon = longitude_offset[idx];
    double sin_lat, cos_lat, sin_lon, cos_lon;
    vp_tm_utm_sincos(lat, lon, &sin_lat, &cos_lat);
    vp_tm_utm_sincos(lon, lon, &sin_lon, &cos_lon);
    latitude_out[idx] = vp_tm_utm_latitude(
        lat, sin_lat, cos_lat, cos_lon, cos_lat * cos_lon, lon
    );
    asinh_out[idx] = vp_tm_utm_asinh(asinh_input[idx]);
    sin_out[idx] = sin_lon;
    cos_out[idx] = cos_lon;
}
"""
)


def test_qualified_tm_math_matches_native_at_domain_edges_and_random_grid():
    rng = np.random.default_rng(20260805)
    n = 100_000
    lat = rng.uniform(-1.47, 1.47, n)
    lon = rng.uniform(-0.06, 0.06, n)
    values = rng.uniform(-0.06, 0.06, n)
    edges = np.array([-0.06, np.nextafter(-0.06, 0.0), 0.0, 0.06])
    lat[: edges.size] = np.array([-1.47, -0.5, 0.0, 1.47])
    lon[: edges.size] = edges
    values[: edges.size] = edges

    inputs = tuple(cp.asarray(item) for item in (lat, lon, values))
    outputs = tuple(cp.empty(n, dtype=cp.float64) for _ in range(4))
    kernel = cp.RawKernel(_TM_MATH_TEST_SOURCE, "tm_qualified_math_test")
    kernel(((n + 255) // 256,), (256,), (*inputs, *outputs, np.int32(n)))
    actual_lat, actual_asinh, actual_sin, actual_cos = (cp.asnumpy(output) for output in outputs)

    expected_lat = np.arctan2(np.sin(lat), np.cos(lat) * np.cos(lon))
    assert_allclose(actual_lat, expected_lat, rtol=0.0, atol=5e-16)
    assert_allclose(actual_asinh, np.arcsinh(values), rtol=0.0, atol=3e-17)
    assert_allclose(actual_sin, np.sin(lon), rtol=0.0, atol=7e-16)
    assert_allclose(actual_cos, np.cos(lon), rtol=0.0, atol=7e-16)


def test_qualified_tm_latitude_guard_preserves_native_principal_branch():
    half_pi = np.pi / 2.0
    lat = np.array(
        [
            -np.inf,
            -2.0 * np.pi,
            -np.pi,
            np.nextafter(-half_pi, -np.inf),
            -half_pi,
            np.nextafter(-half_pi, np.inf),
            0.0,
            np.nextafter(half_pi, -np.inf),
            half_pi,
            np.nextafter(half_pi, np.inf),
            np.pi,
            2.0 * np.pi,
            np.inf,
            np.nan,
        ],
        dtype=np.float64,
    )
    lon = np.zeros_like(lat)
    lon[-3:] = [np.nextafter(0.06, np.inf), np.inf, np.nan]
    values = np.zeros_like(lat)

    inputs = tuple(cp.asarray(item) for item in (lat, lon, values))
    outputs = tuple(cp.empty(lat.size, dtype=cp.float64) for _ in range(4))
    kernel = cp.RawKernel(_TM_MATH_TEST_SOURCE, "tm_qualified_math_test")
    kernel((1,), (256,), (*inputs, *outputs, np.int32(lat.size)))
    actual_lat = cp.asnumpy(outputs[0])

    with np.errstate(invalid="ignore"):
        expected_lat = np.arctan2(np.sin(lat), np.cos(lat) * np.cos(lon))
    assert_allclose(actual_lat, expected_lat, rtol=0.0, atol=1e-15, equal_nan=True)


def _run_exact_tmerc(transformer, lat, lon, implementation_id):
    pipeline = transformer._pipeline
    return fused_transform(
        lat,
        lon,
        projection_name="tmerc",
        direction="forward",
        computed=pipeline.computed,
        src_north_first=pipeline.src_north_first,
        dst_north_first=pipeline.dst_north_first,
        xp=cp,
        precision="fp64",
        transcendental_impl=implementation_id,
    )


def test_public_and_exact_tmerc_variants_match_native_for_extreme_latitudes():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=False)
    lat_host = np.array(
        [
            -np.inf,
            -360.0,
            -181.0,
            np.nextafter(-90.0, -np.inf),
            -90.0,
            np.nextafter(-90.0, np.inf),
            0.0,
            np.nextafter(90.0, -np.inf),
            90.0,
            np.nextafter(90.0, np.inf),
            181.0,
            360.0,
            np.inf,
            np.nan,
        ],
        dtype=np.float64,
    )
    lon_host = np.full_like(lat_host, 3.0)
    lat = cp.asarray(lat_host)
    lon = cp.asarray(lon_host)

    exact_native = _run_exact_tmerc(transformer, lat, lon, NATIVE_LIBDEVICE)
    exact_accelerated = _run_exact_tmerc(transformer, lat, lon, TMERC_FIXED_Q62)
    public_native = transformer.transform_buffers(lat, lon, transcendentals="native")
    public_accelerated = transformer.transform_buffers(lat, lon, transcendentals="accelerated")

    for reference, candidate in (
        (exact_native, exact_accelerated),
        (public_native, public_accelerated),
    ):
        for expected, actual in zip(reference, candidate, strict=True):
            assert_allclose(
                cp.asnumpy(actual),
                cp.asnumpy(expected),
                rtol=0.0,
                atol=1e-8,
                equal_nan=True,
            )


class _CompileSpy:
    def __init__(self):
        self.calls = 0

    def compile(self):
        self.calls += 1


def test_compile_kernels_eagerly_invokes_rawkernel_compile(monkeypatch):
    spy = _CompileSpy()
    monkeypatch.setattr(fused_module, "_get_kernel", lambda *args, **kwargs: spy)

    compile_kernels(["eqc"], precision="fp64")

    assert spy.calls == 2  # forward and inverse


def test_compile_helmert_kernel_eagerly_invokes_rawkernel_compile(monkeypatch):
    spy = _CompileSpy()
    monkeypatch.setattr(fused_module, "_get_helmert_kernel", lambda *args, **kwargs: spy)

    compile_helmert_kernel(transcendental_impl=NATIVE_LIBDEVICE)

    assert spy.calls == 1


def test_compile_svd_kernel_eagerly_invokes_rawkernel_compile(monkeypatch):
    spy = _CompileSpy()
    monkeypatch.setattr(fused_module, "_get_svd_kernel", lambda: spy)

    compile_svd_kernel()

    assert spy.calls == 1


def test_svd_coefficient_cache_key_includes_cuda_device():
    correction = object()

    assert _svd_device_cache_key(correction, 0) != _svd_device_cache_key(correction, 1)


@pytest.mark.parametrize("legacy_stream_name", ["null", "ptds"])
def test_fused_svd_legacy_stream_uses_input_device_for_first_use(legacy_stream_name):
    from vibeproj._datum_corrections import apply_svd_correction, get_datum_correction

    correction = get_datum_correction("EPSG:4267", "EPSG:4269")
    assert correction is not None
    _svd_device_cache.clear()
    lat_host = np.array([35.0, 40.0, 45.0], dtype=np.float64)
    lon_host = np.array([-100.0, -90.0, -80.0], dtype=np.float64)
    lat = cp.asarray(lat_host)
    lon = cp.asarray(lon_host)
    device_id = int(lat.device.id)
    stream = getattr(cp.cuda.Stream, legacy_stream_name)

    actual_lat, actual_lon = fused_svd_correction(
        lat,
        lon,
        correction,
        cp,
        stream=stream,
    )
    stream.synchronize()

    expected_lat, expected_lon = apply_svd_correction(lat_host, lon_host, correction, np)
    assert_allclose(cp.asnumpy(actual_lat), expected_lat, rtol=0.0, atol=1e-12)
    assert_allclose(cp.asnumpy(actual_lon), expected_lon, rtol=0.0, atol=1e-12)
    cache_entry = _svd_device_cache[(device_id, id(correction))]
    assert all(int(array.device.id) == device_id for array in cache_entry[1])


@pytest.mark.parametrize("stream_kind", ["null", "nonblocking"])
def test_public_svd_transform_buffers_supports_explicit_cuda_stream(stream_kind):
    transformer = Transformer.from_crs("EPSG:4267", "EPSG:4269")
    lon_host = np.array([-100.0, -90.0, -80.0], dtype=np.float64)
    lat_host = np.array([35.0, 40.0, 45.0], dtype=np.float64)
    expected_lon, expected_lat = transformer.transform_buffers(lon_host, lat_host)
    lon = cp.asarray(lon_host)
    lat = cp.asarray(lat_host)
    out_lon = cp.empty_like(lon)
    out_lat = cp.empty_like(lat)
    stream = cp.cuda.Stream.null if stream_kind == "null" else cp.cuda.Stream(non_blocking=True)
    _svd_device_cache.clear()

    actual_lon, actual_lat = transformer.transform_buffers(
        lon,
        lat,
        out_x=out_lon,
        out_y=out_lat,
        stream=stream,
    )
    stream.synchronize()

    assert actual_lon is out_lon
    assert actual_lat is out_lat
    assert_allclose(cp.asnumpy(actual_lon), expected_lon, rtol=0.0, atol=1e-12)
    assert_allclose(cp.asnumpy(actual_lat), expected_lat, rtol=0.0, atol=1e-12)


def test_fused_svd_rejects_cross_device_lon_via_mock():
    from vibeproj._datum_corrections import get_datum_correction

    correction = get_datum_correction("EPSG:4267", "EPSG:4269")
    assert correction is not None
    lat = cp.asarray([40.0], dtype=cp.float64)
    wrong_device = int(lat.device.id) + 1
    lon = SimpleNamespace(size=lat.size, device=SimpleNamespace(id=wrong_device))

    with pytest.raises(CoordinateValidationError, match="lon is on CUDA device"):
        fused_svd_correction(lat, lon, correction, cp)


@pytest.mark.parametrize("output_name", ["out_lat", "out_lon"])
def test_fused_svd_rejects_cross_device_output_via_mock(output_name):
    from vibeproj._datum_corrections import get_datum_correction

    correction = get_datum_correction("EPSG:4267", "EPSG:4269")
    assert correction is not None
    lat = cp.asarray([40.0], dtype=cp.float64)
    lon = cp.asarray([-90.0], dtype=cp.float64)
    wrong_device = int(lat.device.id) + 1
    output = SimpleNamespace(device=SimpleNamespace(id=wrong_device))

    with pytest.raises(CoordinateValidationError, match=f"{output_name} is on CUDA device"):
        fused_svd_correction(lat, lon, correction, cp, **{output_name: output})


def test_fused_svd_rejects_cross_device_non_null_stream_via_mock():
    from vibeproj._datum_corrections import get_datum_correction

    correction = get_datum_correction("EPSG:4267", "EPSG:4269")
    assert correction is not None
    lat = cp.asarray([40.0], dtype=cp.float64)
    lon = cp.asarray([-90.0], dtype=cp.float64)
    stream = SimpleNamespace(device_id=int(lat.device.id) + 1)

    with pytest.raises(CoordinateValidationError, match="SVD stream is on CUDA device"):
        fused_svd_correction(lat, lon, correction, cp, stream=stream)


def test_svd_coefficient_cache_rejects_cross_device_arrays_via_mock():
    from vibeproj._datum_corrections import get_datum_correction

    correction = get_datum_correction("EPSG:4267", "EPSG:4269")
    assert correction is not None
    device_id = int(cp.cuda.runtime.getDevice())
    wrong_device = device_id + 1
    fake_array = SimpleNamespace(device=SimpleNamespace(id=wrong_device))
    key = _svd_device_cache_key(correction, device_id)
    _svd_device_cache.clear()
    _svd_device_cache[key] = (correction, (fake_array,) * 6, None)

    try:
        with pytest.raises(ValueError, match="SVD coefficient u_lat is on CUDA device"):
            _get_svd_device_arrays(correction, device_id=device_id)
    finally:
        _svd_device_cache.clear()


def test_svd_coefficients_materialize_on_supplied_nonblocking_stream(monkeypatch):
    from vibeproj._datum_corrections import get_datum_correction

    correction = get_datum_correction("EPSG:4267", "EPSG:4269")
    assert correction is not None
    _svd_device_cache.clear()
    stream = cp.cuda.Stream(non_blocking=True)
    observed_streams = []
    original_asarray = cp.asarray

    def traced_asarray(*args, **kwargs):
        observed_streams.append(cp.cuda.get_current_stream().ptr)
        return original_asarray(*args, **kwargs)

    monkeypatch.setattr(cp, "asarray", traced_asarray)
    arrays = _get_svd_device_arrays(
        correction,
        device_id=int(stream.device_id),
        stream=stream,
    )
    stream.synchronize()

    assert len(arrays) == 6
    assert observed_streams == [stream.ptr] * 6
    assert (int(stream.device_id), id(correction)) in _svd_device_cache


def test_fused_svd_first_use_is_ordered_on_nonblocking_stream():
    from vibeproj._datum_corrections import apply_svd_correction, get_datum_correction

    correction = get_datum_correction("EPSG:4267", "EPSG:4269")
    assert correction is not None
    _svd_device_cache.clear()
    lat_host = np.array([35.0, 40.0, 45.0], dtype=np.float64)
    lon_host = np.array([-100.0, -90.0, -80.0], dtype=np.float64)
    lat = cp.asarray(lat_host)
    lon = cp.asarray(lon_host)
    out_lat = cp.empty_like(lat)
    out_lon = cp.empty_like(lon)
    stream = cp.cuda.Stream(non_blocking=True)

    result = fused_svd_correction(
        lat,
        lon,
        correction,
        cp,
        out_lat=out_lat,
        out_lon=out_lon,
        stream=stream,
    )
    complete = cp.cuda.Event()
    complete.record(stream)
    complete.synchronize()

    expected_lat, expected_lon = apply_svd_correction(lat_host, lon_host, correction, np)
    assert result[0] is out_lat
    assert result[1] is out_lon
    assert_allclose(cp.asnumpy(out_lat), expected_lat, rtol=0.0, atol=1e-12)
    assert_allclose(cp.asnumpy(out_lon), expected_lon, rtol=0.0, atol=1e-12)


def test_svd_cached_coefficients_add_cross_stream_event_dependency():
    from vibeproj._datum_corrections import get_datum_correction

    correction = get_datum_correction("EPSG:4267", "EPSG:4269")
    assert correction is not None
    _svd_device_cache.clear()
    producer = cp.cuda.Stream(non_blocking=True)
    consumer = cp.cuda.Stream(non_blocking=True)

    produced = _get_svd_device_arrays(correction, stream=producer)
    consumed = _get_svd_device_arrays(correction, stream=consumer)
    with consumer:
        total = cp.sum(consumed[0])
    consumer.synchronize()

    assert all(lhs is rhs for lhs, rhs in zip(produced, consumed, strict=True))
    assert_allclose(
        float(total.item()),
        np.asarray(correction.u_lat, dtype=np.float64).sum(),
        rtol=0.0,
        atol=1e-12,
    )


def test_warmed_svd_coefficients_allow_steady_state_cuda_graph_capture():
    from vibeproj._datum_corrections import get_datum_correction

    correction = get_datum_correction("EPSG:4267", "EPSG:4269")
    assert correction is not None
    _svd_device_cache.clear()
    lat = cp.full(1024, 40.0, dtype=cp.float64)
    lon = cp.full(1024, -90.0, dtype=cp.float64)
    out_lat = cp.empty_like(lat)
    out_lon = cp.empty_like(lon)

    fused_svd_correction(
        lat,
        lon,
        correction,
        cp,
        out_lat=out_lat,
        out_lon=out_lon,
    )
    cp.cuda.get_current_stream().synchronize()

    stream = cp.cuda.Stream(non_blocking=True)
    stream.begin_capture()
    with stream:
        fused_svd_correction(
            lat,
            lon,
            correction,
            cp,
            out_lat=out_lat,
            out_lon=out_lon,
            stream=stream,
        )
    graph = stream.end_capture()

    assert "svd_correction" in graph.debug_dot_str()
