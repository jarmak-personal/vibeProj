"""Accuracy tests for bounded Q1.62 CUDA sine/cosine."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device not available", allow_module_level=True)
except Exception as exc:  # pragma: no cover - import-time environment guard
    pytest.skip(f"CUDA device not available: {exc}", allow_module_level=True)

from vibeproj._fixed_trig_device_fns import FIXED_TRIG_DEVICE_FNS  # noqa: E402
from vibeproj import Transformer  # noqa: E402
from vibeproj.fused_kernels import (  # noqa: E402
    _get_helmert_kernel,
    _helmert_kernel_cache,
    fused_helmert_shift,
)
from vibeproj.transcendentals import (  # noqa: E402
    HELMERT_FIXED_Q62,
    NATIVE_LIBDEVICE,
    TranscendentalOperation,
    detect_device_capability,
    resolve_transcendental_strategy,
)


_TEST_SOURCE = (
    FIXED_TRIG_DEVICE_FNS
    + r"""
extern "C" __global__ void fixed_trig_test(
    const double* __restrict__ angles,
    double* __restrict__ out_sin,
    double* __restrict__ out_cos,
    double* __restrict__ out_sin_only,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    vp_fixed_sincos_bounded(angles[idx], &out_sin[idx], &out_cos[idx]);
    out_sin_only[idx] = vp_fixed_sin_bounded(angles[idx]);
}
"""
)


def _run_fixed_trig(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel = cp.RawKernel(_TEST_SOURCE, "fixed_trig_test")
    angles_gpu = cp.asarray(angles, dtype=cp.float64)
    outputs = tuple(cp.empty(angles.size, dtype=cp.float64) for _ in range(3))
    block = 256
    grid = max(1, (angles.size + block - 1) // block)
    kernel((grid,), (block,), (angles_gpu, *outputs, np.int32(angles.size)))
    return tuple(cp.asnumpy(output) for output in outputs)


def test_fixed_trig_dense_bounded_domain_matches_fp64():
    rng = np.random.default_rng(42)
    random_angles = rng.uniform(-np.pi, np.pi, 100_000)
    boundaries = np.array(
        [
            -np.pi,
            -np.pi / 2,
            -np.pi / 4,
            0.0,
            np.pi / 4,
            np.pi / 2,
            np.pi,
        ],
        dtype=np.float64,
    )
    adjacent = np.concatenate(
        (np.nextafter(boundaries, -np.inf), boundaries, np.nextafter(boundaries, np.inf))
    )
    angles = np.concatenate((random_angles, adjacent))

    actual_sin, actual_cos, actual_sin_only = _run_fixed_trig(angles)
    expected_sin = np.sin(angles)
    expected_cos = np.cos(angles)

    assert_allclose(actual_sin, expected_sin, rtol=0.0, atol=7e-16)
    assert_allclose(actual_cos, expected_cos, rtol=0.0, atol=7e-16)
    assert_allclose(actual_sin_only, actual_sin, rtol=0.0, atol=0.0)
    assert np.max(np.abs(actual_sin * actual_sin + actual_cos * actual_cos - 1.0)) < 1e-15


def test_fixed_trig_outside_domain_uses_native_fallback():
    angles = np.array(
        [
            -1000.0,
            -4 * np.pi,
            np.nextafter(-np.pi, -np.inf),
            np.nextafter(np.pi, np.inf),
            4 * np.pi,
            1000.0,
            np.nan,
            -np.inf,
            np.inf,
        ],
        dtype=np.float64,
    )
    actual_sin, actual_cos, actual_sin_only = _run_fixed_trig(angles)

    with np.errstate(invalid="ignore"):
        expected_sin = np.sin(angles)
        expected_cos = np.cos(angles)
    assert_allclose(actual_sin, expected_sin, rtol=0.0, atol=1e-15, equal_nan=True)
    assert_allclose(actual_cos, expected_cos, rtol=0.0, atol=1e-15, equal_nan=True)
    assert_allclose(actual_sin_only, expected_sin, rtol=0.0, atol=1e-15, equal_nan=True)


@pytest.mark.parametrize("with_height", [False, True])
def test_fixed_trig_helmert_matches_fp64_globally(with_height):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4277")
    params = transformer._helmert
    assert params is not None

    rng = np.random.default_rng(123)
    n = 100_000
    lat = cp.asarray(rng.uniform(-89.5, 89.5, n), dtype=cp.float64)
    lon = cp.asarray(rng.uniform(-180.0, 180.0, n), dtype=cp.float64)
    height = cp.asarray(rng.uniform(-500.0, 10_000.0, n), dtype=cp.float64) if with_height else None

    fp64 = fused_helmert_shift(lat, lon, params, cp, h=height, transcendental_impl=NATIVE_LIBDEVICE)
    fixed = fused_helmert_shift(
        lat, lon, params, cp, h=height, transcendental_impl=HELMERT_FIXED_Q62
    )
    cp.cuda.get_current_stream().synchronize()

    radius = params.dst_ellipsoid.a
    dlat_m = (fixed[0] - fp64[0]) * (np.pi / 180.0) * radius
    dlon_m = (fixed[1] - fp64[1]) * (np.pi / 180.0) * radius * cp.cos(fp64[0] * (np.pi / 180.0))
    horizontal_error = cp.hypot(dlat_m, dlon_m)
    assert float(cp.max(horizontal_error).item()) < 1e-8
    if with_height:
        # Height recovery subtracts two Earth-radius-scale quantities, so its
        # conditioning amplifies nanometre ECEF differences into ~0.1 um.
        assert float(cp.max(cp.abs(fixed[2] - fp64[2])).item()) < 2e-7


def test_fixed_trig_helmert_falls_back_for_unbounded_longitude():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4277")
    params = transformer._helmert
    assert params is not None
    lat = cp.asarray([0.0, 45.0, -45.0, 80.0], dtype=cp.float64)
    lon = cp.asarray([360.0, -540.0, 720.0, -1000.0], dtype=cp.float64)

    fp64 = fused_helmert_shift(lat, lon, params, cp, transcendental_impl=NATIVE_LIBDEVICE)
    fixed = fused_helmert_shift(lat, lon, params, cp, transcendental_impl=HELMERT_FIXED_Q62)
    assert_allclose(cp.asnumpy(fixed[0]), cp.asnumpy(fp64[0]), rtol=0.0, atol=1e-13)
    assert_allclose(cp.asnumpy(fixed[1]), cp.asnumpy(fp64[1]), rtol=0.0, atol=1e-13)


def test_helmert_registry_implementation_uses_exact_strategy_cache():
    assert _get_helmert_kernel(HELMERT_FIXED_Q62) is _get_helmert_kernel(HELMERT_FIXED_Q62)


def test_fused_helmert_honors_all_preallocated_output_identities():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4277")
    params = transformer._helmert
    assert params is not None
    lat = cp.asarray([0.0, 45.0], dtype=cp.float64)
    lon = cp.asarray([0.0, -3.0], dtype=cp.float64)
    height = cp.asarray([10.0, 100.0], dtype=cp.float64)
    outputs = tuple(cp.empty_like(lat) for _ in range(3))

    result = fused_helmert_shift(
        lat,
        lon,
        params,
        cp,
        h=height,
        out_lat=outputs[0],
        out_lon=outputs[1],
        out_h=outputs[2],
        transcendental_impl=NATIVE_LIBDEVICE,
    )

    assert all(actual is expected for actual, expected in zip(result, outputs, strict=True))


def test_transformer_compile_warms_auto_selected_helmert_kernel():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    assert transformer._helmert is not None
    expected_impl = resolve_transcendental_strategy(
        TranscendentalOperation.HELMERT,
        "auto",
        device=detect_device_capability(cp),
    ).implementation_id

    _helmert_kernel_cache.clear()
    transformer.compile()

    assert set(_helmert_kernel_cache) == {expected_impl}
