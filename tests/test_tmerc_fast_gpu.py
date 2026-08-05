"""Accuracy and dispatch tests for guarded forward-TM CUDA transcendentals."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device not available", allow_module_level=True)
except Exception as exc:  # pragma: no cover - import-time environment guard
    pytest.skip(f"CUDA device not available: {exc}", allow_module_level=True)

from vibeproj import Transformer  # noqa: E402
from vibeproj.fused_kernels import (  # noqa: E402
    _get_kernel,
    _kernel_cache,
    fused_transform,
)
from vibeproj.transcendentals import (  # noqa: E402
    NATIVE_LIBDEVICE,
    TMERC_FIXED_Q62,
    TranscendentalOperation,
    detect_device_capability,
    resolve_transcendental_strategy,
)


def _run_forward(transformer, lat, lon, implementation_id):
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


def _max_radial_difference(native, fast) -> float:
    return float(cp.max(cp.hypot(fast[0] - native[0], fast[1] - native[1])).item())


@pytest.mark.parametrize(
    "epsg,lat_range,central_lon",
    [
        (32601, (0.0, 84.0), -177.0),
        (32631, (0.0, 84.0), 3.0),
        (32660, (0.0, 84.0), 177.0),
        (32756, (-80.0, 0.0), 153.0),
    ],
)
def test_fast_tmerc_matches_native_across_utm_zones(epsg, lat_range, central_lon):
    rng = np.random.default_rng(epsg)
    n = 50_000
    lat = cp.asarray(rng.uniform(*lat_range, n), dtype=cp.float64)
    lon = cp.asarray(rng.uniform(central_lon - 3.0, central_lon + 3.0, n), dtype=cp.float64)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=False)

    native = _run_forward(transformer, lat, lon, NATIVE_LIBDEVICE)
    fast = _run_forward(transformer, lat, lon, TMERC_FIXED_Q62)

    assert _max_radial_difference(native, fast) < 1e-8


def test_fast_tmerc_guard_boundaries_match_native():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=False)
    central_lon = 3.0
    boundary_degrees = np.rad2deg(0.06)
    offsets = np.array(
        [
            -boundary_degrees,
            np.nextafter(-boundary_degrees, np.inf),
            0.0,
            np.nextafter(boundary_degrees, -np.inf),
            boundary_degrees,
            np.nextafter(boundary_degrees, np.inf),
        ]
    )
    lat_host = np.repeat(np.array([-80.0, -45.0, 0.0, 45.0, 84.0]), offsets.size)
    lon_host = np.tile(central_lon + offsets, 5)
    lat = cp.asarray(lat_host, dtype=cp.float64)
    lon = cp.asarray(lon_host, dtype=cp.float64)

    native = _run_forward(transformer, lat, lon, NATIVE_LIBDEVICE)
    fast = _run_forward(transformer, lat, lon, TMERC_FIXED_Q62)

    assert _max_radial_difference(native, fast) < 1e-8


def test_fast_tmerc_wide_domain_uses_accurate_fallback():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=False)
    rng = np.random.default_rng(8128)
    n = 100_000
    lat = cp.asarray(rng.uniform(-80.0, 80.0, n), dtype=cp.float64)
    lon = cp.asarray(rng.uniform(-57.0, 63.0, n), dtype=cp.float64)

    native = _run_forward(transformer, lat, lon, NATIVE_LIBDEVICE)
    fast = _run_forward(transformer, lat, lon, TMERC_FIXED_Q62)

    assert _max_radial_difference(native, fast) < 1e-8


def test_tmerc_registry_implementation_uses_exact_strategy_cache():
    assert _get_kernel(
        "tmerc", "forward", "float64", transcendental_impl=TMERC_FIXED_Q62
    ) is _get_kernel("tmerc", "forward", "float64", transcendental_impl=TMERC_FIXED_Q62)


def test_transformer_compile_warms_selected_tmerc_strategy():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    expected_impl = resolve_transcendental_strategy(
        TranscendentalOperation.TMERC_FORWARD,
        "auto",
        device=detect_device_capability(cp),
        domain="utm",
        precision="auto",
    ).implementation_id
    _kernel_cache.clear()
    transformer.compile()

    assert ("tmerc", "forward", "float64", expected_impl) in _kernel_cache
    assert ("tmerc", "inverse", "float64", NATIVE_LIBDEVICE) in _kernel_cache


def test_non_utm_tmerc_auto_and_compile_remain_native():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=False)
    assert transformer._pipeline.computed["is_utm"] is False
    lat = cp.asarray([-47.0, -42.0, -35.0], dtype=cp.float64)
    lon = cp.asarray([167.0, 173.0, 178.0], dtype=cp.float64)

    _kernel_cache.clear()
    transformer.compile()
    transformer.transform_buffers(lat, lon, precision="fp64")

    assert ("tmerc", "forward", "float64", NATIVE_LIBDEVICE) in _kernel_cache
    assert ("tmerc", "forward", "float64", TMERC_FIXED_Q62) not in _kernel_cache
