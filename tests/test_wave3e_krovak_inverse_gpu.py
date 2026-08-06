"""GPU production gates for the guarded Krovak inverse log-ratio series."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import math

import numpy as np
import pytest
from pyproj import CRS

from vibeproj import Transformer
import vibeproj.fused_kernels as fused_module
import vibeproj.transcendentals as transcendental_module
from vibeproj.transcendentals import (
    KROVAK_INVERSE_GUARDED_LOG_RATIO,
    NATIVE_LIBDEVICE,
)


cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device not available", allow_module_level=True)
except Exception as exc:  # pragma: no cover - runtime-specific
    pytest.skip(f"CUDA device not available: {exc}", allow_module_level=True)


KROVAK_CODES = (2065, 5513, 8352, 5221, 5514, 8353)
NATIVE_FP64_SOURCE_SHA256 = "c218df1ed25a75219566f6ea06183fd5619f08c348fc3e76a63baf6be945e582"


def _transformer(epsg: int = 5514, *, always_xy: bool = True) -> Transformer:
    target = CRS.from_epsg(epsg)
    return Transformer.from_crs(target.geodetic_crs, target, always_xy=always_xy)


def _run_exact(
    transformer: Transformer,
    first,
    second,
    implementation: str,
    *,
    computed: dict | None = None,
    out_x=None,
    out_y=None,
    stream=None,
):
    pipeline = transformer._pipeline_for_direction("INVERSE")
    return fused_module.fused_transform(
        first,
        second,
        projection_name="krovak",
        direction="inverse",
        computed=pipeline.computed if computed is None else computed,
        src_north_first=pipeline.src_north_first,
        dst_north_first=pipeline.dst_north_first,
        xp=cp,
        out_x=out_x,
        out_y=out_y,
        precision="fp64",
        stream=stream,
        transcendental_impl=implementation,
    )


def _bitwise_equal(actual, expected) -> bool:
    return all(
        bool(cp.all(got.view(cp.uint64) == want.view(cp.uint64)).get())
        for got, want in zip(actual, expected, strict=True)
    )


def _geographic_error_m(actual, expected, scale: float, *, always_xy: bool) -> float:
    actual_lon, actual_lat = actual if always_xy else actual[::-1]
    expected_lon, expected_lat = expected if always_xy else expected[::-1]
    delta_lon = cp.mod(actual_lon - expected_lon + 180.0, 360.0) - 180.0
    east = cp.deg2rad(delta_lon) * scale * cp.cos(cp.deg2rad(expected_lat))
    north = cp.deg2rad(actual_lat - expected_lat) * scale
    finite = cp.isfinite(east) & cp.isfinite(north)
    return float(cp.max(cp.hypot(east[finite], north[finite])).get())


@pytest.mark.parametrize("epsg", KROVAK_CODES)
@pytest.mark.parametrize("always_xy", [False, True])
def test_all_standard_crss_and_axis_modes_stay_below_10nm(epsg, always_xy):
    target = CRS.from_epsg(epsg)
    transformer = _transformer(epsg, always_xy=always_xy)
    expected_semantics = (
        "standard_bessel.regular"
        if epsg in (2065, 5513, 8352)
        else "standard_bessel.north_oriented"
    )
    assert transformer._pipeline.computed["_strategy_krovak_semantics"] == expected_semantics
    longitude = cp.linspace(
        12.1 - target.geodetic_crs.prime_meridian.longitude,
        18.9 - target.geodetic_crs.prime_meridian.longitude,
        4096,
        dtype=cp.float64,
    )
    latitude = cp.linspace(48.2, 51.2, longitude.size, dtype=cp.float64)
    geographic = (longitude, latitude) if always_xy else (latitude, longitude)
    projected = transformer.transform_buffers(
        *geographic, precision="fp64", transcendentals="native"
    )
    native = _run_exact(transformer, *projected, NATIVE_LIBDEVICE)
    candidate = _run_exact(transformer, *projected, KROVAK_INVERSE_GUARDED_LOG_RATIO)
    assert (
        _geographic_error_m(
            candidate,
            native,
            transformer._pipeline.computed["a"],
            always_xy=always_xy,
        )
        <= 1e-8
    )
    candidate_lon = candidate[0] if always_xy else candidate[1]
    native_lon = native[0] if always_xy else native[1]
    assert bool(cp.all(candidate_lon.view(cp.uint64) == native_lon.view(cp.uint64)).get())


@pytest.mark.parametrize(
    "latitude",
    [
        math.nextafter(-80.0, -math.inf),
        -80.0,
        math.nextafter(-80.0, math.inf),
    ],
)
def test_recovered_latitude_guard_boundary_and_nextafter_preserve_native(latitude):
    transformer = _transformer()
    projected = transformer.transform_buffers(
        cp.full(32, 15.0, dtype=cp.float64),
        cp.full(32, latitude, dtype=cp.float64),
        precision="fp64",
        transcendentals="native",
    )
    native = _run_exact(transformer, *projected, NATIVE_LIBDEVICE)
    candidate = _run_exact(transformer, *projected, KROVAK_INVERSE_GUARDED_LOG_RATIO)
    assert (
        _geographic_error_m(candidate, native, transformer._pipeline.computed["a"], always_xy=True)
        <= 1e-8
    )
    if latitude <= -80.0:
        assert _bitwise_equal(candidate, native)


@pytest.mark.parametrize(
    ("x_value", "y_value"),
    [
        (0.0, 0.0),
        (-0.0, 0.0),
        (np.nextafter(0.0, 1.0), np.nextafter(0.0, -1.0)),
        (np.finfo(np.float64).max, np.finfo(np.float64).max),
        (0.0, 6_377_397.155),
        (-0.0, 6_377_397.155),
        (math.nan, 0.0),
        (math.inf, 0.0),
        (0.0, -math.inf),
    ],
)
def test_center_signed_zero_tiny_huge_seam_and_nonfinite_are_bitwise_native(x_value, y_value):
    transformer = _transformer()
    first = cp.full(32, x_value, dtype=cp.float64)
    second = cp.full(32, y_value, dtype=cp.float64)
    native = _run_exact(transformer, first, second, NATIVE_LIBDEVICE)
    candidate = _run_exact(transformer, first, second, KROVAK_INVERSE_GUARDED_LOG_RATIO)
    assert _bitwise_equal(candidate, native)


def test_one_postguard_cold_lane_sends_complete_warp_through_native():
    transformer = _transformer()
    hot = transformer.transform_buffers(
        cp.full(32, 15.0, dtype=cp.float64),
        cp.full(32, 50.0, dtype=cp.float64),
        precision="fp64",
        transcendentals="native",
    )
    cold = transformer.transform_buffers(
        cp.asarray([15.0]),
        cp.asarray([-81.0]),
        precision="fp64",
        transcendentals="native",
    )
    first, second = hot[0].copy(), hot[1].copy()
    first[0], second[0] = cold[0][0], cold[1][0]
    native = _run_exact(transformer, first, second, NATIVE_LIBDEVICE)
    candidate = _run_exact(transformer, first, second, KROVAK_INVERSE_GUARDED_LOG_RATIO)
    assert _bitwise_equal(candidate, native)


def test_native_source_and_abi_are_pinned_while_candidate_has_separate_abi():
    native_source = fused_module._NATIVE_PAIRED_SINCOS_DEVICE_FNS + (
        fused_module._inject_linear_unit_args(
            fused_module._KROVAK_INVERSE_SOURCE.format(
                real_t="double",
                pi=fused_module._PI_LITERALS["float64"],
                tol=fused_module._TOL_LITERALS["float64"],
            )
        )
    )
    assert hashlib.sha256(native_source.encode()).hexdigest() == NATIVE_FP64_SOURCE_SHA256
    assert "krovak_inverse(" in native_source
    assert "cgb0" not in native_source
    assert "log_k" not in native_source

    candidate_source, candidate_name = fused_module._build_projection_guarded_source(
        "krovak", "inverse", KROVAK_INVERSE_GUARDED_LOG_RATIO
    )
    assert candidate_name == "krovak_inverse_guarded_log_ratio"
    assert "cgb0" in candidate_source and "log_k" in candidate_source
    assert "fabs(candidate_phi) <= VP_KROVAK_HOT_PHI_D" in candidate_source
    assert candidate_source.count("__any_sync") == 2
    assert "for (int i = 0; i < 6" not in candidate_source


def test_candidate_cache_resources_stream_graph_and_concurrent_lookup():
    fused_module._kernel_cache.clear()
    native = fused_module._get_kernel(
        "krovak", "inverse", "float64", transcendental_impl=NATIVE_LIBDEVICE
    )
    candidate = fused_module._get_kernel(
        "krovak",
        "inverse",
        "float64",
        transcendental_impl=KROVAK_INVERSE_GUARDED_LOG_RATIO,
    )
    assert candidate is not native
    candidate.compile()
    assert candidate.attributes["num_regs"] <= 40
    assert candidate.attributes["local_size_bytes"] <= 56
    assert candidate.attributes["shared_size_bytes"] == 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        kernels = list(
            executor.map(
                lambda _: fused_module._get_kernel(
                    "krovak",
                    "inverse",
                    "float64",
                    transcendental_impl=KROVAK_INVERSE_GUARDED_LOG_RATIO,
                ),
                range(32),
            )
        )
    assert all(kernel is candidate for kernel in kernels)

    transformer = _transformer()
    projected = transformer.transform_buffers(
        cp.linspace(12.0, 19.0, 4096),
        cp.linspace(48.0, 52.0, 4096),
        precision="fp64",
        transcendentals="native",
    )
    out_x, out_y = cp.empty_like(projected[0]), cp.empty_like(projected[1])
    stream = cp.cuda.Stream(non_blocking=True)
    stream.begin_capture()
    with stream:
        result = _run_exact(
            transformer,
            *projected,
            KROVAK_INVERSE_GUARDED_LOG_RATIO,
            out_x=out_x,
            out_y=out_y,
            stream=stream,
        )
    graph = stream.end_capture()
    dot = graph.debug_dot_str().lower()
    assert result[0] is out_x and result[1] is out_y
    assert dot.count("krovak_inverse_guarded_log_ratio") == 1
    assert "memcpy" not in dot and "memset" not in dot


def test_public_explicit_policy_selects_candidate_while_auto_and_native_stay_native():
    transformer = _transformer()
    projected = transformer.transform_buffers(
        cp.linspace(12.0, 19.0, 4096),
        cp.linspace(48.0, 52.0, 4096),
        precision="fp64",
        transcendentals="native",
    )
    decisions = {
        policy: transformer.explain_strategy(
            direction="INVERSE",
            precision="fp64",
            transcendentals=policy,
            workload_size=projected[0].size,
        )
        .decisions[-1]
        .implementation_id
        for policy in ("native", "auto", "accelerated")
    }
    assert decisions == {
        "native": NATIVE_LIBDEVICE,
        "auto": NATIVE_LIBDEVICE,
        "accelerated": KROVAK_INVERSE_GUARDED_LOG_RATIO,
    }

    native = transformer.transform_buffers(
        *projected,
        direction="INVERSE",
        precision="fp64",
        transcendentals="native",
    )
    automatic = transformer.transform_buffers(
        *projected,
        direction="INVERSE",
        precision="fp64",
        transcendentals="auto",
    )
    assert _bitwise_equal(automatic, native)

    out_x, out_y = cp.empty_like(projected[0]), cp.empty_like(projected[1])
    stream = cp.cuda.Stream(non_blocking=True)
    accelerated = transformer.transform_buffers(
        *projected,
        direction="INVERSE",
        precision="fp64",
        transcendentals="accelerated",
        out_x=out_x,
        out_y=out_y,
        stream=stream,
    )
    stream.synchronize()
    assert accelerated[0] is out_x and accelerated[1] is out_y
    assert (
        _geographic_error_m(
            accelerated, native, transformer._pipeline.computed["a"], always_xy=True
        )
        <= 1e-8
    )


def test_strategy_semantics_is_cached_and_repeated_policy_calls_do_no_setup_work(
    monkeypatch,
):
    transformer = _transformer()
    computed = transformer._pipeline.computed
    assert computed["_strategy_krovak_semantics"] == "standard_bessel.north_oriented"

    def unexpected_isclose(*args, **kwargs):
        raise AssertionError("Krovak policy repeated setup-time numeric predicates")

    monkeypatch.setattr(transcendental_module.math, "isclose", unexpected_isclose)
    for _ in range(2):
        assert (
            transcendental_module.projection_strategy_domain("krovak", "inverse", computed)
            == "krovak.inverse.ellipsoidal.standard_bessel.north_oriented"
        )


@pytest.mark.parametrize(
    ("direction", "dtype"),
    [("forward", "float64"), ("inverse", "float32"), ("inverse", "ds")],
)
def test_candidate_rejects_wrong_operation_or_precision(direction, dtype):
    with pytest.raises(ValueError, match="qualified only"):
        fused_module._get_kernel(
            "krovak",
            direction,
            dtype,
            transcendental_impl=KROVAK_INVERSE_GUARDED_LOG_RATIO,
        )
