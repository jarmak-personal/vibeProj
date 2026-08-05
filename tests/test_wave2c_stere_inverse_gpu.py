"""CUDA qualification checks for STERE inverse Q1.62 phi2."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pyproj import CRS

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device not available", allow_module_level=True)
except Exception as exc:  # pragma: no cover - import-time environment guard
    pytest.skip(f"CUDA device not available: {exc}", allow_module_level=True)

import vibeproj.fused_kernels as fused_module  # noqa: E402
from vibeproj.crs import ProjectionParams, resolve_projection_params  # noqa: E402
from vibeproj.ellipsoid import SPHERE, WGS84, Ellipsoid  # noqa: E402
from vibeproj.pipeline import TransformPipeline  # noqa: E402
from vibeproj.transcendentals import (  # noqa: E402
    NATIVE_LIBDEVICE,
    PROJECTION_FIXED_Q62_MAX_SCALE_M,
    STERE_INVERSE_FIXED_Q62,
)


IMPLEMENTATION = STERE_INVERSE_FIXED_Q62


def _epsg_pipeline(code: int) -> TransformPipeline:
    projected = CRS.from_epsg(code)
    return TransformPipeline(
        resolve_projection_params(projected.geodetic_crs),
        resolve_projection_params(projected),
    )


def _pipeline(ellipsoid: Ellipsoid, *, latitude_origin: float = 90.0) -> TransformPipeline:
    geographic = ProjectionParams("longlat", ellipsoid, north_first=False)
    projected = ProjectionParams(
        "stere",
        ellipsoid,
        lat_0=latitude_origin,
        k_0=0.994,
        north_first=False,
    )
    return TransformPipeline(geographic, projected)


def _run(
    pipeline: TransformPipeline,
    x,
    y,
    implementation: str,
    *,
    computed: dict | None = None,
    out=None,
    stream=None,
):
    kwargs = {}
    if out is not None:
        kwargs.update(out_x=out[0], out_y=out[1])
    return fused_module.fused_transform(
        cp.asarray(x),
        cp.asarray(y),
        projection_name="stere",
        direction="inverse",
        computed=pipeline.computed if computed is None else computed,
        src_north_first=False,
        dst_north_first=False,
        xp=cp,
        precision="fp64",
        transcendental_impl=implementation,
        stream=stream,
        **kwargs,
    )


def _assert_bit_exact(actual, expected) -> None:
    for got, want in zip(actual, expected, strict=True):
        assert_array_equal(cp.asnumpy(got).view(np.uint64), cp.asnumpy(want).view(np.uint64))


def test_stere_inverse_fixed_q62_phi2_compiles_and_is_resource_bounded():
    kernel = fused_module._get_kernel(
        "stere",
        "inverse",
        "float64",
        transcendental_impl=IMPLEMENTATION,
    )
    kernel.compile()
    assert kernel.attributes["shared_size_bytes"] == 0
    assert kernel.attributes["local_size_bytes"] <= 40
    assert kernel.attributes["num_regs"] <= 40
    assert kernel.attributes["max_threads_per_block"] == 256


def test_stere_inverse_fixed_q62_phi2_is_operation_and_precision_locked():
    with pytest.raises(ValueError, match="qualified only"):
        fused_module._get_kernel(
            "stere",
            "forward",
            "float64",
            transcendental_impl=IMPLEMENTATION,
        )
    with pytest.raises(ValueError, match="qualified only"):
        fused_module._get_kernel(
            "stere",
            "inverse",
            "float32",
            transcendental_impl=IMPLEMENTATION,
        )


@pytest.mark.parametrize(
    "epsg",
    [32661, 32761, 3413, 3031, 2985],
    ids=[
        "variant-a-north",
        "variant-a-south",
        "variant-b-north",
        "variant-b-south",
        "variant-c-south",
    ],
)
def test_stere_inverse_fixed_q62_phi2_log_plane_is_within_ten_nanometers(epsg):
    pipeline = _epsg_pipeline(epsg)
    computed = pipeline.computed
    rng = np.random.default_rng(20260844 + epsg)
    count = 200_000
    radius = np.power(10.0, rng.uniform(-12.0, 150.0, count))
    azimuth = rng.uniform(-math.pi, math.pi, count)
    x = computed["x0"] + radius * np.cos(azimuth) * computed["a"]
    y = computed["y0"] + radius * np.sin(azimuth) * computed["a"]
    native = _run(pipeline, x, y, NATIVE_LIBDEVICE)
    candidate = _run(pipeline, x, y, IMPLEMENTATION)

    native_host = [cp.asnumpy(value) for value in native]
    candidate_host = [cp.asnumpy(value) for value in candidate]
    native_finite = np.isfinite(native_host[0]) & np.isfinite(native_host[1])
    candidate_finite = np.isfinite(candidate_host[0]) & np.isfinite(candidate_host[1])
    assert_array_equal(candidate_finite, native_finite)
    angular_error = np.maximum(
        np.abs(candidate_host[0] - native_host[0]),
        np.abs(candidate_host[1] - native_host[1]),
    )
    physical_error = angular_error * math.pi / 180.0 * computed["a"]
    assert float(np.max(physical_error[native_finite])) <= 1e-8
    assert float(np.percentile(physical_error[native_finite], 99)) <= 1e-8


def test_stere_inverse_fixed_q62_phi2_center_axes_and_nonfinite_match_native():
    pipeline = _pipeline(WGS84)
    scale = pipeline.computed["a"]
    radius = np.asarray(
        [
            0.0,
            math.nextafter(0.0, 1.0),
            1e-150,
            1e-12,
            1.0,
            1e12,
            1e150,
            math.inf,
            math.nan,
        ]
    )
    azimuth = np.asarray([0.0, -0.0, math.pi / 2, -math.pi / 2, math.pi, 0.0, 1.0, 0.0, 0.0])
    with np.errstate(invalid="ignore", over="ignore"):
        x = radius * np.cos(azimuth) * scale
        y = radius * np.sin(azimuth) * scale
    native = _run(pipeline, x, y, NATIVE_LIBDEVICE)
    candidate = _run(pipeline, x, y, IMPLEMENTATION)
    _assert_bit_exact(candidate, native)


@pytest.mark.parametrize("eccentricity", [0.0, math.inf, math.nan])
def test_stere_inverse_fixed_q62_phi2_nonellipsoidal_or_malformed_e_is_native(eccentricity):
    pipeline = _pipeline(SPHERE)
    computed = dict(pipeline.computed)
    computed["e"] = eccentricity
    x = cp.asarray([-1e7, -1.0, -0.0, 0.0, 1.0, 1e7])
    y = cp.asarray([1e7, 1.0, 0.0, -0.0, -1.0, -1e7])
    native = _run(pipeline, x, y, NATIVE_LIBDEVICE, computed=computed)
    candidate = _run(pipeline, x, y, IMPLEMENTATION, computed=computed)
    _assert_bit_exact(candidate, native)


@pytest.mark.parametrize(
    ("eccentricity", "eligible"),
    [
        (math.nextafter(0.05, 0.0), False),
        (0.05, True),
        (0.2, True),
        (math.nextafter(0.2, math.inf), False),
    ],
)
def test_stere_inverse_fixed_q62_phi2_eccentricity_guard_edges(eccentricity, eligible):
    pipeline = _pipeline(WGS84)
    computed = dict(pipeline.computed)
    computed["e"] = eccentricity
    x = cp.asarray([-1e12, -1.0, 0.0, 1.0, 1e12]) * computed["a"]
    y = cp.asarray([1e12, 1.0, 0.0, -1.0, -1e12]) * computed["a"]
    native = _run(pipeline, x, y, NATIVE_LIBDEVICE, computed=computed)
    candidate = _run(pipeline, x, y, IMPLEMENTATION, computed=computed)
    if not eligible:
        _assert_bit_exact(candidate, native)
    else:
        error_degrees = cp.maximum(
            cp.abs(candidate[0] - native[0]), cp.abs(candidate[1] - native[1])
        )
        assert float(cp.max(error_degrees)) * math.pi / 180.0 * computed["a"] <= 1e-8


@pytest.mark.parametrize(
    "radius",
    [PROJECTION_FIXED_Q62_MAX_SCALE_M, math.nextafter(PROJECTION_FIXED_Q62_MAX_SCALE_M, math.inf)],
)
def test_stere_inverse_fixed_q62_phi2_scale_boundary(radius):
    ellipsoid = Ellipsoid.from_af(radius, 298.257223563)
    pipeline = _pipeline(ellipsoid)
    x = cp.asarray([-1e12, -1.0, 0.0, 1.0, 1e12]) * radius
    y = cp.asarray([1e12, 1.0, 0.0, -1.0, -1e12]) * radius
    native = _run(pipeline, x, y, NATIVE_LIBDEVICE)
    candidate = _run(pipeline, x, y, IMPLEMENTATION)
    if radius > PROJECTION_FIXED_Q62_MAX_SCALE_M:
        _assert_bit_exact(candidate, native)
    else:
        error_degrees = cp.maximum(
            cp.abs(candidate[0] - native[0]), cp.abs(candidate[1] - native[1])
        )
        assert float(cp.max(error_degrees)) * math.pi / 180.0 * radius <= 1e-8


def test_stere_inverse_fixed_q62_phi2_honors_preallocated_buffers_and_stream():
    pipeline = _epsg_pipeline(3413)
    x = cp.asarray([-1e6, 0.0, 1e6])
    y = cp.asarray([1e6, 0.0, -1e6])
    out = (cp.empty_like(x), cp.empty_like(y))
    stream = cp.cuda.Stream(non_blocking=True)
    got = _run(pipeline, x, y, IMPLEMENTATION, out=out, stream=stream)
    stream.synchronize()
    assert got[0] is out[0]
    assert got[1] is out[1]
