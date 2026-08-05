"""CUDA qualification checks for Wave 2B forward Q1.62 candidates."""

from __future__ import annotations

import math

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
from vibeproj.crs import ProjectionParams  # noqa: E402
from vibeproj.ellipsoid import Ellipsoid, SPHERE, WGS84  # noqa: E402
from vibeproj.pipeline import TransformPipeline  # noqa: E402
from vibeproj.transcendentals import (  # noqa: E402
    GEOS_FORWARD_FIXED_Q62,
    LAEA_FORWARD_POLAR_FIXED_Q62,
    NATIVE_LIBDEVICE,
    PROJECTION_FIXED_Q62_MAX_SCALE_M,
)


def _sphere(radius=PROJECTION_FIXED_Q62_MAX_SCALE_M):
    return Ellipsoid(a=radius, b=radius, f=0.0, e=0.0, es=0.0, n=0.0)


def _pipeline(projection, ellipsoid, *, lat_0=0.0, sweep="x", height=35_785_831.0):
    geographic = ProjectionParams("longlat", ellipsoid, north_first=False)
    extra = {"h": height, "sweep_axis": sweep} if projection == "geos" else {}
    projected = ProjectionParams(
        projection,
        ellipsoid,
        lat_0=lat_0,
        north_first=False,
        extra=extra,
    )
    return TransformPipeline(geographic, projected)


def _run(pipeline, lon, lat, projection, implementation, *, computed=None, out=None, stream=None):
    kwargs = {}
    if out is not None:
        kwargs.update(out_x=out[0], out_y=out[1])
    return fused_module.fused_transform(
        cp.asarray(lon),
        cp.asarray(lat),
        projection_name=projection,
        direction="forward",
        computed=pipeline.computed if computed is None else computed,
        src_north_first=False,
        dst_north_first=False,
        xp=cp,
        precision="fp64",
        transcendental_impl=implementation,
        stream=stream,
        **kwargs,
    )


def _assert_bit_exact(actual, expected):
    for got, want in zip(actual, expected, strict=True):
        assert_array_equal(cp.asnumpy(got).view(np.uint64), cp.asnumpy(want).view(np.uint64))


@pytest.mark.parametrize(
    ("projection", "implementation"),
    [("geos", GEOS_FORWARD_FIXED_Q62), ("laea", LAEA_FORWARD_POLAR_FIXED_Q62)],
)
def test_wave2b_forward_q62_variants_compile_and_are_resource_bounded(projection, implementation):
    kernel = fused_module._get_kernel(
        projection,
        "forward",
        "float64",
        transcendental_impl=implementation,
    )
    kernel.compile()
    assert kernel.attributes["shared_size_bytes"] == 0
    assert kernel.attributes["max_threads_per_block"] == 256
    assert kernel.attributes["num_regs"] <= 48


@pytest.mark.parametrize(
    ("projection", "implementation"),
    [("geos", GEOS_FORWARD_FIXED_Q62), ("laea", LAEA_FORWARD_POLAR_FIXED_Q62)],
)
def test_wave2b_forward_q62_variants_are_operation_and_precision_locked(projection, implementation):
    with pytest.raises(ValueError, match="qualified only"):
        fused_module._get_kernel(
            projection,
            "inverse",
            "float64",
            transcendental_impl=implementation,
        )
    with pytest.raises(ValueError, match="qualified only"):
        fused_module._get_kernel(
            projection,
            "forward",
            "float32",
            transcendental_impl=implementation,
        )


@pytest.mark.parametrize("sweep", ["x", "y"])
@pytest.mark.parametrize("ellipsoid", [SPHERE, WGS84], ids=["sphere", "ellipsoid"])
def test_geos_forward_q62_full_globe_is_within_ten_nanometers(ellipsoid, sweep):
    rng = np.random.default_rng(20260809 + int(ellipsoid.e != 0.0) + (sweep == "y"))
    count = 400_000
    lon = rng.uniform(-90.0, 90.0, count)
    lat = rng.uniform(-90.0, 90.0, count)
    pipeline = _pipeline("geos", ellipsoid, sweep=sweep)
    native = _run(pipeline, lon, lat, "geos", NATIVE_LIBDEVICE)
    candidate = _run(pipeline, lon, lat, "geos", GEOS_FORWARD_FIXED_Q62)
    error = cp.maximum(cp.abs(candidate[0] - native[0]), cp.abs(candidate[1] - native[1]))
    finite = cp.isfinite(error)
    assert float(cp.max(error[finite])) <= 1e-8
    assert float(cp.percentile(error[finite], 99)) <= 1e-8


@pytest.mark.parametrize("sweep", ["x", "y"])
@pytest.mark.parametrize("ellipsoid", [SPHERE, WGS84], ids=["sphere", "ellipsoid"])
def test_geos_forward_q62_visibility_limb_and_nonfinite_are_exact(ellipsoid, sweep):
    pipeline = _pipeline("geos", ellipsoid, sweep=sweep)
    computed = pipeline.computed
    limb = math.degrees(math.acos(computed["a"] / computed["H"]))
    lon = cp.asarray(
        [
            math.nextafter(limb, 0.0),
            limb,
            math.nextafter(limb, math.inf),
            -math.nextafter(limb, 0.0),
            -limb,
            -math.nextafter(limb, math.inf),
            180.0,
            math.nan,
            math.inf,
        ]
    )
    lat = cp.zeros_like(lon)
    native = _run(pipeline, lon, lat, "geos", NATIVE_LIBDEVICE)
    candidate = _run(pipeline, lon, lat, "geos", GEOS_FORWARD_FIXED_Q62)
    # The visibility/sentinel classification is exact. Qualified inside-limb
    # finite coordinates retain the normal <=10 nm approximation contract.
    for got, want in zip(candidate, native, strict=True):
        got_host = cp.asnumpy(got)
        want_host = cp.asnumpy(want)
        assert_array_equal(np.isfinite(got_host), np.isfinite(want_host))
        nonfinite = ~np.isfinite(want_host)
        assert_array_equal(
            got_host.view(np.uint64)[nonfinite], want_host.view(np.uint64)[nonfinite]
        )
    error = cp.maximum(cp.abs(candidate[0] - native[0]), cp.abs(candidate[1] - native[1]))
    finite = cp.isfinite(error)
    assert float(cp.max(error[finite])) <= 1e-8


@pytest.mark.parametrize("sweep", ["x", "y"])
@pytest.mark.parametrize("ellipsoid", [SPHERE, WGS84], ids=["sphere", "ellipsoid"])
def test_geos_forward_q62_analytic_limb_grid_is_bit_exact_native(ellipsoid, sweep):
    pipeline = _pipeline("geos", ellipsoid, sweep=sweep)
    computed = pipeline.computed
    count = 4_096
    latitude = np.arcsin(
        np.linspace(-math.sin(math.radians(81.0)), math.sin(math.radians(81.0)), count)
    )
    flattening_ratio = computed["r_pol2"] / computed["r_eq2"]
    phi_gc = np.arctan(flattening_ratio * np.tan(latitude))
    cos_phi_gc = np.cos(phi_gc)
    r_earth = math.sqrt(computed["r_pol2"]) / np.sqrt(
        1.0 - (computed["r_eq2"] - computed["r_pol2"]) / computed["r_eq2"] * cos_phi_gc * cos_phi_gc
    )
    longitude_limit = np.arccos(
        np.clip(
            computed["r_eq2"] / (computed["H"] * r_earth * cos_phi_gc),
            -1.0,
            1.0,
        )
    )
    longitude = np.rad2deg(
        np.concatenate(
            (
                np.nextafter(longitude_limit, 0.0),
                longitude_limit,
                np.nextafter(longitude_limit, math.inf),
                -np.nextafter(longitude_limit, 0.0),
                -longitude_limit,
                -np.nextafter(longitude_limit, math.inf),
            )
        )
    )
    latitude = np.tile(np.rad2deg(latitude), 6)

    native = _run(pipeline, longitude, latitude, "geos", NATIVE_LIBDEVICE)
    candidate = _run(pipeline, longitude, latitude, "geos", GEOS_FORWARD_FIXED_Q62)
    _assert_bit_exact(candidate, native)


@pytest.mark.parametrize("height", [1.0, 1_000.0, 35_785_831.0, 6.4e16])
def test_geos_forward_q62_height_cancellation_bound_at_maximum_radius(height):
    ellipsoid = _sphere()
    pipeline = _pipeline("geos", ellipsoid, height=height)
    computed = pipeline.computed
    limb = math.degrees(math.acos(computed["a"] / computed["H"]))
    rng = np.random.default_rng(20260810 + int(math.log10(height)))
    lon = rng.uniform(-0.8 * limb, 0.8 * limb, 200_000)
    lat = rng.uniform(-0.1 * limb, 0.1 * limb, 200_000)
    native = _run(pipeline, lon, lat, "geos", NATIVE_LIBDEVICE)
    candidate = _run(pipeline, lon, lat, "geos", GEOS_FORWARD_FIXED_Q62)
    error = cp.maximum(cp.abs(candidate[0] - native[0]), cp.abs(candidate[1] - native[1]))
    finite = cp.isfinite(error)
    assert float(cp.max(error[finite])) <= 1e-8


def test_geos_forward_q62_scale_and_satellite_parameter_guard_edges():
    lon = cp.asarray([-30.0, -5.0, 0.0, 5.0, 30.0])
    lat = cp.asarray([-10.0, -1.0, 0.0, 1.0, 10.0])

    over = _sphere(math.nextafter(PROJECTION_FIXED_Q62_MAX_SCALE_M, math.inf))
    pipeline = _pipeline("geos", over)
    native = _run(pipeline, lon, lat, "geos", NATIVE_LIBDEVICE)
    candidate = _run(pipeline, lon, lat, "geos", GEOS_FORWARD_FIXED_Q62)
    _assert_bit_exact(candidate, native)

    pipeline = _pipeline("geos", _sphere())
    for key, value in (
        ("H", math.nextafter(pipeline.computed["H"], math.inf)),
        ("h", 0.0),
        ("h", math.inf),
        ("H", math.nan),
    ):
        malformed = dict(pipeline.computed)
        malformed[key] = value
        native = _run(pipeline, lon, lat, "geos", NATIVE_LIBDEVICE, computed=malformed)
        candidate = _run(
            pipeline,
            lon,
            lat,
            "geos",
            GEOS_FORWARD_FIXED_Q62,
            computed=malformed,
        )
        _assert_bit_exact(candidate, native)


@pytest.mark.parametrize("lat_0", [-90.0, 90.0])
@pytest.mark.parametrize("ellipsoid", [SPHERE, WGS84], ids=["sphere", "ellipsoid"])
def test_laea_polar_forward_q62_is_within_ten_nanometers(ellipsoid, lat_0):
    rng = np.random.default_rng(20260811 + int(lat_0) + int(ellipsoid.e != 0.0))
    lon = rng.uniform(-180.0, 180.0, 400_000)
    lat = rng.uniform(-90.0, 90.0, 400_000)
    pipeline = _pipeline("laea", ellipsoid, lat_0=lat_0)
    native = _run(pipeline, lon, lat, "laea", NATIVE_LIBDEVICE)
    candidate = _run(pipeline, lon, lat, "laea", LAEA_FORWARD_POLAR_FIXED_Q62)
    error = cp.maximum(cp.abs(candidate[0] - native[0]), cp.abs(candidate[1] - native[1]))
    finite = cp.isfinite(error)
    assert float(cp.max(error[finite])) <= 1e-8


def test_laea_polar_q62_nonpolar_scale_and_sentinel_fallback_are_exact():
    lon = cp.asarray([-180.0, -45.0, 0.0, 45.0, 180.0, math.nan, math.inf])
    lat = cp.asarray([-90.0, -45.0, 0.0, 45.0, 90.0, 0.0, 0.0])
    for lat_0, ellipsoid in (
        (0.0, WGS84),
        (35.0, WGS84),
        (90.0, _sphere(math.nextafter(PROJECTION_FIXED_Q62_MAX_SCALE_M, math.inf))),
    ):
        pipeline = _pipeline("laea", ellipsoid, lat_0=lat_0)
        native = _run(pipeline, lon, lat, "laea", NATIVE_LIBDEVICE)
        candidate = _run(pipeline, lon, lat, "laea", LAEA_FORWARD_POLAR_FIXED_Q62)
        _assert_bit_exact(candidate, native)


@pytest.mark.parametrize(
    ("projection", "implementation", "lat_0"),
    [
        ("geos", GEOS_FORWARD_FIXED_Q62, 0.0),
        ("laea", LAEA_FORWARD_POLAR_FIXED_Q62, 90.0),
    ],
)
def test_wave2b_forward_q62_honors_preallocated_buffers_and_stream(
    projection, implementation, lat_0
):
    pipeline = _pipeline(projection, WGS84, lat_0=lat_0)
    lon = cp.asarray([-20.0, 0.0, 20.0])
    lat = cp.asarray([-10.0, 0.0, 10.0])
    out = (cp.empty_like(lon), cp.empty_like(lat))
    stream = cp.cuda.Stream(non_blocking=True)
    got = _run(pipeline, lon, lat, projection, implementation, out=out, stream=stream)
    stream.synchronize()
    assert got[0] is out[0]
    assert got[1] is out[1]
