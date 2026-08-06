"""Wave 3 signed projected-axis and regular Krovak baselines."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pyproj import CRS
from pyproj import Transformer as PyProjTransformer

from vibeproj import Transformer
from vibeproj.crs import ProjectionParams, resolve_projection_params
from vibeproj.ellipsoid import SPHERE
from vibeproj.pipeline import TransformPipeline


KROVAK_CODES = (5513, 5514)


def _cupy_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device not available")
    except Exception as exc:
        pytest.skip(f"CUDA device not available: {exc}")
    return cp


@pytest.mark.parametrize(
    ("epsg", "north_first", "visualization_north_first", "east_sign", "north_sign"),
    [
        (5513, True, True, -1.0, -1.0),
        (5514, False, False, 1.0, 1.0),
    ],
)
def test_krovak_axis_metadata_keeps_component_order_and_sign_independent(
    epsg, north_first, visualization_north_first, east_sign, north_sign
):
    params = resolve_projection_params(CRS.from_epsg(epsg))
    assert params.north_first is north_first
    assert params.visualization_north_first is visualization_north_first
    assert params.easting_axis_sign == east_sign
    assert params.northing_axis_sign == north_sign
    assert params.x_unit_to_m > 0.0
    assert params.y_unit_to_m > 0.0


@pytest.mark.parametrize(
    ("epsg", "north_first", "visualization_north_first"),
    [
        (3031, False, False),
        (3413, False, False),
        (32661, True, False),
        (32761, True, False),
    ],
)
def test_polar_axis_meridian_directions_preserve_component_order_and_positive_signs(
    epsg, north_first, visualization_north_first
):
    params = resolve_projection_params(CRS.from_epsg(epsg))
    assert params.north_first is north_first
    assert params.visualization_north_first is visualization_north_first
    assert params.easting_axis_sign == 1.0
    assert params.northing_axis_sign == 1.0
    assert params.x_unit_to_m > 0.0
    assert params.y_unit_to_m > 0.0


def test_projection_params_axis_fields_are_appended_after_existing_positional_api():
    params = ProjectionParams(
        "eqc",
        SPHERE,
        1.0,
        2.0,
        3.0,
        4.0,
        0.9,
        5.0,
        6.0,
        7.0,
        8.0,
        9,
        True,
        True,
        {"sentinel": True},
        "existing operation method",
    )
    assert params.operation_method == "existing operation method"
    assert params.easting_axis_sign == 1.0
    assert params.northing_axis_sign == 1.0
    assert params.visualization_north_first is False


@pytest.mark.parametrize("epsg", KROVAK_CODES)
@pytest.mark.parametrize("always_xy", [False, True])
def test_krovak_forward_inverse_and_roundtrip_match_same_datum_pyproj(epsg, always_xy):
    target = CRS.from_epsg(epsg)
    source = target.geodetic_crs
    expected_forward = PyProjTransformer.from_crs(source, target, always_xy=always_xy)
    expected_inverse = PyProjTransformer.from_crs(target, source, always_xy=always_xy)
    actual = Transformer.from_crs(source, target, always_xy=always_xy)
    lon = np.array([12.3, 14.0, 16.5, 18.7])
    lat = np.array([48.4, 49.5, 50.2, 51.1])
    first, second = (lon, lat) if always_xy else (lat, lon)

    expected_x, expected_y = expected_forward.transform(first, second)
    actual_x, actual_y = actual.transform(first, second)
    assert_allclose(actual_x, expected_x, rtol=0.0, atol=2e-6)
    assert_allclose(actual_y, expected_y, rtol=0.0, atol=2e-6)

    expected_first, expected_second = expected_inverse.transform(expected_x, expected_y)
    actual_first, actual_second = actual.transform(actual_x, actual_y, direction="INVERSE")
    assert_allclose(actual_first, expected_first, rtol=0.0, atol=1e-11)
    assert_allclose(actual_second, expected_second, rtol=0.0, atol=1e-11)
    assert_allclose(actual_first, first, rtol=0.0, atol=1e-11)
    assert_allclose(actual_second, second, rtol=0.0, atol=1e-11)


def test_regular_krovak_visualization_preserves_x_southing_y_westing_like_pyproj():
    target = CRS.from_epsg(5513)
    source = target.geodetic_crs
    lon = np.array([13.0, 14.5, 17.0])
    lat = np.array([49.0, 50.0, 50.8])
    native = Transformer.from_crs(source, target, always_xy=False)
    visual = Transformer.from_crs(source, target, always_xy=True)
    oracle_native = PyProjTransformer.from_crs(source, target, always_xy=False)
    oracle_visual = PyProjTransformer.from_crs(source, target, always_xy=True)

    native_southing, native_westing = native.transform(lat, lon)
    visual_southing, visual_westing = visual.transform(lon, lat)
    expected_native = oracle_native.transform(lat, lon)
    expected_visual = oracle_visual.transform(lon, lat)
    assert_allclose(native_southing, expected_native[0], rtol=0.0, atol=2e-6)
    assert_allclose(native_westing, expected_native[1], rtol=0.0, atol=2e-6)
    assert_allclose(visual_southing, expected_visual[0], rtol=0.0, atol=2e-6)
    assert_allclose(visual_westing, expected_visual[1], rtol=0.0, atol=2e-6)
    assert_allclose(visual_southing, native_southing, rtol=0.0, atol=0.0)
    assert_allclose(visual_westing, native_westing, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("always_xy", [False, True])
def test_regular_to_north_oriented_krovak_proj_to_proj_matches_pyproj(always_xy):
    source = CRS.from_epsg(5513)
    target = CRS.from_epsg(5514)
    expected = PyProjTransformer.from_crs(source, target, always_xy=always_xy)
    actual = Transformer.from_crs(source, target, always_xy=always_xy)
    southing = np.array([1_196_831.4, 1_048_524.8, 950_000.0])
    westing = np.array([907_368.2, 774_126.6, 650_000.0])
    expected_x, expected_y = expected.transform(southing, westing)
    actual_x, actual_y = actual.transform(southing, westing)
    assert_allclose(actual_x, expected_x, rtol=0.0, atol=2e-8)
    assert_allclose(actual_y, expected_y, rtol=0.0, atol=2e-8)


@pytest.mark.parametrize("unit", ["m", "us-ft"])
def test_custom_north_krovak_false_offsets_forward_matches_pyproj_and_roundtrips(unit):
    target = CRS.from_user_input(
        "+proj=krovak +lat_0=49.5 +lon_0=24.83333333333333 "
        "+alpha=30.28813975277778 +k=0.9999 +x_0=100 +y_0=200 "
        f"+ellps=bessel +units={unit} +type=crs"
    )
    source = target.geodetic_crs
    expected = PyProjTransformer.from_crs(source, target, always_xy=True)
    actual = Transformer.from_crs(source, target, always_xy=True)
    lon = np.array([13.0, 14.0, 17.5])
    lat = np.array([49.0, 50.0, 50.7])
    expected_x, expected_y = expected.transform(lon, lat)
    actual_x, actual_y = actual.transform(lon, lat)
    tolerance = 3e-6 if unit == "m" else 1e-5
    assert_allclose(actual_x, expected_x, rtol=0.0, atol=tolerance)
    assert_allclose(actual_y, expected_y, rtol=0.0, atol=tolerance)

    # Custom nonzero offsets expose an asymmetric inverse in legacy PROJ
    # pipelines through 9.5. vibeProj intentionally keeps the compatible
    # forward convention but applies it symmetrically on inverse.
    actual_lon, actual_lat = actual.transform(actual_x, actual_y, direction="INVERSE")
    assert_allclose(actual_lon, lon, rtol=0.0, atol=1e-11)
    assert_allclose(actual_lat, lat, rtol=0.0, atol=1e-11)


def test_custom_southing_westing_axis_uses_native_order_for_visualization():
    target = CRS.from_user_input(
        "+proj=krovak +lat_0=49.5 +lon_0=24.83333333333333 "
        "+alpha=30.28813975277778 +k=0.9999 +ellps=bessel "
        "+czech +axis=swu +units=m +type=crs"
    )
    params = resolve_projection_params(target)
    assert params.north_first is True
    assert params.visualization_north_first is True
    assert params.easting_axis_sign == -1.0
    assert params.northing_axis_sign == -1.0


def test_signed_units_offsets_roundtrip_in_generic_xp_pipeline():
    geographic = ProjectionParams("longlat", SPHERE, north_first=False)
    canonical = ProjectionParams(
        "eqc", SPHERE, x_0=100.0, y_0=-200.0, x_unit_to_m=0.3048, y_unit_to_m=2.0
    )
    signed = ProjectionParams(
        "eqc",
        SPHERE,
        x_0=100.0,
        y_0=-200.0,
        x_unit_to_m=0.3048,
        y_unit_to_m=2.0,
        north_first=True,
        easting_axis_sign=-1.0,
        northing_axis_sign=-1.0,
        visualization_north_first=True,
    )
    lon = np.array([-10.0, 0.0, 15.0])
    lat = np.array([-5.0, 20.0, 60.0])
    east, north = TransformPipeline(geographic, canonical).transform(lon, lat, np)
    south, west = TransformPipeline(geographic, signed).transform(lon, lat, np)
    assert_allclose(south, -north, rtol=0.0, atol=0.0)
    assert_allclose(west, -east, rtol=0.0, atol=0.0)
    actual_lon, actual_lat = TransformPipeline(signed, geographic).transform(south, west, np)
    assert_allclose(actual_lon, lon, rtol=0.0, atol=2e-14)
    assert_allclose(actual_lat, lat, rtol=0.0, atol=2e-14)


@pytest.mark.parametrize("epsg", KROVAK_CODES)
@pytest.mark.parametrize("always_xy", [False, True])
def test_krovak_gpu_buffers_stream_and_inverse_match_cpu(epsg, always_xy):
    cp = _cupy_or_skip()
    target = CRS.from_epsg(epsg)
    source = target.geodetic_crs
    transformer = Transformer.from_crs(source, target, always_xy=always_xy)
    lon = np.linspace(12.0, 18.8, 4096)
    lat = np.linspace(48.3, 51.1, 4096)
    first, second = (lon, lat) if always_xy else (lat, lon)
    expected_x, expected_y = transformer.transform(first, second)
    gpu_first = cp.asarray(first)
    gpu_second = cp.asarray(second)
    out_x = cp.empty_like(gpu_first)
    out_y = cp.empty_like(gpu_second)
    stream = cp.cuda.Stream(non_blocking=True)
    actual_x, actual_y = transformer.transform_buffers(
        gpu_first, gpu_second, out_x=out_x, out_y=out_y, stream=stream
    )
    stream.synchronize()
    assert actual_x is out_x
    assert actual_y is out_y
    assert_allclose(cp.asnumpy(actual_x), expected_x, rtol=0.0, atol=2e-6)
    assert_allclose(cp.asnumpy(actual_y), expected_y, rtol=0.0, atol=2e-6)

    inverse_first = cp.empty_like(gpu_first)
    inverse_second = cp.empty_like(gpu_second)
    got_first, got_second = transformer.transform_buffers(
        actual_x,
        actual_y,
        direction="INVERSE",
        out_x=inverse_first,
        out_y=inverse_second,
        stream=stream,
    )
    stream.synchronize()
    assert got_first is inverse_first
    assert got_second is inverse_second
    assert_allclose(cp.asnumpy(got_first), first, rtol=0.0, atol=1e-11)
    assert_allclose(cp.asnumpy(got_second), second, rtol=0.0, atol=1e-11)


@pytest.mark.parametrize("epsg", KROVAK_CODES)
def test_krovak_nonfinite_outputs_are_atomic_and_positive_on_cpu_and_gpu(epsg):
    target = CRS.from_epsg(epsg)
    source = target.geodetic_crs
    transformer = Transformer.from_crs(source, target, always_xy=True)
    lon = np.array([math.nan, math.inf, 14.0])
    lat = np.array([50.0, 50.0, math.inf])
    with pytest.warns(UserWarning, match="non-finite"):
        x, y = transformer.transform(lon, lat)
    assert np.isnan(x[0]) and np.isnan(y[0])
    assert np.isposinf(x[1:]).all()
    assert np.isposinf(y[1:]).all()

    projected_x = np.array([math.nan, math.inf, 0.0])
    projected_y = np.array([0.0, 0.0, math.inf])
    with pytest.warns(UserWarning, match="non-finite"):
        out_lon, out_lat = transformer.transform(projected_x, projected_y, direction="INVERSE")
    assert np.isnan(out_lon[0]) and np.isnan(out_lat[0])
    assert np.isposinf(out_lon[1:]).all()
    assert np.isposinf(out_lat[1:]).all()

    cp = _cupy_or_skip()
    gpu_x, gpu_y = transformer.transform_buffers(cp.asarray(lon), cp.asarray(lat))
    assert_array_equal(cp.asnumpy(gpu_x), x)
    assert_array_equal(cp.asnumpy(gpu_y), y)
    gpu_lon, gpu_lat = transformer.transform_buffers(
        cp.asarray(projected_x), cp.asarray(projected_y), direction="INVERSE"
    )
    assert_array_equal(cp.asnumpy(gpu_lon), out_lon)
    assert_array_equal(cp.asnumpy(gpu_lat), out_lat)


def test_fused_axis_signs_reuse_existing_unit_abi_and_kernel_cache():
    cp = _cupy_or_skip()
    import vibeproj.fused_kernels as fused

    injected = fused._inject_linear_unit_args(fused._KROVAK_FORWARD_SOURCE)
    assert "double x_unit_to_m, double y_unit_to_m" in injected
    assert "easting_axis_sign" not in injected
    assert "northing_axis_sign" not in injected

    geographic = ProjectionParams("longlat", SPHERE, north_first=False)
    canonical = ProjectionParams("eqc", SPHERE, north_first=False)
    signed = ProjectionParams(
        "eqc",
        SPHERE,
        north_first=True,
        easting_axis_sign=-1.0,
        northing_axis_sign=-1.0,
        visualization_north_first=True,
    )
    lon = cp.asarray(np.array([-10.0, 15.0]))
    lat = cp.asarray(np.array([5.0, 60.0]))
    east, north = TransformPipeline(geographic, canonical).transform(lon, lat, cp)
    south, west = TransformPipeline(geographic, signed).transform(lon, lat, cp)
    assert_allclose(cp.asnumpy(south), -cp.asnumpy(north), rtol=0.0, atol=0.0)
    assert_allclose(cp.asnumpy(west), -cp.asnumpy(east), rtol=0.0, atol=0.0)
    keys = [key for key in fused._kernel_cache if key[:2] == ("eqc", "forward")]
    assert len(keys) == 1
