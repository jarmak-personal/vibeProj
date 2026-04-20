"""Direct pyproj-oracle validation for GPU-backed transform APIs."""

from __future__ import annotations

import numpy as np
import pytest
from pyproj import Transformer as PyProjTransformer
from pyproj.enums import TransformDirection

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device not available", allow_module_level=True)
except Exception as exc:  # pragma: no cover - import-time environment guard
    pytest.skip(f"CUDA device not available: {exc}", allow_module_level=True)

from _accuracy_cases import EPSG_SWEEP, UNIT_EQUIVALENT_PAIRS, finite_mask, make_lon_lat_grid  # noqa: E402
from vibeproj import Transformer  # noqa: E402

GRID_N = 15
CHUNK_SIZE = 17


def _to_numpy(x):
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else np.asarray(x)


def _max_planar_error_m(ax, ay, bx, by):
    """Maximum Euclidean planar error in meters, ignoring non-finite points."""
    ax = _to_numpy(ax)
    ay = _to_numpy(ay)
    bx = _to_numpy(bx)
    by = _to_numpy(by)
    mask = finite_mask(ax, ay, bx, by)
    assert mask.sum() > 0, "No finite comparison points"
    err = np.sqrt((ax[mask] - bx[mask]) ** 2 + (ay[mask] - by[mask]) ** 2)
    return float(np.max(err))


def _max_angular_error_deg(lon_a, lat_a, lon_b, lat_b):
    """Maximum absolute angular error in degrees, ignoring non-finite points."""
    lon_a = _to_numpy(lon_a)
    lat_a = _to_numpy(lat_a)
    lon_b = _to_numpy(lon_b)
    lat_b = _to_numpy(lat_b)
    mask = finite_mask(lon_a, lat_a, lon_b, lat_b)
    assert mask.sum() > 0, "No finite comparison points"
    err_lon = float(np.max(np.abs(lon_a[mask] - lon_b[mask])))
    err_lat = float(np.max(np.abs(lat_a[mask] - lat_b[mask])))
    return max(err_lon, err_lat)


@pytest.mark.parametrize(
    "label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg",
    EPSG_SWEEP,
    ids=[case[0] for case in EPSG_SWEEP],
)
def test_gpu_transform_forward_vs_pyproj_epsg(
    label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg
):
    """CuPy-backed transform() forward path matches pyproj."""
    lon, lat = make_lon_lat_grid(lat_range, lon_range, n=GRID_N)
    lon_cp = cp.asarray(lon, dtype=cp.float64)
    lat_cp = cp.asarray(lat, dtype=cp.float64)

    pp = PyProjTransformer.from_crs("EPSG:4326", crs_spec, always_xy=True)
    vp = Transformer.from_crs("EPSG:4326", crs_spec)

    pp_x, pp_y = pp.transform(lon, lat)
    vp_x, vp_y = vp.transform(lon_cp, lat_cp)
    assert isinstance(vp_x, cp.ndarray)
    assert isinstance(vp_y, cp.ndarray)

    max_err = _max_planar_error_m(vp_x, vp_y, pp_x, pp_y)
    assert max_err < fwd_tol_m, (
        f"{label} GPU transform forward error {max_err:.6e} m exceeds {fwd_tol_m} m"
    )


@pytest.mark.parametrize(
    "label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg",
    EPSG_SWEEP,
    ids=[case[0] for case in EPSG_SWEEP],
)
def test_gpu_transform_inverse_vs_pyproj_epsg(
    label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg
):
    """CuPy-backed transform() inverse path matches pyproj."""
    lon, lat = make_lon_lat_grid(lat_range, lon_range, n=GRID_N)
    pp = PyProjTransformer.from_crs("EPSG:4326", crs_spec, always_xy=True)
    vp = Transformer.from_crs("EPSG:4326", crs_spec)

    pp_x, pp_y = pp.transform(lon, lat)
    mask = finite_mask(pp_x, pp_y)
    if mask.sum() == 0:
        pytest.skip(f"No finite forward points for {label}")

    pp_inv_lon, pp_inv_lat = pp.transform(
        pp_x[mask], pp_y[mask], direction=TransformDirection.INVERSE
    )
    vp_inv_lon, vp_inv_lat = vp.transform(
        cp.asarray(pp_x[mask], dtype=cp.float64),
        cp.asarray(pp_y[mask], dtype=cp.float64),
        direction="INVERSE",
    )
    assert isinstance(vp_inv_lon, cp.ndarray)
    assert isinstance(vp_inv_lat, cp.ndarray)

    max_err = _max_angular_error_deg(vp_inv_lon, vp_inv_lat, pp_inv_lon, pp_inv_lat)
    assert max_err < inv_tol_deg, (
        f"{label} GPU transform inverse error {max_err:.6e} deg exceeds {inv_tol_deg} deg"
    )


@pytest.mark.parametrize(
    "label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg",
    EPSG_SWEEP,
    ids=[case[0] for case in EPSG_SWEEP],
)
def test_gpu_transform_buffers_forward_vs_pyproj_epsg(
    label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg
):
    """CuPy-backed transform_buffers() forward path matches pyproj."""
    lon, lat = make_lon_lat_grid(lat_range, lon_range, n=GRID_N)
    lon_cp = cp.asarray(lon, dtype=cp.float64)
    lat_cp = cp.asarray(lat, dtype=cp.float64)

    pp = PyProjTransformer.from_crs("EPSG:4326", crs_spec, always_xy=True)
    vp = Transformer.from_crs("EPSG:4326", crs_spec)

    pp_x, pp_y = pp.transform(lon, lat)
    vp_x, vp_y = vp.transform_buffers(lon_cp, lat_cp, precision="fp64")
    assert isinstance(vp_x, cp.ndarray)
    assert isinstance(vp_y, cp.ndarray)

    max_err = _max_planar_error_m(vp_x, vp_y, pp_x, pp_y)
    assert max_err < fwd_tol_m, (
        f"{label} GPU transform_buffers forward error {max_err:.6e} m exceeds {fwd_tol_m} m"
    )


@pytest.mark.parametrize(
    "label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg",
    EPSG_SWEEP,
    ids=[case[0] for case in EPSG_SWEEP],
)
def test_gpu_transform_buffers_inverse_vs_pyproj_epsg(
    label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg
):
    """CuPy-backed transform_buffers() inverse path matches pyproj."""
    lon, lat = make_lon_lat_grid(lat_range, lon_range, n=GRID_N)
    pp = PyProjTransformer.from_crs("EPSG:4326", crs_spec, always_xy=True)
    vp = Transformer.from_crs("EPSG:4326", crs_spec)

    pp_x, pp_y = pp.transform(lon, lat)
    mask = finite_mask(pp_x, pp_y)
    if mask.sum() == 0:
        pytest.skip(f"No finite forward points for {label}")

    pp_inv_lon, pp_inv_lat = pp.transform(
        pp_x[mask], pp_y[mask], direction=TransformDirection.INVERSE
    )
    vp_inv_lon, vp_inv_lat = vp.transform_buffers(
        cp.asarray(pp_x[mask], dtype=cp.float64),
        cp.asarray(pp_y[mask], dtype=cp.float64),
        direction="INVERSE",
        precision="fp64",
    )
    assert isinstance(vp_inv_lon, cp.ndarray)
    assert isinstance(vp_inv_lat, cp.ndarray)

    max_err = _max_angular_error_deg(vp_inv_lon, vp_inv_lat, pp_inv_lon, pp_inv_lat)
    assert max_err < inv_tol_deg, (
        f"{label} GPU transform_buffers inverse error {max_err:.6e} deg exceeds {inv_tol_deg} deg"
    )


@pytest.mark.parametrize(
    "label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg",
    EPSG_SWEEP,
    ids=[case[0] for case in EPSG_SWEEP],
)
def test_gpu_transform_chunked_forward_vs_pyproj_epsg(
    label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg
):
    """GPU-backed transform_chunked() forward path matches pyproj."""
    lon, lat = make_lon_lat_grid(lat_range, lon_range, n=GRID_N)
    pp = PyProjTransformer.from_crs("EPSG:4326", crs_spec, always_xy=True)
    vp = Transformer.from_crs("EPSG:4326", crs_spec)

    pp_x, pp_y = pp.transform(lon, lat)
    vp_x, vp_y = vp.transform_chunked(lon, lat, chunk_size=CHUNK_SIZE)

    max_err = _max_planar_error_m(vp_x, vp_y, pp_x, pp_y)
    assert max_err < fwd_tol_m, (
        f"{label} GPU transform_chunked forward error {max_err:.6e} m exceeds {fwd_tol_m} m"
    )


@pytest.mark.parametrize(
    "label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg",
    EPSG_SWEEP,
    ids=[case[0] for case in EPSG_SWEEP],
)
def test_gpu_transform_chunked_inverse_vs_pyproj_epsg(
    label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg
):
    """GPU-backed transform_chunked() inverse path matches pyproj."""
    lon, lat = make_lon_lat_grid(lat_range, lon_range, n=GRID_N)
    pp = PyProjTransformer.from_crs("EPSG:4326", crs_spec, always_xy=True)
    vp = Transformer.from_crs("EPSG:4326", crs_spec)

    pp_x, pp_y = pp.transform(lon, lat)
    mask = finite_mask(pp_x, pp_y)
    if mask.sum() == 0:
        pytest.skip(f"No finite forward points for {label}")

    pp_inv_lon, pp_inv_lat = pp.transform(
        pp_x[mask], pp_y[mask], direction=TransformDirection.INVERSE
    )
    vp_inv_lon, vp_inv_lat = vp.transform_chunked(
        pp_x[mask], pp_y[mask], direction="INVERSE", chunk_size=CHUNK_SIZE
    )

    max_err = _max_angular_error_deg(vp_inv_lon, vp_inv_lat, pp_inv_lon, pp_inv_lat)
    assert max_err < inv_tol_deg, (
        f"{label} GPU transform_chunked inverse error {max_err:.6e} deg exceeds {inv_tol_deg} deg"
    )


def _proj_to_proj_inputs(src_crs, lat_range, lon_range):
    """Create projected source coordinates for proj_to_proj GPU tests."""
    lon, lat = make_lon_lat_grid(lat_range, lon_range, n=GRID_N)
    pp_src = PyProjTransformer.from_crs("EPSG:4326", src_crs, always_xy=True)
    src_x, src_y = pp_src.transform(lon, lat)
    mask = finite_mask(src_x, src_y)
    assert mask.sum() > 0, f"No finite projected inputs for {src_crs}"
    return src_x[mask], src_y[mask]


def _assert_proj_to_proj_matches(
    api_name, result_x, result_y, exp_x, exp_y, tol_m, label, direction
):
    max_err = _max_planar_error_m(result_x, result_y, exp_x, exp_y)
    assert max_err < tol_m, (
        f"{label} GPU {api_name} {direction} proj_to_proj error {max_err:.6e} m exceeds {tol_m} m"
    )


@pytest.mark.parametrize(
    "label, src_crs, dst_crs, lat_range, lon_range, tol_m",
    UNIT_EQUIVALENT_PAIRS,
    ids=[case[0] for case in UNIT_EQUIVALENT_PAIRS],
)
def test_gpu_proj_to_proj_transform_unit_pairs_vs_pyproj(
    label, src_crs, dst_crs, lat_range, lon_range, tol_m
):
    """CuPy-backed transform() matches pyproj for projected unit pairs."""
    src_x, src_y = _proj_to_proj_inputs(src_crs, lat_range, lon_range)
    pp = PyProjTransformer.from_crs(src_crs, dst_crs, always_xy=True)
    vp = Transformer.from_crs(src_crs, dst_crs)

    exp_x, exp_y = pp.transform(src_x, src_y)
    got_x, got_y = vp.transform(
        cp.asarray(src_x, dtype=cp.float64), cp.asarray(src_y, dtype=cp.float64)
    )
    assert isinstance(got_x, cp.ndarray)
    assert isinstance(got_y, cp.ndarray)
    _assert_proj_to_proj_matches("transform", got_x, got_y, exp_x, exp_y, tol_m, label, "forward")

    exp_inv_x, exp_inv_y = pp.transform(exp_x, exp_y, direction=TransformDirection.INVERSE)
    got_inv_x, got_inv_y = vp.transform(
        cp.asarray(exp_x, dtype=cp.float64),
        cp.asarray(exp_y, dtype=cp.float64),
        direction="INVERSE",
    )
    _assert_proj_to_proj_matches(
        "transform", got_inv_x, got_inv_y, exp_inv_x, exp_inv_y, tol_m, label, "inverse"
    )


@pytest.mark.parametrize(
    "label, src_crs, dst_crs, lat_range, lon_range, tol_m",
    UNIT_EQUIVALENT_PAIRS,
    ids=[case[0] for case in UNIT_EQUIVALENT_PAIRS],
)
def test_gpu_proj_to_proj_transform_buffers_unit_pairs_vs_pyproj(
    label, src_crs, dst_crs, lat_range, lon_range, tol_m
):
    """CuPy-backed transform_buffers() matches pyproj for projected unit pairs."""
    src_x, src_y = _proj_to_proj_inputs(src_crs, lat_range, lon_range)
    pp = PyProjTransformer.from_crs(src_crs, dst_crs, always_xy=True)
    vp = Transformer.from_crs(src_crs, dst_crs)

    exp_x, exp_y = pp.transform(src_x, src_y)
    got_x, got_y = vp.transform_buffers(
        cp.asarray(src_x, dtype=cp.float64),
        cp.asarray(src_y, dtype=cp.float64),
        precision="fp64",
    )
    assert isinstance(got_x, cp.ndarray)
    assert isinstance(got_y, cp.ndarray)
    _assert_proj_to_proj_matches(
        "transform_buffers", got_x, got_y, exp_x, exp_y, tol_m, label, "forward"
    )

    exp_inv_x, exp_inv_y = pp.transform(exp_x, exp_y, direction=TransformDirection.INVERSE)
    got_inv_x, got_inv_y = vp.transform_buffers(
        cp.asarray(exp_x, dtype=cp.float64),
        cp.asarray(exp_y, dtype=cp.float64),
        direction="INVERSE",
        precision="fp64",
    )
    _assert_proj_to_proj_matches(
        "transform_buffers",
        got_inv_x,
        got_inv_y,
        exp_inv_x,
        exp_inv_y,
        tol_m,
        label,
        "inverse",
    )


@pytest.mark.parametrize(
    "label, src_crs, dst_crs, lat_range, lon_range, tol_m",
    UNIT_EQUIVALENT_PAIRS,
    ids=[case[0] for case in UNIT_EQUIVALENT_PAIRS],
)
def test_gpu_proj_to_proj_transform_chunked_unit_pairs_vs_pyproj(
    label, src_crs, dst_crs, lat_range, lon_range, tol_m
):
    """GPU-backed transform_chunked() matches pyproj for projected unit pairs."""
    src_x, src_y = _proj_to_proj_inputs(src_crs, lat_range, lon_range)
    pp = PyProjTransformer.from_crs(src_crs, dst_crs, always_xy=True)
    vp = Transformer.from_crs(src_crs, dst_crs)

    exp_x, exp_y = pp.transform(src_x, src_y)
    got_x, got_y = vp.transform_chunked(src_x, src_y, chunk_size=CHUNK_SIZE)
    _assert_proj_to_proj_matches(
        "transform_chunked", got_x, got_y, exp_x, exp_y, tol_m, label, "forward"
    )

    exp_inv_x, exp_inv_y = pp.transform(exp_x, exp_y, direction=TransformDirection.INVERSE)
    got_inv_x, got_inv_y = vp.transform_chunked(
        exp_x, exp_y, direction="INVERSE", chunk_size=CHUNK_SIZE
    )
    _assert_proj_to_proj_matches(
        "transform_chunked",
        got_inv_x,
        got_inv_y,
        exp_inv_x,
        exp_inv_y,
        tol_m,
        label,
        "inverse",
    )


@pytest.mark.parametrize(
    "label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg",
    [case for case in EPSG_SWEEP if case[1] in ("EPSG:2222", "EPSG:3435")],
    ids=[f"{case[0]}_ds" for case in EPSG_SWEEP if case[1] in ("EPSG:2222", "EPSG:3435")],
)
def test_gpu_transform_buffers_ds_vs_pyproj_non_meter_tmerc(
    label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg
):
    """Double-single TM buffers stay within the repo's accepted pyproj budget."""
    lon, lat = make_lon_lat_grid(lat_range, lon_range, n=GRID_N)
    pp = PyProjTransformer.from_crs("EPSG:4326", crs_spec, always_xy=True)
    vp = Transformer.from_crs("EPSG:4326", crs_spec)

    pp_x, pp_y = pp.transform(lon, lat)
    vp_x, vp_y = vp.transform_buffers(
        cp.asarray(lon, dtype=cp.float64),
        cp.asarray(lat, dtype=cp.float64),
        precision="ds",
    )
    assert isinstance(vp_x, cp.ndarray)
    assert isinstance(vp_y, cp.ndarray)

    max_err = _max_planar_error_m(vp_x, vp_y, pp_x, pp_y)
    assert max_err < fwd_tol_m, (
        f"{label} GPU transform_buffers ds error {max_err:.6e} m exceeds {fwd_tol_m} m"
    )
