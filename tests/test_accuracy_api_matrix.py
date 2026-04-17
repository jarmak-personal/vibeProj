"""Pyproj-oracle accuracy matrix for every public transform API."""

from __future__ import annotations

import numpy as np
import pytest
from pyproj import Transformer as PyProjTransformer
from pyproj.enums import TransformDirection

from _accuracy_cases import EPSG_SWEEP, UNIT_EQUIVALENT_PAIRS, finite_mask, make_lon_lat_grid
from vibeproj import Transformer

GRID_N = 15
CHUNK_SIZE = 17


def _max_planar_error_m(ax, ay, bx, by):
    """Maximum Euclidean planar error in meters, ignoring non-finite points."""
    mask = finite_mask(ax, ay, bx, by)
    assert mask.sum() > 0, "No finite comparison points"
    err = np.sqrt((ax[mask] - bx[mask]) ** 2 + (ay[mask] - by[mask]) ** 2)
    return float(np.max(err))


def _max_angular_error_deg(lon_a, lat_a, lon_b, lat_b):
    """Maximum absolute angular error in degrees, ignoring non-finite points."""
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
def test_transform_buffers_forward_vs_pyproj_epsg(
    label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg
):
    """transform_buffers forward path matches pyproj across the EPSG sweep."""
    lon, lat = make_lon_lat_grid(lat_range, lon_range, n=GRID_N)
    pp = PyProjTransformer.from_crs("EPSG:4326", crs_spec, always_xy=True)
    vp = Transformer.from_crs("EPSG:4326", crs_spec)

    pp_x, pp_y = pp.transform(lon, lat)
    vp_x, vp_y = vp.transform_buffers(lon.copy(), lat.copy())

    max_err = _max_planar_error_m(vp_x, vp_y, pp_x, pp_y)
    assert max_err < fwd_tol_m, (
        f"{label} transform_buffers forward error {max_err:.6e} m exceeds {fwd_tol_m} m"
    )


@pytest.mark.parametrize(
    "label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg",
    EPSG_SWEEP,
    ids=[case[0] for case in EPSG_SWEEP],
)
def test_transform_buffers_inverse_vs_pyproj_epsg(
    label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg
):
    """transform_buffers inverse path matches pyproj on pyproj-generated inputs."""
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
        pp_x[mask].copy(), pp_y[mask].copy(), direction="INVERSE"
    )

    max_err = _max_angular_error_deg(vp_inv_lon, vp_inv_lat, pp_inv_lon, pp_inv_lat)
    assert max_err < inv_tol_deg, (
        f"{label} transform_buffers inverse error {max_err:.6e} deg exceeds {inv_tol_deg} deg"
    )


@pytest.mark.parametrize(
    "label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg",
    EPSG_SWEEP,
    ids=[case[0] for case in EPSG_SWEEP],
)
def test_transform_chunked_forward_vs_pyproj_epsg(
    label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg
):
    """transform_chunked forward path matches pyproj across the EPSG sweep."""
    lon, lat = make_lon_lat_grid(lat_range, lon_range, n=GRID_N)
    pp = PyProjTransformer.from_crs("EPSG:4326", crs_spec, always_xy=True)
    vp = Transformer.from_crs("EPSG:4326", crs_spec)

    pp_x, pp_y = pp.transform(lon, lat)
    vp_x, vp_y = vp.transform_chunked(lon, lat, chunk_size=CHUNK_SIZE)

    max_err = _max_planar_error_m(vp_x, vp_y, pp_x, pp_y)
    assert max_err < fwd_tol_m, (
        f"{label} transform_chunked forward error {max_err:.6e} m exceeds {fwd_tol_m} m"
    )


@pytest.mark.parametrize(
    "label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg",
    EPSG_SWEEP,
    ids=[case[0] for case in EPSG_SWEEP],
)
def test_transform_chunked_inverse_vs_pyproj_epsg(
    label, crs_spec, lat_range, lon_range, fwd_tol_m, inv_tol_deg
):
    """transform_chunked inverse path matches pyproj on pyproj-generated inputs."""
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
        f"{label} transform_chunked inverse error {max_err:.6e} deg exceeds {inv_tol_deg} deg"
    )


def _proj_to_proj_inputs(src_crs, lat_range, lon_range):
    """Create projected source coordinates for proj_to_proj tests."""
    lon, lat = make_lon_lat_grid(lat_range, lon_range, n=GRID_N)
    pp_src = PyProjTransformer.from_crs("EPSG:4326", src_crs, always_xy=True)
    src_x, src_y = pp_src.transform(lon, lat)
    mask = finite_mask(src_x, src_y)
    assert mask.sum() > 0, f"No finite projected inputs for {src_crs}"
    return src_x[mask], src_y[mask]


def _assert_proj_to_proj_matches(
    api_name,
    result_x,
    result_y,
    exp_x,
    exp_y,
    tol_m,
    label,
    direction,
):
    max_err = _max_planar_error_m(result_x, result_y, exp_x, exp_y)
    assert max_err < tol_m, (
        f"{label} {api_name} {direction} proj_to_proj error {max_err:.6e} m exceeds {tol_m} m"
    )


@pytest.mark.parametrize(
    "label, src_crs, dst_crs, lat_range, lon_range, tol_m",
    UNIT_EQUIVALENT_PAIRS,
    ids=[case[0] for case in UNIT_EQUIVALENT_PAIRS],
)
def test_proj_to_proj_transform_unit_pairs_vs_pyproj(
    label, src_crs, dst_crs, lat_range, lon_range, tol_m
):
    """transform() matches pyproj for projected-to-projected unit-equivalent pairs."""
    src_x, src_y = _proj_to_proj_inputs(src_crs, lat_range, lon_range)
    pp = PyProjTransformer.from_crs(src_crs, dst_crs, always_xy=True)
    vp = Transformer.from_crs(src_crs, dst_crs)

    exp_x, exp_y = pp.transform(src_x, src_y)
    got_x, got_y = vp.transform(src_x, src_y)
    _assert_proj_to_proj_matches("transform", got_x, got_y, exp_x, exp_y, tol_m, label, "forward")

    exp_inv_x, exp_inv_y = pp.transform(exp_x, exp_y, direction=TransformDirection.INVERSE)
    got_inv_x, got_inv_y = vp.transform(exp_x, exp_y, direction="INVERSE")
    _assert_proj_to_proj_matches(
        "transform", got_inv_x, got_inv_y, exp_inv_x, exp_inv_y, tol_m, label, "inverse"
    )


@pytest.mark.parametrize(
    "label, src_crs, dst_crs, lat_range, lon_range, tol_m",
    UNIT_EQUIVALENT_PAIRS,
    ids=[case[0] for case in UNIT_EQUIVALENT_PAIRS],
)
def test_proj_to_proj_transform_buffers_unit_pairs_vs_pyproj(
    label, src_crs, dst_crs, lat_range, lon_range, tol_m
):
    """transform_buffers() matches pyproj for projected-to-projected unit pairs."""
    src_x, src_y = _proj_to_proj_inputs(src_crs, lat_range, lon_range)
    pp = PyProjTransformer.from_crs(src_crs, dst_crs, always_xy=True)
    vp = Transformer.from_crs(src_crs, dst_crs)

    exp_x, exp_y = pp.transform(src_x, src_y)
    got_x, got_y = vp.transform_buffers(src_x.copy(), src_y.copy())
    _assert_proj_to_proj_matches(
        "transform_buffers", got_x, got_y, exp_x, exp_y, tol_m, label, "forward"
    )

    exp_inv_x, exp_inv_y = pp.transform(exp_x, exp_y, direction=TransformDirection.INVERSE)
    got_inv_x, got_inv_y = vp.transform_buffers(
        exp_x.copy(), exp_y.copy(), direction="INVERSE"
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
def test_proj_to_proj_transform_chunked_unit_pairs_vs_pyproj(
    label, src_crs, dst_crs, lat_range, lon_range, tol_m
):
    """transform_chunked() matches pyproj for projected-to-projected unit pairs."""
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
