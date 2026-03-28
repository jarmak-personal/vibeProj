"""Tests for fused NVRTC kernels — validate GPU output matches NumPy xp path.

These tests require CuPy and a CUDA GPU. They are automatically skipped
if CuPy is not installed.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

cp = pytest.importorskip("cupy")

from conftest import GRID_CORNERS  # noqa: E402
from vibeproj import Transformer  # noqa: E402
from vibeproj.fused_kernels import can_fuse  # noqa: E402


# ---------------------------------------------------------------------------
# can_fuse() unit tests
# ---------------------------------------------------------------------------


def test_can_fuse_supported():
    assert can_fuse("tmerc", "forward")
    assert can_fuse("tmerc", "inverse")
    assert can_fuse("webmerc", "forward")
    assert can_fuse("webmerc", "inverse")


def test_can_fuse_all_projections():
    for proj in (
        "tmerc",
        "webmerc",
        "merc",
        "lcc",
        "aea",
        "stere",
        "laea",
        "eqc",
        "sinu",
        "eqearth",
        "cea",
        "ortho",
        "gnom",
        "moll",
        "sterea",
        "geos",
        "robin",
        "wintri",
        "natearth",
        "aeqd",
    ):
        assert can_fuse(proj, "forward"), f"{proj} forward"
        assert can_fuse(proj, "inverse"), f"{proj} inverse"


def test_can_fuse_unsupported():
    assert not can_fuse("unknown", "forward")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_forward_gpu_vs_cpu(crs_from, crs_to, lat, lon, atol=1e-4):
    """Run transform on GPU (fused) and CPU (numpy), compare results."""
    t = Transformer.from_crs(crs_from, crs_to, always_xy=False)

    # CPU path (NumPy)
    lat_np = np.asarray(lat, dtype=np.float64)
    lon_np = np.asarray(lon, dtype=np.float64)
    cpu_x, cpu_y = t.transform(lat_np, lon_np)

    # GPU path (CuPy — should use fused kernel)
    lat_cp = cp.asarray(lat, dtype=cp.float64)
    lon_cp = cp.asarray(lon, dtype=cp.float64)
    gpu_x, gpu_y = t.transform(lat_cp, lon_cp)

    # Compare
    assert_allclose(cp.asnumpy(gpu_x), np.asarray(cpu_x), atol=atol)
    assert_allclose(cp.asnumpy(gpu_y), np.asarray(cpu_y), atol=atol)

    return cpu_x, cpu_y, gpu_x, gpu_y


def _make_grid(min_corner, max_corner, nx=50, ny=50):
    """Create a grid of lat/lon points."""
    x, y = np.meshgrid(
        np.linspace(min_corner[0], max_corner[0], ny, dtype=np.float64),
        np.linspace(min_corner[1], max_corner[1], nx, dtype=np.float64),
    )
    return x.ravel(), y.ravel()


# ---------------------------------------------------------------------------
# Transverse Mercator fused kernel tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("min_corner, max_corner, crs_to", GRID_CORNERS)
def test_tmerc_forward_fused_matches_numpy(min_corner, max_corner, crs_to):
    lat, lon = _make_grid(min_corner, max_corner)
    _run_forward_gpu_vs_cpu("EPSG:4326", crs_to, lat, lon)


@pytest.mark.parametrize("min_corner, max_corner, crs_to", GRID_CORNERS)
def test_tmerc_inverse_fused_matches_numpy(min_corner, max_corner, crs_to):
    """Forward on CPU to get projected coords, then inverse on GPU vs CPU."""
    lat, lon = _make_grid(min_corner, max_corner)

    t = Transformer.from_crs("EPSG:4326", crs_to, always_xy=False)
    # Get projected coordinates via CPU
    proj_x, proj_y = t.transform(np.asarray(lat), np.asarray(lon))

    # Inverse: GPU vs CPU
    t_inv = Transformer.from_crs(crs_to, "EPSG:4326", always_xy=False)

    cpu_lat, cpu_lon = t_inv.transform(np.asarray(proj_x), np.asarray(proj_y))

    gpu_lat, gpu_lon = t_inv.transform(cp.asarray(proj_x), cp.asarray(proj_y))

    assert_allclose(cp.asnumpy(gpu_lat), np.asarray(cpu_lat), atol=1e-7)
    assert_allclose(cp.asnumpy(gpu_lon), np.asarray(cpu_lon), atol=1e-7)


def test_tmerc_fused_roundtrip():
    """Forward then inverse on GPU, check lat/lon matches input."""
    lat = cp.array([48.8566, 40.7128, -33.8587], dtype=cp.float64)
    lon = cp.array([2.3522, -74.006, 151.214], dtype=cp.float64)

    t = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=False)
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")

    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-7)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-7)


# ---------------------------------------------------------------------------
# Web Mercator fused kernel tests
# ---------------------------------------------------------------------------


def test_webmerc_forward_fused_matches_numpy():
    lat = np.array([40.7128, 51.5074, -33.8688, 35.6762])
    lon = np.array([-74.006, -0.1278, 151.2093, 139.6503])
    _run_forward_gpu_vs_cpu("EPSG:4326", "EPSG:3857", lat, lon, atol=0.01)


def test_webmerc_inverse_fused_matches_numpy():
    t = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=False)

    x_np = np.array([-8238310.24, -14226.63])
    y_np = np.array([4970071.58, 6711533.71])
    cpu_a, cpu_b = t.transform(x_np, y_np)

    x_cp = cp.asarray(x_np)
    y_cp = cp.asarray(y_np)
    gpu_a, gpu_b = t.transform(x_cp, y_cp)

    assert_allclose(cp.asnumpy(gpu_a), cpu_a, atol=1e-6)
    assert_allclose(cp.asnumpy(gpu_b), cpu_b, atol=1e-6)


def test_webmerc_fused_roundtrip():
    lat = cp.array([40.0, -30.0, 60.0], dtype=cp.float64)
    lon = cp.array([-100.0, 20.0, 140.0], dtype=cp.float64)

    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=False)
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")

    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-7)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-7)


# ---------------------------------------------------------------------------
# Float32 tests
# ---------------------------------------------------------------------------


def test_tmerc_ds_fp64_equivalent():
    """Double-single (ds) fp32 arithmetic matches fp64 to ~10^-8 meters."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=False)
    rng = np.random.default_rng(42)
    lat = cp.asarray(rng.uniform(45, 55, 10_000), dtype=cp.float64)
    lon = cp.asarray(rng.uniform(0, 6, 10_000), dtype=cp.float64)

    x64, y64 = t.transform_buffers(lat, lon, precision="fp64")
    xds, yds = t.transform_buffers(lat, lon, precision="ds")

    # ds should match fp64 to within ~100 nanometers
    assert_allclose(cp.asnumpy(xds), cp.asnumpy(x64), atol=1e-6)
    assert_allclose(cp.asnumpy(yds), cp.asnumpy(y64), atol=1e-6)


def test_auto_precision_uses_fp64():
    """Auto mode always uses fp64 (trig-dominated — SFU, not ALU bound)."""
    from vibeproj.gpu_detect import select_compute_precision

    assert select_compute_precision() == "fp64"


# ---------------------------------------------------------------------------
# Large array test
# ---------------------------------------------------------------------------


def test_tmerc_fused_large_array():
    """1M coordinates — verify no errors."""
    rng = np.random.default_rng(42)
    lat = cp.asarray(rng.uniform(30, 60, 1_000_000), dtype=cp.float64)
    lon = cp.asarray(rng.uniform(-10, 10, 1_000_000), dtype=cp.float64)

    t = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=False)
    x, y = t.transform(lat, lon)

    assert x.shape == (1_000_000,)
    assert y.shape == (1_000_000,)
    # Sanity: values should be finite
    assert bool(cp.all(cp.isfinite(x)))
    assert bool(cp.all(cp.isfinite(y)))


# ---------------------------------------------------------------------------
# Mercator (ellipsoidal) fused kernel tests
# ---------------------------------------------------------------------------


def test_merc_forward_fused_matches_numpy():
    lat = np.array([40.7128, 51.5074, -33.8688])
    lon = np.array([-74.006, -0.1278, 151.2093])
    _run_forward_gpu_vs_cpu("EPSG:4326", "EPSG:3395", lat, lon, atol=0.01)


def test_merc_fused_roundtrip():
    lat = cp.array([40.0, -30.0, 60.0], dtype=cp.float64)
    lon = cp.array([-74.0, 20.0, 140.0], dtype=cp.float64)
    t = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=False)
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")
    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-7)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-7)


# ---------------------------------------------------------------------------
# Lambert Conformal Conic fused kernel tests
# ---------------------------------------------------------------------------


def test_lcc_forward_fused_matches_numpy():
    lat = np.array([48.8566, 43.6047, 45.764])
    lon = np.array([2.3522, 1.4442, 4.8357])
    _run_forward_gpu_vs_cpu("EPSG:4326", "EPSG:2154", lat, lon, atol=0.01)


def test_lcc_fused_roundtrip():
    lat = cp.array([48.8566, 43.6], dtype=cp.float64)
    lon = cp.array([2.3522, 1.44], dtype=cp.float64)
    t = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=False)
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")
    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-7)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-7)


# ---------------------------------------------------------------------------
# Albers Equal Area fused kernel tests
# ---------------------------------------------------------------------------


def test_aea_forward_fused_matches_numpy():
    lat = np.array([40.0, 35.0, 45.0])
    lon = np.array([-96.0, -100.0, -90.0])
    _run_forward_gpu_vs_cpu("EPSG:4326", "EPSG:5070", lat, lon, atol=0.01)


def test_aea_fused_roundtrip():
    lat = cp.array([40.0, 35.0], dtype=cp.float64)
    lon = cp.array([-96.0, -100.0], dtype=cp.float64)
    t = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=False)
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")
    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-7)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-7)


# ---------------------------------------------------------------------------
# Polar Stereographic fused kernel tests
# ---------------------------------------------------------------------------


def test_stere_forward_fused_matches_numpy():
    lat = np.array([-75.0, -80.0, -70.0])
    lon = np.array([0.0, 45.0, -90.0])
    _run_forward_gpu_vs_cpu("EPSG:4326", "EPSG:3031", lat, lon, atol=0.1)


def test_stere_fused_roundtrip():
    lat = cp.array([-75.0, -80.0], dtype=cp.float64)
    lon = cp.array([30.0, -60.0], dtype=cp.float64)
    t = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=False)
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")
    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-6)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-6)


# ---------------------------------------------------------------------------
# LAEA fused kernel tests
# ---------------------------------------------------------------------------


def test_laea_forward_fused_matches_numpy():
    lat = np.array([52.52, 48.86, 41.39])
    lon = np.array([13.405, 2.35, 2.17])
    _run_forward_gpu_vs_cpu("EPSG:4326", "EPSG:3035", lat, lon, atol=0.01)


def test_laea_fused_roundtrip():
    lat = cp.array([52.52, 48.86], dtype=cp.float64)
    lon = cp.array([13.405, 2.35], dtype=cp.float64)
    t = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=False)
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")
    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-7)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-7)


# ---------------------------------------------------------------------------
# Plate Carrée fused kernel tests
# ---------------------------------------------------------------------------


def test_eqc_forward_fused_matches_numpy():
    lat = np.array([40.0, -30.0, 60.0])
    lon = np.array([-100.0, 20.0, 140.0])
    _run_forward_gpu_vs_cpu("EPSG:4326", "EPSG:4087", lat, lon, atol=0.01)


def test_eqc_fused_roundtrip():
    lat = cp.array([40.0, -30.0], dtype=cp.float64)
    lon = cp.array([-100.0, 20.0], dtype=cp.float64)
    t = Transformer.from_crs("EPSG:4326", "EPSG:4087", always_xy=False)
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")
    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-7)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-7)


# ---------------------------------------------------------------------------
# Equal Earth fused kernel tests
# ---------------------------------------------------------------------------


def test_eqearth_forward_fused_matches_numpy():
    lat = np.array([40.0, -30.0, 60.0])
    lon = np.array([-100.0, 20.0, 140.0])
    _run_forward_gpu_vs_cpu("EPSG:4326", "EPSG:8857", lat, lon, atol=0.1)


def test_eqearth_fused_roundtrip():
    lat = cp.array([40.0, -30.0, 60.0], dtype=cp.float64)
    lon = cp.array([-100.0, 20.0, 140.0], dtype=cp.float64)
    t = Transformer.from_crs("EPSG:4326", "EPSG:8857", always_xy=False)
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")
    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-6)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-6)


# ---------------------------------------------------------------------------
# Cylindrical Equal Area fused kernel tests
# ---------------------------------------------------------------------------


def test_cea_forward_fused_matches_numpy():
    lat = np.array([40.0, -30.0, 60.0])
    lon = np.array([-100.0, 20.0, 140.0])
    _run_forward_gpu_vs_cpu("EPSG:4326", "EPSG:6933", lat, lon, atol=0.1)


def test_cea_fused_roundtrip():
    lat = cp.array([40.0, -30.0, 60.0], dtype=cp.float64)
    lon = cp.array([-100.0, 20.0, 140.0], dtype=cp.float64)
    t = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=False)
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")
    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-6)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-6)


# ---------------------------------------------------------------------------
# Orthographic fused kernel tests (custom CRS — no standard EPSG)
# ---------------------------------------------------------------------------


def test_ortho_fused_roundtrip():
    """Test orthographic via custom CRS definition."""
    from vibeproj.crs import ProjectionParams
    from vibeproj.ellipsoid import WGS84
    from vibeproj.pipeline import TransformPipeline

    params = ProjectionParams(
        projection_name="ortho",
        ellipsoid=WGS84,
        lon_0=0.0,
        lat_0=45.0,
        north_first=False,
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    pipe = TransformPipeline(src, params)

    lat = cp.array([40.0, 50.0, 45.0], dtype=cp.float64)
    lon = cp.array([-5.0, 5.0, 0.0], dtype=cp.float64)
    x, y = pipe.transform(lat, lon, cp)

    inv_pipe = TransformPipeline(params, src)
    lat2, lon2 = inv_pipe.transform(x, y, cp)

    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-7)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-7)


# ---------------------------------------------------------------------------
# Gnomonic fused kernel tests (custom CRS)
# ---------------------------------------------------------------------------


def test_gnom_fused_roundtrip():
    from vibeproj.crs import ProjectionParams
    from vibeproj.ellipsoid import WGS84
    from vibeproj.pipeline import TransformPipeline

    params = ProjectionParams(
        projection_name="gnom",
        ellipsoid=WGS84,
        lon_0=0.0,
        lat_0=45.0,
        north_first=False,
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    pipe = TransformPipeline(src, params)

    lat = cp.array([43.0, 47.0, 45.0], dtype=cp.float64)
    lon = cp.array([-2.0, 2.0, 0.0], dtype=cp.float64)
    x, y = pipe.transform(lat, lon, cp)

    inv_pipe = TransformPipeline(params, src)
    lat2, lon2 = inv_pipe.transform(x, y, cp)

    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-7)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-7)


# ---------------------------------------------------------------------------
# Mollweide fused kernel tests (custom CRS)
# ---------------------------------------------------------------------------


def test_transform_buffers_basic():
    """transform_buffers() fast path — no scalar handling."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=False)
    lat = cp.array([48.8566, 40.7128], dtype=cp.float64)
    lon = cp.array([2.3522, -74.006], dtype=cp.float64)

    # Without pre-allocated output
    x, y = t.transform_buffers(lat, lon)
    assert isinstance(x, cp.ndarray)
    assert x.shape == (2,)

    # With pre-allocated output (zero-copy)
    out_x = cp.empty(2, dtype=cp.float64)
    out_y = cp.empty(2, dtype=cp.float64)
    rx, ry = t.transform_buffers(lat, lon, out_x=out_x, out_y=out_y)

    # Verify same objects returned (no allocation)
    assert rx is out_x
    assert ry is out_y

    # Verify results match
    assert_allclose(cp.asnumpy(out_x), cp.asnumpy(x), atol=1e-10)
    assert_allclose(cp.asnumpy(out_y), cp.asnumpy(y), atol=1e-10)


def test_transform_buffers_roundtrip():
    """transform_buffers forward then inverse with pre-allocated outputs."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=False)
    lat = cp.array([48.8566, 40.7128, -33.8587], dtype=cp.float64)
    lon = cp.array([2.3522, -74.006, 151.214], dtype=cp.float64)

    proj_x = cp.empty(3, dtype=cp.float64)
    proj_y = cp.empty(3, dtype=cp.float64)
    t.transform_buffers(lat, lon, out_x=proj_x, out_y=proj_y)

    lat2 = cp.empty(3, dtype=cp.float64)
    lon2 = cp.empty(3, dtype=cp.float64)
    t.transform_buffers(proj_x, proj_y, direction="INVERSE", out_x=lat2, out_y=lon2)

    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-7)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-7)


def test_transform_buffers_vibeSpatial_pattern():
    """Simulate vibeSpatial's OwnedGeometryArray zero-copy pattern.

    vibeSpatial stores coordinates as separate CuPy float64 x/y arrays.
    This test simulates: pre-allocate output, transform in-place, verify
    no intermediate allocations needed.
    """
    # Simulate a DeviceFamilyGeometryBuffer with 100k points
    n = 100_000
    rng = np.random.default_rng(42)

    # "Device-resident" coordinate buffers (as vibeSpatial would have them)
    device_x = cp.asarray(rng.uniform(30, 60, n), dtype=cp.float64)  # lat
    device_y = cp.asarray(rng.uniform(-10, 10, n), dtype=cp.float64)  # lon

    # Create transformer once (cached by vibeSpatial)
    t = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=False)

    # Pre-allocate output (vibeSpatial creates new FamilyBuffer)
    new_x = cp.empty(n, dtype=cp.float64)
    new_y = cp.empty(n, dtype=cp.float64)

    # Zero-copy transform: device_x/y -> new_x/y, no intermediate allocation
    rx, ry = t.transform_buffers(device_x, device_y, out_x=new_x, out_y=new_y)

    # Verify same buffers returned
    assert rx is new_x
    assert ry is new_y

    # Verify all values are finite
    assert bool(cp.all(cp.isfinite(new_x)))
    assert bool(cp.all(cp.isfinite(new_y)))


def test_moll_fused_roundtrip():
    from vibeproj.crs import ProjectionParams
    from vibeproj.ellipsoid import WGS84
    from vibeproj.pipeline import TransformPipeline

    params = ProjectionParams(
        projection_name="moll",
        ellipsoid=WGS84,
        lon_0=0.0,
        north_first=False,
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    pipe = TransformPipeline(src, params)

    lat = cp.array([40.0, -30.0, 60.0, 0.0], dtype=cp.float64)
    lon = cp.array([-100.0, 20.0, 140.0, 0.0], dtype=cp.float64)
    x, y = pipe.transform(lat, lon, cp)

    inv_pipe = TransformPipeline(params, src)
    lat2, lon2 = inv_pipe.transform(x, y, cp)

    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-6)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-6)


# ---------------------------------------------------------------------------
# Oblique Mercator (Hotine) — EPSG:3168 variant A
# ---------------------------------------------------------------------------


def test_omerc_forward_fused_matches_numpy():
    lat = np.array([4.0, 3.0, 5.5, 2.0])
    lon = np.array([102.25, 101.0, 103.5, 100.0])
    _run_forward_gpu_vs_cpu("EPSG:4326", "EPSG:3168", lat, lon, atol=0.01)


def test_omerc_fused_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:3168", always_xy=False)
    lat = cp.array([4.0, 3.0, 5.5, 2.0], dtype=cp.float64)
    lon = cp.array([102.25, 101.0, 103.5, 100.0], dtype=cp.float64)
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")
    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-6)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-6)


# ---------------------------------------------------------------------------
# Krovak — EPSG:5514 (North Orientated)
# ---------------------------------------------------------------------------


def test_krovak_fused_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:5514", always_xy=False)
    lat = cp.array([50.0, 49.5, 48.0, 50.5], dtype=cp.float64)
    lon = cp.array([14.0, 15.0, 17.0, 12.0], dtype=cp.float64)
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")
    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-6)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-6)


# ---------------------------------------------------------------------------
# Eckert IV (manual pipeline — no standard EPSG)
# ---------------------------------------------------------------------------


def test_eck4_fused_roundtrip():
    from vibeproj.crs import ProjectionParams
    from vibeproj.ellipsoid import WGS84
    from vibeproj.pipeline import TransformPipeline

    params = ProjectionParams(
        projection_name="eck4",
        ellipsoid=WGS84,
        lon_0=0.0,
        north_first=False,
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    pipe = TransformPipeline(src, params)

    lat = cp.array([40.0, -30.0, 60.0, 0.0], dtype=cp.float64)
    lon = cp.array([-100.0, 20.0, 140.0, 0.0], dtype=cp.float64)
    x, y = pipe.transform(lat, lon, cp)

    inv_pipe = TransformPipeline(params, src)
    lat2, lon2 = inv_pipe.transform(x, y, cp)

    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-6)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-6)


# ---------------------------------------------------------------------------
# Eckert VI (manual pipeline — no standard EPSG)
# ---------------------------------------------------------------------------


def test_eck6_fused_roundtrip():
    from vibeproj.crs import ProjectionParams
    from vibeproj.ellipsoid import WGS84
    from vibeproj.pipeline import TransformPipeline

    params = ProjectionParams(
        projection_name="eck6",
        ellipsoid=WGS84,
        lon_0=0.0,
        north_first=False,
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    pipe = TransformPipeline(src, params)

    lat = cp.array([40.0, -30.0, 60.0, 0.0], dtype=cp.float64)
    lon = cp.array([-100.0, 20.0, 140.0, 0.0], dtype=cp.float64)
    x, y = pipe.transform(lat, lon, cp)

    inv_pipe = TransformPipeline(params, src)
    lat2, lon2 = inv_pipe.transform(x, y, cp)

    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-6)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-6)


# ---------------------------------------------------------------------------
# Oblique Stereographic (Netherlands EPSG:28992)
# ---------------------------------------------------------------------------


def test_sterea_forward_fused_matches_numpy():
    lat = np.array([52.37, 51.44, 53.22])
    lon = np.array([4.90, 5.47, 6.57])
    _run_forward_gpu_vs_cpu("EPSG:4326", "EPSG:28992", lat, lon, atol=0.01)


def test_sterea_fused_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=False)
    lat = cp.array([52.15, 52.10], dtype=cp.float64)
    lon = cp.array([5.38, 5.39], dtype=cp.float64)
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")
    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-7)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-7)


# ---------------------------------------------------------------------------
# Custom CRS roundtrip tests (Robin, Wintri, NatEarth, AzEqDist)
# ---------------------------------------------------------------------------


def _custom_roundtrip(proj_name, lat, lon, lat_0=0.0, atol=1e-6):
    from vibeproj.crs import ProjectionParams
    from vibeproj.ellipsoid import WGS84
    from vibeproj.pipeline import TransformPipeline

    params = ProjectionParams(
        projection_name=proj_name, ellipsoid=WGS84, lon_0=0.0, lat_0=lat_0, north_first=False
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    x, y = TransformPipeline(src, params).transform(lat, lon, cp)
    lat2, lon2 = TransformPipeline(params, src).transform(x, y, cp)
    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=atol)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=atol)


def test_robin_fused_roundtrip():
    _custom_roundtrip(
        "robin",
        cp.array([40.0, -30.0, 60.0], dtype=cp.float64),
        cp.array([-100.0, 20.0, 140.0], dtype=cp.float64),
        atol=1e-4,
    )


def test_wintri_fused_roundtrip():
    _custom_roundtrip(
        "wintri",
        cp.array([40.0, -20.0, 50.0], dtype=cp.float64),
        cp.array([-80.0, 30.0, 100.0], dtype=cp.float64),
        atol=0.01,
    )


def test_natearth_fused_roundtrip():
    _custom_roundtrip(
        "natearth",
        cp.array([40.0, -30.0, 60.0], dtype=cp.float64),
        cp.array([-100.0, 20.0, 140.0], dtype=cp.float64),
        atol=1e-7,
    )


def test_aeqd_fused_roundtrip():
    _custom_roundtrip(
        "aeqd",
        cp.array([40.0, 50.0, 45.0], dtype=cp.float64),
        cp.array([-5.0, 5.0, 0.0], dtype=cp.float64),
        lat_0=45.0,
        atol=1e-7,
    )


def test_sinu_fused_roundtrip():
    _custom_roundtrip(
        "sinu",
        cp.array([40.0, -30.0, 60.0, 0.0], dtype=cp.float64),
        cp.array([-100.0, 20.0, 140.0, 0.0], dtype=cp.float64),
        atol=1e-7,
    )


def test_geos_fused_roundtrip():
    """Geostationary roundtrip — use points near sub-satellite point (visible disk)."""
    from vibeproj.crs import ProjectionParams
    from vibeproj.ellipsoid import WGS84
    from vibeproj.pipeline import TransformPipeline

    params = ProjectionParams(
        projection_name="geos",
        ellipsoid=WGS84,
        lon_0=0.0,
        north_first=False,
        extra={"h": 35785831.0},
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    pipe = TransformPipeline(src, params)

    # Points within the visible disk (±~60° from sub-satellite point)
    lat = cp.array([0.0, 20.0, -30.0, 50.0], dtype=cp.float64)
    lon = cp.array([0.0, 10.0, -20.0, 5.0], dtype=cp.float64)
    x, y = pipe.transform(lat, lon, cp)

    inv_pipe = TransformPipeline(params, src)
    lat2, lon2 = inv_pipe.transform(x, y, cp)

    assert_allclose(cp.asnumpy(lat2), cp.asnumpy(lat), atol=1e-6)
    assert_allclose(cp.asnumpy(lon2), cp.asnumpy(lon), atol=1e-6)


# ---------------------------------------------------------------------------
# SVD datum correction kernel
# ---------------------------------------------------------------------------


def test_svd_kernel_compiles():
    from vibeproj.fused_kernels import compile_svd_kernel

    compile_svd_kernel()  # should not raise


def test_svd_kernel_matches_xp():
    """Fused SVD correction kernel matches NumPy xp path."""
    from vibeproj._datum_corrections import apply_svd_correction, get_datum_correction
    from vibeproj.fused_kernels import fused_svd_correction

    correction = get_datum_correction("EPSG:4267", "EPSG:4269")
    assert correction is not None

    # Test points well within CONUS
    lat_np = np.array([35.0, 40.0, 45.0, 30.0, 42.0], dtype=np.float64)
    lon_np = np.array([-100.0, -90.0, -80.0, -110.0, -75.0], dtype=np.float64)

    # NumPy xp path (reference)
    ref_lat, ref_lon = apply_svd_correction(lat_np, lon_np, correction, np)

    # GPU fused kernel
    lat_gpu = cp.asarray(lat_np)
    lon_gpu = cp.asarray(lon_np)
    out_lat, out_lon = fused_svd_correction(lat_gpu, lon_gpu, correction, cp)

    assert_allclose(cp.asnumpy(out_lat), ref_lat, atol=1e-12)
    assert_allclose(cp.asnumpy(out_lon), ref_lon, atol=1e-12)


def test_svd_kernel_negate():
    """Negate flag produces opposite correction."""
    from vibeproj._datum_corrections import get_datum_correction
    from vibeproj.fused_kernels import fused_svd_correction

    correction = get_datum_correction("EPSG:4267", "EPSG:4269")
    lat = cp.array([40.0], dtype=cp.float64)
    lon = cp.array([-90.0], dtype=cp.float64)

    fwd_lat, fwd_lon = fused_svd_correction(lat, lon, correction, cp, negate=False)
    inv_lat, inv_lon = fused_svd_correction(lat, lon, correction, cp, negate=True)

    dlat_fwd = cp.asnumpy(fwd_lat)[0] - 40.0
    dlat_inv = cp.asnumpy(inv_lat)[0] - 40.0
    assert_allclose(dlat_fwd, -dlat_inv, atol=1e-15)

    dlon_fwd = cp.asnumpy(fwd_lon)[0] - (-90.0)
    dlon_inv = cp.asnumpy(inv_lon)[0] - (-90.0)
    assert_allclose(dlon_fwd, -dlon_inv, atol=1e-15)


def test_svd_kernel_preallocated():
    """Pre-allocated output buffers work correctly."""
    from vibeproj._datum_corrections import get_datum_correction
    from vibeproj.fused_kernels import fused_svd_correction

    correction = get_datum_correction("EPSG:4267", "EPSG:4269")
    lat = cp.array([35.0, 40.0, 45.0], dtype=cp.float64)
    lon = cp.array([-100.0, -90.0, -80.0], dtype=cp.float64)
    out_lat = cp.empty(3, dtype=cp.float64)
    out_lon = cp.empty(3, dtype=cp.float64)

    result = fused_svd_correction(lat, lon, correction, cp, out_lat=out_lat, out_lon=out_lon)
    assert result[0] is out_lat
    assert result[1] is out_lon


def test_svd_kernel_end_to_end():
    """NAD27→NAD83 on GPU with SVD correction matches pyproj."""
    import warnings

    from pyproj import Transformer as PT

    warnings.filterwarnings("ignore", message="Best transformation")

    t = Transformer.from_crs("EPSG:4267", "EPSG:4269")
    pt = PT.from_crs("EPSG:4267", "EPSG:4269", always_xy=True)

    # Interior CONUS points (avoid Canada border)
    lons_np = np.array([-100.0, -90.0, -80.0, -110.0, -95.0], dtype=np.float64)
    lats_np = np.array([35.0, 40.0, 38.0, 32.0, 42.0], dtype=np.float64)

    lons_gpu = cp.asarray(lons_np)
    lats_gpu = cp.asarray(lats_np)

    out_x, out_y = t.transform(lons_gpu, lats_gpu)
    ref_x, ref_y = pt.transform(lons_np, lats_np)

    err_m = np.sqrt(
        ((cp.asnumpy(out_y) - ref_y) * 111320) ** 2
        + ((cp.asnumpy(out_x) - ref_x) * 111320 * np.cos(np.radians(lats_np))) ** 2
    )
    assert err_m.max() < 0.05, f"Max error {err_m.max():.4f} m exceeds 5cm threshold"
