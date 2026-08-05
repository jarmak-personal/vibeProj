"""Wave 2 CUDA baseline coverage for equal-area and GEOS kernels."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyproj import CRS, Transformer as PyprojTransformer

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device not available", allow_module_level=True)
except Exception as exc:  # pragma: no cover - import-time environment guard
    pytest.skip(f"CUDA device not available: {exc}", allow_module_level=True)

from vibeproj import Transformer  # noqa: E402
import vibeproj.fused_kernels as fused_module  # noqa: E402
from vibeproj.crs import ProjectionParams  # noqa: E402
from vibeproj.ellipsoid import SPHERE, WGS84  # noqa: E402
from vibeproj.pipeline import TransformPipeline  # noqa: E402


_LAEA_CASES = (
    ("oblique", 45.0),
    ("equatorial", 0.0),
    ("north_pole", 90.0),
    ("south_pole", -90.0),
)

_EQUAL_AREA_PROJ4 = {
    "aea": "+proj=aea +lat_1=20 +lat_2=50 +lat_0=30 +lon_0=10",
    "laea": "+proj=laea +lat_0=35 +lon_0=10",
    "eqearth": "+proj=eqearth +lon_0=10",
    "cea": "+proj=cea +lat_ts=30 +lon_0=10",
}

_NONFINITE_WARNING = "Transform produced non-finite values"


def _projected_crs(definition: str, ellipsoid: str = "+datum=WGS84") -> CRS:
    return CRS.from_proj4(f"{definition} {ellipsoid} +units=m +type=crs")


def _transformers(crs: CRS):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        native = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    oracle = PyprojTransformer.from_crs("EPSG:4326", crs, always_xy=True)
    return native, oracle


@pytest.mark.parametrize(("mode", "lat_0"), _LAEA_CASES)
@pytest.mark.parametrize("ellipsoid", ["+datum=WGS84", "+R=6371000"])
def test_laea_all_modes_match_pyproj_and_roundtrip(mode, lat_0, ellipsoid):
    crs = _projected_crs(f"+proj=laea +lat_0={lat_0} +lon_0=10", ellipsoid)
    native, oracle = _transformers(crs)
    lon = np.array([10.0, 15.0, -20.0, 45.0, -60.0])
    lat = np.array([0.0, 20.0, 50.0, -30.0, 75.0])

    gpu_x, gpu_y = native.transform(cp.asarray(lon), cp.asarray(lat))
    expected_x, expected_y = oracle.transform(lon, lat)
    assert_allclose(cp.asnumpy(gpu_x), expected_x, atol=1e-6, rtol=0.0)
    assert_allclose(cp.asnumpy(gpu_y), expected_y, atol=1e-6, rtol=0.0)

    gpu_lon, gpu_lat = native.transform(gpu_x, gpu_y, direction="INVERSE")
    assert_allclose(cp.asnumpy(gpu_lon), lon, atol=1e-10, rtol=0.0)
    assert_allclose(cp.asnumpy(gpu_lat), lat, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize(("mode", "lat_0"), _LAEA_CASES)
def test_laea_inverse_disk_center_and_nonfinite_contract(mode, lat_0):
    crs = _projected_crs(f"+proj=laea +lat_0={lat_0} +lon_0=10")
    native, _ = _transformers(crs)
    computed = native._pipeline.computed

    if mode in ("oblique", "equatorial"):
        boundary = computed["a"] * computed["D"] * 2.0 * computed["Rq"]
    else:
        boundary = computed["a"] * math.sqrt(2.0 * computed["qp"])
    easting = np.array(
        [0.0, boundary * (1.0 - 1e-12), boundary, boundary * (1.0 + 1e-12), np.nan, np.inf]
    )
    northing = np.zeros_like(easting)

    with pytest.warns(UserWarning, match=_NONFINITE_WARNING):
        cpu_lon, cpu_lat = native.transform(easting, northing, direction="INVERSE")
    gpu_lon, gpu_lat = native.transform(
        cp.asarray(easting), cp.asarray(northing), direction="INVERSE"
    )
    gpu_lon = cp.asnumpy(gpu_lon)
    gpu_lat = cp.asnumpy(gpu_lat)

    unique = np.array([0, 1, 3, 4, 5])
    assert_allclose(gpu_lon[unique], cpu_lon[unique], atol=2e-10, rtol=0.0, equal_nan=True)
    assert_allclose(gpu_lat[unique], cpu_lat[unique], atol=1e-8, rtol=0.0, equal_nan=True)
    assert_allclose([gpu_lon[0], gpu_lat[0]], [10.0, lat_0], atol=1e-12, rtol=0.0)
    assert np.all(np.isfinite([gpu_lon[1], gpu_lat[1]]))
    assert np.all(np.isposinf([gpu_lon[3], gpu_lat[3], gpu_lon[5], gpu_lat[5]]))
    assert np.all(np.isnan([gpu_lon[4], gpu_lat[4]]))


@pytest.mark.parametrize(("mode", "lat_0"), _LAEA_CASES)
def test_laea_inverse_exact_normalized_disk_boundary_is_accepted(mode, lat_0):
    crs = _projected_crs(f"+proj=laea +lat_0={lat_0} +lon_0=10")
    native, _ = _transformers(crs)
    computed = dict(native._pipeline.computed)
    computed.update(a=1.0, D=1.0, x0=0.0, y0=0.0, x_unit_to_m=1.0, y_unit_to_m=1.0)
    boundary = (
        2.0 * computed["Rq"]
        if mode in ("oblique", "equatorial")
        else math.sqrt(2.0 * computed["qp"])
    )
    easting = cp.asarray(
        [math.nextafter(boundary, 0.0), boundary, math.nextafter(boundary, math.inf)]
    )
    northing = cp.zeros_like(easting)
    lon, lat = fused_module.fused_transform(
        easting,
        northing,
        projection_name="laea",
        direction="inverse",
        computed=computed,
        src_north_first=False,
        dst_north_first=False,
        xp=cp,
        precision="fp64",
    )
    lon, lat = cp.asnumpy(lon), cp.asnumpy(lat)
    assert np.all(np.isfinite(lon[:2])) and np.all(np.isfinite(lat[:2]))
    assert np.all(np.isposinf([lon[2], lat[2]]))


@pytest.mark.parametrize(
    ("mode", "lat_0", "antipode_lat"),
    [
        ("equatorial", 0.0, 0.0),
        ("north_pole", 90.0, -90.0),
        ("south_pole", -90.0, 90.0),
    ],
)
def test_laea_antipode_and_latitude_domain_sentinels(mode, lat_0, antipode_lat):
    crs = _projected_crs(f"+proj=laea +lat_0={lat_0} +lon_0=10")
    native, oracle = _transformers(crs)
    lon = np.array([-170.0, 10.0, np.nan, np.inf])
    lat = np.array([antipode_lat, 90.0 + 1e-7, 0.0, 0.0])

    gpu_x, gpu_y = native.transform(cp.asarray(lon), cp.asarray(lat))
    expected_x, expected_y = oracle.transform(lon, lat)
    gpu_x, gpu_y = cp.asnumpy(gpu_x), cp.asnumpy(gpu_y)

    assert np.all(np.isposinf([gpu_x[0], gpu_y[0], gpu_x[1], gpu_y[1], gpu_x[3], gpu_y[3]]))
    assert np.all(np.isnan([gpu_x[2], gpu_y[2]]))
    assert np.array_equal(np.isfinite(gpu_x[[0, 2, 3]]), np.isfinite(expected_x[[0, 2, 3]]))
    assert np.array_equal(np.isfinite(gpu_y[[0, 2, 3]]), np.isfinite(expected_y[[0, 2, 3]]))


@pytest.mark.parametrize("sweep", ["x", "y"])
@pytest.mark.parametrize("height", [35_785_831.0, 40_000_000.0])
def test_geos_sweep_and_height_match_pyproj(sweep, height):
    crs = _projected_crs(f"+proj=geos +lat_0=0 +lon_0=-75 +h={height} +sweep={sweep}")
    native, oracle = _transformers(crs)
    lon = np.array([-75.0, -70.0, -90.0, -40.0, -120.0])
    lat = np.array([0.0, 20.0, -30.0, 10.0, 35.0])

    gpu_x, gpu_y = native.transform(cp.asarray(lon), cp.asarray(lat))
    expected_x, expected_y = oracle.transform(lon, lat)
    assert_allclose(cp.asnumpy(gpu_x), expected_x, atol=2e-6, rtol=0.0)
    assert_allclose(cp.asnumpy(gpu_y), expected_y, atol=2e-6, rtol=0.0)

    gpu_lon, gpu_lat = native.transform(gpu_x, gpu_y, direction="INVERSE")
    assert_allclose(cp.asnumpy(gpu_lon), lon, atol=3e-10, rtol=0.0)
    assert_allclose(cp.asnumpy(gpu_lat), lat, atol=3e-10, rtol=0.0)


@pytest.mark.parametrize("sweep", ["x", "y"])
def test_geos_visibility_inverse_disk_and_scan_domain_sentinels(sweep):
    height = 35_785_831.0
    crs = _projected_crs(f"+proj=geos +lat_0=0 +lon_0=0 +h={height} +sweep={sweep}")
    native, oracle = _transformers(crs)
    computed = native._pipeline.computed

    # Equatorial limb: a/H = cos(lambda), and the inverse discriminant is zero
    # at scan angle acos(sqrt(H^2-a^2)/H).
    lon_limb = math.degrees(math.acos(computed["a"] / computed["H"]))
    lon = np.array(
        [math.nextafter(lon_limb, 0.0), lon_limb, math.nextafter(lon_limb, math.inf), 180.0]
    )
    lat = np.zeros_like(lon)
    gpu_x, gpu_y = native.transform(cp.asarray(lon), cp.asarray(lat))
    expected_x, expected_y = oracle.transform(lon, lat)
    assert np.array_equal(np.isfinite(cp.asnumpy(gpu_x)), np.isfinite(expected_x))
    assert np.array_equal(np.isfinite(cp.asnumpy(gpu_y)), np.isfinite(expected_y))

    scan_limit = math.acos(math.sqrt(computed["H"] ** 2 - computed["a"] ** 2) / computed["H"])
    disk_x = computed["h"] * scan_limit
    half_pi_x = computed["h"] * (math.pi / 2.0)
    easting = np.array(
        [
            math.nextafter(disk_x, 0.0),
            disk_x,
            math.nextafter(disk_x, math.inf),
            disk_x * (1.0 - 1e-12),
            disk_x * (1.0 + 1e-12),
            half_pi_x,
            math.nextafter(half_pi_x, math.inf),
            computed["h"] * (math.pi + 0.01),
            np.nan,
            np.inf,
        ]
    )
    northing = np.zeros_like(easting)
    with (
        np.errstate(invalid="ignore"),
        pytest.warns(UserWarning, match=_NONFINITE_WARNING),
    ):
        cpu_lon, cpu_lat = native.transform(easting, northing, direction="INVERSE")
    gpu_lon, gpu_lat = native.transform(
        cp.asarray(easting), cp.asarray(northing), direction="INVERSE"
    )
    gpu_lon, gpu_lat = cp.asnumpy(gpu_lon), cp.asnumpy(gpu_lat)
    assert_allclose(gpu_lon, cpu_lon, atol=1e-7, rtol=0.0, equal_nan=True)
    assert_allclose(gpu_lat, cpu_lat, atol=1e-7, rtol=0.0, equal_nan=True)
    assert np.all(np.isfinite(gpu_lon[:4])) and np.all(np.isfinite(gpu_lat[:4]))
    assert np.all(np.isposinf(gpu_lon[4:8])) and np.all(np.isposinf(gpu_lat[4:8]))
    assert np.all(np.isnan([gpu_lon[8], gpu_lat[8]]))
    assert np.all(np.isposinf([gpu_lon[9], gpu_lat[9]]))

    fp32_easting = cp.asarray([disk_x, disk_x * (1.0 + 1e-5)])
    fp32_lon, fp32_lat = native.transform_buffers(
        fp32_easting,
        cp.zeros_like(fp32_easting),
        direction="INVERSE",
        precision="fp32",
    )
    assert bool(cp.all(cp.isfinite(cp.stack((fp32_lon[0], fp32_lat[0])))))
    assert bool(cp.all(cp.isinf(cp.stack((fp32_lon[1], fp32_lat[1])))))


@pytest.mark.parametrize("projection", ["aea", "laea", "eqearth", "cea"])
@pytest.mark.parametrize("ellipsoid", ["+datum=WGS84", "+R=6371000"])
def test_shared_equal_area_helpers_match_cpu_and_pyproj(projection, ellipsoid):
    crs = _projected_crs(_EQUAL_AREA_PROJ4[projection], ellipsoid)
    native, oracle = _transformers(crs)
    lon = np.array([-120.0, -80.0, -20.0, 10.0, 40.0, 90.0, 130.0])
    lat = np.array([-90.0, -80.0, -30.0, 0.0, 35.0, 80.0, 90.0])

    cpu_x, cpu_y = native.transform(lon, lat)
    gpu_x, gpu_y = native.transform(cp.asarray(lon), cp.asarray(lat))
    expected_x, expected_y = oracle.transform(lon, lat)
    assert_allclose(cp.asnumpy(gpu_x), cpu_x, atol=1e-6, rtol=0.0)
    assert_allclose(cp.asnumpy(gpu_y), cpu_y, atol=1e-6, rtol=0.0)
    ordinary = np.abs(lat) < 90.0
    assert_allclose(cp.asnumpy(gpu_x)[ordinary], expected_x[ordinary], atol=1e-5, rtol=0.0)
    assert_allclose(cp.asnumpy(gpu_y)[ordinary], expected_y[ordinary], atol=1e-5, rtol=0.0)
    assert_allclose(cp.asnumpy(gpu_x)[~ordinary], cpu_x[~ordinary], atol=1e-7, rtol=0.0)
    assert_allclose(cp.asnumpy(gpu_y)[~ordinary], cpu_y[~ordinary], atol=1e-7, rtol=0.0)

    bad_lon = np.array([np.nan, 0.0, np.inf, 0.0])
    bad_lat = np.array([0.0, np.nan, 0.0, np.inf])
    with (
        np.errstate(invalid="ignore"),
        pytest.warns(UserWarning, match=_NONFINITE_WARNING),
    ):
        cpu_x, cpu_y = native.transform(bad_lon, bad_lat)
    gpu_x, gpu_y = native.transform(cp.asarray(bad_lon), cp.asarray(bad_lat))
    assert_allclose(cp.asnumpy(gpu_x), cpu_x, equal_nan=True)
    assert_allclose(cp.asnumpy(gpu_y), cpu_y, equal_nan=True)


@pytest.mark.parametrize("projection", ["aea", "cea"])
@pytest.mark.parametrize("pole_sign", [-1.0, 1.0])
@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_equal_area_inverse_q_domain_below_at_above_limit(projection, pole_sign, precision):
    definition = _EQUAL_AREA_PROJ4[projection]
    crs = _projected_crs(definition)
    native, forward_oracle = _transformers(crs)
    inverse_oracle = PyprojTransformer.from_crs(crs, "EPSG:4326", always_xy=True)
    computed = native._pipeline.computed

    boundary_x, boundary_y = forward_oracle.transform(10.0, pole_sign * 90.0)
    if projection == "cea":
        easting = np.full(4, boundary_x)
        northing = np.array([boundary_y * 0.99, boundary_y, boundary_y * 1.01, boundary_y * 2.0])
        outside = np.array([2, 3])
    else:
        q_factors = np.array([0.99, 1.0, 1.01])
        q = pole_sign * computed["qp"] * q_factors
        rho = np.sqrt(computed["C"] - computed["n"] * q) / abs(computed["n"])
        easting = np.full(3, computed["x0"])
        northing = computed["y0"] + computed["a"] * (computed["rho0"] - rho)
        # Use the independently projected pole for the exact boundary sample.
        easting[1], northing[1] = boundary_x, boundary_y
        outside = np.array([2])

    with pytest.warns(UserWarning, match=_NONFINITE_WARNING):
        cpu_lon, cpu_lat = native.transform(easting, northing, direction="INVERSE")
    gpu_lon, gpu_lat = native.transform_buffers(
        cp.asarray(easting),
        cp.asarray(northing),
        direction="INVERSE",
        precision=precision,
    )
    gpu_lon, gpu_lat = cp.asnumpy(gpu_lon), cp.asnumpy(gpu_lat)
    oracle_lon, oracle_lat = inverse_oracle.transform(easting, northing)

    atol = 5e-5 if precision == "fp32" else 1e-9
    assert_allclose(gpu_lon[0], cpu_lon[0], atol=atol, rtol=0.0)
    assert_allclose(gpu_lat[0], cpu_lat[0], atol=atol, rtol=0.0)
    assert_allclose(gpu_lon[0], oracle_lon[0], atol=atol, rtol=0.0)
    assert_allclose(gpu_lat[0], oracle_lat[0], atol=atol, rtol=0.0)

    assert np.all(np.isfinite([gpu_lon[1], gpu_lat[1], cpu_lon[1], cpu_lat[1]]))
    assert_allclose(gpu_lat[1], pole_sign * 90.0, atol=atol, rtol=0.0)
    assert_allclose(cpu_lat[1], pole_sign * 90.0, atol=3e-6, rtol=0.0)
    assert np.all(np.isfinite([oracle_lon[1], oracle_lat[1]]))
    assert_allclose(oracle_lat[1], pole_sign * 90.0, atol=3e-6, rtol=0.0)

    assert np.all(np.isposinf(gpu_lon[outside]))
    assert np.all(np.isposinf(gpu_lat[outside]))
    assert np.all(np.isposinf(cpu_lon[outside]))
    assert np.all(np.isposinf(cpu_lat[outside]))
    assert not np.any(np.isfinite(oracle_lat[outside]))


@pytest.mark.parametrize(
    "definition",
    [
        "+proj=aea +lat_1=30 +lat_2=30.00000001 +lat_0=30 +lon_0=10 "
        "+datum=WGS84 +units=m +type=crs",
        "+proj=cea +lat_ts=30 +lon_0=10 +datum=WGS84 +units=ft +type=crs",
    ],
    ids=["aea-near-degenerate", "cea-non-metre"],
)
@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_equal_area_forward_derived_poles_survive_inverse(definition, precision):
    crs = CRS.from_proj4(definition)
    native, _ = _transformers(crs)
    lon = cp.asarray([10.0, 10.0])
    lat = cp.asarray([-90.0, 90.0])

    projected_x, projected_y = native.transform_buffers(lon, lat, precision=precision)
    roundtrip_lon, roundtrip_lat = native.transform_buffers(
        projected_x,
        projected_y,
        direction="INVERSE",
        precision=precision,
    )

    atol = 5e-5 if precision == "fp32" else 2e-10
    assert bool(cp.all(cp.isfinite(projected_x)))
    assert bool(cp.all(cp.isfinite(projected_y)))
    assert_allclose(cp.asnumpy(roundtrip_lon), cp.asnumpy(lon), atol=atol, rtol=0.0)
    assert_allclose(cp.asnumpy(roundtrip_lat), cp.asnumpy(lat), atol=atol, rtol=0.0)


@pytest.mark.parametrize(
    "definition",
    [
        "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +datum=WGS84 +units=m +type=crs",
        "+proj=aea +lat_1=-29.5 +lat_2=-45.5 +lat_0=-23 +lon_0=-96 +datum=WGS84 +units=m +type=crs",
    ],
    ids=["positive-n", "negative-n"],
)
@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_aea_exact_poles_wrap_longitude_before_cone_angle(definition, precision):
    crs = CRS.from_proj4(definition)
    native, _ = _transformers(crs)
    # Every longitude is equivalent to -190 degrees after one or more full
    # rotations around the nonzero central meridian.
    lon = np.array([170.0, -190.0, 530.0, -550.0] * 2)
    lat = np.repeat(np.array([-90.0, 90.0]), 4)

    cpu_x, cpu_y = native.transform(lon, lat)
    gpu_x, gpu_y = native.transform_buffers(cp.asarray(lon), cp.asarray(lat), precision=precision)

    assert_allclose(cp.asnumpy(gpu_x), cpu_x, atol=2e-7, rtol=0.0)
    assert_allclose(cp.asnumpy(gpu_y), cpu_y, atol=2e-7, rtol=0.0)


@pytest.mark.parametrize("projection", ["aea", "cea"])
@pytest.mark.parametrize("pole_sign", [-1.0, 1.0])
@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_equal_area_inverse_q_pole_snap_band_and_true_outside(projection, pole_sign, precision):
    crs = _projected_crs(_EQUAL_AREA_PROJ4[projection])
    native, _ = _transformers(crs)
    inverse_oracle = PyprojTransformer.from_crs(crs, "EPSG:4326", always_xy=True)
    computed = native._pipeline.computed
    q = pole_sign * (computed["qp"] + np.array([0.5e-10, 2.0e-10]))

    if projection == "cea":
        easting = np.full(2, computed["x0"] / computed["x_unit_to_m"])
        physical_y = computed["y0"] + computed["a"] * 0.5 * q / computed["k0"]
        northing = physical_y / computed["y_unit_to_m"]
    else:
        rho = np.sqrt(computed["C"] - computed["n"] * q) / abs(computed["n"])
        easting = np.full(2, computed["x0"] / computed["x_unit_to_m"])
        physical_y = computed["y0"] + computed["a"] * (computed["rho0"] - rho)
        northing = physical_y / computed["y_unit_to_m"]

    with pytest.warns(UserWarning, match=_NONFINITE_WARNING):
        cpu_lon, cpu_lat = native.transform(easting, northing, direction="INVERSE")
    gpu_lon, gpu_lat = native.transform_buffers(
        cp.asarray(easting),
        cp.asarray(northing),
        direction="INVERSE",
        precision=precision,
    )
    gpu_lon, gpu_lat = cp.asnumpy(gpu_lon), cp.asnumpy(gpu_lat)
    _, oracle_lat = inverse_oracle.transform(easting, northing)

    atol = 5e-5 if precision == "fp32" else 3e-6
    assert np.all(np.isfinite([cpu_lon[0], cpu_lat[0], gpu_lon[0], gpu_lat[0]]))
    assert_allclose(cpu_lat[0], pole_sign * 90.0, atol=atol, rtol=0.0)
    assert_allclose(gpu_lat[0], pole_sign * 90.0, atol=atol, rtol=0.0)
    assert np.all(np.isposinf([cpu_lon[1], cpu_lat[1], gpu_lon[1], gpu_lat[1]]))
    if projection == "cea":
        assert not np.isfinite(oracle_lat[1])
    else:
        # PROJ clips this sub-nanoradian AEA excess to the pole. vibeproj's
        # explicitly bounded 1e-10 q snap rejects it as outside the domain.
        assert_allclose(oracle_lat[1], pole_sign * 90.0, atol=3e-6, rtol=0.0)


def test_equal_area_helper_tiny_e_and_pole_inverse_on_device():
    source = (
        fused_module._NATIVE_PAIRED_SINCOS_DEVICE_FNS
        + fused_module._EA_DEVICE_FNS.format(
            real_t="double", pi=fused_module._PI_LITERALS["float64"], tol="1e-14"
        )
        + r"""
extern "C" __global__ void ea_helper_test(
    const double* s, const double* e, double* q, double* phi, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    q[i] = qsfn(s[i], e[i]);
    phi[i] = phi_from_q(q[i], e[i], e[i] * e[i], qsfn(1.0, e[i]));
}
extern "C" __global__ void ea_fraction_test(
    const double* fraction, double* phi, double e, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double qp = qsfn(1.0, e);
    phi[i] = phi_from_q(fraction[i] * qp, e, e * e, qp);
}
"""
    )
    kernel = cp.RawKernel(source, "ea_helper_test")
    cutoff = 1e-10
    eccentricity = np.array(
        [0.0, math.nextafter(cutoff, 0.0), cutoff, math.nextafter(cutoff, math.inf), 1e-8] * 5
    )
    sin_phi = np.repeat(np.array([-1.0, -0.999999, 0.0, 0.999999, 1.0]), 5)
    d_s = cp.asarray(sin_phi)
    d_e = cp.asarray(eccentricity)
    d_q = cp.empty_like(d_s)
    d_phi = cp.empty_like(d_s)
    kernel(((d_s.size + 255) // 256,), (256,), (d_s, d_e, d_q, d_phi, np.int32(d_s.size)))

    expected_q = np.empty_like(sin_phi)
    zero = eccentricity == 0.0
    expected_q[zero] = 2.0 * sin_phi[zero]
    z = eccentricity[~zero] * sin_phi[~zero]
    expected_q[~zero] = (1.0 - eccentricity[~zero] ** 2) * (
        sin_phi[~zero] / (1.0 - z * z) + np.arctanh(z) / eccentricity[~zero]
    )
    assert_allclose(cp.asnumpy(d_q), expected_q, atol=2e-15, rtol=2e-15)
    assert_allclose(np.sin(cp.asnumpy(d_phi)), sin_phi, atol=3e-15, rtol=0.0)

    fractions = np.array(
        [
            -1.0,
            math.nextafter(-1.0, 0.0),
            -1.0 + 1e-15,
            -0.999999,
            0.0,
            0.999999,
            1.0 - 1e-15,
            math.nextafter(1.0, 0.0),
            1.0,
        ]
    )
    d_fractions = cp.asarray(fractions)
    d_fraction_phi = cp.empty_like(d_fractions)
    fraction_kernel = cp.RawKernel(source, "ea_fraction_test")
    fraction_kernel(
        (1,),
        (32,),
        (d_fractions, d_fraction_phi, np.float64(WGS84.e), np.int32(fractions.size)),
    )
    fraction_phi = cp.asnumpy(d_fraction_phi)
    assert np.all(np.isfinite(fraction_phi))
    assert np.all(np.diff(fraction_phi) >= 0.0)
    assert fraction_phi[0] == -math.pi / 2.0
    assert fraction_phi[-1] == math.pi / 2.0

    # Long-double bisection is an independent monotonic oracle for the hard
    # near-pole q/qp fractions where Newton in phi previously stalled.
    e_ld = np.longdouble(WGS84.e)
    es_ld = e_ld * e_ld

    def q_longdouble(s):
        e_s = e_ld * s
        return (1 - es_ld) * (s / (1 - e_s * e_s) + np.arctanh(e_s) / e_ld)

    qp_ld = q_longdouble(np.longdouble(1.0))
    expected_phi = []
    for fraction in fractions:
        if fraction == -1.0:
            expected_phi.append(-math.pi / 2.0)
            continue
        if fraction == 1.0:
            expected_phi.append(math.pi / 2.0)
            continue
        target = np.longdouble(fraction) * qp_ld
        low, high = np.longdouble(-1.0), np.longdouble(1.0)
        for _ in range(100):
            midpoint = (low + high) / 2
            if q_longdouble(midpoint) < target:
                low = midpoint
            else:
                high = midpoint
        expected_phi.append(float(np.arcsin((low + high) / 2)))
    # At one binary64 q/qp ULP from an endpoint, q=qp*fraction can round back
    # to qp; the resulting <=1.5e-8 rad pole collapse is representational.
    assert_allclose(fraction_phi, expected_phi, atol=1.6e-8, rtol=0.0)
    assert_allclose(fraction_phi[[3, 4, 5]], np.asarray(expected_phi)[[3, 4, 5]], atol=2e-12)


@pytest.mark.parametrize(
    "projection",
    sorted(name for name, direction in fused_module._SUPPORTED if direction == "inverse"),
)
def test_all_fused_inverse_paths_preserve_nonfinite_longitude(projection):
    extra = {"h": 35_785_831.0, "sweep_axis": "x"} if projection == "geos" else {}
    ellipsoid = SPHERE if projection == "aeqd" else WGS84
    projected = ProjectionParams(
        projection_name=projection,
        ellipsoid=ellipsoid,
        lat_0=45.0,
        lat_1=20.0,
        lat_2=50.0,
        k_0=0.9996,
        north_first=False,
        extra=extra,
    )
    geographic = ProjectionParams(projection_name="longlat", ellipsoid=ellipsoid, north_first=False)
    pipeline = TransformPipeline(projected, geographic)
    easting = np.array([np.inf, -np.inf])
    northing = np.zeros_like(easting)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        cpu_lon, _ = pipeline.transform(easting, northing, np)
    gpu_lon, _ = pipeline.transform(cp.asarray(easting), cp.asarray(northing), cp)
    assert_allclose(cp.asnumpy(gpu_lon), cpu_lon, equal_nan=True)


@pytest.mark.parametrize("projection", ["laea", "geos"])
def test_wave2_fused_kernel_honors_preallocated_buffers_and_stream(projection):
    definition = (
        "+proj=laea +lat_0=45 +lon_0=10"
        if projection == "laea"
        else "+proj=geos +lat_0=0 +lon_0=10 +h=40000000 +sweep=x"
    )
    crs = _projected_crs(definition)
    native, _ = _transformers(crs)
    lon = cp.asarray([10.0, 15.0, -20.0])
    lat = cp.asarray([0.0, 20.0, 50.0])
    out_x = cp.empty_like(lon)
    out_y = cp.empty_like(lat)
    stream = cp.cuda.Stream(non_blocking=True)

    got_x, got_y = native.transform_buffers(lon, lat, out_x=out_x, out_y=out_y, stream=stream)
    stream.synchronize()
    assert got_x is out_x
    assert got_y is out_y
    assert bool(cp.all(cp.isfinite(out_x)))
    assert bool(cp.all(cp.isfinite(out_y)))
