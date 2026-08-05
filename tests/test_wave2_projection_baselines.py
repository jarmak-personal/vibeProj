"""CPU baselines and exact-domain contracts for azimuthal and GEOS projections."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pyproj import CRS
from pyproj import Transformer as PyProjTransformer

import vibeproj
import vibeproj.fused_kernels as fused_module
import vibeproj.projections.geostationary as geos_module
from vibeproj import Transformer, UnsupportedProjectionError, VibeProjectionError
from vibeproj.crs import ProjectionParams, resolve_projection_params
from vibeproj.ellipsoid import SPHERE, WGS84, Ellipsoid
from vibeproj.pipeline import TransformPipeline
from vibeproj.projections._equal_area import (
    authalic_q_scalar,
    geodetic_latitude_from_authalic_q,
)
from vibeproj.projections.albers_equal_area import AlbersEqualArea
from vibeproj.projections.azimuthal_equidistant import AzimuthalEquidistant
from vibeproj.projections.cylindrical_equal_area import CylindricalEqualArea
from vibeproj.projections.equal_earth import EqualEarth
from vibeproj.projections.geostationary import Geostationary
from vibeproj.projections.lambert_azimuthal_equal_area import LambertAzimuthalEqualArea
from vibeproj.transcendentals import (
    NATIVE_LIBDEVICE,
    DeviceCapability,
    TranscendentalOperation,
    resolve_transcendental_strategy,
)


CPU = DeviceCapability(backend="cpu", name="test CPU")
ADA = DeviceCapability(
    backend="cuda",
    compute_capability=(8, 9),
    fp32_to_fp64_ratio=64,
    name="test Ada",
)
WGS84_LONG_LAT = "+proj=longlat +ellps=WGS84 +type=crs"
EPS_Q_DOMAIN = 1.0e-10


def _projection_domains(transformer: Transformer, direction: str = "FORWARD") -> list[str]:
    explanation = transformer.explain_strategy(
        transcendentals="native", direction=direction, device=CPU
    )
    return [
        decision.domain
        for decision in explanation.decisions
        if decision.operation is TranscendentalOperation.PROJECTION
    ]


def test_geos_cpu_and_cuda_numeric_contract_constants_are_explicit():
    assert geos_module.GEOS_LIMB_TOLERANCE == 0.0
    assert geos_module.GEOS_DISCRIMINANT_TOLERANCE == 0.0
    assert geos_module.GEOS_SCAN_ANGLE_LIMIT == math.pi / 2.0

    source = fused_module._GEOS_INVERSE_SOURCE
    expected_definitions = (
        "#define VP_GEOS_FP32_DISCRIMINANT_TOLERANCE "
        f"{fused_module.GEOS_FP32_DISCRIMINANT_TOLERANCE:.17g}f",
        "#define VP_GEOS_FP64_DISCRIMINANT_TOLERANCE "
        f"{fused_module.GEOS_FP64_DISCRIMINANT_TOLERANCE:.17g}",
        f"#define VP_GEOS_SCAN_ANGLE_LIMIT {fused_module.GEOS_SCAN_ANGLE_LIMIT:.17g}",
    )
    assert all(definition in source for definition in expected_definitions)
    assert "disc < -discriminant_tolerance" in source
    assert source.count("VP_GEOS_SCAN_ANGLE_LIMIT") >= 3


@pytest.mark.parametrize("geometry", ["spherical", "ellipsoidal"])
@pytest.mark.parametrize(
    ("lat_0", "mode"),
    [(0, "equatorial"), (45, "oblique"), (90, "north_pole"), (-90, "south_pole")],
)
def test_laea_public_domain_is_exact_for_geometry_and_mode(geometry, lat_0, mode):
    earth = "R=6378137" if geometry == "spherical" else "ellps=WGS84"
    target = CRS.from_user_input(f"+proj=laea +lat_0={lat_0} +lon_0=0 +{earth} +units=m +type=crs")
    transformer = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)

    assert _projection_domains(transformer) == [f"laea.forward.{geometry}.{mode}"]
    assert _projection_domains(transformer, "INVERSE") == [f"laea.inverse.{geometry}.{mode}"]


@pytest.mark.parametrize(
    ("target_input", "expected"),
    [
        (32661, "stere.forward.ellipsoidal.variant_a.north"),
        (32761, "stere.forward.ellipsoidal.variant_a.south"),
        (5041, "stere.forward.ellipsoidal.variant_a.north"),
        (5042, "stere.forward.ellipsoidal.variant_a.south"),
        (3413, "stere.forward.ellipsoidal.variant_b.north"),
        (3031, "stere.forward.ellipsoidal.variant_b.south"),
        (2985, "stere.forward.ellipsoidal.variant_c.south"),
        (
            "+proj=stere +lat_0=90 +k_0=0.994 +lon_0=37 +R=6378137 +units=m +type=crs",
            "stere.forward.spherical.variant_a.north",
        ),
        (
            "+proj=stere +lat_0=-90 +k_0=0.994 +lon_0=37 +R=6378137 +units=m +type=crs",
            "stere.forward.spherical.variant_a.south",
        ),
        (
            "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=37 +R=6378137 +units=m +type=crs",
            "stere.forward.spherical.variant_b.north",
        ),
        (
            "+proj=stere +lat_0=-90 +lat_ts=-70 +lon_0=37 +R=6378137 +units=m +type=crs",
            "stere.forward.spherical.variant_b.south",
        ),
        (28992, "sterea.forward.ellipsoidal.oblique"),
        (
            "+proj=sterea +lat_0=-37 +lon_0=143 +k=0.9987 +R=6378137 +units=m +type=crs",
            "sterea.forward.spherical.oblique",
        ),
    ],
)
def test_stereographic_method_identity_reaches_public_domain(target_input, expected):
    target = CRS.from_user_input(target_input)
    params = resolve_projection_params(target)
    transformer = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)

    assert params.operation_method == target.coordinate_operation.method_name
    assert _projection_domains(transformer) == [expected]
    assert _projection_domains(transformer, "INVERSE") == [
        expected.replace(".forward.", ".inverse.")
    ]


@pytest.mark.parametrize("sweep", ["x", "y"])
def test_geos_crs_height_sweep_and_public_domains(sweep):
    target = CRS.from_user_input(
        f"+proj=geos +lon_0=-75 +h=40000000 +sweep={sweep} +ellps=WGS84 +units=m +type=crs"
    )
    params = resolve_projection_params(target)
    transformer = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)

    assert params.extra == {"h": 40_000_000.0, "sweep_axis": sweep}
    assert _projection_domains(transformer) == [f"geos.forward.ellipsoidal.sweep_{sweep}"]
    assert _projection_domains(transformer, "INVERSE") == [
        f"geos.inverse.ellipsoidal.sweep_{sweep}"
    ]


@pytest.mark.parametrize(
    ("epsg", "message"),
    [(27701, "Ellipsoidal"), (3295, "Modified"), (3993, "Guam")],
)
def test_public_aeqd_rejects_semantics_not_implemented(epsg, message):
    target = CRS.from_epsg(epsg)
    with pytest.raises(UnsupportedProjectionError, match=message) as raised:
        Transformer.from_crs(target.geodetic_crs, target, always_xy=True)
    assert isinstance(raised.value, VibeProjectionError)
    assert "+R" in str(raised.value)


def test_list_projections_advertises_only_explicit_spherical_aeqd_method():
    methods = vibeproj.list_projections()["aeqd"]["methods"]
    assert methods == ["Azimuthal Equidistant (Spherical)"]
    assert "Modified Azimuthal Equidistant" not in methods
    assert "Guam Projection" not in methods


def test_unknown_projection_error_does_not_advertise_unsupported_aeqd_methods():
    with pytest.raises(UnsupportedProjectionError) as raised:
        Transformer.from_crs("EPSG:4326", "EPSG:5880")
    message = str(raised.value)
    assert "Modified Azimuthal Equidistant" not in message
    assert "Guam Projection" not in message
    assert "'Azimuthal Equidistant'" not in message


def test_spherical_aeqd_remains_supported_and_unqualified():
    target = CRS.from_user_input("+proj=aeqd +lat_0=45 +lon_0=0 +R=6378137 +units=m +type=crs")
    transformer = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)
    explanation = transformer.explain_strategy(transcendentals="accelerated", device=ADA)

    assert explanation.decisions[0].domain == "aeqd.forward.spherical.oblique"
    assert explanation.decisions[0].implementation_id == NATIVE_LIBDEVICE
    assert explanation.decisions[0].fallback is True
    assert _projection_domains(transformer, "INVERSE") == ["aeqd.inverse.spherical.oblique"]
    x, y = transformer.transform(np.array([0.0, 10.0]), np.array([45.0, 50.0]))
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))


def test_pipeline_explain_and_compile_resolve_the_same_exact_domains(monkeypatch):
    target = CRS.from_user_input("+proj=ortho +lat_0=45 +lon_0=0 +R=6378137 +units=m +type=crs")
    transformer = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)
    expected = {
        "ortho.forward.spherical.oblique",
        "ortho.inverse.spherical.oblique",
    }
    pipeline_domains = {
        decision.domain
        for direction in ("FORWARD", "INVERSE")
        for decision in transformer._pipeline_for_direction(direction)
        .build_execution_context(
            precision="fp64",
            transcendentals="native",
            device=CPU,
            workload_size=None,
        )
        .decisions
    }
    explain_domains = {
        *_projection_domains(transformer),
        *_projection_domains(transformer, "INVERSE"),
    }

    observed: list[str] = []
    original = resolve_transcendental_strategy

    def capture(*args, **kwargs):
        observed.append(kwargs["domain"])
        return original(*args, **kwargs)

    monkeypatch.setattr("vibeproj.transcendentals.resolve_transcendental_strategy", capture)
    monkeypatch.setattr("vibeproj.transcendentals.detect_device_capability", lambda: CPU)
    monkeypatch.setattr("vibeproj.fused_kernels.compile_kernels", lambda **kwargs: None)
    transformer.compile(precision="fp64", transcendentals="native")

    assert pipeline_domains == explain_domains == set(observed) == expected


def test_ortho_warm_up_enumerates_only_the_qualified_exact_forward_domain(monkeypatch):
    observed: list[str] = []
    original = resolve_transcendental_strategy

    def capture(*args, **kwargs):
        observed.append(kwargs["domain"])
        return original(*args, **kwargs)

    monkeypatch.setattr("vibeproj.transcendentals.resolve_transcendental_strategy", capture)
    monkeypatch.setattr("vibeproj.transcendentals.detect_device_capability", lambda: CPU)
    monkeypatch.setattr("vibeproj.fused_kernels.compile_kernels", lambda **kwargs: None)
    vibeproj.warm_up(["ortho"], precision="fp64", transcendentals="native")

    assert "ortho.forward.spherical.oblique" in observed
    assert "ortho.forward" not in observed


@pytest.mark.parametrize("earth", ["R=6378137", "ellps=WGS84"])
@pytest.mark.parametrize("lat_0", [0, 45, 90, -90])
def test_laea_cpu_matches_proj_and_roundtrips_all_modes(earth, lat_0):
    target = CRS.from_user_input(f"+proj=laea +lat_0={lat_0} +lon_0=12 +{earth} +units=m +type=crs")
    source = target.geodetic_crs
    ours = Transformer.from_crs(source, target, always_xy=True)
    oracle = PyProjTransformer.from_crs(source, target, always_xy=True)
    lon = np.array([-45.0, 0.0, 12.0, 30.0, 70.0])
    lat = np.array([-65.0, -20.0, 5.0, 35.0, 70.0])

    actual_x, actual_y = ours.transform(lon, lat)
    expected_x, expected_y = oracle.transform(lon, lat)
    np.testing.assert_allclose(actual_x, expected_x, atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(actual_y, expected_y, atol=1e-5, rtol=0.0)

    roundtrip_lon, roundtrip_lat = ours.transform(expected_x, expected_y, direction="INVERSE")
    np.testing.assert_allclose(roundtrip_lon, lon, atol=5e-8, rtol=0.0)
    np.testing.assert_allclose(roundtrip_lat, lat, atol=5e-8, rtol=0.0)


@pytest.mark.parametrize(
    ("lat_0", "lam", "phi"),
    [
        (0, math.pi, 0.0),
        (45, math.pi, -math.pi / 4),
        (90, 0.0, -math.pi / 2),
        (-90, 0.0, math.pi / 2),
    ],
)
def test_laea_exact_antipode_is_atomic_inf_without_runtime_warning(lat_0, lam, phi):
    projection = LambertAzimuthalEqualArea()
    params = ProjectionParams("laea", SPHERE, lat_0=lat_0)
    computed = projection.setup(params)

    with np.errstate(all="raise"):
        x, y = projection.forward(np.array([lam]), np.array([phi]), params, computed, np)

    assert np.isposinf(x[0])
    assert np.isposinf(y[0])


@pytest.mark.parametrize("lat_0", [0, 45, 90, -90])
def test_laea_inverse_disk_boundary_is_exact(lat_0):
    projection = LambertAzimuthalEqualArea()
    params = ProjectionParams("laea", WGS84, lat_0=lat_0)
    computed = projection.setup(params)
    if computed["mode"] in ("north_pole", "south_pole"):
        radius = math.sqrt(2.0 * computed["qp"])
    else:
        radius = 2.0 * computed["Rq"] * computed["D"]
    x = np.array([np.nextafter(radius, 0.0), radius, np.nextafter(radius, math.inf)])
    lam, phi = projection.inverse(x, np.zeros(3), params, computed, np)

    assert np.all(np.isfinite(lam[:2]))
    assert np.all(np.isfinite(phi[:2]))
    assert np.isposinf(lam[2])
    assert np.isposinf(phi[2])


def test_laea_q_tiny_e_and_inverse_endpoints_are_stable():
    for e in (0.0, 1e-15, np.nextafter(1e-10, 0.0), np.nextafter(1e-10, math.inf)):
        assert authalic_q_scalar(0.7, e) == pytest.approx(1.4, abs=2e-15)

    e = WGS84.e
    qp = authalic_q_scalar(1.0, e)
    values = geodetic_latitude_from_authalic_q(np.array([-qp, 0.0, qp]), qp, e, WGS84.es, np)
    np.testing.assert_allclose(values, [-math.pi / 2, 0.0, math.pi / 2], atol=0.0)

    off_domain = geodetic_latitude_from_authalic_q(
        np.array([np.nextafter(-qp, -math.inf), np.nan, np.nextafter(qp, math.inf)]),
        qp,
        e,
        WGS84.es,
        np,
    )
    assert np.isposinf(off_domain[0])
    assert np.isnan(off_domain[1])
    assert np.isposinf(off_domain[2])


def test_phi_from_q_device_namespace_never_reads_reduction_scalar():
    class DeviceNamespaceProxy:
        __name__ = "cupy"

        def __getattr__(self, name):
            return getattr(np, name)

        def all(self, *args, **kwargs):
            raise AssertionError("device reduction must not be read by Python")

    proxy = DeviceNamespaceProxy()
    qp = authalic_q_scalar(1.0, WGS84.e)
    q = np.linspace(-0.99, 0.99, 101) * qp

    actual = geodetic_latitude_from_authalic_q(q, qp, WGS84.e, WGS84.es, proxy)
    expected = geodetic_latitude_from_authalic_q(q, qp, WGS84.e, WGS84.es, np)

    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0.0)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_aea_inverse_q_boundary_accepts_pole_and_rejects_true_outside(sign):
    projection = AlbersEqualArea()
    params = ProjectionParams("aea", WGS84, lon_0=-96.0, lat_0=23.0, lat_1=29.5, lat_2=45.5)
    computed = projection.setup(params)
    qp = computed["qp"]
    q = sign * np.array([qp - 2.0 * EPS_Q_DOMAIN, qp, qp + 2.0 * EPS_Q_DOMAIN])
    rho = np.sqrt(computed["C"] - computed["n"] * q) / abs(computed["n"])
    x = np.zeros(3)
    y = computed["rho0"] - rho

    lon, lat = projection.inverse(x, y, params, computed, np)

    assert np.all(np.isfinite(lon[:2]))
    assert np.all(np.isfinite(lat[:2]))
    assert lat[1] == sign * math.pi / 2.0
    assert np.isposinf(lon[2])
    assert np.isposinf(lat[2])

    oracle = PyProjTransformer.from_crs(
        CRS.from_user_input("+proj=longlat +datum=WGS84 +type=crs"),
        CRS.from_user_input(
            "+proj=aea +lon_0=-96 +lat_0=23 +lat_1=29.5 +lat_2=45.5 +datum=WGS84 +units=m +type=crs"
        ),
        always_xy=True,
    )
    oracle_q = sign * (qp + 1.0e-6)
    oracle_rho = math.sqrt(computed["C"] - computed["n"] * oracle_q) / abs(computed["n"])
    oracle_y = computed["rho0"] - oracle_rho
    oracle_lon, oracle_lat = oracle.transform(0.0, oracle_y * computed["a"], direction="INVERSE")
    assert not math.isfinite(oracle_lon)
    assert not math.isfinite(oracle_lat)


@pytest.mark.parametrize("ellipsoid", [SPHERE, WGS84])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_cea_inverse_q_boundary_is_atomic(ellipsoid, sign):
    projection = CylindricalEqualArea()
    params = ProjectionParams("cea", ellipsoid, lat_1=30.0)
    computed = projection.setup(params)
    qp = computed["qp"]
    q = sign * np.array([qp - 2.0 * EPS_Q_DOMAIN, qp, qp + 2.0 * EPS_Q_DOMAIN])
    y = q / (2.0 * computed["k0"])

    lon, lat = projection.inverse(np.zeros(3), y, params, computed, np)

    assert np.all(np.isfinite(lon[:2]))
    assert np.all(np.isfinite(lat[:2]))
    assert lat[1] == sign * math.pi / 2.0
    assert np.isposinf(lon[2])
    assert np.isposinf(lat[2])


def test_cea_non_metre_forward_poles_remain_inverse_valid():
    target = CRS.from_user_input("+proj=cea +lat_ts=30 +datum=WGS84 +units=us-ft +type=crs")
    transformer = Transformer.from_crs(target.geodetic_crs, target, always_xy=True)
    longitude = np.array([0.0, 0.0])
    latitude = np.array([-90.0, 90.0])

    x, y = transformer.transform(longitude, latitude)
    actual_longitude, actual_latitude = transformer.transform(x, y, direction="INVERSE")

    np.testing.assert_allclose(actual_longitude, longitude, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(actual_latitude, latitude, atol=1e-7, rtol=0.0)


def test_equal_earth_sphere_matches_proj_and_roundtrips():
    target = CRS.from_user_input("+proj=eqearth +lon_0=10 +R=6371000 +units=m +type=crs")
    source = target.geodetic_crs
    ours = Transformer.from_crs(source, target, always_xy=True)
    oracle = PyProjTransformer.from_crs(source, target, always_xy=True)
    longitude = np.array([-170.0, -80.0, 10.0, 75.0, 170.0])
    latitude = np.array([-80.0, -35.0, 0.0, 45.0, 80.0])

    actual_x, actual_y = ours.transform(longitude, latitude)
    expected_x, expected_y = oracle.transform(longitude, latitude)
    np.testing.assert_allclose(actual_x, expected_x, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(actual_y, expected_y, atol=1e-8, rtol=0.0)

    roundtrip_longitude, roundtrip_latitude = ours.transform(
        expected_x, expected_y, direction="INVERSE"
    )
    np.testing.assert_allclose(roundtrip_longitude, longitude, atol=5e-13, rtol=0.0)
    np.testing.assert_allclose(roundtrip_latitude, latitude, atol=5e-13, rtol=0.0)


@pytest.mark.parametrize(
    "projection",
    [
        "+proj=aea +lat_0=23 +lat_1=29.5 +lat_2=45.5 +lon_0=0",
        "+proj=laea +lat_0=35 +lon_0=0",
        "+proj=cea +lat_ts=30 +lon_0=0",
        "+proj=eqearth +lon_0=0",
    ],
)
@pytest.mark.parametrize("earth", ["+R=6371000", "+ellps=WGS84"])
def test_shared_authalic_helper_families_match_proj(projection, earth):
    target = CRS.from_user_input(f"{projection} {earth} +units=m +type=crs")
    source = target.geodetic_crs
    ours = Transformer.from_crs(source, target, always_xy=True)
    oracle = PyProjTransformer.from_crs(source, target, always_xy=True)
    longitude = np.array([-45.0, -15.0, 0.0, 25.0, 50.0])
    latitude = np.array([-60.0, -25.0, 0.0, 35.0, 65.0])

    actual_x, actual_y = ours.transform(longitude, latitude)
    expected_x, expected_y = oracle.transform(longitude, latitude)
    np.testing.assert_allclose(actual_x, expected_x, atol=2e-5, rtol=0.0)
    np.testing.assert_allclose(actual_y, expected_y, atol=2e-5, rtol=0.0)

    roundtrip_longitude, roundtrip_latitude = ours.transform(
        expected_x, expected_y, direction="INVERSE"
    )
    np.testing.assert_allclose(roundtrip_longitude, longitude, atol=5e-8, rtol=0.0)
    np.testing.assert_allclose(roundtrip_latitude, latitude, atol=5e-8, rtol=0.0)


def test_equal_earth_sphere_nonfinite_contract_matches_fused_and_is_warning_free():
    projection = EqualEarth()
    params = ProjectionParams("eqearth", SPHERE)
    computed = projection.setup(params)
    first = np.array([np.nan, np.inf, 0.0, 0.0])
    second = np.array([0.0, 0.0, np.nan, np.inf])

    with np.errstate(all="raise"):
        forward_x, forward_y = projection.forward(first, second, params, computed, np)
        inverse_lon, inverse_lat = projection.inverse(first, second, params, computed, np)

    assert np.isnan(forward_x[0]) and forward_y[0] == 0.0
    assert np.isposinf(forward_x[1]) and forward_y[1] == 0.0
    assert np.isnan(forward_x[2]) and np.isnan(forward_y[2])
    assert np.isnan(forward_x[3]) and np.isnan(forward_y[3])

    assert np.isnan(inverse_lon[0]) and inverse_lat[0] == 0.0
    assert np.isposinf(inverse_lon[1]) and inverse_lat[1] == 0.0
    assert np.isnan(inverse_lon[2]) and np.isnan(inverse_lat[2])
    assert np.isnan(inverse_lon[3]) and np.isnan(inverse_lat[3])


def test_aea_public_forward_poles_remain_inverse_valid():
    projected = ProjectionParams("aea", WGS84, lon_0=-96.0, lat_0=23.0, lat_1=29.5, lat_2=45.5)
    geographic = ProjectionParams("longlat", WGS84, north_first=True)
    forward = TransformPipeline(geographic, projected)
    inverse = TransformPipeline(projected, geographic)
    latitude = np.array([-90.0, 90.0])
    longitude = np.array([-96.0, -96.0])

    x, y = forward.transform(latitude, longitude, np)
    actual_latitude, actual_longitude = inverse.transform(x, y, np)

    np.testing.assert_allclose(actual_latitude, latitude, atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(actual_longitude, longitude, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("sweep", ["x", "y"])
@pytest.mark.parametrize("height", [35_786_023.0, 40_000_000.0])
def test_geos_cpu_matches_proj_with_custom_height_and_sweep(sweep, height):
    target = CRS.from_user_input(
        f"+proj=geos +lon_0=-75 +h={height} +sweep={sweep} +ellps=WGS84 +units=m +type=crs"
    )
    source = target.geodetic_crs
    ours = Transformer.from_crs(source, target, always_xy=True)
    oracle = PyProjTransformer.from_crs(source, target, always_xy=True)
    lon = np.array([-130.0, -100.0, -75.0, -45.0, -20.0])
    lat = np.array([-45.0, -10.0, 0.0, 20.0, 45.0])

    actual_x, actual_y = ours.transform(lon, lat)
    expected_x, expected_y = oracle.transform(lon, lat)
    np.testing.assert_allclose(actual_x, expected_x, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(actual_y, expected_y, atol=1e-6, rtol=0.0)
    roundtrip_lon, roundtrip_lat = ours.transform(expected_x, expected_y, direction="INVERSE")
    np.testing.assert_allclose(roundtrip_lon, lon, atol=2e-10, rtol=0.0)
    np.testing.assert_allclose(roundtrip_lat, lat, atol=2e-10, rtol=0.0)


def test_geos_visibility_scan_domain_and_nonfinite_sentinels():
    projection = Geostationary()
    params = ProjectionParams("geos", WGS84, extra={"h": 35_786_023.0, "sweep_axis": "y"})
    computed = projection.setup(params)
    tangent = math.asin(computed["a"] / computed["H"])
    scan = np.array([tangent, 0.2, np.nextafter(math.pi / 2, 0.0), math.pi / 2])
    x = scan * computed["h"] / computed["a"]
    lon, lat = projection.inverse(x, np.zeros_like(x), params, computed, np)

    assert np.all(np.isfinite([lon[0], lat[0]]))
    assert np.all(np.isposinf(lon[1:]))
    assert np.all(np.isposinf(lat[1:]))

    nonfinite = np.array([np.nan, np.inf, -np.inf])
    lon, lat = projection.inverse(nonfinite, np.zeros(3), params, computed, np)
    assert np.isnan(lon[0]) and np.isnan(lat[0])
    assert np.all(np.isposinf(lon[1:]))
    assert np.all(np.isposinf(lat[1:]))


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (ProjectionParams("geos", WGS84, extra={"h": 0.0, "sweep_axis": "x"}), "height"),
        (
            ProjectionParams("geos", WGS84, extra={"h": 35_786_023.0, "sweep_axis": "z"}),
            "sweep axis",
        ),
        (
            ProjectionParams("geos", WGS84, extra={"h": WGS84.a * 1.1e10, "sweep_axis": "x"}),
            "1e10 radii",
        ),
        (
            ProjectionParams(
                "geos",
                Ellipsoid(1.0, 2.0, -1.0, 0.0, 0.0, 0.0),
                extra={"h": 1.0, "sweep_axis": "x"},
            ),
            "polar radius",
        ),
    ],
)
def test_geos_setup_rejects_invalid_geometry(params, message):
    with pytest.raises(ValueError, match=message):
        Geostationary().setup(params)


@pytest.mark.parametrize(
    "params",
    [
        ProjectionParams("laea", Ellipsoid(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ProjectionParams("laea", Ellipsoid(1.0, 1.0, 0.0, math.nan, math.nan, 0.0)),
        ProjectionParams("laea", SPHERE, lat_0=91.0),
    ],
)
def test_laea_setup_rejects_invalid_geometry(params):
    with pytest.raises(ValueError):
        LambertAzimuthalEqualArea().setup(params)


def test_aeqd_setup_guard_also_applies_to_manual_params():
    with pytest.raises(UnsupportedProjectionError, match="Ellipsoidal"):
        AzimuthalEquidistant().setup(ProjectionParams("aeqd", WGS84))
