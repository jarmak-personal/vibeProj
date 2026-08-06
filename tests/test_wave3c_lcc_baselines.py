"""Wave 3C public-domain and policy baselines for LCC."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyproj import CRS
from pyproj import Transformer as PyProjTransformer

import vibeproj
from vibeproj import Transformer
from vibeproj.transcendentals import (
    LCC_FORWARD_CONFORMAL_REFRAME,
    LCC_FORWARD_CONFORMAL_REFRAME_MIN_ELEMENTS,
    LCC_INVERSE_CONFORMAL_REFRAME,
    LCC_INVERSE_CONFORMAL_REFRAME_MIN_ELEMENTS,
    NATIVE_LIBDEVICE,
    DeviceCapability,
    TranscendentalOperation,
    projection_strategy_domain,
    projection_strategy_domains,
    resolve_transcendental_strategy,
)


ADA = DeviceCapability(
    backend="cuda",
    compute_capability=(8, 9),
    fp32_to_fp64_ratio=64,
    name="test Ada",
)
H100 = DeviceCapability(
    backend="cuda",
    compute_capability=(9, 0),
    fp32_to_fp64_ratio=2,
    name="test H100",
)

LCC_TARGETS = (
    "EPSG:2154",
    "EPSG:3851",
    "EPSG:3112",
    "EPSG:2263",
    "EPSG:2251",
    "+proj=lcc +lat_0=40 +lat_1=40 +lon_0=-96 +R=6371000 +units=m +type=crs",
    "+proj=lcc +lat_0=23 +lat_1=33 +lat_2=45 +lon_0=-96 +R=6371000 +units=m +type=crs",
)


def _computed(target_definition: str) -> dict:
    target = CRS.from_user_input(target_definition)
    return Transformer.from_crs(target.geodetic_crs, target, always_xy=True)._pipeline.computed


@pytest.mark.parametrize("target_definition", LCC_TARGETS)
def test_lcc_public_cpu_forward_inverse_matches_pyproj(target_definition):
    target = CRS.from_user_input(target_definition)
    source = target.geodetic_crs
    random = np.random.default_rng(20260806)
    transformer = Transformer.from_crs(source, target, always_xy=True)
    computed = transformer._pipeline.computed
    longitude_origin = math.degrees(float(computed["lam0"]))
    latitude_origin = float(computed["_strategy_latitude_origin"])
    lon = random.uniform(longitude_origin - 10.0, longitude_origin + 10.0, 1024)
    lat = random.uniform(
        max(-80.0, latitude_origin - 10.0),
        min(80.0, latitude_origin + 10.0),
        1024,
    )
    expected_forward = PyProjTransformer.from_crs(source, target, always_xy=True)
    expected_inverse = PyProjTransformer.from_crs(target, source, always_xy=True)

    expected_x, expected_y = expected_forward.transform(lon, lat)
    actual_x, actual_y = transformer.transform(lon, lat)
    assert_allclose(actual_x, expected_x, rtol=0.0, atol=2e-6)
    assert_allclose(actual_y, expected_y, rtol=0.0, atol=2e-6)

    expected_lon, expected_lat = expected_inverse.transform(expected_x, expected_y)
    actual_lon, actual_lat = transformer.transform(actual_x, actual_y, direction="INVERSE")
    assert_allclose(actual_lon, expected_lon, rtol=0.0, atol=2e-11)
    assert_allclose(actual_lat, expected_lat, rtol=0.0, atol=2e-11)


@pytest.mark.parametrize(
    ("target_definition", "geometry", "variant"),
    [
        (
            "+proj=lcc +lat_0=40 +lat_1=40 +lon_0=-96 +R=6371000 +units=m +type=crs",
            "spherical",
            "1sp",
        ),
        (
            "+proj=lcc +lat_0=23 +lat_1=33 +lat_2=45 +lon_0=-96 +R=6371000 +units=m +type=crs",
            "spherical",
            "2sp",
        ),
        ("EPSG:2154", "ellipsoidal", "2sp"),
        ("EPSG:3851", "ellipsoidal", "2sp"),
    ],
)
def test_lcc_strategy_domains_come_from_numeric_setup(target_definition, geometry, variant):
    computed = _computed(target_definition)
    assert computed["lcc_variant"] == variant
    assert projection_strategy_domain("lcc", "forward", computed) == (
        f"lcc.forward.{geometry}.{variant}.regular_cone"
    )
    assert projection_strategy_domain("lcc", "inverse", computed) == (
        f"lcc.inverse.{geometry}.{variant}"
    )


def test_lcc_registered_domains_are_exact_and_warmup_discoverable():
    forward_domains = projection_strategy_domains("lcc", "forward")
    inverse_domains = projection_strategy_domains("lcc", "inverse")
    assert len(forward_domains) == 12
    assert len(inverse_domains) == 8
    for geometry in ("spherical", "ellipsoidal"):
        for variant in ("1sp", "2sp"):
            forward_prefix = f"lcc.forward.{geometry}.{variant}"
            assert f"{forward_prefix}.regular_cone" in forward_domains
            assert f"{forward_prefix}.near_equator" in forward_domains
            assert f"{forward_prefix}.invalid_setup" in forward_domains
            inverse_prefix = f"lcc.inverse.{geometry}.{variant}"
            assert inverse_prefix in inverse_domains
            assert f"{inverse_prefix}.invalid_setup" in inverse_domains


def test_lcc_forward_near_equator_is_native_but_inverse_remains_qualified():
    computed = _computed("+proj=lcc +lat_0=.1 +lat_1=.1 +lon_0=0 +ellps=WGS84 +units=m +type=crs")
    forward_domain = projection_strategy_domain("lcc", "forward", computed)
    inverse_domain = projection_strategy_domain("lcc", "inverse", computed)
    assert forward_domain == "lcc.forward.ellipsoidal.1sp.near_equator"
    assert inverse_domain == "lcc.inverse.ellipsoidal.1sp"
    assert (
        resolve_transcendental_strategy(
            TranscendentalOperation.PROJECTION,
            "accelerated",
            device=ADA,
            domain=forward_domain,
            precision="fp64",
        ).implementation_id
        == NATIVE_LIBDEVICE
    )
    assert (
        resolve_transcendental_strategy(
            TranscendentalOperation.PROJECTION,
            "accelerated",
            device=ADA,
            domain=inverse_domain,
            precision="fp64",
        ).implementation_id
        == LCC_INVERSE_CONFORMAL_REFRAME
    )


@pytest.mark.parametrize("n", [math.nextafter(0.2, 0.0), math.nextafter(-0.2, 0.0)])
def test_lcc_forward_n_just_inside_boundary_is_native(n):
    computed = _computed("EPSG:2154")
    computed["n"] = n
    assert projection_strategy_domain("lcc", "forward", computed).endswith(".near_equator")


@pytest.mark.parametrize(
    "n", [0.2, math.nextafter(0.2, math.inf), -0.2, math.nextafter(-0.2, -math.inf)]
)
def test_lcc_forward_n_at_or_outside_boundary_is_qualified(n):
    computed = _computed("EPSG:2154")
    computed["n"] = n
    assert projection_strategy_domain("lcc", "forward", computed).endswith(".regular_cone")


def test_lcc_closed_eccentricity_and_scale_boundaries_are_qualified():
    computed = _computed("EPSG:2154")
    computed.update(e=0.1, a=6_400_000.0)
    assert projection_strategy_domain("lcc", "forward", computed).endswith(".regular_cone")
    assert projection_strategy_domain("lcc", "inverse", computed) == "lcc.inverse.ellipsoidal.2sp"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("a", math.nextafter(6_400_000.0, math.inf)),
        ("e", math.nextafter(0.1, math.inf)),
        ("k0", math.nextafter(1.0, 0.0)),
        ("k0", math.nextafter(1.0, math.inf)),
        ("x_unit_to_m", 0.0),
        ("y_unit_to_m", math.inf),
        ("F", math.nan),
        ("rho0", math.inf),
    ],
)
def test_lcc_invalid_or_unqualified_setup_is_native(field, value):
    computed = _computed("EPSG:2154")
    computed[field] = value
    for direction in ("forward", "inverse"):
        domain = projection_strategy_domain("lcc", direction, computed)
        assert domain.endswith(".invalid_setup")
        assert (
            resolve_transcendental_strategy(
                TranscendentalOperation.PROJECTION,
                "accelerated",
                device=ADA,
                domain=domain,
                precision="fp64",
            ).implementation_id
            == NATIVE_LIBDEVICE
        )


@pytest.mark.parametrize(
    ("direction", "minimum", "implementation"),
    [
        ("forward", LCC_FORWARD_CONFORMAL_REFRAME_MIN_ELEMENTS, LCC_FORWARD_CONFORMAL_REFRAME),
        ("inverse", LCC_INVERSE_CONFORMAL_REFRAME_MIN_ELEMENTS, LCC_INVERSE_CONFORMAL_REFRAME),
    ],
)
def test_lcc_policy_threshold_hardware_and_precision(direction, minimum, implementation):
    computed = _computed("EPSG:2154")
    domain = projection_strategy_domain("lcc", direction, computed)
    below = resolve_transcendental_strategy(
        TranscendentalOperation.PROJECTION,
        "auto",
        device=ADA,
        domain=domain,
        precision="fp64",
        workload_size=minimum - 1,
    )
    at = resolve_transcendental_strategy(
        TranscendentalOperation.PROJECTION,
        "auto",
        device=ADA,
        domain=domain,
        precision="fp64",
        workload_size=minimum,
    )
    explicit = resolve_transcendental_strategy(
        TranscendentalOperation.PROJECTION,
        "accelerated",
        device=ADA,
        domain=domain,
        precision="fp64",
        workload_size=1,
    )
    assert below.implementation_id == NATIVE_LIBDEVICE
    assert at.implementation_id == implementation
    assert explicit.implementation_id == implementation
    for device, precision in ((H100, "fp64"), (ADA, "fp32"), (ADA, "ds")):
        assert (
            resolve_transcendental_strategy(
                TranscendentalOperation.PROJECTION,
                "accelerated",
                device=device,
                domain=domain,
                precision=precision,
            ).implementation_id
            == NATIVE_LIBDEVICE
        )


@pytest.mark.parametrize("policy", ["auto", "accelerated"])
def test_lcc_module_warmup_compiles_accelerated_and_native_domain_variants(monkeypatch, policy):
    observed = []
    monkeypatch.setattr("vibeproj.transcendentals.detect_device_capability", lambda: ADA)
    monkeypatch.setattr(
        "vibeproj.fused_kernels.compile_kernels",
        lambda *args, **kwargs: observed.extend(kwargs["projection_variants"]),
    )

    vibeproj.warm_up(["lcc"], precision="fp64", transcendentals=policy)

    assert set(observed) == {
        ("lcc", "forward", NATIVE_LIBDEVICE),
        ("lcc", "forward", LCC_FORWARD_CONFORMAL_REFRAME),
        ("lcc", "inverse", NATIVE_LIBDEVICE),
        ("lcc", "inverse", LCC_INVERSE_CONFORMAL_REFRAME),
    }


def _load_policy_benchmark_module():
    module_name = "vibeproj_wave3c_policy_benchmark"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = Path(__file__).resolve().parents[1] / "benchmarks/bench_transcendental_policy.py"
    specification = importlib.util.spec_from_file_location(module_name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def test_lcc_central_benchmark_covers_domains_mixtures_and_scoped_grid():
    benchmark = _load_policy_benchmark_module()
    core = {
        f"lcc-{direction}-{configuration}"
        for direction in ("forward", "inverse")
        for configuration in (
            "spherical-1sp-north",
            "spherical-2sp-north",
            "ellipsoidal-1sp-north",
            "ellipsoidal-2sp-north",
            "ellipsoidal-2sp-south",
            "ellipsoidal-2sp-zero",
            "ellipsoidal-1sp-near-equator",
        )
    }
    mixtures = {
        f"lcc-forward-ellipsoidal-2sp-mix-{fraction:04d}" for fraction in (0, 1, 10, 100, 500, 1000)
    }
    assert set(benchmark.LCC_CASES) == core | mixtures
    assert set(benchmark.CASE_GROUPS["lcc"]) == core | mixtures
    assert benchmark._requires_default_workload_grid("lcc") is False
    assert benchmark._requires_default_workload_grid("merc") is False
    assert benchmark._requires_default_workload_grid("all") is True

    for index, name in enumerate(benchmark.LCC_CASES):
        case = benchmark._prepare_case(np, name, 128, 20260806 + index)
        specification = benchmark.QUALIFICATION_SPECS[name]
        assert case.name == name
        assert case.family == "lcc"
        assert case.qualification is specification
        assert case.host_x.shape == (128,)
        assert case.host_y.shape == (128,)
        direction = "forward" if name.startswith("lcc-forward-") else "inverse"
        assert specification.direction == direction
        assert specification.domain == projection_strategy_domain(
            "lcc", direction, case.transformer._pipeline_for_direction("FORWARD").computed
        )
        if name == "lcc-forward-ellipsoidal-1sp-near-equator":
            assert specification.implementation_id == NATIVE_LIBDEVICE
            assert name not in benchmark.WORKLOAD_GRID_CASES
            assert benchmark._qualification_workload_sizes((5_000_000,), specification) == (
                5_000_000,
            )
        else:
            expected_id = (
                LCC_FORWARD_CONFORMAL_REFRAME
                if direction == "forward"
                else LCC_INVERSE_CONFORMAL_REFRAME
            )
            expected_minimum = (
                LCC_FORWARD_CONFORMAL_REFRAME_MIN_ELEMENTS
                if direction == "forward"
                else LCC_INVERSE_CONFORMAL_REFRAME_MIN_ELEMENTS
            )
            assert specification.implementation_id == expected_id
            assert specification.min_elements == expected_minimum
            assert name in benchmark.WORKLOAD_GRID_CASES
            assert benchmark._qualification_workload_sizes((5_000_000,), specification) == (
                expected_minimum - 1,
                expected_minimum,
                5_000_000,
            )

    forward_near = benchmark.QUALIFICATION_SPECS["lcc-forward-ellipsoidal-1sp-near-equator"]
    inverse_near = benchmark.QUALIFICATION_SPECS["lcc-inverse-ellipsoidal-1sp-near-equator"]
    assert forward_near.implementation_id == NATIVE_LIBDEVICE
    assert forward_near.domain.endswith(".near_equator")
    assert inverse_near.implementation_id == LCC_INVERSE_CONFORMAL_REFRAME
    assert inverse_near.min_elements == LCC_INVERSE_CONFORMAL_REFRAME_MIN_ELEMENTS
    for name in benchmark.LCC_CASES:
        spec = benchmark.QUALIFICATION_SPECS[name]
        if spec.direction == "forward" and spec.implementation_id != NATIVE_LIBDEVICE:
            assert spec.domain.endswith(".regular_cone")


def test_central_benchmark_native_identity_gate_uses_stable_wall_evidence_only():
    benchmark = _load_policy_benchmark_module()
    assert benchmark._native_identity_noise_pass(
        auto_implementation_ids=[NATIVE_LIBDEVICE],
        native_implementation_ids=[NATIVE_LIBDEVICE],
        wall_speedup=0.99,
        wall_repeat_speedups=[0.96, 1.0, 1.04],
    )
    assert not benchmark._native_identity_noise_pass(
        auto_implementation_ids=[LCC_FORWARD_CONFORMAL_REFRAME],
        native_implementation_ids=[NATIVE_LIBDEVICE],
        wall_speedup=1.2,
        wall_repeat_speedups=[1.2, 1.2, 1.2],
    )


def test_lcc_central_hot_guard_audit_counts_every_controlled_lane_and_warp():
    benchmark = _load_policy_benchmark_module()
    for name in (
        "lcc-forward-ellipsoidal-2sp-north",
        "lcc-forward-ellipsoidal-2sp-south",
        "lcc-inverse-ellipsoidal-2sp-north",
        "lcc-inverse-ellipsoidal-2sp-south",
    ):
        case = benchmark._prepare_case(np, name, 4_096, 20260806)
        computed = case.transformer._pipeline_for_direction("FORWARD").computed
        audit = benchmark._lcc_hot_guard_audit(
            case.qualification.direction, computed, case.host_x, case.host_y
        )
        assert audit["setup_qualified"] is True
        assert audit["qualified_lanes"] == audit["total_lanes"] == 4_096
        assert audit["qualified_warps"] == 128
        assert audit["cold_warps"] == 0
        if case.qualification.direction == "forward":
            assert audit["pole_margin_rad"] > 0.0
            assert audit["theta_margin_rad"] > 0.0
        else:
            assert audit["minimum_radius_ratio"] > 0.0

    near = benchmark._prepare_case(np, "lcc-forward-ellipsoidal-1sp-near-equator", 4_096, 20260806)
    near_audit = benchmark._lcc_hot_guard_audit(
        "forward", near.transformer._pipeline.computed, near.host_x, near.host_y
    )
    assert near_audit["setup_qualified"] is False
    assert near_audit["qualified_lanes"] == 0
