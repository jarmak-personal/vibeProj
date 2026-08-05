"""Exhaustive public transcendental registry and fallback contract."""

from __future__ import annotations

import re
import importlib.util
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vibeproj
from vibeproj import Transformer
from vibeproj.fused_kernels import _SUPPORTED
from vibeproj.transcendentals import (
    AccuracyContract,
    HELMERT_FIXED_Q62,
    HELMERT_FIXED_Q62_MIN_ELEMENTS,
    NATIVE_LIBDEVICE,
    ORTHO_FORWARD_FIXED_Q62,
    ORTHO_FORWARD_FIXED_Q62_MIN_ELEMENTS,
    ORTHO_INVERSE_GUARDED_REFRAME,
    ORTHO_INVERSE_GUARDED_REFRAME_MIN_ELEMENTS,
    PROJECTION_FIXED_Q62_MAX_SCALE_M,
    SINU_FORWARD_FIXED_Q62,
    SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS,
    TMERC_FIXED_Q62,
    TMERC_FIXED_Q62_MIN_ELEMENTS,
    DeviceCapability,
    TranscendentalOperation,
    list_transcendental_strategies,
    resolve_transcendental_strategy,
)


def test_accuracy_contract_preserves_legacy_positional_notes_argument():
    contract = AccuracyContract("reference", 1.0, 2.0, "legacy notes")

    assert contract.reference == "reference"
    assert contract.max_horizontal_error_m == 1.0
    assert contract.max_vertical_error_m == 2.0
    assert contract.notes == "legacy notes"
    assert contract.max_physical_scale_m is None


EXPECTED_FUSED_PATHS = frozenset(
    (family, direction)
    for family in (
        "aea",
        "aeqd",
        "cea",
        "eck4",
        "eck6",
        "eqc",
        "eqearth",
        "geos",
        "gnom",
        "krovak",
        "laea",
        "lcc",
        "merc",
        "moll",
        "natearth",
        "omerc",
        "ortho",
        "robin",
        "sinu",
        "stere",
        "sterea",
        "tmerc",
        "webmerc",
        "wintri",
    )
    for direction in ("forward", "inverse")
)

EXPECTED_REGISTRY_MATRIX = frozenset(
    {
        (
            NATIVE_LIBDEVICE,
            TranscendentalOperation.PROJECTION,
            ("*",),
            (),
            ("auto", "fp64", "fp32", "ds"),
            0,
        ),
        (
            NATIVE_LIBDEVICE,
            TranscendentalOperation.HELMERT,
            ("global",),
            (),
            ("auto", "fp64", "fp32", "ds"),
            0,
        ),
        (
            SINU_FORWARD_FIXED_Q62,
            TranscendentalOperation.PROJECTION,
            ("sinu.forward",),
            ((8, 9),),
            ("auto", "fp64"),
            SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS,
        ),
        (
            ORTHO_FORWARD_FIXED_Q62,
            TranscendentalOperation.PROJECTION,
            ("ortho.forward.spherical.oblique",),
            ((8, 9),),
            ("auto", "fp64"),
            ORTHO_FORWARD_FIXED_Q62_MIN_ELEMENTS,
        ),
        (
            ORTHO_INVERSE_GUARDED_REFRAME,
            TranscendentalOperation.PROJECTION,
            ("ortho.inverse.spherical.equatorial",),
            ((8, 9),),
            ("auto", "fp64"),
            ORTHO_INVERSE_GUARDED_REFRAME_MIN_ELEMENTS,
        ),
        (
            HELMERT_FIXED_Q62,
            TranscendentalOperation.HELMERT,
            ("global",),
            ((8, 9),),
            ("auto", "fp64", "fp32", "ds"),
            HELMERT_FIXED_Q62_MIN_ELEMENTS,
        ),
        (
            NATIVE_LIBDEVICE,
            TranscendentalOperation.TMERC_FORWARD,
            ("global", "utm"),
            (),
            ("auto", "fp64", "fp32", "ds"),
            0,
        ),
        (
            TMERC_FIXED_Q62,
            TranscendentalOperation.TMERC_FORWARD,
            ("utm",),
            ((8, 9),),
            ("auto", "fp64"),
            TMERC_FIXED_Q62_MIN_ELEMENTS,
        ),
    }
)

ADA_4090 = DeviceCapability(
    backend="cuda",
    compute_capability=(8, 9),
    fp32_to_fp64_ratio=64,
    name="contract Ada",
)
HOPPER_H100 = DeviceCapability(
    backend="cuda",
    compute_capability=(9, 0),
    fp32_to_fp64_ratio=2,
    name="contract Hopper",
)
CPU = DeviceCapability(backend="cpu", name="contract CPU")
ORTHO_FORWARD_DOMAIN = "ortho.forward.spherical.oblique"


def _context(family: str, direction: str) -> tuple[TranscendentalOperation, str]:
    if (family, direction) == ("tmerc", "forward"):
        return TranscendentalOperation.TMERC_FORWARD, "utm"
    representative = {
        "aeqd": "spherical.oblique",
        "geos": "ellipsoidal.sweep_y",
        "laea": "ellipsoidal.oblique",
        "ortho": "spherical.oblique",
        "stere": "ellipsoidal.variant_b.north",
        "sterea": "ellipsoidal.oblique",
    }.get(family)
    if representative is not None:
        return (
            TranscendentalOperation.PROJECTION,
            f"{family}.{direction}.{representative}",
        )
    return TranscendentalOperation.PROJECTION, f"{family}.{direction}"


def test_static_inventory_covers_every_fused_family_and_direction():
    """Adding or removing a fused path requires an explicit inventory decision."""
    assert len(EXPECTED_FUSED_PATHS) == 48
    assert frozenset(_SUPPORTED) == EXPECTED_FUSED_PATHS
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs/dev/transcendentals.md").read_text(encoding="utf-8")
    section = text.split("## Complete fused-kernel inventory", maxsplit=1)[1].split(
        "## Coverage matrix", maxsplit=1
    )[0]
    documented_paths = frozenset(
        re.findall(r"^\| `([^`]+)` \| (forward|inverse) \|", section, re.MULTILINE)
    )
    assert documented_paths == EXPECTED_FUSED_PATHS
    assert "| `helmert` | datum shift (forward or inverse pipeline) |" in section


def test_registry_exactly_matches_documented_coverage_matrix():
    registry = list_transcendental_strategies()
    actual = frozenset(
        (
            item.implementation_id,
            item.operation,
            item.domains,
            item.supported_compute_capabilities,
            item.supported_compute_precisions,
            item.min_elements,
        )
        for item in registry
    )
    assert actual == EXPECTED_REGISTRY_MATRIX
    assert isinstance(registry, tuple)
    with pytest.raises(FrozenInstanceError):
        registry[0].family = "mutated"  # type: ignore[misc]


def test_wave1_registry_entries_expose_exact_public_contracts():
    entries = {
        entry.implementation_id: entry
        for entry in list_transcendental_strategies()
        if entry.implementation_id in {SINU_FORWARD_FIXED_Q62, ORTHO_FORWARD_FIXED_Q62}
    }
    assert set(entries) == {SINU_FORWARD_FIXED_Q62, ORTHO_FORWARD_FIXED_Q62}
    expected_thresholds = {
        SINU_FORWARD_FIXED_Q62: SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS,
        ORTHO_FORWARD_FIXED_Q62: ORTHO_FORWARD_FIXED_Q62_MIN_ELEMENTS,
    }
    for implementation_id, entry in entries.items():
        assert entry.operation is TranscendentalOperation.PROJECTION
        assert entry.supported_policies == ("auto", "accelerated")
        assert entry.supported_backends == ("cuda",)
        assert entry.supported_compute_capabilities == ((8, 9),)
        assert entry.min_fp32_to_fp64_ratio == 16
        assert entry.supported_compute_precisions == ("auto", "fp64")
        assert entry.min_elements == expected_thresholds[implementation_id]
        assert entry.accuracy.reference == NATIVE_LIBDEVICE
        assert entry.accuracy.max_horizontal_error_m == 1e-8
        assert entry.accuracy.max_vertical_error_m is None
        assert entry.accuracy.max_physical_scale_m == PROJECTION_FIXED_Q62_MAX_SCALE_M
        assert entry.native_fallback is True
        assert entry.priority == 0


def test_wave1_exact_ids_and_thresholds_are_exported():
    import vibeproj.transcendentals as transcendental_module
    from vibeproj._transcendental_device_fns import (
        PROJECTION_FIXED_Q62_MAX_SCALE_M as DEVICE_MAX_SCALE_M,
    )

    for name in (
        "SINU_FORWARD_FIXED_Q62",
        "SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS",
        "ORTHO_FORWARD_FIXED_Q62",
        "ORTHO_FORWARD_FIXED_Q62_MIN_ELEMENTS",
        "PROJECTION_FIXED_Q62_MAX_SCALE_M",
    ):
        assert name in transcendental_module.__all__
        assert getattr(transcendental_module, name)
    assert PROJECTION_FIXED_Q62_MAX_SCALE_M == DEVICE_MAX_SCALE_M


def test_every_accelerated_registry_domain_resolves_to_its_exact_entry():
    accelerated = tuple(
        entry
        for entry in list_transcendental_strategies()
        if entry.implementation_id != NATIVE_LIBDEVICE
    )
    for entry in accelerated:
        for domain in entry.domains:
            decision = resolve_transcendental_strategy(
                entry.operation,
                "accelerated",
                device=ADA_4090,
                domain=domain,
                precision="fp64",
            )
            assert decision.implementation_id == entry.implementation_id


@pytest.mark.parametrize(
    "call",
    [
        lambda transformer: transformer.transform(object(), object(), precision="half"),
        lambda transformer: transformer.transform_buffers(object(), object(), precision="half"),
        lambda transformer: transformer.transform_chunked(object(), object(), precision="half"),
        lambda transformer: transformer.transform_bounds(0.0, 0.0, 1.0, 1.0, precision="half"),
        lambda transformer: transformer.compile(precision="half"),
        lambda transformer: transformer.explain_strategy(precision="half"),
    ],
)
def test_invalid_precision_fails_before_input_or_device_work(call):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    with pytest.raises(ValueError, match="Invalid precision.*half"):
        call(transformer)


def test_invalid_warm_up_precision_fails_before_device_work():
    with pytest.raises(ValueError, match="Invalid precision.*half"):
        vibeproj.warm_up(precision="half")


def test_resolver_reuses_decisions_but_keys_cache_by_complete_device_context():
    first = resolve_transcendental_strategy(
        TranscendentalOperation.TMERC_FORWARD,
        "accelerated",
        device=ADA_4090,
        domain="utm",
        precision="fp64",
    )
    repeated = resolve_transcendental_strategy(
        TranscendentalOperation.TMERC_FORWARD,
        "accelerated",
        device=ADA_4090,
        domain="utm",
        precision="fp64",
    )
    second_device = DeviceCapability(
        backend="cuda",
        compute_capability=(8, 9),
        fp32_to_fp64_ratio=64,
        name="contract Ada second device",
        device_id=1,
    )
    other = resolve_transcendental_strategy(
        TranscendentalOperation.TMERC_FORWARD,
        "accelerated",
        device=second_device,
        domain="utm",
        precision="fp64",
    )

    assert repeated is first
    assert other is not first
    assert other.device.device_id == 1


@pytest.mark.parametrize(
    "operation,domain,accelerated_id,min_elements",
    [
        (
            TranscendentalOperation.TMERC_FORWARD,
            "utm",
            TMERC_FIXED_Q62,
            TMERC_FIXED_Q62_MIN_ELEMENTS,
        ),
        (
            TranscendentalOperation.HELMERT,
            "global",
            HELMERT_FIXED_Q62,
            HELMERT_FIXED_Q62_MIN_ELEMENTS,
        ),
        (
            TranscendentalOperation.PROJECTION,
            "sinu.forward",
            SINU_FORWARD_FIXED_Q62,
            SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS,
        ),
        (
            TranscendentalOperation.PROJECTION,
            ORTHO_FORWARD_DOMAIN,
            ORTHO_FORWARD_FIXED_Q62,
            ORTHO_FORWARD_FIXED_Q62_MIN_ELEMENTS,
        ),
        (
            TranscendentalOperation.PROJECTION,
            "ortho.inverse.spherical.equatorial",
            ORTHO_INVERSE_GUARDED_REFRAME,
            ORTHO_INVERSE_GUARDED_REFRAME_MIN_ELEMENTS,
        ),
    ],
)
def test_size_aware_auto_crossover_and_explicit_override(
    operation, domain, accelerated_id, min_elements
):
    below = resolve_transcendental_strategy(
        operation,
        "auto",
        device=ADA_4090,
        domain=domain,
        precision="fp64",
        workload_size=min_elements - 1,
    )
    at_crossover = resolve_transcendental_strategy(
        operation,
        "auto",
        device=ADA_4090,
        domain=domain,
        precision="fp64",
        workload_size=min_elements,
    )
    explicit = resolve_transcendental_strategy(
        operation,
        "accelerated",
        device=ADA_4090,
        domain=domain,
        precision="fp64",
        workload_size=32,
    )

    assert below.implementation_id == NATIVE_LIBDEVICE
    assert below.workload_size == min_elements - 1
    assert "below the accelerated crossover" in below.reason
    assert at_crossover.implementation_id == accelerated_id
    assert at_crossover.workload_size == min_elements
    assert explicit.implementation_id == accelerated_id
    assert explicit.workload_size == 32


@pytest.mark.parametrize(
    "crs_to,min_elements,accelerated_id",
    [
        ("EPSG:32631", TMERC_FIXED_Q62_MIN_ELEMENTS, TMERC_FIXED_Q62),
        ("EPSG:4277", HELMERT_FIXED_Q62_MIN_ELEMENTS, HELMERT_FIXED_Q62),
    ],
)
def test_public_explanation_records_size_aware_decisions(crs_to, min_elements, accelerated_id):
    transformer = Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)
    below = transformer.explain_strategy(
        transcendentals="auto",
        precision="fp64",
        device=ADA_4090,
        workload_size=min_elements - 1,
    )
    qualified = transformer.explain_strategy(
        transcendentals="auto",
        precision="fp64",
        device=ADA_4090,
        workload_size=min_elements,
    )

    assert below.workload_size == min_elements - 1
    assert below.decisions[-1].implementation_id == NATIVE_LIBDEVICE
    assert below.decisions[-1].workload_size == min_elements - 1
    assert qualified.workload_size == min_elements
    assert qualified.decisions[-1].implementation_id == accelerated_id
    assert qualified.decisions[-1].workload_size == min_elements


@pytest.mark.parametrize("device", [ADA_4090, HOPPER_H100, CPU])
def test_all_48_fused_paths_resolve_nonempty_decisions_for_every_policy(device):
    for family, direction in EXPECTED_FUSED_PATHS:
        operation, domain = _context(family, direction)
        for policy in ("auto", "native", "accelerated"):
            decision = resolve_transcendental_strategy(
                operation,
                policy,
                device=device,
                domain=domain,
                precision="fp64",
            )
            assert decision.implementation_id
            assert decision.reason
            assert decision.domain == domain
            if policy == "native":
                assert decision.implementation_id == NATIVE_LIBDEVICE
                assert decision.fallback is False
            elif device == ADA_4090 and (family, direction) in {
                ("tmerc", "forward"),
                ("sinu", "forward"),
                ("ortho", "forward"),
            }:
                expected = {
                    "tmerc": TMERC_FIXED_Q62,
                    "sinu": SINU_FORWARD_FIXED_Q62,
                    "ortho": ORTHO_FORWARD_FIXED_Q62,
                }[family]
                assert decision.implementation_id == expected
                assert decision.fallback is False
            else:
                assert decision.implementation_id == NATIVE_LIBDEVICE
                assert decision.fallback is (policy == "accelerated")
                if policy == "accelerated":
                    assert "fell back to native" in decision.reason


def test_public_explanation_covers_all_48_fused_paths_and_policies():
    """No fused stage may disappear from public introspection."""
    transformer = object.__new__(Transformer)
    transformer._helmert = None
    for family, stage_direction in EXPECTED_FUSED_PATHS:
        transformer._pipeline = SimpleNamespace(
            mode=stage_direction,
            projection=SimpleNamespace(name=family),
            computed={"is_utm": family == "tmerc" and stage_direction == "forward"},
        )
        for policy in ("auto", "native", "accelerated"):
            explanation = transformer.explain_strategy(
                transcendentals=policy,
                precision="fp64",
                direction="FORWARD",
                device=HOPPER_H100,
            )
            assert len(explanation.decisions) == 1
            decision = explanation.decisions[0]
            assert decision.implementation_id == NATIVE_LIBDEVICE
            assert decision.reason
            assert decision.fallback is (policy == "accelerated")


@pytest.mark.parametrize("policy", ["auto", "accelerated"])
def test_ada_qualified_helmert_resolves_accelerated(policy):
    decision = resolve_transcendental_strategy(
        TranscendentalOperation.HELMERT,
        policy,
        device=ADA_4090,
        domain="global",
        precision="fp64",
    )
    assert decision.implementation_id == HELMERT_FIXED_Q62
    assert decision.fallback is False
    assert decision.accuracy.reference == NATIVE_LIBDEVICE
    assert decision.accuracy.max_horizontal_error_m == 1e-8
    assert decision.accuracy.max_vertical_error_m == 2e-7


@pytest.mark.parametrize(
    "operation,domain,precision,device",
    [
        (TranscendentalOperation.HELMERT, "global", "fp64", HOPPER_H100),
        (TranscendentalOperation.HELMERT, "global", "fp64", CPU),
        (TranscendentalOperation.TMERC_FORWARD, "utm", "fp64", HOPPER_H100),
        (TranscendentalOperation.TMERC_FORWARD, "utm", "fp64", CPU),
        (TranscendentalOperation.TMERC_FORWARD, "global", "fp64", ADA_4090),
        (TranscendentalOperation.TMERC_FORWARD, "utm", "fp32", ADA_4090),
        (TranscendentalOperation.PROJECTION, "sinu.inverse", "fp64", ADA_4090),
        (
            TranscendentalOperation.PROJECTION,
            "ortho.inverse.spherical.oblique",
            "fp64",
            ADA_4090,
        ),
        (TranscendentalOperation.PROJECTION, "sinu.forward", "fp32", ADA_4090),
        (TranscendentalOperation.PROJECTION, ORTHO_FORWARD_DOMAIN, "ds", ADA_4090),
        (TranscendentalOperation.PROJECTION, "sinu.forward", "fp64", HOPPER_H100),
        (TranscendentalOperation.PROJECTION, ORTHO_FORWARD_DOMAIN, "fp64", CPU),
    ],
)
def test_explicit_accelerated_is_portable_native_fallback(operation, domain, precision, device):
    decision = resolve_transcendental_strategy(
        operation,
        "accelerated",
        device=device,
        domain=domain,
        precision=precision,
    )
    assert decision.implementation_id == NATIVE_LIBDEVICE
    assert decision.fallback is True
    assert decision.reason.startswith("accelerated policy fell back to native:")


def test_auto_hopper_remains_native_without_4090_inference():
    for operation, domain in (
        (TranscendentalOperation.HELMERT, "global"),
        (TranscendentalOperation.TMERC_FORWARD, "utm"),
    ):
        decision = resolve_transcendental_strategy(
            operation,
            "auto",
            device=HOPPER_H100,
            domain=domain,
            precision="fp64",
        )
        assert decision.implementation_id == NATIVE_LIBDEVICE
        assert decision.fallback is False
        assert "not accuracy-qualified" in decision.reason


def test_tmerc_asinh_degree11_operation_contract():
    """The production coefficient sequence stays within its per-operation bound."""
    value = np.linspace(-0.06, 0.06, 100_001, dtype=np.float64)
    value_sq = value * value
    polynomial = np.full_like(value, -63.0 / 2816.0)
    polynomial = polynomial * value_sq + 35.0 / 1152.0
    polynomial = polynomial * value_sq - 5.0 / 112.0
    polynomial = polynomial * value_sq + 3.0 / 40.0
    polynomial = polynomial * value_sq - 1.0 / 6.0
    actual = value * (polynomial * value_sq + 1.0)
    assert float(np.max(np.abs(actual - np.arcsinh(value)))) <= 2e-17


def test_tmerc_atan_identity_operation_contract():
    """Validate the refined-reciprocal correction over the complete fast guard."""
    gaussian_latitude = np.linspace(-np.pi / 2, np.pi / 2, 1001)
    sin_latitude = np.sin(gaussian_latitude)
    cos_latitude = np.cos(gaussian_latitude)
    max_correction = 0.0
    max_error = 0.0
    for longitude_offset in np.linspace(-0.06, 0.06, 1001):
        cos_longitude = np.cos(longitude_offset)
        denominator = cos_longitude * cos_latitude * cos_latitude + sin_latitude * sin_latitude
        reciprocal = (np.float32(1.0) / denominator.astype(np.float32)).astype(np.float64)
        reciprocal *= 2.0 - denominator * reciprocal
        correction = sin_latitude * cos_latitude * (1.0 - cos_longitude) * reciprocal
        correction_sq = correction * correction
        actual = gaussian_latitude + correction * (
            correction_sq * (correction_sq * 0.2 - 1.0 / 3.0) + 1.0
        )
        expected = np.arctan2(
            sin_latitude,
            cos_latitude * cos_longitude,
        )
        max_correction = max(max_correction, float(np.max(np.abs(correction))))
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))

    assert max_correction <= 9.01e-4
    assert max_error <= 2.3e-16


def test_rejected_eck4_inverse_q62_pair_reproduces_edge_amplification():
    root = Path(__file__).resolve().parents[1]
    module_spec = importlib.util.spec_from_file_location(
        "vibeproj_eck4_inverse_rejection",
        root / "benchmarks/eck4_inverse_rejection.py",
    )
    assert module_spec is not None and module_spec.loader is not None
    rejection_module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = rejection_module
    module_spec.loader.exec_module(rejection_module)

    result = rejection_module.reproduce_eck4_inverse_edge()

    assert result["cy_over_c_y"] == -0.999994
    assert result["longitude_rad"] == np.pi
    assert result["research_device_speedup"] == pytest.approx(1.0891340311)
    assert result["research_wall_speedup"] == pytest.approx(1.0891271130)
    assert result["horizontal_error_nm"] == pytest.approx(271.916335, abs=1e-6)
    assert result["horizontal_error_m"] > result["gate_m"]
    assert result["passes_gate"] is False


def _documented_ids(path: Path, heading: str) -> set[str]:
    text = path.read_text(encoding="utf-8")
    section = text.split(heading, maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    return set(re.findall(r"^\| `([^`]+)` \|", section, flags=re.MULTILINE))


def _load_policy_benchmark_module():
    module_name = "vibeproj_bench_transcendental_policy"
    if module_name in sys.modules:
        return sys.modules[module_name]
    root = Path(__file__).resolve().parents[1]
    module_spec = importlib.util.spec_from_file_location(
        module_name,
        root / "benchmarks/bench_transcendental_policy.py",
    )
    assert module_spec is not None and module_spec.loader is not None
    benchmark_module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = benchmark_module
    module_spec.loader.exec_module(benchmark_module)
    return benchmark_module


def test_user_and_developer_coverage_tables_name_exact_registry_ids():
    root = Path(__file__).resolve().parents[1]
    registered_ids = {
        implementation.implementation_id for implementation in list_transcendental_strategies()
    }
    user_ids = _documented_ids(
        root / "docs/user/transcendentals.md", "## Qualified hardware and coverage"
    )
    developer_ids = _documented_ids(root / "docs/dev/transcendentals.md", "## Coverage matrix")
    assert user_ids == registered_ids
    assert developer_ids == registered_ids


def test_documented_auto_thresholds_match_registry_exactly():
    root = Path(__file__).resolve().parents[1]
    expected = {
        HELMERT_FIXED_Q62: HELMERT_FIXED_Q62_MIN_ELEMENTS,
        ORTHO_FORWARD_FIXED_Q62: ORTHO_FORWARD_FIXED_Q62_MIN_ELEMENTS,
        ORTHO_INVERSE_GUARDED_REFRAME: ORTHO_INVERSE_GUARDED_REFRAME_MIN_ELEMENTS,
        SINU_FORWARD_FIXED_Q62: SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS,
        TMERC_FIXED_Q62: TMERC_FIXED_Q62_MIN_ELEMENTS,
    }
    for relative_path, heading in (
        ("docs/user/transcendentals.md", "## Qualified hardware and coverage"),
        ("docs/dev/transcendentals.md", "## Coverage matrix"),
    ):
        text = (root / relative_path).read_text(encoding="utf-8")
        section = text.split(heading, maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
        for implementation_id, min_elements in expected.items():
            row = next(
                line
                for line in section.splitlines()
                if line.startswith(f"| `{implementation_id}` |")
            )
            assert f"| {min_elements:,} |" in row


def test_benchmark_enforced_grid_error_reports_current_default_bounds(monkeypatch, capsys):
    benchmark_module = _load_policy_benchmark_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_transcendental_policy.py",
            "--case",
            "ortho-inverse",
            "--enforce-gates",
            "--workload-sizes",
            "32,5000000",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        benchmark_module.main()

    message = capsys.readouterr().err
    assert "every default workload size from 1 to 5000000" in message


def test_wave1_benchmark_specs_enforce_complete_public_qualification_surface():
    benchmark_module = _load_policy_benchmark_module()
    QUALIFICATION_SPECS = benchmark_module.QUALIFICATION_SPECS
    _prepare_case = benchmark_module._prepare_case
    _qualification_workload_sizes = benchmark_module._qualification_workload_sizes

    expected = {
        "sinu-forward": (SINU_FORWARD_FIXED_Q62, SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS),
        "ortho-forward": (ORTHO_FORWARD_FIXED_Q62, ORTHO_FORWARD_FIXED_Q62_MIN_ELEMENTS),
    }
    for index, (case_name, (implementation_id, min_elements)) in enumerate(expected.items()):
        specification = QUALIFICATION_SPECS[case_name]
        assert specification.implementation_id == implementation_id
        assert specification.min_elements == min_elements
        assert specification.coordinate_contract_m == 1e-8
        assert specification.operation == TranscendentalOperation.PROJECTION.value
        expected_domain = {
            "ortho-forward": ORTHO_FORWARD_DOMAIN,
        }.get(case_name, f"{case_name.removesuffix('-forward')}.forward")
        assert specification.domain == expected_domain
        assert specification.direction == "forward"
        assert specification.max_physical_scale_m == PROJECTION_FIXED_Q62_MAX_SCALE_M
        assert specification.expected_kernel_nodes == 1
        assert _qualification_workload_sizes((32, 5_000_000), specification) == (
            32,
            min_elements - 1,
            min_elements,
            5_000_000,
        )

        case = _prepare_case(np, case_name, 128, 20260805 + index)
        assert np.all(np.isfinite(case.host_x))
        assert np.all(np.isfinite(case.host_y))
        assert np.any(np.isfinite(case.edge_host_x))
        assert np.any(np.isfinite(case.edge_host_y))
        assert np.any(~np.isfinite(case.edge_host_x))
        assert np.any(~np.isfinite(case.edge_host_y))
        assert case.qualification is specification


def test_every_accelerated_registry_id_has_matching_benchmark_contract():
    benchmark_module = _load_policy_benchmark_module()
    specifications = tuple(benchmark_module.QUALIFICATION_SPECS.values())
    accelerated_entries = tuple(
        entry
        for entry in list_transcendental_strategies()
        if entry.implementation_id != NATIVE_LIBDEVICE
    )

    assert {entry.implementation_id for entry in accelerated_entries} == {
        specification.implementation_id
        for specification in specifications
        if specification.implementation_id != NATIVE_LIBDEVICE
    }
    for entry in accelerated_entries:
        matching = tuple(
            specification
            for specification in specifications
            if specification.implementation_id == entry.implementation_id
        )
        assert matching
        for specification in matching:
            assert specification.operation == entry.operation.value
            assert specification.domain in entry.domains
            assert specification.min_elements == entry.min_elements
            assert specification.coordinate_contract_m == entry.accuracy.max_horizontal_error_m
            assert specification.max_physical_scale_m == entry.accuracy.max_physical_scale_m
            if entry.operation is TranscendentalOperation.PROJECTION:
                assert specification.domain.split(".")[1] == specification.direction
