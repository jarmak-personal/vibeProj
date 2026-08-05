"""Exhaustive public transcendental registry and fallback contract."""

from __future__ import annotations

import re
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vibeproj
from vibeproj import Transformer
from vibeproj.fused_kernels import _SUPPORTED
from vibeproj.transcendentals import (
    HELMERT_FIXED_Q62,
    HELMERT_FIXED_Q62_MIN_ELEMENTS,
    NATIVE_LIBDEVICE,
    TMERC_FIXED_Q62,
    TMERC_FIXED_Q62_MIN_ELEMENTS,
    DeviceCapability,
    TranscendentalOperation,
    list_transcendental_strategies,
    resolve_transcendental_strategy,
)


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


def _context(family: str, direction: str) -> tuple[TranscendentalOperation, str]:
    if (family, direction) == ("tmerc", "forward"):
        return TranscendentalOperation.TMERC_FORWARD, "utm"
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
            elif device == ADA_4090 and family == "tmerc" and direction == "forward":
                assert decision.implementation_id == TMERC_FIXED_Q62
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


def _documented_ids(path: Path, heading: str) -> set[str]:
    text = path.read_text(encoding="utf-8")
    section = text.split(heading, maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    return set(re.findall(r"^\| `([^`]+)` \|", section, flags=re.MULTILINE))


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
