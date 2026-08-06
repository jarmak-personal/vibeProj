"""Public API and host resolver tests for transcendental policies."""

from __future__ import annotations

import dataclasses
import math
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

import vibeproj
from vibeproj import DeviceCapability, Transformer
from vibeproj.transcendentals import (
    GEOS_FORWARD_FIXED_Q62,
    GEOS_FORWARD_FIXED_Q62_MIN_ELEMENTS,
    GNOM_INVERSE_GUARDED_RSQRT_REFRAME,
    HELMERT_FIXED_Q62,
    HELMERT_FIXED_Q62_MIN_ELEMENTS,
    LAEA_FORWARD_POLAR_FIXED_Q62,
    LAEA_FORWARD_POLAR_FIXED_Q62_MIN_ELEMENTS,
    LCC_FORWARD_CONFORMAL_REFRAME,
    LCC_INVERSE_CONFORMAL_REFRAME,
    MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY,
    MERC_FORWARD_SPHERICAL_PRODUCT_POLY,
    MERC_INVERSE_EXP_SERIES,
    NATIVE_LIBDEVICE,
    ORTHO_FORWARD_FIXED_Q62,
    ORTHO_FORWARD_FIXED_Q62_MIN_ELEMENTS,
    ORTHO_INVERSE_GUARDED_REFRAME,
    ORTHO_INVERSE_GUARDED_REFRAME_MIN_ELEMENTS,
    SINU_FORWARD_FIXED_Q62,
    SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS,
    STERE_INVERSE_FIXED_Q62,
    TMERC_FIXED_Q62,
    TMERC_FIXED_Q62_MIN_ELEMENTS,
    TranscendentalOperation,
    _resolve_transcendental_strategy_cached,
    list_transcendental_strategies,
    projection_strategy_domain,
    resolve_transcendental_strategy,
)

ADA = DeviceCapability(
    backend="cuda",
    compute_capability=(8, 9),
    fp32_to_fp64_ratio=64,
    name="mock Ada",
    device_id=0,
)
HOPPER = DeviceCapability(
    backend="cuda",
    compute_capability=(9, 0),
    fp32_to_fp64_ratio=2,
    name="mock Hopper",
    device_id=0,
)
CPU = DeviceCapability(backend="cpu", name="CPU")
WEAK_ADA = dataclasses.replace(ADA, fp32_to_fp64_ratio=8, name="mock strong-fp64 Ada")

SPHERICAL_LONG_LAT = "+proj=longlat +R=6378137 +type=crs"
PROJECTION_CRS = {
    "sinu": "+proj=sinu +lon_0=0 +R=6378137 +units=m +type=crs",
    "ortho": "+proj=ortho +lat_0=45 +lon_0=0 +R=6378137 +units=m +type=crs",
}
GLOBAL_TM_0 = "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +type=crs"
GLOBAL_TM_9 = "+proj=tmerc +lat_0=0 +lon_0=9 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +type=crs"
UTM_TM_31 = "+proj=utm +zone=31 +datum=WGS84 +units=m +type=crs"
UTM_TM_32 = "+proj=utm +zone=32 +datum=WGS84 +units=m +type=crs"
ORTHO_FORWARD_DOMAIN = "ortho.forward.spherical.oblique"


def _projection_transformer(projection: str) -> Transformer:
    return Transformer.from_crs(
        SPHERICAL_LONG_LAT,
        PROJECTION_CRS[projection],
        always_xy=True,
    )


@pytest.mark.parametrize("policy", ["auto", "native", "accelerated"])
def test_cpu_transform_accepts_every_policy(policy):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    expected = transformer.transform([2.0], [49.0], transcendentals="native")
    actual = transformer.transform([2.0], [49.0], transcendentals=policy)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "call",
    [
        lambda transformer: transformer.transform(object(), object(), transcendentals="fast"),
        lambda transformer: transformer.transform_buffers(
            object(), object(), transcendentals="fast"
        ),
        lambda transformer: transformer.transform_chunked(
            object(), object(), transcendentals="fast"
        ),
        lambda transformer: transformer.transform_bounds(
            0.0, 0.0, 1.0, 1.0, transcendentals="fast"
        ),
        lambda transformer: transformer.compile(transcendentals="fast"),
        lambda transformer: transformer.explain_strategy(transcendentals="fast"),
    ],
)
def test_invalid_policy_fails_before_input_or_device_work(call):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    with pytest.raises(ValueError, match="Invalid transcendentals policy.*fast"):
        call(transformer)


def test_invalid_warm_up_policy_fails_early():
    with pytest.raises(ValueError, match="Invalid transcendentals policy"):
        vibeproj.warm_up(transcendentals="fast")


def test_registry_is_immutable_and_contains_stable_ids():
    registry = list_transcendental_strategies()
    assert isinstance(registry, tuple)
    assert {entry.implementation_id for entry in registry} == {
        NATIVE_LIBDEVICE,
        HELMERT_FIXED_Q62,
        GEOS_FORWARD_FIXED_Q62,
        GNOM_INVERSE_GUARDED_RSQRT_REFRAME,
        LAEA_FORWARD_POLAR_FIXED_Q62,
        LCC_FORWARD_CONFORMAL_REFRAME,
        LCC_INVERSE_CONFORMAL_REFRAME,
        MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY,
        MERC_FORWARD_SPHERICAL_PRODUCT_POLY,
        MERC_INVERSE_EXP_SERIES,
        ORTHO_FORWARD_FIXED_Q62,
        ORTHO_INVERSE_GUARDED_REFRAME,
        SINU_FORWARD_FIXED_Q62,
        STERE_INVERSE_FIXED_Q62,
        TMERC_FIXED_Q62,
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        registry[0].family = "changed"  # type: ignore[misc]


def test_resolver_selects_an_exact_domain_candidate_from_registry(monkeypatch):
    import vibeproj.transcendentals as transcendental_module

    template = next(
        entry
        for entry in list_transcendental_strategies()
        if entry.implementation_id == TMERC_FIXED_Q62
    )
    synthetic = dataclasses.replace(
        template,
        implementation_id="synthetic.forward.fixed_q62",
        operation=TranscendentalOperation.PROJECTION,
        domains=("synthetic.forward",),
        min_elements=17,
    )
    monkeypatch.setattr(
        transcendental_module,
        "_REGISTRY",
        (*list_transcendental_strategies(), synthetic),
    )
    _resolve_transcendental_strategy_cached.cache_clear()
    try:
        below = resolve_transcendental_strategy(
            TranscendentalOperation.PROJECTION,
            "auto",
            device=ADA,
            domain="synthetic.forward",
            workload_size=16,
        )
        automatic = resolve_transcendental_strategy(
            TranscendentalOperation.PROJECTION,
            "auto",
            device=ADA,
            domain="synthetic.forward",
            workload_size=17,
        )
        explicit = resolve_transcendental_strategy(
            TranscendentalOperation.PROJECTION,
            "accelerated",
            device=ADA,
            domain="synthetic.forward",
            workload_size=1,
        )
    finally:
        _resolve_transcendental_strategy_cached.cache_clear()

    assert below.implementation_id == NATIVE_LIBDEVICE
    assert automatic.implementation_id == synthetic.implementation_id
    assert explicit.implementation_id == synthetic.implementation_id


def test_resolver_rejects_ambiguous_exact_domain_candidates(monkeypatch):
    import vibeproj.transcendentals as transcendental_module

    template = next(
        entry
        for entry in list_transcendental_strategies()
        if entry.implementation_id == TMERC_FIXED_Q62
    )
    candidates = tuple(
        dataclasses.replace(
            template,
            implementation_id=f"synthetic.forward.variant_{index}",
            operation=TranscendentalOperation.PROJECTION,
            domains=("synthetic.forward",),
        )
        for index in range(2)
    )
    monkeypatch.setattr(
        transcendental_module,
        "_REGISTRY",
        (*list_transcendental_strategies(), *candidates),
    )
    _resolve_transcendental_strategy_cached.cache_clear()
    try:
        native = resolve_transcendental_strategy(
            TranscendentalOperation.PROJECTION,
            "native",
            device=ADA,
            domain="synthetic.forward",
        )
        assert native.implementation_id == NATIVE_LIBDEVICE
        with pytest.raises(RuntimeError, match="Ambiguous.*synthetic.forward"):
            resolve_transcendental_strategy(
                TranscendentalOperation.PROJECTION,
                "accelerated",
                device=ADA,
                domain="synthetic.forward",
            )
    finally:
        _resolve_transcendental_strategy_cached.cache_clear()


def test_resolver_filters_disjoint_hardware_candidates_before_ambiguity(monkeypatch):
    import vibeproj.transcendentals as transcendental_module

    template = next(
        entry
        for entry in list_transcendental_strategies()
        if entry.implementation_id == SINU_FORWARD_FIXED_Q62
    )
    ada_variant = dataclasses.replace(
        template,
        implementation_id="synthetic.ada",
        domains=("synthetic.forward",),
        min_elements=17,
    )
    hopper_variant = dataclasses.replace(
        template,
        implementation_id="synthetic.hopper",
        domains=("synthetic.forward",),
        supported_compute_capabilities=((9, 0),),
        min_fp32_to_fp64_ratio=None,
        min_elements=31,
    )
    monkeypatch.setattr(
        transcendental_module,
        "_REGISTRY",
        (*list_transcendental_strategies(), ada_variant, hopper_variant),
    )
    _resolve_transcendental_strategy_cached.cache_clear()
    try:
        for device, variant in ((ADA, ada_variant), (HOPPER, hopper_variant)):
            below = resolve_transcendental_strategy(
                TranscendentalOperation.PROJECTION,
                "auto",
                device=device,
                domain="synthetic.forward",
                workload_size=variant.min_elements - 1,
            )
            at = resolve_transcendental_strategy(
                TranscendentalOperation.PROJECTION,
                "auto",
                device=device,
                domain="synthetic.forward",
                workload_size=variant.min_elements,
            )
            assert below.implementation_id == NATIVE_LIBDEVICE
            assert str(variant.min_elements) in below.reason
            assert at.implementation_id == variant.implementation_id
    finally:
        _resolve_transcendental_strategy_cached.cache_clear()


def test_resolver_uses_priority_then_rejects_equal_priority_overlap(monkeypatch):
    import vibeproj.transcendentals as transcendental_module

    base_registry = list_transcendental_strategies()
    template = next(
        entry for entry in base_registry if entry.implementation_id == SINU_FORWARD_FIXED_Q62
    )
    lower = dataclasses.replace(
        template,
        implementation_id="synthetic.lower",
        domains=("synthetic.forward",),
        priority=10,
        min_elements=11,
    )
    preferred = dataclasses.replace(
        template,
        implementation_id="synthetic.preferred",
        domains=("synthetic.forward",),
        priority=20,
        min_elements=23,
    )
    monkeypatch.setattr(
        transcendental_module,
        "_REGISTRY",
        (*base_registry, lower, preferred),
    )
    _resolve_transcendental_strategy_cached.cache_clear()
    try:
        below = resolve_transcendental_strategy(
            TranscendentalOperation.PROJECTION,
            "auto",
            device=ADA,
            domain="synthetic.forward",
            workload_size=22,
        )
        selected = resolve_transcendental_strategy(
            TranscendentalOperation.PROJECTION,
            "auto",
            device=ADA,
            domain="synthetic.forward",
            workload_size=23,
        )
        assert below.implementation_id == NATIVE_LIBDEVICE
        assert "crossover 23" in below.reason
        assert selected.implementation_id == preferred.implementation_id

        tied = dataclasses.replace(preferred, priority=lower.priority)
        monkeypatch.setattr(
            transcendental_module,
            "_REGISTRY",
            (*base_registry, lower, tied),
        )
        _resolve_transcendental_strategy_cached.cache_clear()
        with pytest.raises(RuntimeError, match="Ambiguous.*priority 10"):
            resolve_transcendental_strategy(
                TranscendentalOperation.PROJECTION,
                "accelerated",
                device=ADA,
                domain="synthetic.forward",
            )
    finally:
        _resolve_transcendental_strategy_cached.cache_clear()


@pytest.mark.parametrize(
    ("operation", "domain", "threshold", "accelerated_id"),
    [
        (
            TranscendentalOperation.TMERC_FORWARD,
            "utm",
            TMERC_FIXED_Q62_MIN_ELEMENTS,
            TMERC_FIXED_Q62,
        ),
        (
            TranscendentalOperation.HELMERT,
            "global",
            HELMERT_FIXED_Q62_MIN_ELEMENTS,
            HELMERT_FIXED_Q62,
        ),
        (
            TranscendentalOperation.PROJECTION,
            "sinu.forward.spherical",
            SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS,
            SINU_FORWARD_FIXED_Q62,
        ),
        (
            TranscendentalOperation.PROJECTION,
            ORTHO_FORWARD_DOMAIN,
            ORTHO_FORWARD_FIXED_Q62_MIN_ELEMENTS,
            ORTHO_FORWARD_FIXED_Q62,
        ),
        (
            TranscendentalOperation.PROJECTION,
            "geos.forward.ellipsoidal.sweep_x",
            GEOS_FORWARD_FIXED_Q62_MIN_ELEMENTS,
            GEOS_FORWARD_FIXED_Q62,
        ),
        (
            TranscendentalOperation.PROJECTION,
            "laea.forward.spherical.north_pole",
            LAEA_FORWARD_POLAR_FIXED_Q62_MIN_ELEMENTS,
            LAEA_FORWARD_POLAR_FIXED_Q62,
        ),
    ],
)
def test_auto_strategy_uses_exact_workload_crossover(operation, domain, threshold, accelerated_id):
    below = resolve_transcendental_strategy(
        operation,
        "auto",
        device=ADA,
        domain=domain,
        workload_size=threshold - 1,
    )
    at = resolve_transcendental_strategy(
        operation,
        "auto",
        device=ADA,
        domain=domain,
        workload_size=threshold,
    )
    explicit = resolve_transcendental_strategy(
        operation,
        "accelerated",
        device=ADA,
        domain=domain,
        workload_size=1,
    )
    unspecified = resolve_transcendental_strategy(
        operation,
        "auto",
        device=ADA,
        domain=domain,
        workload_size=None,
    )

    assert below.implementation_id == NATIVE_LIBDEVICE
    assert below.workload_size == threshold - 1
    assert at.implementation_id == accelerated_id
    assert at.workload_size == threshold
    assert explicit.implementation_id == accelerated_id
    assert unspecified.implementation_id == accelerated_id


@pytest.mark.parametrize(
    ("operation", "domain", "expected"),
    [
        (TranscendentalOperation.HELMERT, "global", HELMERT_FIXED_Q62),
        (TranscendentalOperation.TMERC_FORWARD, "utm", TMERC_FIXED_Q62),
        (
            TranscendentalOperation.PROJECTION,
            "sinu.forward.spherical",
            SINU_FORWARD_FIXED_Q62,
        ),
        (TranscendentalOperation.PROJECTION, ORTHO_FORWARD_DOMAIN, ORTHO_FORWARD_FIXED_Q62),
        (
            TranscendentalOperation.PROJECTION,
            "geos.forward.spherical.sweep_y",
            GEOS_FORWARD_FIXED_Q62,
        ),
        (
            TranscendentalOperation.PROJECTION,
            "laea.forward.spherical.south_pole",
            LAEA_FORWARD_POLAR_FIXED_Q62,
        ),
    ],
)
def test_ada_auto_selects_qualified_implementations(operation, domain, expected):
    decision = resolve_transcendental_strategy(operation, device=ADA, domain=domain)
    assert decision.implementation_id == expected
    assert decision.fallback is False


@pytest.mark.parametrize(
    ("projection", "accelerated_id", "min_elements"),
    [
        ("sinu", SINU_FORWARD_FIXED_Q62, SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS),
        ("ortho", ORTHO_FORWARD_FIXED_Q62, ORTHO_FORWARD_FIXED_Q62_MIN_ELEMENTS),
    ],
)
@pytest.mark.parametrize(
    ("policy", "precision", "size_offset", "accelerated"),
    [
        ("native", "fp64", 0, False),
        ("auto", "auto", 0, True),
        ("auto", "fp64", -1, False),
        ("auto", "fp64", 0, True),
        ("accelerated", "fp64", None, True),
        ("accelerated", "fp32", 0, False),
        ("accelerated", "ds", 0, False),
    ],
)
def test_wave1_public_policy_precision_and_size_matrix(
    projection,
    accelerated_id,
    min_elements,
    policy,
    precision,
    size_offset,
    accelerated,
):
    workload_size = 1 if size_offset is None else min_elements + size_offset
    explanation = _projection_transformer(projection).explain_strategy(
        transcendentals=policy,
        precision=precision,
        workload_size=workload_size,
        device=ADA,
    )
    assert explanation.workload_size == workload_size
    assert len(explanation.decisions) == 1
    decision = explanation.decisions[0]
    expected_domain = ORTHO_FORWARD_DOMAIN if projection == "ortho" else "sinu.forward.spherical"
    assert decision.domain == expected_domain
    assert decision.implementation_id == (accelerated_id if accelerated else NATIVE_LIBDEVICE)
    assert decision.fallback is (policy == "accelerated" and not accelerated)
    assert decision.accuracy.max_horizontal_error_m <= 1e-8


@pytest.mark.parametrize(
    ("domain", "device", "precision", "reason"),
    [
        (
            "sinu.inverse.ellipsoidal",
            ADA,
            "fp64",
            "domain 'sinu.inverse.ellipsoidal' is not accuracy-qualified",
        ),
        (
            "ortho.inverse.spherical.oblique",
            ADA,
            "fp64",
            "domain 'ortho.inverse.spherical.oblique' is not accuracy-qualified",
        ),
        ("sinu.forward.spherical", CPU, "fp64", "cpu backend"),
        (ORTHO_FORWARD_DOMAIN, HOPPER, "fp64", "compute capability"),
        ("sinu.forward.spherical", WEAK_ADA, "fp64", "fp32:fp64 ratio"),
        (ORTHO_FORWARD_DOMAIN, ADA, "fp32", "compute precision 'fp32'"),
        (ORTHO_FORWARD_DOMAIN, ADA, "ds", "compute precision 'ds'"),
    ],
)
def test_wave1_explicit_fallback_explains_exact_failed_qualification(
    domain, device, precision, reason
):
    decision = resolve_transcendental_strategy(
        TranscendentalOperation.PROJECTION,
        "accelerated",
        device=device,
        domain=domain,
        precision=precision,
    )
    assert decision.implementation_id == NATIVE_LIBDEVICE
    assert decision.fallback is True
    assert reason in decision.reason


@pytest.mark.parametrize("policy", ["auto", "native"])
def test_hopper_normally_remains_native(policy):
    decision = resolve_transcendental_strategy(
        TranscendentalOperation.HELMERT, policy, device=HOPPER
    )
    assert decision.implementation_id == NATIVE_LIBDEVICE
    assert decision.fallback is False
    assert "native" in decision.reason


def test_accelerated_unsupported_device_has_explicit_native_fallback():
    decision = resolve_transcendental_strategy(
        TranscendentalOperation.HELMERT, "accelerated", device=HOPPER
    )
    assert decision.implementation_id == NATIVE_LIBDEVICE
    assert decision.fallback is True
    assert "fell back to native" in decision.reason


def test_accelerated_unsupported_domain_has_explicit_native_fallback():
    decision = resolve_transcendental_strategy(
        TranscendentalOperation.TMERC_FORWARD,
        "accelerated",
        device=ADA,
        domain="global",
    )
    assert decision.implementation_id == NATIVE_LIBDEVICE
    assert decision.fallback is True
    assert "not accuracy-qualified" in decision.reason


@pytest.mark.parametrize("precision", ["fp32", "ds"])
def test_tmerc_acceleration_does_not_change_or_override_compute_precision(precision):
    decision = resolve_transcendental_strategy(
        TranscendentalOperation.TMERC_FORWARD,
        "accelerated",
        device=ADA,
        domain="utm",
        precision=precision,
    )
    assert decision.implementation_id == NATIVE_LIBDEVICE
    assert decision.fallback is True
    assert precision in decision.reason


def test_non_tmerc_accelerated_request_is_observable_native_fallback():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")
    explanation = transformer.explain_strategy(transcendentals="accelerated", device=ADA)
    assert explanation.requested_policy == "accelerated"
    assert len(explanation.decisions) == 1
    decision = explanation.decisions[0]
    assert decision.operation is TranscendentalOperation.PROJECTION
    assert decision.domain == "webmerc.forward.spherical.pseudo"
    assert decision.implementation_id == NATIVE_LIBDEVICE
    assert decision.fallback is True
    assert "domain 'webmerc.forward.spherical.pseudo' is not accuracy-qualified" in decision.reason


def test_utm_explanation_resolves_by_policy_without_array_materialization():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    accelerated = transformer.explain_strategy(transcendentals="accelerated", device=ADA)
    native = transformer.explain_strategy(transcendentals="native", device=ADA)
    assert [decision.implementation_id for decision in accelerated.decisions] == [TMERC_FIXED_Q62]
    assert [decision.implementation_id for decision in native.decisions] == [NATIVE_LIBDEVICE]


def test_proj_to_proj_inverse_explanation_includes_each_projection_stage():
    transformer = Transformer.from_crs("EPSG:32631", "EPSG:3857")
    explanation = transformer.explain_strategy(
        transcendentals="accelerated", direction="INVERSE", device=ADA
    )
    assert [decision.domain for decision in explanation.decisions] == [
        "webmerc.inverse.spherical.pseudo",
        "utm",
    ]
    assert explanation.decisions[0].fallback is True
    assert explanation.decisions[1].implementation_id == TMERC_FIXED_Q62


def test_helmert_explanation_includes_projection_and_datum_stage():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    assert transformer._helmert is not None
    explanation = transformer.explain_strategy(transcendentals="accelerated", device=ADA)
    assert [decision.operation for decision in explanation.decisions] == [
        TranscendentalOperation.HELMERT,
        TranscendentalOperation.TMERC_FORWARD,
    ]
    assert explanation.decisions[0].implementation_id == HELMERT_FIXED_Q62
    assert explanation.decisions[1].implementation_id == NATIVE_LIBDEVICE
    assert explanation.decisions[1].fallback is True


@pytest.mark.parametrize(
    ("precision", "policy", "expected"),
    [
        ("auto", "auto", TMERC_FIXED_Q62),
        ("fp64", "accelerated", TMERC_FIXED_Q62),
        ("fp32", "accelerated", NATIVE_LIBDEVICE),
        ("fp64", "native", NATIVE_LIBDEVICE),
    ],
)
def test_compile_resolves_exact_tmerc_cache_variant(monkeypatch, precision, policy, expected):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    observed = []

    def capture(*args, **kwargs):
        variants = kwargs["projection_variants"]
        observed.append(variants)
        with pytest.raises(TypeError):
            variants[0] = ("tmerc", "forward", "changed")

    monkeypatch.setattr("vibeproj.transcendentals.detect_device_capability", lambda: ADA)
    monkeypatch.setattr("vibeproj.fused_kernels.compile_kernels", capture)
    transformer.compile(precision=precision, transcendentals=policy)
    assert observed == [
        (
            ("tmerc", "forward", expected),
            ("tmerc", "inverse", NATIVE_LIBDEVICE),
        )
    ]


@pytest.mark.parametrize(
    ("source", "target"),
    [(GLOBAL_TM_0, UTM_TM_31), (UTM_TM_31, GLOBAL_TM_0)],
)
def test_compile_mixed_global_and_utm_tmerc_keeps_both_forward_variants(
    monkeypatch, source, target
):
    observed = []
    monkeypatch.setattr("vibeproj.transcendentals.detect_device_capability", lambda: ADA)
    monkeypatch.setattr(
        "vibeproj.fused_kernels.compile_kernels",
        lambda *args, **kwargs: observed.append(kwargs["projection_variants"]),
    )

    Transformer.from_crs(source, target).compile(precision="fp64", transcendentals="auto")

    assert observed == [
        (
            ("tmerc", "forward", NATIVE_LIBDEVICE),
            ("tmerc", "forward", TMERC_FIXED_Q62),
            ("tmerc", "inverse", NATIVE_LIBDEVICE),
        )
    ]


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        (
            UTM_TM_31,
            UTM_TM_32,
            (
                ("tmerc", "forward", TMERC_FIXED_Q62),
                ("tmerc", "inverse", NATIVE_LIBDEVICE),
            ),
        ),
        (
            GLOBAL_TM_0,
            GLOBAL_TM_9,
            (
                ("tmerc", "forward", NATIVE_LIBDEVICE),
                ("tmerc", "inverse", NATIVE_LIBDEVICE),
            ),
        ),
    ],
)
def test_compile_deduplicates_matching_tmerc_domains(monkeypatch, source, target, expected):
    observed = []
    monkeypatch.setattr("vibeproj.transcendentals.detect_device_capability", lambda: ADA)
    monkeypatch.setattr(
        "vibeproj.fused_kernels.compile_kernels",
        lambda *args, **kwargs: observed.append(kwargs["projection_variants"]),
    )

    Transformer.from_crs(source, target).compile(precision="fp64", transcendentals="auto")

    assert observed == [expected]


@pytest.mark.parametrize(
    ("precision", "policy"),
    [("fp32", "accelerated"), ("ds", "accelerated"), ("fp64", "native")],
)
def test_compile_unqualified_modes_collapse_mixed_tmerc_to_native(monkeypatch, precision, policy):
    observed = []
    monkeypatch.setattr("vibeproj.transcendentals.detect_device_capability", lambda: ADA)
    monkeypatch.setattr(
        "vibeproj.fused_kernels.compile_kernels",
        lambda *args, **kwargs: observed.append(kwargs["projection_variants"]),
    )

    Transformer.from_crs(GLOBAL_TM_0, UTM_TM_31).compile(
        precision=precision,
        transcendentals=policy,
    )

    assert observed == [
        (
            ("tmerc", "forward", NATIVE_LIBDEVICE),
            ("tmerc", "inverse", NATIVE_LIBDEVICE),
        )
    ]


def test_compile_kernels_deduplicates_exact_variant_cache_keys(monkeypatch):
    import vibeproj.fused_kernels as fused_module

    observed = []

    class CompileSpy:
        def compile(self):
            return None

    def fake_get_kernel(projection, direction, compute_dtype, **kwargs):
        observed.append((projection, direction, compute_dtype, kwargs["transcendental_impl"]))
        return CompileSpy()

    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    monkeypatch.setattr(fused_module, "_get_kernel", fake_get_kernel)
    variants = (
        ("tmerc", "forward", NATIVE_LIBDEVICE),
        ("tmerc", "forward", TMERC_FIXED_Q62),
        ("tmerc", "forward", TMERC_FIXED_Q62),
        ("tmerc", "inverse", NATIVE_LIBDEVICE),
    )

    fused_module.compile_kernels(precision="fp64", projection_variants=variants)

    assert observed == [
        ("tmerc", "forward", "float64", NATIVE_LIBDEVICE),
        ("tmerc", "forward", "float64", TMERC_FIXED_Q62),
        ("tmerc", "inverse", "float64", NATIVE_LIBDEVICE),
    ]


@pytest.mark.parametrize(
    ("projection", "accelerated_id"),
    [
        ("sinu", SINU_FORWARD_FIXED_Q62),
        ("ortho", ORTHO_FORWARD_FIXED_Q62),
    ],
)
@pytest.mark.parametrize(
    ("precision", "policy", "uses_accelerated"),
    [
        ("auto", "auto", True),
        ("fp64", "accelerated", True),
        ("fp32", "accelerated", False),
        ("ds", "accelerated", False),
        ("fp64", "native", False),
    ],
)
def test_wave1_compile_matrix_keeps_companion_inverse_native(
    monkeypatch,
    projection,
    accelerated_id,
    precision,
    policy,
    uses_accelerated,
):
    observed = []
    monkeypatch.setattr("vibeproj.transcendentals.detect_device_capability", lambda: ADA)
    monkeypatch.setattr(
        "vibeproj.fused_kernels.compile_kernels",
        lambda *args, **kwargs: observed.append(kwargs["projection_variants"]),
    )

    _projection_transformer(projection).compile(
        precision=precision,
        transcendentals=policy,
    )

    expected_forward = accelerated_id if uses_accelerated else NATIVE_LIBDEVICE
    assert observed == [
        (
            (projection, "forward", expected_forward),
            (projection, "inverse", NATIVE_LIBDEVICE),
        )
    ]


def test_compile_resolves_helmert_and_non_utm_fallback_independently(monkeypatch):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    projection_impl = []
    helmert_impl = []
    monkeypatch.setattr("vibeproj.transcendentals.detect_device_capability", lambda: ADA)
    monkeypatch.setattr(
        "vibeproj.fused_kernels.compile_kernels",
        lambda *args, **kwargs: projection_impl.append(kwargs["projection_variants"]),
    )
    monkeypatch.setattr(
        "vibeproj.fused_kernels.compile_helmert_kernel",
        lambda **kwargs: helmert_impl.append(kwargs["transcendental_impl"]),
    )
    transformer.compile(transcendentals="accelerated")
    assert projection_impl == [
        (
            ("tmerc", "forward", NATIVE_LIBDEVICE),
            ("tmerc", "inverse", NATIVE_LIBDEVICE),
        )
    ]
    assert helmert_impl == [HELMERT_FIXED_Q62]


def test_warm_up_resolves_every_requested_projection_direction(monkeypatch):
    observed = []

    def capture(*args, **kwargs):
        variants = kwargs["projection_variants"]
        observed.append(variants)
        with pytest.raises(TypeError):
            variants[0] = ("webmerc", "forward", "changed")

    monkeypatch.setattr("vibeproj.transcendentals.detect_device_capability", lambda: ADA)
    monkeypatch.setattr("vibeproj.fused_kernels.compile_kernels", capture)
    vibeproj.warm_up(
        ["tmerc", "webmerc"],
        precision="fp64",
        transcendentals="accelerated",
    )

    assert observed == [
        (
            ("tmerc", "forward", NATIVE_LIBDEVICE),
            ("tmerc", "forward", TMERC_FIXED_Q62),
            ("tmerc", "inverse", NATIVE_LIBDEVICE),
            ("webmerc", "forward", NATIVE_LIBDEVICE),
            ("webmerc", "inverse", NATIVE_LIBDEVICE),
        )
    ]


@pytest.mark.parametrize(
    ("precision", "policy", "expected_sinu", "expected_ortho", "expected_ortho_inverse"),
    [
        (
            "auto",
            "auto",
            SINU_FORWARD_FIXED_Q62,
            ORTHO_FORWARD_FIXED_Q62,
            ORTHO_INVERSE_GUARDED_REFRAME,
        ),
        (
            "fp64",
            "accelerated",
            SINU_FORWARD_FIXED_Q62,
            ORTHO_FORWARD_FIXED_Q62,
            ORTHO_INVERSE_GUARDED_REFRAME,
        ),
        ("fp32", "accelerated", NATIVE_LIBDEVICE, NATIVE_LIBDEVICE, NATIVE_LIBDEVICE),
        ("ds", "accelerated", NATIVE_LIBDEVICE, NATIVE_LIBDEVICE, NATIVE_LIBDEVICE),
        ("fp64", "native", NATIVE_LIBDEVICE, NATIVE_LIBDEVICE, NATIVE_LIBDEVICE),
    ],
)
def test_wave1_warm_up_matrix_keeps_companion_inverses_native(
    monkeypatch,
    precision,
    policy,
    expected_sinu,
    expected_ortho,
    expected_ortho_inverse,
):
    observed = []
    monkeypatch.setattr("vibeproj.transcendentals.detect_device_capability", lambda: ADA)
    monkeypatch.setattr(
        "vibeproj.fused_kernels.compile_kernels",
        lambda *args, **kwargs: observed.append(kwargs["projection_variants"]),
    )

    vibeproj.warm_up(
        ["sinu", "ortho"],
        precision=precision,
        transcendentals=policy,
    )

    assert observed == [
        (
            ("ortho", "forward", expected_ortho),
            ("ortho", "inverse", expected_ortho_inverse),
            ("sinu", "forward", expected_sinu),
            ("sinu", "inverse", NATIVE_LIBDEVICE),
        )
    ]


@pytest.mark.parametrize(
    ("requested_lat_0", "canonical_lat_0", "expected_implementation"),
    [
        (0.0, 0.0, ORTHO_INVERSE_GUARDED_REFRAME),
        (1e-9, 0.0, ORTHO_INVERSE_GUARDED_REFRAME),
        (-1e-9, 0.0, ORTHO_INVERSE_GUARDED_REFRAME),
        (1e-8, 1e-8, NATIVE_LIBDEVICE),
        (-1e-8, -1e-8, NATIVE_LIBDEVICE),
    ],
)
def test_ortho_inverse_canonicalized_origin_controls_public_selection(
    requested_lat_0,
    canonical_lat_0,
    expected_implementation,
):
    target = f"+proj=ortho +lat_0={requested_lat_0:.17g} +lon_0=0 +R=6378137 +units=m +type=crs"
    transformer = Transformer.from_crs(target, "+proj=longlat +R=6378137 +type=crs")
    pipeline = transformer._pipeline_for_direction("FORWARD")

    explanation = transformer.explain_strategy(
        transcendentals="auto",
        precision="fp64",
        workload_size=ORTHO_INVERSE_GUARDED_REFRAME_MIN_ELEMENTS,
        device=ADA,
    )

    assert pipeline.proj_params.lat_0 == canonical_lat_0
    assert pipeline.computed["_strategy_latitude_origin"] == canonical_lat_0
    assert explanation.decisions[0].implementation_id == expected_implementation


@pytest.mark.parametrize(
    ("latitude_origin", "expected_domain", "expected_implementation"),
    [
        (0.0, "gnom.inverse.spherical.equatorial", GNOM_INVERSE_GUARDED_RSQRT_REFRAME),
        (45.0, "gnom.inverse.spherical.oblique_bounded", GNOM_INVERSE_GUARDED_RSQRT_REFRAME),
        (60.0, "gnom.inverse.spherical.oblique_bounded", GNOM_INVERSE_GUARDED_RSQRT_REFRAME),
        (61.0, "gnom.inverse.spherical.oblique_high", NATIVE_LIBDEVICE),
        (90.0, "gnom.inverse.spherical.north_pole", NATIVE_LIBDEVICE),
        (-90.0, "gnom.inverse.spherical.south_pole", NATIVE_LIBDEVICE),
    ],
)
def test_gnom_inverse_origin_domain_controls_public_selection(
    latitude_origin,
    expected_domain,
    expected_implementation,
):
    target = f"+proj=gnom +lat_0={latitude_origin:.17g} +lon_0=0 +R=6378137 +units=m +type=crs"
    transformer = Transformer.from_crs(target, SPHERICAL_LONG_LAT, always_xy=True)
    explanation = transformer.explain_strategy(
        transcendentals="accelerated",
        precision="fp64",
        workload_size=5_000_000,
        device=ADA,
    )
    automatic = transformer.explain_strategy(
        transcendentals="auto",
        precision="fp64",
        workload_size=5_000_000,
        device=ADA,
    )

    assert explanation.decisions[0].domain == expected_domain
    assert explanation.decisions[0].implementation_id == expected_implementation
    assert automatic.decisions[0].implementation_id == NATIVE_LIBDEVICE


@pytest.mark.parametrize(
    ("overrides", "expected_domain"),
    [
        ({"cos_phi0": 0.5}, "gnom.inverse.spherical.oblique_bounded"),
        (
            {"cos_phi0": np.nextafter(0.5, 0.0)},
            "gnom.inverse.spherical.oblique_high",
        ),
        ({"sin_phi0": math.nan}, "gnom.inverse.spherical.invalid_setup"),
        ({"a": math.inf}, "gnom.inverse.spherical.invalid_setup"),
    ],
)
def test_gnom_inverse_domain_pins_exact_cosine_and_invalid_setup(overrides, expected_domain):
    computed = {
        "_strategy_geometry": "spherical",
        "_strategy_latitude_origin": 60.0,
        "sin_phi0": math.sqrt(0.75),
        "cos_phi0": 0.5,
        "a": 6_378_137.0,
        "lam0": 0.0,
        "x0": 0.0,
        "y0": 0.0,
        **overrides,
    }

    assert projection_strategy_domain("gnom", "inverse", computed) == expected_domain


def test_compile_collects_both_directions_of_projected_pipeline(monkeypatch):
    transformer = Transformer.from_crs("EPSG:32631", "EPSG:3857")
    observed = []
    monkeypatch.setattr("vibeproj.transcendentals.detect_device_capability", lambda: ADA)
    monkeypatch.setattr(
        "vibeproj.fused_kernels.compile_kernels",
        lambda *args, **kwargs: observed.append(kwargs["projection_variants"]),
    )

    transformer.compile(precision="fp64", transcendentals="accelerated")

    assert observed == [
        (
            ("tmerc", "forward", TMERC_FIXED_Q62),
            ("tmerc", "inverse", NATIVE_LIBDEVICE),
            ("webmerc", "forward", NATIVE_LIBDEVICE),
            ("webmerc", "inverse", NATIVE_LIBDEVICE),
        )
    ]


def test_transform_propagates_precision_and_policy_independently(monkeypatch):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    x, y = transformer.transform([2.0], [49.0])
    observed = []
    original = transformer._pipeline.transform

    def capture(*args, **kwargs):
        observed.append((kwargs["precision"], kwargs["transcendentals"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(transformer._pipeline, "transform", capture)
    transformer.transform([2.0], [49.0], precision="fp32", transcendentals="native")
    assert observed == [("fp32", "native")]

    transformer.transform(x, y, direction="INVERSE", transcendentals="accelerated")
    assert transformer._inv_pipeline is not None


def test_precision_and_policy_reach_projection_dispatch(monkeypatch):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    observed = []

    def unavailable_fused(*args, **kwargs):
        observed.append((kwargs["precision"], kwargs["transcendentals"]))
        return None

    monkeypatch.setattr("vibeproj.pipeline._try_fused", unavailable_fused)
    transformer.transform([2.0], [49.0], precision="ds", transcendentals="accelerated")
    assert observed == [("ds", "accelerated")]


def test_inverse_lazy_pipeline_propagates_precision_and_policy(monkeypatch):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    x, y = transformer.transform([2.0], [49.0])
    transformer.transform(x, y, direction="INVERSE")
    assert transformer._inv_pipeline is not None
    observed = []
    original = transformer._inv_pipeline.transform

    def capture(*args, **kwargs):
        observed.append((kwargs["precision"], kwargs["transcendentals"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(transformer._inv_pipeline, "transform", capture)
    transformer.transform(
        x,
        y,
        direction="INVERSE",
        precision="fp64",
        transcendentals="native",
    )
    assert observed == [("fp64", "native")]


def test_pickle_roundtrip_has_no_persisted_per_call_policy():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    restored = pickle.loads(pickle.dumps(transformer))
    assert "transcendentals" not in restored.__getstate__()
    explanation = restored.explain_strategy(transcendentals="native", device=CPU)
    assert all(decision.implementation_id == NATIVE_LIBDEVICE for decision in explanation.decisions)


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


def test_invalid_warm_up_precision_fails_early():
    with pytest.raises(ValueError, match="Invalid precision"):
        vibeproj.warm_up(precision="half")


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("EPSG:4326", "EPSG:4326"),
        ("EPSG:4326", "EPSG:4277"),
        ("EPSG:4326", "EPSG:27700"),
        ("EPSG:27700", "EPSG:32631"),
        ("EPSG:4267", "EPSG:4269"),
    ],
)
def test_transform_buffers_returns_every_supplied_output_buffer(source, target):
    transformer = Transformer.from_crs(source, target)
    if source == "EPSG:27700":
        x = np.array([470000.0])
        y = np.array([235000.0])
    else:
        x = np.array([-1.0])
        y = np.array([52.0])
    z = np.array([10.0])
    expected = transformer.transform_buffers(x, y, z)
    out_x = np.empty_like(x)
    out_y = np.empty_like(y)
    out_z = np.empty_like(z)
    actual = transformer.transform_buffers(x, y, z, out_x=out_x, out_y=out_y, out_z=out_z)
    assert actual[0] is out_x
    assert actual[1] is out_y
    assert actual[2] is out_z
    np.testing.assert_allclose(actual, expected)


def test_longlat_helmert_receives_final_axis_mapped_buffers(monkeypatch):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4277")
    out_x = np.empty(1)
    out_y = np.empty(1)
    out_z = np.empty(1)
    observed = {}

    def fake_shift(lat, lon, helmert, xp, **kwargs):
        observed.update(kwargs)
        kwargs["out_lat"][:] = lat
        kwargs["out_lon"][:] = lon
        kwargs["out_h"][:] = kwargs["h"]
        return kwargs["out_lat"], kwargs["out_lon"], kwargs["out_h"]

    monkeypatch.setattr("vibeproj.pipeline._apply_datum_shift", fake_shift)
    result = transformer.transform_buffers(
        np.array([-1.0]),
        np.array([52.0]),
        np.array([10.0]),
        out_x=out_x,
        out_y=out_y,
        out_z=out_z,
    )
    # always_xy=True means public x=lon/y=lat, while Helmert uses lat/lon.
    assert observed["out_lat"] is out_y
    assert observed["out_lon"] is out_x
    assert observed["out_h"] is out_z
    assert result[0] is out_x
    assert result[1] is out_y
    assert result[2] is out_z


def test_strategy_decisions_are_cached_by_full_context(monkeypatch):
    import vibeproj.transcendentals as transcendental_module

    _resolve_transcendental_strategy_cached.cache_clear()
    original = transcendental_module._implementation
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(transcendental_module, "_implementation", counted)
    first = resolve_transcendental_strategy(
        TranscendentalOperation.TMERC_FORWARD,
        "accelerated",
        device=ADA,
        domain="utm",
        precision="fp64",
    )
    second = resolve_transcendental_strategy(
        TranscendentalOperation.TMERC_FORWARD,
        "accelerated",
        device=ADA,
        domain="utm",
        precision="fp64",
    )
    assert second is first
    assert len(calls) == 1  # native lookup occurs only on the first full-context call

    resolve_transcendental_strategy(
        TranscendentalOperation.TMERC_FORWARD,
        "accelerated",
        device=HOPPER,
        domain="utm",
        precision="fp64",
    )
    assert len(calls) == 2


class _FakeDeviceContext:
    def __init__(self, device_id):
        self.id = device_id

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeStream:
    def __init__(self, ptr, device_id=-1):
        self.ptr = ptr
        self.device_id = device_id
        self.entered = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *args):
        self.entered = False
        return False


class _FakeEvent:
    def __init__(self, *, disable_timing=True):
        self.complete = False
        self.stream = None

    def record(self, stream):
        self.stream = stream

    def query(self):
        return self.complete


class _FakeCupy:
    __name__ = "cupy"
    float64 = np.float64

    def __init__(self):
        self.device_id = 0
        self.stream = _FakeStream(0)
        self.empty_calls = []
        self.events = []

        def new_event(**kwargs):
            event = _FakeEvent(**kwargs)
            self.events.append(event)
            return event

        self.cuda = SimpleNamespace(
            Device=lambda device_id: _FakeDeviceContext(device_id),
            Event=new_event,
            get_current_stream=lambda: self.stream,
            runtime=SimpleNamespace(getDevice=lambda: self.device_id),
        )

    def empty(self, size, dtype):
        self.empty_calls.append((self.device_id, size, dtype))
        return np.empty(size, dtype=dtype)


def test_scratch_cache_is_grow_only_and_keyed_by_device_and_stream():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    fake_cp = _FakeCupy()

    def coordinate(size, device_id):
        return SimpleNamespace(size=size, device=SimpleNamespace(id=device_id))

    stream_a = _FakeStream(11)
    with transformer._transform_scratch_context(
        coordinate(8, 0), fake_cp, transformer._pipeline, stream_a
    ) as first:
        assert first is not None
    assert len(fake_cp.empty_calls) == 4
    with transformer._transform_scratch_context(
        coordinate(4, 0), fake_cp, transformer._pipeline, stream_a
    ) as reused:
        assert reused is first
    assert len(fake_cp.empty_calls) == 4

    with transformer._transform_scratch_context(
        coordinate(8, 0), fake_cp, transformer._pipeline, _FakeStream(12)
    ):
        pass
    with transformer._transform_scratch_context(
        coordinate(8, 1), fake_cp, transformer._pipeline, stream_a
    ):
        pass
    assert len(fake_cp.empty_calls) == 12
    assert set(transformer._scratch_slots) == {(0, 11), (0, 12), (1, 11)}


def test_alternating_chunk_streams_allocate_scratch_only_during_warmup():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    fake_cp = _FakeCupy()
    streams = (_FakeStream(21), _FakeStream(22))
    coordinate = SimpleNamespace(size=8, device=SimpleNamespace(id=0))

    for chunk_index in range(8):
        with transformer._transform_scratch_context(
            coordinate,
            fake_cp,
            transformer._pipeline,
            streams[chunk_index % 2],
        ) as scratch:
            assert scratch is not None
    assert len(fake_cp.empty_calls) == 8  # four arrays for each stream, once


def test_scratch_growth_retires_inflight_generation_until_event_completes():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    fake_cp = _FakeCupy()
    stream = _FakeStream(31)

    def coordinate(size):
        return SimpleNamespace(size=size, device=SimpleNamespace(id=0))

    with transformer._transform_scratch_context(
        coordinate(8), fake_cp, transformer._pipeline, stream
    ) as first:
        pass
    with transformer._transform_scratch_context(
        coordinate(32), fake_cp, transformer._pipeline, stream
    ):
        pass

    assert any(scratch is first for scratch, _event in transformer._scratch_retired)
    fake_cp.events[0].complete = True
    with transformer._scratch_slots_lock:
        transformer._prune_scratch_retired_locked()
    assert all(scratch is not first for scratch, _event in transformer._scratch_retired)


def test_scratch_cache_is_bounded_across_many_streams_and_prunes_completed_events():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    fake_cp = _FakeCupy()
    coordinate = SimpleNamespace(size=8, device=SimpleNamespace(id=0))

    for pointer in range(100, 120):
        with transformer._transform_scratch_context(
            coordinate, fake_cp, transformer._pipeline, _FakeStream(pointer)
        ):
            pass

    assert len(transformer._scratch_slots) == 8
    assert len(transformer._scratch_retired) == 12
    for event in fake_cp.events:
        event.complete = True
    with transformer._scratch_slots_lock:
        transformer._prune_scratch_retired_locked()
    assert transformer._scratch_retired == []


def test_chunk_device_buffers_are_cached_per_cuda_device(monkeypatch):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    fake_cp = _FakeCupy()
    fake_cp.ndarray = np.ndarray
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)

    fake_cp.device_id = 0
    device_zero = transformer._get_dev_buffers(8)
    assert transformer._get_dev_buffers(4) is device_zero
    fake_cp.device_id = 1
    device_one = transformer._get_dev_buffers(8)
    assert device_one is not device_zero
    assert set(transformer._device_buffer_cache) == {0, 1}


def test_transform_buffers_resolves_one_immutable_context_from_input_device(monkeypatch):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    fake_cp = _FakeCupy()
    fake_cp.ndarray = np.ndarray
    coordinate = SimpleNamespace(size=32, device=SimpleNamespace(id=7))
    observed = []
    detections = []

    monkeypatch.setattr("vibeproj.transformer.get_array_module", lambda value: fake_cp)

    def fake_detect(xp=None, *, device_id=None):
        detections.append((xp, device_id))
        return dataclasses.replace(ADA, device_id=device_id)

    def fake_transform(pipeline_self, x, y, xp, **kwargs):
        observed.append(kwargs["execution_context"])
        return kwargs["out_x"], kwargs["out_y"]

    monkeypatch.setattr("vibeproj.transcendentals.detect_device_capability", fake_detect)
    monkeypatch.setattr("vibeproj.pipeline.TransformPipeline.transform", fake_transform)
    out_x = np.empty(32)
    out_y = np.empty(32)
    transformer.transform_buffers(coordinate, coordinate, out_x=out_x, out_y=out_y)

    assert detections == [(fake_cp, 7)]
    assert len(observed) == 1
    assert observed[0].device.device_id == 7
    assert observed[0].workload_size == 32
    with pytest.raises(dataclasses.FrozenInstanceError):
        observed[0].workload_size = 1


def test_transform_buffers_rejects_stream_from_another_device(monkeypatch):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    fake_cp = _FakeCupy()
    fake_cp.ndarray = np.ndarray
    coordinate = SimpleNamespace(size=1, device=SimpleNamespace(id=0))
    monkeypatch.setattr("vibeproj.transformer.get_array_module", lambda value: fake_cp)

    with pytest.raises(ValueError, match="stream device 1.*input device 0"):
        transformer.transform_buffers(
            coordinate,
            coordinate,
            out_x=np.empty(1),
            out_y=np.empty(1),
            stream=_FakeStream(123, device_id=1),
        )


@pytest.mark.parametrize(
    ("source", "target", "direction"),
    [
        ("EPSG:4326", "EPSG:4326", "FORWARD"),
        ("EPSG:4326", "EPSG:3857", "FORWARD"),
        ("EPSG:4326", "EPSG:3857", "INVERSE"),
        ("EPSG:32631", "EPSG:3857", "FORWARD"),
        ("EPSG:4326", "EPSG:4277", "FORWARD"),
        ("EPSG:4267", "EPSG:4269", "FORWARD"),
    ],
)
def test_transform_buffers_enqueues_complete_pipeline_in_supplied_stream(
    monkeypatch, source, target, direction
):
    transformer = Transformer.from_crs(source, target)
    fake_cp = _FakeCupy()
    fake_cp.ndarray = np.ndarray
    stream = _FakeStream(99)
    coordinate = SimpleNamespace(size=1, device=SimpleNamespace(id=0))
    out_x = np.empty(1)
    out_y = np.empty(1)
    observed = []

    monkeypatch.setattr("vibeproj.transformer.get_array_module", lambda value: fake_cp)

    def fake_transform(pipeline_self, x, y, xp, **kwargs):
        observed.append((pipeline_self.mode, stream.entered, kwargs["stream"] is stream))
        return kwargs["out_x"], kwargs["out_y"]

    monkeypatch.setattr("vibeproj.pipeline.TransformPipeline.transform", fake_transform)
    transformer.transform_buffers(
        coordinate,
        coordinate,
        direction=direction,
        out_x=out_x,
        out_y=out_y,
        stream=stream,
    )
    assert observed[0][1:] == (True, True)


@pytest.mark.parametrize(
    ("source", "target", "direction"),
    [
        ("EPSG:4326", "EPSG:4277", "FORWARD"),  # Helmert-only longlat
        ("EPSG:4326", "EPSG:27700", "FORWARD"),
        ("EPSG:4326", "EPSG:27700", "INVERSE"),
        ("EPSG:27700", "EPSG:32631", "FORWARD"),
    ],
)
def test_gpu_preallocated_correction_paths_call_no_empty_after_warmup(
    monkeypatch, source, target, direction
):
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    except (ImportError, RuntimeError, OSError):
        pytest.skip("CuPy/CUDA unavailable")

    transformer = Transformer.from_crs(source, target)
    z = cp.asarray([10.0], dtype=cp.float64)
    if source == "EPSG:27700":
        x = cp.asarray([470000.0], dtype=cp.float64)
        y = cp.asarray([235000.0], dtype=cp.float64)
    else:
        x = cp.asarray([-1.0], dtype=cp.float64)
        y = cp.asarray([52.0], dtype=cp.float64)
    if direction == "INVERSE":
        x, y, z = transformer.transform_buffers(x, y, z)

    input_snapshots = tuple(value.copy() for value in (x, y, z))
    outputs = tuple(cp.empty_like(value) for value in (x, y, z))
    transformer.transform_buffers(
        x,
        y,
        z,
        direction=direction,
        out_x=outputs[0],
        out_y=outputs[1],
        out_z=outputs[2],
    )
    cp.cuda.get_current_stream().synchronize()

    empty_calls = []
    original_empty = cp.empty

    def counted_empty(*args, **kwargs):
        empty_calls.append((args, kwargs))
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(cp, "empty", counted_empty)
    result = transformer.transform_buffers(
        x,
        y,
        z,
        direction=direction,
        out_x=outputs[0],
        out_y=outputs[1],
        out_z=outputs[2],
    )
    cp.cuda.get_current_stream().synchronize()

    assert empty_calls == []
    assert all(actual is expected for actual, expected in zip(result, outputs, strict=True))
    for actual, expected in zip((x, y, z), input_snapshots, strict=True):
        cp.testing.assert_array_equal(actual, expected)


def _cupy_or_skip():
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
        return cp
    except (ImportError, RuntimeError, OSError):
        pytest.skip("CuPy/CUDA unavailable")


def test_gpu_svd_only_forward_honors_out_z_on_null_stream():
    cp = _cupy_or_skip()
    transformer = Transformer.from_crs("EPSG:4267", "EPSG:26918")
    assert transformer._svd_correction is not None
    assert transformer._helmert is None
    x = cp.asarray([-75.0, -74.5], dtype=cp.float64)
    y = cp.asarray([40.0, 40.5], dtype=cp.float64)
    z = cp.asarray([10.0, 20.0], dtype=cp.float64)
    outputs = tuple(cp.empty_like(value) for value in (x, y, z))

    result = transformer.transform_buffers(
        x,
        y,
        z,
        out_x=outputs[0],
        out_y=outputs[1],
        out_z=outputs[2],
        stream=cp.cuda.Stream.null,
    )
    cp.cuda.Stream.null.synchronize()

    assert all(actual is expected for actual, expected in zip(result, outputs, strict=True))
    cp.testing.assert_array_equal(outputs[2], z)


def test_gpu_chunk_workspace_reuses_allocations_and_is_thread_safe(monkeypatch):
    cp = _cupy_or_skip()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    inputs = [
        (np.linspace(-4.0, -1.0, 2048), np.linspace(50.0, 54.0, 2048)),
        (np.linspace(-3.0, 0.0, 2048), np.linspace(51.0, 55.0, 2048)),
    ]
    expected = [transformer.transform_chunked(*pair, chunk_size=256) for pair in inputs]
    workspace = transformer._chunk_workspaces[int(cp.cuda.runtime.getDevice())]
    stream_ids = tuple(id(stream) for stream in workspace.streams)

    empty_calls = []
    original_empty = cp.empty

    def counted_empty(*args, **kwargs):
        empty_calls.append((args, kwargs))
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(cp, "empty", counted_empty)
    repeated = transformer.transform_chunked(*inputs[0], chunk_size=256)
    assert empty_calls == []
    assert tuple(id(stream) for stream in workspace.streams) == stream_ids
    for actual, reference in zip(repeated, expected[0], strict=True):
        np.testing.assert_array_equal(actual, reference)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(transformer.transform_chunked, *pair, chunk_size=256) for pair in inputs
        ]
        concurrent = [future.result() for future in futures]
    for result, reference in zip(concurrent, expected, strict=True):
        for actual, wanted in zip(result, reference, strict=True):
            np.testing.assert_array_equal(actual, wanted)


def test_gpu_scratch_cache_stays_bounded_after_many_streams_and_growth():
    cp = _cupy_or_skip()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    streams = []
    for index in range(20):
        stream = cp.cuda.Stream(non_blocking=True)
        streams.append(stream)
        size = 8 << (index % 5)
        x = cp.full(size, -1.0, dtype=cp.float64)
        y = cp.full(size, 52.0, dtype=cp.float64)
        transformer.transform_buffers(
            x,
            y,
            out_x=cp.empty_like(x),
            out_y=cp.empty_like(y),
            stream=stream,
        )

    assert len(transformer._scratch_slots) <= 8
    for stream in streams:
        stream.synchronize()
    with transformer._scratch_slots_lock:
        transformer._prune_scratch_retired_locked()
    assert len(transformer._scratch_slots) <= 8
    assert transformer._scratch_retired == []
