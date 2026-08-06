#!/usr/bin/env python3
"""Qualify public transcendental policies with end-to-end GPU transforms.

This is the production qualification benchmark. It times public
``transcendentals=`` policies through ``Transformer.transform_buffers`` with
pre-allocated device output, reports resolved registry IDs, and compares both
coordinate accuracy and hot-path allocation behavior with native policy.

The raw-kernel scripts ``bench_int64_sincos.py``, ``bench_tmerc_int64.py``,
``bench_helmert_int64.py``, and ``bench_alt_transcendentals.py`` retain useful
research/rejected variants but do not establish public coverage.

Examples:
    uv run python benchmarks/bench_transcendental_policy.py --help
    uv run python benchmarks/bench_transcendental_policy.py --case all --n 5000000
    uv run python benchmarks/bench_transcendental_policy.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vibeproj.transcendentals import (
    GEOS_FORWARD_FIXED_Q62,
    GEOS_FORWARD_FIXED_Q62_MIN_ELEMENTS,
    GNOM_INVERSE_GUARDED_RSQRT_REFRAME,
    HELMERT_FIXED_Q62,
    HELMERT_FIXED_Q62_MIN_ELEMENTS,
    LAEA_FORWARD_POLAR_FIXED_Q62,
    LAEA_FORWARD_POLAR_FIXED_Q62_MIN_ELEMENTS,
    KROVAK_INVERSE_GUARDED_LOG_RATIO,
    KROVAK_INVERSE_GUARDED_LOG_RATIO_MIN_ELEMENTS,
    LCC_FORWARD_CONFORMAL_REFRAME,
    LCC_FORWARD_CONFORMAL_REFRAME_MIN_ELEMENTS,
    LCC_INVERSE_CONFORMAL_REFRAME,
    LCC_INVERSE_CONFORMAL_REFRAME_MIN_ELEMENTS,
    MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY,
    MERC_FORWARD_SPHERICAL_PRODUCT_POLY,
    MERC_FORWARD_SPHERICAL_PRODUCT_POLY_MIN_ELEMENTS,
    MERC_INVERSE_EXP_SERIES,
    MERC_INVERSE_EXP_SERIES_MIN_ELEMENTS,
    NATIVE_LIBDEVICE,
    ORTHO_FORWARD_FIXED_Q62,
    ORTHO_FORWARD_FIXED_Q62_MIN_ELEMENTS,
    ORTHO_INVERSE_GUARDED_REFRAME,
    ORTHO_INVERSE_GUARDED_REFRAME_MIN_ELEMENTS,
    PROJECTION_FIXED_Q62_MAX_SCALE_M,
    SINU_FORWARD_FIXED_Q62,
    SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS,
    SINU_INVERSE_CONVERGENT_NEWTON,
    SINU_INVERSE_CONVERGENT_NEWTON_MIN_ELEMENTS,
    SINU_INVERSE_MERIDIONAL_RECURRENCE,
    STERE_INVERSE_FIXED_Q62,
    STERE_INVERSE_FIXED_Q62_MIN_ELEMENTS,
    TMERC_FIXED_Q62,
    TMERC_FIXED_Q62_MIN_ELEMENTS,
)


POLICIES = ("native", "accelerated", "auto")
EARTH_RADIUS_M = 6_378_137.0
DEFAULT_WORKLOAD_SIZES = (
    1,
    8,
    32,
    64,
    128,
    256,
    512,
    1_024,
    2_048,
    4_096,
    8_192,
    16_384,
    32_768,
    65_536,
    131_072,
    262_144,
    524_288,
    1_048_576,
    2_097_152,
    5_000_000,
)


@dataclass(frozen=True)
class QualificationSpec:
    """Reusable public-policy gates for one family and direction."""

    implementation_id: str
    operation: str
    domain: str
    direction: str
    min_elements: int
    coordinate_contract_m: float
    max_physical_scale_m: float | None = None
    expected_kernel_nodes: int = 1
    auto_enabled: bool = True
    auto_disabled_reason: str | None = None
    explicit_performance_min_elements: int | None = None
    auto_implementation_id: str | None = None
    implementation_min_elements: int | None = None
    representative_mixed_guard_sweep: bool = False


QUALIFICATION_SPECS = {
    "tmerc-forward": QualificationSpec(
        implementation_id=TMERC_FIXED_Q62,
        operation="tmerc.forward",
        domain="utm",
        direction="forward",
        min_elements=TMERC_FIXED_Q62_MIN_ELEMENTS,
        coordinate_contract_m=1e-8,
        auto_enabled=False,
        auto_disabled_reason=(
            "host dispatch cannot prove coordinates are within the in-zone guard; "
            "broad-longitude workloads regress on RTX 4090"
        ),
        explicit_performance_min_elements=TMERC_FIXED_Q62_MIN_ELEMENTS,
    ),
    "tmerc-inverse": QualificationSpec(
        implementation_id=NATIVE_LIBDEVICE,
        operation="projection",
        domain="tmerc.inverse",
        direction="inverse",
        min_elements=0,
        coordinate_contract_m=1e-8,
    ),
    "helmert-forward": QualificationSpec(
        implementation_id=HELMERT_FIXED_Q62,
        operation="helmert",
        domain="global",
        direction="forward",
        min_elements=HELMERT_FIXED_Q62_MIN_ELEMENTS,
        coordinate_contract_m=1e-8,
    ),
    "helmert-inverse": QualificationSpec(
        implementation_id=HELMERT_FIXED_Q62,
        operation="helmert",
        domain="global",
        direction="inverse",
        min_elements=HELMERT_FIXED_Q62_MIN_ELEMENTS,
        coordinate_contract_m=1e-8,
    ),
    "sinu-forward": QualificationSpec(
        implementation_id=SINU_FORWARD_FIXED_Q62,
        operation="projection",
        domain="sinu.forward.spherical",
        direction="forward",
        min_elements=SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS,
        coordinate_contract_m=1e-8,
        max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
    ),
    "sinu-inverse": QualificationSpec(
        implementation_id=SINU_INVERSE_MERIDIONAL_RECURRENCE,
        auto_implementation_id=SINU_INVERSE_CONVERGENT_NEWTON,
        operation="projection",
        domain="sinu.inverse.ellipsoidal",
        direction="inverse",
        min_elements=SINU_INVERSE_CONVERGENT_NEWTON_MIN_ELEMENTS,
        coordinate_contract_m=1e-8,
        max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
        explicit_performance_min_elements=1,
        implementation_min_elements=0,
        representative_mixed_guard_sweep=True,
    ),
    "sinu-inverse-boundary": QualificationSpec(
        implementation_id=SINU_INVERSE_MERIDIONAL_RECURRENCE,
        auto_implementation_id=SINU_INVERSE_CONVERGENT_NEWTON,
        operation="projection",
        domain="sinu.inverse.ellipsoidal",
        direction="inverse",
        min_elements=SINU_INVERSE_CONVERGENT_NEWTON_MIN_ELEMENTS,
        coordinate_contract_m=1e-8,
        max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
        explicit_performance_min_elements=1,
        implementation_min_elements=0,
    ),
    "ortho-forward": QualificationSpec(
        implementation_id=ORTHO_FORWARD_FIXED_Q62,
        operation="projection",
        domain="ortho.forward.spherical.oblique",
        direction="forward",
        min_elements=ORTHO_FORWARD_FIXED_Q62_MIN_ELEMENTS,
        coordinate_contract_m=1e-8,
        max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
    ),
    "ortho-inverse": QualificationSpec(
        implementation_id=ORTHO_INVERSE_GUARDED_REFRAME,
        operation="projection",
        domain="ortho.inverse.spherical.equatorial",
        direction="inverse",
        min_elements=ORTHO_INVERSE_GUARDED_REFRAME_MIN_ELEMENTS,
        coordinate_contract_m=1e-8,
        max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
    ),
    "gnom-inverse-equatorial": QualificationSpec(
        implementation_id=GNOM_INVERSE_GUARDED_RSQRT_REFRAME,
        operation="projection",
        domain="gnom.inverse.spherical.equatorial",
        direction="inverse",
        min_elements=0,
        coordinate_contract_m=1e-8,
        max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
        auto_enabled=False,
        auto_disabled_reason=(
            "host dispatch cannot observe per-coordinate rho; random 10% cold mixtures "
            "regress to about 0.92x native on RTX 4090"
        ),
    ),
    "gnom-inverse-oblique": QualificationSpec(
        implementation_id=GNOM_INVERSE_GUARDED_RSQRT_REFRAME,
        operation="projection",
        domain="gnom.inverse.spherical.oblique_bounded",
        direction="inverse",
        min_elements=0,
        coordinate_contract_m=1e-8,
        max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
        auto_enabled=False,
        auto_disabled_reason=(
            "host dispatch cannot observe per-coordinate rho; random 10% cold mixtures "
            "regress to about 0.92x native on RTX 4090"
        ),
    ),
    **{
        f"geos-forward-{geometry}-sweep-{sweep}": QualificationSpec(
            implementation_id=GEOS_FORWARD_FIXED_Q62,
            operation="projection",
            domain=f"geos.forward.{geometry}.sweep_{sweep}",
            direction="forward",
            min_elements=GEOS_FORWARD_FIXED_Q62_MIN_ELEMENTS,
            coordinate_contract_m=1e-8,
            max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
        )
        for geometry in ("spherical", "ellipsoidal")
        for sweep in ("x", "y")
    },
    **{
        f"laea-forward-{geometry}-{pole}": QualificationSpec(
            implementation_id=LAEA_FORWARD_POLAR_FIXED_Q62,
            operation="projection",
            domain=f"laea.forward.{geometry}.{pole}_pole",
            direction="forward",
            min_elements=LAEA_FORWARD_POLAR_FIXED_Q62_MIN_ELEMENTS,
            coordinate_contract_m=1e-8,
            max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
        )
        for geometry in ("spherical",)
        for pole in ("north", "south")
    },
    **{
        f"stere-inverse-{variant}-{hemisphere}": QualificationSpec(
            implementation_id=STERE_INVERSE_FIXED_Q62,
            operation="projection",
            domain=f"stere.inverse.ellipsoidal.{variant}.{hemisphere}",
            direction="inverse",
            min_elements=STERE_INVERSE_FIXED_Q62_MIN_ELEMENTS,
            coordinate_contract_m=1e-8,
            max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
        )
        for variant, hemisphere in (
            ("variant_a", "north"),
            ("variant_a", "south"),
            ("variant_b", "north"),
            ("variant_b", "south"),
            ("variant_c", "south"),
        )
    },
    **{
        f"merc-{direction}-{geometry}-variant-{variant}": QualificationSpec(
            implementation_id=(
                (
                    MERC_FORWARD_SPHERICAL_PRODUCT_POLY
                    if geometry == "spherical"
                    else MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY
                )
                if direction == "forward"
                else MERC_INVERSE_EXP_SERIES
            ),
            operation="projection",
            domain=f"merc.{direction}.{geometry}.variant_{variant}",
            direction=direction,
            min_elements=(
                (MERC_FORWARD_SPHERICAL_PRODUCT_POLY_MIN_ELEMENTS if geometry == "spherical" else 0)
                if direction == "forward"
                else MERC_INVERSE_EXP_SERIES_MIN_ELEMENTS
            ),
            coordinate_contract_m=1e-8,
            max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
            auto_enabled=direction != "forward" or geometry == "spherical",
            auto_disabled_reason=(
                "10-50% random polar-cap mixtures make warp-wide native fallback "
                "slower than native-only execution"
                if direction == "forward" and geometry == "ellipsoidal"
                else None
            ),
            explicit_performance_min_elements=(
                MERC_FORWARD_SPHERICAL_PRODUCT_POLY_MIN_ELEMENTS
                if direction == "forward" and geometry == "ellipsoidal"
                else None
            ),
        )
        for direction in ("forward", "inverse")
        for geometry in ("spherical", "ellipsoidal")
        for variant in ("a", "b")
    },
    **{
        f"lcc-{direction}-{configuration}": QualificationSpec(
            implementation_id=(
                LCC_FORWARD_CONFORMAL_REFRAME
                if direction == "forward"
                else LCC_INVERSE_CONFORMAL_REFRAME
            ),
            operation="projection",
            domain=(
                f"lcc.{direction}.{geometry}.{variant}.regular_cone"
                if direction == "forward"
                else f"lcc.{direction}.{geometry}.{variant}"
            ),
            direction=direction,
            min_elements=(
                LCC_FORWARD_CONFORMAL_REFRAME_MIN_ELEMENTS
                if direction == "forward"
                else LCC_INVERSE_CONFORMAL_REFRAME_MIN_ELEMENTS
            ),
            coordinate_contract_m=1e-8,
            max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
        )
        for direction in ("forward", "inverse")
        for configuration, geometry, variant in (
            ("spherical-1sp-north", "spherical", "1sp"),
            ("spherical-2sp-north", "spherical", "2sp"),
            ("ellipsoidal-1sp-north", "ellipsoidal", "1sp"),
            ("ellipsoidal-2sp-north", "ellipsoidal", "2sp"),
            ("ellipsoidal-2sp-south", "ellipsoidal", "2sp"),
            ("ellipsoidal-2sp-zero", "ellipsoidal", "2sp"),
        )
    },
    **{
        f"lcc-forward-ellipsoidal-2sp-mix-{mix:04d}": QualificationSpec(
            implementation_id=LCC_FORWARD_CONFORMAL_REFRAME,
            operation="projection",
            domain="lcc.forward.ellipsoidal.2sp.regular_cone",
            direction="forward",
            min_elements=LCC_FORWARD_CONFORMAL_REFRAME_MIN_ELEMENTS,
            coordinate_contract_m=1e-8,
            max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
        )
        for mix in (0, 1, 10, 100, 500, 1000)
    },
    "lcc-forward-ellipsoidal-1sp-near-equator": QualificationSpec(
        implementation_id=NATIVE_LIBDEVICE,
        operation="projection",
        domain="lcc.forward.ellipsoidal.1sp.near_equator",
        direction="forward",
        min_elements=0,
        coordinate_contract_m=1e-8,
        max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
        auto_enabled=False,
        auto_disabled_reason="abs(n) below the proved 0.2 regular-cone boundary",
    ),
    "lcc-inverse-ellipsoidal-1sp-near-equator": QualificationSpec(
        implementation_id=LCC_INVERSE_CONFORMAL_REFRAME,
        operation="projection",
        domain="lcc.inverse.ellipsoidal.1sp",
        direction="inverse",
        min_elements=LCC_INVERSE_CONFORMAL_REFRAME_MIN_ELEMENTS,
        coordinate_contract_m=1e-8,
        max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
    ),
    **{
        f"krovak-inverse-epsg-{epsg}": QualificationSpec(
            implementation_id=KROVAK_INVERSE_GUARDED_LOG_RATIO,
            operation="projection",
            domain=(
                "krovak.inverse.ellipsoidal.standard_bessel.regular"
                if epsg in (2065, 5513, 8352)
                else "krovak.inverse.ellipsoidal.standard_bessel.north_oriented"
            ),
            direction="inverse",
            min_elements=KROVAK_INVERSE_GUARDED_LOG_RATIO_MIN_ELEMENTS,
            coordinate_contract_m=1e-8,
            auto_enabled=False,
            auto_disabled_reason=(
                "the guarded log-ratio is expert opt-in; automatic Krovak inverse "
                "remains native at every workload size"
            ),
            explicit_performance_min_elements=1,
            representative_mixed_guard_sweep=True,
        )
        for epsg in (2065, 5221, 5513, 5514, 8352, 8353)
    },
}
CASES = tuple(QUALIFICATION_SPECS)
MERC_CASES = tuple(name for name in CASES if name.startswith("merc-"))
LCC_CASES = tuple(name for name in CASES if name.startswith("lcc-"))
SINU_INVERSE_CASES = tuple(name for name in CASES if name.startswith("sinu-inverse"))
KROVAK_INVERSE_CASES = tuple(name for name in CASES if name.startswith("krovak-inverse"))
CASE_GROUPS = {
    "merc": MERC_CASES,
    "lcc": LCC_CASES,
    "sinu-inverse": SINU_INVERSE_CASES,
    "krovak-inverse": KROVAK_INVERSE_CASES,
}
WORKLOAD_GRID_CASES = tuple(
    name
    for name, qualification in QUALIFICATION_SPECS.items()
    if qualification.implementation_id != NATIVE_LIBDEVICE
)

# Retained no-go evidence from the Wave 2A production-shaped candidates. The
# corresponding private CUDA paths were removed after measurement; none can be
# selected by the registry. Times are 1M-coordinate fused medians on RTX 4090.
WAVE2A_REJECTED_CANDIDATES = {
    "aeqd.forward.bounded_acos": {
        "native_ms": [0.288, 0.292],
        "candidate_ms": [0.390, 0.394],
        "max_error_m": [1.67e-6, 2.73e-6],
        "reason": "slower and exceeds the 1e-8 m native-relative contract",
    },
    "aeqd.forward.reduced_half_angle_scale": {
        "native_ms": [0.288, 0.292],
        "candidate_ms": [0.318, 0.322],
        "max_error_m": [2.77e-8, 3.00e-8],
        "reason": "slower and exceeds the 1e-8 m native-relative contract",
    },
    "aeqd.inverse.bounded_asin_q62_sincos": {
        "native_ms": [0.302, 0.305],
        "candidate_ms": [0.628, 0.695],
        "max_error_m": [3.8e-9, 6.4e-9],
        "reason": "accuracy passes but synchronized kernel execution is substantially slower",
    },
    "ortho.inverse.bounded_rational": {
        "native_ms": 0.3843,
        "candidate_ms": 0.6042,
        "max_error_m": 6.77e-9,
        "reason": "accuracy passes but mixed guarded execution is slower",
    },
}

# Retained Wave 3C numeric no-go evidence. These rejected variants cannot be
# selected by production policy.
WAVE3C_LCC_REJECTED_CANDIDATES = {
    "lcc.forward.exp_isometric_explicit_zero": {
        "max_error_m": 1.21e-8,
        "reason": "exceeds the 1e-8 m native-relative contract",
    },
    "lcc.forward.guarded_logtan": {
        "speedup_vs_native": [0.72, 0.92],
        "reason": "full-domain warp guards make production-shaped execution slower",
    },
    "lcc.inverse.two_step_newton": {
        "max_error_m": [1.86e-3, 2.91e-3],
        "reason": "millimetre error exceeds the 1e-8 m native-relative contract",
    },
    "lcc.inverse.two_step_direct": {
        "max_error_m": [2.3e-4, 3.6e-4],
        "reason": "sub-millimetre error still exceeds the 1e-8 m contract",
    },
}

WAVE3C_LCC_FORWARD_THRESHOLD_EVIDENCE = {
    "rejected_auto_minimums": [4_096, 8_192, 16_384, 32_768],
    "selected_auto_minimum": LCC_FORWARD_CONFORMAL_REFRAME_MIN_ELEMENTS,
    "rule": "lowest size with meaningful >=1.05 wall margin in every 3x30 repeat/domain",
    "selected_screen_worst_wall_repeat": {
        "accelerated": 1.0742620020953595,
        "auto": 1.0760746548140832,
    },
}


def _qualification_workload_sizes(
    requested_sizes: tuple[int, ...],
    qualification: QualificationSpec,
) -> tuple[int, ...]:
    """Return the requested grid plus crossover boundaries where auto is enabled."""
    if not requested_sizes:
        return ()
    sizes = set(requested_sizes)
    if qualification.auto_enabled:
        if qualification.min_elements == 1:
            # Public transforms are nonempty. N=1 is the exact lower endpoint;
            # retain N=2 as its adjacent crossover confirmation instead of
            # fabricating an unlaunchable N=0 workload.
            sizes.update((1, 2))
        else:
            sizes.update((qualification.min_elements - 1, qualification.min_elements))
    if qualification.explicit_performance_min_elements is not None:
        sizes.add(qualification.explicit_performance_min_elements)
    return tuple(sorted(sizes))


def _candidate_policy_ids(qualification: QualificationSpec) -> dict[str, str]:
    """Return the exact implementation qualified by each candidate policy."""
    return {
        "accelerated": qualification.implementation_id,
        "auto": qualification.auto_implementation_id or qualification.implementation_id,
    }


def _distinct_candidate_policies(qualification: QualificationSpec) -> tuple[str, ...]:
    """Return candidate policies requiring independent qualification evidence."""
    policy_ids = _candidate_policy_ids(qualification)
    if qualification.auto_enabled and policy_ids["auto"] != policy_ids["accelerated"]:
        return ("accelerated", "auto")
    return ("accelerated",)


@dataclass(frozen=True)
class BenchmarkCase:
    """Prepared GPU inputs and metadata for one transform direction."""

    name: str
    family: str
    direction: str
    transformer: Any
    input_x: Any
    input_y: Any
    host_x: np.ndarray
    host_y: np.ndarray
    edge_host_x: np.ndarray
    edge_host_y: np.ndarray
    domain: dict[str, Any]
    edges: dict[str, list[float | str]]
    output_is_geographic: bool
    oracle_from_crs: str
    oracle_to_crs: str
    qualification: QualificationSpec
    geographic_scale_m: float = EARTH_RADIUS_M


def _device_metadata(cp: Any, device_id: int) -> dict[str, Any]:
    device = cp.cuda.Device(device_id)
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    raw_name = properties["name"]
    name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    ratio = int(device.attributes.get("SingleToDoublePrecisionPerfRatio", 0))
    return {
        "id": device_id,
        "name": name,
        "compute_capability": f"{int(properties['major'])}.{int(properties['minor'])}",
        "sm": f"sm_{int(properties['major'])}{int(properties['minor'])}",
        "multiprocessor_count": int(properties["multiProcessorCount"]),
        "single_to_double_precision_perf_ratio": ratio,
    }


def _prepare_tmerc_forward(cp: Any, n: int, rng: np.random.Generator) -> BenchmarkCase:
    from vibeproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    longitude = rng.uniform(0.0, 6.0, n).astype(np.float64)
    latitude = rng.uniform(-80.0, 84.0, n).astype(np.float64)
    boundary_deg = math.degrees(0.06)
    edge_lon = [
        3.0 - boundary_deg,
        np.nextafter(3.0 - boundary_deg, 3.0),
        3.0,
        np.nextafter(3.0 + boundary_deg, 3.0),
        3.0 + boundary_deg,
        np.nextafter(3.0 + boundary_deg, math.inf),
        -57.0,
        63.0,
        3.0,
        3.0,
    ]
    edge_lat = [-80.0, -45.0, 0.0, 45.0, 84.0, 80.0, -60.0, 60.0, -100.0, 100.0]
    return BenchmarkCase(
        name="tmerc-forward",
        family="tmerc",
        direction="FORWARD",
        transformer=transformer,
        input_x=cp.asarray(longitude),
        input_y=cp.asarray(latitude),
        host_x=longitude,
        host_y=latitude,
        edge_host_x=np.asarray([*edge_lon, np.nan, np.inf, -np.inf], dtype=np.float64),
        edge_host_y=np.asarray([*edge_lat, 0.0, 0.0, 0.0], dtype=np.float64),
        domain={
            "crs": "EPSG:4326 -> EPSG:32631",
            "longitude_degrees": [0.0, 6.0],
            "latitude_degrees": [-80.0, 84.0],
            "utm_central_meridian_degrees": 3.0,
        },
        edges={
            "longitude_degrees": [*edge_lon, "nan", "+inf", "-inf"],
            "latitude_degrees": [*edge_lat, 0.0, 0.0, 0.0],
        },
        output_is_geographic=False,
        oracle_from_crs="EPSG:4326",
        oracle_to_crs="EPSG:32631",
        qualification=QUALIFICATION_SPECS["tmerc-forward"],
    )


def _prepare_tmerc_inverse(cp: Any, n: int, rng: np.random.Generator) -> BenchmarkCase:
    from pyproj import Transformer as PyProjTransformer

    from vibeproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    longitude = rng.uniform(0.0, 6.0, n).astype(np.float64)
    latitude = rng.uniform(-80.0, 84.0, n).astype(np.float64)
    oracle = PyProjTransformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    easting, northing = oracle.transform(longitude, latitude)
    easting = np.asarray(easting, dtype=np.float64)
    northing = np.asarray(northing, dtype=np.float64)
    edge_easting = [166_021.443, 500_000.0, 833_978.557, -5_000_000.0, 6_000_000.0]
    edge_northing = [0.0, 9_328_093.831, 10_000_000.0, -1_000_000.0, 11_000_000.0]
    return BenchmarkCase(
        name="tmerc-inverse",
        family="tmerc",
        direction="INVERSE",
        transformer=transformer,
        input_x=cp.asarray(easting),
        input_y=cp.asarray(northing),
        host_x=easting,
        host_y=northing,
        edge_host_x=np.asarray([*edge_easting, np.nan, np.inf, -np.inf], dtype=np.float64),
        edge_host_y=np.asarray([*edge_northing, 0.0, 0.0, 0.0], dtype=np.float64),
        domain={
            "crs": "EPSG:32631 -> EPSG:4326 (inverse call)",
            "source_longitude_degrees": [0.0, 6.0],
            "source_latitude_degrees": [-80.0, 84.0],
        },
        edges={
            "easting_m": [*edge_easting, "nan", "+inf", "-inf"],
            "northing_m": [*edge_northing, 0.0, 0.0, 0.0],
        },
        output_is_geographic=True,
        oracle_from_crs="EPSG:4326",
        oracle_to_crs="EPSG:32631",
        qualification=QUALIFICATION_SPECS["tmerc-inverse"],
    )


def _prepare_helmert(cp: Any, n: int, rng: np.random.Generator, direction: str) -> BenchmarkCase:
    from pyproj import Transformer as PyProjTransformer
    from vibeproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4277", always_xy=True)
    longitude = rng.uniform(-8.0, 2.0, n).astype(np.float64)
    latitude = rng.uniform(49.0, 61.0, n).astype(np.float64)
    if direction == "FORWARD":
        input_x, input_y = longitude, latitude
        crs = "EPSG:4326 -> EPSG:4277"
        edge_x = [-180.0, 0.0, 180.0, 360.0, -540.0, 720.0]
        edge_y = [-90.0, -89.0, 0.0, 89.0, 90.0, 45.0]
    else:
        oracle = PyProjTransformer.from_crs("EPSG:4326", "EPSG:4277", always_xy=True)
        input_x, input_y = oracle.transform(longitude, latitude)
        input_x = np.asarray(input_x, dtype=np.float64)
        input_y = np.asarray(input_y, dtype=np.float64)
        crs = "EPSG:4277 -> EPSG:4326 (inverse call)"
        edge_x = [-180.0, 0.0, 180.0, 360.0, -540.0, 720.0]
        edge_y = [-90.0, -89.0, 0.0, 89.0, 90.0, 45.0]
    name = f"helmert-{direction.lower()}"
    return BenchmarkCase(
        name=name,
        family="helmert",
        direction=direction,
        transformer=transformer,
        input_x=cp.asarray(input_x),
        input_y=cp.asarray(input_y),
        host_x=input_x,
        host_y=input_y,
        edge_host_x=np.asarray([*edge_x, np.nan, np.inf, -np.inf], dtype=np.float64),
        edge_host_y=np.asarray([*edge_y, 0.0, 0.0, 0.0], dtype=np.float64),
        domain={
            "crs": crs,
            "source_longitude_degrees": [-8.0, 2.0],
            "source_latitude_degrees": [49.0, 61.0],
        },
        edges={
            "longitude_degrees": [*edge_x, "nan", "+inf", "-inf"],
            "latitude_degrees": [*edge_y, 0.0, 0.0, 0.0],
        },
        output_is_geographic=True,
        oracle_from_crs="EPSG:4326",
        oracle_to_crs="EPSG:4277",
        qualification=QUALIFICATION_SPECS[name],
    )


_SPHERICAL_LONG_LAT_CRS = "+proj=longlat +R=6378137 +type=crs"


def _json_edge_values(values: np.ndarray) -> list[float | str]:
    result: list[float | str] = []
    for value in values:
        if np.isnan(value):
            result.append("nan")
        elif np.isposinf(value):
            result.append("+inf")
        elif np.isneginf(value):
            result.append("-inf")
        else:
            result.append(float(value))
    return result


def _prepare_geographic_projection_forward(
    cp: Any,
    n: int,
    rng: np.random.Generator,
    *,
    name: str,
    target_crs: str,
    longitude_range: tuple[float, float],
    latitude_range: tuple[float, float],
    edge_longitude: np.ndarray,
    edge_latitude: np.ndarray,
    source_crs: str = _SPHERICAL_LONG_LAT_CRS,
    family: str | None = None,
    uniform_surface: bool = False,
) -> BenchmarkCase:
    """Prepare one reusable geographic forward-projection qualification case."""
    from vibeproj import Transformer

    transformer = Transformer.from_crs(
        source_crs,
        target_crs,
        always_xy=True,
    )
    longitude = rng.uniform(*longitude_range, n).astype(np.float64)
    if uniform_surface:
        sin_min, sin_max = np.sin(np.deg2rad(latitude_range))
        latitude = np.rad2deg(
            np.arcsin(rng.uniform(min(sin_min, sin_max), max(sin_min, sin_max), n))
        ).astype(np.float64)
    else:
        latitude = rng.uniform(*latitude_range, n).astype(np.float64)
    resolved_family = family or name.removesuffix("-forward")
    return BenchmarkCase(
        name=name,
        family=resolved_family,
        direction="FORWARD",
        transformer=transformer,
        input_x=cp.asarray(longitude),
        input_y=cp.asarray(latitude),
        host_x=longitude,
        host_y=latitude,
        edge_host_x=edge_longitude,
        edge_host_y=edge_latitude,
        domain={
            "crs": f"{source_crs} -> {target_crs}",
            "longitude_degrees": list(longitude_range),
            "latitude_degrees": list(latitude_range),
        },
        edges={
            "longitude_degrees": _json_edge_values(edge_longitude),
            "latitude_degrees": _json_edge_values(edge_latitude),
        },
        output_is_geographic=False,
        oracle_from_crs=source_crs,
        oracle_to_crs=target_crs,
        qualification=QUALIFICATION_SPECS[name],
    )


def _prepare_sinu_forward(cp: Any, n: int, rng: np.random.Generator) -> BenchmarkCase:
    edge_latitude = np.asarray(
        [
            -90.0,
            np.nextafter(-90.0, math.inf),
            0.0,
            np.nextafter(90.0, -math.inf),
            90.0,
            np.nextafter(-90.0, -math.inf),
            np.nextafter(90.0, math.inf),
            -100.0,
            100.0,
            -np.inf,
            np.inf,
            np.nan,
            40.0,
            40.0,
        ],
        dtype=np.float64,
    )
    edge_longitude = np.asarray(
        [-180.0, -180.0, 0.0, 180.0, 180.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.inf, np.nan],
        dtype=np.float64,
    )
    return _prepare_geographic_projection_forward(
        cp,
        n,
        rng,
        name="sinu-forward",
        target_crs="+proj=sinu +lon_0=0 +R=6378137 +units=m +type=crs",
        longitude_range=(-170.0, 170.0),
        latitude_range=(-80.0, 80.0),
        edge_longitude=edge_longitude,
        edge_latitude=edge_latitude,
    )


def _prepare_sinu_inverse(
    cp: Any,
    n: int,
    rng: np.random.Generator,
    *,
    name: str,
) -> BenchmarkCase:
    """Prepare the ellipsoidal inverse hot domain and exact meridional edges."""
    from pyproj import CRS
    from pyproj import Transformer as PyProjTransformer
    from vibeproj import Transformer

    target_crs = (
        "+proj=sinu +lon_0=0 +a=6400000 +es=.012 +units=m +type=crs"
        if name == "sinu-inverse-boundary"
        else "ESRI:54008"
    )
    target = CRS.from_user_input(target_crs)
    geographic_crs = target.geodetic_crs.to_string()
    transformer = Transformer.from_crs(target, target.geodetic_crs, always_xy=True)
    longitude = rng.uniform(-179.0, 179.0, n).astype(np.float64)
    latitude = rng.uniform(-89.9, 89.9, n).astype(np.float64)
    oracle_forward = PyProjTransformer.from_crs(target.geodetic_crs, target, always_xy=True)
    easting, northing = oracle_forward.transform(longitude, latitude)
    easting = np.asarray(easting, dtype=np.float64)
    northing = np.asarray(northing, dtype=np.float64)
    computed = transformer._pipeline.computed
    pole = float(computed["y0"] + computed["a"] * computed["meridional_pole"])
    edge_easting = np.asarray(
        [0.0, 1.0, 1.0e7, 1.0, 1.0, math.inf, -math.inf, math.nan, 0.0],
        dtype=np.float64,
    )
    edge_northing = np.asarray(
        [
            0.0,
            math.nextafter(pole, 0.0),
            pole,
            math.nextafter(pole, math.inf),
            -pole,
            0.0,
            0.0,
            0.0,
            math.nan,
        ],
        dtype=np.float64,
    )
    return BenchmarkCase(
        name=name,
        family="sinu",
        direction="FORWARD",
        transformer=transformer,
        input_x=cp.asarray(easting),
        input_y=cp.asarray(northing),
        host_x=easting,
        host_y=northing,
        edge_host_x=edge_easting,
        edge_host_y=edge_northing,
        domain={
            "crs": f"{target.to_string()} -> {geographic_crs}",
            "setup": (
                "exact qualified es=0.012, a=6400000m boundary"
                if name == "sinu-inverse-boundary"
                else "ESRI:54008 / WGS84 representative"
            ),
            "source_longitude_degrees": [-179.0, 179.0],
            "source_latitude_degrees": [-89.9, 89.9],
            "recurrence_hot_latitude_degrees": [-89.9, 89.9],
            "auto_coordinate_domain": "complete native inverse image",
        },
        edges={
            "easting_m": _json_edge_values(edge_easting),
            "northing_m": _json_edge_values(edge_northing),
        },
        output_is_geographic=True,
        oracle_from_crs=target.to_string(),
        oracle_to_crs=geographic_crs,
        qualification=QUALIFICATION_SPECS[name],
        geographic_scale_m=float(computed["a"]),
    )


def _prepare_ortho_forward(cp: Any, n: int, rng: np.random.Generator) -> BenchmarkCase:
    edge_latitude = np.asarray(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            -45.0,
            -90.0,
            90.0,
            np.nextafter(-90.0, -math.inf),
            np.nextafter(90.0, math.inf),
            -100.0,
            100.0,
            -np.inf,
            np.inf,
            np.nan,
            40.0,
            40.0,
        ],
        dtype=np.float64,
    )
    edge_longitude = np.asarray(
        [
            np.nextafter(90.0, 0.0),
            90.0,
            np.nextafter(90.0, math.inf),
            120.0,
            180.0,
            -180.0,
            180.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            np.inf,
            np.nan,
        ],
        dtype=np.float64,
    )
    return _prepare_geographic_projection_forward(
        cp,
        n,
        rng,
        name="ortho-forward",
        target_crs="+proj=ortho +lat_0=45 +lon_0=0 +R=6378137 +units=m +type=crs",
        longitude_range=(-30.0, 30.0),
        latitude_range=(20.0, 70.0),
        edge_longitude=edge_longitude,
        edge_latitude=edge_latitude,
    )


def _prepare_ortho_inverse(cp: Any, n: int, rng: np.random.Generator) -> BenchmarkCase:
    """Prepare the proved spherical-equatorial interior disk and guard edges."""
    from vibeproj import Transformer

    radius = EARTH_RADIUS_M
    target_crs = "+proj=ortho +lat_0=0 +lon_0=0 +R=6378137 +units=m +type=crs"
    rho = np.sqrt(rng.uniform(1e-16, np.nextafter(1.0, 0.0), n)) * radius
    azimuth = rng.uniform(-math.pi, math.pi, n)
    easting = (rho * np.cos(azimuth)).astype(np.float64)
    northing = (rho * np.sin(azimuth)).astype(np.float64)
    guard_radius = math.sqrt(0.99) * radius
    edge_easting = np.asarray(
        [
            0.0,
            -0.0,
            0.5 * radius,
            -0.5 * radius,
            0.0,
            guard_radius,
            np.nextafter(guard_radius, math.inf),
            radius,
            np.nextafter(radius, math.inf),
            1.1 * radius,
            math.inf,
            -math.inf,
            math.nan,
        ],
        dtype=np.float64,
    )
    edge_northing = np.asarray(
        [0.0, -0.0, 0.0, 0.0, 0.5 * radius] + [0.0] * 8,
        dtype=np.float64,
    )
    return BenchmarkCase(
        name="ortho-inverse",
        family="ortho",
        direction="FORWARD",
        transformer=Transformer.from_crs(target_crs, _SPHERICAL_LONG_LAT_CRS, always_xy=True),
        input_x=cp.asarray(easting),
        input_y=cp.asarray(northing),
        host_x=easting,
        host_y=northing,
        edge_host_x=edge_easting,
        edge_host_y=edge_northing,
        domain={
            "crs": f"{target_crs} -> {_SPHERICAL_LONG_LAT_CRS}",
            "normalized_rho_squared_sampling_interval": "[1e-16, 1.0)",
            "accelerated_guard": "1e-16 < rho_squared <= 0.99; non-axis finite points",
            "origin_mode": "equatorial",
        },
        edges={
            "easting_m": _json_edge_values(edge_easting),
            "northing_m": _json_edge_values(edge_northing),
        },
        output_is_geographic=True,
        oracle_from_crs=target_crs,
        oracle_to_crs=_SPHERICAL_LONG_LAT_CRS,
        qualification=QUALIFICATION_SPECS["ortho-inverse"],
    )


def _prepare_gnom_inverse(
    cp: Any,
    n: int,
    rng: np.random.Generator,
    *,
    origin: str,
) -> BenchmarkCase:
    """Prepare the production spherical Gnom inverse guard and cold edges."""
    from vibeproj import Transformer

    latitude_origin = 0.0 if origin == "equatorial" else 45.0
    name = f"gnom-inverse-{origin}"
    radius = EARTH_RADIUS_M
    target_crs = f"+proj=gnom +lat_0={latitude_origin:.17g} +lon_0=0 +R=6378137 +units=m +type=crs"
    rho_squared = rng.uniform(1e-12, np.nextafter(0.02, 0.0), n)
    azimuth = rng.uniform(-math.pi, math.pi, n)
    rho = np.sqrt(rho_squared) * radius
    easting = (rho * np.cos(azimuth)).astype(np.float64)
    northing = (rho * np.sin(azimuth)).astype(np.float64)

    minimum_radius = 1e-12
    maximum_radius = math.sqrt(0.02)
    diagonal = math.sqrt(0.5)
    edge_normalized_x = np.asarray(
        [
            0.0,
            -0.0,
            0.05,
            0.0,
            diagonal * np.nextafter(minimum_radius, 0.0),
            diagonal * minimum_radius,
            diagonal * np.nextafter(minimum_radius, math.inf),
            diagonal * np.nextafter(maximum_radius, 0.0),
            diagonal * maximum_radius,
            diagonal * np.nextafter(maximum_radius, math.inf),
            0.2,
            math.inf,
            -math.inf,
            math.nan,
        ],
        dtype=np.float64,
    )
    edge_normalized_y = np.asarray(
        [
            0.0,
            -0.0,
            0.0,
            0.05,
            diagonal * np.nextafter(minimum_radius, 0.0),
            diagonal * minimum_radius,
            diagonal * np.nextafter(minimum_radius, math.inf),
            diagonal * np.nextafter(maximum_radius, 0.0),
            diagonal * maximum_radius,
            diagonal * np.nextafter(maximum_radius, math.inf),
            0.2,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    edge_easting = edge_normalized_x * radius
    edge_northing = edge_normalized_y * radius
    return BenchmarkCase(
        name=name,
        family="gnom",
        direction="FORWARD",
        transformer=Transformer.from_crs(target_crs, _SPHERICAL_LONG_LAT_CRS, always_xy=True),
        input_x=cp.asarray(easting),
        input_y=cp.asarray(northing),
        host_x=easting,
        host_y=northing,
        edge_host_x=edge_easting,
        edge_host_y=edge_northing,
        domain={
            "crs": f"{target_crs} -> {_SPHERICAL_LONG_LAT_CRS}",
            "normalized_rho_squared_sampling_interval": "[1e-12, 0.02)",
            "accelerated_guard": "1e-24 < rho_squared <= 0.02; non-axis finite points",
            "origin_mode": origin,
            "minimum_abs_cos_phi0": 0.5,
        },
        edges={
            "easting_m": _json_edge_values(edge_easting),
            "northing_m": _json_edge_values(edge_northing),
        },
        output_is_geographic=True,
        oracle_from_crs=target_crs,
        oracle_to_crs=_SPHERICAL_LONG_LAT_CRS,
        qualification=QUALIFICATION_SPECS[name],
    )


def _earth_crs_parts(geometry: str) -> tuple[str, float, float]:
    if geometry == "spherical":
        return "+R=6378137", EARTH_RADIUS_M, EARTH_RADIUS_M
    return "+ellps=WGS84", EARTH_RADIUS_M, 6_356_752.314245179


def _geos_visible_samples(
    n: int,
    rng: np.random.Generator,
    *,
    equatorial_radius: float,
    polar_radius: float,
    satellite_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the visible ellipsoid without biasing the timed set to the limb."""
    longitude_parts: list[np.ndarray] = []
    latitude_parts: list[np.ndarray] = []
    remaining = n
    satellite_distance = satellite_height + equatorial_radius
    flattening_ratio = (polar_radius / equatorial_radius) ** 2
    while remaining:
        batch = max(4_096, int(remaining * 2.75))
        longitude = rng.uniform(-math.pi, math.pi, batch)
        latitude = np.arcsin(rng.uniform(-1.0, 1.0, batch))
        phi_gc = np.arctan(flattening_ratio * np.tan(latitude))
        cos_phi_gc = np.cos(phi_gc)
        r_earth = polar_radius / np.sqrt(1.0 - (1.0 - flattening_ratio) * cos_phi_gc * cos_phi_gc)
        point_x = r_earth * cos_phi_gc * np.cos(longitude)
        visible = satellite_distance * point_x >= equatorial_radius**2
        count = min(remaining, int(np.count_nonzero(visible)))
        longitude_parts.append(longitude[visible][:count])
        latitude_parts.append(latitude[visible][:count])
        remaining -= count
    return (
        np.rad2deg(np.concatenate(longitude_parts)),
        np.rad2deg(np.concatenate(latitude_parts)),
    )


def _geos_limb_samples(
    n: int,
    rng: np.random.Generator,
    *,
    equatorial_radius: float,
    polar_radius: float,
    satellite_height: float,
    outside: bool,
) -> tuple[np.ndarray, np.ndarray]:
    satellite_distance = satellite_height + equatorial_radius
    flattening_ratio = (polar_radius / equatorial_radius) ** 2
    latitude_parts: list[np.ndarray] = []
    longitude_parts: list[np.ndarray] = []
    remaining = n
    while remaining:
        batch = max(4_096, int(remaining * 1.5))
        latitude = np.arcsin(
            rng.uniform(-math.sin(math.radians(81.0)), math.sin(math.radians(81.0)), batch)
        )
        phi_gc = np.arctan(flattening_ratio * np.tan(latitude))
        cos_phi_gc = np.cos(phi_gc)
        r_earth = polar_radius / np.sqrt(1.0 - (1.0 - flattening_ratio) * cos_phi_gc * cos_phi_gc)
        ratio = equatorial_radius**2 / (satellite_distance * r_earth * cos_phi_gc)
        valid = np.isfinite(ratio) & (ratio < 1.0)
        latitude = latitude[valid]
        longitude_limit = np.arccos(np.clip(ratio[valid], -1.0, 1.0))
        count = min(remaining, latitude.size)
        distance = np.power(10.0, rng.uniform(-12.0, -3.0, count))
        factor = 1.0 + distance if outside else 1.0 - distance
        sign = np.where(rng.random(count) < 0.5, -1.0, 1.0)
        latitude_parts.append(latitude[:count])
        longitude_parts.append(sign * factor * longitude_limit[:count])
        remaining -= count
    return (
        np.rad2deg(np.concatenate(longitude_parts)),
        np.rad2deg(np.concatenate(latitude_parts)),
    )


def _geos_analytic_limb_samples(
    *,
    equatorial_radius: float,
    polar_radius: float,
    satellite_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact/adjacent analytic limb longitudes over both hemispheres."""
    satellite_distance = satellite_height + equatorial_radius
    flattening_ratio = (polar_radius / equatorial_radius) ** 2
    latitude = np.linspace(-80.0, 80.0, 513, dtype=np.float64)
    latitude_rad = np.deg2rad(latitude)
    phi_gc = np.arctan(flattening_ratio * np.tan(latitude_rad))
    cos_phi_gc = np.cos(phi_gc)
    r_earth = polar_radius / np.sqrt(1.0 - (1.0 - flattening_ratio) * cos_phi_gc * cos_phi_gc)
    ratio = equatorial_radius**2 / (satellite_distance * r_earth * cos_phi_gc)
    valid = np.isfinite(ratio) & (ratio < 1.0)
    latitude = latitude[valid]
    limb = np.rad2deg(np.arccos(np.clip(ratio[valid], -1.0, 1.0)))

    positive = np.stack(
        (np.nextafter(limb, 0.0), limb, np.nextafter(limb, math.inf)),
        axis=1,
    )
    negative_limb = -limb
    negative = np.stack(
        (
            np.nextafter(negative_limb, 0.0),
            negative_limb,
            np.nextafter(negative_limb, -math.inf),
        ),
        axis=1,
    )
    longitude = np.concatenate((positive, negative), axis=1).reshape(-1)
    return longitude, np.repeat(latitude, 6)


def _prepare_geos_forward(
    cp: Any,
    n: int,
    rng: np.random.Generator,
    *,
    geometry: str,
    sweep: str,
) -> BenchmarkCase:
    from vibeproj import Transformer

    earth, equatorial_radius, polar_radius = _earth_crs_parts(geometry)
    satellite_height = 35_785_831.0
    source_crs = f"+proj=longlat {earth} +type=crs"
    target_crs = (
        f"+proj=geos +lon_0=0 +h={satellite_height:g} +sweep={sweep} {earth} +units=m +type=crs"
    )
    longitude, latitude = _geos_visible_samples(
        n,
        rng,
        equatorial_radius=equatorial_radius,
        polar_radius=polar_radius,
        satellite_height=satellite_height,
    )
    inside_lon, inside_lat = _geos_limb_samples(
        4_096,
        rng,
        equatorial_radius=equatorial_radius,
        polar_radius=polar_radius,
        satellite_height=satellite_height,
        outside=False,
    )
    outside_lon, outside_lat = _geos_limb_samples(
        1_024,
        rng,
        equatorial_radius=equatorial_radius,
        polar_radius=polar_radius,
        satellite_height=satellite_height,
        outside=True,
    )
    analytic_lon, analytic_lat = _geos_analytic_limb_samples(
        equatorial_radius=equatorial_radius,
        polar_radius=polar_radius,
        satellite_height=satellite_height,
    )
    edge_longitude = np.concatenate(
        (
            analytic_lon,
            inside_lon,
            outside_lon,
            [0.0, 0.0, np.inf, -np.inf, np.nan],
        )
    ).astype(np.float64)
    edge_latitude = np.concatenate(
        (
            analytic_lat,
            inside_lat,
            outside_lat,
            [90.0, -90.0, 0.0, 0.0, 0.0],
        )
    ).astype(np.float64)
    name = f"geos-forward-{geometry}-sweep-{sweep}"
    return BenchmarkCase(
        name=name,
        family="geos",
        direction="FORWARD",
        transformer=Transformer.from_crs(source_crs, target_crs, always_xy=True),
        input_x=cp.asarray(longitude),
        input_y=cp.asarray(latitude),
        host_x=longitude,
        host_y=latitude,
        edge_host_x=edge_longitude,
        edge_host_y=edge_latitude,
        domain={
            "crs": f"{source_crs} -> {target_crs}",
            "timed_distribution": "uniform-area sphere samples conditioned on visibility",
            "randomized_limb_stress_coordinates": int(inside_lon.size + outside_lon.size),
            "analytic_exact_nextafter_limb_coordinates": int(analytic_lon.size),
            "sweep_axis": sweep,
            "geometry": geometry,
        },
        edges={
            "longitude_degrees": _json_edge_values(edge_longitude),
            "latitude_degrees": _json_edge_values(edge_latitude),
        },
        output_is_geographic=False,
        oracle_from_crs=source_crs,
        oracle_to_crs=target_crs,
        qualification=QUALIFICATION_SPECS[name],
    )


def _prepare_laea_polar_forward(
    cp: Any,
    n: int,
    rng: np.random.Generator,
    *,
    geometry: str,
    pole: str,
) -> BenchmarkCase:
    earth, _, _ = _earth_crs_parts(geometry)
    source_crs = f"+proj=longlat {earth} +type=crs"
    latitude_origin = 90.0 if pole == "north" else -90.0
    antipode = -latitude_origin
    edge_longitude = np.asarray(
        [-180.0, 0.0, 180.0, 0.0, 0.0, np.inf, -np.inf, np.nan],
        dtype=np.float64,
    )
    edge_latitude = np.asarray(
        [
            latitude_origin,
            latitude_origin,
            antipode,
            np.nextafter(antipode, 0.0),
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    name = f"laea-forward-{geometry}-{pole}"
    return _prepare_geographic_projection_forward(
        cp,
        n,
        rng,
        name=name,
        target_crs=(f"+proj=laea +lat_0={latitude_origin:g} +lon_0=0 {earth} +units=m +type=crs"),
        longitude_range=(-180.0, 180.0),
        latitude_range=(-90.0, 90.0),
        edge_longitude=edge_longitude,
        edge_latitude=edge_latitude,
        source_crs=source_crs,
        family="laea",
        uniform_surface=True,
    )


_STERE_INVERSE_EPSG = {
    ("variant_a", "north"): 32661,
    ("variant_a", "south"): 32761,
    ("variant_b", "north"): 3413,
    ("variant_b", "south"): 3031,
    ("variant_c", "south"): 2985,
}


def _prepare_stere_inverse(
    cp: Any,
    n: int,
    rng: np.random.Generator,
    *,
    variant: str,
    hemisphere: str,
) -> BenchmarkCase:
    """Prepare realistic polar timing plus separate full/log-plane edges."""
    from pyproj import CRS

    from vibeproj import Transformer

    epsg = _STERE_INVERSE_EPSG[(variant, hemisphere)]
    projected = CRS.from_epsg(epsg)
    oracle_projected = projected
    if variant == "variant_c":
        oracle_projected = CRS.from_user_input(
            "+proj=stere +lat_0=-90 +lat_ts=-67 +lon_0=140 "
            "+x_0=300000 +y_0=200000 +a=6378388 +rf=297 +units=m +type=crs"
        )
    transformer = Transformer.from_crs(projected, projected.geodetic_crs, always_xy=True)
    computed = transformer._pipeline_for_direction("FORWARD").computed
    sign = computed["sign"]
    latitude = sign * np.deg2rad(rng.uniform(45.0, 90.0, n))
    longitude = rng.uniform(-math.pi, math.pi, n)
    phi_adjusted = sign * latitude
    sin_phi = np.sin(phi_adjusted)
    e_sin = computed["e"] * sin_phi
    t = np.tan(0.5 * (math.pi / 2.0 - phi_adjusted)) / ((1.0 - e_sin) / (1.0 + e_sin)) ** (
        0.5 * computed["e"]
    )

    rho = computed["akm1"] * t
    easting = computed["x0"] + computed["a"] * rho * np.sin(longitude)
    northing = computed["y0"] - sign * computed["a"] * rho * np.cos(longitude)

    log_rho = np.power(10.0, np.linspace(-15.0, 140.0, 96))
    edge_azimuth = np.linspace(-math.pi, math.pi, log_rho.size, endpoint=False)
    edge_easting = computed["x0"] + computed["a"] * log_rho * np.cos(edge_azimuth)
    edge_northing = computed["y0"] + computed["a"] * log_rho * np.sin(edge_azimuth)
    edge_easting = np.concatenate(
        (
            [computed["x0"], np.nextafter(computed["x0"], math.inf)],
            edge_easting,
            [1e300, -1e300, math.inf, -math.inf, math.nan],
        )
    ).astype(np.float64)
    edge_northing = np.concatenate(
        (
            [computed["y0"], computed["y0"]],
            edge_northing,
            [-1e300, 1e300, 0.0, 0.0, math.nan],
        )
    ).astype(np.float64)
    name = f"stere-inverse-{variant}-{hemisphere}"
    return BenchmarkCase(
        name=name,
        family="stere",
        direction="FORWARD",
        transformer=transformer,
        input_x=cp.asarray(easting),
        input_y=cp.asarray(northing),
        host_x=easting,
        host_y=northing,
        edge_host_x=edge_easting,
        edge_host_y=edge_northing,
        domain={
            "crs": f"EPSG:{epsg} -> its geodetic CRS",
            "timed_distribution": "full longitude, polar latitude magnitude uniform [45, 90] degrees",
            "adversarial_accuracy_only": "normalized radius logspace 1e-15 through 1e140",
            "geometry": "ellipsoidal",
            "variant": variant,
            "hemisphere": hemisphere,
            "eccentricity_guard": "0.05 <= e <= 0.2",
        },
        edges={
            "easting_m": _json_edge_values(edge_easting),
            "northing_m": _json_edge_values(edge_northing),
        },
        output_is_geographic=True,
        oracle_from_crs=oracle_projected,
        oracle_to_crs=oracle_projected.geodetic_crs,
        qualification=QUALIFICATION_SPECS[name],
        geographic_scale_m=float(computed["a"]),
    )


def _prepare_merc(
    cp: Any,
    n: int,
    rng: np.random.Generator,
    *,
    direction: str,
    geometry: str,
    variant: str,
) -> BenchmarkCase:
    """Prepare one full-domain regular Mercator public qualification case."""
    from pyproj import CRS
    from vibeproj import Transformer

    targets = {
        ("ellipsoidal", "a"): "EPSG:3395",
        ("ellipsoidal", "b"): "EPSG:3994",
        (
            "spherical",
            "a",
        ): "+proj=merc +k=0.87 +lon_0=12 +R=6371000 +units=m +type=crs",
        (
            "spherical",
            "b",
        ): "+proj=merc +lat_ts=33 +lon_0=12 +R=6371000 +units=m +type=crs",
    }
    target = CRS.from_user_input(targets[(geometry, variant)])
    source = target.geodetic_crs
    name = f"merc-{direction}-{geometry}-variant-{variant}"
    forward = Transformer.from_crs(source, target, always_xy=True)
    computed = forward._pipeline.computed
    a = float(computed["a"])
    k0 = float(computed["k0"])
    lam0_degrees = math.degrees(float(computed["lam0"]))
    x0 = float(computed["x0"])
    y0 = float(computed["y0"])

    longitude = rng.uniform(lam0_degrees - 179.0, lam0_degrees + 179.0, n).astype(np.float64)
    latitude = rng.uniform(-89.999, 89.999, n).astype(np.float64)
    if n:
        latitude[0] = 0.0
    if n > 1:
        latitude[1] = -0.0

    clamp = 89.999
    if direction == "forward":
        transformer = forward
        input_x, input_y = longitude, latitude
        edge_x = np.asarray(
            [
                lam0_degrees,
                -0.0,
                1e300,
                math.nan,
                math.inf,
                -math.inf,
                lam0_degrees,
                lam0_degrees,
                lam0_degrees,
                lam0_degrees,
                lam0_degrees,
                lam0_degrees,
            ],
            dtype=np.float64,
        )
        edge_y = np.asarray(
            [
                0.0,
                -0.0,
                1e300,
                0.0,
                0.0,
                0.0,
                math.nextafter(clamp, -math.inf),
                clamp,
                math.nextafter(clamp, math.inf),
                math.nan,
                math.inf,
                -math.inf,
            ],
            dtype=np.float64,
        )
        output_is_geographic = False
        oracle_from, oracle_to = source, target
        domain = {
            "crs": f"{source.name} -> {target.name}",
            "longitude_degrees": [lam0_degrees - 179.0, lam0_degrees + 179.0],
            "latitude_degrees": [-89.999, 89.999],
            "includes_signed_zero_and_clamp_nextafters": True,
            "accelerated_hot_latitude_degrees": [-89.9, 89.9],
        }
    else:
        transformer = Transformer.from_crs(target, source, always_xy=True)
        normalized_x = rng.uniform(-math.pi, math.pi, n)
        normalized_psi = rng.uniform(-4.0, 4.0, n)
        extreme = np.arange(n) % 10 == 0
        normalized_psi[extreme] = rng.uniform(-750.0, 750.0, int(np.count_nonzero(extreme)))
        if n:
            normalized_psi[0] = 0.0
        if n > 1:
            normalized_psi[1] = -0.0
        input_x = normalized_x * (a * k0) + x0
        input_y = normalized_psi * (a * k0) + y0
        edge_x = np.asarray(
            [x0, -0.0, 1e300, math.nan, math.inf, -math.inf, x0, x0, x0, x0, x0],
            dtype=np.float64,
        )
        edge_y = np.asarray(
            [
                y0,
                -0.0,
                1e300,
                y0,
                y0,
                y0,
                -1e300,
                math.nextafter(y0, math.inf),
                math.nan,
                math.inf,
                -math.inf,
            ],
            dtype=np.float64,
        )
        output_is_geographic = True
        oracle_from, oracle_to = target, source
        domain = {
            "crs": f"{target.name} -> {source.name}",
            "normalized_psi_typical": [-4.0, 4.0],
            "normalized_psi_extreme": [-750.0, 750.0],
            "extreme_fraction": 0.1,
            "includes_signed_zero": True,
        }

    return BenchmarkCase(
        name=name,
        family="merc",
        direction="FORWARD",
        transformer=transformer,
        input_x=cp.asarray(input_x),
        input_y=cp.asarray(input_y),
        host_x=np.asarray(input_x, dtype=np.float64),
        host_y=np.asarray(input_y, dtype=np.float64),
        edge_host_x=edge_x,
        edge_host_y=edge_y,
        domain=domain,
        edges={"first": _json_edge_values(edge_x), "second": _json_edge_values(edge_y)},
        output_is_geographic=output_is_geographic,
        oracle_from_crs=oracle_from,
        oracle_to_crs=oracle_to,
        qualification=QUALIFICATION_SPECS[name],
        geographic_scale_m=a,
    )


_LCC_TARGETS = {
    "spherical-1sp-north": (
        "+proj=lcc +lat_0=40 +lat_1=40 +lon_0=-96 +R=6371000 +units=m +type=crs"
    ),
    "spherical-2sp-north": (
        "+proj=lcc +lat_0=23 +lat_1=33 +lat_2=45 +lon_0=-96 +R=6371000 +units=m +type=crs"
    ),
    "ellipsoidal-1sp-north": (
        "+proj=lcc +lat_0=40 +lat_1=40 +lon_0=-96 +ellps=WGS84 +units=m +type=crs"
    ),
    "ellipsoidal-1sp-near-equator": (
        "+proj=lcc +lat_0=.1 +lat_1=.1 +lon_0=0 +ellps=WGS84 +units=m +type=crs"
    ),
    "ellipsoidal-2sp-north": "EPSG:2154",
    "ellipsoidal-2sp-south": "EPSG:3851",
    "ellipsoidal-2sp-zero": (
        "+proj=lcc +lat_0=10 +lat_1=0 +lat_2=30 +lon_0=-96 +ellps=WGS84 +units=m +type=crs"
    ),
}


def _prepare_lcc(
    cp: Any,
    n: int,
    rng: np.random.Generator,
    *,
    direction: str,
    configuration: str,
    mix_per_mille: int | None = None,
) -> BenchmarkCase:
    """Prepare LCC 1SP/2SP, sphere/ellipsoid, sign, and cold-mixture cases."""
    from pyproj import CRS
    from pyproj import Transformer as PyProjTransformer

    from vibeproj import Transformer

    target = CRS.from_user_input(_LCC_TARGETS[configuration])
    source = target.geodetic_crs
    forward = Transformer.from_crs(source, target, always_xy=True)
    computed = forward._pipeline.computed
    a = float(computed["a"])
    longitude_origin = math.degrees(float(computed["lam0"]))
    southern = float(computed["n"]) < 0.0
    latitude_range = (-75.0, -5.0) if southern else (5.0, 75.0)
    longitude = rng.uniform(longitude_origin - 45.0, longitude_origin + 45.0, n).astype(np.float64)
    latitude = rng.uniform(*latitude_range, n).astype(np.float64)

    name = f"lcc-{direction}-{configuration}"
    if mix_per_mille is not None:
        name = f"lcc-forward-ellipsoidal-2sp-mix-{mix_per_mille:04d}"
        count = round(n * mix_per_mille / 1000.0)
        if count:
            latitude[rng.choice(n, count, replace=False)] = -89.999

    if direction == "forward":
        transformer = forward
        input_x, input_y = longitude, latitude
        edge_x = np.asarray(
            [longitude_origin, longitude_origin, math.nan, math.inf, -math.inf],
            dtype=np.float64,
        )
        edge_y = np.asarray([90.0, -90.0, 45.0, 45.0, 45.0], dtype=np.float64)
        output_is_geographic = False
        oracle_from, oracle_to = source, target
    else:
        transformer = Transformer.from_crs(target, source, always_xy=True)
        oracle = PyProjTransformer.from_crs(source, target, always_xy=True)
        input_x, input_y = oracle.transform(longitude, latitude)
        input_x = np.asarray(input_x, dtype=np.float64)
        input_y = np.asarray(input_y, dtype=np.float64)
        apex_x = float(computed["x0"])
        apex_y = float(computed["y0"] + a * computed["rho0"])
        edge_x = np.asarray(
            [apex_x, math.nan, math.inf, -math.inf, 1e300, -1e300], dtype=np.float64
        )
        edge_y = np.asarray([apex_y, apex_y, apex_y, apex_y, -1e300, 1e300], dtype=np.float64)
        output_is_geographic = True
        oracle_from, oracle_to = target, source

    return BenchmarkCase(
        name=name,
        family="lcc",
        direction="FORWARD",
        transformer=transformer,
        input_x=cp.asarray(input_x),
        input_y=cp.asarray(input_y),
        host_x=np.asarray(input_x, dtype=np.float64),
        host_y=np.asarray(input_y, dtype=np.float64),
        edge_host_x=edge_x,
        edge_host_y=edge_y,
        domain={
            "crs": f"{source.name} -> {target.name}"
            if direction == "forward"
            else f"{target.name} -> {source.name}",
            "geometry": "spherical" if float(computed["e"]) == 0.0 else "ellipsoidal",
            "variant": computed["lcc_variant"],
            "cone_sign": "south"
            if southern
            else ("zero-standard-parallel" if configuration.endswith("zero") else "north"),
            "longitude_degrees": [longitude_origin - 45.0, longitude_origin + 45.0],
            "source_latitude_degrees": list(latitude_range),
            "opposite_cone_extreme_fraction": (
                None if mix_per_mille is None else mix_per_mille / 1000.0
            ),
        },
        edges={"first": _json_edge_values(edge_x), "second": _json_edge_values(edge_y)},
        output_is_geographic=output_is_geographic,
        oracle_from_crs=oracle_from,
        oracle_to_crs=oracle_to,
        qualification=QUALIFICATION_SPECS[name],
        geographic_scale_m=a,
    )


def _lcc_hot_guard_audit(
    direction: str,
    computed: dict[str, Any],
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, Any]:
    """Mirror LCC's production predicates for a deterministic hot-probe audit."""
    nn = float(computed["n"])
    F = float(computed["F"])
    rho0 = float(computed["rho0"])
    e = float(computed["e"])
    k0 = float(computed["k0"])
    lam0 = float(computed["lam0"])
    a = float(computed["a"])
    x0 = float(computed["x0"])
    y0 = float(computed["y0"])
    x_unit = float(computed["x_unit_to_m"])
    y_unit = float(computed["y_unit_to_m"])
    setup_qualified = (
        all(np.isfinite(value) for value in (nn, F, rho0, e, k0, lam0, a, x0, y0))
        and nn != 0.0
        and F != 0.0
        and 0.0 <= e <= 0.1
        and k0 == 1.0
        and 0.0 < a <= PROJECTION_FIXED_Q62_MAX_SCALE_M
        and np.isfinite(x_unit)
        and x_unit != 0.0
        and np.isfinite(y_unit)
        and y_unit != 0.0
        and (direction == "inverse" or abs(nn) >= 0.2)
    )
    if direction == "forward":
        phi = np.deg2rad(second)
        lam = np.deg2rad(first) - lam0
        lam -= 2.0 * math.pi * np.rint(lam / (2.0 * math.pi))
        tan_half = np.tan(0.5 * (0.5 * math.pi - phi))
        if e == 0.0:
            rho = F * np.exp(nn * np.log(tan_half)) * k0
        else:
            correction = e * np.arctanh(e * np.sin(phi))
            rho = F * np.power(tan_half / np.exp(-correction), nn) * k0
        theta = nn * lam
        mask = (
            setup_qualified
            & np.isfinite(first)
            & np.isfinite(second)
            & np.isfinite(phi)
            & (np.abs(phi) < 0.5 * math.pi)
            & np.isfinite(lam)
            & np.isfinite(rho)
            & (np.abs(theta) <= math.pi)
            & np.isfinite(rho * np.sin(theta))
            & np.isfinite(rho0 - rho * np.cos(theta))
        )
        pole_margin = float(0.5 * math.pi - np.max(np.abs(phi)))
        theta_margin = float(math.pi - np.max(np.abs(theta)))
        minimum_ratio = None
    else:
        cx = (first * x_unit - x0) / a
        cy = (second * y_unit - y0) / a
        work_cx = cx.copy()
        dy = rho0 - cy
        rho = np.sqrt(work_cx * work_cx + dy * dy)
        if nn < 0.0:
            rho = -rho
            work_cx = -work_cx
            dy = -dy
        ratio = rho / (F * k0)
        candidate_lam = np.arctan2(work_cx, dy) / nn
        if e == 0.0:
            psi = -np.log(ratio) / nn
            candidate_phi = 0.5 * math.pi - 2.0 * np.arctan(np.exp(-psi))
        else:
            ts = np.power(ratio, 1.0 / nn)
            candidate_phi = 0.5 * math.pi - 2.0 * np.arctan(ts)
            for iteration in range(6):
                sin_phi = np.sin(candidate_phi)
                correction = e * np.arctanh(e * sin_phi)
                scaled_ts = ts * np.exp(-correction)
                dphi = 0.5 * math.pi - 2.0 * np.arctan(scaled_ts) - candidate_phi
                step = dphi
                if iteration == 5:
                    e_squared = e * e
                    contraction = (
                        e_squared
                        * (1.0 - sin_phi * sin_phi)
                        / (1.0 - e_squared * sin_phi * sin_phi)
                    )
                    step = np.where(np.abs(dphi) >= 1e-14, dphi / (1.0 - contraction), dphi)
                candidate_phi += step
        mask = (
            setup_qualified
            & np.isfinite(first)
            & np.isfinite(second)
            & np.isfinite(cx)
            & np.isfinite(cy)
            & np.isfinite(work_cx)
            & np.isfinite(dy)
            & (ratio > 0.0)
            & np.isfinite(candidate_lam)
            & np.isfinite(candidate_phi)
        )
        pole_margin = None
        theta_margin = None
        minimum_ratio = float(np.min(ratio))
    mask = np.asarray(mask, dtype=bool)
    warp_count = (mask.size + 31) // 32
    padded = np.pad(mask, (0, warp_count * 32 - mask.size), constant_values=False)
    qualified_warps = int(np.count_nonzero(np.all(padded.reshape(-1, 32), axis=1)))
    return {
        "setup_qualified": bool(setup_qualified),
        "qualified_lanes": int(np.count_nonzero(mask)),
        "total_lanes": int(mask.size),
        "qualified_warps": qualified_warps,
        "cold_warps": int(warp_count - qualified_warps),
        "pole_margin_rad": pole_margin,
        "theta_margin_rad": theta_margin,
        "minimum_radius_ratio": minimum_ratio,
    }


def _prepare_krovak_inverse(
    cp: Any,
    n: int,
    rng: np.random.Generator,
    *,
    epsg: int,
) -> BenchmarkCase:
    """Prepare one public standard-Bessel Krovak inverse qualification case."""
    from pyproj import CRS
    from pyproj import Transformer as PyProjTransformer

    from vibeproj import Transformer

    projected = CRS.from_epsg(epsg)
    geographic = projected.geodetic_crs
    prime_meridian = float(geographic.prime_meridian.longitude)
    longitude = rng.uniform(12.0 - prime_meridian, 19.0 - prime_meridian, n)
    latitude = rng.uniform(48.0, 51.3, n)
    oracle_forward = PyProjTransformer.from_crs(geographic, projected, always_xy=True)
    public_x, public_y = oracle_forward.transform(longitude, latitude)
    public_x = np.asarray(public_x, dtype=np.float64)
    public_y = np.asarray(public_y, dtype=np.float64)
    transformer = Transformer.from_crs(projected, geographic, always_xy=True)
    computed = transformer._pipeline.computed
    regular = epsg in (2065, 5513, 8352)

    # Public regular Krovak is X=Southing/Y=Westing even with always_xy=True;
    # do not reinterpret these buffers as generic easting/northing.
    edge_x = np.asarray(
        [0.0, public_x[0], math.nan, math.inf, -math.inf, public_x[-1]],
        dtype=np.float64,
    )
    edge_y = np.asarray(
        [0.0, public_y[0], public_y[0], public_y[0], public_y[0], public_y[-1]],
        dtype=np.float64,
    )
    name = f"krovak-inverse-epsg-{epsg}"
    return BenchmarkCase(
        name=name,
        family="krovak",
        direction="FORWARD",
        transformer=transformer,
        input_x=cp.asarray(public_x),
        input_y=cp.asarray(public_y),
        host_x=public_x,
        host_y=public_y,
        edge_host_x=edge_x,
        edge_host_y=edge_y,
        domain={
            "epsg": epsg,
            "crs": f"{projected.name} -> {geographic.name}",
            "geodetic_crs": geographic.name,
            "operation_method": computed["_strategy_operation_method"],
            "public_axis_order": "X=Southing/Y=Westing" if regular else "X=Easting/Y=Northing",
            "prime_meridian": geographic.prime_meridian.name,
            "greenwich_longitude_degrees": [12.0, 19.0],
            "source_latitude_degrees": [48.0, 51.3],
        },
        edges={"first": _json_edge_values(edge_x), "second": _json_edge_values(edge_y)},
        output_is_geographic=True,
        oracle_from_crs=projected,
        oracle_to_crs=geographic,
        qualification=QUALIFICATION_SPECS[name],
        geographic_scale_m=float(computed["a"]),
    )


def _prepare_case(cp: Any, name: str, n: int, seed: int) -> BenchmarkCase:
    rng = np.random.default_rng(seed)
    if name == "tmerc-forward":
        return _prepare_tmerc_forward(cp, n, rng)
    if name == "tmerc-inverse":
        return _prepare_tmerc_inverse(cp, n, rng)
    if name == "helmert-forward":
        return _prepare_helmert(cp, n, rng, "FORWARD")
    if name == "helmert-inverse":
        return _prepare_helmert(cp, n, rng, "INVERSE")
    if name == "sinu-forward":
        return _prepare_sinu_forward(cp, n, rng)
    if name.startswith("sinu-inverse"):
        return _prepare_sinu_inverse(cp, n, rng, name=name)
    if name == "ortho-forward":
        return _prepare_ortho_forward(cp, n, rng)
    if name == "ortho-inverse":
        return _prepare_ortho_inverse(cp, n, rng)
    if name.startswith("gnom-inverse-"):
        return _prepare_gnom_inverse(cp, n, rng, origin=name.rsplit("-", maxsplit=1)[1])
    if name.startswith("geos-forward-"):
        _, _, geometry, _, sweep = name.split("-")
        return _prepare_geos_forward(cp, n, rng, geometry=geometry, sweep=sweep)
    if name.startswith("laea-forward-"):
        _, _, geometry, pole = name.split("-")
        return _prepare_laea_polar_forward(cp, n, rng, geometry=geometry, pole=pole)
    if name.startswith("stere-inverse-"):
        _, _, variant, hemisphere = name.split("-")
        return _prepare_stere_inverse(cp, n, rng, variant=variant, hemisphere=hemisphere)
    if name.startswith("merc-"):
        _, direction, geometry, _, variant = name.split("-")
        return _prepare_merc(
            cp,
            n,
            rng,
            direction=direction,
            geometry=geometry,
            variant=variant,
        )
    if name.startswith("lcc-forward-ellipsoidal-2sp-mix-"):
        return _prepare_lcc(
            cp,
            n,
            rng,
            direction="forward",
            configuration="ellipsoidal-2sp-north",
            mix_per_mille=int(name.rsplit("-", maxsplit=1)[1]),
        )
    if name.startswith("lcc-"):
        _, direction, *configuration_parts = name.split("-")
        return _prepare_lcc(
            cp,
            n,
            rng,
            direction=direction,
            configuration="-".join(configuration_parts),
        )
    if name.startswith("krovak-inverse-epsg-"):
        return _prepare_krovak_inverse(
            cp,
            n,
            rng,
            epsg=int(name.rsplit("-", maxsplit=1)[1]),
        )
    raise ValueError(f"Unknown benchmark case: {name}")


def _resolved_strategy(
    transformer: Any,
    policy: str,
    direction: str,
    *,
    precision: str,
    workload_size: int | None,
) -> dict[str, Any]:
    explanation = transformer.explain_strategy(
        transcendentals=policy,
        direction=direction,
        precision=precision,
        workload_size=workload_size,
    )
    decisions = []
    for decision in explanation.decisions:
        decisions.append(
            {
                "family": decision.family,
                "operation": decision.operation,
                "implementation_id": decision.implementation_id,
                "fallback": decision.fallback,
                "reason": decision.reason,
                "workload_size": decision.workload_size,
                "accuracy": asdict(decision.accuracy),
            }
        )
    return {
        "requested_policy": explanation.requested_policy,
        "direction": explanation.direction,
        "implementation_ids": sorted({item["implementation_id"] for item in decisions}),
        "decisions": decisions,
    }


def _percentile(samples: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(samples, dtype=np.float64), q))


def _distribution(samples: list[float], n: int) -> dict[str, Any]:
    median = statistics.median(samples)
    return {
        "samples_ms": samples,
        "min_ms": min(samples),
        "p05_ms": _percentile(samples, 5),
        "median_ms": median,
        "p95_ms": _percentile(samples, 95),
        "max_ms": max(samples),
        "stdev_ms": statistics.pstdev(samples),
        "mcoords_per_s": n / median / 1000.0,
    }


def _native_identity_noise_pass(
    *,
    auto_implementation_ids: list[str],
    native_implementation_ids: list[str],
    wall_speedup: float,
    wall_repeat_speedups: list[float],
) -> bool:
    """Gate identical-native AUTO timing only on stable public wall measurements.

    At tiny crossover-adjacent workloads, CUDA event medians are quantized too
    coarsely to compare two launches of the exact same cached native kernel.
    Candidate event gates remain strict once AUTO resolves an accelerated ID.
    """
    identical_native_variant = (
        auto_implementation_ids == native_implementation_ids == [NATIVE_LIBDEVICE]
    )
    return (
        identical_native_variant
        and wall_speedup >= 0.98
        and all(speedup >= 0.95 for speedup in wall_repeat_speedups)
    )


def _time_interleaved(
    cp: Any,
    launches: dict[str, Callable[[], None]],
    *,
    n: int,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[dict[str, Any], int, int]:
    """Time launches in a rotating order with one CUDA event pair per launch."""
    names = list(launches)
    for index in range(warmup):
        order = names[index % len(names) :] + names[: index % len(names)]
        for name in order:
            launches[name]()
    cp.cuda.get_current_stream().synchronize()

    memory_pool = cp.get_default_memory_pool()
    used_before = int(memory_pool.used_bytes())
    repeat_results: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    all_samples: dict[str, list[float]] = {name: [] for name in names}

    for repeat in range(repeats):
        repeat_events: dict[str, list[tuple[Any, Any]]] = {name: [] for name in names}
        for index in range(iterations):
            offset = (index + repeat) % len(names)
            order = names[offset:] + names[:offset]
            for name in order:
                start, end = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                launches[name]()
                end.record()
                repeat_events[name].append((start, end))
        cp.cuda.get_current_stream().synchronize()
        for name in names:
            samples = [
                float(cp.cuda.get_elapsed_time(start, end)) for start, end in repeat_events[name]
            ]
            all_samples[name].extend(samples)
            repeat_results[name].append(_distribution(samples, n))

    used_after = int(memory_pool.used_bytes())
    results: dict[str, Any] = {}
    native_repeat_medians = [item["median_ms"] for item in repeat_results["native"]]
    native_all_median = statistics.median(all_samples["native"])
    for name in names:
        aggregate = _distribution(all_samples[name], n)
        aggregate["speedup_vs_native"] = native_all_median / aggregate["median_ms"]
        aggregate["repeats"] = repeat_results[name]
        aggregate["repeat_speedups_vs_native"] = [
            native_repeat_medians[index] / result["median_ms"]
            for index, result in enumerate(repeat_results[name])
        ]
        results[name] = aggregate
    return results, used_before, used_after


def _time_synchronized_wall_interleaved(
    cp: Any,
    launches: dict[str, Callable[[], None]],
    *,
    n: int,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    """Measure warmed public dispatch plus device completion wall time."""
    names = list(launches)
    stream = cp.cuda.get_current_stream()
    for index in range(warmup):
        order = names[index % len(names) :] + names[: index % len(names)]
        for name in order:
            launches[name]()
            stream.synchronize()

    repeat_results: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    all_samples: dict[str, list[float]] = {name: [] for name in names}
    for repeat in range(repeats):
        repeat_samples: dict[str, list[float]] = {name: [] for name in names}
        for index in range(iterations):
            offset = (index + repeat) % len(names)
            order = names[offset:] + names[:offset]
            for name in order:
                # Start from an idle stream. The timed interval includes all
                # public-call normalization, strategy/device resolution,
                # argument packing, submission, and completion synchronization.
                stream.synchronize()
                started_ns = time.perf_counter_ns()
                launches[name]()
                stream.synchronize()
                elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
                repeat_samples[name].append(elapsed_ms)
        for name in names:
            samples = repeat_samples[name]
            all_samples[name].extend(samples)
            repeat_results[name].append(_distribution(samples, n))

    results: dict[str, Any] = {}
    native_repeat_medians = [item["median_ms"] for item in repeat_results["native"]]
    native_all_median = statistics.median(all_samples["native"])
    for name in names:
        aggregate = _distribution(all_samples[name], n)
        aggregate["speedup_vs_native"] = native_all_median / aggregate["median_ms"]
        aggregate["repeats"] = repeat_results[name]
        aggregate["repeat_speedups_vs_native"] = [
            native_repeat_medians[index] / result["median_ms"]
            for index, result in enumerate(repeat_results[name])
        ]
        results[name] = aggregate
    return results


def _measure_allocator_calls(
    cp: Any,
    launch: Callable[[], tuple[Any, ...]],
    *,
    calls: int = 5,
) -> dict[str, Any]:
    """Count allocator API calls in separate, untimed steady-state transforms."""
    original_allocator = cp.cuda.get_allocator()
    per_call: list[list[int]] = []
    identities: list[bool] = []

    def counting_allocator(size: int) -> Any:
        per_call[-1].append(int(size))
        return original_allocator(size)

    cp.cuda.set_allocator(counting_allocator)
    try:
        for _ in range(calls):
            per_call.append([])
            result = launch()
            identities.append(bool(result[-1]))
        cp.cuda.get_current_stream().synchronize()
    finally:
        cp.cuda.set_allocator(original_allocator)

    return {
        "instrumentation": "CuPy allocator wrapper around untimed public transform calls",
        "steady_state_calls": calls,
        "allocator_calls_per_transform": [len(sizes) for sizes in per_call],
        "requested_bytes_per_transform": [sum(sizes) for sizes in per_call],
        "requested_sizes_per_transform": per_call,
        "returned_preallocated_outputs_per_transform": identities,
    }


def _capture_cuda_graph(
    cp: Any,
    launch: Callable[[Any], tuple[Any, ...]],
) -> dict[str, Any]:
    """Capture one untimed call and report every CUDA graph node."""
    stream = cp.cuda.Stream(non_blocking=True)
    try:
        stream.begin_capture()
        with stream:
            result = launch(stream)
        graph = stream.end_capture()
    except Exception as exc:
        # Ending an invalidated capture resets stream capture state when CUDA
        # permits it; preserve the original failure as qualification evidence.
        try:
            stream.end_capture()
        except Exception:
            pass
        return {
            "captured": False,
            "error": f"{type(exc).__name__}: {exc}",
            "kernel_nodes": None,
            "memcpy_nodes": None,
            "memset_nodes": None,
            "kernel_names": [],
            "memcpy_labels": [],
            "returned_preallocated_outputs": False,
        }

    dot = graph.debug_dot_str()
    node_blocks = re.findall(r'"graph_[^"]+_node_\d+"\[(.*?)\];', dot, re.DOTALL)
    kernel_names: list[str] = []
    memcpy_labels: list[str] = []
    memset_labels: list[str] = []
    for block in node_blocks:
        label_match = re.search(r'label="(.*?)"', block, re.DOTALL)
        raw_label = label_match.group(1).strip() if label_match else block.strip()
        label = raw_label.replace("\n", " | ")
        if 'shape="octagon"' in block:
            parts = [part.strip() for part in raw_label.splitlines() if part.strip()]
            kernel_names.append(parts[-1] if parts else label)
        elif "MEMCPY" in label:
            memcpy_labels.append(label)
        elif "MEMSET" in label:
            memset_labels.append(label)
    return {
        "captured": True,
        "kernel_nodes": len(kernel_names),
        "memcpy_nodes": len(memcpy_labels),
        "memset_nodes": len(memset_labels),
        "total_nodes": len(node_blocks),
        "kernel_names": kernel_names,
        "memcpy_labels": memcpy_labels,
        "memset_labels": memset_labels,
        "returned_preallocated_outputs": bool(result[-1]),
    }


def _kernel_resources(cp: Any, case: BenchmarkCase) -> dict[str, Any]:
    """Report compiled resource usage and block occupancy for exact variants."""
    from vibeproj.fused_kernels import _get_helmert_kernel, _get_kernel

    device_properties = cp.cuda.runtime.getDeviceProperties(int(cp.cuda.Device().id))
    threads_per_sm = int(device_properties["maxThreadsPerMultiProcessor"])
    result = {}
    variants = [
        ("native", NATIVE_LIBDEVICE),
        ("accelerated", case.qualification.implementation_id),
    ]
    auto_implementation_id = _candidate_policy_ids(case.qualification)["auto"]
    if auto_implementation_id not in {NATIVE_LIBDEVICE, case.qualification.implementation_id}:
        variants.append(("auto", auto_implementation_id))
    for label, implementation_id in variants:
        if case.family == "helmert":
            kernel = _get_helmert_kernel(implementation_id)
        else:
            kernel = _get_kernel(
                case.family,
                case.qualification.direction,
                "float64",
                transcendental_impl=implementation_id,
            )
        kernel.compile()
        active_blocks = int(
            cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(kernel.kernel.ptr, 256, 0)
        )
        result[label] = {
            "attributes": {key: int(value) for key, value in kernel.attributes.items()},
            "active_blocks_per_sm_at_256_threads": active_blocks,
            "thread_occupancy_fraction": min(1.0, active_blocks * 256 / threads_per_sm),
        }
    return result


def _merc_forward_polar_cap_sweep(
    cp: Any,
    case: BenchmarkCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Retain the mixed polar-cap evidence that disables ellipsoidal auto."""
    if not case.name.startswith("merc-forward-ellipsoidal-"):
        return {"required": False, "rows": []}

    threshold = case.qualification.explicit_performance_min_elements
    if threshold is None:
        raise RuntimeError("Ellipsoidal Mercator requires an explicit performance size")
    workload_sizes = tuple(sorted({threshold, args.n}))
    rows = []
    for size_index, n in enumerate(workload_sizes):
        rng = np.random.default_rng(args.seed + 70_000 + size_index)
        base_longitude = rng.uniform(-179.0, 179.0, n)
        hot_latitude = rng.uniform(-80.0, 80.0, n)
        for fallback_fraction in (0.0, 0.001, 0.01, 0.10, 0.50, 1.0):
            latitude = hot_latitude.copy()
            fallback_count = round(fallback_fraction * n)
            fallback_mask = np.zeros(n, dtype=bool)
            if fallback_count:
                selected = rng.choice(n, fallback_count, replace=False)
                fallback_mask[selected] = True
                latitude[selected] = rng.choice((-89.95, 89.95), fallback_count)
            input_x = cp.asarray(base_longitude)
            input_y = cp.asarray(latitude)
            outputs = {
                policy: (cp.empty(n, dtype=cp.float64), cp.empty(n, dtype=cp.float64))
                for policy in ("native", "accelerated")
            }

            def invoke(policy: str) -> None:
                case.transformer.transform_buffers(
                    input_x,
                    input_y,
                    direction=case.direction,
                    out_x=outputs[policy][0],
                    out_y=outputs[policy][1],
                    precision=args.precision,
                    transcendentals=policy,
                )

            launches = {policy: lambda policy=policy: invoke(policy) for policy in outputs}
            device_timing, _, _ = _time_interleaved(
                cp,
                launches,
                n=n,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            wall_timing = _time_synchronized_wall_interleaved(
                cp,
                launches,
                n=n,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            invoke("native")
            invoke("accelerated")
            cp.cuda.get_current_stream().synchronize()
            native = tuple(cp.asnumpy(value) for value in outputs["native"])
            accelerated = tuple(cp.asnumpy(value) for value in outputs["accelerated"])
            cold_lanes_bitwise_native = all(
                np.array_equal(
                    actual[fallback_mask].view(np.uint64), expected[fallback_mask].view(np.uint64)
                )
                for actual, expected in zip(accelerated, native, strict=True)
            )
            error = _coordinate_error(
                *accelerated,
                *native,
                geographic=False,
            )
            rows.append(
                {
                    "n": n,
                    "fallback_fraction": fallback_fraction,
                    "fallback_count": fallback_count,
                    "pattern": "random lanes at +/-89.95 degrees; remaining latitudes uniform [-80,80]",
                    "device_timing": device_timing,
                    "wall_timing": wall_timing,
                    "cold_lanes_bitwise_native": cold_lanes_bitwise_native,
                    "error_vs_native_m": error,
                }
            )

    negative_auto_evidence = any(
        row["fallback_fraction"] in (0.10, 0.50)
        and (
            row["device_timing"]["accelerated"]["speedup_vs_native"] < 1.0
            or row["wall_timing"]["accelerated"]["speedup_vs_native"] < 1.0
        )
        for row in rows
    )
    correctness_pass = all(
        row["cold_lanes_bitwise_native"]
        and row["error_vs_native_m"]["nonfinite_match"]
        and row["error_vs_native_m"]["max_m"] <= case.qualification.coordinate_contract_m
        for row in rows
    )
    return {
        "required": True,
        "workload_sizes": list(workload_sizes),
        "fractions": [0.0, 0.001, 0.01, 0.10, 0.50, 1.0],
        "rows": rows,
        "all_cold_lanes_bitwise_native": all(row["cold_lanes_bitwise_native"] for row in rows),
        "negative_auto_evidence_pass": negative_auto_evidence,
        "qualification_pass": correctness_pass and negative_auto_evidence,
    }


def _sinu_inverse_mixed_guard_sweep(
    cp: Any,
    case: BenchmarkCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Qualify explicit warp fallback and full-domain AUTO on identical mixtures."""
    if not case.qualification.representative_mixed_guard_sweep:
        return {"required": False, "qualification_pass": True, "rows": []}

    from pyproj import Transformer as PyProjTransformer

    n = max(args.n, 32_000)
    rng = np.random.default_rng(args.seed + 110_000)
    longitude = rng.uniform(-179.0, 179.0, n)
    latitude = rng.uniform(-80.0, 80.0, n)
    oracle_forward = PyProjTransformer.from_crs(
        case.oracle_to_crs, case.oracle_from_crs, always_xy=True
    )
    hot_projected = oracle_forward.transform(longitude, latitude)
    hot_x = np.asarray(hot_projected[0], dtype=np.float64)
    hot_y = np.asarray(hot_projected[1], dtype=np.float64)
    computed = case.transformer._pipeline_for_direction(case.direction).computed
    pole = float(computed["y0"] + computed["a"] * computed["meridional_pole"])
    cold_values = (
        (1.0, pole),
        (1.0, -pole),
        (1.0, math.nextafter(pole, math.inf)),
        (math.nan, float(computed["y0"])),
    )
    rows = []
    fractions = (0.0, 0.001, 0.01, 0.10, 0.50, 1.0)
    for fraction in fractions:
        cold_count = round(fraction * n)
        cold_mask = np.zeros(n, dtype=bool)
        if cold_count:
            selected = rng.choice(n, cold_count, replace=False)
            cold_mask[selected] = True
        input_x_host = hot_x.copy()
        input_y_host = hot_y.copy()
        for pattern_index, lane_index in enumerate(np.flatnonzero(cold_mask)):
            input_x_host[lane_index], input_y_host[lane_index] = cold_values[
                pattern_index % len(cold_values)
            ]

        cold_warp_lane_mask = np.zeros(n, dtype=bool)
        for warp_start in range(0, n, 32):
            warp_stop = min(warp_start + 32, n)
            if bool(np.any(cold_mask[warp_start:warp_stop])):
                cold_warp_lane_mask[warp_start:warp_stop] = True
        cold_warp_count = sum(
            bool(np.any(cold_mask[start : min(start + 32, n)])) for start in range(0, n, 32)
        )

        input_x = cp.asarray(input_x_host)
        input_y = cp.asarray(input_y_host)
        outputs = {
            policy: (cp.empty(n, dtype=cp.float64), cp.empty(n, dtype=cp.float64))
            for policy in POLICIES
        }

        def invoke(policy: str) -> None:
            case.transformer.transform_buffers(
                input_x,
                input_y,
                direction=case.direction,
                out_x=outputs[policy][0],
                out_y=outputs[policy][1],
                precision=args.precision,
                transcendentals=policy,
            )

        launches = {policy: lambda policy=policy: invoke(policy) for policy in POLICIES}
        device_timing, _, _ = _time_interleaved(
            cp,
            launches,
            n=n,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        wall_timing = _time_synchronized_wall_interleaved(
            cp,
            launches,
            n=n,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        for policy in POLICIES:
            invoke(policy)
        cp.cuda.get_current_stream().synchronize()
        host_outputs = {
            policy: (cp.asnumpy(out_x), cp.asnumpy(out_y))
            for policy, (out_x, out_y) in outputs.items()
        }
        native_x, native_y = host_outputs["native"]
        errors = {
            policy: _coordinate_error(
                actual_x,
                actual_y,
                native_x,
                native_y,
                geographic=True,
                geographic_scale_m=case.geographic_scale_m,
            )
            for policy, (actual_x, actual_y) in host_outputs.items()
            if policy != "native"
        }
        accelerated_x, accelerated_y = host_outputs["accelerated"]
        recurrence_fallback_bitwise = bool(
            np.array_equal(
                accelerated_x[cold_warp_lane_mask].view(np.uint64),
                native_x[cold_warp_lane_mask].view(np.uint64),
            )
            and np.array_equal(
                accelerated_y[cold_warp_lane_mask].view(np.uint64),
                native_y[cold_warp_lane_mask].view(np.uint64),
            )
        )
        rows.append(
            {
                "cold_fraction_requested": fraction,
                "cold_lane_count": int(cold_count),
                "cold_lane_fraction_achieved": cold_count / n,
                "cold_warp_count": int(cold_warp_count),
                "cold_warp_fraction_achieved": cold_warp_count / math.ceil(n / 32),
                "fallback_lane_count_achieved": int(np.count_nonzero(cold_warp_lane_mask)),
                "fallback_lane_fraction_achieved": float(np.mean(cold_warp_lane_mask)),
                "cold_pattern": (
                    "random lanes cycling north/south pole, outside pole, and NaN easting; "
                    "explicit recurrence fallback is complete-warp"
                ),
                "device_timing": device_timing,
                "wall_timing": wall_timing,
                "error_vs_native_m": errors,
                "recurrence_fallback_warp_lanes_bitwise_native": (recurrence_fallback_bitwise),
            }
        )

    def timing_pass(row: dict[str, Any], policy: str) -> bool:
        return all(
            timing[policy]["speedup_vs_native"] >= 1.05
            and all(speedup >= 1.05 for speedup in timing[policy]["repeat_speedups_vs_native"])
            for timing in (row["device_timing"], row["wall_timing"])
        )

    correctness_pass = all(
        row["recurrence_fallback_warp_lanes_bitwise_native"]
        and all(
            error["max_m"] is not None
            and error["max_m"] <= case.qualification.coordinate_contract_m
            and error["nonfinite_match"]
            for error in row["error_vs_native_m"].values()
        )
        for row in rows
    )
    hot_row = next(row for row in rows if row["cold_fraction_requested"] == 0.0)
    ten_percent_row = next(row for row in rows if row["cold_fraction_requested"] == 0.10)
    all_cold_row = next(row for row in rows if row["cold_fraction_requested"] == 1.0)
    all_cold_wall = all_cold_row["wall_timing"]["accelerated"]
    all_cold_no_regression = all_cold_wall["speedup_vs_native"] >= 0.98 and all(
        speedup >= 0.95 for speedup in all_cold_wall["repeat_speedups_vs_native"]
    )
    recurrence_ten_percent_misses_auto_gate = (
        ten_percent_row["wall_timing"]["accelerated"]["speedup_vs_native"] < 1.05
    )
    auto_full_domain_performance = all(timing_pass(row, "auto") for row in rows)
    return {
        "required": True,
        "n": n,
        "fractions": list(fractions),
        "rows": rows,
        "recurrence_hot_performance_pass": timing_pass(hot_row, "accelerated"),
        "recurrence_ten_percent_misses_auto_gate": (recurrence_ten_percent_misses_auto_gate),
        "recurrence_all_cold_no_regression_pass": all_cold_no_regression,
        "auto_full_domain_performance_pass": auto_full_domain_performance,
        "all_recurrence_fallback_warp_lanes_bitwise_native": all(
            row["recurrence_fallback_warp_lanes_bitwise_native"] for row in rows
        ),
        "qualification_pass": correctness_pass
        and timing_pass(hot_row, "accelerated")
        and recurrence_ten_percent_misses_auto_gate
        and all_cold_no_regression
        and auto_full_domain_performance,
    }


def _krovak_inverse_mixed_guard_sweep(
    cp: Any,
    case: BenchmarkCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Qualify complete-warp Krovak fallback with homogeneous cold warps."""
    from pyproj import CRS
    from pyproj import Transformer as PyProjTransformer

    n = max(args.n, 32_000)
    rng = np.random.default_rng(args.seed + 130_000)
    epsg = int(case.domain["epsg"])
    projected = CRS.from_epsg(epsg)
    geographic = projected.geodetic_crs
    prime_meridian = float(geographic.prime_meridian.longitude)
    longitude = rng.uniform(12.0 - prime_meridian, 19.0 - prime_meridian, n)
    latitude = rng.uniform(48.0, 51.3, n)
    oracle_forward = PyProjTransformer.from_crs(geographic, projected, always_xy=True)
    hot_projected = oracle_forward.transform(longitude, latitude)
    hot_x = np.asarray(hot_projected[0], dtype=np.float64)
    hot_y = np.asarray(hot_projected[1], dtype=np.float64)
    cold_groups = (
        ("zero_radius", 0.0, 0.0),
        ("nan_first", math.nan, float(hot_y[0])),
        ("positive_inf_second", float(hot_x[0]), math.inf),
        ("huge_positive", 1e308, 1e308),
        ("huge_opposed", -1e308, 1e308),
    )
    fractions = (0.0, 0.001, 0.01, 0.10, 0.50, 1.0)
    warp_count = math.ceil(n / 32)
    rows = []
    for fraction in fractions:
        cold_warp_count = round(fraction * warp_count)
        input_x_host = hot_x.copy()
        input_y_host = hot_y.copy()
        cold_warp_mask = np.zeros(n, dtype=bool)
        for warp_index in range(cold_warp_count):
            start = warp_index * 32
            stop = min(start + 32, n)
            _, cold_x, cold_y = cold_groups[warp_index % len(cold_groups)]
            input_x_host[start:stop] = cold_x
            input_y_host[start:stop] = cold_y
            cold_warp_mask[start:stop] = True

        input_x = cp.asarray(input_x_host)
        input_y = cp.asarray(input_y_host)
        outputs = {
            policy: (cp.empty(n, dtype=cp.float64), cp.empty(n, dtype=cp.float64))
            for policy in POLICIES
        }

        def invoke(policy: str) -> None:
            case.transformer.transform_buffers(
                input_x,
                input_y,
                direction=case.direction,
                out_x=outputs[policy][0],
                out_y=outputs[policy][1],
                precision=args.precision,
                transcendentals=policy,
            )

        launches = {policy: lambda policy=policy: invoke(policy) for policy in POLICIES}
        device_timing, _, _ = _time_interleaved(
            cp,
            launches,
            n=n,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        wall_timing = _time_synchronized_wall_interleaved(
            cp,
            launches,
            n=n,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        for policy in POLICIES:
            invoke(policy)
        cp.cuda.get_current_stream().synchronize()
        host_outputs = {
            policy: (cp.asnumpy(out_x), cp.asnumpy(out_y))
            for policy, (out_x, out_y) in outputs.items()
        }
        native_x, native_y = host_outputs["native"]
        accelerated_x, accelerated_y = host_outputs["accelerated"]
        auto_x, auto_y = host_outputs["auto"]
        accelerated_error = _coordinate_error(
            accelerated_x,
            accelerated_y,
            native_x,
            native_y,
            geographic=True,
            geographic_scale_m=case.geographic_scale_m,
        )
        accelerated_cold_bitwise = bool(
            np.array_equal(
                accelerated_x[cold_warp_mask].view(np.uint64),
                native_x[cold_warp_mask].view(np.uint64),
            )
            and np.array_equal(
                accelerated_y[cold_warp_mask].view(np.uint64),
                native_y[cold_warp_mask].view(np.uint64),
            )
        )
        auto_bitwise_native = bool(
            np.array_equal(auto_x.view(np.uint64), native_x.view(np.uint64))
            and np.array_equal(auto_y.view(np.uint64), native_y.view(np.uint64))
        )
        auto_strategy = _resolved_strategy(
            case.transformer,
            "auto",
            case.direction,
            precision=args.precision,
            workload_size=n,
        )
        rows.append(
            {
                "cold_fraction_requested": fraction,
                "cold_warp_count": cold_warp_count,
                "cold_warp_fraction_achieved": cold_warp_count / warp_count,
                "cold_lane_count": int(np.count_nonzero(cold_warp_mask)),
                "cold_pattern": "homogeneous complete warps cycling named edge groups",
                "homogeneous_warp_edge_groups": [group[0] for group in cold_groups],
                "device_timing": device_timing,
                "wall_timing": wall_timing,
                "error_vs_native_m": accelerated_error,
                "accelerated_cold_warp_lanes_bitwise_native": accelerated_cold_bitwise,
                "auto_strategy": auto_strategy,
                "auto_bitwise_native": auto_bitwise_native,
            }
        )

    correctness_pass = all(
        row["accelerated_cold_warp_lanes_bitwise_native"]
        and row["auto_strategy"]["implementation_ids"] == [NATIVE_LIBDEVICE]
        and row["auto_bitwise_native"]
        and row["error_vs_native_m"]["max_m"] is not None
        and row["error_vs_native_m"]["max_m"] <= case.qualification.coordinate_contract_m
        and row["error_vs_native_m"]["nonfinite_match"]
        for row in rows
    )
    return {
        "required": True,
        "n": n,
        "fractions": list(fractions),
        "warp_layout": "selected cold warps are homogeneous; no random per-lane mixing",
        "rows": rows,
        "auto_native_at_every_fraction_pass": all(
            row["auto_strategy"]["implementation_ids"] == [NATIVE_LIBDEVICE]
            and row["auto_bitwise_native"]
            for row in rows
        ),
        "qualification_pass": correctness_pass,
    }


def _guarded_inverse_fallback_sweep(
    cp: Any,
    case: BenchmarkCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Measure deliberately mixed per-lane native fallback percentages."""
    if case.name.startswith("merc-forward-ellipsoidal-"):
        return _merc_forward_polar_cap_sweep(cp, case, args)
    if case.family == "sinu" and case.qualification.direction == "inverse":
        return _sinu_inverse_mixed_guard_sweep(cp, case, args)
    if case.family == "krovak":
        return _krovak_inverse_mixed_guard_sweep(cp, case, args)
    if case.name != "ortho-inverse" and not case.name.startswith("gnom-inverse-"):
        return {"required": False, "rows": []}

    n = args.n if case.family == "gnom" else min(args.n, 1_000_000)
    rng = np.random.default_rng(args.seed + 50_000)
    azimuth = rng.uniform(-math.pi, math.pi, n)
    if case.family == "gnom":
        safe_radius_squared = rng.uniform(0.001, 0.019, n)
        fallback_radius_squared = rng.uniform(
            np.nextafter(0.02, math.inf), np.nextafter(0.04, 0.0), n
        )
        fallback_pattern = "normalized rho_squared > 0.02"
    else:
        safe_radius_squared = rng.uniform(0.05, 0.88, n)
        fallback_radius_squared = rng.uniform(
            np.nextafter(0.99, math.inf), np.nextafter(1.0, 0.0), n
        )
        fallback_pattern = "0.99 < normalized rho_squared < 1.0"
    safe_radius = np.sqrt(safe_radius_squared) * EARTH_RADIUS_M
    safe_x = safe_radius * np.cos(azimuth)
    safe_y = safe_radius * np.sin(azimuth)
    fallback_radius = np.sqrt(fallback_radius_squared) * EARTH_RADIUS_M
    fallback_x = fallback_radius * np.cos(azimuth)
    fallback_y = fallback_radius * np.sin(azimuth)
    rows = []
    for fallback_fraction in (0.0, 0.001, 0.01, 0.10, 0.50, 1.0):
        fallback_count = round(fallback_fraction * n)
        mask = np.zeros(n, dtype=bool)
        if fallback_count:
            mask[rng.choice(n, size=fallback_count, replace=False)] = True
        input_x = cp.asarray(np.where(mask, fallback_x, safe_x))
        input_y = cp.asarray(np.where(mask, fallback_y, safe_y))
        outputs = {
            policy: (cp.empty(n, dtype=cp.float64), cp.empty(n, dtype=cp.float64))
            for policy in ("native", "accelerated")
        }

        def invoke(policy: str) -> None:
            case.transformer.transform_buffers(
                input_x,
                input_y,
                direction=case.direction,
                out_x=outputs[policy][0],
                out_y=outputs[policy][1],
                precision=args.precision,
                transcendentals=policy,
            )

        timing = _time_synchronized_wall_interleaved(
            cp,
            {policy: lambda policy=policy: invoke(policy) for policy in outputs},
            n=n,
            warmup=args.warmup if case.family == "gnom" else min(args.warmup, 5),
            iterations=args.iterations if case.family == "gnom" else min(args.iterations, 20),
            repeats=args.repeats,
        )
        invoke("native")
        invoke("accelerated")
        cp.cuda.get_current_stream().synchronize()
        native = tuple(cp.asnumpy(value) for value in outputs["native"])
        accelerated = tuple(cp.asnumpy(value) for value in outputs["accelerated"])
        fallback_bitwise_native = bool(
            all(
                np.array_equal(actual[mask].view(np.uint64), expected[mask].view(np.uint64))
                for actual, expected in zip(accelerated, native, strict=True)
            )
        )
        rows.append(
            {
                "fallback_fraction": fallback_fraction,
                "fallback_count": fallback_count,
                "timing": timing,
                "fallback_lanes_bitwise_native": fallback_bitwise_native,
            }
        )
    return {
        "required": True,
        "n": n,
        "pattern": f"random lane mixture; valid fallback points use {fallback_pattern}",
        "rows": rows,
        "all_fallback_lanes_bitwise_native": all(
            row["fallback_lanes_bitwise_native"] for row in rows
        ),
        "qualification_pass": all(row["fallback_lanes_bitwise_native"] for row in rows),
    }


def _coordinate_error(
    actual_x: np.ndarray,
    actual_y: np.ndarray,
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    *,
    geographic: bool,
    geographic_scale_m: float = EARTH_RADIUS_M,
) -> dict[str, Any]:
    nonfinite_match = bool(
        np.array_equal(np.isnan(actual_x), np.isnan(reference_x))
        and np.array_equal(np.isnan(actual_y), np.isnan(reference_y))
        and np.array_equal(np.isposinf(actual_x), np.isposinf(reference_x))
        and np.array_equal(np.isposinf(actual_y), np.isposinf(reference_y))
        and np.array_equal(np.isneginf(actual_x), np.isneginf(reference_x))
        and np.array_equal(np.isneginf(actual_y), np.isneginf(reference_y))
    )
    finite = (
        np.isfinite(actual_x)
        & np.isfinite(actual_y)
        & np.isfinite(reference_x)
        & np.isfinite(reference_y)
    )
    actual_x_finite = actual_x[finite]
    actual_y_finite = actual_y[finite]
    reference_x_finite = reference_x[finite]
    reference_y_finite = reference_y[finite]
    if geographic:
        dlat_m = np.deg2rad(actual_y_finite - reference_y_finite) * geographic_scale_m
        delta_longitude = (actual_x_finite - reference_x_finite + 180.0) % 360.0 - 180.0
        dlon_m = (
            np.deg2rad(delta_longitude)
            * geographic_scale_m
            * np.cos(np.deg2rad(reference_y_finite))
        )
        values = np.hypot(dlat_m, dlon_m)
    else:
        values = np.hypot(
            actual_x_finite - reference_x_finite,
            actual_y_finite - reference_y_finite,
        )
    if values.size == 0:
        return {
            "finite_coordinates": 0,
            "max_m": None,
            "p99_m": None,
            "rms_m": None,
            "nonfinite_match": nonfinite_match,
        }
    return {
        "finite_coordinates": int(values.size),
        "max_m": float(np.max(values)),
        "p99_m": float(np.percentile(values, 99)),
        "rms_m": float(np.sqrt(np.mean(values * values))),
        "nonfinite_match": nonfinite_match,
    }


def _pyproj_reference(case: BenchmarkCase, limit: int) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer as PyProjTransformer
    from pyproj.enums import TransformDirection

    oracle = PyProjTransformer.from_crs(
        case.oracle_from_crs,
        case.oracle_to_crs,
        always_xy=True,
    )
    direction = (
        TransformDirection.FORWARD if case.direction == "FORWARD" else TransformDirection.INVERSE
    )
    out_x, out_y = oracle.transform(case.host_x[:limit], case.host_y[:limit], direction=direction)
    return np.asarray(out_x, dtype=np.float64), np.asarray(out_y, dtype=np.float64)


def _edge_accuracy(cp: Any, case: BenchmarkCase, precision: str) -> dict[str, Any]:
    """Run boundary/out-of-domain probes separately from the timed distribution."""
    # Mercator and LCC guards are warp-atomic. Launch each edge as one homogeneous
    # warp so a NaN/Inf probe cannot make signed-zero, clamp, or huge-finite
    # evidence vacuously execute the native path.
    groups = (
        tuple(
            (
                cp.full(32, first, dtype=cp.float64),
                cp.full(32, second, dtype=cp.float64),
            )
            for first, second in zip(case.edge_host_x, case.edge_host_y, strict=True)
        )
        if case.family in {"merc", "lcc", "sinu"}
        else ((cp.asarray(case.edge_host_x), cp.asarray(case.edge_host_y)),)
    )
    host_groups: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        policy: [] for policy in POLICIES
    }
    for policy in host_groups:
        for edge_x, edge_y in groups:
            out_x = cp.empty(edge_x.size, dtype=cp.float64)
            out_y = cp.empty(edge_y.size, dtype=cp.float64)
            case.transformer.transform_buffers(
                edge_x,
                edge_y,
                direction=case.direction,
                out_x=out_x,
                out_y=out_y,
                precision=precision,
                transcendentals=policy,
            )
            host_groups[policy].append((cp.asnumpy(out_x), cp.asnumpy(out_y)))
    outputs = {
        policy: (
            np.concatenate([group[0] for group in results]),
            np.concatenate([group[1] for group in results]),
        )
        for policy, results in host_groups.items()
    }
    native_x, native_y = outputs["native"]
    by_policy = {
        policy: _coordinate_error(
            actual_x,
            actual_y,
            native_x,
            native_y,
            geographic=case.output_is_geographic,
            geographic_scale_m=case.geographic_scale_m,
        )
        for policy, (actual_x, actual_y) in outputs.items()
        if policy != "native"
    }
    # Retain the original accelerated summary fields for artifact consumers,
    # while making disjoint explicit/AUTO implementations independently visible.
    return {**by_policy["accelerated"], "by_policy": by_policy}


def _scale_guard_behavior(cp: Any, case: BenchmarkCase, precision: str) -> dict[str, Any]:
    """Verify uniform native execution above a projection's scale contract."""
    maximum_scale = case.qualification.max_physical_scale_m
    if maximum_scale is None:
        return {"required": False, "qualification_pass": True, "probes": []}

    from vibeproj import Transformer

    probes = []
    raw_scales = (
        (maximum_scale, np.nextafter(maximum_scale, math.inf), 1e12)
        if case.family in {"merc", "stere", "lcc"}
        or (case.family == "sinu" and case.qualification.direction == "inverse")
        else (np.nextafter(maximum_scale, math.inf), 1e12)
    )
    for raw_scale in raw_scales:
        physical_scale = float(raw_scale)
        definition_scale = (
            maximum_scale
            if case.family == "sinu" and physical_scale > maximum_scale
            else physical_scale
        )
        source_crs = f"+proj=longlat +R={definition_scale:.17g} +type=crs"
        projection_parameters = f"+proj={case.family} +lon_0=0"
        if case.family in {"ortho", "gnom"}:
            equatorial_inverse = case.name == "ortho-inverse" or case.name.endswith("-equatorial")
            projection_parameters += " +lat_0=0" if equatorial_inverse else " +lat_0=45"
        elif case.family == "laea":
            latitude_origin = 90 if case.name.endswith("-north") else -90
            projection_parameters += f" +lat_0={latitude_origin}"
        elif case.family == "geos":
            sweep = case.name.rsplit("-", maxsplit=1)[1]
            projection_parameters += f" +h=35785831 +sweep={sweep}"
        elif case.family == "stere":
            latitude_origin = -90 if case.name.endswith("-south") else 90
            projection_parameters += f" +lat_0={latitude_origin} +k_0=0.994"
            source_crs = f"+proj=longlat +a={definition_scale:.17g} +rf=298.257223563 +type=crs"
        elif case.family == "merc":
            if case.name.endswith("variant-b"):
                projection_parameters += " +lat_ts=33"
            else:
                projection_parameters += " +k=0.87"
            if "ellipsoidal" in case.name:
                source_crs = f"+proj=longlat +a={definition_scale:.17g} +rf=298.257223563 +type=crs"
        elif case.family == "lcc":
            if "near-equator" in case.name:
                projection_parameters += " +lat_0=.1 +lat_1=.1"
            elif "1sp" in case.name:
                projection_parameters += " +lat_0=40 +lat_1=40"
            else:
                projection_parameters += " +lat_0=23 +lat_1=33 +lat_2=45"
            if "ellipsoidal" in case.name:
                source_crs = f"+proj=longlat +a={definition_scale:.17g} +rf=298.257223563 +type=crs"
        elif case.family == "sinu" and case.qualification.direction == "inverse":
            source_crs = f"+proj=longlat +a={definition_scale:.17g} +rf=298.257223563 +type=crs"
        earth = (
            f"+a={definition_scale:.17g} +rf=298.257223563"
            if (case.family in {"merc", "stere", "lcc"} and "ellipsoidal" in case.name)
            or (case.family == "sinu" and case.qualification.direction == "inverse")
            else f"+R={definition_scale:.17g}"
        )
        target_crs = f"{projection_parameters} {earth} +units=m +type=crs"
        if (
            case.name == "ortho-inverse"
            or case.family in {"gnom", "stere"}
            or (case.family == "merc" and case.qualification.direction == "inverse")
            or (case.family == "lcc" and case.qualification.direction == "inverse")
            or (case.family == "sinu" and case.qualification.direction == "inverse")
        ):
            transformer = Transformer.from_crs(target_crs, source_crs, always_xy=True)
            if (
                case.family == "sinu"
                and case.qualification.direction == "inverse"
                and physical_scale > maximum_scale
            ):
                inverse_pipeline = transformer._pipeline_for_direction("FORWARD")
                inverse_pipeline.computed = {
                    **inverse_pipeline.computed,
                    "a": physical_scale,
                }
            first_host = (
                np.asarray([0.1, -0.3, 0.7, 0.0, np.inf, np.nan], dtype=np.float64) * physical_scale
            )
            second_host = (
                np.asarray([0.2, 0.4, -0.1, 0.0, 0.0, 0.0], dtype=np.float64) * physical_scale
            )
        else:
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
            if case.family == "sinu" and physical_scale > maximum_scale:
                forward_pipeline = transformer._pipeline_for_direction("FORWARD")
                forward_pipeline.computed = {
                    **forward_pipeline.computed,
                    "a": physical_scale,
                }
            first_host = np.asarray([-180.0, -120.0, 0.0, 120.0, 180.0, np.inf, np.nan])
            second_host = np.asarray([-90.0, -45.0, 0.0, 45.0, 90.0, 40.0, 40.0])
        first = cp.asarray(first_host)
        second = cp.asarray(second_host)
        candidate_policies = _distinct_candidate_policies(case.qualification)
        evaluated_policies = ("native", *candidate_policies)
        outputs = {}
        for policy in evaluated_policies:
            out_x = cp.empty(first.size, dtype=cp.float64)
            out_y = cp.empty(second.size, dtype=cp.float64)
            transformer.transform_buffers(
                first,
                second,
                out_x=out_x,
                out_y=out_y,
                precision=precision,
                transcendentals=policy,
            )
            outputs[policy] = (cp.asnumpy(out_x), cp.asnumpy(out_y))
        native_expected = physical_scale > maximum_scale
        native_x, native_y = outputs["native"]
        policy_probes = {}
        for policy in candidate_policies:
            actual_x, actual_y = outputs[policy]
            bitwise_native = bool(
                np.array_equal(actual_x.view(np.uint64), native_x.view(np.uint64))
                and np.array_equal(actual_y.view(np.uint64), native_y.view(np.uint64))
            )
            error = _coordinate_error(
                actual_x,
                actual_y,
                native_x,
                native_y,
                geographic=case.output_is_geographic,
                geographic_scale_m=physical_scale,
            )
            strategy = _resolved_strategy(
                transformer,
                policy,
                case.direction,
                precision=precision,
                workload_size=int(first.size),
            )
            expected_scale_implementation = (
                NATIVE_LIBDEVICE
                if (
                    case.family == "lcc"
                    or (case.family == "sinu" and case.qualification.direction == "inverse")
                )
                and native_expected
                else _candidate_policy_ids(case.qualification)[policy]
            )
            strategy_remains_selected = strategy["implementation_ids"] == [
                expected_scale_implementation
            ]
            behavior_pass = (
                bitwise_native
                if native_expected
                else (
                    error["max_m"] is not None
                    and error["max_m"] <= case.qualification.coordinate_contract_m
                    and error["nonfinite_match"]
                )
            )
            policy_probes[policy] = {
                "strategy": strategy,
                "strategy_remains_selected": strategy_remains_selected,
                "bitwise_native_outputs": bitwise_native,
                "error_vs_native_m": error,
                "behavior_pass": behavior_pass,
            }
        accelerated_probe = policy_probes["accelerated"]
        probes.append(
            {
                "physical_scale_m": float(physical_scale),
                "strategy": accelerated_probe["strategy"],
                "strategy_remains_selected": accelerated_probe["strategy_remains_selected"],
                "bitwise_native_outputs": accelerated_probe["bitwise_native_outputs"],
                "native_expected": native_expected,
                "error_vs_native_m": accelerated_probe["error_vs_native_m"],
                "behavior_pass": accelerated_probe["behavior_pass"],
                "policies": policy_probes,
            }
        )
    return {
        "required": True,
        "maximum_qualified_scale_m": maximum_scale,
        "probes": probes,
        "qualification_pass": all(
            policy_probe["strategy_remains_selected"] and policy_probe["behavior_pass"]
            for probe in probes
            for policy_probe in probe["policies"].values()
        ),
    }


def _stere_inverse_e_guard_behavior(
    cp: Any,
    case: BenchmarkCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Prove native fallback and wall no-regression outside the eccentricity band."""
    if case.name != "stere-inverse-variant_a-north":
        return {"required": False, "qualification_pass": True, "probes": []}

    from pyproj import CRS

    from vibeproj import Transformer

    n = min(args.n, 1_000_000)
    rng = np.random.default_rng(args.seed + 90_000)
    probes = []
    for eccentricity in (np.nextafter(0.05, 0.0), np.nextafter(0.2, math.inf)):
        target = CRS.from_user_input(
            "+proj=stere +lat_0=90 +k_0=0.994 +lon_0=0 "
            f"+a=6378137 +es={eccentricity * eccentricity:.17g} +units=m +type=crs"
        )
        transformer = Transformer.from_crs(target, target.geodetic_crs, always_xy=True)
        radius = np.sqrt(rng.random(n)) * 0.5 * 6_378_137.0
        angle = rng.uniform(-math.pi, math.pi, n)
        input_x = cp.asarray(radius * np.cos(angle))
        input_y = cp.asarray(radius * np.sin(angle))
        outputs = {
            policy: (cp.empty(n, dtype=cp.float64), cp.empty(n, dtype=cp.float64))
            for policy in ("native", "accelerated")
        }

        def invoke(policy: str) -> None:
            transformer.transform_buffers(
                input_x,
                input_y,
                out_x=outputs[policy][0],
                out_y=outputs[policy][1],
                precision=args.precision,
                transcendentals=policy,
            )

        timing = _time_synchronized_wall_interleaved(
            cp,
            {policy: lambda policy=policy: invoke(policy) for policy in outputs},
            n=n,
            warmup=min(args.warmup, 5),
            iterations=min(args.iterations, 20),
            repeats=args.repeats,
        )
        for policy in outputs:
            invoke(policy)
        cp.cuda.get_current_stream().synchronize()
        native_x, native_y = (cp.asnumpy(value) for value in outputs["native"])
        accelerated_x, accelerated_y = (cp.asnumpy(value) for value in outputs["accelerated"])
        bitwise_native = bool(
            np.array_equal(accelerated_x.view(np.uint64), native_x.view(np.uint64))
            and np.array_equal(accelerated_y.view(np.uint64), native_y.view(np.uint64))
        )
        strategy = _resolved_strategy(
            transformer,
            "accelerated",
            "FORWARD",
            precision=args.precision,
            workload_size=n,
        )
        repeat_speedups = timing["accelerated"]["repeat_speedups_vs_native"]
        probes.append(
            {
                "eccentricity": float(eccentricity),
                "strategy": strategy,
                "bitwise_native_outputs": bitwise_native,
                "wall_timing": timing,
                "no_wall_regression_pass": timing["accelerated"]["speedup_vs_native"] >= 0.98
                and all(speedup >= 0.95 for speedup in repeat_speedups),
            }
        )
    return {
        "required": True,
        "qualified_eccentricity_interval": "[0.05, 0.2]",
        "probes": probes,
        "qualification_pass": all(
            probe["bitwise_native_outputs"] and probe["no_wall_regression_pass"] for probe in probes
        ),
    }


def _merc_parameter_guard_behavior(
    cp: Any,
    case: BenchmarkCase,
    precision: str,
) -> dict[str, Any]:
    """Exercise Mercator setup and derived-conditioning fallbacks publicly."""
    if case.family != "merc":
        return {"required": False, "qualification_pass": True, "probes": []}

    pipeline = case.transformer._pipeline_for_direction(case.direction)
    original = pipeline.computed
    n = 256
    first = cp.resize(case.input_x, n)
    second = cp.resize(case.input_y, n)
    probes = []

    def execute(input_first, input_second) -> dict[str, Any]:
        outputs = {}
        for policy in ("native", "accelerated"):
            out_x = cp.empty(input_first.size, dtype=cp.float64)
            out_y = cp.empty(input_second.size, dtype=cp.float64)
            case.transformer.transform_buffers(
                input_first,
                input_second,
                direction=case.direction,
                out_x=out_x,
                out_y=out_y,
                precision=precision,
                transcendentals=policy,
            )
            outputs[policy] = (cp.asnumpy(out_x), cp.asnumpy(out_y))
        bitwise_native = all(
            np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
            for actual, expected in zip(outputs["accelerated"], outputs["native"], strict=True)
        )
        error = _coordinate_error(
            *outputs["accelerated"],
            *outputs["native"],
            geographic=case.output_is_geographic,
            geographic_scale_m=case.geographic_scale_m,
        )
        return {"bitwise_native_outputs": bitwise_native, "error_vs_native_m": error}

    if case.qualification.direction == "forward":
        # Deterministic EPSG:3395 witness: the product-polynomial result is one
        # binary64 ULP from the native pow expression at this polar latitude.
        hot_first = cp.full(32, 23.5, dtype=cp.float64)
        hot_second = cp.full(32, -89.35820071200712, dtype=cp.float64)
    else:
        # Deterministic inverse witness retained as a homogeneous warp so the
        # warp-atomic guard cannot make branch evidence input-order dependent.
        hot_first = cp.full(32, 11_478_839.31124818, dtype=cp.float64)
        hot_second = cp.full(32, 4_691_815.763041037, dtype=cp.float64)
    hot_probe = execute(hot_first, hot_second)
    hot_probe["nonvacuous_pass"] = (
        not hot_probe["bitwise_native_outputs"] if "ellipsoidal" in case.name else True
    )

    coordinate_fallback_probes = []
    hot_first = float(case.host_x[min(2, case.host_x.size - 1)])
    hot_second = float(case.host_y[min(2, case.host_y.size - 1)])
    for component in (0, 1):
        for label, value in (
            ("nan", math.nan),
            ("positive_inf", math.inf),
            ("negative_inf", -math.inf),
        ):
            input_first = cp.full(32, hot_first, dtype=cp.float64)
            input_second = cp.full(32, hot_second, dtype=cp.float64)
            target = input_first if component == 0 else input_second
            target.fill(value)
            homogeneous = execute(input_first, input_second)
            coordinate_fallback_probes.append(
                {
                    "name": f"component_{component}_{label}_homogeneous",
                    **homogeneous,
                }
            )

            input_first.fill(hot_first)
            input_second.fill(hot_second)
            target = input_first if component == 0 else input_second
            target[0] = value
            mixed = execute(input_first, input_second)
            coordinate_fallback_probes.append(
                {"name": f"component_{component}_{label}_mixed", **mixed}
            )
    mutations: list[tuple[str, dict[str, float]]] = [
        ("eccentricity_nextabove", {"e": math.nextafter(0.1, math.inf)}),
        (
            "scale_nextabove",
            {"a": math.nextafter(PROJECTION_FIXED_Q62_MAX_SCALE_M, math.inf)},
        ),
        ("nonfinite_central_meridian", {"lam0": math.inf}),
        ("nonfinite_x_offset", {"x0": math.inf}),
        ("nonfinite_y_offset", {"y0": math.nan}),
        ("nonfinite_x_unit", {"x_unit_to_m": math.inf}),
        ("zero_y_unit", {"y_unit_to_m": 0.0}),
    ]
    if case.qualification.direction == "forward":
        mutations.extend(
            [
                ("scale_factor_nextabove", {"k0": math.nextafter(1.0, math.inf)}),
            ]
        )
    else:
        mutations.extend(
            [
                ("tiny_scale_factor_overflow", {"k0": math.ulp(0.0)}),
                ("nonfinite_series_coefficient", {"conformal_to_geodetic": (math.nan,) * 6}),
            ]
        )

    try:
        for label, mutation in mutations:
            pipeline.computed = {**original, **mutation}
            result = execute(first, second)
            probes.append(
                {
                    "name": label,
                    "mutated_fields": sorted(mutation),
                    **result,
                }
            )
    finally:
        pipeline.computed = original

    if case.qualification.direction == "forward":
        # Raw longitude and lam0 are both finite, but their subtraction after
        # degree-to-radian conversion overflows. This pins the derived-value
        # guard independently of the setup guard.
        pipeline.computed = {**original, "lam0": np.finfo(np.float64).max}
        overflow_first = cp.full(32, -np.finfo(np.float64).max, dtype=cp.float64)
        overflow_second = cp.full(32, 45.0, dtype=cp.float64)
        try:
            result = execute(overflow_first, overflow_second)
            probes.append(
                {
                    "name": "derived_longitude_overflow",
                    "mutated_fields": ["lam0"],
                    **result,
                }
            )
        finally:
            pipeline.computed = original

    return {
        "required": True,
        "hot_probe": hot_probe,
        "coordinate_fallback_probes": coordinate_fallback_probes,
        "probes": probes,
        "qualification_pass": (
            hot_probe["nonvacuous_pass"]
            and hot_probe["error_vs_native_m"]["max_m"] <= case.qualification.coordinate_contract_m
            and all(probe["bitwise_native_outputs"] for probe in coordinate_fallback_probes)
            and all(probe["bitwise_native_outputs"] for probe in probes)
        ),
    }


def _lcc_parameter_guard_behavior(
    cp: Any,
    case: BenchmarkCase,
    precision: str,
) -> dict[str, Any]:
    """Exercise LCC setup, coordinate, warp-atomic, and cone-boundary guards."""
    if case.family != "lcc":
        return {"required": False, "qualification_pass": True, "probes": []}

    pipeline = case.transformer._pipeline_for_direction(case.direction)
    original = pipeline.computed
    n = 256
    first = cp.resize(case.input_x, n)
    second = cp.resize(case.input_y, n)

    def execute(input_first, input_second) -> dict[str, Any]:
        outputs = {}
        for policy in ("native", "accelerated"):
            out_x = cp.empty(input_first.size, dtype=cp.float64)
            out_y = cp.empty(input_second.size, dtype=cp.float64)
            case.transformer.transform_buffers(
                input_first,
                input_second,
                direction=case.direction,
                out_x=out_x,
                out_y=out_y,
                precision=precision,
                transcendentals=policy,
            )
            outputs[policy] = (cp.asnumpy(out_x), cp.asnumpy(out_y))
        bitwise_native = all(
            np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
            for actual, expected in zip(outputs["accelerated"], outputs["native"], strict=True)
        )
        error = _coordinate_error(
            *outputs["accelerated"],
            *outputs["native"],
            geographic=case.output_is_geographic,
            geographic_scale_m=case.geographic_scale_m,
        )
        return {"bitwise_native_outputs": bitwise_native, "error_vs_native_m": error}

    hot_n = 4_096
    longitude_origin = math.degrees(float(original["lam0"]))
    if case.qualification.direction == "forward" and float(original["n"]) < 0.0:
        # Deterministic EPSG:3851 witness retained independently of workload N
        # and seed; the conformal candidate differs from native by one ULP.
        hot_host_first = np.full(hot_n, 169.3024197984506, dtype=np.float64)
        hot_host_second = np.full(hot_n, -9.308100979853066, dtype=np.float64)
    elif case.qualification.direction == "forward":
        hot_host_first = np.linspace(longitude_origin - 45.0, longitude_origin + 45.0, hot_n)
        hot_host_second = np.linspace(5.0, 75.0, hot_n)
    else:
        from pyproj import Transformer as PyProjTransformer

        latitude_range = (-75.0, -5.0) if float(original["n"]) < 0.0 else (5.0, 75.0)
        hot_longitude = np.linspace(longitude_origin - 45.0, longitude_origin + 45.0, hot_n)
        hot_latitude = np.linspace(*latitude_range, hot_n)
        oracle_forward = PyProjTransformer.from_crs(
            case.oracle_to_crs, case.oracle_from_crs, always_xy=True
        )
        hot_projected = oracle_forward.transform(hot_longitude, hot_latitude)
        hot_host_first = np.asarray(hot_projected[0], dtype=np.float64)
        hot_host_second = np.asarray(hot_projected[1], dtype=np.float64)
    hot_guard_audit = _lcc_hot_guard_audit(
        case.qualification.direction, original, hot_host_first, hot_host_second
    )
    hot_first, hot_second = cp.asarray(hot_host_first), cp.asarray(hot_host_second)
    hot_probe = execute(hot_first, hot_second)
    hot_strategy = _resolved_strategy(
        case.transformer,
        "accelerated",
        case.direction,
        precision=precision,
        workload_size=hot_n,
    )
    hot_strategy_selected = hot_strategy["implementation_ids"] == [
        case.qualification.implementation_id
    ]
    if case.qualification.implementation_id == NATIVE_LIBDEVICE:
        hot_nonvacuous = hot_strategy_selected and hot_probe["bitwise_native_outputs"]
    else:
        hot_nonvacuous = (
            hot_strategy_selected
            and hot_guard_audit["qualified_lanes"] == hot_guard_audit["total_lanes"]
            and hot_guard_audit["cold_warps"] == 0
            and not hot_probe["bitwise_native_outputs"]
        )
    hot_probe.update(
        {
            "strategy": hot_strategy,
            "strategy_selected": hot_strategy_selected,
            "guard_audit": hot_guard_audit,
            "nonvacuous_pass": hot_nonvacuous,
        }
    )

    setup_mutations: list[tuple[str, dict[str, float]]] = [
        ("zero_cone_constant", {"n": 0.0}),
        ("nonfinite_cone_constant", {"n": math.nan}),
        ("zero_F", {"F": 0.0}),
        ("nonfinite_F", {"F": math.nan}),
        ("nonfinite_rho0", {"rho0": math.inf}),
        ("eccentricity_below_zero", {"e": math.nextafter(0.0, -math.inf)}),
        ("eccentricity_nextabove", {"e": math.nextafter(0.1, math.inf)}),
        ("scale_factor_nextbelow", {"k0": math.nextafter(1.0, 0.0)}),
        ("scale_factor_nextabove", {"k0": math.nextafter(1.0, math.inf)}),
        ("nonfinite_central_meridian", {"lam0": math.inf}),
        ("zero_scale", {"a": 0.0}),
        ("scale_nextabove", {"a": math.nextafter(PROJECTION_FIXED_Q62_MAX_SCALE_M, math.inf)}),
        ("nonfinite_x_offset", {"x0": math.inf}),
        ("nonfinite_y_offset", {"y0": math.nan}),
        ("zero_x_unit", {"x_unit_to_m": 0.0}),
        ("nonfinite_y_unit", {"y_unit_to_m": math.inf}),
    ]
    if case.qualification.direction == "forward":
        setup_mutations.extend(
            [
                ("positive_cone_nextinside", {"n": math.nextafter(0.2, 0.0)}),
                ("negative_cone_nextinside", {"n": math.nextafter(-0.2, 0.0)}),
            ]
        )
    else:
        setup_mutations.append(("negative_radius_ratio", {"F": -float(original["F"])}))
    setup_probes = []
    try:
        for label, mutation in setup_mutations:
            pipeline.computed = {**original, **mutation}
            setup_probes.append({"name": label, **execute(first, second)})
    finally:
        pipeline.computed = original

    qualified_boundary_probes = []
    for label, mutation in (
        ("eccentricity_exact_upper", {"e": 0.1}),
        ("scale_exact_upper", {"a": PROJECTION_FIXED_Q62_MAX_SCALE_M}),
    ):
        pipeline.computed = {**original, **mutation}
        try:
            result = execute(hot_first, hot_second)
            strategy = _resolved_strategy(
                case.transformer,
                "accelerated",
                case.direction,
                precision=precision,
                workload_size=n,
            )
        finally:
            pipeline.computed = original
        expected_bitwise = case.qualification.implementation_id == NATIVE_LIBDEVICE
        strategy_selected = strategy["implementation_ids"] == [case.qualification.implementation_id]
        qualified_boundary_probes.append(
            {
                "name": label,
                **result,
                "strategy": strategy,
                "strategy_selected": strategy_selected,
                "nonvacuous_pass": strategy_selected
                and (not expected_bitwise or result["bitwise_native_outputs"]),
            }
        )

    coordinate_probes = []
    ordinary_first = float(case.host_x[min(2, case.host_x.size - 1)])
    ordinary_second = float(case.host_y[min(2, case.host_y.size - 1)])
    coordinate_edges = list(zip(case.edge_host_x, case.edge_host_y, strict=True))
    if case.qualification.direction == "inverse":
        # Huge finite raw coordinates are qualified and intentionally converge
        # to the finite pole limit even when normalized rho overflows to +inf.
        # The exact apex is pinned by the direct ABI-level GPU test; public CRS
        # reconstruction can round its raw y representation by one ULP.
        coordinate_edges = coordinate_edges[1:4]
    for index, (edge_first, edge_second) in enumerate(coordinate_edges):
        homogeneous_first = cp.full(32, edge_first, dtype=cp.float64)
        homogeneous_second = cp.full(32, edge_second, dtype=cp.float64)
        coordinate_probes.append(
            {"name": f"edge_{index}_homogeneous", **execute(homogeneous_first, homogeneous_second)}
        )
        mixed_first = cp.full(32, ordinary_first, dtype=cp.float64)
        mixed_second = cp.full(32, ordinary_second, dtype=cp.float64)
        mixed_first[0], mixed_second[0] = edge_first, edge_second
        coordinate_probes.append(
            {"name": f"edge_{index}_mixed", **execute(mixed_first, mixed_second)}
        )

    return {
        "required": True,
        "hot_probe": hot_probe,
        "coordinate_fallback_probes": coordinate_probes,
        "qualified_boundary_probes": qualified_boundary_probes,
        "probes": setup_probes,
        "qualification_pass": (
            hot_probe["nonvacuous_pass"]
            and hot_probe["error_vs_native_m"]["max_m"] <= case.qualification.coordinate_contract_m
            and all(probe["bitwise_native_outputs"] for probe in coordinate_probes)
            and all(probe["bitwise_native_outputs"] for probe in setup_probes)
            and all(
                probe["nonvacuous_pass"]
                and probe["error_vs_native_m"]["max_m"] <= case.qualification.coordinate_contract_m
                for probe in qualified_boundary_probes
            )
        ),
    }


def _sinu_inverse_parameter_guard_behavior(
    cp: Any,
    case: BenchmarkCase,
    precision: str,
) -> dict[str, Any]:
    """Exercise the exact host setup domain for both Sinu inverse candidates."""
    if case.family != "sinu" or case.qualification.direction != "inverse":
        return {"required": False, "qualification_pass": True, "probes": []}

    from pyproj import CRS
    from pyproj import Transformer as PyProjTransformer

    from vibeproj import Transformer

    pipeline = case.transformer._pipeline_for_direction(case.direction)
    original = pipeline.computed
    candidate_policies = _distinct_candidate_policies(case.qualification)
    policy_ids = _candidate_policy_ids(case.qualification)
    longitude = np.linspace(-179.0, 179.0, 4_096, dtype=np.float64)
    latitude = np.linspace(-89.8, 89.8, 4_096, dtype=np.float64)
    oracle_forward = PyProjTransformer.from_crs(
        case.oracle_to_crs, case.oracle_from_crs, always_xy=True
    )
    projected = oracle_forward.transform(longitude, latitude)
    hot_first = cp.asarray(np.asarray(projected[0], dtype=np.float64))
    hot_second = cp.asarray(np.asarray(projected[1], dtype=np.float64))

    def execute(
        transformer: Any = case.transformer,
        input_first: Any = hot_first,
        input_second: Any = hot_second,
    ) -> dict[str, Any]:
        active_pipeline = transformer._pipeline_for_direction(case.direction)
        active_scale = float(active_pipeline.computed["a"])
        error_scale = (
            active_scale if math.isfinite(active_scale) and active_scale > 0.0 else EARTH_RADIUS_M
        )
        outputs = {}
        strategies = {}
        for policy in ("native", *candidate_policies):
            out_x = cp.empty(input_first.size, dtype=cp.float64)
            out_y = cp.empty(input_second.size, dtype=cp.float64)
            transformer.transform_buffers(
                input_first,
                input_second,
                direction=case.direction,
                out_x=out_x,
                out_y=out_y,
                precision=precision,
                transcendentals=policy,
            )
            outputs[policy] = (cp.asnumpy(out_x), cp.asnumpy(out_y))
            strategies[policy] = _resolved_strategy(
                transformer,
                policy,
                case.direction,
                precision=precision,
                workload_size=int(input_first.size),
            )
        native_x, native_y = outputs["native"]
        policy_results = {}
        for policy in candidate_policies:
            actual_x, actual_y = outputs[policy]
            policy_results[policy] = {
                "strategy": strategies[policy],
                "bitwise_native_outputs": bool(
                    np.array_equal(actual_x.view(np.uint64), native_x.view(np.uint64))
                    and np.array_equal(actual_y.view(np.uint64), native_y.view(np.uint64))
                ),
                "error_vs_native_m": _coordinate_error(
                    actual_x,
                    actual_y,
                    native_x,
                    native_y,
                    geographic=True,
                    geographic_scale_m=error_scale,
                ),
            }
        return {"policies": policy_results}

    hot_probe = execute()
    for policy, result in hot_probe["policies"].items():
        result["strategy_selected"] = result["strategy"]["implementation_ids"] == [
            policy_ids[policy]
        ]
        result["nonvacuous_pass"] = (
            result["strategy_selected"] and not result["bitwise_native_outputs"]
        )

    coefficients = tuple(original["meridional_coefficients"])
    setup_mutations: list[tuple[str, dict[str, Any]]] = [
        ("eccentricity_zero", {"es": 0.0}),
        ("eccentricity_negative", {"es": math.nextafter(0.0, -math.inf)}),
        ("eccentricity_nextabove", {"es": math.nextafter(0.012, math.inf)}),
        ("eccentricity_nonfinite", {"es": math.nan}),
        ("leading_coefficient_zero", {"meridional_coefficients": (0.0, *coefficients[1:])}),
        (
            "leading_coefficient_negative",
            {
                "meridional_coefficients": (
                    math.nextafter(0.0, -math.inf),
                    *coefficients[1:],
                )
            },
        ),
        ("central_meridian_nonfinite", {"lam0": math.inf}),
        ("scale_zero", {"a": 0.0}),
        ("scale_nonfinite", {"a": math.inf}),
        (
            "scale_nextabove",
            {"a": math.nextafter(PROJECTION_FIXED_Q62_MAX_SCALE_M, math.inf)},
        ),
        ("x_offset_nonfinite", {"x0": math.inf}),
        ("y_offset_nonfinite", {"y0": math.nan}),
        ("x_unit_zero", {"x_unit_to_m": 0.0}),
        ("x_unit_nonfinite", {"x_unit_to_m": math.inf}),
        ("y_unit_zero", {"y_unit_to_m": 0.0}),
        ("y_unit_nonfinite", {"y_unit_to_m": math.nan}),
    ]
    for index in range(8):
        mutated = list(coefficients)
        mutated[index] = math.nan
        setup_mutations.append(
            (f"coefficient_{index}_nonfinite", {"meridional_coefficients": tuple(mutated)})
        )

    setup_probes = []
    try:
        for label, mutation in setup_mutations:
            pipeline.computed = {**original, **mutation}
            result = execute()
            for policy, policy_result in result["policies"].items():
                policy_result["native_strategy_pass"] = policy_result["strategy"][
                    "implementation_ids"
                ] == [NATIVE_LIBDEVICE]
                policy_result["behavior_pass"] = (
                    policy_result["native_strategy_pass"]
                    and policy_result["bitwise_native_outputs"]
                )
            setup_probes.append({"name": label, "mutated_fields": sorted(mutation), **result})
    finally:
        pipeline.computed = original

    qualified_boundary_probes = []
    for label, mutation in (
        ("eccentricity_exact_upper", {"es": 0.012}),
        ("scale_exact_upper", {"a": PROJECTION_FIXED_Q62_MAX_SCALE_M}),
    ):
        if label == "eccentricity_exact_upper":
            target = CRS.from_user_input(
                "+proj=sinu +lon_0=0 +a=6400000 +es=.012 +units=m +type=crs"
            )
            boundary_transformer = Transformer.from_crs(target, target.geodetic_crs, always_xy=True)
            boundary_forward = PyProjTransformer.from_crs(
                target.geodetic_crs, target, always_xy=True
            )
            boundary_projected = boundary_forward.transform(longitude, latitude)
            result = execute(
                boundary_transformer,
                cp.asarray(np.asarray(boundary_projected[0], dtype=np.float64)),
                cp.asarray(np.asarray(boundary_projected[1], dtype=np.float64)),
            )
        else:
            pipeline.computed = {**original, **mutation}
            try:
                result = execute()
            finally:
                pipeline.computed = original
        for policy, policy_result in result["policies"].items():
            error = policy_result["error_vs_native_m"]
            policy_result["strategy_selected"] = policy_result["strategy"][
                "implementation_ids"
            ] == [policy_ids[policy]]
            policy_result["behavior_pass"] = (
                policy_result["strategy_selected"]
                and error["max_m"] is not None
                and error["max_m"] <= case.qualification.coordinate_contract_m
                and error["nonfinite_match"]
            )
        qualified_boundary_probes.append(
            {"name": label, "mutated_fields": sorted(mutation), **result}
        )

    hot_pass = all(
        result["nonvacuous_pass"]
        and result["error_vs_native_m"]["max_m"] is not None
        and result["error_vs_native_m"]["max_m"] <= case.qualification.coordinate_contract_m
        and result["error_vs_native_m"]["nonfinite_match"]
        for result in hot_probe["policies"].values()
    )
    return {
        "required": True,
        "hot_probe": hot_probe,
        "probes": setup_probes,
        "qualified_boundary_probes": qualified_boundary_probes,
        "qualification_pass": hot_pass
        and all(
            result["behavior_pass"]
            for probe in setup_probes
            for result in probe["policies"].values()
        )
        and all(
            result["behavior_pass"]
            for probe in qualified_boundary_probes
            for result in probe["policies"].values()
        ),
    }


def _krovak_parameter_guard_behavior(
    cp: Any,
    case: BenchmarkCase,
    precision: str,
) -> dict[str, Any]:
    """Exercise Krovak setup, unit/sign, scale, and restored-state replay."""
    pipeline = case.transformer._pipeline_for_direction(case.direction)
    original = pipeline.computed
    n = min(4_096, case.host_x.size)
    first = cp.asarray(case.host_x[:n])
    second = cp.asarray(case.host_y[:n])

    def execute() -> dict[str, Any]:
        outputs = {}
        for policy in ("native", "accelerated", "auto"):
            out_x = cp.empty(n, dtype=cp.float64)
            out_y = cp.empty(n, dtype=cp.float64)
            case.transformer.transform_buffers(
                first,
                second,
                direction=case.direction,
                out_x=out_x,
                out_y=out_y,
                precision=precision,
                transcendentals=policy,
            )
            outputs[policy] = (cp.asnumpy(out_x), cp.asnumpy(out_y))
        native_x, native_y = outputs["native"]
        accelerated_x, accelerated_y = outputs["accelerated"]
        auto_x, auto_y = outputs["auto"]
        return {
            "accelerated_strategy": _resolved_strategy(
                case.transformer,
                "accelerated",
                case.direction,
                precision=precision,
                workload_size=n,
            ),
            "auto_strategy": _resolved_strategy(
                case.transformer,
                "auto",
                case.direction,
                precision=precision,
                workload_size=n,
            ),
            "accelerated_bitwise_native": bool(
                np.array_equal(accelerated_x.view(np.uint64), native_x.view(np.uint64))
                and np.array_equal(accelerated_y.view(np.uint64), native_y.view(np.uint64))
            ),
            "auto_bitwise_native": bool(
                np.array_equal(auto_x.view(np.uint64), native_x.view(np.uint64))
                and np.array_equal(auto_y.view(np.uint64), native_y.view(np.uint64))
            ),
            "error_vs_native_m": _coordinate_error(
                accelerated_x,
                accelerated_y,
                native_x,
                native_y,
                geographic=True,
                geographic_scale_m=case.geographic_scale_m,
            ),
        }

    hot = execute()
    hot["strategy_selected"] = hot["accelerated_strategy"]["implementation_ids"] == [
        KROVAK_INVERSE_GUARDED_LOG_RATIO
    ]
    hot["auto_native_pass"] = (
        hot["auto_strategy"]["implementation_ids"] == [NATIVE_LIBDEVICE]
        and hot["auto_bitwise_native"]
    )

    cgb = tuple(original["cgb"])
    mutations: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "setup": [
            (
                "custom_method",
                {
                    "_strategy_operation_method": "Custom Krovak",
                    "_strategy_krovak_semantics": "custom",
                },
            ),
            (
                "spherical_geometry",
                {
                    "_strategy_geometry": "spherical",
                    "_strategy_krovak_semantics": "spherical",
                },
            ),
            (
                "eccentricity_nonfinite",
                {"e": math.nan, "_strategy_krovak_semantics": "invalid_setup"},
            ),
            ("B_nonfinite", {"B": math.inf, "_strategy_krovak_semantics": "invalid_setup"}),
            ("k_nonfinite", {"k": math.nan, "_strategy_krovak_semantics": "invalid_setup"}),
            ("cone_nonpositive", {"n": 0.0, "_strategy_krovak_semantics": "invalid_setup"}),
            (
                "radius_nonpositive",
                {"r_0_norm": 0.0, "_strategy_krovak_semantics": "invalid_setup"},
            ),
            (
                "tan_half_p_nonpositive",
                {"tan_half_p": 0.0, "_strategy_krovak_semantics": "invalid_setup"},
            ),
            (
                "sin_alpha_nonfinite",
                {"sin_alpha_c": math.nan, "_strategy_krovak_semantics": "invalid_setup"},
            ),
            (
                "cos_alpha_nonfinite",
                {"cos_alpha_c": math.inf, "_strategy_krovak_semantics": "invalid_setup"},
            ),
            (
                "series_nonfinite",
                {
                    "cgb": (math.nan, *cgb[1:]),
                    "_strategy_krovak_semantics": "invalid_setup",
                },
            ),
            (
                "log_k_nonfinite",
                {"log_k": math.inf, "_strategy_krovak_semantics": "invalid_setup"},
            ),
            (
                "central_meridian_nonfinite",
                {"lam0": math.nan, "_strategy_krovak_semantics": "invalid_setup"},
            ),
            (
                "x_offset_nonfinite",
                {"x0": math.inf, "_strategy_krovak_semantics": "invalid_setup"},
            ),
            (
                "y_offset_nonfinite",
                {"y0": math.nan, "_strategy_krovak_semantics": "invalid_setup"},
            ),
        ],
        "units": [
            (
                "x_unit_zero",
                {"x_unit_to_m": 0.0, "_strategy_krovak_semantics": "invalid_setup"},
            ),
            (
                "y_unit_nonfinite",
                {"y_unit_to_m": math.inf, "_strategy_krovak_semantics": "invalid_setup"},
            ),
            (
                "wrong_easting_sign",
                {
                    "easting_axis_sign": -original["easting_axis_sign"],
                    "_strategy_krovak_semantics": "invalid_setup",
                },
            ),
            (
                "wrong_northing_sign",
                {
                    "northing_axis_sign": -original["northing_axis_sign"],
                    "_strategy_krovak_semantics": "invalid_setup",
                },
            ),
        ],
        "scale": [
            (
                "non_bessel_scale",
                {"a": original["a"] + 1.0, "_strategy_krovak_semantics": "custom"},
            ),
            (
                "scale_nonfinite",
                {"a": math.inf, "_strategy_krovak_semantics": "invalid_setup"},
            ),
        ],
    }
    probes: dict[str, list[dict[str, Any]]] = {category: [] for category in mutations}
    try:
        for category, category_mutations in mutations.items():
            for label, mutation in category_mutations:
                pipeline.computed = {**original, **mutation}
                result = execute()
                result["native_dispatch_pass"] = result["accelerated_strategy"][
                    "implementation_ids"
                ] == [NATIVE_LIBDEVICE]
                result["behavior_pass"] = (
                    result["native_dispatch_pass"]
                    and result["accelerated_bitwise_native"]
                    and result["auto_bitwise_native"]
                )
                probes[category].append(
                    {"name": label, "mutated_fields": sorted(mutation), **result}
                )
    finally:
        pipeline.computed = original

    replay = execute()
    replay["strategy_selected"] = replay["accelerated_strategy"]["implementation_ids"] == [
        KROVAK_INVERSE_GUARDED_LOG_RATIO
    ]
    replay_error = replay["error_vs_native_m"]
    replay["behavior_pass"] = (
        replay["strategy_selected"]
        and replay["auto_strategy"]["implementation_ids"] == [NATIVE_LIBDEVICE]
        and replay["auto_bitwise_native"]
        and replay_error["max_m"] is not None
        and replay_error["max_m"] <= case.qualification.coordinate_contract_m
        and replay_error["nonfinite_match"]
    )
    hot_error = hot["error_vs_native_m"]
    hot_pass = (
        hot["strategy_selected"]
        and hot["auto_native_pass"]
        and hot_error["max_m"] is not None
        and hot_error["max_m"] <= case.qualification.coordinate_contract_m
        and hot_error["nonfinite_match"]
    )
    return {
        "required": True,
        "hot_probe": hot,
        "setup_probes": probes["setup"],
        "unit_probes": probes["units"],
        "scale_probes": probes["scale"],
        "replay": replay,
        "qualification_pass": hot_pass
        and replay["behavior_pass"]
        and all(probe["behavior_pass"] for category in probes.values() for probe in category),
    }


def _parameter_guard_behavior(
    cp: Any,
    case: BenchmarkCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if case.family == "merc":
        return _merc_parameter_guard_behavior(cp, case, args.precision)
    if case.family == "lcc":
        return _lcc_parameter_guard_behavior(cp, case, args.precision)
    if case.family == "sinu" and case.qualification.direction == "inverse":
        return _sinu_inverse_parameter_guard_behavior(cp, case, args.precision)
    if case.family == "krovak":
        return _krovak_parameter_guard_behavior(cp, case, args.precision)
    return _stere_inverse_e_guard_behavior(cp, case, args)


def _run_case(cp: Any, case: BenchmarkCase, args: argparse.Namespace) -> dict[str, Any]:
    outputs = {
        policy: (
            cp.empty(args.n, dtype=cp.float64),
            cp.empty(args.n, dtype=cp.float64),
        )
        for policy in POLICIES
    }
    launches: dict[str, Callable[[], None]] = {}
    invocations: dict[str, Callable[[Any | None], tuple[Any, ...]]] = {}
    for policy in POLICIES:
        out_x, out_y = outputs[policy]

        def invoke(
            stream: Any | None = None,
            policy: str = policy,
            out_x: Any = out_x,
            out_y: Any = out_y,
        ) -> tuple[Any, ...]:
            result = case.transformer.transform_buffers(
                case.input_x,
                case.input_y,
                direction=case.direction,
                out_x=out_x,
                out_y=out_y,
                precision=args.precision,
                transcendentals=policy,
                stream=stream,
            )
            return (*result, result[0] is out_x and result[1] is out_y)

        def launch(invoke: Callable[[Any | None], tuple[Any, ...]] = invoke) -> None:
            invoke(None)

        launches[policy] = launch
        invocations[policy] = invoke

    timing, pool_before, pool_after = _time_interleaved(
        cp,
        launches,
        n=args.n,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    wall_timing = _time_synchronized_wall_interleaved(
        cp,
        launches,
        n=args.n,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    instrumentation = {}
    for policy in POLICIES:
        invoke = invocations[policy]
        instrumentation[policy] = {
            "allocator": _measure_allocator_calls(cp, lambda invoke=invoke: invoke(None)),
            "cuda_graph": _capture_cuda_graph(cp, invoke),
        }

    for launch in launches.values():
        launch()
    cp.cuda.get_current_stream().synchronize()
    host_outputs = {
        policy: (cp.asnumpy(out_x), cp.asnumpy(out_y)) for policy, (out_x, out_y) in outputs.items()
    }
    native_x, native_y = host_outputs["native"]
    errors_vs_native = {
        policy: _coordinate_error(
            out_x,
            out_y,
            native_x,
            native_y,
            geographic=case.output_is_geographic,
            geographic_scale_m=case.geographic_scale_m,
        )
        for policy, (out_x, out_y) in host_outputs.items()
    }

    oracle_n = min(args.oracle_n, args.n)
    oracle_x, oracle_y = _pyproj_reference(case, oracle_n)
    errors_vs_pyproj = {
        policy: _coordinate_error(
            out_x[:oracle_n],
            out_y[:oracle_n],
            oracle_x,
            oracle_y,
            geographic=case.output_is_geographic,
            geographic_scale_m=case.geographic_scale_m,
        )
        for policy, (out_x, out_y) in host_outputs.items()
    }
    edge_error_vs_native = _edge_accuracy(cp, case, args.precision)
    scale_guard = _scale_guard_behavior(cp, case, args.precision)
    parameter_guard = _parameter_guard_behavior(cp, case, args)
    kernel_resources = _kernel_resources(cp, case)
    fallback_sweep = _guarded_inverse_fallback_sweep(cp, case, args)

    strategies = {
        policy: _resolved_strategy(
            case.transformer,
            policy,
            case.direction,
            precision=args.precision,
            workload_size=args.n,
        )
        for policy in POLICIES
    }
    accelerated_ids = strategies["accelerated"]["implementation_ids"]
    qualification = case.qualification
    accelerated_is_native = qualification.implementation_id == NATIVE_LIBDEVICE
    expected_accelerated_ids = [qualification.implementation_id]
    candidate_policy_ids = _candidate_policy_ids(qualification)
    candidate_policies = _distinct_candidate_policies(qualification)
    auto_implementation_id = candidate_policy_ids["auto"]
    expected_auto_id = (
        auto_implementation_id
        if qualification.auto_enabled
        and not accelerated_is_native
        and args.n >= qualification.min_elements
        else NATIVE_LIBDEVICE
    )
    repeat_speedups = timing["accelerated"]["repeat_speedups_vs_native"]
    wall_repeat_speedups = wall_timing["accelerated"]["repeat_speedups_vs_native"]
    auto_repeat_speedups = timing["auto"]["repeat_speedups_vs_native"]
    auto_wall_repeat_speedups = wall_timing["auto"]["repeat_speedups_vs_native"]
    coordinate_contract_m = qualification.coordinate_contract_m
    accelerated_native_nonfinite_match = errors_vs_native["accelerated"]["nonfinite_match"]
    native_pyproj_error = errors_vs_pyproj["native"]["max_m"]
    native_pyproj_nonfinite_match = errors_vs_pyproj["native"]["nonfinite_match"]
    accelerated_pyproj_nonfinite_match = errors_vs_pyproj["accelerated"]["nonfinite_match"]
    candidate_native_contract_passes = {
        policy: errors_vs_native[policy]["max_m"] is not None
        and errors_vs_native[policy]["max_m"] <= coordinate_contract_m
        and errors_vs_native[policy]["nonfinite_match"]
        for policy in candidate_policies
    }
    candidate_edge_contract_passes = {
        policy: edge_error_vs_native["by_policy"][policy]["max_m"] is not None
        and edge_error_vs_native["by_policy"][policy]["max_m"] <= coordinate_contract_m
        and edge_error_vs_native["by_policy"][policy]["nonfinite_match"]
        for policy in candidate_policies
    }
    candidate_pyproj_regression_passes = {
        policy: errors_vs_pyproj[policy]["max_m"] is not None
        and native_pyproj_error is not None
        and errors_vs_pyproj[policy]["max_m"] <= native_pyproj_error + coordinate_contract_m
        and (not native_pyproj_nonfinite_match or errors_vs_pyproj[policy]["nonfinite_match"])
        for policy in candidate_policies
    }
    allocators = [instrumentation[policy]["allocator"] for policy in POLICIES]
    graphs = [instrumentation[policy]["cuda_graph"] for policy in POLICIES]
    gates = {
        "speedup_threshold": 1.05,
        "accelerated_resolved_native_only": accelerated_is_native,
        "native_policy_resolution_pass": strategies["native"]["implementation_ids"]
        == [NATIVE_LIBDEVICE],
        "accelerated_policy_resolution_pass": accelerated_ids == expected_accelerated_ids,
        "auto_policy_resolution_pass": strategies["auto"]["implementation_ids"]
        == [expected_auto_id],
        "median_speedup_pass": accelerated_is_native
        or timing["accelerated"]["speedup_vs_native"] >= 1.05,
        "three_repeat_speedup_pass": accelerated_is_native
        or all(speedup >= 1.05 for speedup in repeat_speedups),
        "wall_median_speedup_pass": accelerated_is_native
        or wall_timing["accelerated"]["speedup_vs_native"] >= 1.05,
        "wall_three_repeat_speedup_pass": accelerated_is_native
        or all(speedup >= 1.05 for speedup in wall_repeat_speedups),
        "auto_median_speedup_pass": "auto" not in candidate_policies
        or timing["auto"]["speedup_vs_native"] >= 1.05,
        "auto_three_repeat_speedup_pass": "auto" not in candidate_policies
        or all(speedup >= 1.05 for speedup in auto_repeat_speedups),
        "auto_wall_median_speedup_pass": "auto" not in candidate_policies
        or wall_timing["auto"]["speedup_vs_native"] >= 1.05,
        "auto_wall_three_repeat_speedup_pass": "auto" not in candidate_policies
        or all(speedup >= 1.05 for speedup in auto_wall_repeat_speedups),
        "no_steady_state_allocator_calls": all(
            count == 0
            for allocator in allocators
            for count in allocator["allocator_calls_per_transform"]
        ),
        "preallocated_output_identity": all(
            all(allocator["returned_preallocated_outputs_per_transform"])
            for allocator in allocators
        )
        and all(graph["returned_preallocated_outputs"] for graph in graphs),
        "cuda_graph_capture_pass": all(graph["captured"] for graph in graphs),
        "expected_kernel_topology_pass": all(
            graph["captured"] and graph["kernel_nodes"] == qualification.expected_kernel_nodes
            for graph in graphs
        ),
        "no_copy_nodes": all(graph["captured"] and graph["memcpy_nodes"] == 0 for graph in graphs),
        "no_memset_nodes": all(
            graph["captured"] and graph["memset_nodes"] == 0 for graph in graphs
        ),
        "native_coordinate_contract_m": coordinate_contract_m,
        "native_coordinate_contract_pass": all(candidate_native_contract_passes.values()),
        "candidate_native_coordinate_contract_passes": candidate_native_contract_passes,
        "accelerated_native_nonfinite_match": accelerated_native_nonfinite_match,
        "edge_coordinate_contract_pass": all(candidate_edge_contract_passes.values()),
        "candidate_edge_coordinate_contract_passes": candidate_edge_contract_passes,
        "scale_guard_native_behavior_pass": scale_guard["qualification_pass"],
        "parameter_guard_native_behavior_pass": parameter_guard["qualification_pass"],
        "fallback_sweep_qualification_pass": not fallback_sweep.get("required", False)
        or bool(fallback_sweep.get("qualification_pass", False)),
        "no_pyproj_regression": all(candidate_pyproj_regression_passes.values()),
        "candidate_pyproj_regression_passes": candidate_pyproj_regression_passes,
        "accelerated_pyproj_nonfinite_match": accelerated_pyproj_nonfinite_match,
        "native_pyproj_nonfinite_match": native_pyproj_nonfinite_match,
    }
    required_gate_names = [
        "median_speedup_pass",
        "three_repeat_speedup_pass",
        "wall_median_speedup_pass",
        "wall_three_repeat_speedup_pass",
        "native_policy_resolution_pass",
        "accelerated_policy_resolution_pass",
        "auto_policy_resolution_pass",
        "no_steady_state_allocator_calls",
        "preallocated_output_identity",
        "cuda_graph_capture_pass",
        "expected_kernel_topology_pass",
        "no_copy_nodes",
        "no_memset_nodes",
        "native_coordinate_contract_pass",
        "edge_coordinate_contract_pass",
        "scale_guard_native_behavior_pass",
        "parameter_guard_native_behavior_pass",
        "fallback_sweep_qualification_pass",
        "no_pyproj_regression",
    ]
    if "auto" in candidate_policies:
        required_gate_names.extend(
            (
                "auto_median_speedup_pass",
                "auto_three_repeat_speedup_pass",
                "auto_wall_median_speedup_pass",
                "auto_wall_three_repeat_speedup_pass",
            )
        )
    gates["qualification_pass"] = all(bool(gates[name]) for name in required_gate_names)
    return {
        "case": case.name,
        "family": case.family,
        "direction": case.direction.lower(),
        "precision": args.precision,
        "n": args.n,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "domain": case.domain,
        "edge_inputs": case.edges,
        "qualification_spec": asdict(qualification),
        "strategies": strategies,
        "device_timing": timing,
        "wall_timing": wall_timing,
        "error_vs_native_m": errors_vs_native,
        "edge_error_vs_native_m": edge_error_vs_native,
        "scale_guard": scale_guard,
        "parameter_guard": parameter_guard,
        "error_vs_pyproj_m": errors_vs_pyproj,
        "instrumentation": instrumentation,
        "kernel_resources": kernel_resources,
        "fallback_sweep": fallback_sweep,
        "hot_path": {
            "preallocated_outputs": True,
            "expected_kernel_nodes": qualification.expected_kernel_nodes,
            "copy_direction": (
                "No transfer direction is inferred; CUDA graph labels are reported verbatim."
            ),
            "memory_pool_observation": (
                "Secondary residency metric only; allocator-call instrumentation is authoritative."
            ),
            "memory_pool_used_bytes_before": pool_before,
            "memory_pool_used_bytes_after": pool_after,
            "memory_pool_growth_bytes": pool_after - pool_before,
        },
        "gates": gates,
    }


def _run_topology_probes(cp: Any, precision: str) -> list[dict[str, Any]]:
    """Capture representative multi-stage projection/Helmert/SVD pipelines."""
    from pyproj import Transformer as PyProjTransformer
    from vibeproj import Transformer

    n = 4096
    longitude = np.linspace(-5.0, 2.0, n, dtype=np.float64)
    latitude = np.linspace(50.0, 60.0, n, dtype=np.float64)
    conus_longitude = np.linspace(-120.0, -70.0, n, dtype=np.float64)
    conus_latitude = np.linspace(25.0, 50.0, n, dtype=np.float64)
    nad27_utm = PyProjTransformer.from_crs("EPSG:4267", "EPSG:26718", always_xy=True)
    easting, northing = nad27_utm.transform(conus_longitude, conus_latitude)
    specs = (
        (
            "helmert-plus-projection",
            Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True),
            longitude,
            latitude,
            None,
            2,
        ),
        (
            "helmert-3d-geographic",
            Transformer.from_crs("EPSG:4326", "EPSG:4277", always_xy=True),
            longitude,
            latitude,
            np.linspace(-500.0, 10_000.0, n, dtype=np.float64),
            1,
        ),
        (
            "svd-geographic",
            Transformer.from_crs("EPSG:4267", "EPSG:4269", always_xy=True),
            conus_longitude,
            conus_latitude,
            None,
            1,
        ),
        (
            "projection-svd-projection",
            Transformer.from_crs("EPSG:26718", "EPSG:26918", always_xy=True),
            np.asarray(easting, dtype=np.float64),
            np.asarray(northing, dtype=np.float64),
            None,
            3,
        ),
    )
    probes = []
    for name, transformer, host_x, host_y, host_z, expected_kernel_nodes in specs:
        input_x = cp.asarray(host_x)
        input_y = cp.asarray(host_y)
        input_z = cp.asarray(host_z) if host_z is not None else None
        out_x = cp.empty(n, dtype=cp.float64)
        out_y = cp.empty(n, dtype=cp.float64)
        out_z = cp.empty(n, dtype=cp.float64) if host_z is not None else None
        transformer.compile(precision=precision, transcendentals="native")

        def invoke(stream: Any | None = None) -> tuple[Any, ...]:
            result = transformer.transform_buffers(
                input_x,
                input_y,
                input_z,
                out_x=out_x,
                out_y=out_y,
                out_z=out_z,
                precision=precision,
                transcendentals="native",
                stream=stream,
            )
            output_identity = result[0] is out_x and result[1] is out_y
            if out_z is not None:
                output_identity = output_identity and result[2] is out_z
            return (*result, output_identity)

        invoke(None)
        cp.cuda.get_current_stream().synchronize()
        allocator = _measure_allocator_calls(cp, lambda: invoke(None))
        graph = _capture_cuda_graph(cp, invoke)
        gates = {
            "no_steady_state_allocator_calls": all(
                count == 0 for count in allocator["allocator_calls_per_transform"]
            ),
            "preallocated_output_identity": all(
                allocator["returned_preallocated_outputs_per_transform"]
            )
            and graph["returned_preallocated_outputs"],
            "cuda_graph_capture_pass": graph["captured"],
            "expected_kernel_topology_pass": graph["captured"]
            and graph["kernel_nodes"] == expected_kernel_nodes,
            "no_copy_nodes": graph["captured"] and graph["memcpy_nodes"] == 0,
            "no_memset_nodes": graph["captured"] and graph["memset_nodes"] == 0,
        }
        gates["qualification_pass"] = all(gates.values())
        probes.append(
            {
                "topology": name,
                "n": n,
                "expected_kernel_nodes": expected_kernel_nodes,
                "allocator": allocator,
                "cuda_graph": graph,
                "gates": gates,
            }
        )
    return probes


def _run_workload_grid(
    cp: Any,
    args: argparse.Namespace,
    selected_cases: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Measure public dispatch over the grid plus each crossover boundary."""
    rows = []
    case_names = tuple(name for name in WORKLOAD_GRID_CASES if name in selected_cases)
    for case_index, case_name in enumerate(case_names):
        qualification = QUALIFICATION_SPECS[case_name]
        workload_sizes = _qualification_workload_sizes(args.workload_sizes, qualification)
        for n in workload_sizes:
            case = _prepare_case(cp, case_name, n, args.seed + 10_000 + case_index)
            outputs = {
                policy: (cp.empty(n, dtype=cp.float64), cp.empty(n, dtype=cp.float64))
                for policy in POLICIES
            }
            launches: dict[str, Callable[[], None]] = {}
            for policy in POLICIES:
                out_x, out_y = outputs[policy]

                def launch(
                    policy: str = policy,
                    out_x: Any = out_x,
                    out_y: Any = out_y,
                ) -> None:
                    case.transformer.transform_buffers(
                        case.input_x,
                        case.input_y,
                        direction=case.direction,
                        out_x=out_x,
                        out_y=out_y,
                        precision=args.precision,
                        transcendentals=policy,
                    )

                launches[policy] = launch

            device_timing, _, _ = _time_interleaved(
                cp,
                launches,
                n=n,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            wall_timing = _time_synchronized_wall_interleaved(
                cp,
                launches,
                n=n,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            strategies = {
                policy: _resolved_strategy(
                    case.transformer,
                    policy,
                    case.direction,
                    precision=args.precision,
                    workload_size=n,
                )
                for policy in POLICIES
            }
            accelerated_id = qualification.implementation_id
            min_elements = qualification.min_elements
            below_crossover = qualification.auto_enabled and n < min_elements
            auto_implementation_id = qualification.auto_implementation_id or accelerated_id
            expected_auto_id = (
                auto_implementation_id
                if qualification.auto_enabled and not below_crossover
                else NATIVE_LIBDEVICE
            )
            auto_wall_speedup = wall_timing["auto"]["speedup_vs_native"]
            auto_wall_repeat_speedups = wall_timing["auto"]["repeat_speedups_vs_native"]
            auto_device_speedup = device_timing["auto"]["speedup_vs_native"]
            auto_device_repeat_speedups = device_timing["auto"]["repeat_speedups_vs_native"]
            accelerated_wall = wall_timing["accelerated"]
            accelerated_device = device_timing["accelerated"]
            explicit_performance_required = (
                qualification.auto_enabled and not below_crossover
            ) or (
                qualification.explicit_performance_min_elements is not None
                and n >= qualification.explicit_performance_min_elements
            )
            gates = {
                "explicit_accelerated_override_pass": strategies["accelerated"][
                    "implementation_ids"
                ]
                == [accelerated_id],
                "size_aware_auto_resolution_pass": strategies["auto"]["implementation_ids"]
                == [expected_auto_id],
                # Identical native execution can jitter around 1.0 at tiny N.
                # Bound synchronized-wall aggregate noise to 2% and every
                # wall repeat to 5%. CUDA-event medians at ~0.03 ms are
                # quantized too coarsely to gate identical cached kernels.
                "small_auto_native_identity_noise_pass": (
                    qualification.auto_enabled and not below_crossover
                )
                or _native_identity_noise_pass(
                    auto_implementation_ids=strategies["auto"]["implementation_ids"],
                    native_implementation_ids=strategies["native"]["implementation_ids"],
                    wall_speedup=auto_wall_speedup,
                    wall_repeat_speedups=auto_wall_repeat_speedups,
                ),
                "qualified_auto_wall_median_pass": not qualification.auto_enabled
                or below_crossover
                or auto_wall_speedup >= 1.05,
                "qualified_auto_wall_three_repeat_pass": not qualification.auto_enabled
                or below_crossover
                or all(speedup >= 1.05 for speedup in auto_wall_repeat_speedups),
                "qualified_auto_device_median_pass": not qualification.auto_enabled
                or below_crossover
                or auto_device_speedup >= 1.05,
                "qualified_auto_device_three_repeat_pass": not qualification.auto_enabled
                or below_crossover
                or all(speedup >= 1.05 for speedup in auto_device_repeat_speedups),
                "qualified_explicit_wall_median_pass": not explicit_performance_required
                or accelerated_wall["speedup_vs_native"] >= 1.05,
                "qualified_explicit_wall_three_repeat_pass": not explicit_performance_required
                or all(
                    speedup >= 1.05 for speedup in accelerated_wall["repeat_speedups_vs_native"]
                ),
                "qualified_explicit_device_median_pass": not explicit_performance_required
                or accelerated_device["speedup_vs_native"] >= 1.05,
                "qualified_explicit_device_three_repeat_pass": not explicit_performance_required
                or all(
                    speedup >= 1.05 for speedup in accelerated_device["repeat_speedups_vs_native"]
                ),
            }
            gates["qualification_pass"] = all(gates.values())
            rows.append(
                {
                    "case": case_name,
                    "n": n,
                    "precision": args.precision,
                    "qualified_implementation_id": accelerated_id,
                    "qualified_auto_implementation_id": auto_implementation_id,
                    "min_elements": min_elements,
                    "threshold_boundary": qualification.auto_enabled
                    and n in ({1, 2} if min_elements == 1 else {min_elements - 1, min_elements}),
                    "auto_enabled": qualification.auto_enabled,
                    "explicit_performance_required": explicit_performance_required,
                    "below_crossover": below_crossover,
                    "strategies": strategies,
                    "device_timing": device_timing,
                    "wall_timing": wall_timing,
                    "gates": gates,
                }
            )
    return rows


def _print_human(results: dict[str, Any]) -> None:
    device = results["meta"]["device"]
    print(f"GPU: {device['name']} ({device['sm']})")
    print(
        f"N={results['meta']['n']:,}; warmup={results['meta']['warmup']}; "
        f"iterations={results['meta']['iterations']}; repeats={results['meta']['repeats']}"
    )
    for result in results["results"]:
        print()
        print(f"{result['case']} ({result['precision']})")
        print("CUDA-event device execution:")
        print(
            f"{'policy':<14} {'resolved IDs':<38} {'median ms':>10} "
            f"{'Mcoord/s':>10} {'vs native':>10} {'p05..p95 ms':>20}"
        )
        for policy in POLICIES:
            ids = ",".join(result["strategies"][policy]["implementation_ids"])
            stats = result["device_timing"][policy]
            spread = f"{stats['p05_ms']:.4f}..{stats['p95_ms']:.4f}"
            print(
                f"{policy:<14} {ids:<38} {stats['median_ms']:>10.4f} "
                f"{stats['mcoords_per_s']:>10.1f} "
                f"{stats['speedup_vs_native']:>9.3f}x {spread:>20}"
            )
        print("synchronized public-dispatch wall medians:")
        for policy in POLICIES:
            stats = result["wall_timing"][policy]
            print(
                f"  {policy:<12} {stats['median_ms']:.4f} ms "
                f"({stats['speedup_vs_native']:.3f}x vs native; "
                f"p05..p95={stats['p05_ms']:.4f}..{stats['p95_ms']:.4f} ms)"
            )
        error = result["error_vs_native_m"]["accelerated"]
        print(
            "accelerated error vs native: "
            f"max={error['max_m']!s} m, p99={error['p99_m']!s} m, "
            f"RMS={error['rms_m']!s} m"
        )
        print(
            "gates: speedup={median_speedup_pass}, three-repeat={three_repeat_speedup_pass}, "
            "wall-speedup={wall_median_speedup_pass}, "
            "wall-three-repeat={wall_three_repeat_speedup_pass}, "
            "native-error={native_coordinate_contract_pass}, "
            "edge-error={edge_coordinate_contract_pass}, "
            "pyproj-regression={no_pyproj_regression}, "
            "no-allocation={no_steady_state_allocator_calls}, "
            "output-identity={preallocated_output_identity}, "
            "kernel-topology={expected_kernel_topology_pass}, "
            "no-copy={no_copy_nodes}".format(**result["gates"])
        )
    if results["topology_probes"]:
        print()
        print("Untimed CUDA-graph topology probes")
        for probe in results["topology_probes"]:
            graph = probe["cuda_graph"]
            print(
                f"  {probe['topology']}: kernels={graph['kernel_nodes']}/"
                f"{probe['expected_kernel_nodes']}, copies={graph['memcpy_nodes']}, "
                f"allocator calls={sum(probe['allocator']['allocator_calls_per_transform'])}, "
                f"pass={probe['gates']['qualification_pass']}"
            )
    if results["workload_grid"]:
        print()
        print("Synchronized public-dispatch workload grid")
        for row in results["workload_grid"]:
            accelerated = row["wall_timing"]["accelerated"]
            auto = row["wall_timing"]["auto"]
            print(
                f"  {row['case']:<16} N={row['n']:>8,}: "
                f"accelerated={accelerated['speedup_vs_native']:.3f}x, "
                f"auto={auto['speedup_vs_native']:.3f}x, "
                f"pass={row['gates']['qualification_pass']}"
            )


def _write_json(results: dict[str, Any], destination: str) -> None:
    payload = json.dumps(results, indent=2, allow_nan=False) + "\n"
    if destination == "-":
        print(payload, end="")
        return
    Path(destination).write_text(payload, encoding="utf-8")
    print(f"Results written to {destination}", file=sys.stderr)


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import cupy as cp
    except ImportError as exc:
        raise SystemExit("CuPy is required for this GPU benchmark") from exc

    cp.cuda.Device(args.device).use()
    cp.cuda.Device(args.device).synchronize()
    selected_cases = CASES if args.case == "all" else CASE_GROUPS.get(args.case, (args.case,))
    results = [
        _run_case(cp, _prepare_case(cp, name, args.n, args.seed + index), args)
        for index, name in enumerate(selected_cases)
    ]
    topology_probes = _run_topology_probes(cp, args.precision) if args.case == "all" else []
    workload_grid = _run_workload_grid(cp, args, selected_cases) if args.workload_sizes else []
    return {
        "meta": {
            "device": _device_metadata(cp, args.device),
            "python": platform.python_version(),
            "cupy": cp.__version__,
            "vibeproj_benchmark": "public transcendental policy qualification",
            "n": args.n,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "oracle_n": min(args.oracle_n, args.n),
            "policies": list(POLICIES),
            "workload_sizes": list(args.workload_sizes),
            "qualification_specs": {
                name: asdict(specification) for name, specification in QUALIFICATION_SPECS.items()
            },
            "wave2a_rejected_candidates": WAVE2A_REJECTED_CANDIDATES,
            "wave3c_lcc_rejected_candidates": WAVE3C_LCC_REJECTED_CANDIDATES,
            "wave3c_lcc_forward_threshold_evidence": WAVE3C_LCC_FORWARD_THRESHOLD_EVIDENCE,
            "ortho_inverse_threshold_note": (
                "N=524288 is the first tested size whose three actual-public repeats "
                "all reached the 1.05 gate in the retained qualification runs; every "
                "larger tested size also passed, so the conservative auto threshold "
                "is 524288."
            ),
            "workload_grid_includes_threshold_boundaries": True,
            "interleaved_cuda_events": True,
            "cuda_event_measurement": "device execution between CUDA events",
            "wall_measurement": (
                "perf_counter around warmed public transform_buffers dispatch plus "
                "completion synchronization"
            ),
        },
        "results": results,
        "topology_probes": topology_probes,
        "workload_grid": workload_grid,
    }


def _requires_default_workload_grid(case: str) -> bool:
    """Whether enforced qualification requires the full historical 20-size grid."""
    if (
        case in {"merc", "lcc", "krovak-inverse"}
        or case.startswith("sinu-inverse")
        or case.startswith("krovak-inverse-")
    ):
        return False
    selected_cases = CASES if case == "all" else CASE_GROUPS.get(case, (case,))
    return any(QUALIFICATION_SPECS[name].auto_enabled for name in selected_cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", *CASE_GROUPS, *CASES), default="all")
    parser.add_argument("--n", type=int, default=1_000_000, help="coordinates per case")
    parser.add_argument("--warmup", type=int, default=10, help="interleaved warmup rounds")
    parser.add_argument("--iterations", type=int, default=50, help="samples per repeat/policy")
    parser.add_argument("--repeats", type=int, default=3, help="independent timed repeats")
    parser.add_argument(
        "--oracle-n", type=int, default=100_000, help="leading coordinates checked vs pyproj"
    )
    parser.add_argument("--precision", choices=("auto", "fp64", "fp32", "ds"), default="fp64")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--workload-sizes",
        type=lambda value: tuple(int(item) for item in value.split(",") if item),
        default=DEFAULT_WORKLOAD_SIZES,
        metavar="N[,N...]",
        help="synchronized wall-time qualification sizes; pass an empty string to disable",
    )
    parser.add_argument(
        "--enforce-gates",
        action="store_true",
        help="exit nonzero if any qualification gate fails",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        metavar="PATH",
        help="emit JSON to PATH, or stdout when PATH is omitted",
    )
    args = parser.parse_args()
    if (
        min(args.n, args.iterations, args.repeats, args.oracle_n) <= 0
        or args.warmup < 0
        or any(size <= 0 for size in args.workload_sizes)
    ):
        parser.error(
            "n, iterations, repeats, and oracle-n must be positive; warmup must be non-negative"
        )
    if (
        args.enforce_gates
        and _requires_default_workload_grid(args.case)
        and not set(DEFAULT_WORKLOAD_SIZES).issubset(args.workload_sizes)
    ):
        parser.error(
            "--enforce-gates requires every default workload size from "
            f"{DEFAULT_WORKLOAD_SIZES[0]} to {DEFAULT_WORKLOAD_SIZES[-1]}"
        )
    if args.enforce_gates and args.precision not in ("auto", "fp64"):
        parser.error("--enforce-gates requires fp64-equivalent precision ('auto' or 'fp64')")

    results = run(args)
    if args.json:
        _write_json(results, args.json)
    else:
        _print_human(results)
    if args.enforce_gates and not all(
        result["gates"]["qualification_pass"] for result in results["results"]
    ):
        raise SystemExit("One or more transcendental qualification gates failed")
    if args.enforce_gates and not all(
        probe["gates"]["qualification_pass"] for probe in results["topology_probes"]
    ):
        raise SystemExit("One or more CUDA topology qualification gates failed")
    requires_workload_grid = (
        args.case == "all" or args.case in CASE_GROUPS or args.case in WORKLOAD_GRID_CASES
    )
    if args.enforce_gates and requires_workload_grid and not results["workload_grid"]:
        raise SystemExit("The workload-size qualification grid is required by --enforce-gates")
    if args.enforce_gates and not all(
        row["gates"]["qualification_pass"] for row in results["workload_grid"]
    ):
        raise SystemExit("One or more workload-size qualification gates failed")


if __name__ == "__main__":
    main()
