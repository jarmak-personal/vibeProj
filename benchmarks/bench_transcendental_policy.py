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


QUALIFICATION_SPECS = {
    "tmerc-forward": QualificationSpec(
        implementation_id=TMERC_FIXED_Q62,
        operation="tmerc.forward",
        domain="utm",
        direction="forward",
        min_elements=TMERC_FIXED_Q62_MIN_ELEMENTS,
        coordinate_contract_m=1e-8,
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
        domain="sinu.forward",
        direction="forward",
        min_elements=SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS,
        coordinate_contract_m=1e-8,
        max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
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
}
CASES = tuple(QUALIFICATION_SPECS)
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


def _qualification_workload_sizes(
    requested_sizes: tuple[int, ...],
    qualification: QualificationSpec,
) -> tuple[int, ...]:
    """Return the requested grid with threshold-1 and threshold included."""
    if not requested_sizes:
        return ()
    return tuple(
        sorted(
            {
                *requested_sizes,
                qualification.min_elements - 1,
                qualification.min_elements,
            }
        )
    )


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


def _prepare_spherical_projection_forward(
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
) -> BenchmarkCase:
    """Prepare one reusable spherical forward-projection qualification case."""
    from vibeproj import Transformer

    transformer = Transformer.from_crs(
        _SPHERICAL_LONG_LAT_CRS,
        target_crs,
        always_xy=True,
    )
    longitude = rng.uniform(*longitude_range, n).astype(np.float64)
    latitude = rng.uniform(*latitude_range, n).astype(np.float64)
    family = name.removesuffix("-forward")
    return BenchmarkCase(
        name=name,
        family=family,
        direction="FORWARD",
        transformer=transformer,
        input_x=cp.asarray(longitude),
        input_y=cp.asarray(latitude),
        host_x=longitude,
        host_y=latitude,
        edge_host_x=edge_longitude,
        edge_host_y=edge_latitude,
        domain={
            "crs": f"{_SPHERICAL_LONG_LAT_CRS} -> {target_crs}",
            "longitude_degrees": list(longitude_range),
            "latitude_degrees": list(latitude_range),
        },
        edges={
            "longitude_degrees": _json_edge_values(edge_longitude),
            "latitude_degrees": _json_edge_values(edge_latitude),
        },
        output_is_geographic=False,
        oracle_from_crs=_SPHERICAL_LONG_LAT_CRS,
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
    return _prepare_spherical_projection_forward(
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
    return _prepare_spherical_projection_forward(
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
    if name == "ortho-forward":
        return _prepare_ortho_forward(cp, n, rng)
    if name == "ortho-inverse":
        return _prepare_ortho_inverse(cp, n, rng)
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
    for label, implementation_id in (
        ("native", NATIVE_LIBDEVICE),
        ("accelerated", case.qualification.implementation_id),
    ):
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


def _ortho_inverse_fallback_sweep(
    cp: Any,
    case: BenchmarkCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Measure deliberately mixed per-lane native fallback percentages."""
    if case.name != "ortho-inverse":
        return {"required": False, "rows": []}

    n = min(args.n, 1_000_000)
    rng = np.random.default_rng(args.seed + 50_000)
    azimuth = rng.uniform(-math.pi, math.pi, n)
    safe_radius = np.sqrt(rng.uniform(0.05, 0.88, n)) * EARTH_RADIUS_M
    safe_x = safe_radius * np.cos(azimuth)
    safe_y = safe_radius * np.sin(azimuth)
    fallback_radius = (
        np.sqrt(rng.uniform(np.nextafter(0.99, math.inf), np.nextafter(1.0, 0.0), n))
        * EARTH_RADIUS_M
    )
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
            warmup=min(args.warmup, 5),
            iterations=min(args.iterations, 20),
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
        "pattern": (
            "random lane mixture; fallback points are valid outer-disk coordinates "
            "with 0.99 < normalized rho_squared < 1.0"
        ),
        "rows": rows,
        "all_fallback_lanes_bitwise_native": all(
            row["fallback_lanes_bitwise_native"] for row in rows
        ),
    }


def _coordinate_error(
    actual_x: np.ndarray,
    actual_y: np.ndarray,
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    *,
    geographic: bool,
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
    if geographic:
        dlat_m = np.deg2rad(actual_y - reference_y) * EARTH_RADIUS_M
        delta_longitude = (actual_x - reference_x + 180.0) % 360.0 - 180.0
        dlon_m = np.deg2rad(delta_longitude) * EARTH_RADIUS_M * np.cos(np.deg2rad(reference_y))
        radial = np.hypot(dlat_m, dlon_m)
    else:
        radial = np.hypot(actual_x - reference_x, actual_y - reference_y)
    values = radial[finite]
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
    edge_x = cp.asarray(case.edge_host_x)
    edge_y = cp.asarray(case.edge_host_y)
    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for policy in ("native", "accelerated"):
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
        outputs[policy] = (cp.asnumpy(out_x), cp.asnumpy(out_y))
    native_x, native_y = outputs["native"]
    accelerated_x, accelerated_y = outputs["accelerated"]
    return _coordinate_error(
        accelerated_x,
        accelerated_y,
        native_x,
        native_y,
        geographic=case.output_is_geographic,
    )


def _scale_guard_behavior(cp: Any, case: BenchmarkCase, precision: str) -> dict[str, Any]:
    """Verify uniform native execution above a projection's scale contract."""
    maximum_scale = case.qualification.max_physical_scale_m
    if maximum_scale is None:
        return {"required": False, "qualification_pass": True, "probes": []}

    from vibeproj import Transformer

    probes = []
    for raw_scale in (np.nextafter(maximum_scale, math.inf), 1e12):
        physical_scale = float(raw_scale)
        source_crs = f"+proj=longlat +R={physical_scale:.17g} +type=crs"
        projection_parameters = f"+proj={case.family} +lon_0=0"
        if case.family == "ortho":
            projection_parameters += " +lat_0=0" if case.name == "ortho-inverse" else " +lat_0=45"
        target_crs = f"{projection_parameters} +R={physical_scale:.17g} +units=m +type=crs"
        if case.name == "ortho-inverse":
            transformer = Transformer.from_crs(target_crs, source_crs, always_xy=True)
            first_host = (
                np.asarray([0.1, -0.3, 0.7, 0.0, np.inf, np.nan], dtype=np.float64) * physical_scale
            )
            second_host = (
                np.asarray([0.2, 0.4, -0.1, 0.0, 0.0, 0.0], dtype=np.float64) * physical_scale
            )
        else:
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
            first_host = np.asarray([-180.0, -120.0, 0.0, 120.0, 180.0, np.inf, np.nan])
            second_host = np.asarray([-90.0, -45.0, 0.0, 45.0, 90.0, 40.0, 40.0])
        first = cp.asarray(first_host)
        second = cp.asarray(second_host)
        outputs = {}
        for policy in ("native", "accelerated"):
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
        native_x, native_y = outputs["native"]
        accelerated_x, accelerated_y = outputs["accelerated"]
        bitwise_native = bool(
            np.array_equal(accelerated_x.view(np.uint64), native_x.view(np.uint64))
            and np.array_equal(accelerated_y.view(np.uint64), native_y.view(np.uint64))
        )
        strategy = _resolved_strategy(
            transformer,
            "accelerated",
            "FORWARD",
            precision=precision,
            workload_size=int(first.size),
        )
        strategy_remains_selected = strategy["implementation_ids"] == [
            case.qualification.implementation_id
        ]
        probes.append(
            {
                "physical_scale_m": float(physical_scale),
                "strategy": strategy,
                "strategy_remains_selected": strategy_remains_selected,
                "bitwise_native_outputs": bitwise_native,
            }
        )
    return {
        "required": True,
        "maximum_qualified_scale_m": maximum_scale,
        "probes": probes,
        "qualification_pass": all(
            probe["strategy_remains_selected"] and probe["bitwise_native_outputs"]
            for probe in probes
        ),
    }


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
        )
        for policy, (out_x, out_y) in host_outputs.items()
    }
    edge_error_vs_native = _edge_accuracy(cp, case, args.precision)
    scale_guard = _scale_guard_behavior(cp, case, args.precision)
    kernel_resources = _kernel_resources(cp, case)
    fallback_sweep = _ortho_inverse_fallback_sweep(cp, case, args)

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
    expected_auto_id = (
        qualification.implementation_id
        if not accelerated_is_native and args.n >= qualification.min_elements
        else NATIVE_LIBDEVICE
    )
    repeat_speedups = timing["accelerated"]["repeat_speedups_vs_native"]
    wall_repeat_speedups = wall_timing["accelerated"]["repeat_speedups_vs_native"]
    coordinate_contract_m = qualification.coordinate_contract_m
    accelerated_native_error = errors_vs_native["accelerated"]["max_m"]
    native_pyproj_error = errors_vs_pyproj["native"]["max_m"]
    accelerated_pyproj_error = errors_vs_pyproj["accelerated"]["max_m"]
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
        "native_coordinate_contract_pass": accelerated_native_error is not None
        and accelerated_native_error <= coordinate_contract_m,
        "edge_coordinate_contract_pass": edge_error_vs_native["max_m"] is not None
        and edge_error_vs_native["max_m"] <= coordinate_contract_m
        and edge_error_vs_native["nonfinite_match"],
        "scale_guard_native_behavior_pass": scale_guard["qualification_pass"],
        "no_pyproj_regression": accelerated_pyproj_error is not None
        and native_pyproj_error is not None
        and accelerated_pyproj_error <= native_pyproj_error + coordinate_contract_m,
    }
    required_gate_names = (
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
        "no_pyproj_regression",
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

            timing = _time_synchronized_wall_interleaved(
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
            below_crossover = n < min_elements
            expected_auto_id = NATIVE_LIBDEVICE if below_crossover else accelerated_id
            auto_speedup = timing["auto"]["speedup_vs_native"]
            auto_repeat_speedups = timing["auto"]["repeat_speedups_vs_native"]
            gates = {
                "explicit_accelerated_override_pass": strategies["accelerated"][
                    "implementation_ids"
                ]
                == [accelerated_id],
                "size_aware_auto_resolution_pass": strategies["auto"]["implementation_ids"]
                == [expected_auto_id],
                # Identical native execution can jitter around 1.0 at tiny N.
                # Bound aggregate noise to 2% and every repeat to 5%.
                "small_auto_no_wall_regression_pass": not below_crossover
                or (
                    auto_speedup >= 0.98
                    and all(speedup >= 0.95 for speedup in auto_repeat_speedups)
                ),
                "qualified_auto_wall_median_pass": below_crossover or auto_speedup >= 1.05,
                "qualified_auto_wall_three_repeat_pass": below_crossover
                or all(speedup >= 1.05 for speedup in auto_repeat_speedups),
            }
            gates["qualification_pass"] = all(gates.values())
            rows.append(
                {
                    "case": case_name,
                    "n": n,
                    "precision": args.precision,
                    "qualified_implementation_id": accelerated_id,
                    "min_elements": min_elements,
                    "threshold_boundary": n in {min_elements - 1, min_elements},
                    "below_crossover": below_crossover,
                    "strategies": strategies,
                    "wall_timing": timing,
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
    selected_cases = CASES if args.case == "all" else (args.case,)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", *CASES), default="all")
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
    if args.enforce_gates and not set(DEFAULT_WORKLOAD_SIZES).issubset(args.workload_sizes):
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
    requires_workload_grid = args.case == "all" or args.case in WORKLOAD_GRID_CASES
    if args.enforce_gates and requires_workload_grid and not results["workload_grid"]:
        raise SystemExit("The workload-size qualification grid is required by --enforce-gates")
    if args.enforce_gates and not all(
        row["gates"]["qualification_pass"] for row in results["workload_grid"]
    ):
        raise SystemExit("One or more workload-size qualification gates failed")


if __name__ == "__main__":
    main()
