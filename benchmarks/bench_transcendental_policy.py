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


POLICIES = ("native", "accelerated", "auto")
CASES = ("tmerc-forward", "tmerc-inverse", "helmert-forward", "helmert-inverse")
WORKLOAD_GRID_CASES = ("tmerc-forward", "helmert-forward", "helmert-inverse")
EARTH_RADIUS_M = 6_378_137.0
DEFAULT_WORKLOAD_SIZES = (
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
    expected_kernel_nodes: int


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
        expected_kernel_nodes=1,
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
        expected_kernel_nodes=1,
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
        expected_kernel_nodes=1,
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

    if case.family == "tmerc":
        oracle = PyProjTransformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    else:
        oracle = PyProjTransformer.from_crs("EPSG:4326", "EPSG:4277", always_xy=True)
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
    accelerated_is_native = accelerated_ids == ["native.libdevice"]
    repeat_speedups = timing["accelerated"]["repeat_speedups_vs_native"]
    wall_repeat_speedups = wall_timing["accelerated"]["repeat_speedups_vs_native"]
    coordinate_contract_m = 1e-8
    accelerated_native_error = errors_vs_native["accelerated"]["max_m"]
    native_pyproj_error = errors_vs_pyproj["native"]["max_m"]
    accelerated_pyproj_error = errors_vs_pyproj["accelerated"]["max_m"]
    allocators = [instrumentation[policy]["allocator"] for policy in POLICIES]
    graphs = [instrumentation[policy]["cuda_graph"] for policy in POLICIES]
    gates = {
        "speedup_threshold": 1.05,
        "accelerated_resolved_native_only": accelerated_is_native,
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
            graph["captured"] and graph["kernel_nodes"] == case.expected_kernel_nodes
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
        "no_pyproj_regression": accelerated_pyproj_error is not None
        and native_pyproj_error is not None
        and accelerated_pyproj_error <= native_pyproj_error + coordinate_contract_m,
    }
    required_gate_names = (
        "median_speedup_pass",
        "three_repeat_speedup_pass",
        "wall_median_speedup_pass",
        "wall_three_repeat_speedup_pass",
        "no_steady_state_allocator_calls",
        "preallocated_output_identity",
        "cuda_graph_capture_pass",
        "expected_kernel_topology_pass",
        "no_copy_nodes",
        "no_memset_nodes",
        "native_coordinate_contract_pass",
        "edge_coordinate_contract_pass",
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
        "strategies": strategies,
        "device_timing": timing,
        "wall_timing": wall_timing,
        "error_vs_native_m": errors_vs_native,
        "edge_error_vs_native_m": edge_error_vs_native,
        "error_vs_pyproj_m": errors_vs_pyproj,
        "instrumentation": instrumentation,
        "hot_path": {
            "preallocated_outputs": True,
            "expected_kernel_nodes": case.expected_kernel_nodes,
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
    """Measure synchronized public dispatch over logarithmic workload sizes."""
    from vibeproj.transcendentals import (
        HELMERT_FIXED_Q62,
        HELMERT_FIXED_Q62_MIN_ELEMENTS,
        NATIVE_LIBDEVICE,
        TMERC_FIXED_Q62,
        TMERC_FIXED_Q62_MIN_ELEMENTS,
    )

    qualifications = {
        "tmerc-forward": (TMERC_FIXED_Q62, TMERC_FIXED_Q62_MIN_ELEMENTS),
        "helmert-forward": (HELMERT_FIXED_Q62, HELMERT_FIXED_Q62_MIN_ELEMENTS),
        "helmert-inverse": (HELMERT_FIXED_Q62, HELMERT_FIXED_Q62_MIN_ELEMENTS),
    }
    rows = []
    case_names = tuple(name for name in WORKLOAD_GRID_CASES if name in selected_cases)
    for case_index, case_name in enumerate(case_names):
        for n in args.workload_sizes:
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
            accelerated_id, min_elements = qualifications[case_name]
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
            "--enforce-gates requires every default logarithmic workload size from 32 to 5000000"
        )

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
