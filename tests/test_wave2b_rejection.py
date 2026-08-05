from __future__ import annotations

import importlib.util
from pathlib import Path


BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "wave2b_rejection.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("wave2b_rejection", BENCHMARK_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retained_wave2b_results_are_explicit_no_go() -> None:
    benchmark = _load_benchmark()
    report = benchmark.reproduce_wave2b_rejection(n=3_200)

    assert report["rejected_implementations_retained"] is False
    assert report["research_results"]
    assert all(not result["qualification_pass"] for result in report["research_results"].values())


def test_guard_profiles_capture_warp_divergence() -> None:
    benchmark = _load_benchmark()
    profiles = benchmark.reproduce_wave2b_rejection(n=32_000)["profiles"]

    laea = profiles["laea.forward.ellipsoidal.oblique"]["algebraic_beta"]
    geos = profiles["geos.inverse.ellipsoidal.sweep_x"]
    assert laea["eligible_lane_fraction"] > laea["all_eligible_warp_fraction"]
    assert geos["eligible_lane_fraction"] > geos["all_eligible_warp_fraction"]
    assert laea["native_fallback_warp_fraction"] > 0.5
    assert geos["native_fallback_warp_fraction"] > 0.5
