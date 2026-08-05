from __future__ import annotations

import importlib.util
from pathlib import Path


BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "wave2c_rejection.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("wave2c_rejection", BENCHMARK_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retained_wave2c_results_reject_only_forward_candidates() -> None:
    benchmark = _load_benchmark()
    report = benchmark.reproduce_wave2c_rejection(n=3_200)

    assert report["rejected_implementations_retained"] is False
    assert report["decision"]["status"] == "forward_candidates_rejected"
    assert report["decision"]["promoted"] == ["stere.inverse.fixed_q62"]
    assert report["research_results"]
    assert all(not result["qualification_pass"] for result in report["research_results"].values())


def test_wave2c_profiles_capture_rd_locality_and_warp_fallback() -> None:
    benchmark = _load_benchmark()
    profiles = benchmark.reproduce_wave2c_rejection(n=32_000)["profiles"]

    full = profiles["sterea.forward.ellipsoidal.oblique.full_globe"]
    rd = profiles["sterea.forward.ellipsoidal.oblique.rd_bbox"]
    assert rd["eligible_lane_fraction"] > full["eligible_lane_fraction"]
    assert rd["all_eligible_warp_fraction"] > full["all_eligible_warp_fraction"]
    assert full["native_fallback_warp_fraction"] > 0.99
