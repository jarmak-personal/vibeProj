#!/usr/bin/env python3
"""Reproduce why the Wave 2C stereographic forward prototypes were rejected.

The rejected forward CUDA implementations are intentionally absent from
production; Polar Stereographic inverse was independently promoted as
``stere.inverse.fixed_q62``. This artifact records synchronized RTX 4090 measurements and
deterministically reconstructs the correctness-guard eligibility that caused
the full-domain failures. In particular, an outlined STEREA forward candidate
is fast inside the Netherlands RD bbox, but that bbox is not a distinct public
strategy domain; randomly ordered full-globe input in the same exact domain is
slower than native.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from vibeproj.crs import ProjectionParams
from vibeproj.ellipsoid import SPHERE, WGS84
from vibeproj.projections.oblique_stereographic import ObliqueStereographic
from vibeproj.projections.stereographic import PolarStereographic


SEED = 20260805
RD_SEED = 555
DEFAULT_COORDINATES = 1_000_000
WARP_SIZE = 32
ERROR_GATE_M = 1e-8
SPEEDUP_GATE = 1.05
Q62_SCALE_LIMIT_M = 6_400_000.0
RD_LONGITUDE_DEGREES = (3.2, 7.3)
RD_LATITUDE_DEGREES = (50.7, 53.7)

# This is the essential cold path of the strongest forward prototype. Its hot
# path used sin(chi)=(w-1)/(w+1), cos(chi)=sqrt(1-sin(chi)^2), and Q1.62 lam_s.
# The helper restored the complete native pair atomically when either the
# angular guard or denominator-amplification guard failed.
OUTLINED_STEREA_NATIVE_HELPER = r"""
__device__ __noinline__ void vp_sterea_forward_native_pairs_cold(
    double chi_argument,
    double lam_s,
    double* sin_chi,
    double* cos_chi,
    double* sin_lam_s,
    double* cos_lam_s
) {
    const double chi = asin(chi_argument);
    sincos(chi, sin_chi, cos_chi);
    sincos(lam_s, sin_lam_s, cos_lam_s);
}
""".strip()

# Ratios are native/candidate, so values below one are regressions. The first
# four screens used N=2,097,152, 10 warmups, and three interleaved blocks of 30
# preallocated one-kernel launches. Device and wall measurements used the same
# synchronized launches. The outlined RD screen used N=5,000,000 and five
# blocks of 30 launches; its raw device medians are retained below.
RESEARCH_RESULTS: dict[str, dict[str, object]] = {
    "stere.forward.q62_longitude.full_globe": {
        "n": 2_097_152,
        "device_speedup_repeats": [0.9407457, 0.9406175, 0.9406175],
        "wall_speedup_repeats": [0.9410956, 0.9407469, 0.9406099],
        "max_error_m": 1.862645149230957e-9,
        "registers": 40,
        "local_bytes": 40,
        "shared_bytes": 0,
        "reason": "accuracy passes, but antipodal scale fallbacks make the full globe slower",
    },
    "sterea.forward.q62_pair.full_globe": {
        "n": 2_097_152,
        "device_speedup_repeats": [0.8957550, 0.8960417, 0.8959950],
        "wall_speedup_repeats": [0.8960395, 0.8961170, 0.8962233],
        "max_error_m": 1.4551915228366852e-9,
        "registers": 40,
        "local_bytes": 40,
        "shared_bytes": 0,
        "reason": "the conformal-antipode amplification guard makes almost every full-globe lane native",
    },
    "sterea.forward.algebraic_chi_current.full_globe": {
        "n": 2_097_152,
        "device_speedup_repeats": [0.9012872, 0.9013165, 0.9005950],
        "wall_speedup_repeats": [0.9014411, 0.9014215, 0.9006107],
        "max_error_m": 1.982e-9,
        "registers": 40,
        "local_bytes": 40,
        "shared_bytes": 0,
        "reason": "full-globe fallback restores accuracy but costs more than native",
    },
    "sterea.forward.algebraic_chi_outlined.rd_bbox.direct_kernel": {
        "n": 5_000_000,
        "native_device_ms": [3.15286, 3.15254, 2.96970, 2.89451, 2.89468],
        "candidate_device_ms": [2.60113, 2.60372, 2.60250, 2.60244, 2.60226],
        "device_speedup_repeats": [
            1.2121116592,
            1.2107830335,
            1.1410951009,
            1.1122292925,
            1.1123715540,
        ],
        "median_device_speedup": 1.1410951009,
        "max_error_m": 2.213e-9,
        "dataset": {
            "seed": RD_SEED,
            "longitude_degrees_uniform": RD_LONGITUDE_DEGREES,
            "latitude_degrees_uniform": RD_LATITUDE_DEGREES,
        },
        "reason": "raw fused kernel passes locally, but excludes production dispatch overhead",
    },
    "sterea.forward.algebraic_chi_outlined.full_globe.direct_kernel": {
        "n": 5_000_000,
        "native_device_ms": 3.0277,
        "candidate_device_ms": 3.3606,
        "median_device_speedup": 0.9010,
        "max_error_m": 1.982e-9,
        "reason": "same implementation and exact domain regress on full-globe input",
    },
    "sterea.forward.algebraic_chi_outlined.rd_bbox.public_path": {
        "n": 5_000_000,
        "device_speedup_repeats": [1.0436422356, 1.0440755706, 1.0440865494],
        "wall_speedup_repeats": [1.0437245101, 1.0440558889, 1.0439298868],
        "native_device_median_ms": 6.993632,
        "candidate_device_median_ms": 6.697728,
        "native_wall_median_ms": 6.9951835,
        "candidate_wall_median_ms": 6.7001055,
        "fast_coordinates": 5_000_000,
        "max_error_m": 2.328306e-9,
        "p99_error_m": 1.513679e-9,
        "native_registers": 34,
        "candidate_registers": 40,
        "native_local_bytes": 40,
        "candidate_local_bytes": 72,
        "shared_bytes": 0,
        "dataset": {
            "seed": RD_SEED,
            "longitude_degrees_uniform": RD_LONGITUDE_DEGREES,
            "latitude_degrees_uniform": RD_LATITUDE_DEGREES,
        },
        "reason": "accuracy passes, but every public-path repeat misses the 1.05x gate",
    },
    "sterea.forward.algebraic_chi_outlined.full_globe.public_path": {
        "n": 5_000_000,
        "device_speedup_repeats": [0.9584381403, 0.9584363105, 0.9583725250],
        "wall_speedup_repeats": [0.9583858803, 0.9584144815, 0.9583544418],
        "native_device_median_ms": 7.774928,
        "candidate_device_median_ms": 8.112128,
        "native_wall_median_ms": 7.776125,
        "candidate_wall_median_ms": 8.113725,
        "fast_coordinates": 13_959,
        "max_error_m": 1.982481e-9,
        "p99_error_m": 0.0,
        "native_registers": 34,
        "candidate_registers": 40,
        "native_local_bytes": 40,
        "candidate_local_bytes": 72,
        "shared_bytes": 0,
        "dataset": {
            "seed": RD_SEED,
            "longitude_degrees_uniform": [-180.0, 180.0],
            "latitude_distribution": "degrees(asin(U[-1,1]))",
        },
        "reason": "only 0.27918% of lanes are fast and the full exact domain regresses",
    },
}


def _warp_profile(eligible: np.ndarray) -> dict[str, float | int]:
    complete = eligible[: eligible.size - eligible.size % WARP_SIZE]
    warp_eligible = complete.reshape(-1, WARP_SIZE).all(axis=1)
    return {
        "coordinates": int(eligible.size),
        "eligible_lane_fraction": float(np.mean(eligible)),
        "native_fallback_lane_fraction": float(np.mean(~eligible)),
        "all_eligible_warp_fraction": float(np.mean(warp_eligible)),
        "native_fallback_warp_fraction": float(np.mean(~warp_eligible)),
    }


def _stere_profile(
    *, ellipsoid, latitude_origin: float, n: int, rng: np.random.Generator
) -> dict[str, float | int]:
    computed = PolarStereographic().setup(
        ProjectionParams("stere", ellipsoid, lat_0=latitude_origin, k_0=0.994)
    )
    latitude = np.arcsin(rng.uniform(-1.0, 1.0, n))
    phi_adjusted = computed["sign"] * latitude
    sin_phi = np.sin(phi_adjusted)
    e_sin = computed["e"] * sin_phi
    t = np.tan(0.5 * (math.pi / 2.0 - phi_adjusted)) / ((1.0 - e_sin) / (1.0 + e_sin)) ** (
        0.5 * computed["e"]
    )
    physical_rho = np.abs(computed["akm1"] * t * computed["a"])
    eligible = (
        np.isfinite(physical_rho) & (physical_rho > 0.0) & (physical_rho <= Q62_SCALE_LIMIT_M)
    )
    return _warp_profile(eligible)


def _sterea_forward_terms(
    *, n: int, rng: np.random.Generator, rd_bbox: bool
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    computed = ObliqueStereographic().setup(
        ProjectionParams(
            "sterea",
            WGS84,
            lat_0=52.156160556,
            lon_0=5.387638889,
            k_0=0.9999079,
        )
    )
    if rd_bbox:
        longitude = np.deg2rad(rng.uniform(*RD_LONGITUDE_DEGREES, n)) - computed["lam0"]
        latitude = np.deg2rad(rng.uniform(*RD_LATITUDE_DEGREES, n))
    else:
        longitude = rng.uniform(-math.pi, math.pi, n)
        latitude = np.arcsin(rng.uniform(-1.0, 1.0, n))
    longitude -= 2.0 * math.pi * np.rint(longitude / (2.0 * math.pi))
    sin_phi = np.sin(latitude)
    conformal_s = ((1.0 + sin_phi) / (1.0 - sin_phi)) * (
        (1.0 - computed["e"] * sin_phi) / (1.0 + computed["e"] * sin_phi)
    ) ** computed["e"]
    w = computed["c"] * conformal_s ** computed["n"]
    sin_chi = (w - 1.0) / (w + 1.0)
    cos_chi = np.sqrt(np.maximum(0.0, 1.0 - sin_chi * sin_chi))
    lam_s = computed["n"] * longitude
    sin_lam_s = np.sin(lam_s)
    cos_lam_s = np.cos(lam_s)
    denominator = 1.0 + computed["sin_chi0"] * sin_chi + computed["cos_chi0"] * cos_chi * cos_lam_s
    amplification = np.abs(2.0 * computed["R"] * computed["k0"] * computed["a"] / denominator)
    eligible = (
        np.isfinite(sin_chi)
        & (np.abs(sin_chi) <= 1.0)
        & (np.abs(lam_s) <= math.pi)
        & np.isfinite(amplification)
        & (amplification > 0.0)
        & (amplification <= Q62_SCALE_LIMIT_M)
    )
    return computed, eligible, sin_chi, cos_chi, sin_lam_s, cos_lam_s


def reproduce_wave2c_rejection(n: int = DEFAULT_COORDINATES) -> dict[str, object]:
    """Return deterministic guard profiles and retained measured evidence."""
    stere_domains = {}
    for geometry, ellipsoid in (("spherical", SPHERE), ("ellipsoidal", WGS84)):
        for variant in ("variant_a", "variant_b", "variant_c", "custom"):
            for hemisphere, latitude_origin in (("north", 90.0), ("south", -90.0)):
                stere_domains[f"stere.forward.{geometry}.{variant}.{hemisphere}"] = _stere_profile(
                    ellipsoid=ellipsoid,
                    latitude_origin=latitude_origin,
                    n=n,
                    rng=np.random.default_rng(SEED),
                )
    _, full_eligible, *_ = _sterea_forward_terms(
        n=n, rng=np.random.default_rng(SEED), rd_bbox=False
    )
    _, rd_eligible, *_ = _sterea_forward_terms(
        n=n, rng=np.random.default_rng(RD_SEED), rd_bbox=True
    )
    results = {}
    for name, measurement in RESEARCH_RESULTS.items():
        speedups = measurement.get("wall_speedup_repeats")
        if speedups is None:
            median_speedup = float(measurement.get("median_device_speedup", math.nan))
        else:
            median_speedup = float(np.median(speedups))
        max_error = float(measurement["max_error_m"])
        results[name] = {
            **measurement,
            "accuracy_pass": max_error <= ERROR_GATE_M,
            "local_speedup_pass": median_speedup >= SPEEDUP_GATE,
            "complete_exact_domain": "full_globe" in name,
            "qualification_pass": max_error <= ERROR_GATE_M
            and median_speedup >= SPEEDUP_GATE
            and "full_globe" in name,
        }
    return {
        "seed": SEED,
        "coordinates_per_profile": n,
        "warp_size": WARP_SIZE,
        "accuracy_gate_m": ERROR_GATE_M,
        "speedup_gate": SPEEDUP_GATE,
        "exact_domain_matrix": {
            "stere": {
                "geometry": ["spherical", "ellipsoidal"],
                "variant": ["variant_a", "variant_b", "variant_c", "custom"],
                "hemisphere": ["north", "south"],
            },
            "sterea": {
                "forward": ["sterea.forward.ellipsoidal.oblique"],
            },
        },
        "profiles": {
            **stere_domains,
            "sterea.forward.ellipsoidal.oblique.full_globe": _warp_profile(full_eligible),
            "sterea.forward.ellipsoidal.oblique.rd_bbox": _warp_profile(rd_eligible),
        },
        "research_results": results,
        "outlined_sterea_native_helper": OUTLINED_STEREA_NATIVE_HELPER,
        "decision": {
            "status": "forward_candidates_rejected",
            "promoted": ["stere.inverse.fixed_q62"],
            "reason": (
                "No forward candidate meets both the 10 nm and 1.05x gates over its "
                "complete exact public domain. The positive RD bbox is not independently "
                "dispatchable without data inspection, allocation, or synchronization."
            ),
        },
        "reproduce_command": "uv run python benchmarks/wave2c_rejection.py",
        "rejected_implementations_retained": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_COORDINATES)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    payload = json.dumps(reproduce_wave2c_rejection(args.n), indent=2, sort_keys=True)
    print(payload)
    if args.json is not None:
        args.json.write_text(payload + "\n")


if __name__ == "__main__":
    main()
