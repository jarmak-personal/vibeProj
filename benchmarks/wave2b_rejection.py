#!/usr/bin/env python3
"""Reproduce why the rejected Wave 2B candidates did not qualify.

The rejected experimental CUDA kernels are intentionally absent from
production. This artifact deterministically reconstructs their correctness
guards over the full LAEA globe/disk and GEOS visible surface, then reports the
retained RTX 4090 measurements. A candidate cannot help a randomly ordered
workload when almost every warp contains at least one coordinate that requires
exact native math.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from vibeproj.crs import ProjectionParams
from vibeproj.ellipsoid import SPHERE, WGS84
from vibeproj.projections._equal_area import authalic_q
from vibeproj.projections.geostationary import Geostationary
from vibeproj.projections.lambert_azimuthal_equal_area import (
    LambertAzimuthalEqualArea,
)


SEED = 20260805
DEFAULT_COORDINATES = 1_000_000
WARP_SIZE = 32
ERROR_GATE_M = 1e-8
SPEEDUP_GATE = 1.05

# Compact results from synchronized three-repeat research screens. They keep
# the rejection reviewable without dead production implementation IDs or
# machine-local artifact paths; the deterministic guard profiles below remain
# directly reproducible.
RESEARCH_RESULTS = {
    "laea.forward.ellipsoidal.oblique.algebraic_beta": {
        "n": 5_000_000,
        "device_speedup": 0.8341699123613556,
        "wall_speedup": 0.8351019564543412,
        "max_error_m": 7.957223565613414e-09,
        "reason": "accuracy passes, full-globe throughput fails",
    },
    "laea.forward.ellipsoidal.oblique.algebraic_beta_q62_lambda": {
        "n": 5_000_000,
        "device_speedup": 0.8729597309401326,
        "wall_speedup": 0.8735542740400323,
        "max_error_m": 1.1519824039985408e-08,
        "reason": "full-globe accuracy and throughput both fail",
    },
    "laea.forward.spherical.oblique.spherical_direct_q62": {
        "n": 200_000,
        "device_speedup": 0.8681,
        "wall_speedup": 0.8799,
        "max_error_m": 5.890201144234053e-09,
        "reason": "accuracy passes after guard, full-globe throughput fails",
    },
    "laea.forward.spherical.oblique.spherical_direct_native": {
        "n": 200_000,
        "device_speedup": 0.7900,
        "wall_speedup": 0.8078,
        "max_error_m": 6.054e-09,
        "reason": "accuracy passes after guard, full-globe throughput fails",
    },
    "laea.inverse.ellipsoidal.oblique.half_angle": {
        "n": 200_000,
        "device_speedup": 0.9291213483480976,
        "wall_speedup": 0.9349230504512839,
        "max_error_m": 7.978413549535443e-09,
        "reason": "accuracy passes, full-disk throughput fails",
    },
    "geos.inverse.ellipsoidal.sweep_x.q62_sincos": {
        "n": 200_000,
        "device_speedup": 0.7988591240624697,
        "wall_speedup": 0.8129387037030428,
        "max_error_m": 0.0,
        "reason": "random visible-limb warps become bitwise-native and slower",
    },
    "geos.forward.lambda_only_q62": {
        "n": 5_000_000,
        "device_speedup": 1.033,
        "wall_speedup": 1.036,
        "max_error_m": 3.725e-09,
        "reason": "accuracy passes but throughput does not meet the 1.05x gate",
    },
    "laea.forward.ellipsoidal.polar.q62_lambda": {
        "n": 5_000_000,
        "device_speedup": 1.0453,
        "wall_speedup": 1.0462,
        "max_error_m": 3.725e-09,
        "reason": "accuracy passes but public throughput does not meet the 1.05x gate",
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


def _laea_forward_profile(
    *, geometry: str, mode: str, n: int, rng: np.random.Generator
) -> dict[str, object]:
    ellipsoid = SPHERE if geometry == "spherical" else WGS84
    latitude_origin = 0.0 if mode == "equatorial" else 45.0
    projection = LambertAzimuthalEqualArea()
    computed = projection.setup(ProjectionParams("laea", ellipsoid, lat_0=latitude_origin))
    longitude = rng.uniform(-math.pi, math.pi, n)
    latitude = np.arcsin(rng.uniform(-1.0, 1.0, n))
    q = authalic_q(np.sin(latitude), computed["e"], np)
    sin_beta = q / computed["qp"]
    cos_beta = np.sqrt(np.maximum(0.0, (1.0 - sin_beta) * (1.0 + sin_beta)))
    if mode == "equatorial":
        denominator = 1.0 + cos_beta * np.cos(longitude)
    else:
        denominator = (
            1.0
            + computed["sin_beta0"] * sin_beta
            + computed["cos_beta0"] * cos_beta * np.cos(longitude)
        )
    algebraic_eligible = (np.abs(sin_beta) <= 0.99) & np.isfinite(denominator) & (denominator > 0.2)
    result: dict[str, object] = {"algebraic_beta": _warp_profile(algebraic_eligible)}
    if geometry == "spherical":
        spherical_direct_eligible = (
            (np.abs(np.sin(latitude)) <= 0.95) & np.isfinite(denominator) & (denominator > 0.2)
        )
        result["spherical_direct"] = _warp_profile(spherical_direct_eligible)
    return result


def _laea_inverse_profile(
    *, geometry: str, mode: str, n: int, rng: np.random.Generator
) -> dict[str, float | int]:
    ellipsoid = SPHERE if geometry == "spherical" else WGS84
    latitude_origin = 0.0 if mode == "equatorial" else 45.0
    computed = LambertAzimuthalEqualArea().setup(
        ProjectionParams("laea", ellipsoid, lat_0=latitude_origin)
    )
    t = np.sqrt(rng.uniform(0.0, np.nextafter(1.0, 0.0), n))
    azimuth = rng.uniform(-math.pi, math.pi, n)
    rho = 2.0 * computed["Rq"] * t
    y_adjusted = rho * np.sin(azimuth)
    sin_ce = 2.0 * t * np.sqrt(np.maximum(0.0, (1.0 - t) * (1.0 + t)))
    cos_ce = 1.0 - 2.0 * t * t
    if mode == "equatorial":
        sin_beta = y_adjusted * sin_ce / rho
    else:
        sin_beta = (
            cos_ce * computed["sin_beta0"] + y_adjusted * sin_ce * computed["cos_beta0"] / rho
        )
    eligible = (t > 1e-12) & (t <= 0.99) & (np.abs(sin_beta) <= 0.95)
    return _warp_profile(eligible)


def _geos_inverse_profile(
    *, geometry: str, sweep: str, n: int, rng: np.random.Generator
) -> dict[str, float | int]:
    ellipsoid = SPHERE if geometry == "spherical" else WGS84
    computed = Geostationary().setup(
        ProjectionParams(
            "geos",
            ellipsoid,
            extra={"h": 35_785_831.0, "sweep_axis": sweep},
        )
    )
    latitude_parts = []
    longitude_parts = []
    remaining = n
    while remaining:
        batch = max(4_096, int(remaining * 1.25))
        latitude = np.arcsin(
            rng.uniform(-math.sin(math.radians(81.0)), math.sin(math.radians(81.0)), batch)
        )
        phi_gc = np.arctan(computed["r_pol2"] / computed["r_eq2"] * np.tan(latitude))
        cos_phi_gc = np.cos(phi_gc)
        r_earth = math.sqrt(computed["r_pol2"]) / np.sqrt(
            1.0
            - (computed["r_eq2"] - computed["r_pol2"]) / computed["r_eq2"] * cos_phi_gc * cos_phi_gc
        )
        ratio = computed["r_eq2"] / (computed["H"] * r_earth * cos_phi_gc)
        valid = np.isfinite(ratio) & (ratio < 1.0)
        latitude = latitude[valid]
        longitude_limit = np.arccos(np.clip(ratio[valid], -1.0, 1.0))
        count = min(remaining, latitude.size)
        latitude = latitude[:count]
        longitude_limit = longitude_limit[:count]
        near_limb = rng.random(count) < 0.20
        fraction = rng.uniform(-1.0, 1.0, count)
        distance = np.power(10.0, rng.uniform(-9.0, -3.0, count))
        sign = np.where(rng.random(count) < 0.5, -1.0, 1.0)
        fraction = np.where(near_limb, sign * (1.0 - distance), fraction)
        latitude_parts.append(latitude)
        longitude_parts.append(longitude_limit * fraction)
        remaining -= count
    latitude = np.concatenate(latitude_parts)
    longitude = np.concatenate(longitude_parts)
    phi_gc = np.arctan(computed["r_pol2"] / computed["r_eq2"] * np.tan(latitude))
    sin_pgc, cos_pgc = np.sin(phi_gc), np.cos(phi_gc)
    r_earth = math.sqrt(computed["r_pol2"]) / np.sqrt(
        1.0 - (computed["r_eq2"] - computed["r_pol2"]) / computed["r_eq2"] * cos_pgc * cos_pgc
    )
    point_x = r_earth * cos_pgc * np.cos(longitude)
    point_y = r_earth * cos_pgc * np.sin(longitude)
    point_z = r_earth * sin_pgc
    satellite_x = computed["H"] - point_x
    if sweep == "x":
        x_angle = np.arctan2(point_y, np.hypot(satellite_x, point_z))
        y_angle = np.arctan2(point_z, satellite_x)
    else:
        x_angle = np.arctan2(point_y, satellite_x)
        y_angle = np.arctan2(point_z, np.hypot(satellite_x, point_y))
    eligible = (np.abs(x_angle) <= 0.035) & (np.abs(y_angle) <= 0.035)
    return _warp_profile(eligible)


def reproduce_wave2b_rejection(n: int = DEFAULT_COORDINATES) -> dict[str, object]:
    """Return deterministic guard eligibility and retained measured evidence."""
    profiles: dict[str, object] = {}
    for geometry in ("spherical", "ellipsoidal"):
        for mode in ("equatorial", "oblique"):
            profiles[f"laea.forward.{geometry}.{mode}"] = _laea_forward_profile(
                geometry=geometry,
                mode=mode,
                n=n,
                rng=np.random.default_rng(SEED),
            )
            profiles[f"laea.inverse.{geometry}.{mode}"] = _laea_inverse_profile(
                geometry=geometry,
                mode=mode,
                n=n,
                rng=np.random.default_rng(SEED),
            )
        for sweep in ("x", "y"):
            profiles[f"geos.inverse.{geometry}.sweep_{sweep}"] = _geos_inverse_profile(
                geometry=geometry,
                sweep=sweep,
                n=n,
                rng=np.random.default_rng(SEED),
            )
    results = {
        name: {
            **measurement,
            "accuracy_pass": measurement["max_error_m"] <= ERROR_GATE_M,
            "device_speedup_pass": measurement["device_speedup"] >= SPEEDUP_GATE,
            "wall_speedup_pass": measurement["wall_speedup"] >= SPEEDUP_GATE,
            "qualification_pass": (
                measurement["max_error_m"] <= ERROR_GATE_M
                and measurement["device_speedup"] >= SPEEDUP_GATE
                and measurement["wall_speedup"] >= SPEEDUP_GATE
            ),
        }
        for name, measurement in RESEARCH_RESULTS.items()
    }
    return {
        "seed": SEED,
        "coordinates_per_profile": n,
        "warp_size": WARP_SIZE,
        "accuracy_gate_m": ERROR_GATE_M,
        "speedup_gate": SPEEDUP_GATE,
        "profiles": profiles,
        "research_results": results,
        "reproduce_command": "uv run python benchmarks/wave2b_rejection.py",
        "rejected_implementations_retained": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_COORDINATES)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    payload = json.dumps(reproduce_wave2b_rejection(args.n), indent=2, sort_keys=True)
    print(payload)
    if args.json is not None:
        args.json.write_text(payload + "\n")


if __name__ == "__main__":
    main()
