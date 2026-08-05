#!/usr/bin/env python3
"""Deterministically reproduce the rejected Eckert IV inverse Q1.62 edge.

The rejected experiment replaced the native ``sin(theta), cos(theta)`` pair
in Eckert IV inverse with the shared Q1.62 pair. Its bulk RTX 4090 timing was
benign, but the edge below amplifies sub-ULP trig error beyond the 10 nm
end-to-end contract. No production implementation uses this candidate.
"""

from __future__ import annotations

import json
import math


Q62_SCALE = 1 << 62
EARTH_RADIUS_M = 6_378_137.0
ECK4_C_X = 0.42223820031577120149
ECK4_C_P = 3.57079632679489661923
EDGE_CY_OVER_C_Y = -0.999994
EDGE_LONGITUDE_RAD = math.pi
ERROR_GATE_M = 1e-8
RESEARCH_DEVICE_SPEEDUP = 1.089134031099977
RESEARCH_WALL_SPEEDUP = 1.0891271130152704


def _q62_multiply(first: int, second: int) -> int:
    negative = (first < 0) != (second < 0)
    magnitude = (abs(first) * abs(second) + (1 << 61)) >> 62
    return -magnitude if negative else magnitude


def _q62_sincos(angle: float) -> tuple[float, float]:
    """Mirror the CUDA nearest-quadrant Q1.62 pair with integer arithmetic."""
    quadrant = round(angle * 0.6366197723675813430755350534900574)
    residual = round((angle - quadrant * 1.5707963267948966192313216916397514) * Q62_SCALE)
    residual_sq = _q62_multiply(residual, residual)

    sin_horner = 12966
    for coefficient in (
        -3526632,
        740592679,
        -115532457973,
        12708570377060,
        -915017067148291,
        38430716820228232,
        -768614336404564608,
        4611686018427387904,
    ):
        sin_horner = coefficient + _q62_multiply(residual_sq, sin_horner)

    cos_horner = -720
    for coefficient in (
        220414,
        -52899477,
        9627704831,
        -1270857037706,
        114377133393536,
        -6405119470038039,
        192153584101141152,
        -2305843009213693952,
        4611686018427387904,
    ):
        cos_horner = coefficient + _q62_multiply(residual_sq, cos_horner)

    sin_residual = _q62_multiply(residual, sin_horner)
    sin_fixed, cos_fixed = (
        (sin_residual, cos_horner),
        (cos_horner, -sin_residual),
        (-sin_residual, -cos_horner),
        (-cos_horner, sin_residual),
    )[quadrant & 3]
    return sin_fixed / Q62_SCALE, cos_fixed / Q62_SCALE


def reproduce_eck4_inverse_edge() -> dict[str, float | bool]:
    """Return the rejected candidate's deterministic horizontal edge error."""
    theta = math.asin(EDGE_CY_OVER_C_Y)
    native_sin = math.sin(theta)
    native_cos = math.cos(theta)
    fixed_sin, fixed_cos = _q62_sincos(theta)

    native_latitude = math.asin((theta + native_sin * native_cos + 2.0 * native_sin) / ECK4_C_P)
    fixed_latitude = math.asin((theta + fixed_sin * fixed_cos + 2.0 * fixed_sin) / ECK4_C_P)
    normalized_x = EDGE_LONGITUDE_RAD * ECK4_C_X * (1.0 + native_cos)
    fixed_longitude = normalized_x / (ECK4_C_X * (1.0 + fixed_cos))

    latitude_error_m = (fixed_latitude - native_latitude) * EARTH_RADIUS_M
    longitude_error_m = (
        (fixed_longitude - EDGE_LONGITUDE_RAD) * EARTH_RADIUS_M * math.cos(native_latitude)
    )
    horizontal_error_m = math.hypot(latitude_error_m, longitude_error_m)
    return {
        "cy_over_c_y": EDGE_CY_OVER_C_Y,
        "longitude_rad": EDGE_LONGITUDE_RAD,
        "latitude_error_m": latitude_error_m,
        "longitude_error_m": longitude_error_m,
        "horizontal_error_m": horizontal_error_m,
        "horizontal_error_nm": horizontal_error_m * 1e9,
        "gate_m": ERROR_GATE_M,
        "passes_gate": horizontal_error_m <= ERROR_GATE_M,
        "research_device_speedup": RESEARCH_DEVICE_SPEEDUP,
        "research_wall_speedup": RESEARCH_WALL_SPEEDUP,
    }


if __name__ == "__main__":
    print(json.dumps(reproduce_eck4_inverse_edge(), indent=2, sort_keys=True))
