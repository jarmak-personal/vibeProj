"""Shared CPU/xp authalic-latitude helpers for equal-area projections."""

from __future__ import annotations

import math

from vibeproj.projections.base import EPS_ANGLE, EPS_CONV

# Public-coordinate unit conversions can reconstruct an exact pole a few ulps
# beyond +/-qp. AEA and CEA share this named representational snap band before
# applying their strict material off-domain rejection.
AUTHALIC_POLE_TOLERANCE = EPS_ANGLE


def authalic_q_scalar(sin_phi: float, eccentricity: float) -> float:
    """Return Snyder's authalic q for a scalar sine of geodetic latitude."""
    if eccentricity == 0.0:
        return 2.0 * sin_phi
    e_sin = eccentricity * sin_phi
    return (1.0 - eccentricity * eccentricity) * (
        sin_phi / (1.0 - e_sin * e_sin) + math.atanh(e_sin) / eccentricity
    )


def authalic_q(sin_phi, eccentricity: float, xp):
    """Return Snyder's authalic q for an array namespace."""
    if eccentricity == 0.0:
        return 2.0 * sin_phi
    e_sin = eccentricity * sin_phi
    return (1.0 - eccentricity * eccentricity) * (
        sin_phi / (1.0 - e_sin * e_sin) + xp.arctanh(e_sin) / eccentricity
    )


def snap_authalic_q_to_pole(q, qp: float, xp):
    """Snap only the shared representational pole band to exact +/-qp."""
    near_pole = xp.abs(xp.abs(q) - qp) <= AUTHALIC_POLE_TOLERANCE
    return xp.where(near_pole, xp.copysign(qp, q), q)


def geodetic_latitude_from_authalic_q(q, qp: float, eccentricity: float, es: float, xp):
    """Invert authalic q without device-to-host convergence reads."""
    numpy_host = getattr(xp, "__name__", None) == "numpy"
    finite = xp.isfinite(q)
    outside = finite & (xp.abs(q) > qp)
    bounded_q = xp.clip(xp.where(finite, q, 0.0), -qp, qp)
    if eccentricity == 0.0:
        phi = xp.arcsin(bounded_q / 2.0)
        return xp.where(finite, xp.where(outside, xp.inf, phi), q)

    sin_phi = xp.clip(bounded_q / qp, -1.0, 1.0)
    for _ in range(15):
        e_sin = eccentricity * sin_phi
        one_minus = 1.0 - e_sin * e_sin
        actual_q = (1.0 - es) * (sin_phi / one_minus + xp.arctanh(e_sin) / eccentricity)
        derivative = 2.0 * (1.0 - es) / (one_minus * one_minus)
        delta = (actual_q - bounded_q) / derivative
        sin_phi = xp.clip(sin_phi - delta, -1.0, 1.0)
        if numpy_host and bool(xp.all(xp.abs(delta) < EPS_CONV)):
            break
    phi = xp.arcsin(sin_phi)
    return xp.where(finite, xp.where(outside, xp.inf, phi), q)


__all__ = [
    "AUTHALIC_POLE_TOLERANCE",
    "authalic_q",
    "authalic_q_scalar",
    "geodetic_latitude_from_authalic_q",
    "snap_authalic_q_to_pole",
]
