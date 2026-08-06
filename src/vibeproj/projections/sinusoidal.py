"""Sinusoidal (Sanson-Flamsteed) projection.

Equal-area pseudocylindrical projection. Used by NASA MODIS land products.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import EPS_DENOM, Projection
from vibeproj.exceptions import UnsupportedProjectionError

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams


class Sinusoidal(Projection):
    """Equal-area pseudocylindrical projection with sinusoidal meridians."""

    name = "sinu"

    def setup(self, params: ProjectionParams) -> dict:
        semi_major_axis = params.ellipsoid.a
        if (
            not math.isfinite(semi_major_axis)
            or not 0.0 < semi_major_axis <= SINU_MAX_SEMI_MAJOR_AXIS_M
        ):
            raise UnsupportedProjectionError(
                "Sinusoidal supports finite positive semi-major axes no larger than "
                f"{SINU_MAX_SEMI_MAJOR_AXIS_M} m; got {semi_major_axis!r}."
            )
        es = params.ellipsoid.es
        if not math.isfinite(es) or not 0.0 <= es <= SINU_MAX_ECCENTRICITY_SQUARED:
            raise UnsupportedProjectionError(
                "Sinusoidal supports spherical and terrestrial ellipsoids with finite "
                f"0 <= eccentricity squared <= {SINU_MAX_ECCENTRICITY_SQUARED}; "
                f"got {es!r}."
            )
        meridional_coefficients = _meridional_coefficients(es)
        return {
            "a": semi_major_axis,
            "es": es,
            "meridional_coefficients": meridional_coefficients,
            "meridional_pole": meridional_coefficients[0] * math.pi / 2.0,
            "meridional_recurrence_hot_cy_limit": _meridional_arc(
                _SINU_RECURRENCE_HOT_PHI, meridional_coefficients, math
            ),
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        es = computed["es"]
        if es == 0.0:
            return lam * xp.cos(phi), phi

        sin_phi = xp.sin(phi)
        x = lam * xp.cos(phi) / xp.sqrt(1.0 - es * sin_phi * sin_phi)
        y = _meridional_arc(phi, computed["meridional_coefficients"], xp)
        return x, y

    def inverse(self, x, y, params, computed, xp):
        es = computed["es"]
        if es != 0.0:
            pole = computed["meridional_pole"]
            invalid = xp.abs(y) > pole
            target = xp.where(invalid, 0.0, y)
            phi = xp.clip(
                target / computed["meridional_coefficients"][0], -math.pi / 2, math.pi / 2
            )
            for _ in range(10):
                sin_phi = xp.sin(phi)
                one_minus = 1.0 - es * sin_phi * sin_phi
                derivative = (1.0 - es) / (one_minus * xp.sqrt(one_minus))
                phi -= (
                    _meridional_arc(phi, computed["meridional_coefficients"], xp) - target
                ) / derivative

            sin_phi = xp.sin(phi)
            denominator = xp.cos(phi) / xp.sqrt(1.0 - es * sin_phi * sin_phi)
            lam = x / denominator
            return xp.where(invalid, xp.inf, lam), xp.where(invalid, xp.inf, phi)

        phi = y
        cos_phi = xp.cos(phi)
        # Guard against division by zero at poles (cos(±π/2) = 0)
        cos_phi = xp.where(xp.abs(cos_phi) < EPS_DENOM, EPS_DENOM, cos_phi)
        lam = x / cos_phi
        return lam, phi


# The seventh-order series is qualified against PROJ through this inclusive
# bound at semi-major axes up to 6,400,000 m. It includes terrestrial Earth
# and Mars ellipsoids while excluding highly eccentric custom bodies for which
# the truncation error is material.
SINU_MAX_SEMI_MAJOR_AXIS_M = 6_400_000.0
SINU_MAX_ECCENTRICITY_SQUARED = 0.012
_MERIDIONAL_SERIES_ORDER = 7
_SINU_RECURRENCE_HOT_PHI = math.radians(89.9)


def _sine_power_fourier_coefficient(power: int, harmonic: int) -> float:
    """Return the cos(2 * harmonic * phi) coefficient of sin(phi) ** (2 * power)."""
    if harmonic > power:
        return 0.0
    denominator = float(4**power)
    if harmonic == 0:
        return math.comb(2 * power, power) / denominator
    return 2.0 * (-1.0) ** harmonic * math.comb(2 * power, power - harmonic) / denominator


def _meridional_coefficients(es: float) -> tuple[float, ...]:
    """Build the seventh-order normalized meridional-distance Fourier series.

    Expanding ``(1-es) / (1-es*sin(phi)**2)**(3/2)`` and integrating its
    Fourier series gives nanometre-level agreement with PROJ for terrestrial
    ellipsoids.  Keeping this derivation in one helper avoids projection-local
    magic constants and makes the coefficients reusable by future families.
    """
    binomial_series = [1.0]
    for order in range(1, _MERIDIONAL_SERIES_ORDER + 1):
        binomial_series.append(binomial_series[-1] * (2 * order + 1) / (2 * order))

    coefficients: list[float] = []
    for harmonic in range(_MERIDIONAL_SERIES_ORDER + 1):
        coefficient = 1.0 if harmonic == 0 else 0.0
        es_power = 1.0
        for order in range(1, _MERIDIONAL_SERIES_ORDER + 1):
            es_power *= es
            term = binomial_series[order] * _sine_power_fourier_coefficient(
                order, harmonic
            ) - binomial_series[order - 1] * _sine_power_fourier_coefficient(order - 1, harmonic)
            if harmonic:
                term /= 2 * harmonic
            coefficient += term * es_power
        coefficients.append(coefficient)
    return tuple(coefficients)


def _meridional_arc(phi, coefficients: tuple[float, ...], xp):
    """Return meridional distance divided by the ellipsoid semi-major axis."""
    result = coefficients[0] * phi
    for harmonic, coefficient in enumerate(coefficients[1:], start=1):
        result += coefficient * xp.sin(2.0 * harmonic * phi)
    return result


register("sinu", Sinusoidal())
