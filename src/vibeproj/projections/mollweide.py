"""Mollweide projection.

Equal-area pseudocylindrical projection for world thematic maps.
Used in climate science and as component of Goode Homolosine.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import EPS_CONV, EPS_DENOM, Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams

_SQRT2 = math.sqrt(2.0)


class Mollweide(Projection):
    name = "moll"

    def setup(self, params: ProjectionParams) -> dict:
        return {
            "a": params.ellipsoid.a,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        # Newton iteration: solve 2θ + sin(2θ) = π·sin(φ)
        theta = phi  # initial guess
        pi_sin_phi = math.pi * xp.sin(phi)
        for _ in range(20):
            denom = 2.0 + 2.0 * xp.cos(2.0 * theta)
            # Guard against zero denominator when theta = ±π/2
            denom = xp.where(xp.abs(denom) < EPS_DENOM, EPS_DENOM, denom)
            dtheta = -(2.0 * theta + xp.sin(2.0 * theta) - pi_sin_phi) / denom
            theta = theta + dtheta
            if hasattr(dtheta, "__len__"):
                if xp.all(xp.abs(dtheta) < EPS_CONV):
                    break
            elif abs(float(dtheta)) < EPS_CONV:
                break
        x = lam * 2.0 * _SQRT2 / math.pi * xp.cos(theta)
        y = _SQRT2 * xp.sin(theta)
        return x, y

    def inverse(self, x, y, params, computed, xp):
        theta = xp.arcsin(xp.clip(y / _SQRT2, -1.0, 1.0))
        phi = xp.arcsin(xp.clip((2.0 * theta + xp.sin(2.0 * theta)) / math.pi, -1.0, 1.0))
        cos_theta = xp.cos(theta)
        # Guard against division by zero at poles (cos(±π/2) = 0)
        cos_theta = xp.where(xp.abs(cos_theta) < EPS_DENOM, EPS_DENOM, cos_theta)
        lam = x * math.pi / (2.0 * _SQRT2 * cos_theta)
        return lam, phi


register("moll", Mollweide())
