"""Mollweide projection.

Equal-area pseudocylindrical projection for world thematic maps.
Used in climate science and as component of Goode Homolosine.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import Projection

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
            dtheta = -(2.0 * theta + xp.sin(2.0 * theta) - pi_sin_phi) / (
                2.0 + 2.0 * xp.cos(2.0 * theta)
            )
            theta = theta + dtheta
            if hasattr(dtheta, "__len__"):
                if xp.all(xp.abs(dtheta) < 1e-14):
                    break
            elif abs(float(dtheta)) < 1e-14:
                break
        x = lam * 2.0 * _SQRT2 / math.pi * xp.cos(theta)
        y = _SQRT2 * xp.sin(theta)
        return x, y

    def inverse(self, x, y, params, computed, xp):
        theta = xp.arcsin(xp.clip(y / _SQRT2, -1.0, 1.0))
        phi = xp.arcsin(xp.clip((2.0 * theta + xp.sin(2.0 * theta)) / math.pi, -1.0, 1.0))
        lam = x * math.pi / (2.0 * _SQRT2 * xp.cos(theta))
        return lam, phi


register("moll", Mollweide())
