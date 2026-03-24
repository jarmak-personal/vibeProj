"""Eckert VI projection.

Equal-area pseudocylindrical projection for world thematic maps.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import EPS_CONV, EPS_DENOM, Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams

_C_p = 1.0 + math.pi / 2.0  # 2.5707963...
_SQRT_2_PLUS_PI = math.sqrt(2.0 + math.pi)  # denominator for x and y
_C_x = 1.0 / _SQRT_2_PLUS_PI
_C_y = 2.0 / _SQRT_2_PLUS_PI


class EckertVI(Projection):
    name = "eck6"

    def setup(self, params: ProjectionParams) -> dict:
        return {
            "a": params.ellipsoid.a,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        # Newton iteration: solve θ + sin(θ) = C_p·sin(φ)
        p = _C_p * xp.sin(phi)
        theta = phi  # initial guess
        for _ in range(20):
            V = theta + xp.sin(theta) - p
            denom = 1.0 + xp.cos(theta)
            denom = xp.where(xp.abs(denom) < EPS_DENOM, EPS_DENOM, denom)
            dtheta = -V / denom
            theta = theta + dtheta
            if hasattr(dtheta, "__len__"):
                if xp.all(xp.abs(dtheta) < EPS_CONV):
                    break
            elif abs(float(dtheta)) < EPS_CONV:
                break
        x = _C_x * lam * (1.0 + xp.cos(theta))
        y = _C_y * theta
        return x, y

    def inverse(self, x, y, params, computed, xp):
        theta = y / _C_y
        sin_t = xp.sin(theta)
        phi = xp.arcsin(xp.clip((theta + sin_t) / _C_p, -1.0, 1.0))
        denom = _C_x * (1.0 + xp.cos(theta))
        denom = xp.where(xp.abs(denom) < EPS_DENOM, EPS_DENOM, denom)
        lam = x / denom
        return lam, phi


register("eck6", EckertVI())
