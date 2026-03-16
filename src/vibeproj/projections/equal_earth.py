"""Equal Earth projection (2018).

Modern equal-area pseudocylindrical projection with pleasing aesthetics.
Polynomial formulas — closed-form forward and inverse.
EPSG: 8857, 8858, 8859.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams

# Polynomial coefficients (Šavrič, Patterson & Jenny 2018)
_A1 = 1.340264
_A2 = -0.081106
_A3 = 0.000893
_A4 = 0.003796


class EqualEarth(Projection):
    name = "eqearth"

    def setup(self, params: ProjectionParams) -> dict:
        return {
            "a": params.ellipsoid.a,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        theta = xp.arcsin(math.sqrt(3.0) / 2.0 * xp.sin(phi))
        t2 = theta * theta
        t6 = t2 * t2 * t2
        d = _A1 + 3 * _A2 * t2 + t6 * (7 * _A3 + 9 * _A4 * t2)
        x = lam * xp.cos(theta) / d
        y = theta * (_A1 + _A2 * t2 + t6 * (_A3 + _A4 * t2))
        return x, y

    def inverse(self, x, y, params, computed, xp):
        # Newton iteration to recover theta from y
        theta = y
        for _ in range(12):
            t2 = theta * theta
            t6 = t2 * t2 * t2
            fy = theta * (_A1 + _A2 * t2 + t6 * (_A3 + _A4 * t2)) - y
            fpy = _A1 + 3 * _A2 * t2 + t6 * (7 * _A3 + 9 * _A4 * t2)
            theta = theta - fy / fpy
        t2 = theta * theta
        t6 = t2 * t2 * t2
        d = _A1 + 3 * _A2 * t2 + t6 * (7 * _A3 + 9 * _A4 * t2)
        lam = x * d / xp.cos(theta)
        phi = xp.arcsin(xp.sin(theta) * 2.0 / math.sqrt(3.0))
        return lam, phi


register("eqearth", EqualEarth())
