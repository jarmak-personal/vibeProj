"""Sinusoidal (Sanson-Flamsteed) projection.

Equal-area pseudocylindrical projection. Used by NASA MODIS land products.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams


class Sinusoidal(Projection):
    name = "sinu"

    def setup(self, params: ProjectionParams) -> dict:
        return {
            "a": params.ellipsoid.a,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        x = lam * xp.cos(phi)
        y = phi
        return x, y

    def inverse(self, x, y, params, computed, xp):
        phi = y
        lam = x / xp.cos(phi)
        return lam, phi


register("sinu", Sinusoidal())
