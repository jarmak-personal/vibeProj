"""Gnomonic projection.

All great circles map to straight lines. Used for navigation and seismology.
Only displays less than one hemisphere.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import EPS_DENOM, Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams


class Gnomonic(Projection):
    name = "gnom"

    def setup(self, params: ProjectionParams) -> dict:
        phi0 = math.radians(params.lat_0)
        return {
            "a": params.ellipsoid.a,
            "sin_phi0": math.sin(phi0),
            "cos_phi0": math.cos(phi0),
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        sin_phi0 = computed["sin_phi0"]
        cos_phi0 = computed["cos_phi0"]
        sin_phi = xp.sin(phi)
        cos_phi = xp.cos(phi)
        cos_lam = xp.cos(lam)
        cos_c = sin_phi0 * sin_phi + cos_phi0 * cos_phi * cos_lam
        # Guard against division by zero at hemisphere boundary (cos_c = 0)
        cos_c = xp.where(xp.abs(cos_c) < EPS_DENOM, EPS_DENOM, cos_c)
        x = cos_phi * xp.sin(lam) / cos_c
        y = (cos_phi0 * sin_phi - sin_phi0 * cos_phi * cos_lam) / cos_c
        return x, y

    def inverse(self, x, y, params, computed, xp):
        sin_phi0 = computed["sin_phi0"]
        cos_phi0 = computed["cos_phi0"]
        rho = xp.sqrt(x * x + y * y)
        c = xp.arctan(rho)
        sin_c = xp.sin(c)
        cos_c = xp.cos(c)
        phi = xp.arcsin(cos_c * sin_phi0 + y * sin_c * cos_phi0 / xp.maximum(rho, EPS_DENOM))
        lam = xp.arctan2(x * sin_c, rho * cos_phi0 * cos_c - y * sin_phi0 * sin_c)
        return lam, phi


register("gnom", Gnomonic())
