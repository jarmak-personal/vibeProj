"""Geostationary Satellite View projection.

Perspective projection from geostationary orbit. Critical for GOES, Meteosat,
Himawari satellite imagery. Sweep axis Y variant (GOES standard).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams


class Geostationary(Projection):
    name = "geos"

    def setup(self, params: ProjectionParams) -> dict:
        e = params.ellipsoid
        h = params.extra.get("h", 35785831.0)  # satellite height in meters
        # Radius ratio
        r_eq = e.a
        r_pol = e.b
        inv_r_eq2 = 1.0 / (r_eq * r_eq)
        inv_r_pol2 = 1.0 / (r_pol * r_pol)
        H = h + r_eq  # distance from Earth center to satellite
        return {
            "a": e.a,
            "H": H,
            "r_eq2": r_eq * r_eq,
            "r_pol2": r_pol * r_pol,
            "inv_r_eq2": inv_r_eq2,
            "inv_r_pol2": inv_r_pol2,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        H = computed["H"]
        r_eq2 = computed["r_eq2"]
        r_pol2 = computed["r_pol2"]
        a = computed["a"]

        # Geographic to geocentric latitude
        phi_gc = xp.arctan(r_pol2 / r_eq2 * xp.tan(phi))
        cos_phi_gc = xp.cos(phi_gc)

        # Distance from Earth center to surface point
        r_earth = a / xp.sqrt(1.0 + (r_eq2 - r_pol2) / r_pol2 * cos_phi_gc * cos_phi_gc)
        # Actually use simplified approach
        r_earth = a  # For the normalized projection

        cos_phi = xp.cos(phi_gc)
        sin_phi = xp.sin(phi_gc)
        cos_lam = xp.cos(lam)

        Sx = H - r_earth * cos_phi * cos_lam
        Sy = -r_earth * cos_phi * xp.sin(lam)
        Sz = r_earth * sin_phi

        # Sweep Y (GOES standard)
        x = xp.arctan(Sy / Sx) / a  # normalized
        y = xp.arctan(Sz / xp.sqrt(Sx * Sx + Sy * Sy)) / a  # normalized
        return x, y

    def inverse(self, x, y, params, computed, xp):
        H = computed["H"]
        a = computed["a"]

        x_proj = x * a
        y_proj = y * a

        sin_x = xp.sin(x_proj)
        cos_x = xp.cos(x_proj)
        sin_y = xp.sin(y_proj)
        cos_y = xp.cos(y_proj)

        # Back-project
        a_coeff = sin_x * sin_x + cos_x * cos_x * (
            cos_y * cos_y + sin_y * sin_y * (a * a) / (computed["r_pol2"])
        )
        b_coeff = -2 * H * cos_x * cos_y
        c_coeff = H * H - a * a

        discrim = b_coeff * b_coeff - 4 * a_coeff * c_coeff
        discrim = xp.maximum(discrim, 0.0)

        r_s = (-b_coeff - xp.sqrt(discrim)) / (2 * a_coeff)

        Sx = r_s * cos_x * cos_y
        Sy = -r_s * sin_x
        Sz = r_s * cos_x * sin_y

        lam = xp.arctan2(Sy, H - Sx)
        phi = xp.arctan(Sz * H / (xp.sqrt((H - Sx) ** 2 + Sy**2) * computed["r_pol2"] / (a * a)))

        return lam, phi


register("geos", Geostationary())
