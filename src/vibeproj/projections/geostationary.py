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
            "h": h,
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
        h = computed["h"]
        r_eq2 = computed["r_eq2"]
        r_pol2 = computed["r_pol2"]
        a = computed["a"]

        # Geographic to geocentric latitude
        phi_gc = xp.arctan(r_pol2 / r_eq2 * xp.tan(phi))
        cos_phi_gc = xp.cos(phi_gc)
        sin_phi_gc = xp.sin(phi_gc)

        # Geocentric earth radius (CGMS standard)
        r_pol = math.sqrt(r_pol2)
        r_earth = r_pol / xp.sqrt(1.0 - (r_eq2 - r_pol2) / r_eq2 * cos_phi_gc * cos_phi_gc)

        cos_lam = xp.cos(lam)

        Sx = H - r_earth * cos_phi_gc * cos_lam
        Sy = -r_earth * cos_phi_gc * xp.sin(lam)
        Sz = r_earth * sin_phi_gc

        # Sweep Y (GOES-R PUG): x = atan2(-Sy, Sx), y = asin(Sz/|S|)
        sn = xp.sqrt(Sx * Sx + Sy * Sy + Sz * Sz)
        x = xp.arctan2(-Sy, Sx) * (h / a)
        y = xp.arcsin(xp.clip(Sz / sn, -1.0, 1.0)) * (h / a)
        return x, y

    def inverse(self, x, y, params, computed, xp):
        H = computed["H"]
        h = computed["h"]
        r_eq2 = computed["r_eq2"]
        r_pol2 = computed["r_pol2"]
        a = computed["a"]

        # Recover scanning angles (pipeline passes x_norm = x_physical / a)
        x_angle = x * a / h
        y_angle = y * a / h

        sin_x = xp.sin(x_angle)
        cos_x = xp.cos(x_angle)
        sin_y = xp.sin(y_angle)
        cos_y = xp.cos(y_angle)

        # Ray-ellipsoid intersection (sweep Y geometry)
        a_coeff = cos_y * cos_y + sin_y * sin_y * r_eq2 / r_pol2
        b_coeff = -2 * H * cos_y * cos_x
        c_coeff = H * H - a * a

        discrim = b_coeff * b_coeff - 4 * a_coeff * c_coeff
        discrim = xp.maximum(discrim, 0.0)

        r_s = (-b_coeff - xp.sqrt(discrim)) / (2 * a_coeff)

        # Reconstruct ground point from scanning angles (sweep Y)
        P_x = H - r_s * cos_y * cos_x
        P_y = r_s * cos_y * sin_x
        P_z = r_s * sin_y

        lam = xp.arctan2(P_y, P_x)
        # Geocentric → geodetic latitude (CGMS standard: r_eq²/r_pol² factor)
        phi = xp.arctan(P_z * r_eq2 / (xp.sqrt(P_x**2 + P_y**2) * r_pol2))

        return lam, phi


register("geos", Geostationary())
