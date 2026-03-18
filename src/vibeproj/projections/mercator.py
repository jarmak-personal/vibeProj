"""Mercator and Web Mercator projections.

Standard Mercator (ellipsoidal): EPSG:3395 and similar.
Web Mercator (spherical pseudo-Mercator): EPSG:3857 — used by all web map tiles.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams

# ~89.999° in radians — avoids tan(π/2) singularity at poles while preserving
# sub-meter accuracy at extreme latitudes.
_MAX_LAT_RAD = math.radians(89.999)


class Mercator(Projection):
    """Ellipsoidal Mercator projection (variant A / 1SP)."""

    name = "merc"

    def setup(self, params: ProjectionParams) -> dict:
        e = params.ellipsoid
        return {
            "a": e.a,
            "e": e.e,
            "es": e.es,
            "k0": params.k_0,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        e = computed["e"]
        # Clamp latitude to avoid singularity at poles (tan(π/2) = inf)
        phi = xp.clip(phi, -_MAX_LAT_RAD, _MAX_LAT_RAD)
        if e == 0:
            # Spherical case
            x = lam
            y = xp.log(xp.tan(math.pi / 4.0 + phi * 0.5))
        else:
            # Ellipsoidal Mercator
            e_sin_phi = e * xp.sin(phi)
            x = lam
            y = xp.log(
                xp.tan(math.pi / 4.0 + phi * 0.5)
                * ((1.0 - e_sin_phi) / (1.0 + e_sin_phi)) ** (e / 2.0)
            )
        return x, y

    def inverse(self, x, y, params, computed, xp):
        e = computed["e"]
        lam = x
        if e == 0:
            phi = 2.0 * xp.arctan(xp.exp(y)) - math.pi / 2.0
        else:
            # Iterative inverse for ellipsoidal Mercator
            phi = 2.0 * xp.arctan(xp.exp(y)) - math.pi / 2.0
            for _ in range(7):
                e_sin_phi = e * xp.sin(phi)
                phi = (
                    2.0
                    * xp.arctan(xp.exp(y) * ((1.0 + e_sin_phi) / (1.0 - e_sin_phi)) ** (e / 2.0))
                    - math.pi / 2.0
                )
        return lam, phi


class WebMercator(Projection):
    """Spherical Pseudo-Mercator (EPSG:3857).

    Uses spherical formulas with WGS84 semi-major axis.
    This is the projection used by virtually all web mapping platforms.
    """

    name = "webmerc"

    def setup(self, params: ProjectionParams) -> dict:
        return {
            "a": params.ellipsoid.a,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        # Clamp latitude to avoid singularity at poles (tan(π/2) = inf)
        phi = xp.clip(phi, -_MAX_LAT_RAD, _MAX_LAT_RAD)
        x = lam
        y = xp.log(xp.tan(math.pi / 4.0 + phi * 0.5))
        return x, y

    def inverse(self, x, y, params, computed, xp):
        lam = x
        phi = 2.0 * xp.arctan(xp.exp(y)) - math.pi / 2.0
        return lam, phi


register("merc", Mercator())
register("webmerc", WebMercator())
