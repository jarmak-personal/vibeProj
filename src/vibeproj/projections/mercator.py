"""Mercator and Web Mercator projections.

Standard Mercator (ellipsoidal): EPSG:3395 and similar.
Web Mercator (spherical pseudo-Mercator): EPSG:3857 — used by all web map tiles.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.exceptions import CRSResolutionError
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
        variant = _mercator_variant(params.operation_method)
        k0 = params.k_0
        if variant == "variant_b":
            if params.lat_1 is None:
                raise CRSResolutionError(
                    "Mercator variant B requires an explicit latitude of standard parallel"
                )
            latitude_standard_parallel = math.radians(params.lat_1)
            sin_latitude = math.sin(latitude_standard_parallel)
            k0 = math.cos(latitude_standard_parallel) / math.sqrt(
                1.0 - e.es * sin_latitude * sin_latitude
            )
        return {
            "a": e.a,
            "e": e.e,
            "es": e.es,
            "k0": k0,
            "mercator_variant": variant,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        e = computed["e"]
        # Clamp latitude to avoid singularity at poles (tan(π/2) = inf)
        if xp.any(xp.abs(phi) > _MAX_LAT_RAD):
            warnings.warn(
                "Latitude values clamped to ±89.999° to avoid Mercator pole singularity. "
                "Mercator is undefined at the poles.",
                RuntimeWarning,
                stacklevel=2,
            )
        phi = xp.clip(phi, -_MAX_LAT_RAD, _MAX_LAT_RAD)
        k0 = computed["k0"]
        if e == 0:
            # Spherical case
            x = k0 * lam
            y = k0 * xp.log(xp.tan(math.pi / 4.0 + phi * 0.5))
        else:
            # Ellipsoidal Mercator
            e_sin_phi = e * xp.sin(phi)
            x = k0 * lam
            y = k0 * xp.log(
                xp.tan(math.pi / 4.0 + phi * 0.5)
                * ((1.0 - e_sin_phi) / (1.0 + e_sin_phi)) ** (e / 2.0)
            )
        return x, y

    def inverse(self, x, y, params, computed, xp):
        e = computed["e"]
        k0 = computed["k0"]
        lam = x / k0
        y = y / k0
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


def _mercator_variant(operation_method: str | None) -> str:
    variants = {
        "Mercator (variant A)": "variant_a",
        "Mercator (1SP)": "variant_a",
        "Mercator (variant B)": "variant_b",
        "Mercator (2SP)": "variant_b",
    }
    return variants.get(operation_method, "custom")


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
        if xp.any(xp.abs(phi) > _MAX_LAT_RAD):
            warnings.warn(
                "Latitude values clamped to ±89.999° to avoid Mercator pole singularity. "
                "Mercator is undefined at the poles.",
                RuntimeWarning,
                stacklevel=2,
            )
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
