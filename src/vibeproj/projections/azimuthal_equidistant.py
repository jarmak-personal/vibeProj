"""Azimuthal Equidistant projection.

Preserves distance from center point. Used in seismology, radio coverage,
telecommunications, and the UN emblem.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import EPS_ANGLE, EPS_DENOM, Projection
from vibeproj.exceptions import UnsupportedProjectionError

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams


class AzimuthalEquidistant(Projection):
    """Azimuthal projection preserving distances from the center point."""

    name = "aeqd"

    def setup(self, params: ProjectionParams) -> dict:
        if params.operation_method == "Guam Projection":
            raise UnsupportedProjectionError(
                "Guam Projection is not implemented; the available AEQD formulas are spherical only"
            )
        if params.operation_method == "Modified Azimuthal Equidistant":
            raise UnsupportedProjectionError(
                "Modified Azimuthal Equidistant is not implemented; "
                "the available AEQD formulas are spherical only"
            )
        if params.ellipsoid.es != 0.0:
            raise UnsupportedProjectionError(
                "Ellipsoidal Azimuthal Equidistant is not implemented; "
                "use an explicit spherical CRS (+R) for spherical AEQD semantics"
            )
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
        sin_lam = xp.sin(lam)

        cos_c = sin_phi0 * sin_phi + cos_phi0 * cos_phi * cos_lam
        cos_c = xp.clip(cos_c, -1.0, 1.0)
        c = xp.arccos(cos_c)
        center = xp.abs(c) < EPS_ANGLE
        safe_sin_c = xp.where(center, 1.0, xp.sin(c))
        k = xp.where(center, 1.0, c / safe_sin_c)

        x = k * cos_phi * sin_lam
        y = k * (cos_phi0 * sin_phi - sin_phi0 * cos_phi * cos_lam)
        return x, y

    def inverse(self, x, y, params, computed, xp):
        sin_phi0 = computed["sin_phi0"]
        cos_phi0 = computed["cos_phi0"]
        c = xp.sqrt(x * x + y * y)
        sin_c = xp.sin(c)
        cos_c = xp.cos(c)

        phi = xp.where(
            c < EPS_ANGLE,
            xp.arcsin(xp.clip(sin_phi0, -1.0, 1.0)) + 0 * x,
            xp.arcsin(
                xp.clip(
                    cos_c * sin_phi0 + y * sin_c * cos_phi0 / xp.maximum(c, EPS_DENOM), -1.0, 1.0
                )
            ),
        )
        lam = xp.arctan2(x * sin_c, c * cos_phi0 * cos_c - y * sin_phi0 * sin_c)
        return lam, phi


register("aeqd", AzimuthalEquidistant())
