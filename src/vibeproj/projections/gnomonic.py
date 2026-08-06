"""Gnomonic projection.

All great circles map to straight lines. Used for navigation and seismology.
Only displays less than one hemisphere.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.exceptions import UnsupportedProjectionError
from vibeproj.projections import register
from vibeproj.projections.base import Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams


# PROJ rejects the horizon and the numerically ill-conditioned sliver directly
# inside it.  Keeping this as the single host/device contract constant avoids
# returning mirrored or enormous finite coordinates for the hidden hemisphere.
GNOM_MIN_COS_C = 1e-10


class Gnomonic(Projection):
    """Gnomonic projection — all great circles map to straight lines."""

    name = "gnom"

    def setup(self, params: ProjectionParams) -> dict:
        if params.ellipsoid.es != 0.0:
            raise UnsupportedProjectionError(
                "Ellipsoidal Gnomonic is not implemented; use an explicit spherical "
                "ellipsoid (+R) when spherical semantics are intended."
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
        finite_input = xp.isfinite(lam) & xp.isfinite(phi)
        work_lam = xp.where(finite_input, lam, 0.0)
        work_phi = xp.where(finite_input, phi, 0.0)
        sin_phi = xp.sin(work_phi)
        cos_phi = xp.cos(work_phi)
        cos_lam = xp.cos(work_lam)
        cos_c = sin_phi0 * sin_phi + cos_phi0 * cos_phi * cos_lam
        visible = finite_input & (cos_c >= GNOM_MIN_COS_C)
        safe_cos_c = xp.where(visible, cos_c, 1.0)
        x = cos_phi * xp.sin(work_lam) / safe_cos_c
        y = (cos_phi0 * sin_phi - sin_phi0 * cos_phi * cos_lam) / safe_cos_c
        nan_input = xp.isnan(lam) | xp.isnan(phi)
        invalid = xp.where(nan_input, xp.nan, xp.inf)
        x = xp.where(visible, x, invalid)
        y = xp.where(visible, y, invalid)
        return x, y

    def inverse(self, x, y, params, computed, xp):
        sin_phi0 = computed["sin_phi0"]
        cos_phi0 = computed["cos_phi0"]
        finite = xp.isfinite(x) & xp.isfinite(y)
        work_x = xp.where(finite, x, 0.0)
        work_y = xp.where(finite, y, 0.0)
        rho = xp.hypot(work_x, work_y)
        norm = xp.hypot(1.0, rho)
        sin_c = rho / norm
        cos_c = 1.0 / norm
        safe_rho = xp.where(rho == 0.0, 1.0, rho)
        unit_x = work_x / safe_rho
        unit_y = work_y / safe_rho
        phi_argument = cos_c * sin_phi0 + unit_y * sin_c * cos_phi0
        phi = xp.arcsin(xp.clip(phi_argument, -1.0, 1.0))
        lam = xp.arctan2(
            unit_x * sin_c,
            cos_phi0 * cos_c - unit_y * sin_phi0 * sin_c,
        )
        nan_input = xp.isnan(x) | xp.isnan(y)
        invalid = xp.where(nan_input, xp.nan, xp.inf)
        phi = xp.where(finite, phi, invalid)
        lam = xp.where(finite, lam, invalid)
        return lam, phi


register("gnom", Gnomonic())
