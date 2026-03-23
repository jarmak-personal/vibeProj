"""Polar Stereographic projection.

Conformal azimuthal projection for polar regions.
Used by UPS (EPSG:32661/32761) and polar scientific grids (EPSG:3031, 3413).

Math from PROJ stere.c and Snyder, "Map Projections: A Working Manual".
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from vibeproj.projections import register
from vibeproj.projections.base import Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams

_EPS10 = 1e-10
_HALF_PI = math.pi / 2.0


class PolarStereographic(Projection):
    """Polar Stereographic (variants A, B, C)."""

    name = "stere"

    def setup(self, params: ProjectionParams) -> dict:
        e = params.ellipsoid
        ec = e.e
        es = e.es

        lat_0 = math.radians(params.lat_0)
        k0 = params.k_0

        # Determine if north or south polar
        is_south = lat_0 < 0
        abs_lat_0 = abs(lat_0)

        if abs(abs_lat_0 - _HALF_PI) < _EPS10:
            # Polar case (latitude of origin is at the pole)
            sign = -1.0 if is_south else 1.0
            # Normalized (without a) — pipeline multiplies by a
            akm1 = k0 * 2.0 / math.sqrt((1 + ec) ** (1 + ec) * (1 - ec) ** (1 - ec))
        else:
            sin_lat = math.sin(lat_0)
            cos_lat = math.cos(lat_0)
            e_sin = ec * abs(sin_lat)
            t = math.tan(0.5 * (_HALF_PI - abs_lat_0)) / ((1 - e_sin) / (1 + e_sin)) ** (0.5 * ec)
            m = cos_lat / math.sqrt(1.0 - es * sin_lat * sin_lat)
            # Normalized (without a) — pipeline multiplies by a
            akm1 = m / t
            sign = -1.0 if is_south else 1.0

        return {
            "akm1": akm1,
            "sign": sign,
            "is_south": is_south,
            "e": ec,
            "es": es,
            "a": e.a,
            "k0": k0,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        akm1 = computed["akm1"]
        sign = computed["sign"]
        e = computed["e"]

        phi_adj = sign * phi
        sin_phi = xp.sin(phi_adj)
        e_sin = e * sin_phi

        t = xp.tan(0.5 * (_HALF_PI - phi_adj)) / ((1.0 - e_sin) / (1.0 + e_sin)) ** (0.5 * e)
        rho = akm1 * t

        lam_adj = sign * lam
        x = rho * xp.sin(lam_adj)
        y = -sign * rho * xp.cos(lam_adj)
        return x, y

    def inverse(self, x, y, params, computed, xp):
        akm1 = computed["akm1"]
        sign = computed["sign"]
        e = computed["e"]

        x_adj = x
        y_adj = -sign * y

        rho = xp.sqrt(x_adj * x_adj + y_adj * y_adj)

        ts = rho / akm1
        half_e = 0.5 * e

        # Iterative phi from ts (same as LCC inverse)
        phi = _HALF_PI - 2.0 * xp.arctan(ts)
        for _ in range(15):
            e_sin = e * xp.sin(phi)
            dphi = _HALF_PI - 2.0 * xp.arctan(ts * ((1.0 - e_sin) / (1.0 + e_sin)) ** half_e) - phi
            phi = phi + dphi
            if xp is np:
                if hasattr(dphi, "__len__"):
                    if xp.all(xp.abs(dphi) < 1e-14):
                        break
                elif abs(float(dphi)) < 1e-14:
                    break

        lam = xp.arctan2(x_adj, y_adj)
        phi = sign * phi
        lam = sign * lam
        return lam, phi


register("stere", PolarStereographic())
