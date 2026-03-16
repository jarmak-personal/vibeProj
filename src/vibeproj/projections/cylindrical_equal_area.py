"""Cylindrical Equal Area projection.

Equal-area cylindrical projection. Includes Lambert, Behrmann, Gall-Peters variants.
EPSG: 6933 (EASE-Grid 2.0), 3410 (EASE-Grid).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams


class CylindricalEqualArea(Projection):
    name = "cea"

    def setup(self, params: ProjectionParams) -> dict:
        lat_ts = math.radians(params.lat_1) if params.lat_1 != 0 else 0.0
        e = params.ellipsoid
        k0 = math.cos(lat_ts) / math.sqrt(1.0 - e.es * math.sin(lat_ts) ** 2)
        # qp = q at the pole
        if e.e > 1e-10:
            qp = (1 - e.es) * (
                1.0 / (1.0 - e.es) - (1.0 / (2.0 * e.e)) * math.log((1.0 - e.e) / (1.0 + e.e))
            )
        else:
            qp = 2.0
        return {
            "a": e.a,
            "e": e.e,
            "es": e.es,
            "k0": k0,
            "qp": qp,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        k0 = computed["k0"]
        e = computed["e"]
        x = lam * k0
        if e < 1e-10:
            y = xp.sin(phi) / k0
        else:
            sin_phi = xp.sin(phi)
            e_sin = e * sin_phi
            q = (1 - e * e) * (
                sin_phi / (1 - e_sin * e_sin) - (0.5 / e) * xp.log((1 - e_sin) / (1 + e_sin))
            )
            y = 0.5 * q / k0
        return x, y

    def inverse(self, x, y, params, computed, xp):
        k0 = computed["k0"]
        e = computed["e"]
        es = computed["es"]
        lam = x / k0
        if e < 1e-10:
            phi = xp.arcsin(xp.clip(y * k0, -1.0, 1.0))
        else:
            q = 2.0 * y * k0
            phi = xp.arcsin(xp.clip(q / 2.0, -1.0, 1.0))
            for _ in range(15):
                sin_phi = xp.sin(phi)
                e_sin = e * sin_phi
                one_minus = 1.0 - e_sin * e_sin
                dphi = (one_minus * one_minus / (2.0 * xp.cos(phi))) * (
                    q / (1.0 - es)
                    - sin_phi / one_minus
                    + (0.5 / e) * xp.log((1.0 - e_sin) / (1.0 + e_sin))
                )
                phi = phi + dphi
                if hasattr(dphi, "__len__"):
                    if xp.all(xp.abs(dphi) < 1e-14):
                        break
                elif abs(float(dphi)) < 1e-14:
                    break
        return lam, phi


register("cea", CylindricalEqualArea())
