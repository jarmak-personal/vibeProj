"""Cylindrical Equal Area projection.

Equal-area cylindrical projection. Includes Lambert, Behrmann, Gall-Peters variants.
EPSG: 6933 (EASE-Grid 2.0), 3410 (EASE-Grid).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections._equal_area import (
    authalic_q,
    authalic_q_scalar,
    geodetic_latitude_from_authalic_q,
    snap_authalic_q_to_pole,
)
from vibeproj.projections.base import Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams


class CylindricalEqualArea(Projection):
    """Equal-area cylindrical projection (Lambert, Behrmann, Gall-Peters variants)."""

    name = "cea"

    def setup(self, params: ProjectionParams) -> dict:
        lat_ts = math.radians(params.lat_1) if params.lat_1 is not None else 0.0
        e = params.ellipsoid
        k0 = math.cos(lat_ts) / math.sqrt(1.0 - e.es * math.sin(lat_ts) ** 2)
        qp = authalic_q_scalar(1.0, e.e)
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
        y = 0.5 * authalic_q(xp.sin(phi), e, xp) / k0
        return x, y

    def inverse(self, x, y, params, computed, xp):
        k0 = computed["k0"]
        e = computed["e"]
        es = computed["es"]
        qp = computed["qp"]
        lam = x / k0
        q = 2.0 * y * k0
        q = snap_authalic_q_to_pole(q, qp, xp)
        invalid_q = xp.isfinite(q) & (xp.abs(q) > qp)
        phi = geodetic_latitude_from_authalic_q(q, qp, e, es, xp)
        lam = xp.where(invalid_q, xp.inf, lam)
        return lam, phi


register("cea", CylindricalEqualArea())
