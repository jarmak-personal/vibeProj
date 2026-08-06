"""Albers Equal Area Conic projection.

Equal-area conic projection. The standard for CONUS mapping (EPSG:5070)
and Australian mapping (EPSG:3577).

Math from PROJ aea.c and Snyder, "Map Projections: A Working Manual" (USGS PP 1395).
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
from vibeproj.projections.base import EPS_ANGLE, Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams
_HALF_PI = math.pi / 2.0


class AlbersEqualArea(Projection):
    """Equal-area conic projection (EPSG method 9822)."""

    name = "aea"

    def setup(self, params: ProjectionParams) -> dict:
        e = params.ellipsoid
        ec = e.e
        es = e.es

        phi0 = math.radians(params.lat_0)
        phi1 = math.radians(params.lat_1) if params.lat_1 is not None else phi0
        phi2 = math.radians(params.lat_2) if params.lat_2 is not None else phi1

        sin_phi1 = math.sin(phi1)
        cos_phi1 = math.cos(phi1)
        m1 = cos_phi1 / math.sqrt(1.0 - es * sin_phi1 * sin_phi1)
        q1 = authalic_q_scalar(sin_phi1, ec)

        if abs(phi1 - phi2) < EPS_ANGLE:
            n = sin_phi1
        else:
            sin_phi2 = math.sin(phi2)
            cos_phi2 = math.cos(phi2)
            m2 = cos_phi2 / math.sqrt(1.0 - es * sin_phi2 * sin_phi2)
            q2 = authalic_q_scalar(sin_phi2, ec)
            n = (m1 * m1 - m2 * m2) / (q2 - q1)

        C = m1 * m1 + n * q1
        q0 = authalic_q_scalar(math.sin(phi0), ec)
        qp = authalic_q_scalar(1.0, ec)
        # Normalized (without a) — pipeline multiplies by a
        rho0 = math.sqrt(C - n * q0) / n

        return {
            "n": n,
            "C": C,
            "rho0": rho0,
            "qp": qp,
            "e": ec,
            "es": es,
            "a": e.a,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        n = computed["n"]
        C = computed["C"]
        rho0 = computed["rho0"]
        e = computed["e"]

        q = authalic_q(xp.sin(phi), e, xp)
        rho = xp.sqrt(xp.maximum(C - n * q, 0.0)) / n

        theta = n * lam
        x = rho * xp.sin(theta)
        y = rho0 - rho * xp.cos(theta)
        return x, y

    def inverse(self, x, y, params, computed, xp):
        n = computed["n"]
        C = computed["C"]
        rho0 = computed["rho0"]
        e = computed["e"]
        es = computed["es"]
        qp = computed["qp"]

        dy = rho0 - y
        rho = xp.sqrt(x * x + dy * dy)
        if n < 0:
            rho = -rho
            x = -x
            dy = -dy

        lam = xp.arctan2(x, dy) / n
        q = (C - (rho * n) ** 2) / n

        q = snap_authalic_q_to_pole(q, qp, xp)
        invalid_q = xp.isfinite(q) & (xp.abs(q) > qp)
        phi = geodetic_latitude_from_authalic_q(q, qp, e, es, xp)
        lam = xp.where(invalid_q, xp.inf, lam)

        return lam, phi


register("aea", AlbersEqualArea())
