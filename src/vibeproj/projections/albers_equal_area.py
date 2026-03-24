"""Albers Equal Area Conic projection.

Equal-area conic projection. The standard for CONUS mapping (EPSG:5070)
and Australian mapping (EPSG:3577).

Math from PROJ aea.c and Snyder, "Map Projections: A Working Manual" (USGS PP 1395).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import EPS_ANGLE, EPS_CONV, EPS_DENOM, Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams
_HALF_PI = math.pi / 2.0


def _qsfn(sin_phi, e):
    """Compute q-function for Albers (scalar)."""
    if e < EPS_ANGLE:
        return 2.0 * sin_phi
    e_sin = e * sin_phi
    return (1.0 - e * e) * (
        sin_phi / (1.0 - e_sin * e_sin)
        - (1.0 / (2.0 * e)) * math.log((1.0 - e_sin) / (1.0 + e_sin))
    )


def _qsfn_array(sin_phi, e, xp):
    """Vectorized q-function."""
    if e < EPS_ANGLE:
        return 2.0 * sin_phi
    e_sin = e * sin_phi
    return (1.0 - e * e) * (
        sin_phi / (1.0 - e_sin * e_sin) - (1.0 / (2.0 * e)) * xp.log((1.0 - e_sin) / (1.0 + e_sin))
    )


class AlbersEqualArea(Projection):
    name = "aea"

    def setup(self, params: ProjectionParams) -> dict:
        e = params.ellipsoid
        ec = e.e
        es = e.es

        phi0 = math.radians(params.lat_0)
        phi1 = math.radians(params.lat_1) if params.lat_1 != 0 else phi0
        phi2 = math.radians(params.lat_2) if params.lat_2 != 0 else phi1

        sin_phi1 = math.sin(phi1)
        cos_phi1 = math.cos(phi1)
        m1 = cos_phi1 / math.sqrt(1.0 - es * sin_phi1 * sin_phi1)
        q1 = _qsfn(sin_phi1, ec)

        if abs(phi1 - phi2) < EPS_ANGLE:
            n = sin_phi1
        else:
            sin_phi2 = math.sin(phi2)
            cos_phi2 = math.cos(phi2)
            m2 = cos_phi2 / math.sqrt(1.0 - es * sin_phi2 * sin_phi2)
            q2 = _qsfn(sin_phi2, ec)
            n = (m1 * m1 - m2 * m2) / (q2 - q1)

        C = m1 * m1 + n * q1
        q0 = _qsfn(math.sin(phi0), ec)
        # Normalized (without a) — pipeline multiplies by a
        rho0 = math.sqrt(C - n * q0) / n

        return {
            "n": n,
            "C": C,
            "rho0": rho0,
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

        q = _qsfn_array(xp.sin(phi), e, xp)
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

        rho = xp.sqrt(x * x + (rho0 - y) ** 2)
        if n < 0:
            rho = -rho

        theta = xp.arctan2(x, rho0 - y)
        if n < 0:
            theta = -theta

        lam = theta / n
        q = (C - (rho * n) ** 2) / n

        # Iterative inverse for phi from q
        phi = xp.arcsin(xp.clip(q / 2.0, -1.0, 1.0))
        for _ in range(15):
            sin_phi = xp.sin(phi)
            e_sin = e * sin_phi
            one_minus_es_sin2 = 1.0 - e_sin * e_sin
            cos_phi = xp.cos(phi)
            cos_phi = xp.where(xp.abs(cos_phi) < EPS_DENOM, EPS_DENOM, cos_phi)
            dphi = (one_minus_es_sin2 * one_minus_es_sin2 / (2.0 * cos_phi)) * (
                q / (1.0 - es)
                - sin_phi / one_minus_es_sin2
                + (1.0 / (2.0 * e)) * xp.log((1.0 - e_sin) / (1.0 + e_sin))
            )
            phi = phi + dphi
            if hasattr(dphi, "__len__"):
                if xp.all(xp.abs(dphi) < EPS_CONV):
                    break
            elif abs(float(dphi)) < EPS_CONV:
                break

        return lam, phi


register("aea", AlbersEqualArea())
